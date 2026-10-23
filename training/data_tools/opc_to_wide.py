#!/usr/bin/env python3
"""
Выгрузка OPC -> сводные таблицы агрегата с метками ремонтов.

Вход:
  SRC — длинная выгрузка OPC (CSV или parquet; можно списком файлов): TimeValue, Kod_ItemOPC, Name, CurrValue[, CurrStatus];
      одна запись = новое значение датчика.
      CurrStatus — качество значения OPC: берутся только записи с качеством из STATUS_GOOD (192), остальные пропускаются.
  REPAIRS_FILE — файл заявок: ежедневные снимки, в столбце «Оборудование» JSON агрегатов с их заявками («Ремонты»).

Результат в OUT_DIR/<агрегат>/:
  hourly.parquet — строка на час; метка 07:00 = интервал 06:00–07:00.
      Числовой датчик — среднее за час, взвешенное по времени (значение действует от своей записи до следующей).
      Текстовый датчик — состояние, которое держалось в часе дольше всего.
      Если значение датчика известно меньше половины часа — ячейка пустая.
  instant.parquet — протянутые значения без усреднения: строка на каждую секунду, в которую пришла хоть одна запись;
      у остальных датчиков — их последнее значение.
  В обеих таблицах: «Состояние» (Работа / Плановый ремонт / Аварийный ремонт / Резерв / Нет данных — вне периода файла
      заявок) и служебные «_Заявка», «_Тип», «_Категория». В часовой — состояние, преобладающее в часе.
  repairs.csv — заявки агрегата: тип, категория, статус, начало, окончание, присвоенное состояние, содержание работ.
  sensors.csv — датчики: код, столбец, числовой/текстовый, число записей, период, частые текстовые значения.
В конце — самопроверка: случайные ячейки обеих таблиц пересчитываются прямо из исходных записей.

Метки — по типу и категории заявки:
  ХР, ВПр, ЗРР, ИСП (холодный резерв, вынужденный простой, заявленный режим, испытания) -> Резерв
  остальные (АР, СР, ТР, КР, …) — только по категории: «Плановый» -> Плановый ремонт, любая другая -> Аварийный ремонт
  При пересечении заявок побеждает: Аварийный > Плановый > Резерв > Работа. Отменённые и отклонённые заявки не учитываются.

Загрузка прямо из базы MSSQL — отдельный скрипт opc_to_wide_db.py (таблицы он строит этим же скриптом).

Запуск: python opc_to_wide.py [выгрузка.csv|.parquet]
Требования: pip install pandas numpy pyarrow
"""
import os, sys, re, glob, time, json, ast, shutil
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv

# ============================ ПАРАМЕТРЫ ============================
SRC = 'opc_export.csv'              # выгрузка OPC: .csv или .parquet (или список файлов)
REPAIRS_FILE = 'Ремонты_НИТЭЦ.csv'  # файл заявок
REPAIRS_SHIFT_H = 5                 # время в файле заявок на 5 ч раньше времени выгрузки (другой часовой пояс):
                                    # столько часов прибавляется к заявкам — метки встают на время выгрузки,
                                    # само время выгрузки не меняется; 0 — без сдвига
UNIT = 'К-2'                        # агрегат: его датчики («КА-2 …», «К-2 …») и общестанционные (без агрегата в имени)
OUT_DIR = 'opc_wide'                # результат: OUT_DIR/<агрегат>/
WORK_DIR = 'opc_work'               # выгрузка в parquet по месяцам — делается один раз (папка от старого скрипта тоже подходит)
DATE_FROM = None                    # с какой даты строить таблицы, например '2023-11-01'; записи раньше не берутся
                                    # (датчик пуст, пока после этой даты не придёт его первая запись); None — с начала выгрузки
TEST_DAYS = None                    # число — пробный прогон: только первые N суток, результат в OUT_DIR_test; None — всё
CONTINUATION_H = 0                  # внеплановый СР/ТР/КР, начатый внутри планового ремонта или не позже N ч после него, —
                                    # продолжение планового (Плановый ремонт); 0 — по общему правилу (Аварийный ремонт)
STATUS_GOOD = (192,)                # CurrStatus (качество OPC), записи с которым берутся; остальные пропускаются;
                                    # None — качество не проверять
# ===================================================================

COL_TIME, COL_KOD, COL_NAME, COL_VAL, COL_STATUS = 'TimeValue', 'Kod_ItemOPC', 'Name', 'CurrValue', 'CurrStatus'
NUMERIC_MIN_SHARE = 0.9     # датчик числовой, если ≥ 90% значений — числа (прочие его значения — пропуски)
MIN_COVERAGE = 0.5          # доля часа с известным значением, иначе ячейка часовой таблицы пустая
CHUNK_ROWS = 50_000         # строк в блоке записи протянутой таблицы (память ~ блок × датчиков × 10 байт)
CHECK_CELLS = 500           # сколько ячеек часовой таблицы перепроверить по исходным записям (0 — без проверки)
BLOCK_MB = 256              # блок чтения CSV

STATE_COL = 'Состояние'
ST_NODATA, ST_WORK, ST_RES, ST_PLAN, ST_FAIL = 'Нет данных', 'Работа', 'Резерв', 'Плановый ремонт', 'Аварийный ремонт'
STATES = [ST_NODATA, ST_WORK, ST_RES, ST_PLAN, ST_FAIL]       # по возрастанию важности: при пересечении побеждает правая
LABEL_COLS = [STATE_COL, '_Заявка', '_Тип', '_Категория']
TYPES = {'СР': 'средний ремонт', 'ТР': 'текущий ремонт', 'ХР': 'холодный резерв', 'КР': 'капитальный ремонт',
         'ВПр': 'вынужденный простой', 'ЗРР': 'заявленный режим работы', 'АР': 'аварийный ремонт', 'ИСП': 'испытания'}
EXCLUDE_STATUS = r'отклон|отмен|аннул|удал'
LAT2CYR = str.maketrans('ABCEHKMOPTX', 'АВСЕНКМОРТХ')
ALIASES = {'КА': 'К', 'ТА': 'Т', 'ГА': 'Г'}                   # «КА-2» и «К-2» — один агрегат

try:                                                          # консоль Windows: кириллица без ошибки charmap
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ------------------------------------------------------------------ чтение файлов
def detect_encoding(path):
    import codecs
    with open(path, 'rb') as f:
        raw = f.read(1 << 20)
    if raw.startswith(codecs.BOM_UTF8):
        return 'utf-8-sig'
    try:                                                      # неполная буква в конце прочитанного куска — не ошибка
        codecs.getincrementaldecoder('utf-8')().decode(raw, final=False)
        return 'utf-8'
    except UnicodeDecodeError:
        return 'cp1251'


def detect_sep(path, enc):
    with open(path, encoding=enc, errors='ignore') as f:
        head = f.readline()
    return max([';', ',', '\t', '|'], key=head.count)


def unit_key(name, strict=False):
    """Агрегат по началу имени: «КА-2 Расход…», «K2», «К-2» -> «К-2». strict — None, если агрегата в начале нет."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = re.sub(r'\s+', ' ', str(name).replace('\xa0', ' ').replace('–', '-').replace('—', '-')).strip().upper().translate(LAT2CYR)
    m = re.match(r'^([А-ЯЁ]{1,3})\s*[-/ ]?\s*(\d+)(?=\D|$)', s)
    if m:
        return f'{ALIASES.get(m.group(1), m.group(1))}-{int(m.group(2))}'
    return None if strict else (s or None)


def parse_json(v):
    """JSON из ячейки: обычный, с одинарными кавычками или «дважды закодированный» (строка внутри строки)."""
    for _ in range(3):
        if isinstance(v, (list, dict)) or v is None:
            return v
        if not isinstance(v, str) or not v.strip():
            return None
        try:
            v = json.loads(v)
        except Exception:
            try:
                v = ast.literal_eval(v)
            except Exception:
                return None
    return v if isinstance(v, (list, dict)) else None


def parse_time(s, state):
    """Время записей: формат подбирается по первому блоку, редкие исключения — медленным разбором."""
    if 'fmt' not in state:
        sample = s.dropna().iloc[:20000]
        cands = ['ISO8601', '%d.%m.%Y %H:%M:%S.%f', '%d.%m.%Y %H:%M:%S', '%d.%m.%Y %H:%M']
        state['fmt'] = max(cands, key=lambda f: pd.to_datetime(sample, format=f, errors='coerce').notna().mean())
        log(f'Формат времени: {state["fmt"]}')
    t = pd.to_datetime(s, format=state['fmt'], errors='coerce')
    bad = t.isna() & s.notna()
    if bad.any():
        t[bad] = pd.to_datetime(s[bad], format='mixed', dayfirst=True, errors='coerce')
    return t


def _as_text(col):
    """Столбец parquet -> строки (как при чтении CSV): None — пусто; целые коды без «.0»."""
    if col.dtype == object:
        return col.map(lambda x: None if x is None or (isinstance(x, float) and x != x) else str(x))
    if pd.api.types.is_float_dtype(col) and (col.dropna() % 1 == 0).all():
        col = col.astype('Int64')
    out = col.astype('string').astype(object)
    out[col.isna().to_numpy()] = None
    return out


def source_batches(path):
    """Выгрузка блоками: (есть ли CurrStatus, генератор (DataFrame, доля файла)). CSV — строки; parquet — приводится к строкам,
    время в parquet может быть уже датой."""
    size = os.path.getsize(path)
    if path.lower().endswith('.parquet'):
        pf = pq.ParquetFile(path)
        header = list(pf.schema_arrow.names)
        log(f'Шаг 1: читаю {path} ({size / 2**30:.1f} ГБ, parquet, {pf.metadata.num_rows:,} строк)')
    else:
        enc = detect_encoding(path)
        sep = detect_sep(path, enc)
        log(f'Шаг 1: читаю {path} ({size / 2**30:.1f} ГБ, кодировка {enc}, разделитель {sep!r})')
        with open(path, encoding=enc, errors='ignore') as f:
            header = [h.strip().strip('"').lstrip('\ufeff') for h in f.readline().rstrip('\r\n').split(sep)]
    miss = [c for c in (COL_TIME, COL_KOD, COL_NAME, COL_VAL) if c not in header]
    if miss:
        sys.exit(f'В выгрузке {path} нет столбцов {miss}; есть: {header}')
    use_status = bool(STATUS_GOOD) and COL_STATUS in header
    if STATUS_GOOD and not use_status:
        log(f'  !! в {path} нет столбца {COL_STATUS} — качество значений не проверяется')
    cols_in = [COL_TIME, COL_KOD, COL_NAME, COL_VAL] + ([COL_STATUS] if use_status else [])

    def gen():
        if path.lower().endswith('.parquet'):
            n_all = max(pf.metadata.num_rows, 1)
            for batch in pf.iter_batches(batch_size=2_000_000, columns=cols_in):
                ch = batch.to_pandas()
                for c in cols_in:
                    if not (c == COL_TIME and pd.api.types.is_datetime64_any_dtype(ch[c])):
                        ch[c] = _as_text(ch[c])
                yield ch, len(ch) / n_all
        else:
            reader = pacsv.open_csv(
                path, read_options=pacsv.ReadOptions(block_size=BLOCK_MB << 20, encoding='utf8' if enc.startswith('utf-8') else enc,
                                                     column_names=header, skip_rows=1),
                parse_options=pacsv.ParseOptions(delimiter=sep),
                convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in cols_in}, include_columns=cols_in))
            for batch in reader:
                yield batch.to_pandas(), batch.nbytes * 0.55 / max(size, 1)       # грубая оценка прогресса
    return use_status, gen()


# ------------------------------------------------------------------ шаг 1: выгрузка -> parquet по месяцам (один раз)
def stage1():
    long_dir, reg_path = os.path.join(WORK_DIR, 'long'), os.path.join(WORK_DIR, 'registry.json')
    files = SRC if isinstance(SRC, (list, tuple)) else [SRC]
    src_sig = [[os.path.abspath(p), os.path.getsize(p), int(os.path.getmtime(p))] for p in files]
    status_sig = sorted(STATUS_GOOD) if STATUS_GOOD else None
    if os.path.exists(os.path.join(long_dir, '_DONE')) and os.path.exists(reg_path):
        reg = json.load(open(reg_path, encoding='utf-8'))
        if reg.get('src') == src_sig and reg.get('status_good') == status_sig:
            log(f'Выгрузка уже переведена в parquet ({long_dir}) — шаг 1 пропущен.')
            return reg
        log(f'В {WORK_DIR} — другая выгрузка или другое правило качества (CurrStatus): шаг 1 выполняется заново.')
    shutil.rmtree(long_dir, ignore_errors=True)
    os.makedirs(long_dir)
    kod2sid, names, stats, txt_counts = {}, defaultdict(Counter), {}, defaultdict(Counter)
    fmt, n_rows, n_bad, part, t0 = {}, 0, 0, 0, time.time()
    n_bad_status, status_seen, bad_kods = 0, Counter(), {}
    total = sum(os.path.getsize(p) for p in files)
    done = 0
    for path in files:
        use_status, batches = source_batches(path)
        size = os.path.getsize(path)
        for ch, frac_file in batches:
            done += frac_file * size
            n_rows += len(ch)
            if pd.api.types.is_datetime64_any_dtype(ch[COL_TIME]):           # parquet: время уже датой
                t = ch[COL_TIME].dt.tz_localize(None) if ch[COL_TIME].dt.tz is not None else ch[COL_TIME]
            else:
                t = parse_time(ch[COL_TIME], fmt)
            ok = t.notna()
            n_bad += int((~ok).sum())
            ch, t = ch[ok], t[ok]
            if use_status:                                    # качество OPC: берутся только записи из STATUS_GOOD
                stv = pd.to_numeric(ch[COL_STATUS].str.strip(), errors='coerce')
                status_seen.update(stv.fillna(-1).astype(np.int64).value_counts().to_dict())
                good = stv.isin(STATUS_GOOD)
                n_bad_status += int((~good).sum())
                for k_, nm_ in ch.loc[~good, [COL_KOD, COL_NAME]].drop_duplicates(COL_KOD).itertuples(index=False):
                    bad_kods.setdefault(str(k_).strip(), nm_)
                ch, t = ch[good], t[good]
            kod = ch[COL_KOD].str.strip()
            for (k, nm), c in ch.assign(k=kod).groupby(['k', COL_NAME], sort=False).size().items():
                names[k][nm] += c
            for k in pd.unique(kod):
                kod2sid.setdefault(k, len(kod2sid))
            sid = kod.map(kod2sid).astype('int32')
            s = ch[COL_VAL].str.strip()
            num = pd.to_numeric(s.str.replace(',', '.', regex=False), errors='coerce')
            txt = s.where(num.isna() & s.notna() & (s != ''))
            for (i, v), c in txt.groupby(sid).value_counts().items():
                txt_counts[int(i)][v] += int(c)
            agg = pd.DataFrame({'sid': sid.values, 't': t.values, 'isnum': num.notna().values}).groupby('sid').agg(
                n=('t', 'size'), n_num=('isnum', 'sum'), t_min=('t', 'min'), t_max=('t', 'max'))
            for i, r in agg.iterrows():
                st = stats.get(int(i))
                if st is None:
                    stats[int(i)] = [int(r.n), int(r.n_num), r.t_min, r.t_max]
                else:
                    st[0] += int(r.n); st[1] += int(r.n_num); st[2] = min(st[2], r.t_min); st[3] = max(st[3], r.t_max)
            df = pd.DataFrame({'sid': sid.values, 't': t.values.astype('datetime64[ms]'), 'v': num.values.astype('float64'),
                               'txt': txt.values})
            ym = (t.dt.year * 100 + t.dt.month).values
            for m in np.unique(ym):
                d = os.path.join(long_dir, f'ym={m // 100}-{m % 100:02d}')
                os.makedirs(d, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(df[ym == m], preserve_index=False),
                               os.path.join(d, f'part-{part:06d}.parquet'), compression='zstd')
            part += 1
            el, frac = time.time() - t0, min(done / total, 0.99)
            log(f'  {n_rows:,} записей, датчиков {len(kod2sid)}, ~{frac:.0%}, {el / 60:.1f} мин'
                + (f', осталось ~{el / frac * (1 - frac) / 60:.0f} мин' if frac > 0.02 else ''))
    sensors = []
    for k, sid in kod2sid.items():
        if sid not in stats:
            continue
        n, n_num, tmin, tmax = stats[sid]
        sensors.append(dict(sid=sid, kod=k, name=names[k].most_common(1)[0][0] if names[k] else str(k), n_names=len(names[k]),
                            n=n, n_num=n_num, share_num=n_num / n, t_min=str(tmin), t_max=str(tmax),
                            txt_values=dict(txt_counts[sid].most_common(50))))
    no_good = [nm for k, nm in bad_kods.items() if k not in kod2sid]       # у датчика нет ни одной записи с хорошим качеством
    reg = dict(sensors=sensors, n_rows=n_rows, n_bad_time=n_bad, n_bad_status=n_bad_status, src=src_sig, status_good=status_sig,
               status_seen={str(k): int(v) for k, v in status_seen.most_common()})
    if status_seen:
        log(f'  Качество (CurrStatus): ' + ', '.join(f'{k} — {v:,}' for k, v in status_seen.most_common(8))
            + f'; записей не {"/".join(map(str, STATUS_GOOD))} пропущено: {n_bad_status:,} ({n_bad_status / max(n_rows, 1):.1%})')
    if no_good:
        log(f'  !! датчиков без единой записи с качеством {"/".join(map(str, STATUS_GOOD))}: {len(no_good)} — в таблицы не попадут; '
            'например: ' + ', '.join(map(str, no_good[:5])))
    json.dump(reg, open(reg_path, 'w', encoding='utf-8'), ensure_ascii=False, default=str)
    open(os.path.join(long_dir, '_DONE'), 'w').close()
    log(f'Шаг 1 готов: {n_rows:,} записей, датчиков {len(sensors)}, нераспознанное время у {n_bad:,}; {(time.time() - t0) / 60:.1f} мин')
    return reg


def read_month(long_dir, m, sids, t_stop=None, t_start=None):
    """Записи выбранных датчиков за месяц: по (датчик, время); из записей с одинаковым временем — последняя в выгрузке."""
    files = sorted(glob.glob(os.path.join(long_dir, f'ym={m}', '*.parquet')))
    cols = dict(sid=np.zeros(0, np.int32), t=np.zeros(0, np.int64), v=np.zeros(0), txt=np.zeros(0, object))
    if not files:
        return cols
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df.sid.isin(sids)]
    t = df.t.values.astype('datetime64[ms]').astype(np.int64)
    if t_stop is not None or t_start is not None:
        keep = np.ones(len(t), bool)
        if t_stop is not None:
            keep &= t < t_stop
        if t_start is not None:
            keep &= t >= t_start
        df, t = df[keep], t[keep]
    sid = df.sid.values.astype(np.int32)
    o = np.lexsort((t, sid))                                  # устойчивая сортировка: порядок выгрузки сохраняется
    sid, t = sid[o], t[o]
    last = np.r_[(sid[1:] != sid[:-1]) | (t[1:] != t[:-1]), True] if len(t) else np.zeros(0, bool)
    return dict(sid=sid[last], t=t[last], v=df.v.values[o][last].astype('float64'), txt=df.txt.values[o][last].astype(object))


# ------------------------------------------------------------------ заявки и метки
def repair_type(s):
    """Тип заявки: сокращение («ТР», «ВПр») или полное название («Текущий ремонт») -> сокращение; иначе как есть."""
    s0 = re.sub(r'\s+', ' ', str(s or '')).strip()
    k = s0.upper().translate(LAT2CYR).replace('.', '').replace(' ', '')
    for key, full in TYPES.items():
        if k == key.upper() or full in s0.lower():
            return key
    return s0


def repair_category(s):
    low = str(s or '').strip().lower()
    if low.startswith('внеплан'):
        return 'Внеплановый'
    if low.startswith('план'):
        return 'Плановый'
    if low in ('но', 'н/о') or low.startswith('неотлож'):
        return 'НО'
    if low in ('ав',) or low.startswith('аварий'):
        return 'АВ'
    return str(s or '').strip()


def state_of(typ, cat):
    if typ in ('ХР', 'ВПр', 'ЗРР', 'ИСП'):                     # резерв, вынужденный простой, заявленный режим, испытания
        return ST_RES
    return ST_PLAN if cat == 'Плановый' else ST_FAIL          # остальные — только по категории


def text_time(s):
    """«19:01 30.07.25» или «24:00 03.08.25» (24:00 — конец суток) -> Timestamp."""
    m = re.match(r'\s*(\d{1,2}):(\d{2})\s+(\d{1,2})\.(\d{1,2})\.(\d{2,4})\s*$', s) if isinstance(s, str) else None
    if not m:
        return None
    h, mi, d, mo, y = map(int, m.groups())
    try:
        return pd.Timestamp(year=y + (2000 if y < 100 else 0), month=mo, day=d) + pd.Timedelta(hours=h, minutes=mi)
    except ValueError:
        return None


def snapshot_dates(s):
    """Столбец «Дата» файла заявок -> Timestamp. Каждая строка разбирается по своему виду: «2025-01-05[ 10:00]»
    (год первым) или «05.01.2024[ 10:00[:00]]» (день первым). Иначе pandas угадывает один формат по первым строкам,
    и строки в другом формате (например, за другие годы) молча пропадают, а «05.01» читается как 5 мая."""
    s = pd.Series(s, dtype=object).astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    iso = s.str.match(r'^\d{4}-\d{1,2}-\d{1,2}')
    dot = s.str.match(r'^\d{1,2}\.\d{1,2}\.\d{2,4}')
    if iso.any():
        out[iso] = pd.to_datetime(s[iso], format='ISO8601', errors='coerce')
    if dot.any():
        out[dot] = pd.to_datetime(s[dot], format='mixed', dayfirst=True, errors='coerce')
    rest = ~iso & ~dot & s.ne('')
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], format='mixed', dayfirst=True, errors='coerce')
    return out


def ticket_time(z, text_key, iso_key):
    t = text_time(z.get(text_key))
    if t is None and isinstance(z.get(iso_key), str):
        t = pd.to_datetime(z.get(iso_key), errors='coerce')
        t = None if pd.isna(t) else t.round('min')
    return t


def load_repairs(target):
    """Все заявки агрегата (последняя версия каждой) с присвоенным состоянием; период, который покрывает файл заявок."""
    enc = detect_encoding(REPAIRS_FILE)
    df = pd.read_csv(REPAIRS_FILE, sep=detect_sep(REPAIRS_FILE, enc), encoding=enc, dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('"').lstrip('﻿') for c in df.columns]
    if 'ОбъектПУ' in df and df['ОбъектПУ'].str.strip().nunique() > 1:
        sys.exit(f'В файле заявок несколько станций: {sorted(df["ОбъектПУ"].str.strip().unique())} — оставьте одну.')
    df['_d'] = snapshot_dates(df['Дата'])
    if df['_d'].isna().any():
        log(f'  !! в файле заявок {int(df["_d"].isna().sum())} строк с нераспознанной датой — пропущены; примеры: '
            + ', '.join(repr(v) for v in df.loc[df['_d'].isna(), 'Дата'].unique()[:5]))
    df = df.dropna(subset=['_d']).sort_values('_d', kind='stable')
    tickets, seen = {}, Counter()
    for d, raw in zip(df['_d'], df['Оборудование']):
        items = parse_json(raw)
        for it in (items if isinstance(items, list) else [items] if isinstance(items, dict) else []):
            if not isinstance(it, dict):
                continue
            u = unit_key(it.get('Оборудование'))
            seen[u] += 1
            if u != target:
                continue
            reps = it.get('Ремонты')
            for z in (parse_json(reps) if isinstance(reps, str) else reps) or []:
                if not isinstance(z, dict):
                    continue
                key = z.get('Номер заявки') or json.dumps(z, ensure_ascii=False, sort_keys=True)
                cur = tickets.get(key)
                if cur is None:
                    tickets[key] = dict(z=z, first=d, last=d, n=1)
                else:
                    cur['n'] += 1
                    if d >= cur['last']:
                        cur.update(z=z, last=d)
    cov = (df['_d'].min().normalize(), df['_d'].max().normalize() + pd.Timedelta('1D'))
    log(f'Заявки: {REPAIRS_FILE}, снимков {len(df):,} за {cov[0]:%Y-%m-%d} — {cov[1] - pd.Timedelta("1D"):%Y-%m-%d}; '
        f'заявок агрегата {target}: {len(tickets)}')
    if not tickets:
        log(f'  !! заявок агрегата {target} нет — всё время будет «{ST_WORK}». Агрегаты в файле: '
            + ', '.join(f'{k} ({v})' for k, v in seen.most_common(20)))
    rows = []
    for num, t in tickets.items():
        z = t['z']
        typ, cat = repair_type(z.get('Тип ремонта')), repair_category(z.get('Категория ремонта'))
        rows.append(dict(заявка=str(z.get('Номер заявки') or ''), тип=typ, категория=cat, статус=str(z.get('Статус заявки') or '').strip(),
                         начало=ticket_time(z, 'Начало', 'Итоговое начало'), окончание=ticket_time(z, 'Окончание', 'Итоговое окончание'),
                         состояние=state_of(typ, cat), примечание='', содержание=str(z.get('Содержание работ') or '').strip(),
                         создана=z.get('Дата создания'), снимков=t['n'], первый_снимок=t['first'].date(), последний_снимок=t['last'].date()))
    R = pd.DataFrame(rows, columns=['заявка', 'тип', 'категория', 'статус', 'начало', 'окончание', 'состояние', 'примечание',
                                    'содержание', 'создана', 'снимков', 'первый_снимок', 'последний_снимок'])
    R['начало'] = pd.to_datetime(R['начало'])
    R['окончание'] = pd.to_datetime(R['окончание'])
    R.insert(R.columns.get_loc('начало'), 'начало по заявке', R['начало'])          # исходное время файла заявок
    R.insert(R.columns.get_loc('окончание'), 'окончание по заявке', R['окончание'])
    if REPAIRS_SHIFT_H:                                       # часовой пояс заявок -> время выгрузки
        dt = pd.Timedelta(hours=REPAIRS_SHIFT_H)
        R['начало'], R['окончание'] = R['начало'] + dt, R['окончание'] + dt
        cov = (cov[0] + dt, cov[1] + dt)
        log(f'  время заявок сдвинуто на {REPAIRS_SHIFT_H:+g} ч (REPAIRS_SHIFT_H)')
    R['учтена'] = R.начало.notna()
    R.loc[R.начало.isna(), 'примечание'] = 'нет времени начала'
    bad = R.статус.str.contains(EXCLUDE_STATUS, case=False, regex=True, na=False)
    R.loc[bad, 'учтена'] = False
    R.loc[bad, 'примечание'] = 'отменена / отклонена'
    R['конец_разметки'] = R.окончание.fillna(cov[1])
    R.loc[R.окончание.isna() & R.учтена, 'примечание'] = 'открыта: до конца файла заявок'
    back = R.учтена & (R.конец_разметки <= R.начало)
    R.loc[back, 'учтена'] = False
    R.loc[back, 'примечание'] = 'окончание не позже начала'
    if CONTINUATION_H:
        plan = R[R.учтена & R.состояние.eq(ST_PLAN)]
        cand = R.учтена & R.состояние.eq(ST_FAIL) & R.тип.isin(['СР', 'ТР', 'КР']) & ~R.категория.isin(['Плановый', 'АВ', 'НО'])
        for i in R.index[cand]:
            t0 = R.at[i, 'начало']
            if ((plan.начало <= t0) & (t0 <= plan.конец_разметки + pd.Timedelta(hours=CONTINUATION_H))).any():
                R.loc[i, ['состояние', 'примечание']] = [ST_PLAN, 'продолжение планового ремонта']
    unknown = sorted(set(R.тип) - set(TYPES) - {''})
    if unknown:
        log(f'  !! незнакомые типы заявок: {unknown} — состояние по категории (плановая -> плановый ремонт, иначе аварийный)')
    return R.sort_values('начало').reset_index(drop=True), cov


class Labeler:
    """Состояние агрегата в любой момент по интервалам заявок."""

    def __init__(self, R, cov):
        r = R[R.учтена].copy()
        r['_p'] = r.состояние.map(STATES.index)
        r = r.sort_values(['_p', 'начало'])                   # более важные рисуются позже и побеждают
        ms = lambda s: s.values.astype('datetime64[ms]').astype(np.int64)
        self.s, self.e = ms(r.начало), ms(r.конец_разметки)
        self.code = r._p.to_numpy(np.int8)
        self.info = [r.заявка.to_numpy(object), r.тип.to_numpy(object), r.категория.to_numpy(object)]
        self.c0, self.c1 = (np.datetime64(c, 'ms').astype(np.int64) for c in cov)

    def at(self, ts):
        """ts — отсортированные моменты (мс). Коды состояний и номер заявки в списке (−1 — нет)."""
        code = np.where((ts >= self.c0) & (ts < self.c1), STATES.index(ST_WORK), STATES.index(ST_NODATA)).astype(np.int8)
        tk = np.full(len(ts), -1, np.int32)
        for i in range(len(self.s)):
            a, b = np.searchsorted(ts, [self.s[i], self.e[i]])
            code[a:b], tk[a:b] = self.code[i], i
        return code, tk

    def columns(self, code, tk):
        out = {STATE_COL: np.array(STATES, dtype=object)[code]}
        for k, arr in zip(LABEL_COLS[1:], self.info):
            out[k] = np.r_[arr, None].astype(object)[tk]
        return out

    def for_hours(self, labels):
        """Преобладающее состояние в каждом часе (по минутам); метка часа — его конец."""
        end = pd.DatetimeIndex(labels).values.astype('datetime64[ms]').astype(np.int64)
        pts = ((end - 3_600_000)[:, None] + (np.arange(60) * 60_000 + 30_000)[None]).ravel()
        code, tk = self.at(pts)
        code, tk = code.reshape(-1, 60), tk.reshape(-1, 60)
        maj = np.stack([(code == c).sum(1) for c in range(len(STATES))], 1).argmax(1).astype(np.int8)
        first = (code == maj[:, None]).argmax(1)
        return self.columns(maj, tk[np.arange(len(maj)), first])


# ------------------------------------------------------------------ часовые признаки
def bin_integrals(S, E, Vs, edges):
    """Интегралы кусочно-постоянных функций по интервалам сетки edges (секунды).
    S, E — начала и концы отрезков (по возрастанию, без пересечений); Vs — значения на отрезках."""
    L = E - S
    i = np.searchsorted(S, edges, side='right') - 1
    valid = i >= 0
    ii = np.clip(i, 0, None)
    part = np.where(valid, np.clip(np.minimum(edges, E[ii]) - S[ii], 0, None), 0.0)
    out = []
    for V in Vs:
        cum = np.concatenate([[0.0], np.cumsum(V * L)])
        out.append(np.diff(np.where(valid, cum[ii] + V[ii] * part, 0.0)))
    return out


def segments(t, vals, carry, u_len):
    """Отрезки действия значений за месяц: от записи до следующей; значение с прошлого месяца действует с начала месяца."""
    has_c, c_val = carry
    if has_c and (len(t) == 0 or t[0] > 0):
        t = np.r_[0.0, t]
        vals = np.r_[np.array([c_val], dtype=vals.dtype), vals]
    return t, np.r_[t[1:], u_len], vals


def num_hours(t, v, carry, n_h, u_len):
    """Числовой датчик: среднее за час, взвешенное по времени (каждое значение — с весом длительности его действия)."""
    T, E, V = segments(t, v, carry, u_len)
    if not len(T):
        return None
    edges = np.arange(n_h + 1) * 3600.0
    known = ~np.isnan(V)
    ref = V[known][0] if known.any() else 0.0                  # сдвиг для точности счёта
    I1, Cv = bin_integrals(T, E, [np.where(known, V - ref, 0.0), known.astype(float)], edges)
    return np.where(Cv >= MIN_COVERAGE * 3600, I1 / np.where(Cv > 0, Cv, 1.0) + ref, np.nan)


def txt_hours(t, x, carry, n_h, u_len):
    """Текстовый датчик: состояние, державшееся в часе дольше всего."""
    T, E, X = segments(t, x, carry, u_len)
    states = [s for s in pd.unique(X) if s is not None]
    if not len(T) or not states:
        return None
    Tk = np.vstack(bin_integrals(T, E, [(X == s).astype(float) for s in states], np.arange(n_h + 1) * 3600.0))
    dom = np.array(states, dtype=object)[Tk.argmax(0)]
    return np.where(Tk.sum(0) >= MIN_COVERAGE * 3600, dom, None)


def text_values(txt, v):
    """Значение текстового датчика: текст как есть, число — строкой («1», «0.5»), пусто — None."""
    out = np.empty(len(txt), dtype=object)
    for i, (z, vv) in enumerate(zip(txt, v)):
        out[i] = z if isinstance(z, str) else (None if vv != vv else f'{vv:g}')
    return out


def hourly_month(rec, sens, carry, u0, u1):
    """Часовые значения всех датчиков за месяц; carry — последнее значение каждого датчика (обновляется)."""
    n_h = int((u1 - u0) / pd.Timedelta('1h'))
    u_len = n_h * 3600.0
    tsec = (rec['t'] - np.datetime64(u0, 'ms').astype(np.int64)) / 1000.0
    b = np.flatnonzero(np.r_[True, rec['sid'][1:] != rec['sid'][:-1], True]) if len(rec['sid']) else np.zeros(1, int)
    where = {int(rec['sid'][b[j]]): (b[j], b[j + 1]) for j in range(len(b) - 1)}
    cols = {}
    for r in sens.itertuples():
        a, z = where.get(r.sid, (0, 0))
        t = tsec[a:z]
        c = carry.get(r.sid, (False, None))
        if r.kind == 'num':
            v = rec['v'][a:z]
            res = num_hours(t, v, (c[0], np.nan if c[1] is None else c[1]), n_h, u_len)
            if z > a:
                carry[r.sid] = (True, v[-1])
            if res is not None:
                cols[r.col] = res.astype('float32')
        else:
            x = text_values(rec['txt'][a:z], rec['v'][a:z])
            res = txt_hours(t, x, c, n_h, u_len)
            if z > a:
                carry[r.sid] = (True, x[-1])
            if res is not None:
                cols[r.col] = res
    idx = pd.DatetimeIndex(u0 + pd.to_timedelta((np.arange(n_h) + 1) * 3600, unit='s'), name='TimeValue')
    return pd.DataFrame(cols, index=idx)


# ------------------------------------------------------------------ протянутая таблица
class InstantWriter:
    """Строка на каждую секунду, в которую пришла хоть одна запись; у остальных датчиков — их последнее значение."""

    def __init__(self, sens, lab, path):
        self.sens, self.lab = sens, lab
        n_sid = int(sens.sid.max()) + 1
        self.num, self.txt = sens[sens.kind == 'num'], sens[sens.kind == 'text']
        self.pos_num = np.full(n_sid, -1, np.int32); self.pos_num[self.num.sid.values] = np.arange(len(self.num))
        self.pos_txt = np.full(n_sid, -1, np.int32); self.pos_txt[self.txt.sid.values] = np.arange(len(self.txt))
        self.carry_num = np.full(len(self.num), np.nan, np.float32)
        self.carry_txt = np.full(len(self.txt), -1, np.int32)
        self.states, self.codes = [[] for _ in range(len(self.txt))], {}
        fields, self.order = [pa.field('TimeValue', pa.timestamp('ms'))], []
        for r in sens.itertuples():
            if r.kind == 'num':
                fields.append(pa.field(r.col, pa.float32())); self.order.append(('n', int(self.pos_num[r.sid])))
            else:
                fields.append(pa.field(r.col, pa.string())); self.order.append(('t', int(self.pos_txt[r.sid])))
        fields += [pa.field(k, pa.string()) for k in LABEL_COLS]
        self.schema = pa.schema(fields)
        self.writer = pq.ParquetWriter(path, self.schema, compression='zstd')
        self.rows = 0

    def _code(self, k, st):
        c = self.codes.get((k, st))
        if c is None:
            c = self.codes[(k, st)] = len(self.states[k])
            self.states[k].append(st)
        return c

    @staticmethod
    def _ffill(M, P, carry):
        M[0], P[0] = carry, True
        idx = np.where(P, np.arange(len(M), dtype=np.int32)[:, None], np.int32(0))
        np.maximum.accumulate(idx, axis=0, out=idx)
        return np.take_along_axis(M, idx, 0)

    def add_month(self, rec):
        if not len(rec['t']):
            return
        tr = rec['t'] // 1000 * 1000
        o = np.lexsort((rec['t'], rec['sid'], tr))
        sid, tr, v, x = rec['sid'][o], tr[o], rec['v'][o], rec['txt'][o]
        last = np.r_[(tr[1:] != tr[:-1]) | (sid[1:] != sid[:-1]), True]      # в одной секунде — последняя запись датчика
        sid, tr, v, x = sid[last], tr[last], v[last], x[last]
        ts_u, inv = np.unique(tr, return_inverse=True)
        pn, pt = self.pos_num[sid], self.pos_txt[sid]
        code = np.full(len(sid), -1, np.int32)
        for i in np.flatnonzero(pt >= 0):
            st = x[i] if isinstance(x[i], str) else (None if v[i] != v[i] else f'{v[i]:g}')
            code[i] = -1 if st is None else self._code(int(pt[i]), st)
        for a in range(0, len(ts_u), CHUNK_ROWS):
            b = min(a + CHUNK_ROWS, len(ts_u))
            r0, r1 = np.searchsorted(inv, [a, b])
            ri, n = inv[r0:r1] - a + 1, b - a + 1
            sn, stx = pn[r0:r1] >= 0, pt[r0:r1] >= 0
            M = np.full((n, len(self.num)), np.nan, np.float32); P = np.zeros(M.shape, bool)
            M[ri[sn], pn[r0:r1][sn]] = v[r0:r1][sn]; P[ri[sn], pn[r0:r1][sn]] = True
            M = self._ffill(M, P, self.carry_num); self.carry_num = M[-1].copy()
            Cc = np.full((n, len(self.txt)), -1, np.int32); Q = np.zeros(Cc.shape, bool)
            Cc[ri[stx], pt[r0:r1][stx]] = code[r0:r1][stx]; Q[ri[stx], pt[r0:r1][stx]] = True
            Cc = self._ffill(Cc, Q, self.carry_txt); self.carry_txt = Cc[-1].copy()
            Mt, Ct = np.ascontiguousarray(M[1:].T), np.ascontiguousarray(Cc[1:].T)
            cols = [pa.array(ts_u[a:b].astype('datetime64[ms]'), type=pa.timestamp('ms'))]
            for kind, p in self.order:
                cols.append(pa.array(Mt[p], type=pa.float32(), from_pandas=True) if kind == 'n'
                            else pa.array(np.array(self.states[p] + [None], dtype=object)[Ct[p]], type=pa.string()))
            lc = self.lab.columns(*self.lab.at(ts_u[a:b]))
            cols += [pa.array(lc[k], type=pa.string()) for k in LABEL_COLS]
            self.writer.write_table(pa.Table.from_arrays(cols, schema=self.schema))
        self.rows += len(ts_u)

    def close(self):
        self.writer.close()


# ------------------------------------------------------------------ самопроверка
def self_check(W, inst_path, pick, parts):
    """Случайные ячейки часовой и строки протянутой таблицы пересчитываются прямо из исходных записей (простым циклом).
    pick — проверяемые датчики, parts — их записи, собранные по месяцам."""
    rng = np.random.default_rng(0)
    recs = {}
    for p in pick.itertuples():
        ch = parts.get(p.sid)
        if not ch or p.col not in W.columns:
            continue
        t = np.concatenate([a[0] for a in ch])
        v = np.concatenate([a[1] for a in ch])
        recs[p.sid] = (t, v if p.kind == 'num' else text_values(np.concatenate([a[2] for a in ch]), v))
    pick = pick[pick.sid.isin(list(recs))]
    if pick.empty:
        return
    lab_ms = W.index.values.astype('datetime64[ms]').astype(np.int64)
    ok = total = 0
    bad = []
    for p in pick.itertuples():
        t, vals = recs[p.sid]
        busy = np.unique(np.searchsorted(lab_ms, t, 'right'))           # часы, в которых были записи
        busy = busy[busy < len(lab_ms)]
        hours = np.unique(np.r_[rng.choice(busy, min(len(busy), CHECK_CELLS // len(pick) // 2), replace=False) if len(busy) else [],
                                rng.choice(len(lab_ms), min(len(lab_ms), CHECK_CELLS // len(pick) // 2), replace=False)]).astype(int)
        for h in hours:
            b = lab_ms[h]; a = b - 3_600_000
            i0, i1 = np.searchsorted(t, a, 'left'), np.searchsorted(t, b, 'left')
            pts = ([(a, vals[i0 - 1])] if i0 > 0 and not (i0 < len(t) and t[i0] == a) else []) + [(t[i], vals[i]) for i in range(i0, i1)]
            dur = [(pts[k + 1][0] if k + 1 < len(pts) else b) - pts[k][0] for k in range(len(pts))]
            if p.kind == 'num':
                kn = [(val, d) for (_, val), d in zip(pts, dur) if val == val]
                cov = sum(d for _, d in kn)
                exp = sum(val * d for val, d in kn) / cov if cov >= MIN_COVERAGE * 3_600_000 else np.nan
                got = W[p.col].iloc[h]
                sc = max(1.0, abs(exp) if exp == exp else 1.0)
                good = (exp != exp and got != got) or (exp == exp and got == got and abs(exp - got) <= 2e-4 * sc)
            else:
                acc = defaultdict(float)
                for (_, val), d in zip(pts, dur):
                    if val is not None:
                        acc[val] += d
                g = W[p.col].iloc[h]
                g = None if g is None or g != g else g
                if sum(acc.values()) < MIN_COVERAGE * 3_600_000:
                    good = g is None
                else:
                    good = g is not None and abs(acc.get(g, -1) - max(acc.values())) < 1e-6
                exp, got = max(acc, key=acc.get) if acc else None, g
            total += 1
            ok += good
            if not good and len(bad) < 5:
                bad.append(f'{p.col} @ {W.index[h]}: ожидалось {exp}; в таблице {got}')
    log(f'Самопроверка часовой таблицы (среднее, текстовое состояние): '
        f'совпало {ok} из {total} ячеек' + ('' if ok == total else ' — РАСХОЖДЕНИЯ:\n    ' + '\n    '.join(bad)))
    # протянутая: значение в строке = последняя запись датчика не позже этой секунды
    pf = pq.ParquetFile(inst_path)
    ok = total = 0
    for g in rng.choice(pf.num_row_groups, min(10, pf.num_row_groups), replace=False):
        tb = pf.read_row_group(int(g), columns=['TimeValue'] + pick.col.tolist()).to_pandas()
        ts = tb.TimeValue.values.astype('datetime64[ms]').astype(np.int64)
        for i in rng.choice(len(tb), min(20, len(tb)), replace=False):
            for p in pick.itertuples():
                t, vals = recs[p.sid]
                j = np.searchsorted(t, ts[i] + 1000, 'left') - 1
                got = tb[p.col].iloc[i]
                if p.kind == 'num':
                    e = vals[j] if j >= 0 else np.nan
                    good = (e != e and got != got) or (e == e and abs(float(np.float32(e)) - got) <= 1e-6 * max(1, abs(e)))
                else:
                    e = vals[j] if j >= 0 else None
                    good = (e is None and (got is None or got != got)) or e == got
                total += 1
                ok += bool(good)
    log(f'Самопроверка протянутой таблицы: совпало {ok} из {total} значений')



# ------------------------------------------------------------------ главное
def main():
    t0 = time.time()
    reg = stage1()
    long_dir = os.path.join(WORK_DIR, 'long')
    target = unit_key(UNIT)
    sens = pd.DataFrame(reg['sensors'])
    sens['агрегат'] = sens.name.map(lambda s: unit_key(s, strict=True))
    found = Counter(sens['агрегат'].dropna())
    if not found.get(target):
        sys.exit(f'Датчиков агрегата {target} не найдено; агрегаты в выгрузке: {dict(found.most_common(20))}')
    sens = sens[sens['агрегат'].eq(target) | sens['агрегат'].isna()].copy()
    sens['kind'] = np.where(sens.share_num >= NUMERIC_MIN_SHARE, 'num', 'text')
    dup = sens.name.duplicated(keep=False)
    sens['col'] = np.where(dup, sens.name + ' [' + sens.kod.astype(str) + ']', sens.name)
    sens = sens.sort_values('col').reset_index(drop=True)
    log(f'Датчиков {target}: {int(sens["агрегат"].notna().sum())}, общестанционных: {int(sens["агрегат"].isna().sum())} '
        f'(числовых {int((sens.kind == "num").sum())}, текстовых {int((sens.kind == "text").sum())})')

    out_dir = os.path.join(OUT_DIR + ('_test' if TEST_DAYS else ''), target)
    os.makedirs(out_dir, exist_ok=True)
    R, cov = load_repairs(target)
    R.drop(columns='конец_разметки').to_csv(os.path.join(out_dir, 'repairs.csv'), sep=';', index=False, encoding='utf-8-sig')
    with open(os.path.join(out_dir, 'labels.json'), 'w', encoding='utf-8') as f:      # для shift_labels.py: не сдвигать дважды
        json.dump({'сдвиг_заявок_ч': float(REPAIRS_SHIFT_H), 'пояснение': 'время заявок + сдвиг = время выгрузки'},
                  f, ensure_ascii=False, indent=1)
    cnt = R[R.учтена].groupby(['тип', 'категория', 'состояние']).size()
    log('  тип / категория -> состояние: ' + '; '.join(f'{a} / {b or "—"} -> {c} ({n})' for (a, b, c), n in cnt.items()))
    lab = Labeler(R, cov)

    t_first = pd.to_datetime(sens.t_min).min().floor('h')
    t_last = pd.to_datetime(sens.t_max).max()
    t_start = None
    if DATE_FROM:
        d0 = pd.Timestamp(DATE_FROM).floor('h')
        if d0 >= t_last:
            sys.exit(f'DATE_FROM = {DATE_FROM}: позже последней записи выгрузки ({t_last:%Y-%m-%d %H:%M})')
        if d0 > t_first:
            t_first = d0
            t_start = np.datetime64(d0, 'ms').astype(np.int64)
            gone = int((pd.to_datetime(sens.t_max) < d0).sum())
            log(f'Таблицы — с {d0:%Y-%m-%d %H:%M} (DATE_FROM); записи раньше не берутся'
                + (f'; датчиков без записей после этой даты: {gone} — их столбцы будут пустыми' if gone else ''))
    t_end = t_last.ceil('h') + (pd.Timedelta('1h') if t_last == t_last.ceil('h') else pd.Timedelta(0))
    if TEST_DAYS:
        t_end = min(t_end, t_first + pd.Timedelta(days=TEST_DAYS))
        log(f'ПРОБНЫЙ ПРОГОН: первые {TEST_DAYS} сут -> {out_dir}')
    t_stop = np.datetime64(t_end, 'ms').astype(np.int64)
    months = [str(p) for p in pd.period_range(t_first, t_end - pd.Timedelta('1ms'), freq='M')]

    inst_path = os.path.join(out_dir, 'instant.parquet')
    inst = InstantWriter(sens, lab, inst_path)
    live = sens[pd.to_datetime(sens.t_min) < t_end]                      # датчики для самопроверки: 8 числовых и 4 текстовых
    pick = pd.concat([g.sample(min(k, len(g)), random_state=0) for g, k in ((live[live.kind == 'num'], 8), (live[live.kind == 'text'], 4))])
    check = defaultdict(list)
    carry, parts, t1 = {}, [], time.time()
    for i, m in enumerate(months):
        u0 = pd.Timestamp(m + '-01')
        u1 = min(u0 + pd.offsets.MonthBegin(1), t_end)
        rec = read_month(long_dir, m, sens.sid.values, t_stop, t_start)
        inst.add_month(rec)
        parts.append(hourly_month(rec, sens, carry, u0, u1))
        if CHECK_CELLS:
            for s in pick.sid:
                k = rec['sid'] == s
                if k.any():
                    check[s].append((rec['t'][k], rec['v'][k], rec['txt'][k]))
        el = time.time() - t1
        log(f'  {m}: записей {len(rec["t"]):,}; {el / 60:.1f} мин, осталось ~{el / (i + 1) * (len(months) - i - 1) / 60:.0f} мин')
    inst.close()

    W = pd.concat(parts)
    order = []
    for r in sens.itertuples():
        order.append(r.col)
    W = W[[c for c in order if c in W.columns]]
    W = W[W.index > t_first]
    for k, v in lab.for_hours(W.index).items():
        W[k] = v
    h_path = os.path.join(out_dir, 'hourly.parquet')
    W.reset_index().to_parquet(h_path, compression='zstd', index=False)

    so = sens[['kod', 'col', 'агрегат', 'kind', 'n', 'share_num', 't_min', 't_max']].copy()
    so['агрегат'] = so['агрегат'].fillna('общестанционный')
    so['частые значения'] = [json.dumps(dict(list(v.items())[:10]), ensure_ascii=False) if k == 'text' else ''
                             for v, k in zip(sens.txt_values, sens.kind)]
    so.to_csv(os.path.join(out_dir, 'sensors.csv'), sep=';', index=False, encoding='utf-8-sig')
    st = W[STATE_COL].value_counts()
    log(f'{h_path}: {len(W):,} ч × {int((sens.kind == "num").sum())} числовых и {int((sens.kind == "text").sum())} текстовых датчиков; '
        + ', '.join(f'{k}: {v:,} ч' for k, v in st.items()))
    log(f'{inst_path}: {inst.rows:,} строк, {os.path.getsize(inst_path) / 2**30:.2f} ГБ')
    if CHECK_CELLS:
        self_check(W, inst_path, pick, check)
    log(f'Готово за {(time.time() - t0) / 60:.1f} мин')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        SRC = sys.argv[1]
    main()
