#!/usr/bin/env python3
"""
База MSSQL -> часовые таблицы агрегатов с метками ремонтов (вход ноутбуков 01_EDA и 02).

Подключение и запросы — как в fetch.py. Какие агрегаты и из какой базы брать — словарь UNITS {агрегат: база};
у каждой базы свои таблицы ItemOPC, ValueOPC, DiskretStatus. Агрегаты обрабатываются по очереди, каждый — в свои папки.

1. Датчики агрегата — из ItemOPC, имя начинается на номер агрегата: «КА-4 …», «KA-4 …», «К-4 …», «КА4 …» (русские или
   латинские буквы в любой смеси). Имя запрашивается в Unicode (nvarchar), поэтому кириллица не бьётся.
2. Загрузка: все записи каждого датчика после DATE_FROM — в RAW_DIR/<агрегат>/<код>.parquet (время, CurrStatus, значение,
   текст дискретного сигнала). Повторный запуск догружает только новое. После первого датчика с данными в лог выводятся
   его первые строки — видно, что значения и кодировка в порядке.
3. Часовая таблица — только по записям с CurrStatus = STATUS_GOOD: число — среднее за час, взвешенное по времени
   (значение действует от своей записи до следующей); текст — державшийся в часе дольше всего; известно меньше
   половины часа — пусто. Метка часа — его конец: 07:00 = 06:00–07:00.
4. Метки по файлу заявок: ХР, ВПр, ЗРР, ИСП — «Резерв»; остальные по категории: «Плановый» — «Плановый ремонт»,
   иначе «Аварийный ремонт». При пересечении: аварийный > плановый > резерв > работа; вне периода файла — «Нет данных».

Где что лежит (папки — рядом со скриптом / там, откуда запущен):
  RAW_DIR/<агрегат>/<код>.parquet    — загруженные записи датчика (файл появляется, когда датчик загружен целиком);
  RAW_DIR/<агрегат>/sensors_list.csv — датчики агрегата;
  OUT_DIR/<агрегат>/hourly.parquet   — часовая таблица; рядом repairs.csv (заявки) и sensors.csv (датчики);
  logs/opc_db_<дата>.log             — лог.
Ctrl+C в любой момент: загруженные датчики текущего агрегата сохранены, его таблица сразу строится из них, следующие
агрегаты не трогаются; повторный запуск продолжит. Ошибка по одному агрегату не останавливает остальные.

Запуск: python opc_to_wide_db.py              — все агрегаты из UNITS: загрузить (догрузить) и построить таблицы
        python opc_to_wide_db.py К-4 К-5      — только эти агрегаты
        python opc_to_wide_db.py --table      — без базы: только построить таблицы из уже загруженного (можно с агрегатами)
Требования: pip install pandas numpy pyarrow pymssql
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)          # предупреждения библиотек (numpy и т.п.)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os, re, sys, json, ast, time, logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================ ПАРАМЕТРЫ ============================
# агрегат -> база, где его данные ('К-4', 'КА-4', 'KA-4' — одно и то же). Если у агрегата другой сервер или логин —
# вместо строки словарь: 'К-7': {'database': '...', 'server': '...', 'user': '...', 'password': '...'}
UNITS = {
    'К-4': 'monitoring_2_SD02403775',
    # 'К-5': '...',
}
DATE_FROM = '2023-01-01'              # с какой даты грузить и строить таблицу
DATE_TO = None                        # по какую дату (не включая), например '2026-01-01'; None — до последней записи

# подключение — те же значения, что в fetch.py (общие для всех агрегатов)
SERVER = ''
USER = ''
PASSWORD = ''

STATUS_GOOD = 192                     # в таблицу идут только записи с этим CurrStatus
REPAIRS_FILE = 'Ремонты_НИТЭЦ.csv'    # файл заявок (ежедневные снимки, JSON в столбце «Оборудование»)
REPAIRS_SHIFT_H = 5                   # время в файле заявок на 5 ч раньше времени выгрузки (другой часовой пояс):
                                      # столько часов прибавляется к заявкам; время выгрузки не меняется; 0 — без сдвига
RAW_DIR = 'opc_raw'                   # загруженные записи
OUT_DIR = 'opc_wide'                  # результат
RELOAD = False                        # True — загрузить всё заново, а не только новое
MIN_COVERAGE = 0.5                    # значение известно меньше этой доли часа — ячейка пустая
# ===================================================================

ST_NODATA, ST_WORK, ST_RES, ST_PLAN, ST_FAIL = 'Нет данных', 'Работа', 'Резерв', 'Плановый ремонт', 'Аварийный ремонт'
STATES = [ST_NODATA, ST_WORK, ST_RES, ST_PLAN, ST_FAIL]              # по возрастанию важности
TYPES = {'СР': 'средний ремонт', 'ТР': 'текущий ремонт', 'ХР': 'холодный резерв', 'КР': 'капитальный ремонт',
         'ВПр': 'вынужденный простой', 'ЗРР': 'заявленный режим работы', 'АР': 'аварийный ремонт', 'ИСП': 'испытания'}
RESERVE_TYPES = ('ХР', 'ВПр', 'ЗРР', 'ИСП')
EXCLUDE_STATUS = r'отклон|отмен|аннул|удал'
LAT2CYR = str.maketrans('ABCEHKMOPTXaceopx', 'АВСЕНКМОРТХасеорх')     # латинские буквы, похожие на русские
RAW_SCHEMA = pa.schema([('t', pa.timestamp('ms')), ('s', pa.int64()), ('v', pa.float64()), ('txt', pa.string())])

# запросы (таблицы — в базе агрегата); имя и текст — сразу в Unicode, чтобы кириллица не билась
SQL_ITEMS = """
    SELECT  Kod_ItemOPC, CAST(Name AS nvarchar(400)) AS Name, TypeChannel, Kod_Diskret
    FROM    ItemOPC
    WHERE   {cond}
"""
SQL_ANALOG = """
    SELECT  TimeValue, CurrStatus, CurrValue, NULL AS Txt
    FROM    ValueOPC
    WHERE   Kod_ItemOPC = %s AND TimeValue {op} %s {until}
"""
SQL_DIAG = "SELECT COUNT(*), MIN(TimeValue), MAX(TimeValue) FROM ValueOPC WHERE Kod_ItemOPC = %s"
SQL_DISKRET = """
    SELECT  v.TimeValue, v.CurrStatus, v.CurrValue, CAST(ds.Description AS nvarchar(200)) AS Txt
    FROM    ValueOPC v
    LEFT JOIN DiskretStatus ds ON ds.Kod_Diskret = %s AND ds.StatusVal = v.CurrValue
    WHERE   v.Kod_ItemOPC = %s AND v.TimeValue {op} %s {until}
"""

log = logging.getLogger('opc_db')


def setup_log():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')         # консоль Windows: кириллица без ошибок
    except Exception:
        pass
    os.makedirs('logs', exist_ok=True)
    fmt = logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(f'logs/opc_db_{pd.Timestamp.now():%Y%m%d}.log', encoding='utf-8')):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


# ------------------------------------------------------------------ имена и кодировка
def unit_of(name):
    """Агрегат по началу имени: «КА-4 …», «KA-4 …», «К-4 …», «КА4 …» -> «К-4»; не похоже — None. «КА-40» — это К-40."""
    s = str(name or '').replace('\xa0', ' ').replace('–', '-').strip().upper().translate(LAT2CYR)
    m = re.match(r'^КА?\s*-?\s*(\d+)(?!\d)', s)
    return f'К-{int(m.group(1))}' if m else None


_MOJI_LAT = re.compile('[À-ÿ]{2,}')                         # «ÊÀ-4»: cp1251, прочитанная как latin-1
_MOJI_UTF = re.compile('[РС][Ѐ-Џѐ-џҐґ‘-› -¿]')   # «РљРђ»: UTF-8 как cp1251


def fix_text(x):
    """Строка, испорченная неверной кодировкой, -> нормальная кириллица; нормальная строка не меняется."""
    if not isinstance(x, str) or not x:
        return x
    if len(_MOJI_UTF.findall(x)) >= 2:
        try:
            return x.encode('cp1251').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    if _MOJI_LAT.search(x):
        try:
            return x.encode('latin-1').decode('cp1251')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return x


def py(x):
    """Значение для параметра запроса — обычный тип Python (pymssql не принимает типы numpy); пусто — None."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    x = x.item() if hasattr(x, 'item') else x
    return int(x) if isinstance(x, float) and x.is_integer() else x


def sql_time(t):
    """Время для запроса строкой «ГГГГММДД чч:мм:сс»: так SQL Server читает дату одинаково при любом языке и формате даты
    сервера («2023-01-05» на русском сервере может прочитаться как 1 мая). Доли секунды отбрасываются."""
    return pd.Timestamp(t).strftime('%Y%m%d %H:%M:%S')


# ------------------------------------------------------------------ загрузка из базы
def find_sensors(conn, target):
    """Датчики агрегата из ItemOPC: код (как в базе), имя, тип канала, код расшифровки дискретного сигнала."""
    n = target.split('-')[1]
    pref = [k + a for k in ('К', 'K') for a in ('А', 'A', '')]                 # КА / KA / КA / KА / К / K
    cond = ' OR '.join(f"Name LIKE N'{p}{sep}{n}%'" for p in pref for sep in ('-', ' ', ''))
    with conn.cursor() as cur:
        cur.execute(SQL_ITEMS.format(cond=cond))
        items = pd.DataFrame(cur.fetchall(), columns=['kod_db', 'name', 'type', 'kod_diskret'])
    items['name'] = items['name'].astype(str).str.strip()
    fixed = items['name'].map(fix_text)
    if (fixed != items['name']).any():
        i = (fixed != items['name']).idxmax()
        log.info(f'Кодировка имён: исправлено {int((fixed != items["name"]).sum())}, например «{items.name[i]}» -> «{fixed[i]}»')
    items['name'] = fixed
    if items['name'].str.contains(r'\?{2,}').any():
        log.warning('В именах есть «???» — символы потеряны на сервере, исправить нельзя')
    sens = items[items['name'].map(unit_of).eq(target)].drop_duplicates('kod_db').reset_index(drop=True)
    sens['kod'] = sens['kod_db'].astype(str).str.strip()
    sens['type'] = pd.to_numeric(sens['type'], errors='coerce').fillna(0).astype(int)
    log.info(f'Датчиков {target}: {len(sens)} (дискретных {int((sens.type == 2).sum())})'
             + ('; первые имена:\n    ' + '\n    '.join(sens['name'].head(10)) if len(sens) else ''))
    return sens


def load_raw(conn, sens, raw_dir, t_from, t_to):
    """Записи каждого датчика -> raw_dir/<код>.parquet; без RELOAD — только новее уже загруженных.
    Ctrl+C — остановка: возвращает True (загруженное остаётся)."""
    os.makedirs(raw_dir, exist_ok=True)
    t0, n, total, shown, n_diag = time.time(), len(sens), 0, set(), 0
    until = 'AND TimeValue < %s' if DATE_TO else ''
    try:
        for i, r in enumerate(sens.itertuples(), 1):
            path = os.path.join(raw_dir, f'{r.kod}.parquet')
            old = pd.read_parquet(path) if os.path.exists(path) and not RELOAD else None
            if old is not None and 's' not in old:                          # файл прежней версии — там только хорошие
                old['s'] = STATUS_GOOD
            if old is not None and len(old):
                op, since = '>=', old['t'].max().floor('s')                # догрузка (повтор той же секунды уберут дубли)
            else:
                op, since, old = '>=', t_from, None
            t1 = time.time()
            with conn.cursor() as cur:
                if r.type == 2:
                    sql = SQL_DISKRET.format(op=op, until=until.replace('TimeValue', 'v.TimeValue'))
                    params = (py(r.kod_diskret), py(r.kod_db), sql_time(since)) + ((sql_time(t_to),) if DATE_TO else ())
                else:
                    sql = SQL_ANALOG.format(op=op, until=until)
                    params = (py(r.kod_db), sql_time(since)) + ((sql_time(t_to),) if DATE_TO else ())
                cur.execute(sql, params)
                rows = []
                while True:                                                # порциями, как в fetch.py
                    part = cur.fetchmany(50_000)
                    if not part:
                        break
                    rows += part
            new = pd.DataFrame(rows, columns=['t', 's', 'v', 'txt'])
            added = 0
            if len(new):
                new['t'] = pd.to_datetime(new['t'])
                new['s'] = pd.to_numeric(new['s'], errors='coerce').fillna(-1).astype('int64')
                new['v'] = pd.to_numeric(new['v'], errors='coerce')
                new['txt'] = new['txt'].map(fix_text).astype(object).where(new['txt'].notna(), None)
                df = pd.concat([old, new], ignore_index=True) if old is not None else new
                df = df.sort_values('t', kind='stable').drop_duplicates('t', keep='last')
                pq.write_table(pa.Table.from_pandas(df, schema=RAW_SCHEMA, preserve_index=False), path + '.tmp', compression='zstd')
                os.replace(path + '.tmp', path)                            # файл появляется целиком
                added = len(df) - (len(old) if old is not None else 0)     # новых (повтор последней секунды не в счёт)
                kind = 'discrete' if r.type == 2 else 'analog'
                if kind not in shown:                                      # первые строки — проверить значения и кодировку
                    shown.add(kind)
                    good = int((new['s'] == STATUS_GOOD).sum())
                    log.info(f'ПРОВЕРКА — {"дискретный" if r.type == 2 else "аналоговый"} датчик «{r.name}» (код {r.kod}): '
                             f'{len(new):,} записей, {new.t.min():%Y-%m-%d %H:%M} — {new.t.max():%Y-%m-%d %H:%M}; '
                             f'CurrStatus: ' + ', '.join(f'{k} — {v:,}' for k, v in new['s'].value_counts().head(6).items())
                             + f' (в таблицу пойдёт {good:,} с CurrStatus {STATUS_GOOD})\n'
                             + new.head(8).rename(columns={'t': 'TimeValue', 's': 'CurrStatus', 'v': 'CurrValue', 'txt': 'текст'})
                             .to_string(index=False))
            total += added
            el = time.time() - t0
            log.info(f'[{i}/{n}] {r.name}: +{added:,} записей за {time.time() - t1:.1f} с '
                     f'(всего {total:,}; прошло {el / 60:.1f} мин, осталось ~{el / i * (n - i) / 60:.0f} мин)')
            if not len(new) and old is None and n_diag < 3:                 # первая загрузка, а записей нет — почему?
                n_diag += 1
                with conn.cursor() as cur:
                    cur.execute(SQL_DIAG, (py(r.kod_db),))
                    cnt, t_min, t_max = cur.fetchone()
                log.warning(f'  нет записей с {DATE_FROM}: в ValueOPC по коду {r.kod_db!r} ({type(py(r.kod_db)).__name__}) '
                            + (f'всего {cnt:,} записей, {t_min} — {t_max}' if cnt else 'нет ни одной записи — данные этого '
                               'датчика не в этой базе или у него другой код в ValueOPC'))
            if i == min(10, n) and total == 0:
                log.warning(f'У первых {i} датчиков нет ни одной записи с {DATE_FROM} — см. строки «нет записей» выше')
        log.info(f'Загрузка закончена: {total:,} новых записей за {(time.time() - t0) / 60:.1f} мин')
        return False
    except KeyboardInterrupt:
        done = sum(os.path.exists(os.path.join(raw_dir, f'{k}.parquet')) for k in sens['kod'])
        log.info(f'Остановлено (Ctrl+C). Есть данные по {done} из {n} датчиков — таблица строится из них; '
                 'повторный запуск продолжит загрузку')
        return True


# ------------------------------------------------------------------ часовая таблица
def _hour_integrals(ts, ends, w, edges):
    """Интеграл кусочно-постоянной функции (w на [ts, ends)) по часам сетки edges (секунды)."""
    cum = np.r_[0.0, np.cumsum(w * (ends - ts))]
    j = np.searchsorted(ts, edges, 'right') - 1
    jj = np.clip(j, 0, None)
    F = np.where(j >= 0, cum[jj] + w[jj] * np.clip(np.minimum(edges, ends[jj]) - ts[jj], 0, None), 0.0)
    return np.diff(F)


def hourly_values(df, kind, edges, t_base):
    """Значения датчика по часам: число — среднее, взвешенное по времени; текст — державшееся дольше всего."""
    ts = (df['t'] - t_base).dt.total_seconds().to_numpy()
    ends = np.r_[ts[1:], edges[-1]]                                   # значение действует до следующей записи
    if kind == 'num':
        v = df['v'].to_numpy(float)
        ok = ~np.isnan(v)
        ref = np.nanmedian(v) if ok.any() else 0.0                    # сдвиг — для точности сумм
        s = _hour_integrals(ts, ends, np.where(ok, v - ref, 0.0), edges)
        c = _hour_integrals(ts, ends, ok.astype(float), edges)
        return np.where(c >= MIN_COVERAGE * 3600, s / np.where(c > 0, c, 1) + ref, np.nan).astype('float32')
    x = df['txt'].where(df['txt'].notna(), df['v'].map(lambda z: None if z != z else f'{z:g}')).to_numpy(object)
    states = [s for s in pd.unique(x) if s is not None]
    if not states:
        return np.full(len(edges) - 1, None, object)
    T = np.vstack([_hour_integrals(ts, ends, (x == s).astype(float), edges) for s in states])
    return np.where(T.sum(0) >= MIN_COVERAGE * 3600, np.array(states, object)[T.argmax(0)], None)


def build_hourly(sens, raw_dir, t_from, t_to):
    data, n_bad = {}, 0
    for r in sens.itertuples():
        path = os.path.join(raw_dir, f'{r.kod}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if 's' in df:                                             # только хорошее качество
                n_bad += int((df['s'] != STATUS_GOOD).sum())
                df = df[df['s'] == STATUS_GOOD]
            df = df[(df['t'] >= t_from) & (df['t'] < t_to)]
            if len(df):
                data[r.kod] = df
    if not data:
        sys.exit(f'Нет загруженных записей с CurrStatus = {STATUS_GOOD} — таблицу строить не из чего')
    t_first = max(min(d['t'].min() for d in data.values()), t_from).floor('h')
    t_last = min(max(d['t'].max() for d in data.values()).ceil('h'), t_to)
    edges = np.arange(int((t_last - t_first) / pd.Timedelta('1h')) + 1) * 3600.0
    index = pd.DatetimeIndex(t_first + pd.to_timedelta(edges[1:], unit='s'), name='TimeValue')   # метка — конец часа
    names = sens['name'].where(~sens['name'].duplicated(keep=False), sens['name'] + ' [' + sens['kod'] + ']')
    cols, rows = {}, []
    for r, col in zip(sens.itertuples(), names):
        df = data.get(r.kod)
        kind = 'text' if r.type == 2 or (df is not None and df['txt'].notna().mean() > 0.1) else 'num'
        if df is not None:
            cols[col] = hourly_values(df, kind, edges, t_first)
        rows.append({'kod': r.kod, 'col': col, 'kind': kind, 'тип канала': r.type, 'записей': 0 if df is None else len(df),
                     't_min': None if df is None else df['t'].min(), 't_max': None if df is None else df['t'].max()})
    W = pd.DataFrame(cols, index=index)
    log.info(f'Часовая таблица: {len(W):,} ч ({index[0]:%Y-%m-%d} — {index[-1]:%Y-%m-%d}), датчиков с данными {len(cols)} '
             f'из {len(sens)}; записей с CurrStatus ≠ {STATUS_GOOD} не взято: {n_bad:,}')
    return W, pd.DataFrame(rows)


# ------------------------------------------------------------------ заявки -> метки
def parse_json(v):
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


def repair_type(s):
    s0 = ' '.join(str(s or '').split())
    k = s0.upper().translate(LAT2CYR).replace('.', '').replace(' ', '')
    return next((key for key, full in TYPES.items() if k == key.upper() or full in s0.lower()), s0)


def repair_category(s):
    low = str(s or '').strip().lower()
    if low.startswith('внеплан'):
        return 'Внеплановый'
    if low.startswith('план'):
        return 'Плановый'
    if low in ('но', 'н/о') or low.startswith('неотлож'):
        return 'НО'
    if low == 'ав' or low.startswith('аварий'):
        return 'АВ'
    return str(s or '').strip()


def state_of(typ, cat):
    if typ in RESERVE_TYPES:
        return ST_RES
    return ST_PLAN if cat == 'Плановый' else ST_FAIL


def ticket_time(z, text_key, iso_key):
    """«19:01 30.07.25» (24:00 — конец суток), иначе ISO-поле."""
    m = re.match(r'\s*(\d{1,2}):(\d{2})\s+(\d{1,2})\.(\d{1,2})\.(\d{2,4})\s*$', str(z.get(text_key) or ''))
    if m:
        h, mi, d, mo, y = map(int, m.groups())
        try:
            return pd.Timestamp(year=y + (2000 if y < 100 else 0), month=mo, day=d) + pd.Timedelta(hours=h, minutes=mi)
        except ValueError:
            pass
    t = pd.to_datetime(z.get(iso_key), errors='coerce') if isinstance(z.get(iso_key), str) else pd.NaT
    return None if pd.isna(t) else t.round('min')


def snapshot_dates(s):
    """«2025-01-05[ …]» или «05.01.2024[ …]» — каждая строка по своему виду (смешанные форматы не теряются)."""
    s = pd.Series(s, dtype=object).astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    iso = s.str.match(r'^\d{4}-\d{1,2}-\d{1,2}')
    if iso.any():
        out[iso] = pd.to_datetime(s[iso], format='ISO8601', errors='coerce')
    rest = ~iso & s.ne('')
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], format='mixed', dayfirst=True, errors='coerce')
    return out


_REPAIRS = {}


def repairs_table():
    """Файл заявок целиком (один раз на все агрегаты): UTF-8, а если не читается — Windows-1251."""
    if 'df' not in _REPAIRS:
        with open(REPAIRS_FILE, 'rb') as f:
            head = f.readline()
        sep = max([b';', b',', b'\t'], key=head.count).decode()
        for enc in ('utf-8-sig', 'cp1251'):
            try:
                _REPAIRS['df'] = pd.read_csv(REPAIRS_FILE, sep=sep, encoding=enc, dtype=str, keep_default_na=False)
                log.info(f'Файл заявок {REPAIRS_FILE}: кодировка {enc}, строк {len(_REPAIRS["df"]):,}')
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f'{REPAIRS_FILE}: не читается ни в UTF-8, ни в Windows-1251')
    return _REPAIRS['df'].copy()


def read_repairs(target):
    df = repairs_table()
    df.columns = [c.strip().strip('"').lstrip('﻿') for c in df.columns]
    df['_d'] = snapshot_dates(df['Дата'])
    df = df.dropna(subset=['_d']).sort_values('_d', kind='stable')
    tickets = {}
    for items in df['Оборудование']:
        items = parse_json(items)
        for it in items if isinstance(items, list) else [items] if isinstance(items, dict) else []:
            if not isinstance(it, dict) or unit_of(it.get('Оборудование')) != target:
                continue
            reps = it.get('Ремонты')
            for z in (parse_json(reps) if isinstance(reps, str) else reps) or []:
                if isinstance(z, dict):
                    tickets[z.get('Номер заявки') or json.dumps(z, ensure_ascii=False, sort_keys=True)] = z   # последняя версия
    cov = (df['_d'].min().normalize(), df['_d'].max().normalize() + pd.Timedelta('1D'))
    rows = []
    for z in tickets.values():
        typ, cat = repair_type(z.get('Тип ремонта')), repair_category(z.get('Категория ремонта'))
        rows.append(dict(заявка=str(z.get('Номер заявки') or ''), тип=typ, категория=cat,
                         статус=str(z.get('Статус заявки') or '').strip(), начало=ticket_time(z, 'Начало', 'Итоговое начало'),
                         окончание=ticket_time(z, 'Окончание', 'Итоговое окончание'), состояние=state_of(typ, cat),
                         содержание=str(z.get('Содержание работ') or '').strip()))
    R = pd.DataFrame(rows, columns=['заявка', 'тип', 'категория', 'статус', 'начало', 'окончание', 'состояние', 'содержание'])
    R['начало'], R['окончание'] = pd.to_datetime(R['начало']), pd.to_datetime(R['окончание'])
    R.insert(R.columns.get_loc('начало'), 'начало по заявке', R['начало'])          # исходное время файла заявок
    R.insert(R.columns.get_loc('окончание'), 'окончание по заявке', R['окончание'])
    if REPAIRS_SHIFT_H:                                       # часовой пояс заявок -> время выгрузки
        dt = pd.Timedelta(hours=REPAIRS_SHIFT_H)
        R['начало'], R['окончание'] = R['начало'] + dt, R['окончание'] + dt
        cov = (cov[0] + dt, cov[1] + dt)
    R['учтена'] = R['начало'].notna() & ~R['статус'].str.contains(EXCLUDE_STATUS, case=False, regex=True, na=False)
    R['конец'] = R['окончание'].fillna(cov[1])                        # открытая заявка — до конца файла заявок
    R.loc[R['конец'] <= R['начало'], 'учтена'] = False
    log.info(f'Заявки: {REPAIRS_FILE}, {cov[0]:%Y-%m-%d} — {cov[1] - pd.Timedelta("1D"):%Y-%m-%d}; агрегата {target}: '
             f'{len(R)}, учтено {int(R.учтена.sum())}; ' + ', '.join(f'{k} — {v}' for k, v in R[R.учтена].состояние.value_counts().items())
             + (f'; время заявок сдвинуто на {REPAIRS_SHIFT_H:+g} ч' if REPAIRS_SHIFT_H else ''))
    return R.sort_values('начало').reset_index(drop=True), cov


def label_hours(index, R, cov):
    """Состояние, преобладающее в каждом часе (по минутам), и заявка, которая его дала."""
    ms = lambda x: np.asarray(x, 'datetime64[ms]').astype(np.int64)
    pts = (ms(index)[:, None] - 3_600_000 + (np.arange(60) * 60_000 + 30_000)[None]).ravel()
    code = np.where((pts >= ms(cov[0])) & (pts < ms(cov[1])), STATES.index(ST_WORK), STATES.index(ST_NODATA)).astype(np.int8)
    tk = np.full(len(pts), -1)
    r = R[R.учтена].assign(p=lambda d: d.состояние.map(STATES.index)).sort_values(['p', 'начало'])   # важные — поверх
    for k, z in enumerate(r.itertuples()):
        a, b = np.searchsorted(pts, [ms(z.начало), ms(z.конец)])
        code[a:b], tk[a:b] = z.p, k
    code, tk = code.reshape(-1, 60), tk.reshape(-1, 60)
    maj = np.stack([(code == c).sum(1) for c in range(len(STATES))], 1).argmax(1)
    k = tk[np.arange(len(maj)), (code == maj[:, None]).argmax(1)]
    pick = lambda col: np.r_[r[col].to_numpy(object), None][k]
    return {'Состояние': np.array(STATES, object)[maj], '_Заявка': pick('заявка'), '_Тип': pick('тип'), '_Категория': pick('категория')}


# ------------------------------------------------------------------ главное
def run_unit(target, conf, table_only, t_from, t_to):
    """Один агрегат: загрузка из его базы (если не table_only) и часовая таблица.
    Возвращает итог: 'готово', 'готово (загрузка остановлена)' (нажат Ctrl+C) или почему пропущен."""
    conf = conf if isinstance(conf, dict) else {'database': conf}
    db = conf.get('database', '')
    raw_dir, out_dir = os.path.join(RAW_DIR, target), os.path.join(OUT_DIR, target)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    list_path = os.path.join(raw_dir, 'sensors_list.csv')
    stopped = False
    if table_only:                                              # без базы: таблица из уже загруженного
        if not os.path.exists(list_path):
            log.warning(f'{target}: нет {list_path} — сначала загрузка')
            return 'пропущен: ещё не загружался'
        sens = pd.read_csv(list_path, sep=';', encoding='utf-8-sig', dtype={'kod': str})
        log.info(f'{target}: таблица из загруженного в {os.path.abspath(raw_dir)} (без базы)')
    else:
        import pymssql
        server = conf.get('server', SERVER)
        log.info(f'{target}: с {t_from:%Y-%m-%d}' + (f' по {t_to:%Y-%m-%d}' if DATE_TO else '') + f'; подключение к {server} / {db}')
        conn = pymssql.connect(server=server, user=conf.get('user', USER), password=conf.get('password', PASSWORD), database=db)
        with conn.cursor() as cur:
            cur.execute('SELECT DB_NAME()')
            log.info(f'Подключение установлено, база: {cur.fetchone()[0]}')
        try:
            sens = find_sensors(conn, target)
            if sens.empty:
                log.warning(f'{target}: в {db}.ItemOPC нет датчиков этого агрегата — пропущен')
                return f'пропущен: в {db} нет датчиков'
            sens[['kod', 'name', 'type', 'kod_diskret']].to_csv(list_path, sep=';', index=False, encoding='utf-8-sig')
            log.info(f'Загрузка в {os.path.abspath(raw_dir)}')
            stopped = load_raw(conn, sens, raw_dir, t_from, t_to)
        finally:
            conn.close()

    W, info = build_hourly(sens, raw_dir, t_from, t_to)
    R, cov = read_repairs(target)
    for k, v in label_hours(W.index, R, cov).items():
        W[k] = v
    W.reset_index().to_parquet(os.path.join(out_dir, 'hourly.parquet'), compression='zstd', index=False)
    R.drop(columns='конец').to_csv(os.path.join(out_dir, 'repairs.csv'), sep=';', index=False, encoding='utf-8-sig')
    with open(os.path.join(out_dir, 'labels.json'), 'w', encoding='utf-8') as f:      # для shift_labels.py: не сдвигать дважды
        json.dump({'сдвиг_заявок_ч': float(REPAIRS_SHIFT_H), 'пояснение': 'время заявок + сдвиг = время выгрузки'},
                  f, ensure_ascii=False, indent=1)
    info.to_csv(os.path.join(out_dir, 'sensors.csv'), sep=';', index=False, encoding='utf-8-sig')
    log.info(f'{target} готов: {os.path.abspath(os.path.join(out_dir, "hourly.parquet"))}; часов по состояниям: '
             + ', '.join(f'{k} — {v:,}' for k, v in W['Состояние'].value_counts().items()))
    return 'готово (загрузка остановлена)' if stopped else 'готово'


def main():
    setup_log()
    t_from = pd.Timestamp(DATE_FROM)
    t_to = pd.Timestamp(DATE_TO) if DATE_TO else pd.Timestamp.now().ceil('h')
    table_only = '--table' in sys.argv
    units = {}
    for u, conf in UNITS.items():
        if unit_of(u) is None:
            sys.exit(f'UNITS: не понял номер агрегата в {u!r} (пример: \'К-4\')')
        units[unit_of(u)] = conf
    only = {unit_of(a) for a in sys.argv[1:] if not a.startswith('--')}
    if only - {None}:
        miss = only - set(units)
        if miss:
            sys.exit(f'Агрегатов {sorted(miss)} нет в UNITS')
        units = {u: c for u, c in units.items() if u in only}
    log.info(f'Агрегаты: {", ".join(units)}' + (' — только таблицы из загруженного' if table_only else ''))
    result = {}
    for target, conf in units.items():
        log.info('=' * 30 + f' {target} ' + '=' * 30)
        try:
            result[target] = run_unit(target, conf, table_only, t_from, t_to)
            stopped = 'остановлена' in result[target]
        except KeyboardInterrupt:
            result[target] = 'остановлено'
            stopped = True
        except SystemExit as e:
            result[target] = f'ОШИБКА: {e}'
            log.error(f'{target}: {e}')
            stopped = False
        except Exception as e:                                  # ошибка по агрегату не останавливает остальные
            result[target] = f'ОШИБКА: {e}'
            log.exception(f'{target}: ошибка')
            stopped = False
        if stopped:
            log.info('Остановлено пользователем — следующие агрегаты не обрабатываются')
            break
    log.info('Итог: ' + '; '.join(f'{u} — {r}' for u, r in result.items()))


if __name__ == '__main__':
    main()
