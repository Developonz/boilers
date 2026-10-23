"""
Общие функции для ноутбуков 00–03 (EDA, детектирование аномалий, сравнение котлов).

Что здесь есть:
  * ШАБЛОНЫ ИМЁН ДАТЧИКОВ — все регулярные выражения для поиска датчиков по имени, одним блоком (ниже, после вывода);
  * список датчиков из Excel (лист на котёл: Kod, col, агрегат) — какие датчики брать в модели;
  * оценка без AUC: точный интервал частоты, точное p, ранг окна, упреждение;
  * оформление выводов и графиков (say, short, setup_plots, ...);
  * загрузка часовых средних и мгновенных данных (мгновенные НЕ усредняются: берутся «снимки» значений
    на регулярной сетке с малым шагом и кэшируются);
  * заявки на ремонт (repairs.csv от opc_to_wide.py);
  * эпизоды состояний, ремонты, фактический режим работы по данным (пуск / работа / останов);
  * «общее состояние котла» по данным (главные компоненты всех датчиков и состояний оборудования);
  * «метод аналогов»: ожидаемое значение и разброс датчика по похожим моментам нормальной работы.
"""
import os
import re
import glob
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
try:
    from IPython.display import display, Markdown
except ImportError:  # Plain Python execution does not require a Jupyter kernel.
    def display(value):
        print(value)
    Markdown = str

TIME_COL, STATE_COL = 'TimeValue', 'Состояние'
ST_WORK, ST_FAIL, ST_PLAN, ST_RES, ST_NODATA = 'Работа', 'Аварийный ремонт', 'Плановый ремонт', 'Резерв', 'Нет данных'
EV_FAIL, EV_PLAN, EV_RES = 'Авария', 'План', 'Резерв'
FEATURE_RX = re.compile(r'__(min|max|std|n)$')         # часовые признаки внутри часа (opc_to_wide.py): «Датчик__max» и т.п.

# фактический режим по данным
M_STOP, M_START, M_RUN, M_PRESTOP = 'Останов', 'Пуск', 'Работа', 'Перед остановом'

COLORS = {ST_WORK: '#2e8b57', ST_FAIL: '#d62728', ST_PLAN: '#1f77b4', ST_RES: '#e6a23c', ST_NODATA: '#c7c7c7',
          M_STOP: '#8c8c8c', M_START: '#ff9f1c', M_PRESTOP: '#9467bd',
          EV_FAIL: '#d62728', EV_PLAN: '#1f77b4', EV_RES: '#e6a23c'}


# ----------------------------------------------------------------------------- вывод

def say(*lines):
    """Выводы списком (markdown); пустые строки пропускаются."""
    lines = [str(l) for l in lines if l]
    if lines:
        display(Markdown('\n'.join(f'- {l}' for l in lines)))


def md(text):
    display(Markdown(text))


LAT2CYR = str.maketrans('ABCEHKMOPTXaceopx', 'АВСЕНКМОРТХасеорх')   # латинские буквы, похожие на русские («KA-8» = «КА-8»)

# ============================================================================= ШАБЛОНЫ ИМЁН ДАТЧИКОВ
# Все регулярные выражения для поиска датчиков по имени. В ноутбуках 01 и 02 их можно заменить параметрами RX_… (None —
# эти). {N} — номер агрегата (подставляется из UNIT: для К-2 — 2). Имя проверяется как есть, с латиницей, заменённой на
# кириллицу («KA-2» = «КА-2», «T» = «Т»), и с развёрнутым сокращением в начале (RX_ABBREV), регистр не важен.

RX_UNIT_PREFIX = r'^[КK][АA]?\s*-?\s*\d+\s+'            # префикс агрегата в имени: «КА-8 », «KA-8 », «К-8 », «КА8 »

# сокращение первым словом (после префикса агрегата) -> полное слово: «T пит. воды», «t пв», «Темп. …», «Т-ра …» ->
# «Температура …»; «F пара», «Расх. …» -> «Расход …»; «P …», «Давл. …» -> «Давление …»
RX_ABBREV = {'Расход': r'f|расх', 'Температура': r'т|t|темп|т-ра|t-ра', 'Давление': r'р|p|давл'}

# датчик «котёл работает / стоит» — только расход перегретого пара: «Расход перегретого пара», «F перегретого пара»
# (сокращение F разворачивается), с префиксом агрегата или без; если подходят несколько — берётся самое короткое имя
RX_RUN = r'^([КK][АA]?\s*-?\s*\d+\s+)?Расход перегретого пара'

# условия работы (задаются снаружи котла, модель их не проверяет, а учитывает): аналоги, AAKR, MSET
RX_FEED_WATER_TEMP = r'^(КА?-?{N} )?Температура (питательной воды|пит\.? ?воды|ПВ|п\.? ?в\.?)(?=\s|,|$)(?!.*(\bза\b|\bпосле\b|выход))'
#   ^ питательная вода ДО экономайзера (задаёт турбинный цех); «… за ВЭК», «после», «на выходе» — нет (на ~100 °C теплее)
RX_COLD_AIR_TEMP = r'(^|\s)(т|температура)\.?\s+(холодного\s+воздуха|воздуха\s+холодного)'
RX_CONDITIONS = {'холодный воздух': RX_COLD_AIR_TEMP, 'питательная вода': RX_FEED_WATER_TEMP}

# дополнительные условия регрессии (бустингу лишние входы не мешают)
RX_REG_CONDITIONS = {'обороты питателей': r'^(КА?-?{N} )?Обороты ПСУ-{N}[АБВГ]$',
                     'уставки шиберов': r'^(КА?-?{N} )?УП шибера ({N}Ш ВВ-\d|Ш ГВ-{N}[АБВГ]|Ш ХВ-{N}[АБВГ]|Ш РГВ ДВ-{N}[АБ])$',
                     'калориферы': r'^(КА?-?{N} )?(Расход сетевой воды на калориферы|Т воды до калорифера)$'}

# состояния механизмов (текстовые датчики): мельницы ММТ-А…Г (питатели ПСУ и ВГД повторяют ММТ), дутьевые вентиляторы
# и дымососы (скорость 1 / 2), задвижка #КВ-1; признак в моделях — «не в обычном состоянии»
RX_STATES = [r'^(КА?-?{N} )?ММТ-[АБВГ] Состояние$', r'^(КА?-?{N} )?Д[ВС]-[АБ] Состояние$', r'^(КА?-?{N} )?{N}КВ-1 Состояние$']
RX_MILLS = None                                          # «мельниц в работе» по токам — не нужно: состояния ММТ заданы явно

# небаланс вода − пар (EDA, раздел 10.1)
RX_FEED_WATER_FLOW = r'^(КА?-?{N} )?Расход (питательной воды|пит\.? ?воды|ПВ|п\.? ?в\.?)(?=\s|,|$)'
RX_STEAM_FLOW = r'^(КА?-?{N} )?Расход (острого|перегретого|свежего) пара|^(КА?-?{N} )?Расход пара (за|от) котл'

# Historical name: marks normative/setpoint names for EDA review only.
# A match is not a reason to exclude the signal from model training.
RX_NOT_MEASUREMENT = r'нормативн|задание'

# подозрительные значения текстовых датчиков (EDA, раздел 9): тайм-ауты, самоход, срабатывание защит
RX_SUSPICIOUS_TEXT = r'тайм|самоход|авари|неисправ|ошиб|отказ|notswitch|сработ'

ABBREV_RX = [(re.compile(rf'^(?:{a})(?:°с?)?(?:\.\s*|\s+|$)', re.I), full) for full, a in RX_ABBREV.items()]
UNIT_PREFIX = RX_UNIT_PREFIX


def unit_rx(pattern, unit):
    """Шаблон для агрегата: {N} -> его номер (нет номера — любое число). Списки и словари шаблонов — поэлементно."""
    n = unit_number(unit)
    sub = lambda p_: p_.replace('{N}', str(n) if n is not None else r'\d+') if isinstance(p_, str) else p_
    if isinstance(pattern, dict):
        return {k: sub(v) for k, v in pattern.items()}
    if isinstance(pattern, (list, tuple)):
        return [sub(v) for v in pattern]
    return sub(pattern)


def expand_abbrev(name):
    """Сокращение в начале имени (после префикса агрегата) -> полное слово: «КА-3 T пит. воды» -> «КА-3 Температура пит. воды».
    Имена без сокращения не меняются («Ток …», «Тяга …», «Разрежение …»)."""
    s_ = ' '.join(str(name).split())
    m = re.match(UNIT_PREFIX, s_, re.I)
    head, body = (s_[:m.end()], s_[m.end():]) if m else ('', s_)
    for rx, full in ABBREV_RX:
        mm = rx.match(body)
        if mm:
            rest = body[mm.end():]
            return head + full + (' ' + rest if rest else '')
    return s_


def short(name, n=48):
    """Короткое имя датчика для графиков: без префикса агрегата («КА-2 », «KA-3 »), не длиннее n символов."""
    s = re.sub(r'\s+', ' ', str(name)).strip()
    s = re.sub(UNIT_PREFIX, '', s)
    return s if len(s) <= n else s[:n - 1] + '…'


def fmt_list(xs, n=8, f=str):
    xs = list(xs)
    if not xs:
        return '—'
    return ', '.join(f(x) for x in xs[:n]) + (f' … (+{len(xs) - n})' if len(xs) > n else '')


def sensor_review_decisions(quality, manual_excluded=()):
    """EDA annotations are advisory; only the user's explicit list excludes."""
    excluded = list(dict.fromkeys(manual_excluded))
    unknown = sorted(set(excluded) - set(quality.index))
    if unknown:
        raise ValueError(f'Не найдены датчики из явного списка исключений: {unknown}')
    return dict(sensors_good=quality.index[quality['итог'].eq('годен')].tolist(),
                sensors_caution=quality.index[quality['итог'].eq('с оговорками')].tolist(),
                sensors_excluded=excluded,
                sensors_review=quality.index[quality['итог'].ne('годен') & ~quality.index.isin(excluded)].tolist(),
                sensor_qc_policy='advisory; exclusions are manual only')


def pct(x, d=1):
    return '—' if x is None or not np.isfinite(x) else f'{100 * x:.{d}f}%'


def setup_plots():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 130, 'figure.figsize': (14, 4),
        'figure.constrained_layout.use': True,
        'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False,
        'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False,
    })
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 70)


def date_axis(ax):
    loc = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def shade_events(ax, events, alpha=0.15, legend=False):
    """Периоды ремонтов — заливка по оси времени."""
    for e in events.itertuples():
        ax.axvspan(e.start, e.end, color=COLORS.get(e.type, 'gray'), alpha=alpha, lw=0)
    if legend:
        add_legend(ax, {ST_FAIL: COLORS[EV_FAIL], ST_PLAN: COLORS[EV_PLAN]}, alpha=0.3)


def add_legend(ax, items, alpha=1.0, **kw):
    """Легенда из цветных прямоугольников: items = {подпись: цвет}."""
    handles = [Patch(color=c, alpha=alpha, label=l) for l, c in items.items()]
    kw.setdefault('ncol', min(len(handles), 6))
    kw.setdefault('loc', 'upper left')
    return ax.legend(handles=handles, **kw)


def boxplot(ax, data, labels, horizontal=False, **kw):
    """Ящики с подписями — работает в старых и новых версиях matplotlib (labels / tick_labels, vert / orientation)."""
    data = [np.asarray(d, float)[np.isfinite(np.asarray(d, float))] for d in data]
    kw.setdefault('showfliers', False)
    try:
        ax.boxplot(data, orientation='horizontal' if horizontal else 'vertical', **kw)
    except TypeError:
        ax.boxplot(data, vert=not horizontal, **kw)
    pos = np.arange(1, len(data) + 1)
    if horizontal:
        ax.set_yticks(pos); ax.set_yticklabels(labels)
    else:
        ax.set_xticks(pos); ax.set_xticklabels(labels)


def corr_linkage(C, method='complete'):
    """Иерархическая кластеризация по матрице корреляций: расстояние 1 − |ρ| (с защитой от погрешностей: |ρ| ≤ 1,
    пропуски = 0, матрица строго симметрична)."""
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    A = np.clip(np.nan_to_num(np.abs(np.asarray(C, float))), 0, 1)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 1)
    return hierarchy.linkage(squareform(np.clip(1 - A, 0, 1), checks=False), method)


def heatmap(ax, M, xlabels=None, ylabels=None, cmap='RdBu_r', vmin=None, vmax=None, cbar=None,
            annot=False, fmt='{:.0f}', annot_min=None, xrot=0, fontsize=8):
    """Тепловая карта таблицы: подписи строк/столбцов, цветовая шкала, при желании — числа в ячейках."""
    A = np.asarray(M, dtype=float)
    im = ax.imshow(A, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=fontsize)
    if xlabels is not None:
        step = max(1, len(xlabels) // 30)
        ax.set_xticks(range(0, len(xlabels), step))
        ax.set_xticklabels(list(xlabels)[::step], fontsize=fontsize, rotation=xrot, ha='right' if xrot else 'center')
    ax.grid(False)
    if annot:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if np.isfinite(v) and (annot_min is None or abs(v) >= annot_min):
                    ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=fontsize - 1,
                            color='white' if im.norm(v) > 0.6 else 'black')
    if cbar:
        plt.colorbar(im, ax=ax, label=cbar, fraction=0.025, pad=0.01)
    return im


def time_heatmap(ax, frame, cmap='magma', vmin=None, vmax=None, cbar=None, ylabels=None, fontsize=8):
    """Тепловая карта «строки × время»: frame — индекс по времени, столбцы — строки карты."""
    x = frame.index
    step = (x[1] - x[0]) if len(x) > 1 else pd.Timedelta('1h')
    xe = mdates.date2num(list(x) + [x[-1] + step])
    im = ax.pcolormesh(xe, np.arange(frame.shape[1] + 1), frame.to_numpy(float).T, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='flat')
    ax.set_ylim(frame.shape[1], 0)
    ax.set_yticks(np.arange(frame.shape[1]) + 0.5)
    ax.set_yticklabels(ylabels if ylabels is not None else [short(c, 40) for c in frame.columns], fontsize=fontsize)
    ax.xaxis_date()
    date_axis(ax)
    ax.grid(False)
    if cbar:
        plt.colorbar(im, ax=ax, label=cbar, fraction=0.025, pad=0.01)
    return im


def ribbon(ax, series, colors, y=0, h=1):
    """Цветная полоса категорий во времени (состояния, режимы)."""
    ep = episodes(series)
    for v, bars in ep.groupby(ep['value'].astype(str), sort=False):
        c = colors.get(bars['value'].iloc[0])
        if c is not None:
            ax.broken_barh(list(zip(bars.start, bars.end - bars.start)), (y, h), color=c, lw=0)
    ax.set_yticks([])


# ----------------------------------------------------------------------------- группы датчиков

GROUP_RULES = [
    ('Химконтроль', ['ph ', 'рн ', 'пров-ть', 'проводим', 'содержание na', ' na ', 'кремн', 'кислород в пит']),
    ('Подшипники', ['подшипник']),
    ('Положения шиберов и клапанов', ['уп шибера', 'уп клапана', 'положение']),
    ('Электрофильтр и золоудаление', ['поле №', 'струйного насоса', 'питател', 'шнек', 'скрубер', 'скруббер',
                                     'э/ф', 'электроф', 'золо']),
    ('Барабан', ['барабан', 'рнп']),
    ('Мельницы и пылесистема', ['ммт', 'аэросмес', 'горелк', 'псу', 'пыл', 'бункер']),
    ('Топка и факел', ['факел', 'топк', 'возгорани', 'альфа', 'мазут']),
    ('Пароперегреватель и впрыски', ['ширм', 'пароохлад', 'впрыск', 'змеевик', 'перегрет', 'острого пара',
                                    'т пара', 'температура пара', 'ипк', 'паросборн', 'регулятор температуры']),
    ('Питательная вода и ВЭК', ['пит. вод', 'питательн', 'вэк']),
    ('Дутьё и тяга', ['вгд', 'дв-', 'дс-', 'дымосос', 'вентилятор', 'разрежение', 'ток двигателя']),
    ('Воздух и газы', ['взп', 'газ', 'воздух']),
    ('Режим и показатели', ['нагрузк', 'q2', 'q4', 'q5', 'q6', 'кпд', 'задание', 'небаланс']),
]
TEXT_GROUP_RULES = [('Блокировки и защиты', ['блокиров', 'защит']),
                    ('Арматура и механизмы', [''])]


def sensor_group(name, rules=None):
    s = ' ' + re.sub(r'\s+', ' ', str(name).lower()) + ' '
    for g, kws in (rules or GROUP_RULES):
        if any(k in s for k in kws):
            return g
    return 'Прочее'


def find_col(cols, keywords, exclude=()):
    """Первый столбец, в имени которого есть ключевое слово (по приоритету ключевых слов). Имя проверяется как есть и с
    развёрнутым сокращением в начале («F острого пара» = «Расход острого пара»)."""
    for k in keywords:
        for c in cols:
            for s in dict.fromkeys([str(c).lower(), expand_abbrev(c).lower()]):
                if k in s and not any(e in s for e in exclude):
                    return c
    return None


def run_column(cols, run_col=None):
    """Датчик «котёл работает / стоит»: заданный run_col или расход перегретого пара по шаблону RX_RUN (из нескольких —
    самое короткое имя). Не нашёлся — понятная ошибка со списком похожих датчиков (тогда задайте RUN_COL в 01_EDA и 02
    или в UNIT_PARAMS блокнота 00)."""
    if run_col:
        if run_col not in cols:
            raise KeyError(f'RUN_COL «{run_col}» нет среди числовых датчиков агрегата')
        return run_col
    hits = [x for x in cols if name_hit(RX_RUN, x)]
    c = min(hits, key=lambda x: (len(norm_name(x)), str(x))) if hits else None
    if c is None:
        low = lambda x: str(x).lower().translate(LAT2CYR)
        steam = [x for x in cols if re.search(r'(расход|^f |\sf )\s*\S*\s*пар(а|\b)', low(x)) or 'нагруз' in low(x) or 'паропроизв' in low(x)]
        like = steam + [x for x in cols if x not in steam and 'расход' in low(x)]
        raise KeyError('Не найден датчик работы котла «Расход перегретого пара» / «F перегретого пара» (шаблон RX_RUN). '
                       'Задайте RUN_COL — имя датчика, который больше нуля, когда котёл работает. Похожие датчики '
                       '(сначала расход пара и нагрузка): ' + fmt_list(like, 25))
    return c


def run_column_by_data(df, cols, state, min_score=0.8):
    """Датчик «работает / стоит» по данным, если по имени не нашёлся: в размеченной «Работе» он обычно больше четверти своей
    медианы, а в ремонтах и резервах — меньше 10% (около нуля). Оценка = доля «Работы» выше 25% медианы × доля
    ремонтов/резервов ниже 10%. При равной оценке выше датчики расхода пара и нагрузки (по имени).
    Возвращает (лучший датчик или None, таблица кандидатов)."""
    on = state.eq('Работа').to_numpy()
    off = state.isin(['Аварийный ремонт', 'Плановый ремонт', 'Резерв']).to_numpy()
    low = lambda x: str(x).lower().translate(LAT2CYR)
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors='coerce').to_numpy(float)
        a, b = x[on], x[off]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 100 or len(b) < 24:
            continue
        m = float(np.median(a))
        if not m > 0:
            continue
        hi_on, lo_off = float((a > 0.25 * m).mean()), float((b < 0.1 * m).mean())
        prio = 2 if re.search(r'(расход|^f |\sf )\s*\S*\s*пар(а|\b)', low(c)) else 1 if ('нагруз' in low(c) or 'паропроизв' in low(c)) else 0
        rows.append({'датчик': c, 'оценка': round(hi_on * lo_off, 3), 'в «Работе» выше 25% медианы': hi_on,
                     'в ремонтах и резервах около нуля': lo_off, 'медиана в «Работе»': m, 'по имени — расход пара / нагрузка': prio})
    if not rows:
        return None, pd.DataFrame()
    t = pd.DataFrame(rows)
    t['_r'] = (t['оценка'] / 0.05).round()                          # оценки в пределах 0.05 — равны, дальше решает имя
    t = t.sort_values(['_r', 'по имени — расход пара / нагрузка', 'оценка'], ascending=False).drop(columns='_r').reset_index(drop=True)
    best = t['датчик'].iloc[0] if t['оценка'].iloc[0] >= min_score else None
    return best, t


# ----------------------------------------------------------------------------- чтение

def _detect_sep(path):
    with open(path, encoding='utf-8-sig', errors='ignore') as f:
        head = f.readline()
    return max([';', ',', '\t'], key=head.count)


def read_table(path, features=False):
    """Таблица из parquet / csv / xlsx. features=False — без часовых признаков «Датчик__min/__max/__std/__n»."""
    keep = lambda cols: [c for c in cols if features or not FEATURE_RX.search(str(c))]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.parquet')))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if ext == '.parquet':
        import pyarrow.parquet as pq
        return pd.read_parquet(path, columns=keep(pq.read_schema(path).names))
    d = pd.read_csv(path, sep=_detect_sep(path), encoding='utf-8-sig', low_memory=False)
    return d[keep(d.columns)]


def to_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype('float64')
    return pd.to_numeric(s.astype(str).str.replace('\xa0', '', regex=False).str.replace(' ', '', regex=False)
                         .str.replace(',', '.', regex=False), errors='coerce')


def split_types(df, min_num_share=0.9):
    """Числовые, текстовые, служебные (_...) и пустые столбцы."""
    num, txt, service, empty = [], [], [], []
    for c in df.columns:
        if c in (TIME_COL, STATE_COL):
            continue
        if str(c).startswith('_'):
            service.append(c)
            continue
        v = df[c].dropna()
        if len(v) == 0:
            empty.append(c)
        elif pd.api.types.is_numeric_dtype(v) or to_num(v.iloc[:5000]).notna().mean() >= min_num_share:
            num.append(c)
        else:
            txt.append(c)
    return dict(num=num, txt=txt, service=service, empty=empty)


def load_hourly(path, freq='1h'):
    """Часовые средние -> регулярная сетка. Метки времени округляются (06:59:59.94 -> 07:00).
    Признаки внутри часа (Датчик__max и т.п.) не загружаются — для них load_hour_features."""
    raw = read_table(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if TIME_COL not in raw.columns:
        raise ValueError(f'Нет столбца времени «{TIME_COL}»')
    raw[TIME_COL] = pd.to_datetime(raw[TIME_COL], errors='coerce')
    n_nat = int(raw[TIME_COL].isna().sum())
    raw = raw.dropna(subset=[TIME_COL]).sort_values(TIME_COL, kind='stable')
    types = split_types(raw)
    for c in types['num']:
        raw[c] = to_num(raw[c]).astype('float32')
    key = raw[TIME_COL].dt.round(freq)
    g = raw.groupby(key)
    df = g[types['num']].mean()
    other = types['txt'] + types['service'] + ([STATE_COL] if STATE_COL in raw.columns else [])
    if other:
        df = df.join(g[other].last())
    df = df.asfreq(freq)
    df.index.name = TIME_COL
    step = raw[TIME_COL].diff().dropna()
    info = dict(rows_raw=len(raw), no_time=n_nat, merged_dups=int(key.duplicated().sum()),
                step_median=step.median() if len(step) else None,
                big_steps=int((step > 2 * step.median()).sum()) if len(step) else 0)
    return df, types, info


def load_hour_features(path, sensors, kinds=('min', 'max', 'std', 'n')):
    """Признаки внутри часа (из opc_to_wide.py) для датчиков sensors: DataFrame с столбцами «Датчик__max» и т.п.
    Пусто, если в файле их нет (старая часовая таблица)."""
    import pyarrow.parquet as pq
    names = set(pq.read_schema(path).names) if str(path).lower().endswith('.parquet') else set()
    cols = [f'{c}__{k}' for c in sensors for k in kinds if f'{c}__{k}' in names]
    if not cols:
        return pd.DataFrame()
    d = pd.read_parquet(path, columns=[TIME_COL] + cols)
    return d.set_index(pd.to_datetime(d[TIME_COL]).dt.round('1h')).drop(columns=TIME_COL)


# ----------------------------------------------------------------------------- мгновенные данные

def _instant_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.parquet'))) or sorted(glob.glob(os.path.join(path, '*.csv')))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f'Нет файлов в {path}')
    return files


def _iter_file(f, batch_rows):
    if f.lower().endswith('.parquet'):
        import pyarrow.parquet as pq
        # buffer_size: столбцы читаются по частям, а не целиком (важно для одного большого файла)
        for b in pq.ParquetFile(f, buffer_size=1 << 23).iter_batches(batch_size=batch_rows):
            yield b.to_pandas()
    else:
        yield from pd.read_csv(f, sep=_detect_sep(f), encoding='utf-8-sig', chunksize=batch_rows, low_memory=False)


def _probe_instant(files, batch_rows, n_places=12, rows_each=5000):
    """
    Образец строк из разных мест данных (а не только из начала) — для определения типов столбцов.
    Parquet: по одной небольшой порции из n_places групп строк, равномерно по всем файлам (группа целиком
    не читается — в одном большом файле она может быть огромной). CSV: порции с n_places позиций по размеру файла.
    """
    import io
    rows_each = min(rows_each, batch_rows)
    parts = []
    pq_files = [f for f in files if f.lower().endswith('.parquet')]
    if pq_files:
        import pyarrow.parquet as pq
        groups = [(f, rg) for f in pq_files for rg in range(pq.ParquetFile(f).num_row_groups)]
        pick = [groups[i] for i in sorted(set(np.linspace(0, len(groups) - 1, min(n_places, len(groups))).astype(int)))]
        for f, rg in pick:
            it = pq.ParquetFile(f, buffer_size=1 << 23).iter_batches(batch_size=rows_each, row_groups=[int(rg)])
            b = next(it, None)
            if b is not None:
                parts.append(b.to_pandas())
    for f in [f for f in files if not f.lower().endswith('.parquet')]:
        sep, size = _detect_sep(f), os.path.getsize(f)
        with open(f, 'rb') as fh:
            header = fh.readline()
            for pos in np.linspace(len(header), max(size - 1, len(header)), max(1, n_places // max(len(files), 1))).astype(int):
                fh.seek(int(pos))
                if pos > len(header):
                    fh.readline()                               # добираемся до начала строки
                lines = [fh.readline() for _ in range(rows_each)]
                chunk = b''.join(l for l in lines if l)
                if chunk:
                    parts.append(pd.read_csv(io.BytesIO(header + chunk), sep=sep, encoding='utf-8-sig', low_memory=False))
    p = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    p.columns = [str(c).strip() for c in p.columns]
    return p


def _instant_types(files, probe):
    """
    Типы столбцов мгновенных данных. Для parquet — по схеме файла: числовой столбец -> датчик, строковый -> текстовое
    состояние (если только его значения в образце почти все не числа в виде строк). Для CSV — по образцу.
    """
    types = split_types(probe)
    pq_files = [f for f in files if f.lower().endswith('.parquet')]
    if not pq_files:
        return types
    import pyarrow as pa
    import pyarrow.parquet as pq
    ftype = {}                                               # тип столбца — первый непустой по всем файлам
    for f in pq_files:
        for fld in pq.read_schema(f):
            t = fld.type.value_type if pa.types.is_dictionary(fld.type) else fld.type
            c = str(fld.name).strip()
            if c not in ftype or pa.types.is_null(ftype[c]):
                ftype[c] = t
    num, txt = [], []
    for c, t in ftype.items():
        if c in (TIME_COL, STATE_COL) or c.startswith('_') or c.startswith('__index_level'):
            continue
        if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t) or pa.types.is_decimal(t):
            num.append(c)
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            v = probe[c].dropna() if c in probe.columns else pd.Series(dtype=object)
            (num if len(v) and to_num(v.iloc[:5000]).notna().mean() >= 0.9 else txt).append(c)
        elif pa.types.is_null(t):
            v = probe[c].dropna() if c in probe.columns else pd.Series(dtype=object)
            (txt if len(v) and to_num(v.iloc[:5000]).notna().mean() < 0.9 else num).append(c)   # пуст во всех схемах
    service = [c for c in ftype if c.startswith('_') and not c.startswith('__index_level')]
    return dict(num=num, txt=txt, service=service, empty=[])



def _batches(f, batch_rows):
    """Блоки строк файла: для parquet — pyarrow RecordBatch (без перевода в pandas), для CSV — DataFrame."""
    if f.lower().endswith('.parquet'):
        import pyarrow.parquet as pq
        yield from pq.ParquetFile(f, buffer_size=1 << 23).iter_batches(batch_size=batch_rows)
    else:
        for ch in pd.read_csv(f, sep=_detect_sep(f), encoding='utf-8-sig', chunksize=batch_rows, low_memory=False):
            ch.columns = [str(c).strip() for c in ch.columns]
            yield ch


_NAT = np.iinfo('int64').min


def _batch_times(b):
    """Время строк блока в нс (int64; нераспознанное — _NAT)."""
    import pyarrow as pa
    if isinstance(b, pa.RecordBatch):
        names = [str(n).strip() for n in b.schema.names]
        col = b.column(names.index(TIME_COL))
        if pa.types.is_timestamp(col.type):
            t = col.cast(pa.timestamp('ns')).to_numpy(zero_copy_only=False)
        else:
            t = pd.to_datetime(col.to_pandas(), errors='coerce').to_numpy()
    else:
        t = pd.to_datetime(b[TIME_COL], errors='coerce').to_numpy()
    return t.astype('datetime64[ns]').astype('int64')


def _batch_rows(b, ix, num, txt):
    """Выбранные строки блока -> (числа float32, тексты object). Для parquet берутся только эти строки."""
    import pyarrow as pa
    if isinstance(b, pa.RecordBatch):
        sub = b.take(pa.array(np.asarray(ix, 'int64'))).to_pandas()
        sub.columns = [str(c).strip() for c in sub.columns]
    else:
        sub = b.iloc[np.asarray(ix)]
    Vd = sub.reindex(columns=num)
    for c in num:
        if Vd[c].dtype == object:
            Vd[c] = to_num(Vd[c])
    return Vd.to_numpy('float32'), (sub.reindex(columns=txt).to_numpy(object) if txt else None)


def snapshot_instant(path, step='5min', max_gap='2h', ffill=False, batch_rows=200_000, verbose=True):
    """
    Мгновенные (протянутые по событиям) значения -> регулярная сетка «снимков» с шагом step, БЕЗ усреднения:
    в каждый момент сетки берётся сырое значение, действовавшее в этот момент (последняя строка не позже него).

    Зачем сетка: исходные строки идут неравномерно — строка появляется, когда меняется хоть один датчик. Если учить
    модель прямо на строках, периоды частых изменений (переходные процессы) получили бы в десятки раз больший вес,
    чем спокойная работа, а окна «последние N шагов» были бы разной длины по времени. Снимок с малым шагом сохраняет
    мгновенные значения и при этом даёт каждой минуте работы одинаковый вес.

    Файл читается блоками; из каждого блока в pandas переводятся только строки, попавшие на моменты сетки, поэтому
    даже многогигабайтный файл обрабатывается без загрузки в память целиком. Если строк нет дольше max_gap — данных
    нет (NaN), а не «значение держится». Текстовые состояния — тоже значение в момент сетки.
    Возвращает DataFrame (числа — float32, тексты — category) и сведения о чтении.
    """
    files = _instant_files(path)
    probe = _probe_instant(files, 20_000)
    types = _instant_types(files, probe)
    num = types['num'] + types['empty']
    txt = types['txt'] + types['service'] + ([STATE_COL] if STATE_COL in probe.columns else [])
    s_ns, gap_ns = pd.Timedelta(step).value, pd.Timedelta(max_gap).value
    codes = [dict() for _ in txt]              # значение -> код, по каждому текстовому столбцу

    def encode(S):
        C = np.full(S.shape, -1, dtype='int16')
        for j in range(S.shape[1]):
            loc, uniq = pd.factorize(S[:, j])
            if len(uniq):
                gmap = np.array([codes[j].setdefault(str(u).strip(), len(codes[j])) for u in uniq], dtype='int16')
                C[:, j] = np.where(loc >= 0, gmap[np.maximum(loc, 0)], -1)
        return C

    G_parts, V_parts, C_parts = [], [], []
    carry, next_g = None, None                 # последняя строка предыдущего блока: (t, V, C); следующий момент сетки
    n_rows = n_back = n_batches = 0
    t_start = time.time()
    for fi, f in enumerate(files):
        for b in _batches(f, batch_rows):
            tv = _batch_times(b)
            order = np.flatnonzero(tv != _NAT)
            if not np.all(np.diff(tv[order]) >= 0):
                order = order[np.argsort(tv[order], kind='stable')]
            if carry is not None:
                back = tv[order] < carry[0]
                if back.any():                                  # строки «из прошлого» (нарушен порядок файлов)
                    n_back += int(back.sum())
                    order = order[~back]
            if not len(order):
                continue
            tvs = tv[order]
            n_rows += len(order)
            full_V = full_C = None
            if ffill:                                           # пустая ячейка = «не изменилось»: нужны все строки
                full_V, S_ = _batch_rows(b, order, num, txt)
                full_C = encode(S_) if txt else None
                if carry is not None:
                    full_V = np.vstack([carry[1], full_V]); full_C = np.vstack([carry[2], full_C]) if txt else None
                full_V = pd.DataFrame(full_V).ffill().to_numpy('float32')[(carry is not None):]
                if txt:
                    full_C = pd.DataFrame(full_C).replace(-1, np.nan).ffill().fillna(-1).to_numpy('int16')[(carry is not None):]
            if next_g is None:
                next_g = (tvs[0] // s_ns + 1) * s_ns
            if tvs[-1] > next_g:                                # моменты сетки до последней строки блока
                g = np.arange(next_g, tvs[-1], s_ns, dtype='int64')
                r = np.searchsorted(tvs, g, side='right') - 1   # -1 — действует строка из прошлого блока
                use = np.unique(r[r >= 0])
                if full_V is not None:
                    Vb, Cb = full_V[use], (full_C[use] if txt else None)
                else:
                    Vb, Sb = _batch_rows(b, order[use], num, txt)
                    Cb = encode(Sb) if txt else None
                inb = r >= 0
                pos = np.searchsorted(use, np.maximum(r, 0))
                Vg = np.full((len(g), len(num)), np.nan, dtype='float32')
                Vg[inb] = Vb[pos[inb]]
                t_src = np.where(inb, tvs[np.maximum(r, 0)], carry[0] if carry is not None else _NAT)
                if carry is not None:
                    Vg[~inb] = carry[1]
                stale = (g - t_src) > gap_ns
                Vg[stale] = np.nan
                G_parts.append(g)
                V_parts.append(Vg)
                if txt:
                    Cg = np.full((len(g), len(txt)), -1, dtype='int16')
                    Cg[inb] = Cb[pos[inb]]
                    if carry is not None:
                        Cg[~inb] = carry[2]
                    Cg[stale] = -1
                    C_parts.append(Cg)
                next_g = int(g[-1]) + s_ns
            if full_V is not None:
                carry = (tvs[-1], full_V[-1:].copy(), full_C[-1:].copy() if txt else None)
            else:
                Vl, Sl = _batch_rows(b, order[-1:], num, txt)
                carry = (tvs[-1], Vl, encode(Sl) if txt else None)
            n_batches += 1
            if verbose and n_batches % 50 == 0:
                print(f'  файл {fi + 1}/{len(files)}, блоков {n_batches}, строк {n_rows:,}, '
                      f'{(time.time() - t_start) / 60:.1f} мин', flush=True)
    if carry is None:
        raise ValueError('Мгновенные данные пусты')
    if next_g is not None and carry[0] >= next_g:
        g = np.arange(next_g, carry[0] + 1, s_ns, dtype='int64')
        G_parts.append(g)
        V_parts.append(np.repeat(carry[1], len(g), axis=0))
        if txt:
            C_parts.append(np.repeat(carry[2], len(g), axis=0))
    idx = pd.DatetimeIndex(pd.to_datetime(np.concatenate(G_parts)), name=TIME_COL)
    out = pd.DataFrame(np.vstack(V_parts), index=idx, columns=num)
    if txt:
        Cm = np.vstack(C_parts)
        for j, c in enumerate(txt):
            out[c] = pd.Categorical.from_codes(Cm[:, j].astype('int32'), categories=list(codes[j].keys()))
    info = dict(rows=n_rows, files=len(files), batches=n_batches, back_rows=n_back, step=str(step),
                minutes=(time.time() - t_start) / 60, num=types['num'], txt=types['txt'],
                service=types['service'], empty=types['empty'])
    return out, info


def instant_profile(path, n_places=12, rows_each=20_000):
    """
    Как часто обновляются мгновенные данные — чтобы выбрать шаг сетки снимков, не загружая файл целиком.
    Берутся n_places непрерывных кусков по rows_each строк из разных мест файла(ов); для каждого числового датчика —
    сколько времени в среднем проходит между изменениями его значения. Число строк — из метаданных parquet.
    """
    import pyarrow.parquet as pq
    files = _instant_files(path)
    pq_files = [f for f in files if f.lower().endswith('.parquet')]
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in pq_files) if pq_files else None
    size_gb = sum(os.path.getsize(f) for f in files) / 2 ** 30
    parts = []
    if pq_files:
        groups = [(f, rg) for f in pq_files for rg in range(pq.ParquetFile(f).num_row_groups)]
        pick = [groups[i] for i in sorted(set(np.linspace(0, len(groups) - 1, min(n_places, len(groups))).astype(int)))]
        for f, rg in pick:
            b = next(pq.ParquetFile(f, buffer_size=1 << 23).iter_batches(batch_size=rows_each, row_groups=[int(rg)]), None)
            if b is not None:
                d = b.to_pandas(); d.columns = [str(c).strip() for c in d.columns]; parts.append(d)
        t0 = pd.to_datetime(pq.ParquetFile(pq_files[0]).read_row_group(0, columns=None).column(0).to_pandas()).min()
        lastf = pq.ParquetFile(pq_files[-1])
        t1 = pd.to_datetime(lastf.read_row_group(lastf.num_row_groups - 1).column(0).to_pandas()).max()
    else:
        for i, ch in enumerate(_batches(files[0], rows_each)):
            parts.append(ch)
            if i >= n_places:
                break
        t0 = t1 = None
    types = split_types(pd.concat(parts, ignore_index=True)) if parts else dict(num=[], txt=[])
    row_dt, ch_int = [], {}
    for d in parts:
        t = pd.to_datetime(d[TIME_COL], errors='coerce')
        ok = t.notna()
        d, t = d[ok.to_numpy()], t[ok]
        if len(d) < 100:
            continue
        span = (t.max() - t.min()).total_seconds()
        if span <= 0:
            continue
        row_dt.append(t.diff().dt.total_seconds().median())
        V = d[types['num']].apply(to_num)
        nch = (V.diff().abs() > 0).sum()
        for c in types['num']:
            ch_int.setdefault(c, []).append(span / nch[c] if nch[c] > 0 else np.inf)
    change_s = pd.Series({c: float(np.median(v)) for c, v in ch_int.items()})
    fin = change_s[np.isfinite(change_s)]
    med = float(fin.median()) if len(fin) else 300.0
    rec = next((s for s in (60, 120, 300, 600) if s >= med), 600)
    n_num, n_txt = len(types['num']), len(types['txt'])
    span_h = (t1 - t0).total_seconds() / 3600 if t0 is not None and t1 is not None else None
    mem = {f'{s // 60} мин': (span_h * 3600 / s) * (n_num * 4 + n_txt * 2) / 2 ** 30 for s in (60, 300, 600)} if span_h else {}
    return dict(files=len(files), size_gb=size_gb, rows=total, t0=t0, t1=t1,
                rows_per_hour=(total / span_h) if total and span_h else None,
                row_dt_s=float(np.median(row_dt)) if row_dt else None, change_s=change_s,
                recommended_step=f'{rec // 60}min', grid_gb=mem, n_num=n_num, n_txt=n_txt)


def load_instant_grid(path, step='5min', cache_dir='cache', refresh=False, **kw):
    """snapshot_instant с кэшем в parquet (повторный запуск — секунды)."""
    stamp = max(int(os.path.getmtime(f)) for f in _instant_files(path))     # данные пересобраны — кэш другой
    tag = re.sub(r'[^\w]+', '_', os.path.normpath(path))[-60:] + '_' + re.sub(r'[^\w]+', '', str(step)) + f'_{stamp}'
    cdir = os.path.join(cache_dir, f'snap_{tag}')
    meta_f, data_f = os.path.join(cdir, 'info.json'), os.path.join(cdir, 'grid.parquet')
    if not refresh and os.path.exists(meta_f) and os.path.exists(data_f):
        info = json.load(open(meta_f, encoding='utf-8'))
        info['from_cache'] = True
        return pd.read_parquet(data_f), info
    out, info = snapshot_instant(path, step=step, **kw)
    os.makedirs(cdir, exist_ok=True)
    out.to_parquet(data_f)
    json.dump(info, open(meta_f, 'w', encoding='utf-8'), ensure_ascii=False, default=str)
    info['from_cache'] = False
    return out, info


def load_grid(kind, hourly_path, instant_path=None, step='5min', cache_dir='cache', **kw):
    """
    Данные для модели: kind='hourly' — часовые средние (сетка 1 ч); kind='instant' — снимки мгновенных значений
    с шагом step (без усреднения). Возвращает (df, types, info); типы столбцов — как у часовых данных.
    """
    if kind == 'hourly':
        df, types, info = load_hourly(hourly_path, '1h')
        return df, types, dict(info, step='1h', from_cache=False)
    df, info = load_instant_grid(instant_path, step, cache_dir, **kw)
    types = dict(num=[c for c in info['num'] + info['empty'] if c in df.columns], txt=[c for c in info['txt'] if c in df.columns],
                 service=[c for c in info['service'] if c in df.columns], empty=[])
    return df, types, info


# ----------------------------------------------------------------------------- заявки на ремонт

def load_repairs(path):
    """Заявки агрегата из repairs.csv, который пишет opc_to_wide.py: номер, тип, категория, статус, начало, окончание,
    присвоенное состояние (по типу и категории), учтена ли, содержание работ."""
    R = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    for c in ('начало', 'окончание'):
        R[c] = pd.to_datetime(R[c], errors='coerce')
    if 'учтена' in R:
        R['учтена'] = R['учтена'].astype(str).str.lower().isin(['true', '1', 'да'])
    return R


# ----------------------------------------------------------------------------- состояния, ремонты, режим

def episodes(s):
    """Непрерывные участки одинакового значения: value, start, end, hours."""
    s = s.astype(object).where(s.notna(), None)
    grp = (s != s.shift()).cumsum()
    ix = s.index.to_series()
    step = s.index[1] - s.index[0] if len(s) > 1 else pd.Timedelta('1h')
    ep = pd.DataFrame({'value': s.groupby(grp).first(), 'start': ix.groupby(grp).first(),
                       'end': ix.groupby(grp).last() + step}).reset_index(drop=True)
    ep['hours'] = (ep.end - ep.start) / pd.Timedelta('1h')
    return ep


def repair_events(state, merge_gap_h=24):
    """Ремонты и резервы из меток: соседние эпизоды одного типа с разрывом < merge_gap_h ч объединяются."""
    ep = episodes(state)
    out = []
    for r in ep[ep.value.isin([ST_FAIL, ST_PLAN, ST_RES])].itertuples():
        if out and out[-1]['state'] == r.value and (r.start - out[-1]['end']) <= pd.Timedelta(hours=merge_gap_h):
            out[-1]['end'] = r.end
            out[-1]['n_ep'] += 1
        else:
            out.append(dict(state=r.value, start=r.start, end=r.end, n_ep=1))
    ev = pd.DataFrame(out, columns=['state', 'start', 'end', 'n_ep'])
    ev['type'] = ev.state.map({ST_FAIL: EV_FAIL, ST_PLAN: EV_PLAN, ST_RES: EV_RES})
    ev['hours'] = (ev.end - ev.start) / pd.Timedelta('1h')
    return ev.reset_index(drop=True)


def run_threshold(x, labels=None, frac=0.25):
    """Порог «котёл работает»: доля от медианы показателя в размеченной работе."""
    ref = x[labels.eq(ST_WORK)] if labels is not None else x
    ref = ref[ref > 0].dropna()
    return float(frac * ref.median()) if len(ref) else float(np.nanmedian(x))


def detect_mode(x, thr, startup_h=24, prestop_h=3, min_stop_h=2, ffill_h=3):
    """
    Фактический режим по показателю нагрузки x: Останов / Пуск (первые startup_h ч после включения) /
    Перед остановом (последние prestop_h ч) / Работа. Остановы короче min_stop_h ч не считаются.
    Возвращает режим, часы с начала работы и часы до останова.
    """
    running = (x.ffill(limit=ffill_h) > thr)
    grp = (running != running.shift()).cumsum()
    size = running.groupby(grp).transform('size')
    first_grp, last_grp = grp.iloc[0], grp.iloc[-1]
    short_stop = ~running & (size < min_stop_h) & (grp != first_grp) & (grp != last_grp)
    running = running | short_stop
    grp = (running != running.shift()).cumsum()
    since = running.groupby(grp).cumcount().astype(float)
    to_stop = running.groupby(grp).cumcount(ascending=False).astype(float)
    first_grp, last_grp = grp.iloc[0], grp.iloc[-1]
    since[grp.eq(first_grp)] = np.inf          # начало данных — момент пуска неизвестен
    to_stop[grp.eq(last_grp)] = np.inf         # конец данных — останов неизвестен
    since[~running] = np.nan
    to_stop[~running] = np.nan
    mode = pd.Series(np.where(running, M_RUN, M_STOP), index=x.index, dtype=object)
    mode[running & (to_stop < prestop_h)] = M_PRESTOP
    mode[running & (since < startup_h)] = M_START
    return mode, since, to_stop


def change_rate(s, run, per_h=1):
    """Доля часов работы, когда значение изменилось за час (для сетки мельче часа — по значениям раз в час)."""
    v = s.where(run)
    if per_h > 1:
        v = v.iloc[::per_h]
    pairs = v.notna() & v.shift().notna()
    return float((v != v.shift())[pairs].mean()) if pairs.sum() > 48 else np.nan


def frozen_mask(s, run, min_h=24, min_change_rate=0.9, per_h=1):
    """
    Залипание: одно и то же значение ≥ min_h часов подряд при работе котла. Проверяется только у датчиков,
    которые обычно меняются каждый час (доля часов с изменением ≥ min_change_rate): у положений шиберов, заданий
    и т.п. неизменное значение сутками — норма, а не неисправность. per_h — шагов сетки в часе.
    """
    cr = change_rate(s, run, per_h)
    if not np.isfinite(cr) or cr < min_change_rate:
        return pd.Series(False, index=s.index)
    v = s.where(run)
    key = (v != v.shift()).cumsum()
    size = v.groupby(key).transform('size')
    return v.notna() & (size >= min_h * per_h)


def is_step_like(s, run, max_change=0.05, max_unique_share=0.01, per_h=1):
    """Ступенчатый параметр: меняется редко (< 5% часов) и принимает мало значений (< 1% уникальных), —
    например, вводимые вручную или по лабораторному анализу потери (q4, q6). Это не измерение процесса:
    каждое изменение для модели выглядело бы «небывалым»."""
    v = s[run].dropna()
    if len(v) < 100:
        return False
    cr = change_rate(s, run, per_h)
    return bool(np.isfinite(cr) and cr < max_change and v.nunique() / len(v) < max_unique_share)


# ----------------------------------------------------------------------------- общее состояние и метод аналогов

def _nan_quantiles_sorted(A, qs):
    """Квантили по оси 1 с пропусками: A (n, k, p) -> (len(qs), n, p) и число непустых (n, p)."""
    B = np.ascontiguousarray(A.transpose(0, 2, 1))   # (n, p, k): сортировка по непрерывной оси быстрее
    B.sort(axis=-1)                                  # NaN уходят в конец
    cnt = (~np.isnan(B)).sum(axis=-1)
    out = []
    for q in qs:
        pos = np.clip(q * (cnt - 1), 0, None)
        lo = np.floor(pos).astype(np.intp)
        hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
        w = pos - lo
        vlo = np.take_along_axis(B, lo[..., None], axis=-1)[..., 0]
        vhi = np.take_along_axis(B, hi[..., None], axis=-1)[..., 0]
        v = vlo * (1 - w) + vhi * w
        v[cnt == 0] = np.nan
        out.append(v)
    return np.array(out), cnt


class StateSpace:
    """
    «Общее состояние котла» по данным — без ручного выбора режимных датчиков.

    Все числовые датчики (в робастных единицах: (x − медиана) / нормальный разброс) и состояния оборудования
    (One-hot: столбец «механизм = значение», 1 или 0) сжимаются методом главных компонент (PCA) до нескольких
    координат. Они описывают основную СОВМЕСТНУЮ изменчивость котла — то, что меняет сразу много датчиков: нагрузку,
    погоду, набор работающего оборудования. Какие датчики входят в каждую координату, видно из loadings().

    Координаты момента считаются устойчиво: пропуски не участвуют, а датчики, явно выбивающиеся из общей картины
    (остаток > robust_thr своих обычных разбросов), при втором проходе не учитываются — иначе неисправный датчик
    сдвигал бы оценку состояния и маскировал сам себя.
    """

    def __init__(self, var=0.8, max_pc=10, min_pc=2, state_weight=1.0, min_share=0.01, max_levels=5,
                 robust_thr=4.0, clip=10.0, max_fit=30_000, seed=0):
        self.var, self.max_pc, self.min_pc = var, max_pc, min_pc
        self.state_weight, self.min_share, self.max_levels = state_weight, min_share, max_levels
        self.robust_thr, self.clip, self.max_fit, self.seed = robust_thr, clip, max_fit, seed

    def _matrix(self, X, T=None):
        U = ((X[self.xcols].astype('float64') - self.med) / self.sd).clip(-self.clip, self.clip)
        parts = [U.to_numpy()]
        for c, lv in self.levels:
            v = T[c] if T is not None and c in T.columns else pd.Series(np.nan, index=X.index)
            known = v.notna().to_numpy()
            vv = v.astype(object).to_numpy()
            for l_ in lv:
                parts.append(np.where(known, (vv == l_).astype(float), np.nan)[:, None] * self.state_weight)
        return np.hstack(parts)

    def fit(self, X, T=None):
        from sklearn.decomposition import PCA
        self.xcols = list(X.columns)
        q = X.astype('float64').quantile([.25, .5, .75])
        self.med = q.loc[.5]
        sd = (q.loc[.75] - q.loc[.25]) / 1.349
        self.sd = sd.where(sd > 0, X.astype('float64').std()).fillna(1.0).replace(0, 1.0)
        self.levels = []
        if T is not None:
            for c in T.columns:
                vc = T[c].astype(object).value_counts(normalize=True)
                lv = [v for v in vc.index[:self.max_levels] if vc[v] >= self.min_share]
                if len(lv) >= 2:                                   # состояние, которое реально меняется
                    self.levels.append((c, lv))
        self.cols = self.xcols + [f'{c}={l_}' for c, lv in self.levels for l_ in lv]
        A = self._matrix(X, T)
        self.mu = np.nan_to_num(np.nanmean(A, axis=0))
        rng = np.random.default_rng(self.seed)
        rows = np.arange(len(A)) if len(A) <= self.max_fit else np.sort(rng.choice(len(A), self.max_fit, replace=False))
        A0 = np.nan_to_num(A[rows] - self.mu)
        n_comp = int(min(self.max_pc + 5, A0.shape[1], len(A0) - 1))
        pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=self.seed).fit(A0)
        cum = np.cumsum(pca.explained_variance_ratio_)
        r = int(np.clip(np.searchsorted(cum, self.var) + 1, self.min_pc, min(self.max_pc, n_comp)))
        self.V = pca.components_[:r].T.copy()                     # d × r, столбцы ортонормированы
        self.ratio = pca.explained_variance_ratio_[:r]
        self.names = [f'ГК{i + 1}' for i in range(r)]
        _, res = self._project(A[rows])
        self.rsd = np.nan_to_num(1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res, axis=0)), axis=0), nan=1.0)
        self.rsd = np.maximum(self.rsd, 0.05)
        return self

    def _project(self, A, use=None, chunk=10_000, ridge=1e-3):
        """Координаты по имеющимся (и не исключённым) столбцам: взвешенный МНК, у каждой строки свой набор столбцов."""
        V = self.V
        d, r = V.shape
        P = (V[:, :, None] * V[:, None, :]).reshape(d, r * r)
        S = np.empty((len(A), r))
        R = np.empty_like(A)
        for a in range(0, len(A), chunk):
            Ac = A[a:a + chunk] - self.mu
            w = ~np.isnan(Ac)
            if use is not None:
                w &= use[a:a + chunk]
            Ac0 = np.where(w, Ac, 0.0)
            G = (np.eye(r).ravel()[None, :] - (~w).astype(float) @ P).reshape(-1, r, r) + ridge * np.eye(r)
            rhs = Ac0 @ V
            s = np.linalg.solve(G, rhs[..., None])[..., 0]
            S[a:a + chunk] = s
            R[a:a + chunk] = Ac - s @ V.T
        return S, R

    def transform(self, X, T=None, robust=True):
        """Координаты «общего состояния» для строк X (и состояний T): DataFrame ГК1…ГКr."""
        A = self._matrix(X, T)
        S, R = self._project(A)
        if robust:
            bad = np.abs(R) > self.robust_thr * self.rsd
            redo = bad.any(axis=1)
            if redo.any():
                S[redo], _ = self._project(A[redo], use=~bad[redo])
        return pd.DataFrame(S, index=X.index, columns=self.names)

    def loadings(self, top=6):
        """Из чего состоит каждая координата: столбцы с наибольшими весами (знак — направление)."""
        rows = []
        for i, nm in enumerate(self.names):
            w = pd.Series(self.V[:, i], index=self.cols)
            big = w.abs().nlargest(top).index
            rows.append({'координата': nm, 'доля дисперсии': float(self.ratio[i]),
                         'главные датчики': ', '.join(f'{short(c, 34)} ({w[c]:+.2f})' for c in big)})
        return pd.DataFrame(rows)


def neighbor_radius(R, k=100, n_sample=3000, seed=0):
    """Типичное расстояние до k-го ближайшего соседа (медиана по выборке строк) — «масштаб похожести»."""
    from sklearn.neighbors import NearestNeighbors
    A = np.nan_to_num(np.asarray(R, float))
    k = int(min(k, len(A) - 1))
    if k < 1:
        return 1.0
    rng = np.random.default_rng(seed)
    q = A[rng.choice(len(A), min(n_sample, len(A)), replace=False)]
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(q)
    return float(np.median(d[:, -1])) or 1.0


class AnalogModel:
    """
    «Метод аналогов» — ожидаемое значение каждого датчика по похожим моментам нормальной работы.

    R — координаты «общего состояния» (ГК1…ГКr из StateSpace; по желанию — ещё время), X — датчики.
    1. Линейная часть: для каждого датчика — линейная зависимость от координат состояния (гребневая регрессия
       по норме). Убирает основную «гладкую» зависимость и не даёт смещения на краях (самая высокая нагрузка и т.п.).
    2. Аналоги: k ближайших по состоянию моментов нормы. По ним — медиана и межквартильный размах того, что
       осталось после линейной части (нелинейности, влияние состояний оборудования).
    Ожидание = линейная часть + медиана аналогов; разброс = размах / 1.349 (не меньше floor_frac общего разброса
    датчика и range_frac его рабочего диапазона). z = (факт − ожидание) / разброс.
    dist — среднее расстояние до аналогов по координатам состояния: «насколько момент вообще похож на норму».

    scales — масштаб отдельных координат (например, {'время': 20}: 20 суток = 1 единица расстояния);
    координаты из no_linear не входят в линейную часть и в dist (время — чтобы не экстраполировать тренд).
    """

    def __init__(self, k=100, floor_frac=0.1, min_valid=10, linear=True, scales=None, no_linear=('время',),
                 max_elems=4_000_000, range_frac=0.02, max_train=None, seed=0):
        self.k, self.floor_frac, self.min_valid, self.linear = k, floor_frac, min_valid, linear
        self.scales, self.no_linear, self.max_elems = scales or {}, set(no_linear), max_elems
        self.range_frac, self.max_train, self.seed = range_frac, max_train, seed

    def _prep_R(self, R):
        return ((R[self.rcols] - self.r_med) / self.r_scale).fillna(0.0).to_numpy('float64')

    def _lin_design(self, R):
        L = (R[self.lcols] - self.r_med[self.lcols]).fillna(0.0).to_numpy('float64')
        return np.column_stack([np.ones(len(L)), L])

    def fit(self, R, X):
        from sklearn.neighbors import NearestNeighbors
        self.rcols = list(R.columns)
        self.lcols = [c for c in self.rcols if c not in self.no_linear]
        self.didx = [self.rcols.index(c) for c in self.lcols]
        self.r_med = R.median()
        self.r_scale = pd.Series(1.0, index=self.rcols)
        for c, v in self.scales.items():
            if c in self.r_scale.index:
                self.r_scale[c] = v
        self.cols = list(X.columns)
        Xv = X.to_numpy('float64')
        qx = X.quantile([0.005, 0.25, 0.75, 0.995])
        glob_sc = ((qx.loc[0.75] - qx.loc[0.25]) / 1.349).replace(0, np.nan).fillna(X.std()).fillna(0)
        span = (qx.loc[0.995] - qx.loc[0.005]).fillna(0)          # рабочий размах датчика
        # разброс снизу: 10% обычного разброса и 2% рабочего размаха (у почти постоянного датчика, например шибера
        # на 99–100%, иначе изменение на доли процента давало бы огромный z)
        self.floor = np.maximum(np.maximum(glob_sc.to_numpy() * self.floor_frac, span.to_numpy() * self.range_frac), 1e-6)
        # датчик не менялся на обучении (например, ступенчатый параметр) — судить не по чему: z = NaN
        self.floor[(span.to_numpy() <= 0) & (glob_sc.to_numpy() <= 0)] = np.nan
        self.B = np.zeros((len(self.lcols) + 1, Xv.shape[1]))
        if self.linear:
            L = self._lin_design(R)
            lam = 1e-3 * len(L) * np.eye(L.shape[1]); lam[0, 0] = 0
            ok = ~np.isnan(Xv)
            full_cols = ok.all(axis=0)
            if full_cols.any():
                self.B[:, full_cols] = np.linalg.solve(L.T @ L + lam, L.T @ Xv[:, full_cols])
            for j in np.flatnonzero(~full_cols):
                m = ok[:, j]
                if m.sum() > 3 * L.shape[1]:
                    Lm = L[m]
                    self.B[:, j] = np.linalg.solve(Lm.T @ Lm + lam * m.mean(), Lm.T @ Xv[m, j])
                elif m.any():
                    self.B[0, j] = np.nanmedian(Xv[:, j])
            Xv = Xv - L @ self.B
        Rn = self._prep_R(R)
        if self.max_train and len(Rn) > self.max_train:          # «память» аналогов прореживается равномерно по времени
            keep = np.linspace(0, len(Rn) - 1, int(self.max_train)).astype(int)
            Rn, Xv = Rn[keep], Xv[keep]
        self.Xtr, self.Rtr = Xv, Rn
        self.kk = min(self.k, len(Rn))
        self.nn = NearestNeighbors(n_neighbors=self.kk).fit(Rn)
        return self

    def expected(self, R, X=None):
        """Ожидание, разброс и z для строк R (и фактических значений X); в self.dist — расстояние до аналогов."""
        Rq = self._prep_R(R)
        n, p = len(Rq), len(self.cols)
        med, sc = np.full((n, p), np.nan), np.full((n, p), np.nan)
        dist = np.full(n, np.nan)
        batch = max(10, int(self.max_elems // max(self.kk * p, 1)))
        for s in range(0, n, batch):
            rq = np.arange(s, min(s + batch, n))
            _, nb = self.nn.kneighbors(Rq[rq])
            (q25, q50, q75), cnt = _nan_quantiles_sorted(self.Xtr[nb], [0.25, 0.5, 0.75])
            q50[cnt < self.min_valid] = np.nan
            med[rq] = q50
            sc[rq] = np.maximum((q75 - q25) / 1.349, self.floor)
            D = Rq[rq][:, None, self.didx] - self.Rtr[nb][:, :, self.didx]
            dist[rq] = np.sqrt((D ** 2).sum(axis=-1)).mean(axis=1)
        if self.linear:
            med = med + self._lin_design(R) @ self.B
        self.dist = pd.Series(dist, index=R.index)
        med = pd.DataFrame(med, index=R.index, columns=self.cols)
        sc = pd.DataFrame(sc, index=R.index, columns=self.cols)
        z = (X[self.cols] - med) / sc if X is not None else None
        return med, sc, z


def analog_crossfit(R, X, blocks, k=100, floor_frac=0.1, **kw):
    """z для обучающих строк без подглядывания: аналоги для блока b ищутся только в остальных блоках.
    Возвращает ожидание, разброс, z и расстояние до аналогов."""
    med = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    sc, z = med.copy(), med.copy()
    dist = pd.Series(np.nan, index=X.index)
    for b in np.unique(blocks):
        tr, te = blocks != b, blocks == b
        if tr.sum() < 50 or te.sum() == 0:
            continue
        m = AnalogModel(k, floor_frac, **kw).fit(R[tr], X[tr])
        mb, sb, zb = m.expected(R[te], X[te])
        med.loc[te], sc.loc[te], z.loc[te] = mb.to_numpy(), sb.to_numpy(), zb.to_numpy()
        dist[te] = m.dist.to_numpy()
    return med, sc, z, dist


def time_blocks(index, n_blocks):
    """Номер временного блока (подряд идущие куски равной длины по числу строк)."""
    n = len(index)
    return np.minimum((np.arange(n) * n_blocks) // max(n, 1), n_blocks - 1)


def fmt_df(d, fmts):
    """Копия таблицы с отформатированными числами (без Styler: работает при повторяющихся подписях)."""
    d = d.copy()
    for c, f in fmts.items():
        if c in d.columns:
            d[c] = d[c].map(lambda v: '—' if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
                            else f.format(v))
    return d


def days_to_next(index, starts):
    """Для каждого момента — сколько суток до ближайшего следующего события (inf, если событий дальше нет)."""
    fs = np.sort(pd.to_datetime(pd.Series(starts)).dropna().to_numpy())
    t = pd.DatetimeIndex(index).to_numpy()
    if not len(fs):
        return pd.Series(np.inf, index=index)
    pos = np.searchsorted(fs, t, side='right')
    ok = pos < len(fs)
    out = np.full(len(t), np.inf)
    out[ok] = (fs[pos[ok]] - t[ok]) / np.timedelta64(1, 'h') / 24
    return pd.Series(out, index=index)


def days_to_nearest(index, starts):
    """Для каждого момента — сколько суток до ближайшего события (до или после)."""
    fs = np.sort(pd.to_datetime(pd.Series(starts)).dropna().to_numpy())
    t = pd.DatetimeIndex(index).to_numpy()
    if not len(fs):
        return pd.Series(np.inf, index=index)
    d = np.abs((t[:, None] - fs[None, :]) / np.timedelta64(1, 'h')).min(axis=1) / 24
    return pd.Series(d, index=index)



# ----------------------------------------------------------------------------- модели для ноутбука 02

class Prep:
    """
    Подготовка признаков для моделей.
    Числовые датчики: (x − медиана) / нормальный разброс (межквартильный размах / 1.35), обрезка ±10, пропуск = 0
    («как обычно»). Текстовые датчики: булевы признаки «механизм не в своём обычном состоянии» (1/0). Берутся только
    те, что в норме хотя бы min_share времени бывают не в обычном состоянии (остальные почти не меняются), и из почти
    одинаковых булевых (|ρ| ≥ dedup, например задвижки, которые всегда двигаются вместе) остаётся одна.
    """

    def __init__(self, min_share=0.02, dedup=0.95, cols=None):
        """cols — явный список текстовых датчиков-состояний (обычное состояние — самое частое в данных обучения);
        None — отбор автоматически по min_share."""
        self.min_share, self.dedup, self.cols = min_share, dedup, cols

    def fit(self, X, T=None):
        self.num = list(X.columns)
        q = X.astype('float64').quantile([.25, .5, .75])
        self.med = q.loc[.5].fillna(0).to_numpy()
        sd = ((q.loc[.75] - q.loc[.25]) / 1.349).to_numpy()
        std = X.astype('float64').std().fillna(1).to_numpy()
        self.sd = np.where(sd > 0, sd, np.where(std > 0, std, 1.0))
        self.bools = []
        if T is not None and T.shape[1]:
            cand = []
            for c in (T.columns if self.cols is None else [c for c in self.cols if c in T.columns]):
                vc = T[c].astype(object).value_counts(normalize=True)
                if len(vc) >= 2 and (self.cols is not None or 1 - vc.iloc[0] >= self.min_share):
                    cand.append((c, vc.index[0]))
            if cand:
                B = self._bools(T, cand)
                C = np.nan_to_num(np.corrcoef(B.T)) if B.shape[1] > 1 else np.ones((1, 1))
                keep = []
                for j in range(len(cand)):
                    if all(abs(C[j, k]) < self.dedup for k in keep):
                        keep.append(j)
                self.bools = [cand[j] for j in keep]
        return self

    @staticmethod
    def _bools(T, pairs):
        out = np.zeros((len(T), len(pairs)), dtype='float32')
        for j, (c, main) in enumerate(pairs):
            v = T[c].astype(object).to_numpy()
            out[:, j] = np.where(pd.isna(v), 0.0, (v != main).astype('float32'))
        return out

    @property
    def names(self):
        return self.num + [f'{c} ≠ {m}' for c, m in self.bools]

    def transform(self, X, T=None):
        Z = (X[self.num].to_numpy('float64') - self.med) / self.sd
        Z = np.nan_to_num(np.clip(Z, -10, 10)).astype('float32')
        if self.bools:
            Z = np.hstack([Z, self._bools(T, self.bools)])
        return Z


class PCADet:
    """PCA: сжимает норму до главных компонент (доля дисперсии var) и восстанавливает; ошибка — по датчикам."""
    name = 'PCA'

    def __init__(self, var=0.9):
        self.var = var

    def fit(self, F, prep):
        from sklearn.decomposition import PCA
        self.n = len(prep.num)
        self.pca = PCA(n_components=self.var, svd_solver='full').fit(F)
        return self

    def errors(self, F, prep):
        return np.abs(F - self.pca.inverse_transform(self.pca.transform(F)))[:, :self.n]


class IForestDet:
    """Isolation Forest: насколько легко отделить момент от нормы случайными разбиениями (одно число, без датчиков)."""
    name = 'Isolation Forest'

    def fit(self, F, prep):
        from sklearn.ensemble import IsolationForest
        self.m = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(F)
        return self

    def errors(self, F, prep):
        return -self.m.score_samples(F)[:, None]


class AnalogDet:
    """Аналоги (k ближайших соседей): ожидание каждого датчика по 100 самым похожим моментам нормы с той же нагрузкой
    и тем же состоянием механизмов (булевы признаки); ошибка — отклонение от ожидания в разбросах аналогов."""
    name = 'Аналоги'

    def __init__(self, load_col, k=100, max_memory=20_000):
        self.load_col, self.k, self.max_memory = load_col, k, max_memory

    def _R(self, F, prep):
        cols = ([prep.num.index(self.load_col)] if self.load_col in prep.num else []) + list(range(len(prep.num), F.shape[1]))
        names = (['нагрузка'] if self.load_col in prep.num else []) + [f'b{j}' for j in range(F.shape[1] - len(prep.num))]
        return pd.DataFrame(F[:, cols], columns=names)

    def fit(self, F, prep):
        self.n = len(prep.num)
        X = pd.DataFrame(F[:, :self.n])
        self.am = AnalogModel(self.k, max_train=self.max_memory).fit(self._R(F, prep), X)
        return self

    signed = True

    def errors(self, F, prep):
        _, _, z = self.am.expected(self._R(F, prep), pd.DataFrame(F[:, :self.n]))
        return np.nan_to_num(z.to_numpy())                         # со знаком: выше (+) или ниже (−) ожидания


class AEDet:
    """Автоэнкодер: нейросеть сжимает момент (датчики + булевы состояния) и восстанавливает датчики; ошибка — по датчикам."""
    name = 'Автоэнкодер'

    def __init__(self, epochs=40, max_train=8000, seed=0):
        self.epochs, self.max_train, self.seed = epochs, max_train, seed

    def fit(self, F, prep):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        self.n = len(prep.num)
        rows = np.arange(len(F)) if len(F) <= self.max_train else np.sort(np.random.default_rng(self.seed).choice(len(F), self.max_train, replace=False))
        F = F[rows]
        d, p = F.shape[1], self.n
        inp = tf.keras.Input((d,))
        h = tf.keras.layers.Dense(min(256, 2 * p), activation='relu')(inp)
        code = tf.keras.layers.Dense(max(8, p // 8), activation='relu')(h)
        h = tf.keras.layers.Dense(min(256, 2 * p), activation='relu')(code)
        self.net = tf.keras.Model(inp, tf.keras.layers.Dense(p)(h))
        self.net.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        self.net.fit(F, F[:, :p], epochs=self.epochs, batch_size=128, verbose=0, validation_split=0.1,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        return self

    def errors(self, F, prep):
        return np.abs(self.net.predict(F, verbose=0, batch_size=4096) - F[:, :self.n])


class LSTMDet:
    """LSTM-автоэнкодер: то же для окна из L последовательных шагов — видит необычную динамику.
    На вход — окна (число окон × L × признаки); ошибка датчика — средняя по окну."""
    name = 'LSTM-автоэнкодер'

    def __init__(self, L=24, epochs=15, max_train=8000, seed=0):
        self.L, self.epochs, self.max_train, self.seed = L, epochs, max_train, seed

    def fit(self, W, prep):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        self.n = len(prep.num)
        if len(W) > self.max_train:
            W = W[np.sort(np.random.default_rng(self.seed).choice(len(W), self.max_train, replace=False))]
        inp = tf.keras.Input((self.L, W.shape[2]))
        h = tf.keras.layers.Dense(64, activation='relu')(inp)
        code = tf.keras.layers.LSTM(32)(h)
        h = tf.keras.layers.RepeatVector(self.L)(code)
        h = tf.keras.layers.LSTM(32, return_sequences=True)(h)
        self.net = tf.keras.Model(inp, tf.keras.layers.Dense(self.n)(h))
        self.net.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        self.net.fit(W, W[:, :, :self.n], epochs=self.epochs, batch_size=128, verbose=0, validation_split=0.1,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
        return self

    def errors(self, W, prep, mask=None):
        """mask (окна × L × датчики) — пропуски: там ошибка не считается (среднее по шагам, где значение было)."""
        out = []
        for a in range(0, len(W), 2048):
            w = W[a:a + 2048]
            e = np.abs(self.net.predict(w, verbose=0, batch_size=512) - w[:, :, :self.n])
            if mask is not None:
                m = ~mask[a:a + 2048]
                e = (e * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1)
            else:
                e = e.mean(axis=1)
            out.append(e)
        return np.vstack(out) if out else np.zeros((0, self.n))


class MahalDet:
    """Махаланобис: насколько сочетание значений необычно с учётом того, как датчики в норме меняются вместе
    (устойчивая оценка ковариации Ledoit–Wolf). Одно число на момент."""
    name = 'Махаланобис'

    def fit(self, F, prep):
        from sklearn.covariance import LedoitWolf
        rows = np.arange(len(F)) if len(F) <= 20_000 else np.linspace(0, len(F) - 1, 20_000).astype(int)
        lw = LedoitWolf().fit(F[rows])
        self.P, self.mu = lw.precision_, lw.location_
        return self

    def errors(self, F, prep):
        D = F - self.mu
        M = D @ self.P
        self.last_contrib = D * M / F.shape[1]       # вклад признака: d_j·(P·d)_j / p; сумма по признакам = ошибке
        return (np.einsum('ij,ij->i', M, D) / F.shape[1])[:, None]


class TwinsDet:
    """Близнецы: датчики, которые в норме почти повторяют друг друга (|ρ| ≥ corr). Для каждого — прямая
    «датчик ≈ a + b × медиана остальных близнецов»; ошибка — отход от неё в обычных разбросах. У датчиков без
    близнецов ошибка 0. Получает все датчики, включая почти-дубли (all_sensors): дубли — это и есть близнецы,
    а остальные методы видят набор без дублей."""
    name = 'Близнецы'
    all_sensors = True

    def __init__(self, corr=0.95):
        self.corr = corr

    def fit(self, F, prep):
        from scipy.cluster import hierarchy
        self.n = n = len(prep.num)
        rows = np.arange(len(F)) if len(F) <= 10_000 else np.linspace(0, len(F) - 1, 10_000).astype(int)
        Z = F[rows, :n]
        C = pd.DataFrame(Z).rank().corr().fillna(0).to_numpy()
        cl = hierarchy.fcluster(corr_linkage(C, 'complete'), 1 - self.corr, 'distance')
        self.models = []
        for g in np.unique(cl):
            idx = np.flatnonzero(cl == g)
            if len(idx) < 2:
                continue
            for j in idx:
                others = [o for o in idx if o != j]
                ref = np.median(Z[:, others], axis=1)
                if ref.std() == 0:
                    continue
                k_, b_ = np.polyfit(ref, Z[:, j], 1)
                r = Z[:, j] - (k_ * ref + b_)
                m_, s_ = np.median(r), 1.4826 * np.median(np.abs(r - np.median(r)))
                if s_ > 0:
                    self.models.append((j, others, k_, b_, m_, s_))
        return self

    def errors(self, F, prep):
        E = np.zeros((len(F), self.n))
        for j, others, k_, b_, m_, s_ in self.models:
            E[:, j] = np.abs((F[:, j] - (k_ * np.median(F[:, others], axis=1) + b_) - m_) / s_)
        return E


# ----------------------------------------------------------------------------- расширенные условия работы

COND_LOAD_CHANGE, COND_SINCE_START, COND_MILLS = 'условие: изменение нагрузки за час', 'условие: часов после пуска', 'условие: мельниц в работе'


def condition_spec(df, run_col, run_thr, per_h, extra_patterns, mill_pattern, normal, candidates):
    """Какие датчики — расширенные условия работы. Внешние датчики — по шаблонам имён (из подходящих — с меньшим
    числом пропусков); мельницы — по токам двигателей: «работает», если ток больше половины обычного рабочего."""
    extra = {}
    for nm, rx in (extra_patterns or {}).items():
        hits = [c for c in candidates if name_hit(rx, c) and c != run_col]
        if hits:
            extra[nm] = min(hits, key=lambda c: df.loc[normal, c].isna().mean())
    mills = {}
    for c in (candidates if mill_pattern else []):
        if name_hit(mill_pattern, c):
            v = df.loc[normal, c].astype(float).dropna()
            if len(v) and v.quantile(0.9) > 0:
                mills[c] = float(0.5 * v.quantile(0.9))
    return dict(run_col=run_col, run_thr=float(run_thr), per_h=int(per_h), extra=extra, mills=mills)


def condition_frame(df, spec):
    """Производные условия по таблице df (регулярная сетка): изменение нагрузки за час, часы после пуска (до 72),
    число работающих мельниц. Для новых данных нужна история хотя бы за 3 суток до оцениваемого момента."""
    x = df[spec['run_col']].astype(float)
    run = (x > spec['run_thr']).to_numpy()
    per_h = spec['per_h']
    out = pd.DataFrame(index=df.index)
    out[COND_LOAD_CHANGE] = (x - x.shift(per_h)).where(run).fillna(0.0).astype('float32')
    grp = np.cumsum(np.r_[True, run[1:] != run[:-1]]) if len(run) else np.zeros(0, int)
    since = pd.Series(1, index=df.index).groupby(grp).cumcount().to_numpy() / per_h
    out[COND_SINCE_START] = np.where(run, np.minimum(since, 72.0), 0.0).astype('float32')
    if spec['mills']:
        out[COND_MILLS] = np.sum([(df[c].astype(float) > thr).to_numpy() for c, thr in spec['mills'].items()], axis=0).astype('float32')
    return out


def condition_columns(spec):
    """Все условия: нагрузка, внешние датчики, производные."""
    return ([spec['run_col']] + [c for c in spec['extra'].values() if c != spec['run_col']]
            + [COND_LOAD_CHANGE, COND_SINCE_START] + ([COND_MILLS] if spec['mills'] else []))


def norm_name(c):
    """Имя столбца без лишних пробелов (в выгрузке бывают пробелы в конце и двойные)."""
    return ' '.join(str(c).split())


def poisson_ci(k, t, conf=0.95):
    """Точный 95% интервал для частоты: k событий за время t (в месяцах) -> (от, до) событий в месяц."""
    from scipy import stats
    a = (1 - conf) / 2
    lo = stats.chi2.ppf(a, 2 * k) / 2 / t if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - a, 2 * k + 2) / 2 / t
    return lo, hi


def window_rank(ratio, bg, scored, smax, L, step=24):
    """Ранг окна перед аварией: доля окон фона длиной L шагов, у которых максимум отношения ниже, чем у окна перед аварией
    (0.5 — как у случайного, 1 — выше всех окон фона)."""
    if not np.isfinite(smax) or L < 1:
        return np.nan
    r = np.where(bg, ratio, np.nan)
    vals = []
    for a in range(0, len(r) - L, step):
        seg = r[a:a + L]
        if np.isfinite(seg).sum() >= 0.5 * L and bg[a:a + L].all():
            vals.append(np.nanmax(seg))
    if not vals:
        return np.nan
    vals = np.array(vals)
    return float((vals < smax).mean() + 0.5 * (vals == smax).mean())


def lead_hours(in_alarm, t, stop):
    """Упреждение: сколько часов до останова горела непрерывная тревога (0 — в последние часы тревоги не было)."""
    pos = np.flatnonzero((t <= stop - pd.Timedelta('3h')) & (t >= stop - pd.Timedelta(days=60)))
    if not len(pos) or not in_alarm[pos[-1]]:
        return 0.0
    k = len(pos) - 1
    while k > 0 and in_alarm[pos[k - 1]]:
        k -= 1
    return float((stop - t[pos[k]]) / pd.Timedelta('1h'))


def poisson_binomial_sf(p, k):
    """P(число успехов ≥ k), если испытания независимы и у i-го вероятность успеха p[i] (точно, через свёртку).
    Нужна для «случайного уровня»: сколько аварий «поймал» бы метод, тревожащий наугад с той же частотой."""
    p = np.clip(np.nan_to_num(np.asarray(p, float)), 0, 1)
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return float(dist[int(k):].sum()) if k > 0 else 1.0


def new_verdict(row, windows=(1, 3, 7, 14, 30), main_d=7):
    """Вывод простыми словами по строке новых метрик (02 раздел 6.0, 00, 03 раздел 2.1): (коротко, подробно).
    Ловит ли вид аварии чаще случайного (обычное и осторожное p по окнам D) и много ли у него тревог."""
    def get(k, d=np.nan):
        v = row.get(k, d) if hasattr(row, 'get') else d
        return d if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
    ds = [d for d in windows if f'D={d}: p' in row]
    D0 = main_d if main_d in ds else (ds[-1] if ds else None)
    far, bgp = get('ложных в мес.'), get('в тревоге, % фона')
    sig = [d for d in ds if get(f'D={d}: p', 1.0) < 0.05]
    sig_c = [d for d in ds if get(f'D={d}: p осторожно', 1.0) < 0.05]
    lst = lambda xs: ', '.join(map(str, xs))
    if sig_c:
        det_s, det = 'чаще случайного', f'ловит аварии чаще случайного, и по осторожной оценке тоже (окна {lst(sig_c)} сут)'
    elif sig:
        det_s = 'чаще случайного только против спокойного фона'
        det = (f'чаще случайного только против спокойного фона (окна {lst(sig)} сут), по осторожной оценке — нет: тревоги, '
               'скорее, идут рядом с любыми остановами и пусками, а не именно перед авариями')
    else:
        det_s, det = 'не лучше случайного', 'аварии ловит не чаще, чем метод, тревожащий наугад с той же частотой, — ни в одном окне'
        kn = re.match(r'\s*(\d+)\s+из\s+(\d+)', str(get(f'D={D0}: с тревогой', ''))) if D0 is not None else None
        if kn and int(kn.group(1)) == int(kn.group(2)) > 0:          # поймал все, но аварий слишком мало для малого p
            det_s = 'аварий мало для вывода'
            det = (f'поймал все аварии ({kn.group(2)}), но их так мало, что и случайный метод поймал бы столько же с '
                   f'вероятностью {get(f"D={D0}: p"):.2f} — у одного котла это не доказательство; смотрите 03, раздел 2.1 '
                   '(аварии всех котлов вместе)')
    if not np.isfinite(far) or not np.isfinite(bgp):
        noise = 'частота тревог неизвестна'
    else:
        noise = 'тревог мало' if far <= 1 and bgp <= 5 else ('тревог умеренно' if far <= 3 and bgp <= 10 else 'тревог много')
    short_ = f'{det_s}; {noise}'
    parts = []
    if np.isfinite(far):
        ci = f' (95%: {get("95%: от"):.2f}–{get("до"):.2f})' if np.isfinite(get('95%: от')) else ''
        parts.append(f'{noise}: {far:.2f} ложных в месяц{ci}, тревога горит {bgp:.1f}% времени фона.')
    if D0 is not None:
        parts.append(f'За {D0} сут до останова тревога была перед {get(f"D={D0}: с тревогой", "—")} аварий, случайный метод поймал '
                     f'бы {get(f"D={D0}: случайно"):.1f} (p = {get(f"D={D0}: p"):.3f}, осторожно {get(f"D={D0}: p осторожно"):.3f}).')
    parts.append(f'Итог: {det}.')
    ctrl = get('контроль (к фону)')
    if np.isfinite(ctrl) and ctrl >= 2:
        parts.append(f'Перед исправными остановами (резерв, плановый ремонт) тревог в {ctrl:.1f} раза больше, чем на фоне, — '
                     'метод реагирует и на сам подход к останову.')
    return short_, ' '.join(parts)


def first_test_month(index, normal, min_steps=720):
    """С какого месяца начинать проверку «как в эксплуатации»: первое число первого месяца, до которого набралось не меньше
    min_steps шагов нормальной работы (по умолчанию 720 ч — месяц). Раньше модели месяца не на чем учиться. None — не набралось."""
    idx = pd.DatetimeIndex(index)
    cum = np.cumsum(np.asarray(normal, bool))
    for m in pd.date_range(idx[0].to_period('M').to_timestamp() + pd.offsets.MonthBegin(1), idx[-1], freq='MS'):
        k = idx.searchsorted(m)                                  # шагов до начала месяца
        if k and cum[k - 1] >= min_steps:
            return f'{m:%Y-%m-%d}'
    return None


def sensor_list(path, unit, columns, sensors_csv=None):
    """
    Name-based Excel selection for EDA/legacy callers. Explicit exclusions are
    honored; codes are ignored because the supplied workbook codes were mis-sorted.
    sensors_csv remains in the signature for compatibility and is not read.
    Return (matched columns, counts/missing names), including sheets without Kod.
    """
    from boiler_monitoring import canonical_name
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    n_unit = unit_number(unit)
    rows = [d for name, d in sheets.items() if unit_number(str(name).translate(LAT2CYR)) == n_unit and re.match(r'^\s*[КK]', str(name))]
    if not rows:
        rows = [d[d['агрегат'].astype(str).str.strip().map(lambda u: unit_number(u) == n_unit)] for d in sheets.values() if 'агрегат' in d]
    L = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['Kod', 'col', 'агрегат'])
    norm = lambda x: ' '.join(str(x).split())
    role = L.get('роль', pd.Series('', index=L.index)).fillna('').astype(str).str.strip().str.lower()
    selected = L.loc[~role.isin(['исключить', 'exclude'])]
    exact, canonical = {}, {}
    for col in columns:
        exact.setdefault(norm(col), []).append(col)
        canonical.setdefault(canonical_name(col, unit), []).append(col)
    matched, missing = set(), []
    for name in selected.get('col', pd.Series(dtype=str)).dropna():
        hits = exact.get(norm(name), []) or canonical.get(canonical_name(name, unit), [])
        if len(hits) > 1:
            raise ValueError(f'{unit}: неоднозначное имя {name}: {hits}')
        if hits:
            matched.add(hits[0])
        else:
            missing.append(norm(name))
    keep = [c for c in columns if c in matched]
    return keep, dict(listed=len(L), found=len(keep), missing=missing,
                     excluded_by_role=int(role.isin(['исключить', 'exclude']).sum()))


def unit_number(unit):
    """Номер агрегата: «К-3», «КА-3», «K3» -> 3 (None — нет числа)."""
    m = re.search(r'\d+', str(unit))
    return int(m.group(0)) if m else None


def common_name(col, unit):
    """Общее имя датчика для сравнения агрегатов: без префикса агрегата («КА-3 ») и с номером агрегата в обозначениях
    оборудования, заменённым на #: «КА-3 Т подшипника ММТ-3Г» -> «Т подшипника ММТ-#Г», «УП шибера 3Ш ВВ-1» -> «УП шибера #Ш ВВ-1».
    Номер заменяется только вплотную перед буквой: после дефиса («-3Г») и в первом коде оборудования («3Ш», «3РК»);
    «О2», «2 ст.», «№3», «гор.1», позиции «1Л / 2Л» не меняются."""
    n = unit_number(unit)
    s = norm_name(col).translate(LAT2CYR)                     # «KA-8 T металла» и «КА-8 Т металла» — одно имя
    if n is None:
        return s
    s = re.sub(rf'^КА?\s*-?\s*{n}\s+', '', s)
    s = re.sub(rf'-{n}(?=[А-ЯЁA-Z])', '-#', s)                              # ММТ-2Г, ДС-2А, ВГД-2В
    s = re.sub(rf'(?<![^\s]){n}(?=[А-ЯЁA-Z])', '#', s, count=1)            # первый код оборудования: 2Ш, 2РК, 2КВ (но не «2Л» позиции)
    return expand_abbrev(s)                                                 # «F пара» и «Расход пара», «T …» и «Температура …» — одно имя


def match_columns(cols, patterns):
    """Все столбцы, имя которых (без лишних пробелов) подходит под какой-нибудь шаблон; порядок — как в cols."""
    rx = [re.compile(p, re.I) for p in (patterns.values() if isinstance(patterns, dict) else patterns)]
    return [c for c in cols if any(name_hit(r, c) for r in rx)]


def name_hit(rx, c):
    """Подходит ли имя под шаблон — как есть или с латинскими буквами, заменёнными на похожие русские («KA-8 …»)."""
    r = rx if isinstance(rx, re.Pattern) else re.compile(rx, re.I)
    s = norm_name(c)
    for v in dict.fromkeys([s, s.translate(LAT2CYR), expand_abbrev(s), expand_abbrev(s.translate(LAT2CYR))]):
        if r.search(v):
            return True
    return False


def wide_condition_spec(spec, df, patterns, normal, candidates, max_missing=0.5):
    """Расширенный набор условий для регрессии: условия spec плюс все датчики по шаблонам patterns (уставки, обороты
    питателей, калориферы). Берутся только меняющиеся в норме, с пропусками не больше max_missing.
    Возвращает (новый spec, {датчик: почему не взят})."""
    extra, skipped = dict(spec['extra']), {}
    for c in match_columns(candidates, patterns):
        if c == spec['run_col'] or c in extra.values():
            continue
        v = df.loc[normal, c].astype(float)
        if v.isna().mean() > max_missing:
            skipped[c] = f'пропусков в норме {v.isna().mean():.0%}'
        elif v.nunique() < 2:
            skipped[c] = 'в норме не меняется'
        else:
            extra[norm_name(c)] = c
    return dict(spec, extra=extra), skipped


def _cond_split(cond, prep):
    """Номера столбцов-условий и проверяемых датчиков среди prep.num."""
    ci = [prep.num.index(c) for c in cond if c in prep.num]
    s = set(ci)
    return ci, [j for j in range(len(prep.num)) if j not in s]


class AnalogCondDet(AnalogDet):
    """analogs: как аналоги, но похожие моменты ищутся по расширенным условиям работы (нагрузка, её изменение за час,
    часы после пуска, число работающих мельниц, внешние датчики) и состояниям механизмов. Датчики-условия сами не
    проверяются (ошибка 0)."""
    name = 'analogs'
    with_conditions = True

    def __init__(self, cond, k=100, max_memory=20_000):
        self.cond, self.k, self.max_memory = list(cond), k, max_memory

    def _R(self, F, prep):
        ci, _ = _cond_split(self.cond, prep)
        nb = F.shape[1] - len(prep.num)
        return pd.DataFrame(F[:, ci + list(range(len(prep.num), F.shape[1]))],
                            columns=[f'c{j}' for j in range(len(ci))] + [f'b{j}' for j in range(nb)])

    def fit(self, F, prep):
        self.n = len(prep.num)
        _, self.ti = _cond_split(self.cond, prep)
        self.am = AnalogModel(self.k, max_train=self.max_memory).fit(self._R(F, prep), pd.DataFrame(F[:, self.ti]))
        return self

    def errors(self, F, prep):
        _, _, z = self.am.expected(self._R(F, prep), pd.DataFrame(F[:, self.ti]))
        E = np.zeros((len(F), self.n))
        E[:, self.ti] = np.nan_to_num(z.to_numpy())               # со знаком, как у аналогов
        return E


class CondNetDet:
    """Условная нейросеть: по условиям работы и состояниям механизмов предсказывает все датчики; ошибка датчика —
    |факт − прогноз|. Сами датчики на вход не подаются, поэтому неисправность не может «подтянуть» прогноз к себе
    (как в автоэнкодере, который восстанавливает датчик по соседним)."""
    name = 'Условная нейросеть'
    with_conditions = True

    def __init__(self, cond, epochs=40, max_train=20_000, seed=0):
        self.cond, self.epochs, self.max_train, self.seed = list(cond), epochs, max_train, seed

    def _in(self, F, prep):
        ci, _ = _cond_split(self.cond, prep)
        return F[:, ci + list(range(len(prep.num), F.shape[1]))]

    def fit(self, F, prep):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        self.n = len(prep.num)
        _, self.ti = _cond_split(self.cond, prep)
        if len(F) > self.max_train:
            F = F[np.sort(np.random.default_rng(self.seed).choice(len(F), self.max_train, replace=False))]
        Xin, Y = self._in(F, prep), F[:, self.ti]
        inp = tf.keras.Input((Xin.shape[1],))
        h = tf.keras.layers.Dense(128, activation='relu')(inp)
        h = tf.keras.layers.Dense(128, activation='relu')(h)
        self.net = tf.keras.Model(inp, tf.keras.layers.Dense(len(self.ti))(h))
        self.net.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        self.net.fit(Xin, Y, epochs=self.epochs, batch_size=256, verbose=0, validation_split=0.1,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        return self

    def errors(self, F, prep):
        E = np.zeros((len(F), self.n))
        E[:, self.ti] = np.abs(self.net.predict(self._in(F, prep), verbose=0, batch_size=4096) - F[:, self.ti])
        return E


def _fit_boost(X, y, specs, max_iter, depth, lr, seed):
    """Градиентный бустинг для одного датчика: по модели на каждую пару (потери, квантиль) из specs.
    Постоянный датчик — None. Один поток: датчики обучаются параллельно (_boost_many)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    if np.ptp(y) == 0:
        return None
    out = []
    with threadpool_limits(1):
        for loss, a in specs:
            kw = dict(quantile=a) if loss == 'quantile' else {}
            out.append(HistGradientBoostingRegressor(loss=loss, max_iter=max_iter, max_depth=depth, learning_rate=lr,
                                                     early_stopping=False, random_state=seed, **kw).fit(X, y))
    return out


def _boost_many(X, Y, specs, max_iter=60, depth=3, lr=0.1, seed=0, n_jobs=-1):
    """По бустингу на каждый столбец Y — параллельно по датчикам (все ядра)."""
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(_fit_boost)(X, Y[:, j], specs, max_iter, depth, lr, seed)
                                                    for j in range(Y.shape[1]))      # max_nbytes=None — без временных файлов


class BoostDet:
    """Бустинг «условия → датчики»: для каждого датчика градиентный бустинг (каждое следующее дерево исправляет ошибки
    предыдущих) по условиям работы и состояниям механизмов предсказывает медиану и квартили 25% и 75%. Отклонение —
    (факт − медиана) / разброс, как у аналогов: разброс свой в каждом режиме. Датчики-условия не проверяются."""
    name = 'Бустинг'
    with_conditions = True
    signed = True

    def __init__(self, cond, max_iter=60, depth=3, lr=0.1, max_train=6000, floor_frac=0.1, range_frac=0.02, seed=0):
        self.cond, self.max_iter, self.depth, self.lr = list(cond), max_iter, depth, lr
        self.max_train, self.floor_frac, self.range_frac, self.seed = max_train, floor_frac, range_frac, seed

    def _in(self, F, prep):
        ci, _ = _cond_split(self.cond, prep)
        return F[:, ci + list(range(len(prep.num), F.shape[1]))]

    def fit(self, F, prep):
        self.n = len(prep.num)
        _, self.ti = _cond_split(self.cond, prep)
        if len(F) > self.max_train:
            F = F[np.sort(np.random.default_rng(self.seed).choice(len(F), self.max_train, replace=False))]
        Y = F[:, self.ti].astype('float64')
        q = np.quantile(Y, [0.005, 0.25, 0.75, 0.995], axis=0)
        self.floor = np.maximum(np.maximum((q[2] - q[1]) / 1.349 * self.floor_frac, (q[3] - q[0]) * self.range_frac), 1e-6)
        self.models = _boost_many(self._in(F, prep), Y, [('quantile', 0.25), ('quantile', 0.5), ('quantile', 0.75)],
                                  self.max_iter, self.depth, self.lr, self.seed)
        return self

    def expected(self, F, prep):
        """Медиана и разброс проверяемых датчиков (в нормированных единицах)."""
        Xin = self._in(F, prep)
        med = np.full((len(F), len(self.ti)), np.nan)
        sc = np.full((len(F), len(self.ti)), np.nan)
        for k, ms in enumerate(self.models):
            if ms is not None:
                p25, p50, p75 = (m.predict(Xin) for m in ms)
                med[:, k], sc[:, k] = p50, np.maximum(np.abs(p75 - p25) / 1.349, self.floor[k])
        return med, sc

    def errors(self, F, prep):
        med, sc = self.expected(F, prep)
        E = np.zeros((len(F), self.n))
        E[:, self.ti] = np.nan_to_num((F[:, self.ti] - med) / sc)
        return E


class AnalogBoostDet(AnalogCondDet):
    """analogs с бустинг-поправкой: бустинг учится предсказывать, где analogs систематически ошибаются — их
    отклонение на незнакомых данных (3 части года) по условиям работы, состояниям механизмов и расстоянию до аналогов, —
    и вычитает это из отклонения. Остальное — как у analogs."""
    name = 'analogs с поправкой'

    def __init__(self, cond, k=100, max_memory=20_000, max_iter=60, depth=3, lr=0.1, max_train=6000, seed=0):
        super().__init__(cond, k, max_memory)
        self.max_iter, self.depth, self.lr, self.max_train, self.seed = max_iter, depth, lr, max_train, seed

    def fit(self, F, prep):
        self.n = len(prep.num)
        _, self.ti = _cond_split(self.cond, prep)
        R, X = self._R(F, prep), pd.DataFrame(F[:, self.ti])
        _, _, z, dist = analog_crossfit(R, X, time_blocks(R.index, 3), self.k, max_train=self.max_memory)
        self.dist_fill = float(np.nanmedian(dist)) if dist.notna().any() else 0.0
        feats = np.column_stack([R.to_numpy(), dist.fillna(self.dist_fill).to_numpy()])
        Z = np.nan_to_num(z.to_numpy())
        rows = np.arange(len(F)) if len(F) <= self.max_train else np.sort(np.random.default_rng(self.seed).choice(len(F), self.max_train, replace=False))
        self.corr = _boost_many(feats[rows], Z[rows], [('absolute_error', None)], self.max_iter, self.depth, self.lr, self.seed)
        self.am = AnalogModel(self.k, max_train=self.max_memory).fit(R, X)
        return self

    def errors(self, F, prep):
        R = self._R(F, prep)
        _, _, z = self.am.expected(R, pd.DataFrame(F[:, self.ti]))
        Z = np.nan_to_num(z.to_numpy())
        feats = np.column_stack([R.to_numpy(), self.am.dist.fillna(self.dist_fill).to_numpy()])
        for k, ms in enumerate(self.corr):
            if ms is not None:
                Z[:, k] -= ms[0].predict(feats)
        E = np.zeros((len(F), self.n))
        E[:, self.ti] = Z
        return E


def on_residuals(det):
    """Пометить метод как «вторую ступень»: он получает не датчики, а остатки «analogs» (факт − ожидание при таких
    же условиях, в разбросах аналогов)."""
    det.on_residuals = True
    return det


class RareStates:
    """Редкие состояния механизмов: сумма −log10(доля в норме) по состояниям, которые в норме бывают реже 1% времени
    (тайм-аут, самоход, срабатывание защиты); состояние, державшееся дольше held_steps шагов, считается устоявшимся."""
    name = 'Редкие состояния механизмов'

    def __init__(self, cols, held_steps):
        self.cols, self.held_steps = list(cols), int(held_steps)

    def fit(self, T):
        self.freq = {c: T[c].astype(object).value_counts(normalize=True) for c in self.cols}
        self.n = len(T)
        return self

    def score(self, df):
        s = np.zeros(len(df))
        for c in self.cols:
            v = df[c].astype(object)
            held = v.groupby((v != v.shift()).cumsum()).cumcount().to_numpy()
            p = v.map(self.freq[c]).astype(float).where(v.notna(), 1.0).fillna(0.5 / (self.n + 1)).to_numpy()
            s += np.where((p < 0.01) & (held < self.held_steps), -np.log10(p), 0.0)
        return s


def combine_ratios(R, how):
    """Объединение методов по отношениям «ошибка / порог» (строки — моменты, столбцы — методы):
    how='max' — хотя бы один метод (наибольшее отношение); 'second' — два метода согласны (второе по величине).
    Возвращает объединённое отношение (NaN, если ни один метод не оценивал) и номер «ведущего» метода."""
    R = np.asarray(R, float)
    ok = np.isfinite(R).any(axis=1)
    Z = np.where(np.isfinite(R), R, -np.inf)
    comb = Z.max(axis=1) if how == 'max' else (np.sort(Z, axis=1)[:, -2] if Z.shape[1] > 1 else Z[:, 0])
    comb = np.where(ok, np.where(np.isfinite(comb), comb, 0.0), np.nan)
    return comb, Z.argmax(axis=1)


def slow(E, index, hours):
    """«Медленный сдвиг»: ошибка каждого датчика усредняется (со знаком) за последние hours часов."""
    if not hours:
        return E
    return pd.DataFrame(E, index=index).rolling(f'{hours}h', min_periods=1).mean().to_numpy()


def calibrate(E_parts, top, slow_h=0, alarm_q=0.995, floor_rel=0.2):
    """Обычная ошибка датчиков и порог по ошибкам на проверочных данных (список (ошибки, индекс)).
    Обычная ошибка = 99-й процентиль |ошибки| датчика, не меньше floor_rel × медианы по датчикам (иначе датчик,
    почти не менявшийся на проверке, давал бы огромные отношения). Порог = квантиль alarm_q ошибки момента."""
    A = np.abs(np.vstack([slow(E, ix, slow_h) for E, ix in E_parts]))
    q = np.nanpercentile(A, 99, axis=0)
    q = np.maximum(np.nan_to_num(q), floor_rel * np.nanmedian(q) if np.isfinite(np.nanmedian(q)) else 0)
    q = np.where(q > 1e-9, q, 1.0)
    thr = float(np.nanquantile(top_mean(A / q, top), alarm_q))
    return q, thr


def windows(F, ends, L):
    """Окна F[i−L+1 … i] для строк ends (номера строк в F). Возвращает массив (len(ends), L, признаки)."""
    from numpy.lib.stride_tricks import sliding_window_view
    V = sliding_window_view(F, L, axis=0)                  # (n−L+1, признаки, L)
    return np.ascontiguousarray(V[np.asarray(ends) - L + 1].transpose(0, 2, 1))


def top_mean(R, k):
    """Средняя из k наибольших нормированных ошибок датчиков в каждой строке («несколько самых выбившихся датчиков»)."""
    R = np.nan_to_num(np.asarray(R, float))
    k = min(k, R.shape[1])
    return np.sort(R, axis=1)[:, -k:].mean(axis=1)


def rolling_mean(s, n_steps, gap='3h'):
    """Скользящее среднее по n_steps шагам внутри непрерывных участков (после разрыва > gap — заново)."""
    s = s.dropna()
    seg = (s.index.to_series().diff() > pd.Timedelta(gap)).cumsum()
    return s.groupby(seg).transform(lambda v: v.rolling(n_steps, min_periods=1).mean())


# ----------------------------------------------------------------------------- вклад датчиков перед авариями (графики)

FC_VAL, FC_BG = 'в неделю перед аварией (×)', 'обычно вдали от аварий (×)'


def failure_contrib_bars(F, title=None, top=8, ncol=3):
    """Столбики: какие датчики сильнее всего отклонялись в неделю перед каждой аварией (таблица failure_contrib_*.csv
    одного или нескольких агрегатов). Длина — во сколько раз отклонение выше обычной ошибки датчика (в самые необычные
    10% часов недели); красный — вне обычного (> 1), серый — в пределах; чёрная черта — как датчик обычно отклоняется
    вдали от аварий (если столбик намного длиннее черты — датчик перед аварией вёл себя необычно)."""
    keys = list(dict.fromkeys(zip(F['агрегат'], F['авария (останов)'])))
    if not keys:
        return
    ncol = min(ncol, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, (0.33 * top + 1.1) * nrow + 0.4), squeeze=False)
    for ax, (u, a) in zip(axes.ravel(), keys):
        g = F[(F['агрегат'] == u) & (F['авария (останов)'] == a)].sort_values(FC_VAL, ascending=False).head(top)[::-1]
        v = g[FC_VAL].to_numpy(float)
        yy = np.arange(len(g))
        ax.barh(yy, v, color=np.where(v > 1, '#c0392b', '#95a5a6'))
        ax.scatter(g[FC_BG].to_numpy(float), yy, marker='|', s=180, color='black', zorder=3)
        ax.axvline(1, color='gray', ls='--', lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([short(c, 42) for c in g['датчик']], fontsize=8)
        ax.set_title(f'{u} · авария {a}', fontsize=10)
        ax.set_xlabel('во сколько раз выше обычного', fontsize=8)
        ax.grid(axis='y', visible=False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    fig.suptitle((title + '\n' if title else '') + 'красный — вне обычного, серый — в пределах; черта — как датчик обычно отклоняется вдали от аварий',
                 fontsize=11, fontweight='bold')
    plt.show()
