"""Исходные алгоритмы legacy 1.4: классы, OOF, калибровка и экспорт.

Секции исполняет legacy_fleet в отдельном пространстве каждого эксперимента.
Это служебный исходный код, не самостоятельный сценарий запуска.
"""

# %% setup
import os, re, json, time, warnings, importlib
warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import joblib
from scipy.cluster import hierarchy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from IPython.display import display

import k2_lib as K
K = importlib.reload(K)                     # всегда свежая версия k2_lib.py, даже без перезапуска ядра
if not hasattr(K, 'calibrate'):
    raise ImportError(f'Рядом с ноутбуком старая k2_lib.py ({K.__file__}) — замените её новой версией.')
from k2_lib import say, md, short, fmt_list, pct, STATE_COL, ST_WORK, ST_FAIL, ST_PLAN, ST_NODATA, EV_FAIL, EV_PLAN, EV_RES, M_STOP, M_START, M_RUN
K.setup_plots()
HAS_TF = False
if USE_NEURAL:
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        HAS_TF = True
    except ImportError:
        print('TensorFlow недоступен в выбранном окружении — нейросети пропускаются.')
BG_GAP_D = EXCL_FAIL_D = PRE_WINDOW_D + NEUTRAL_D              # фон и норма для обучения — дальше окна и нейтральной недели
MERGE_GAP_H, N_FOLDS = globals().get('MERGE_GAP_H', 12), 3
import legacy_group as LG
LG = importlib.reload(LG)


# %% training
months = pd.date_range(pd.Timestamp(FIRST_TEST).to_period('M').to_timestamp(), df.index[-1], freq='MS')
t_ix = df.index
OUT = os.path.join(RES_DIR, KIND_TAG); os.makedirs(OUT, exist_ok=True)                  # кэш результатов методов агрегата
DATA_FILE = HOURLY_PATH if DATA_KIND == 'hourly' else INSTANT_PATH
TRAIN_DESC = {'hours': f'последние {TRAIN_HOURS} ч нормальной работы (не дальше {TRAIN_MAX_DAYS} сут)',
              'days': f'нормальная работа за последние {TRAIN_DAYS} сут', 'all': 'вся нормальная работа до начала месяца'}[TRAIN_MODE]
SIG = dict(unit=UNIT, kind=KIND_TAG, first=FIRST_TEST, train=TRAIN_DESC, q=ALARM_Q, smooth=SMOOTH_H, sens=len(sens), v=13, states=STATE_COLS, cover=MIN_TRAIN_COVER,
           pre=PRE_WINDOW_D, excl=EXCL_FAIL_D,                      # меняется окно — результаты пересчитываются
           data=int(os.path.getmtime(DATA_FILE)) if os.path.exists(DATA_FILE) else 0)      # данные пересобраны — пересчёт
MAIN_PREFIX = ('analogs:', 'Регрессия:', 'Остатки ', 'Близнецы', 'analogs (мин. набор):', 'Регрессия (мин. набор):',
               'analogs (общий набор):', 'Регрессия (общий набор):')            # виды основных методов (раздел 4)
if USE_SENSOR_LIST:                                   # другой список датчиков — результаты пересчитываются
    SIG = dict(SIG, sensor_list=sorted(num_all + txt_cols))

def wanted(name):
    """Считать ли метод: RUN_ONLY — только перечисленные; иначе все."""
    if RUN_ONLY is not None:
        return name in RUN_ONLY
    return True
RESID = {}                                       # остатки первой ступени для текущего месяца (4.3–4.5)
REG = {}                                         # посчитанные методы: имя -> (конструктор, виды) — для экспорта (раздел 9)

def masks(m0):
    """Часы обучения модели на момент m0: нормальная работа до m0 без EXCL_FAIL_D суток перед уже случившимися авариями,
    в окне TRAIN_MODE ('hours' — последние TRAIN_HOURS таких часов, не дальше TRAIN_MAX_DAYS; 'days' — за TRAIN_DAYS суток; 'all')."""
    known = fails.start[fails.start < m0]
    pre = (K.days_to_next(t_ix, known) <= EXCL_FAIL_D).to_numpy() if len(known) else np.zeros(len(df), bool)
    ok = normal.to_numpy() & enough.to_numpy() & ~pre & (t_ix < m0)
    if TRAIN_MODE == 'days':
        return ok & (t_ix >= m0 - pd.Timedelta(days=TRAIN_DAYS))
    if TRAIN_MODE == 'hours':
        ok &= t_ix >= m0 - pd.Timedelta(days=TRAIN_MAX_DAYS)
        pos = np.flatnonzero(ok)
        if len(pos) > hs(TRAIN_HOURS):
            ok[pos[:-hs(TRAIN_HOURS)]] = False                   # оставить последние TRAIN_HOURS часов нормальной работы
    return ok

def data_of(det):
    """Данные метода: (числа, пропуски, тексты). Близнецам — все датчики с почти-дублями (дубли и есть близнецы);
    регрессии — датчики и расширенные условия; остальным методам с условиями — датчики и условия analogs; методам на
    остатках — остатки первой ступени (без состояний механизмов: они уже учтены в первой ступени); остальным — датчики без дублей."""
    if getattr(det, 'on_residuals', False):
        return RESID['X'], RESID['M'], None
    sub = getattr(det, 'subset', None)
    if sub is not None and (getattr(det, 'reg', False) or getattr(det, 'with_conditions', False)):
        key = (getattr(det, 'reg', False), tuple(sub))           # модель на части проверяемых датчиков (раздел 8.1)
        if key not in SUBSET_DATA:
            base = X_reg if key[0] else X_cond
            Xs = base[[c for c in base.columns if c in set(sub) or c not in set(TARGETS)]]
            SUBSET_DATA[key] = (Xs, Xs.isna().to_numpy())
        return SUBSET_DATA[key] + (T,)
    if getattr(det, 'reg', False):
        return X_reg, MISS_reg, T
    if getattr(det, 'with_conditions', False):
        return X_cond, MISS_cond, T
    if getattr(det, 'all_sensors', False):
        return X_all, MISS_all, T
    return X, MISS, T

SUBSET_DATA = {}

def with_subset(det, sub):
    """Тот же метод, но проверяет только датчики sub (условия и состояния механизмов — все)."""
    det.subset = list(sub)
    return det

def common_for(det):
    """Общие параметры папки модели; у модели на части датчиков «есть ли данные» проверяется только по её датчикам."""
    sub = getattr(det, 'subset', None)
    if sub is None:
        return COMMON
    keep = set(sub) | set(COND_RAW)
    return dict(COMMON, presence_cols=[c for c in sens if c in keep] or list(sub))

def sig_of(det):
    """Отпечаток для кэша: у методов с другими данными — свой (меняется набор данных — пересчитывается только он)."""
    if getattr(det, 'subset', None) is not None:
        return dict(SIG, cond=COND_COLS_REG if getattr(det, 'reg', False) else COND_COLS, subset=list(det.subset))
    if getattr(det, 'reg', False):
        return dict(SIG, cond=COND_COLS_REG)
    if getattr(det, 'on_residuals', False) or getattr(det, 'with_conditions', False):
        return dict(SIG, cond=COND_COLS)
    if getattr(det, 'all_sensors', False):
        return dict(SIG, all_sensors=X_all.shape[1])
    return SIG

def regression():
    """Регрессия «условия → датчики» (бустинг) на расширенных условиях: основной метод 4.2 и первая ступень 4.5."""
    det = K.BoostDet(COND_COLS_REG)
    det.reg = True
    return det

def blank_cols(Xs, Ms, blank):
    """Датчики, которых почти не было в обучающем году, — как пропуск: модель их не проверяет (ошибка 0)."""
    idx = [Xs.columns.get_loc(c) for c in blank if c in Xs.columns]
    if not idx:
        return Xs, Ms
    Xs, Ms = Xs.copy(), Ms.copy()
    Xs.iloc[:, idx] = np.nan
    Ms[:, idx] = True
    return Xs, Ms

def inputs(det, prep, rows, train=False, max_rows=None):
    """Данные для модели по маске строк: признаки или (для LSTM) окна из L шагов; индекс строк; маска пропусков."""
    Xd, Md, Td = data_of(det)
    blank = getattr(det, 'blank', None)
    L = getattr(det, 'L', 0)
    if not L:
        pos = np.flatnonzero(rows)
        if max_rows and len(pos) > max_rows:
            pos = pos[np.linspace(0, len(pos) - 1, max_rows).astype(int)]
        ix = t_ix[pos]
        Xs, Ms = (Xd.loc[ix], Md[pos]) if not blank else blank_cols(Xd.loc[ix], Md[pos], blank)
        return prep.transform(Xs, Td.loc[ix] if Td is not None else None), ix, Ms
    pr = np.flatnonzero(rows)
    p0, p1 = max(0, pr.min() - L + 1), pr.max() + 1
    rng = t_ix[p0:p1]
    Xr, Mr = (Xd.loc[rng], Md[p0:p1]) if not blank else blank_cols(Xd.loc[rng], Md[p0:p1], blank)
    F = prep.transform(Xr, Td.loc[rng] if Td is not None else None)
    ok = (rows if train else scorable.to_numpy())[p0:p1].astype(int)
    ends = np.flatnonzero(pd.Series(ok).rolling(L, min_periods=L).sum().eq(L).to_numpy() & rows[p0:p1])
    lim = det.max_train if train else max_rows
    if lim and len(ends) > lim:
        ends = ends[np.linspace(0, len(ends) - 1, lim).astype(int)] if not train else np.sort(np.random.default_rng(0).choice(ends, lim, replace=False))
    return K.windows(F, ends, L), rng[ends], K.windows(Mr, ends, L)

def model_errors(det, prep, rows, max_rows=None):
    """Ошибки датчиков по маске строк (пропуски — без ошибки)."""
    F, ix, Mi = inputs(det, prep, rows, max_rows=max_rows)
    if not len(ix):
        return None, ix
    if getattr(det, 'L', 0):
        return det.errors(F, prep, mask=Mi), ix
    E = det.errors(F, prep)
    return (np.where(Mi, 0.0, E) if E.shape[1] > 1 else E), ix

def train_one(make, m0, views, oof_rows='часть'):
    """Модель на момент m0: проверка на незнакомых частях года -> обычные ошибки и пороги; затем модель на всём годе.
    oof_rows=None — ошибки на незнакомых частях по всем строкам года (нужно первой ступени методов на остатках)."""
    tr = masks(m0)
    if tr.sum() < hs(500):
        return None
    cover = data_of(make())[0].loc[tr].notna().mean()      # новые датчики (мало значений в обучающем году) — не проверяются
    blank = [c for c in cover.index[cover < MIN_TRAIN_COVER]]
    def make_b():
        det = make()
        det.blank = blank
        return det
    pos = np.flatnonzero(tr)
    blocks = K.time_blocks(t_ix[pos], N_FOLDS)
    parts, cparts = [], []
    for b in range(N_FOLDS):
        fit_m, ev_m = tr.copy(), np.zeros(len(t_ix), bool)
        fit_m[pos[blocks == b]] = False
        ev_m[pos[blocks == b]] = True
        det = make_b()
        Xd, _, Td = data_of(det)
        prep = K.Prep(cols=STATE_COLS).fit(Xd.loc[fit_m], Td.loc[fit_m] if Td is not None else None)
        det.fit(inputs(det, prep, fit_m, train=True)[0], prep)
        E, ix = model_errors(det, prep, ev_m, max_rows=hs(3000) if oof_rows == 'часть' else oof_rows)
        if E is not None:
            parts.append((E, ix))
            C = getattr(det, 'last_contrib', None)          # вклады признаков (Махаланобис) на незнакомых данных
            if C is not None and len(C) == len(ix):          # по названиям: булевы признаки в частях года разные
                cparts.append(pd.DataFrame(np.abs(C), columns=list(prep.names)[:C.shape[1]]))
    cal = {v['name']: K.calibrate(parts, v['top'], v.get('slow', 0), ALARM_Q) for v in views}
    det = make_b()
    Xd, _, Td = data_of(det)
    prep = K.Prep(cols=STATE_COLS).fit(Xd.loc[tr], Td.loc[tr] if Td is not None else None)
    det.fit(inputs(det, prep, tr, train=True)[0], prep)
    qc = None
    if cparts:                                           # обычный вклад признака итоговой модели: 99-й процентиль |вклада|
        A = pd.concat(cparts, ignore_index=True).reindex(columns=list(prep.names)).to_numpy(float)
        qc = np.nan_to_num(np.nanpercentile(A, 99, axis=0)) if np.isfinite(A).any() else np.ones(A.shape[1])
        qc = np.where(np.maximum(qc, 0.2 * np.median(qc)) > 1e-12, np.maximum(qc, 0.2 * np.median(qc)), 1.0)
    return dict(prep=prep, det=det, cal=cal, parts=parts, qc=qc, blank=blank)

def cached(name, compute, sig=None):
    sig = sig or SIG
    f = os.path.join(OUT, name.replace(' ', '_') + '.joblib')
    if not RECOMPUTE and os.path.exists(f):
        r = joblib.load(f)
        if r.get('sig') == sig:
            print(f'{name}: результаты загружены из {f}')
            return r['views']
    t0 = time.time()
    views = compute()
    joblib.dump(dict(views=views, sig=sig), f, compress=3)
    print(f'{name}: {(time.time() - t0) / 60:.1f} мин')
    return views

def new_out(views):
    return {v['name']: dict(score=pd.Series(np.nan, index=t_ix), thr=pd.Series(np.nan, index=t_ix), contrib=None,
                            model_cutoff=pd.Series(pd.NaT, index=t_ix, dtype='datetime64[ns]')) for v in views}

def fill(out, tm, E, ix, m0, in_m, views, C=None):
    """Ошибки датчиков за месяц -> ошибка момента по каждому виду, порог месяца, вклад датчиков
    (у Махаланобиса — вклад признака в расстояние / его обычный вклад)."""
    keep = ix >= m0
    for v in views:
        q, thr = tm['cal'][v['name']]
        R = np.abs(K.slow(E, ix, v.get('slow', 0)))[keep] / q
        o = out[v['name']]
        o['model_cutoff'].loc[ix[keep]] = pd.Timestamp(tm.get('trained_cutoff', m0))
        o['score'].loc[ix[keep]] = K.top_mean(R, v['top'])
        o['thr'].loc[in_m] = thr
        if R.shape[1] > 1:
            if o['contrib'] is None:
                o['contrib'] = np.full((len(t_ix), R.shape[1]), np.nan, dtype='float16')
                o['cols'] = list(tm['prep'].num)          # какие датчики в столбцах вклада
            o['contrib'][t_ix.get_indexer(ix[keep])] = R.astype('float16')
        elif C is not None and tm.get('qc') is not None:
            n = len(tm['prep'].num)                      # в разбивку — только датчики (их набор одинаков во всех месяцах)
            Rc = (np.abs(C[keep]) / tm['qc'])[:, :n]
            if o['contrib'] is None:
                o['contrib'] = np.full((len(t_ix), n), np.nan, dtype='float16')
                o['cols'] = list(tm['prep'].num)
            o['contrib'][t_ix.get_indexer(ix[keep])] = Rc.astype('float16')

def run_online(name, make, views):
    """Ежемесячный прогон модели; views — варианты ошибки момента (сколько датчиков, медленный сдвиг)."""
    def compute():
        out = new_out(views)
        slow_max = max(v.get('slow', 0) for v in views)
        for i, m0 in enumerate(months):
            t0 = time.time()
            m1 = m0 + pd.offsets.MonthBegin(1)
            in_m = (t_ix >= m0) & (t_ix < m1)
            te = scorable.to_numpy() & enough.to_numpy() & in_m
            if not te.any():
                continue
            tm = train_one(make, m0, views)
            if tm is None:
                continue
            rows = te | (scorable.to_numpy() & enough.to_numpy() & (t_ix >= m0 - pd.Timedelta(hours=slow_max)) & (t_ix < m0)) if slow_max else te
            E, ix = model_errors(tm['det'], tm['prep'], rows)
            if E is None:
                continue
            fill(out, tm, E, ix, m0, in_m, views, getattr(tm['det'], 'last_contrib', None))
            if i == 0 or (i == 1 and out[views[0]['name']]['score'].notna().sum() == 0):
                print(f'  {name}: месяц — {time.time() - t0:.0f} с, всего месяцев {len(months)}', flush=True)
        return out
    return cached(name, compute, sig_of(make()))

# --- оценка

RES, MET = {}, {}
import k2_deploy as D
D = importlib.reload(D)
import sklearn
DEPLOY = os.path.join(MODELS_DIR, UNIT if KIND_TAG == 'hourly' else f'{UNIT}_{KIND_TAG}')
M_NEXT = df.index[-1] + STEP
COMMON = dict(run_col=run_col, run_thr=float(run_thr), startup_steps=hs(startup_h), prestop_steps=hs(3), min_stop_steps=hs(2),
              ffill_steps=hs(3), smooth_steps=hs(SMOOTH_H), persist_steps=hs(PERSIST_H), merge_steps=hs(MERGE_GAP_H), step=str(STEP),
              min_present=0.7, presence_cols=list(sens), cond_spec=COND_SPEC, data_kind=KIND_TAG, trained_until=str(df.index[-1]),
              train_days=TRAIN_DAYS, train_window=TRAIN_DESC, alarm_q=ALARM_Q, pre_window_d=PRE_WINDOW_D, created=pd.Timestamp.now().isoformat(timespec='seconds'),
              versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, tensorflow=tf.__version__ if HAS_TF else None))
# метод -> (папка, ядро, вход)
EXPORT_AS = {'analogs': ('analogs', 'analogs', 'conditions'), 'Регрессия': ('regression', 'boost', 'conditions'),
             'analogs (мин. набор)': ('analogs_min', 'analogs', 'conditions'), 'Регрессия (мин. набор)': ('regression_min', 'boost', 'conditions'),
             'analogs (общий набор)': ('analogs_common', 'analogs', 'conditions'), 'Регрессия (общий набор)': ('regression_common', 'boost', 'conditions'),
             'Близнецы': ('twins', 'twins', 'sensors'), 'AAKR': ('aakr', 'kernel', 'conditions'), 'MSET': ('mset', 'kernel', 'conditions'),
             'AAKR (все датчики)': ('aakr_all', 'kernel', 'conditions'), 'MSET (все датчики)': ('mset_all', 'kernel', 'conditions')}
EXPORTED, VIEW_DIR, FINAL = {}, {}, {}                  # папка -> метод; вид -> папка; папка -> обученная модель (для проверки)
HISTORY_COUNTS = {}
STAGE1_TM, RESID_TM = {}, {}                             # по виду первой ступени: итоговая первая ступень; модели на остатках

def vcfg(tm, views):
    return [dict(name=v['name'], top=v['top'], slow_h=v.get('slow', 0), q=tm['cal'][v['name']][0], thr=tm['cal'][v['name']][1]) for v in views]

def quality(views):
    return {v['name']: {k: (float(x) if isinstance(x, (int, float, np.floating)) else x) for k, x in MET[v['name']].items()}
            for v in views if v['name'] in MET}

QUALITY_OF = lambda n: quality([dict(name=n)]).get(n)

def export_sig(**kw):
    return json.loads(json.dumps(dict(SIG, **kw), ensure_ascii=False, default=str))

def already(path, sig):
    """Папка уже экспортирована с теми же данными и настройками — заново не обучаем."""
    f = os.path.join(path, 'config.json')
    return not RECOMPUTE and os.path.exists(f) and json.load(open(f, encoding='utf-8')).get('sig') == sig

def register(path, name, views):
    EXPORTED[path] = name
    VIEW_DIR.update({v['name']: path for v in views})

def export_now(name, make, views):
    """Сразу после оценки метода: итоговая модель на последнем годе -> models/<агрегат>/<метод>/final/."""
    if not EXPORT or name not in EXPORT_AS:
        return
    folder, core, inp = EXPORT_AS[name]
    det0 = make()
    cond = getattr(det0, 'export_cond', None)               # классические AAKR / MSET — только производные условия
    if cond is None:
        cond = (COND_COLS_REG if getattr(det0, 'reg', False) else COND_COLS) if inp == 'conditions' else None
    sub = getattr(det0, 'subset', None)
    path, sig = os.path.join(DEPLOY, folder, 'final'), export_sig(method=name, cond=cond, **({'subset': sub} if sub else {}))
    if already(path, sig):
        register(path, name, views)
        if globals().get('VERBOSE', False):
            print(f'  {name}: папка {path} уже сохранена — не переобучаем')
        return
    t0 = time.time()
    tm = train_one(make, M_NEXT, views)
    if tm is None:
        return
    extra = dict(sig=sig, **({'L': tm['det'].L} if core == 'lstm' else {}))
    actual_common = dict(common_for(det0), trained_until=str(pd.Timestamp(tm.get('trained_cutoff', M_NEXT)) - STEP),
                         training_rows_by_unit=tm.get('training_rows_by_unit', {}))
    D.export_model(path, name, core, tm['det'], tm['prep'], vcfg(tm, views), actual_common, inp=inp,
                   cond=cond or (), qc=tm['qc'], quality=quality(views), extra=extra)
    register(path, name, views)
    FINAL[path] = (tm, views, None)
    if globals().get('VERBOSE', False):
        print(f'  {name}: итоговая модель сохранена в {path} ({time.time() - t0:.0f} с)')

def export_history(name):
    """Модели месяцев для дашборда: для каждого месяца проверки — та же модель, что дала график (обучена только на
    данных до начала месяца). Без них дашборд оценивал бы прошлое итоговой моделью, а она «помнит» свой обучающий год."""
    if not EXPORT or name.replace(' (мин. набор)', '').replace(' (общий набор)', '') not in HISTORY_FOR or name not in EXPORT_AS:
        return
    make, views = REG[name]
    folder, core, inp = EXPORT_AS[name]
    common = common_for(make())
    out_dir = os.path.join(DEPLOY, folder)
    sub = getattr(make(), 'subset', None)
    sig = export_sig(method=name, history=True, **({'subset': sub} if sub else {}))
    t0, n_new, n_ready = time.time(), 0, 0
    for m0 in months:
        path = os.path.join(out_dir, f'{m0:%Y-%m}')
        if already(path, sig):
            n_ready += 1
            continue
        tm = GROUP_ENGINE.model_for_export(make, m0, views)
        if tm is None:
            continue
        D.export_model(path, f'{name} ({m0:%Y-%m})', core, tm['det'], tm['prep'], vcfg(tm, views),
                       dict(common, trained_until=str(pd.Timestamp(tm.get('trained_cutoff', m0)) - STEP),
                            training_rows_by_unit=tm.get('training_rows_by_unit', {})), inp=inp, cond=list(getattr(tm['det'], 'cond', ())),
                       qc=tm['qc'], extra=dict(sig=sig))
        n_new += 1
        n_ready += 1
    HISTORY_COUNTS[name] = n_ready
    if globals().get('VERBOSE', False):
        print(f'  {name}: модели месяцев для дашборда — {out_dir}/<ГГГГ-ММ>/ (новых {n_new}, {time.time() - t0:.0f} с)')

def elapsed_text(started):
    seconds = time.perf_counter() - started
    return f'{seconds:.1f} с' if seconds < 60 else f'{seconds / 60:.1f} мин'


def run_and_show(name, make, views):
    if not wanted(name):
        print(f'{name}: пропущено (RUN_ONLY)')
        return
    started = time.perf_counter()
    before_ops, before_exports = len(GROUP_ENGINE.operation_audit), len(EXPORTED)
    print(f'{GROUP_NAME} · {UNIT} · {name}', flush=True)
    out = run_online(name, make, views)
    REG[name] = (make, views)
    for v in views:
        RES[v['name']] = out[v['name']]
        show(v['name'])
    export_now(name, make, views)
    export_history(name)
    ops = GROUP_ENGINE.operation_audit[before_ops:]
    fits = sum(int(r['fit_calls']) for r in ops)
    import legacy_metrics as _report_metrics
    hours = _report_metrics.alarm_series(out[views[0]['name']], globals())[3].sum() / PER_H
    print(f"  Готово за {elapsed_text(started)}; рассчитано {hours:.0f} ч; "
          f"обучений: {fits}; экспорт: финальных {len(EXPORTED) - before_exports}, месячных {HISTORY_COUNTS.get(name, 0)}; оценки: {globals().get('_LAST_SCORE_SOURCE', 'расчёт')}.", flush=True)
GROUP_ENGINE.install(globals())


# %% residual
import kernel_models as KM
KM = importlib.reload(KM)
from kernel_models import AAKR, MSET
CLASSIC = ' (все датчики)'
MAIN_PREFIX = tuple(globals().get('MAIN_PREFIX', ())) + ('AAKR:', 'MSET:', f'AAKR{CLASSIC}:', f'MSET{CLASSIC}:')
DERIVED_COND = [c for c in COND_COLS if c.startswith('условие: ')]

class KernelDet:
    """AAKR / MSET как метод ноутбука. Ошибка датчика — (факт − ожидание) / разброс, со знаком (как у analogs);
    дальше — обычная ошибка датчика, порог и тревога ноутбука. Датчики-условия не проверяются (ошибка 0).
    classic=True — классический вариант «… (все датчики)»: одно расстояние по всем датчикам (условия тоже — просто
    датчики, сам датчик входит в своё расстояние), без линейной поправки и устойчивого прохода; проверяются все датчики,
    в том числе условия. Производные условия (изменение нагрузки, часы после пуска) — не датчики: в расстоянии не участвуют."""
    with_conditions = True

    def __init__(self, cond, kind, classic=False):
        self.kind, self.classic = kind, classic
        self.cond = list(DERIVED_COND if classic else cond)
        self.export_cond = self.cond if classic else None

    def _split(self, F, prep):
        ci, ti = K._cond_split(self.cond, prep)
        n = len(prep.num)
        return F[:, ti].astype(float), F[:, ci].astype(float), F[:, n:].astype(float), ti, n

    def fit(self, F, prep):
        X, C_, B, ti, _ = self._split(F, prep)
        if self.classic:
            par = dict(mode='all', linear=False, robust=False, cond_weight=0.0, state_weight=1.0)
            self.m = AAKR(h=AAKR_H, **par) if self.kind == 'aakr' else MSET(h=MSET_H, **par)
            run_ = [k for k, j in enumerate(ti) if prep.num[j] == run_col]
            self.m.strat_col = run_[0] if run_ else None        # память — равномерно по нагрузке и механизмам
        else:
            self.m = AAKR(h=AAKR_H) if self.kind == 'aakr' else MSET(h=MSET_H)
        self.m.fit(X, C_, B)
        return self

    def errors(self, F, prep):
        X, C_, B, ti, n = self._split(F, prep)
        Ex, S = self.m.predict(X, C_, B)
        E = np.zeros((len(F), n))
        E[:, ti] = np.nan_to_num((X - Ex) / S)
        return E


TI = list(range(len(TARGETS)))                   # проверяемые датчики — первые столбцы и X_cond, и X_reg
RESID['M'] = MISS_cond[:, TI]                    # их пропуски (одинаковы у обеих первых ступеней)
# вид первой ступени -> название, конструктор, ядро для внедрения, условия, префикс папок models/<агрегат>/
STAGES = {'analogs': dict(title='Остатки analogs', make=lambda: K.AnalogCondDet(COND_COLS), core='analogs', cond=COND_COLS,
                           stage_title='первая ступень: analogs', prefix='resid_'),
          'регрессия': dict(title='Остатки регрессии', make=regression, core='boost', cond=COND_COLS_REG,
                            stage_title='первая ступень: Регрессия', prefix='resid_reg_')}
STAGES.update({
    'AAKR': dict(title='Остатки AAKR', make=lambda: KernelDet(COND_COLS, 'aakr'),
                 core='kernel', cond=COND_COLS, stage_title='первая ступень: AAKR', prefix='resid_aakr_'),
    'MSET': dict(title='Остатки MSET', make=lambda: KernelDet(COND_COLS, 'mset'),
                 core='kernel', cond=COND_COLS, stage_title='первая ступень: MSET', prefix='resid_mset_')})
_base_aliases = {'analogs': 'analogs', 'aakr': 'AAKR', 'mset': 'MSET',
                 'regression': 'регрессия', 'регрессия': 'регрессия'}
try:
    SELECTED_RESIDUAL_BASES = list(dict.fromkeys(_base_aliases[str(b).strip().lower()] for b in RESIDUAL_BASES))
except KeyError as ex:
    raise ValueError(f'Неизвестная первая ступень {ex.args[0]!r}; доступны analogs, AAKR, MSET, regression.') from ex
# вторая ступень: ключ -> (конструктор, название в видах, сколько датчиков в ошибке момента, папка, ядро)
SECOND = {'PCA': (lambda: K.PCADet(), 'PCA', 5, 'pca', 'pca'),
          'Махаланобис': (lambda: K.MahalDet(), 'Махаланобис', 1, 'mahalanobis', 'mahalanobis'),
          'Isolation Forest': (lambda: K.IForestDet(), 'Isolation Forest', 1, 'iforest', 'iforest')}
if HAS_TF:
    SECOND.update({'Автоэнкодер': (lambda: K.AEDet(epochs=RESIDUAL_AE_EPOCHS, max_train=RESIDUAL_MAX_TRAIN), 'автоэнкодер', 5, 'autoencoder', 'autoencoder'),
                   'LSTM-автоэнкодер': (lambda: K.LSTMDet(L=RESIDUAL_LSTM_STEPS, epochs=RESIDUAL_LSTM_EPOCHS, max_train=RESIDUAL_MAX_TRAIN), 'LSTM-автоэнкодер', 5, 'lstm_ae', 'lstm')})
_method_aliases = {'PCA': 'PCA', 'Махаланобис': 'Махаланобис', 'Isolation Forest': 'Isolation Forest',
                   'AE': 'Автоэнкодер', 'LSTM': 'LSTM-автоэнкодер',
                   'Автоэнкодер': 'Автоэнкодер', 'LSTM-автоэнкодер': 'LSTM-автоэнкодер'}
try:
    _selected_methods = list(dict.fromkeys(_method_aliases[m] for m in RESIDUAL_METHODS))
except KeyError as ex:
    raise ValueError(f'Неизвестная вторая ступень {ex.args[0]!r}; доступны {list(_method_aliases)}.') from ex
_unavailable = [m for m in _selected_methods if m not in SECOND]
if _unavailable:
    print('Пропущены нейросети (USE_NEURAL=False или нет TensorFlow):', ', '.join(_unavailable))
SECOND = {k: SECOND[k] for k in _selected_methods if k in SECOND}
if _selected_methods and not SECOND and SELECTED_RESIDUAL_BASES:
    raise ValueError('Не осталось доступных вторых ступеней: для AE/LSTM нужны USE_NEURAL=True и готовое окружение TensorFlow.')
RESID_L = max((getattr(s[0](), 'L', 0) for s in SECOND.values()), default=0)
BAG = [k for k in SECOND if k != 'Isolation Forest'] if RESIDUAL_BAGGING else []        # бэггинг: все методы на остатках, кроме IF
RESID_OUT = {}

def resid_methods(kind):
    """Методы второй ступени для вида первой ступени: ключ -> (конструктор, виды)."""
    t = STAGES[kind]['title']
    return {k: ((lambda mk=mk: K.on_residuals(mk())), [dict(name=f'{t} → {label}', top=top)])
            for k, (mk, label, top, _, _) in SECOND.items()}

def bag_view(kind):
    return dict(name=f'{STAGES[kind]["title"]}: бэггинг (среднее)', top=1)

def set_resid(parts, test):
    """Остатки первой ступени на весь ряд: сначала от модели на всём годе (месяц и история перед ним),
    поверх — на незнакомых данных для обучающего года."""
    Z = np.full((len(t_ix), len(TARGETS)), np.nan, np.float32)
    for E, ix in test + parts:
        Z[t_ix.get_indexer(ix)] = E[:, TI]
    RESID['X'] = pd.DataFrame(Z, index=t_ix, columns=TARGETS)

def stage1(kind, m0, L=0):
    """Первая ступень на момент m0: остатки на обучающем годе (на незнакомых данных) и в месяце m0
    с историей L шагов перед ним (для окон LSTM)."""
    tm1 = train_one(STAGES[kind]['make'], m0, [dict(name='z', top=1)], oof_rows=None)
    if tm1 is None:
        return None
    rows = scorable.to_numpy() & enough.to_numpy() & (t_ix >= m0 - L * STEP) & (t_ix < m0 + pd.offsets.MonthBegin(1))
    E, ix = model_errors(tm1['det'], tm1['prep'], rows) if rows.any() else (None, None)
    set_resid(tm1['parts'], [(E, ix)] if E is not None else [])
    return tm1

def bag_ratio(tm, v, E, ix):
    """Отношение «ошибка момента / порог» одной модели бэггинга."""
    q, thr = tm['cal'][v['name']]
    return pd.Series(K.top_mean(np.abs(E) / q, v['top']) / thr, index=ix)

def bag_threshold(members):
    """Порог бэггинга: 99.5% среднего отношения моделей на незнакомых частях года (как пороги самих моделей)."""
    S = pd.concat([bag_ratio(tm, v, np.vstack([e for e, _ in tm['parts']]), np.concatenate([i for _, i in tm['parts']]))
                   for tm, v in members], axis=1)
    return float(np.nanquantile(S.mean(axis=1), ALARM_Q))

def run_resid(kind):
    st, methods, bv = STAGES[kind], resid_methods(kind), bag_view(kind)
    views = [v for _, vs in methods.values() for v in vs] + ([bv] if len(BAG) >= 2 else [])
    def compute():
        out = new_out(views)
        for i, m0 in enumerate(months):
            t0 = time.time()
            in_m = (t_ix >= m0) & (t_ix < m0 + pd.offsets.MonthBegin(1))
            te = scorable.to_numpy() & enough.to_numpy() & in_m
            if not te.any() or stage1(kind, m0, RESID_L) is None:
                continue
            got = {}
            for key, (mk, vs) in methods.items():
                tm = train_one(mk, m0, vs)
                if tm is None:
                    continue
                E, ix = model_errors(tm['det'], tm['prep'], te)
                if E is not None:
                    C = getattr(tm['det'], 'last_contrib', None)
                    fill(out, tm, E, ix, m0, in_m, vs, C)
                    if key in BAG:
                        got[key] = (tm, vs[0], E, ix, C)
            if len(got) >= 2:                            # бэггинг: среднее отношений, вклады — среднее нормированных ошибок датчиков
                c = pd.concat([bag_ratio(tm, v, E, ix) for tm, v, E, ix, _ in got.values()], axis=1).mean(axis=1)
                o = out[bv['name']]
                o['model_cutoff'].loc[c.index] = m0
                o['score'].loc[c.index] = c.to_numpy()
                o['thr'].loc[in_m] = bag_threshold([(tm, v) for tm, v, _, _, _ in got.values()])
                Ms = [pd.DataFrame(np.abs(E) / tm['cal'][v['name']][0] if E.shape[1] > 1 else np.abs(C) / tm['qc'], index=ix)
                      for tm, v, E, ix, C in got.values() if E.shape[1] > 1 or (C is not None and tm.get('qc') is not None)]
                if Ms:
                    Mb = pd.concat(Ms).groupby(level=0).mean()
                    if o['contrib'] is None:
                        o['contrib'] = np.full((len(t_ix), len(TARGETS)), np.nan, dtype='float16')
                        o['cols'] = list(TARGETS)
                    o['contrib'][t_ix.get_indexer(Mb.index)] = Mb.to_numpy().astype('float16')
            if i == 0:
                if globals().get('VERBOSE', False):
                    print(f'  {st["title"]}: месяц — {time.time() - t0:.0f} с, всего месяцев {len(months)}', flush=True)
        return out
    return cached(st['title'], compute, dict(SIG, cond=st['cond'], methods=sorted(methods), bag=BAG))

def export_resid(kind):
    """Методы на остатках -> папки с общей первой ступенью (своя копия в stage1/ каждой папки)."""
    st, methods = STAGES[kind], resid_methods(kind)
    sig = export_sig(method=st['title'], cond=st['cond'], methods=sorted(methods))
    paths = {k: os.path.join(DEPLOY, st['prefix'] + SECOND[k][3], 'final') for k in methods}
    if all(already(p, sig) for p in paths.values()):
        for k, (_, vs) in methods.items():
            register(paths[k], vs[0]['name'], vs)
        if globals().get('VERBOSE', False):
            print(f'  {st["title"]}: папки уже сохранены — не переобучаем')
        return
    t0 = time.time()
    tm1 = stage1(kind, M_NEXT, RESID_L)
    if tm1 is None:
        return
    STAGE1_TM[kind], RESID_TM[kind] = tm1, {}
    for key, (mk, views) in methods.items():
        tm = train_one(mk, M_NEXT, views)
        if tm is None:
            continue
        core = SECOND[key][4]
        D.export_model(paths[key], views[0]['name'], core, tm['det'], tm['prep'], vcfg(tm, views), COMMON, inp='residuals', cond=st['cond'],
                       qc=tm['qc'], stage1=(tm1['prep'], tm1['det'], st['core'], st['stage_title'], st['cond']), quality=quality(views),
                       extra=dict(sig=sig, **({'L': tm['det'].L} if core == 'lstm' else {})))
        register(paths[key], views[0]['name'], views)
        FINAL[paths[key]] = (tm, views, kind)
        RESID_TM[kind][key] = (tm, views[0], paths[key])
        if globals().get('VERBOSE', False):
            print(f'  {views[0]["name"]}: сохранено в {paths[key]}', flush=True)
    if globals().get('VERBOSE', False):
        print(f'  {st["title"]}: итоговые модели — {(time.time() - t0) / 60:.1f} мин')

def bagging(kind):
    """Бэггинг вида: оценка и экспорт (общая первая ступень + вторые ступени моделей)."""
    bv, st = bag_view(kind), STAGES[kind]
    if bv['name'] not in RESID_OUT.get(kind, {}):
        if globals().get('VERBOSE', False):
            print(f'{bv["name"]}: не посчитан (нужно хотя бы два метода на остатках).')
        return
    RES[bv['name']] = RESID_OUT[kind][bv['name']]
    show(bv['name'])
    if globals().get('VERBOSE', False):
        say(f'В бэггинге: {fmt_list([resid_methods(kind)[k][1][0]["name"] for k in BAG])}.')
    got = RESID_TM.get(kind, {})
    bag = [k for k in BAG if k in got]
    path = os.path.join(DEPLOY, st['prefix'] + 'bagging', 'final')
    if EXPORT and len(bag) >= 2:
        D.export_bagging(path, bv['name'], [got[k][2] for k in bag], os.path.join(got[bag[0]][2], 'stage1'),
                         dict(name=bv['name'], top=1, thr=bag_threshold([got[k][:2] for k in bag]),
                              качество_на_проверке=QUALITY_OF(bv['name'])), COMMON)
        register(path, bv['name'], [bv])
        if globals().get('VERBOSE', False):
            print(f'  Бэггинг: сохранено в {path}')
    elif EXPORT and os.path.exists(os.path.join(path, 'config.json')):
        register(path, bv['name'], [bv])             # модели на остатках не переобучались — прежняя папка бэггинга

def run_two_stage(kind):
    """Первая ступень вида kind + все методы второй ступени и их бэггинг: оценка, графики, экспорт."""
    st = STAGES[kind]
    if not wanted(st['title']):
        print(f'{st["title"]}: пропущено (RUN_ONLY)')
        return
    started = time.perf_counter()
    before_ops, before_exports = len(GROUP_ENGINE.operation_audit), len(EXPORTED)
    print(f'{GROUP_NAME} · {UNIT} · {st["title"]}', flush=True)
    RESID_OUT[kind] = out = run_resid(kind)
    for _, vs in resid_methods(kind).values():
        for v in vs:
            RES[v['name']] = out[v['name']]
            show(v['name'])
    if EXPORT:
        export_resid(kind)
    bagging(kind)
    ops = GROUP_ENGINE.operation_audit[before_ops:]
    print(f"  Готово за {elapsed_text(started)}; обучений: {sum(int(r['fit_calls']) for r in ops)}; "
          f"финальных экспортов: {len(EXPORTED) - before_exports}.", flush=True)
GROUP_ENGINE.install_residual_stage(globals())

