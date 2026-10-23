"""
Внедрение моделей котлов: экспорт каждой модели в свою папку (models/<агрегат>/<метод>/<ГГГГ-ММ | final>/) и код инференса.

Часть 1 — ИНФЕРЕНС. Функции и классы этой части копируются в model.py каждой папки (только те, что нужны этой
модели), поэтому папка самодостаточна: config.json, sensors.csv, scaler.json, параметры и model.py.
Часть 2 — ЭКСПОРТ (вызывается из ноутбука 02, раздел 7): запись файлов модели и сборка её model.py.
"""
import os
import json
import inspect

import numpy as np
import pandas as pd

import kernel_models as KM                  # AAKR / MSET: их код инференса копируется в model.py папки (ядро «kernel»)
from kernel_models import AAKR, MSET, Linear, robust_scale, spd_solve, _chol_solve

# ============================================================================== ЧАСТЬ 1. ИНФЕРЕНС

TIME_COL = 'TimeValue'
M_STOP, M_START, M_RUN, M_PRESTOP = 'стоит', 'пуск', 'работа', 'перед остановом'
COND_LOAD_CHANGE, COND_SINCE_START, COND_MILLS = 'условие: изменение нагрузки за час', 'условие: часов после пуска', 'условие: мельниц в работе'


def read_json(folder, name):
    with open(os.path.join(folder, name), encoding='utf-8') as f:
        return json.load(f)


def prepare(df):
    """Часовая таблица -> время в индексе (из столбца TimeValue или индекса), по возрастанию, сетка 1 ч без пропусков."""
    d = df.set_index(TIME_COL) if TIME_COL in df.columns else df.copy()
    d.columns = [str(c).strip() for c in d.columns]          # в выгрузке бывают пробелы в конце имени (при обучении обрезаны)
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index)).round('1h')
    d = d[~d.index.duplicated(keep='last')].sort_index()
    return d.asfreq('1h')


def detect_mode(x, thr, startup, prestop, min_stop, ffill):
    """Режим по датчику нагрузки: стоит / пуск (первые startup шагов после включения) / перед остановом / работа.
    Остановы короче min_stop шагов не считаются; пропуски нагрузки до ffill шагов заполняются."""
    running = (x.ffill(limit=ffill) > thr)
    grp = (running != running.shift()).cumsum()
    size = running.groupby(grp).transform('size')
    first_grp, last_grp = grp.iloc[0], grp.iloc[-1]
    running = running | (~running & (size < min_stop) & (grp != first_grp) & (grp != last_grp))
    grp = (running != running.shift()).cumsum()
    since = running.groupby(grp).cumcount().astype(float)
    to_stop = running.groupby(grp).cumcount(ascending=False).astype(float)
    first_grp, last_grp = grp.iloc[0], grp.iloc[-1]
    since[grp.eq(first_grp)] = np.inf
    to_stop[grp.eq(last_grp)] = np.inf
    mode = pd.Series(np.where(running, M_RUN, M_STOP), index=x.index, dtype=object)
    mode[running & (to_stop < prestop)] = M_PRESTOP
    mode[running & (since < startup)] = M_START
    return mode


def condition_frame(df, spec):
    """Производные условия работы: изменение нагрузки за час, часы после пуска (до 72), число работающих мельниц."""
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


def rolling_mean(s, n_steps, gap='3h'):
    """Скользящее среднее по n_steps шагам внутри непрерывных участков (после разрыва > gap — заново)."""
    s = s.dropna()
    seg = (s.index.to_series().diff() > pd.Timedelta(gap)).cumsum()
    return s.groupby(seg).transform(lambda v: v.rolling(n_steps, min_periods=1).mean())


def slow(E, index, hours):
    """Медленный сдвиг: ошибка каждого датчика (со знаком), усреднённая за последние hours часов."""
    if not hours:
        return E
    return pd.DataFrame(E, index=index).rolling(f'{hours}h', min_periods=1).mean().to_numpy()


def top_mean(R, k):
    """Средняя из k наибольших нормированных ошибок датчиков в каждой строке."""
    R = np.nan_to_num(np.asarray(R, float))
    k = min(k, R.shape[1])
    return np.sort(R, axis=1)[:, -k:].mean(axis=1)


def windows(F, ends, L):
    """Окна F[i−L+1 … i] для строк ends: (число окон, L, признаки)."""
    from numpy.lib.stride_tricks import sliding_window_view
    return np.ascontiguousarray(sliding_window_view(F, L, axis=0)[np.asarray(ends) - L + 1].transpose(0, 2, 1))


def mean_finite(A, axis):
    """Среднее по оси без пропусков; если значений нет — NaN (без предупреждений)."""
    ok = np.isfinite(A)
    n = ok.sum(axis=axis)
    return np.where(n > 0, np.where(ok, A, 0.0).sum(axis=axis) / np.maximum(n, 1), np.nan)


def alarm_episodes(ratio, persist, merge, step):
    """Эпизоды тревоги: отношение к порогу > 1 не меньше persist шагов подряд; разрыв меньше merge шагов — склейка."""
    above = (ratio > 1).fillna(False).to_numpy()
    idx = ratio.index
    eps, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if j - i >= persist:
                if (eps and (idx[i] - eps[-1][1]) < merge * step
                        and ratio.loc[eps[-1][1]:idx[i] - step].notna().all()):
                    eps[-1][1] = idx[j - 1] + step
                else:
                    eps.append([idx[i], idx[j - 1] + step])
            i = j
        else:
            i += 1
    return eps


def _nan_quantiles_sorted(A, qs):
    """Квантили по оси 1 с пропусками: A (n, k, p) -> (len(qs), n, p) и число непустых (n, p)."""
    B = np.ascontiguousarray(A.transpose(0, 2, 1))
    B.sort(axis=-1)
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


class Scaler:
    """Нормировка входа, как при обучении: (x − медиана) / масштаб, обрезка ±10, пропуск -> 0 (обычное значение);
    текстовые датчики -> 1, если механизм не в своём обычном состоянии (scaler.json)."""

    def __init__(self, folder):
        d = read_json(folder, 'scaler.json')
        self.num, self.bools = d['num'], [tuple(b) for b in d['bools']]
        self.med, self.sd = np.array(d['med'], float), np.array(d['sd'], float)

    def transform(self, df):
        X = df.reindex(columns=self.num).astype('float64').to_numpy()
        Z = np.nan_to_num(np.clip((X - self.med) / self.sd, -10, 10)).astype('float32')
        if self.bools:
            B = np.zeros((len(df), len(self.bools)), dtype='float32')
            for j, (c, main) in enumerate(self.bools):
                if c in df.columns:
                    v = df[c].astype(object).to_numpy()
                    B[:, j] = np.where(pd.isna(v), 0.0, (v != main).astype('float32'))
            Z = np.hstack([Z, B])
        return Z, np.isnan(X)


class Result:
    """Результат score(): hourly, mode, sensor_errors, contrib_share, expected, alarms, missing (см. README)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return 'Result(' + ', '.join(self.__dict__) + ')'


# ---------------------------------------------------------------- ядра моделей: ошибки по нормированному входу F

class AnalogCore:
    """Аналоги: для каждого часа — k самых похожих часов нормы по условиям (память в params.npz); ожидание датчика —
    их медиана (+ линейная поправка на условия), разброс — их межквартильный размах; ошибка — (факт − ожидание) / разброс."""

    def __init__(self, folder):
        from sklearn.neighbors import NearestNeighbors
        p = np.load(os.path.join(folder, 'params.npz'))
        for k in p.files:
            setattr(self, k, p[k])
        self.kk, self.min_valid, self.n = int(self.kk), int(self.min_valid), int(self.n)
        self.nn = NearestNeighbors(n_neighbors=self.kk).fit(self.Rtr)

    def expected(self, F):
        R = F[:, self.r_idx].astype('float64')
        Rq = (R - self.r_med) / self.r_scale
        n, p = len(Rq), self.Xtr.shape[1]
        med, sc, dist = np.full((n, p), np.nan), np.full((n, p), np.nan), np.full(n, np.nan)
        batch = max(10, int(4_000_000 // max(self.kk * p, 1)))
        for s in range(0, n, batch):
            rq = np.arange(s, min(s + batch, n))
            _, nb = self.nn.kneighbors(Rq[rq])
            (q25, q50, q75), cnt = _nan_quantiles_sorted(self.Xtr[nb].astype('float64'), [0.25, 0.5, 0.75])
            q50[cnt < self.min_valid] = np.nan
            med[rq], sc[rq] = q50, np.maximum((q75 - q25) / 1.349, self.floor)
            D = Rq[rq][:, None, self.didx] - self.Rtr[nb][:, :, self.didx]
            dist[rq] = np.sqrt((D ** 2).sum(axis=-1)).mean(axis=1)
        L = np.column_stack([np.ones(n), R[:, self.lidx] - self.r_med[self.lidx]])
        return med + L @ self.B, sc, dist

    def errors(self, F):
        med, sc, dist = self.expected(F)
        E = np.zeros((len(F), self.n))
        E[:, self.t_idx] = np.nan_to_num((F[:, self.t_idx] - med) / sc)
        return dict(E=E, med=med, sc=sc, dist=dist)


class AnalogBoostCore(AnalogCore):
    """analogs с поправкой: отклонение аналогов минус предсказанная бустингом систематическая ошибка (corr.joblib)."""

    def __init__(self, folder):
        import joblib
        super().__init__(folder)
        c = joblib.load(os.path.join(folder, 'corr.joblib'))
        self.corr, self.dist_fill = c['models'], c['dist_fill']

    def errors(self, F):
        med, sc, dist = self.expected(F)
        Z = np.nan_to_num((F[:, self.t_idx] - med) / sc)
        feats = np.column_stack([F[:, self.r_idx], np.where(np.isnan(dist), self.dist_fill, dist)])
        P = np.zeros_like(Z)
        for k, ms in enumerate(self.corr):
            if ms is not None:
                P[:, k] = ms[0].predict(feats)
        E = np.zeros((len(F), self.n))
        E[:, self.t_idx] = Z - P
        return dict(E=E, med=med + P * sc, sc=sc, dist=dist)      # ожидание с поправкой


class BoostCore:
    """Бустинг «условия → датчики»: медиана и квартили каждого датчика по условиям (boost.joblib);
    ошибка — (факт − медиана) / разброс."""

    def __init__(self, folder):
        import joblib
        b = joblib.load(os.path.join(folder, 'boost.joblib'))
        self.in_idx, self.t_idx, self.n, self.floor, self.models = b['in_idx'], b['t_idx'], b['n'], b['floor'], b['models']

    def errors(self, F):
        Xin = F[:, self.in_idx]
        med = np.full((len(F), len(self.t_idx)), np.nan)
        sc = np.full((len(F), len(self.t_idx)), np.nan)
        for k, ms in enumerate(self.models):
            if ms is not None:
                p25, p50, p75 = (m.predict(Xin) for m in ms)
                med[:, k], sc[:, k] = p50, np.maximum(np.abs(p75 - p25) / 1.349, self.floor[k])
        E = np.zeros((len(F), self.n))
        E[:, self.t_idx] = np.nan_to_num((F[:, self.t_idx] - med) / sc)
        return dict(E=E, med=med, sc=sc)


class PcaCore:
    """PCA: час сжимается до главных компонент нормы и восстанавливается; ошибка датчика — |факт − восстановленное|."""

    def __init__(self, folder):
        p = np.load(os.path.join(folder, 'params.npz'))
        self.mean, self.comps, self.n = p['mean'], p['comps'], int(p['n'])

    def errors(self, F):
        rec = (F - self.mean) @ self.comps.T @ self.comps + self.mean
        return dict(E=np.abs(F - rec)[:, :self.n])


class MahalCore:
    """Махаланобис: насколько сочетание значений необычно с учётом совместных изменений в норме; вклад признака —
    d_j·(P·d)_j / p (сумма вкладов = ошибке)."""

    def __init__(self, folder):
        p = np.load(os.path.join(folder, 'params.npz'))
        self.mu, self.P = p['mu'], p['P']

    def errors(self, F):
        D = F - self.mu
        C = D * (D @ self.P) / F.shape[1]
        return dict(E=C.sum(axis=1)[:, None], C=C)


class IForestCore:
    """Isolation Forest (sklearn, iforest.joblib): насколько легко отделить час от нормы. Разбивки по датчикам нет."""

    def __init__(self, folder):
        import joblib
        self.m = joblib.load(os.path.join(folder, 'iforest.joblib'))

    def errors(self, F):
        return dict(E=-self.m.score_samples(F)[:, None])


class AeCore:
    """Автоэнкодер (net.keras): ошибка датчика — |факт − восстановленное|."""

    def __init__(self, folder):
        import tensorflow as tf
        self.net = tf.keras.models.load_model(os.path.join(folder, 'net.keras'))
        self.n = int(read_json(folder, 'config.json')['n'])

    def errors(self, F):
        return dict(E=np.abs(self.net.predict(F, verbose=0, batch_size=4096) - F[:, :self.n]))


class LstmCore:
    """LSTM-автоэнкодер (net.keras): окно из L часов; ошибка датчика — средняя по окну (без пропущенных значений)."""

    def __init__(self, folder):
        import tensorflow as tf
        self.net = tf.keras.models.load_model(os.path.join(folder, 'net.keras'))
        cfg = read_json(folder, 'config.json')
        self.n, self.L = int(cfg['n']), int(cfg['L'])

    def errors(self, W, mask):
        out = []
        for a in range(0, len(W), 2048):
            w = W[a:a + 2048]
            e = np.abs(self.net.predict(w, verbose=0, batch_size=512) - w[:, :, :self.n])
            m = ~mask[a:a + 2048]
            out.append((e * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1))
        return dict(E=np.vstack(out) if out else np.zeros((0, self.n)))


class CondNetCore:
    """Условная нейросеть (net.keras): по условиям работы предсказывает датчики; ошибка — |факт − прогноз|."""

    def __init__(self, folder):
        import tensorflow as tf
        self.net = tf.keras.models.load_model(os.path.join(folder, 'net.keras'))
        p = np.load(os.path.join(folder, 'params.npz'))
        self.in_idx, self.t_idx, self.n = p['in_idx'], p['t_idx'], int(p['n'])

    def errors(self, F):
        E = np.zeros((len(F), self.n))
        E[:, self.t_idx] = np.abs(self.net.predict(F[:, self.in_idx], verbose=0, batch_size=4096) - F[:, self.t_idx])
        return dict(E=E)


class TwinsCore:
    """Близнецы: датчик ≈ a + b × медиана остальных близнецов; ошибка — отход от прямой в обычных разбросах (twins.json)."""

    def __init__(self, folder):
        t = read_json(folder, 'twins.json')
        self.n, self.models = t['n'], t['models']

    def errors(self, F):
        E = np.zeros((len(F), self.n))
        for m in self.models:
            ref = np.median(F[:, m['others']], axis=1)
            E[:, m['j']] = np.abs((F[:, m['j']] - (m['k'] * ref + m['b']) - m['m']) / m['s'])
        return dict(E=E)


class KernelCore:
    """AAKR / MSET: ожидание датчика — по похожим часам памяти (params.npz) с учётом условий работы, состояний механизмов
    и связанных с ним датчиков; ошибка — (факт − ожидание) / разброс остатка. Код моделей — классы AAKR и MSET ниже."""

    def __init__(self, folder):
        p = np.load(os.path.join(folder, 'params.npz'))
        kind = str(p['kind'])
        m = MSET(local=int(p['local']), lam=float(p['lam']), mode=str(p['mode'])) if kind == 'mset' else AAKR(mode=str(p['mode']))
        m.h, m.robust, m.robust_thr = float(p['h']), bool(p['robust']), float(p['robust_thr'])
        m.cond_weight, m.state_weight = float(p['cond_weight']), float(p['state_weight'])
        for k in ('xm', 'xs', 'cm', 'cs', 'nb', 'Mc', 'Mx'):
            setattr(m, k, p[k])
        m.p = int(p['p'])
        m.lin = None
        if 'lin_beta' in p.files:
            m.lin = Linear()
            m.lin.beta, m.lin.cm, m.lin.cs = p['lin_beta'], p['lin_cm'], p['lin_cs']
        self.m = m
        self.c_idx, self.t_idx, self.n = p['c_idx'].astype(int), p['t_idx'].astype(int), int(p['n'])

    def errors(self, F):
        X, C, B = F[:, self.t_idx].astype(float), F[:, self.c_idx].astype(float), F[:, self.n:].astype(float)
        Ex, S = self.m.predict(X, C, B)
        E = np.zeros((len(F), self.n))
        E[:, self.t_idx] = np.nan_to_num((X - Ex) / S)
        return dict(E=E, med=Ex, sc=S)


class RareCore:
    """Редкие состояния механизмов: сумма −log10(доля в норме) по состояниям реже 1% (rare.json);
    состояние, державшееся дольше held_steps шагов, — устоявшееся. Вклад — по каждому текстовому датчику."""

    def __init__(self, folder):
        r = read_json(folder, 'rare.json')
        self.cols, self.held, self.n_fit, self.freq = r['cols'], r['held_steps'], r['n'], r['freq']

    def score(self, df):
        C = np.zeros((len(df), len(self.cols)))
        for j, c in enumerate(self.cols):
            if c not in df.columns:
                continue
            v = df[c].astype(object)
            held = v.groupby((v != v.shift()).cumsum()).cumcount().to_numpy()
            p = v.map(self.freq[c]).astype(float).where(v.notna(), 1.0).fillna(0.5 / (self.n_fit + 1)).to_numpy()
            C[:, j] = np.where((p < 0.01) & (held < self.held), -np.log10(p), 0.0)
        return C.sum(axis=1), C


# ---------------------------------------------------------------- модель: вход -> ядро -> ошибка момента, тревоги, вклады

class Model:
    """Одна модель из папки: config.json (параметры, виды, пороги), sensors.csv (датчики и их обычные ошибки),
    scaler.json (нормировка), файлы параметров; для моделей на остатках — первая ступень в stage1/."""

    def __init__(self, folder, stage1=None):
        self.folder = folder
        self.cfg = read_json(folder, 'config.json')
        self.core = CORES[self.cfg['core']](folder)
        self.scaler = Scaler(folder) if os.path.exists(os.path.join(folder, 'scaler.json')) else None
        if self.cfg['input'] == 'residuals':
            self.stage1 = stage1 if stage1 is not None else Model(os.path.join(folder, 'stage1'))
        sens = pd.read_csv(os.path.join(folder, 'sensors.csv'), encoding='utf-8-sig')
        self.cols = sens['датчик'].tolist()
        self.q = {v['name']: (sens[f'обычная ошибка: {v["name"]}'].to_numpy(float) if f'обычная ошибка: {v["name"]}' in sens
                              else np.array([v.get('q', 1.0)], float)) for v in self.cfg['views']}
        self.qc = sens['обычный вклад'].to_numpy(float) if 'обычный вклад' in sens else None

    def needed(self):
        """Столбцы, которые нужны модели во входной таблице."""
        cfg = self.cfg
        need = [cfg['run_col']]
        if self.scaler is not None and cfg['input'] != 'residuals':
            need += [c for c in self.scaler.num if not c.startswith('условие: ')] + [c for c, _ in self.scaler.bools]
        if cfg['input'] == 'residuals':
            need += self.stage1.needed()
        if cfg['core'] == 'rare':
            need += self.core.cols
        if cfg.get('cond_spec'):
            need += list(cfg['cond_spec']['mills'])
        return list(dict.fromkeys(need))

    def _inputs(self, df, rows, ok):
        cfg = self.cfg
        d = df
        if cfg['input'] in ('conditions', 'residuals') and cfg.get('cond_spec'):
            cf = condition_frame(df, cfg['cond_spec'])
            d = pd.concat([df, cf[[c for c in cf.columns if c not in df.columns]]], axis=1)
        blank = [c for c in (cfg.get('blank_cols') or []) if c in self.scaler.num]   # датчики, которых почти не было при
        if cfg['input'] == 'residuals':                                              # обучении: модель их не проверяет
            Z = pd.DataFrame(self.stage1.raw_errors(df, rows, ok), index=df.index, columns=self.stage1.cols)[self.scaler.num]
            if blank:
                Z = Z.copy()
                Z[blank] = np.nan
            F, _ = self.scaler.transform(Z)
            miss = d.reindex(columns=self.scaler.num).isna().to_numpy()
            if blank:
                miss[:, [self.scaler.num.index(c) for c in blank]] = True
            return F, miss
        if blank:
            d = d.copy()
            d[[c for c in blank if c in d.columns]] = np.nan
        return self.scaler.transform(d)

    def raw_errors(self, df, rows, ok):
        """Ошибки датчиков по всем строкам df (NaN — строка не оценивалась); первая ступень для моделей на остатках."""
        key = (id(df), rows.tobytes())
        if getattr(self, '_memo', (None,))[0] != key:
            E, pos, _, _ = self.errors(df, rows, ok)
            Z = np.full((len(df), E.shape[1]), np.nan)
            Z[pos] = E
            self._memo = (key, Z)
        return self._memo[1]

    def errors(self, df, rows, ok):
        """Ошибки датчиков в оцениваемых строках: (E, номера строк, вклады C или None, ожидание или None)."""
        if self.cfg['core'] == 'rare':
            pos = np.flatnonzero(rows)
            s, C = self.core.score(df)
            return s[pos, None], pos, C[pos], None
        F, miss = self._inputs(df, rows, ok)
        L = getattr(self.core, 'L', 0)
        if L:
            full = pd.Series(ok.astype(int)).rolling(L, min_periods=L).sum().eq(L).to_numpy()
            pos = np.flatnonzero(full & rows)
            if not len(pos):
                return np.zeros((0, self.core.n)), pos, None, None
            return self.core.errors(windows(F, pos, L), windows(miss, pos, L))['E'], pos, None, None
        pos = np.flatnonzero(rows)
        out = self.core.errors(F[pos])
        E = out['E']
        if E.shape[1] > 1:
            E = np.where(miss[pos], 0.0, E)
        exp = None
        if 'med' in out:                                   # ожидание и разброс — в единицах датчиков
            t = np.asarray(self.core.t_idx)
            sd, m0 = self.scaler.sd[t], self.scaler.med[t]
            names = [self.scaler.num[j] for j in t]
            ix = df.index[pos]
            exp = pd.concat({'ожидание': pd.DataFrame(out['med'] * sd + m0, index=ix, columns=names),
                             'разброс': pd.DataFrame(out['sc'] * sd, index=ix, columns=names),
                             'факт': df.reindex(columns=names).loc[ix]}, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
        return E, pos, out.get('C'), exp

    def score(self, df, prepared=False):
        """Оценка часовой таблицы df (история ≥ 4 суток до оцениваемых часов). Возвращает Result."""
        df = df if prepared else prepare(df)
        cfg = self.cfg
        mode = detect_mode(df[cfg['run_col']].astype(float), cfg['run_thr'], cfg['startup_steps'], cfg['prestop_steps'],
                           cfg['min_stop_steps'], cfg['ffill_steps'])
        ok = mode.isin([M_RUN, M_PRESTOP]).to_numpy()
        rows = ok.copy()
        if cfg.get('use_presence', True):
            rows &= df.reindex(columns=cfg['presence_cols']).notna().mean(axis=1).to_numpy() >= cfg['min_present']
        E, pos, C, exp = self.errors(df, rows, ok)
        return self._result(df, mode, E, pos, C, exp)

    def _result(self, df, mode, E, pos, C, exp):
        cfg = self.cfg
        step = pd.Timedelta(cfg['step'])
        running = mode.ne(M_STOP)
        ix = df.index[pos]
        cols, errs, shares, eps = {}, {}, {}, []
        for v in cfg['views']:
            name = v['name']
            R = np.abs(slow(E, ix, v.get('slow_h', 0))) / self.q[name]
            s = pd.Series(top_mean(R, v['top']), index=ix)
            sm = rolling_mean(s, cfg['smooth_steps']).reindex(df.index)
            ratio = (sm / v['thr']).where(running, 0.0)
            M = None
            if R.shape[1] > 1:
                M = pd.DataFrame(R, index=ix, columns=self.cols)
            elif C is not None:
                M = pd.DataFrame(np.abs(C) / self.qc if self.qc is not None else C, index=ix, columns=self.cols)
            if M is not None:
                errs[name] = M
                tot = M.sum(axis=1).replace(0, np.nan)
                shares[name] = M.div(tot, axis=0) * 100
            flag = pd.Series(False, index=df.index)
            for a, b in alarm_episodes(ratio.where(mode.isin([M_RUN, M_PRESTOP]) & s.reindex(df.index).notna()), cfg['persist_steps'], cfg['merge_steps'], step):
                flag[(df.index >= a) & (df.index < b)] = True
                top = (', '.join(f'{c} ×{x:.1f}' for c, x in M.loc[(M.index >= a) & (M.index < b)].quantile(0.9).nlargest(3).items())
                       if M is not None and ((M.index >= a) & (M.index < b)).any() else '')
                eps.append(dict(вид=name, начало=a, конец=b, часов=(b - a) / pd.Timedelta('1h'), пик=float(ratio[a:b - step].max()),
                                **{'главные датчики': top}))
            cols.update({(name, 'ошибка'): s.reindex(df.index), (name, 'сглаженная'): sm, (name, 'порог'): float(v['thr']),
                         (name, 'отношение'): ratio, (name, 'выше порога'): ratio > 1, (name, 'тревога'): flag})
        hourly = pd.DataFrame(cols, index=df.index)
        hourly.columns = pd.MultiIndex.from_tuples(hourly.columns, names=['вид', 'поле'])
        return Result(hourly=hourly, mode=mode, sensor_errors=errs, contrib_share=shares, expected=exp,
                      alarms=pd.DataFrame(eps, columns=['вид', 'начало', 'конец', 'часов', 'пик', 'главные датчики']),
                      missing=[c for c in self.needed() if c not in df.columns])


class Union:
    """Объединение моделей, которые ловят разное: отношение «сглаженная ошибка / свой порог» каждой модели;
    «хотя бы один» — наибольшее, «два согласны» — второе по величине; вклады — от ведущей модели."""

    def __init__(self, folder):
        self.cfg = read_json(folder, 'config.json')
        self.members = [(Model(os.path.join(folder, 'members', m['folder'])), m['view']) for m in self.cfg['members']]

    def score(self, df, prepared=False):
        df = df if prepared else prepare(df)
        cfg = self.cfg
        res = [(m.score(df, prepared=True), v, m) for m, v in self.members]
        R = np.column_stack([(r.hourly[(v, 'сглаженная')] / r.hourly[(v, 'порог')]).to_numpy(float) for r, v, _ in res])
        mode = res[0][0].mode
        running = mode.ne(M_STOP)
        step = pd.Timedelta(cfg['step'])
        cols_u = list(dict.fromkeys(c for r, v, _ in res if v in r.sensor_errors for c in r.sensor_errors[v].columns))
        cols, errs, shares, eps = {}, {}, {}, []
        for vv in cfg['views']:
            ok = np.isfinite(R).any(axis=1)
            Z = np.where(np.isfinite(R), R, -np.inf)
            comb = Z.max(axis=1) if vv['how'] == 'max' else np.sort(Z, axis=1)[:, -2]
            comb = pd.Series(np.where(ok, np.where(np.isfinite(comb), comb, 0.0), np.nan), index=df.index)
            lead = Z.argmax(axis=1)
            ratio = (comb / vv['thr']).where(running, 0.0)
            M = pd.DataFrame(np.nan, index=df.index, columns=cols_u)
            for j, (r, v, _) in enumerate(res):
                if v in r.sensor_errors:
                    rows = df.index[(lead == j) & ok]
                    part = r.sensor_errors[v].reindex(rows)
                    M.loc[rows, part.columns] = part.to_numpy()
            M = M[ok & running.to_numpy()]
            errs[vv['name']] = M
            shares[vv['name']] = M.div(M.sum(axis=1).replace(0, np.nan), axis=0) * 100
            flag = pd.Series(False, index=df.index)
            for a, b in alarm_episodes(ratio.where(mode.isin([M_RUN, M_PRESTOP]) & comb.notna()), cfg['persist_steps'], cfg['merge_steps'], step):
                flag[(df.index >= a) & (df.index < b)] = True
                eps.append(dict(вид=vv['name'], начало=a, конец=b, часов=(b - a) / pd.Timedelta('1h'), пик=float(ratio[a:b - step].max()),
                                **{'главные датчики': ', '.join(res[k][1] for k in pd.Series(lead[(df.index >= a) & (df.index < b)]).value_counts().index[:2])}))
            cols.update({(vv['name'], 'ошибка'): comb, (vv['name'], 'сглаженная'): comb, (vv['name'], 'порог'): float(vv['thr']),
                         (vv['name'], 'отношение'): ratio, (vv['name'], 'выше порога'): ratio > 1, (vv['name'], 'тревога'): flag,
                         (vv['name'], 'ведущая модель'): pd.Series(np.where(ok, np.array([v for _, v, _ in res])[lead], None), index=df.index)})
        hourly = pd.DataFrame(cols, index=df.index)
        hourly.columns = pd.MultiIndex.from_tuples(hourly.columns, names=['вид', 'поле'])
        return Result(hourly=hourly, mode=mode, sensor_errors=errs, contrib_share=shares, expected=None,
                      alarms=pd.DataFrame(eps, columns=['вид', 'начало', 'конец', 'часов', 'пик', 'главные датчики']),
                      missing=sorted({c for r, _, _ in res for c in r.missing}), members={v: r for r, v, _ in res})


class Bagging:
    """Бэггинг моделей на остатках: среднее отношений «ошибка / свой порог» (первая ступень общая, stage1/);
    вклады — среднее нормированных ошибок датчиков по моделям."""

    def __init__(self, folder):
        self.cfg = read_json(folder, 'config.json')
        self.stage1 = Model(os.path.join(folder, 'stage1'))
        self.members = [Model(os.path.join(folder, 'members', m), stage1=self.stage1) for m in self.cfg['members']]

    def score(self, df, prepared=False):
        df = df if prepared else prepare(df)
        cfg = self.cfg
        res = [m.score(df, prepared=True) for m in self.members]
        views = [m.cfg['views'][0]['name'] for m in self.members]
        R = np.column_stack([(r.hourly[(v, 'ошибка')] / r.hourly[(v, 'порог')]).to_numpy(float) for r, v in zip(res, views)])
        c = pd.Series(mean_finite(R, axis=1), index=df.index)
        ok = c.notna().to_numpy()
        mode = res[0].mode
        running = mode.ne(M_STOP)
        step = pd.Timedelta(cfg['step'])
        vv = cfg['views'][0]
        sm = rolling_mean(c, cfg['smooth_steps']).reindex(df.index)
        ratio = (sm / vv['thr']).where(running, 0.0)
        mats = [r.sensor_errors[v] for r, v in zip(res, views) if v in r.sensor_errors]
        idx = df.index[ok]
        M = None
        if mats:
            cols_u = list(dict.fromkeys(cc for m in mats for cc in m.columns))
            A = np.stack([m.reindex(index=idx, columns=cols_u).to_numpy(float) for m in mats])
            M = pd.DataFrame(mean_finite(A, axis=0), index=idx, columns=cols_u)
        flag = pd.Series(False, index=df.index)
        eps = []
        for a, b in alarm_episodes(ratio.where(mode.isin([M_RUN, M_PRESTOP]) & c.notna()), cfg['persist_steps'], cfg['merge_steps'], step):
            flag[(df.index >= a) & (df.index < b)] = True
            top = (', '.join(f'{k} ×{x:.1f}' for k, x in M.loc[(M.index >= a) & (M.index < b)].quantile(0.9).nlargest(3).items())
                   if M is not None and ((M.index >= a) & (M.index < b)).any() else '')
            eps.append(dict(вид=vv['name'], начало=a, конец=b, часов=(b - a) / pd.Timedelta('1h'), пик=float(ratio[a:b - step].max()),
                            **{'главные датчики': top}))
        name = vv['name']
        hourly = pd.DataFrame({(name, 'ошибка'): c, (name, 'сглаженная'): sm, (name, 'порог'): float(vv['thr']), (name, 'отношение'): ratio,
                               (name, 'выше порога'): ratio > 1, (name, 'тревога'): flag}, index=df.index)
        hourly.columns = pd.MultiIndex.from_tuples(hourly.columns, names=['вид', 'поле'])
        return Result(hourly=hourly, mode=mode, sensor_errors={name: M} if M is not None else {},
                      contrib_share={name: M.div(M.sum(axis=1).replace(0, np.nan), axis=0) * 100} if M is not None else {},
                      expected=None, alarms=pd.DataFrame(eps, columns=['вид', 'начало', 'конец', 'часов', 'пик', 'главные датчики']),
                      missing=sorted({c for r in res for c in r.missing}), members=dict(zip(views, res)))


def load(folder=None):
    """Загрузить модель из папки (по умолчанию — папка этого файла)."""
    folder = folder or os.path.dirname(os.path.abspath(__file__))
    kind = read_json(folder, 'config.json')['core']
    return {'union': Union, 'bagging': Bagging}.get(kind, Model)(folder)


CORES_ALL = {'analogs': AnalogCore, 'analogs_boost': AnalogBoostCore, 'boost': BoostCore, 'pca': PcaCore, 'mahalanobis': MahalCore,
             'iforest': IForestCore, 'autoencoder': AeCore, 'lstm': LstmCore, 'cond_net': CondNetCore, 'twins': TwinsCore,
             'rare': RareCore, 'kernel': KernelCore}
CORES = CORES_ALL                                   # в model.py — только ядра этой модели

# ============================================================================== ЧАСТЬ 2. ЭКСПОРТ

COMMON_CODE = [read_json, prepare, detect_mode, condition_frame, rolling_mean, slow, top_mean, windows, mean_finite, alarm_episodes,
               Scaler, Result]
CORE_CODE = {'analogs': [_nan_quantiles_sorted, AnalogCore], 'analogs_boost': [_nan_quantiles_sorted, AnalogCore, AnalogBoostCore],
             'boost': [BoostCore], 'pca': [PcaCore], 'mahalanobis': [MahalCore], 'iforest': [IForestCore], 'autoencoder': [AeCore],
             'lstm': [LstmCore], 'cond_net': [CondNetCore], 'twins': [TwinsCore], 'rare': [RareCore],
             'kernel': [robust_scale, Linear, AAKR, MSET, spd_solve, _chol_solve, KernelCore]}
DEPS = {'analogs': 'scikit-learn', 'analogs_boost': 'scikit-learn, joblib', 'boost': 'scikit-learn, joblib', 'iforest': 'scikit-learn, joblib',
        'autoencoder': 'tensorflow', 'lstm': 'tensorflow', 'cond_net': 'tensorflow'}
CORE_CLASS_NAME = {k: v.__name__ for k, v in CORES_ALL.items()}


def _cores_of(folder):
    """Какие ядра нужны модели в папке (с первой ступенью и вложенными моделями)."""
    cfg = read_json(folder, 'config.json')
    out = []
    if cfg['core'] in ('union', 'bagging'):
        for sub in ('stage1',):
            if os.path.isdir(os.path.join(folder, sub)):
                out += _cores_of(os.path.join(folder, sub))
        mdir = os.path.join(folder, 'members')
        for m in sorted(os.listdir(mdir)):
            out += _cores_of(os.path.join(mdir, m))
    else:
        out.append(cfg['core'])
        if os.path.isdir(os.path.join(folder, 'stage1')):
            out += _cores_of(os.path.join(folder, 'stage1'))
    return list(dict.fromkeys(out))


def write_code(folder, title):
    """model.py модели: только нужный ей код инференса (общие функции + её ядра + класс модели)."""
    cfg = read_json(folder, 'config.json')
    cores = _cores_of(folder)
    objs = list(COMMON_CODE)
    for c in cores:
        objs += [o for o in CORE_CODE[c] if o not in objs]
    objs.append(Model)
    top = {'union': Union, 'bagging': Bagging}.get(cfg['core'], Model)
    if top is not Model:
        objs.append(top)
    deps = ['numpy', 'pandas'] + sorted({d for c in cores for d in DEPS.get(c, '').split(', ') if d})
    head = f'''"""
Модель котла: {title}.
Код инференса этой папки (создан k2_deploy.py). Вход — готовая часовая таблица, как hourly.parquet от opc_to_wide.py:
столбец TimeValue (или время в индексе) и столбцы датчиков, как есть. Нужна история ≥ 4 суток до оцениваемых часов.

    m = load('путь/к/этой/папке')
    r = m.score(df)
    r.hourly         — таблица по часам; столбцы (вид, поле): ошибка, сглаженная, порог, отношение (сглаженная / порог),
                       выше порога, тревога (выше порога не меньше {cfg['persist_steps']} ч подряд, с учётом склейки)
    r.mode           — режим котла по часам: стоит / пуск / работа / перед остановом (оцениваются работа и перед остановом)
    r.sensor_errors  — {{вид: таблица час × датчик}}: ошибка датчика / его обычная ошибка (1 — граница нормы датчика)
    r.contrib_share  — {{вид: таблица час × датчик}}: доля датчика в сумме нормированных ошибок, %
    r.expected       — ожидание, разброс и факт по датчикам (у analogs, AAKR, MSET и регрессии), иначе None
    r.alarms         — эпизоды тревог: вид, начало, конец, часов, пик отношения, главные датчики
    r.missing        — каких нужных датчиков нет во входной таблице (они считаются пропусками)

Зависимости: {', '.join(deps)}.
"""
import os
import json

import numpy as np
import pandas as pd

TIME_COL = 'TimeValue'
M_STOP, M_START, M_RUN, M_PRESTOP = 'стоит', 'пуск', 'работа', 'перед остановом'
COND_LOAD_CHANGE, COND_SINCE_START, COND_MILLS = {COND_LOAD_CHANGE!r}, {COND_SINCE_START!r}, {COND_MILLS!r}

'''
    body = '\n\n'.join(inspect.getsource(o).rstrip() for o in objs)
    cores_map = 'CORES = {' + ', '.join(f"'{c}': {CORE_CLASS_NAME[c]}" for c in cores) + '}'
    load_src = (f'def load(folder=None):\n    """Загрузить модель из папки (по умолчанию — папка этого файла)."""\n'
                f'    return {top.__name__}(folder or os.path.dirname(os.path.abspath(__file__)))\n')
    tail = '\n\n\n' + cores_map + '\n\n\n' + load_src
    with open(os.path.join(folder, 'model.py'), 'w', encoding='utf-8') as f:
        f.write(head + body + tail)
    with open(os.path.join(folder, 'example.py'), 'w', encoding='utf-8') as f:
        f.write(EXAMPLE.format(title=title))


EXAMPLE = '''"""Пример: {title}. Оценить последние 7 суток часовой таблицы (плюс 4 суток истории)."""
import os
import importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(folder):
    """Загрузка модели по пути к папке (так можно держать в проекте несколько моделей сразу)."""
    spec = importlib.util.spec_from_file_location('model_' + os.path.basename(os.path.abspath(folder)), os.path.join(folder, 'model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load(folder)


if __name__ == '__main__':
    m = load_model(HERE)
    df = pd.read_parquet('hourly.parquet')             # часовая таблица: TimeValue и датчики
    r = m.score(df.tail(24 * 11))
    print(r.hourly.xs('отношение', axis=1, level='поле').tail(10))
    print(r.alarms)
    if r.missing:
        print('нет датчиков:', r.missing[:10])
'''


def _save_json(folder, name, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return str(o)
    with open(os.path.join(folder, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=conv)


def _write_scaler(folder, prep):
    _save_json(folder, 'scaler.json', dict(num=list(prep.num), med=np.asarray(prep.med, float), sd=np.asarray(prep.sd, float),
                                           bools=[[c, m] for c, m in prep.bools]))


def _write_core(folder, core, det, prep):
    """Параметры ядра: params.npz (числа), *.joblib (sklearn), net.keras (нейросети), *.json (правила)."""
    import joblib
    n = len(prep.num) if prep is not None else 0
    nb = len(prep.bools) if prep is not None else 0
    if core in ('analogs', 'analogs_boost'):
        am = det.am
        if hasattr(det, 'cond'):                              # analogs: условия — выбранные столбцы
            ci = [prep.num.index(c) for c in det.cond if c in prep.num]
            ti = [j for j in range(n) if j not in set(ci)]
            r_idx = ci + list(range(n, n + nb))
        else:                                                 # Аналоги: нагрузка + состояния механизмов
            r_idx = ([prep.num.index(det.load_col)] if det.load_col in prep.num else []) + list(range(n, n + nb))
            ti = list(range(n))
        np.savez_compressed(os.path.join(folder, 'params.npz'), r_idx=np.array(r_idx), t_idx=np.array(ti),
                            r_med=am.r_med[am.rcols].to_numpy(float), r_scale=am.r_scale[am.rcols].to_numpy(float),
                            lidx=np.array([am.rcols.index(c) for c in am.lcols]), didx=np.array(am.didx), B=am.B,
                            Xtr=am.Xtr.astype('float32'), Rtr=am.Rtr, floor=am.floor, kk=am.kk, min_valid=am.min_valid, n=n)
        if core == 'analogs_boost':
            joblib.dump(dict(models=det.corr, dist_fill=det.dist_fill), os.path.join(folder, 'corr.joblib'), compress=3)
    elif core == 'boost':
        ci = [prep.num.index(c) for c in det.cond if c in prep.num]
        joblib.dump(dict(in_idx=np.array(ci + list(range(n, n + nb))), t_idx=np.array(det.ti), n=n, floor=det.floor, models=det.models),
                    os.path.join(folder, 'boost.joblib'), compress=3)
    elif core == 'pca':
        np.savez_compressed(os.path.join(folder, 'params.npz'), mean=det.pca.mean_, comps=det.pca.components_, n=det.n)
    elif core == 'mahalanobis':
        np.savez_compressed(os.path.join(folder, 'params.npz'), mu=det.mu, P=det.P)
    elif core == 'iforest':
        joblib.dump(det.m, os.path.join(folder, 'iforest.joblib'), compress=3)
    elif core in ('autoencoder', 'lstm'):
        det.net.save(os.path.join(folder, 'net.keras'))
    elif core == 'cond_net':
        ci = [prep.num.index(c) for c in det.cond if c in prep.num]
        det.net.save(os.path.join(folder, 'net.keras'))
        np.savez_compressed(os.path.join(folder, 'params.npz'), in_idx=np.array(ci + list(range(n, n + nb))), t_idx=np.array(det.ti), n=n)
    elif core == 'twins':
        _save_json(folder, 'twins.json', dict(n=det.n, models=[dict(j=int(j), others=[int(o) for o in others], k=float(k_), b=float(b_),
                                                                     m=float(m_), s=float(s_)) for j, others, k_, b_, m_, s_ in det.models]))
    elif core == 'kernel':                                    # AAKR / MSET (раздел 4.7 ноутбука 02)
        m = det.m
        ci = [prep.num.index(c) for c in det.cond if c in prep.num]
        ti = [j for j in range(n) if j not in set(ci)]
        kind = 'mset' if type(m).__name__ == 'MSET' else 'aakr'   # по имени: ноутбук перезагружает kernel_models (reload)
        par = dict(kind=kind, mode=getattr(m, 'mode', 'per_sensor'), h=float(m.h),
                   robust=bool(m.robust), robust_thr=float(m.robust_thr), cond_weight=float(m.cond_weight),
                   state_weight=float(m.state_weight), xm=m.xm, xs=m.xs, cm=m.cm, cs=m.cs, nb=m.nb, Mc=m.Mc, Mx=m.Mx, p=int(m.p),
                   c_idx=np.array(ci, int), t_idx=np.array(ti, int), n=n, local=int(getattr(m, 'local', 0)), lam=float(getattr(m, 'lam', 0.0)))
        if getattr(m, 'lin', None) is not None:
            par.update(lin_beta=m.lin.beta, lin_cm=m.lin.cm, lin_cs=m.lin.cs)
        np.savez_compressed(os.path.join(folder, 'params.npz'), **par)
    elif core == 'rare':
        _save_json(folder, 'rare.json', dict(cols=det.cols, held_steps=det.held_steps, n=det.n,
                                             freq={c: {str(k): float(v) for k, v in det.freq[c].items()} for c in det.cols}))
    else:
        raise ValueError(core)


def export_model(folder, title, core, det, prep, views, common, inp='sensors', cond=(), qc=None, stage1=None, quality=None, extra=None):
    """
    Одна модель -> папка: config.json, sensors.csv, scaler.json, параметры ядра, model.py, example.py.
    views — [{name, top, slow_h, q (обычные ошибки датчиков или одно число), thr}]; common — общие параметры
    (датчик нагрузки и порог, пуск, сглаживание, устойчивость, склейка, шаг, наличие датчиков, дата обучения).
    stage1 — (prep, det) первой ступени для моделей на остатках (analogs) или (prep, det, ядро, название, условия),
    например регрессия: (prep, det, 'boost', 'первая ступень: Регрессия', условия). Возвращает путь.
    """
    import shutil
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    if prep is not None:
        _write_scaler(folder, prep)
    _write_core(folder, core, det, prep)
    one = core in ('mahalanobis', 'iforest', 'rare')          # одно число на час
    if core == 'rare':
        names = list(det.cols)
    elif core == 'mahalanobis':
        names = list(prep.names)
    else:
        names = list(prep.num)
    sens = pd.DataFrame({'датчик': names})
    sens['роль'] = ['условие' if c in set(cond) else ('состояние механизма' if ' ≠ ' in c and core == 'mahalanobis' else
                                                     ('текстовый' if core == 'rare' else 'проверяется')) for c in names]
    vcfg = []
    for v in views:
        q = np.atleast_1d(np.asarray(v['q'], float))
        d = dict(name=v['name'], top=int(v['top']), slow_h=int(v.get('slow_h', 0)), thr=float(v['thr']))
        if one or len(q) == 1:
            d['q'] = float(q[0])
        else:
            sens[f'обычная ошибка: {v["name"]}'] = q
        if quality and v['name'] in quality:
            d['качество на проверке'] = quality[v['name']]
        vcfg.append(d)
    if qc is not None:
        sens['обычный вклад'] = np.asarray(qc, float)
    sens.to_csv(os.path.join(folder, 'sensors.csv'), index=False, encoding='utf-8-sig')
    cfg = dict(title=title, core=core, input=inp, views=vcfg, n=len(prep.num) if prep is not None else 0, **common)
    if core == 'rare':
        cfg['use_presence'] = False
    if inp not in ('conditions', 'residuals'):
        cfg.pop('cond_spec', None)
    blank = [c for c in (getattr(det, 'blank', None) or []) if prep is not None and c in prep.num]
    if blank:                                                 # мало данных при обучении — модель эти датчики не проверяет
        cfg['blank_cols'] = blank
    cfg.update(extra or {})
    _save_json(folder, 'config.json', cfg)
    if stage1 is not None:                                    # (prep, det[, ядро, название, условия]) — по умолчанию analogs
        p1, d1, core1, title1, cond1 = (tuple(stage1) + (None,) * 3)[:5]
        export_model(os.path.join(folder, 'stage1'), title1 or 'первая ступень: analogs', core1 or 'analogs', d1, p1, [], common,
                     inp='conditions', cond=cond if cond1 is None else cond1)
    write_code(folder, title)
    return folder


def _member_name(src):
    """Имя вложенной модели: папка метода (у итоговых моделей последняя часть пути — «final»)."""
    name = os.path.basename(os.path.normpath(src))
    return os.path.basename(os.path.dirname(os.path.normpath(src))) if name == 'final' else name


def export_union(folder, title, members, views, common):
    """Объединение: members — [(папка уже экспортированной модели, вид)]; модели копируются в members/."""
    import shutil
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(os.path.join(folder, 'members'))
    mcfg = []
    for src, view in members:
        name = _member_name(src)
        shutil.copytree(src, os.path.join(folder, 'members', name))
        mcfg.append(dict(folder=name, view=view))
    cfg = dict(title=title, core='union', input='members', members=mcfg, views=views,
               **{k: common[k] for k in ('step', 'persist_steps', 'merge_steps', 'smooth_steps', 'trained_until')})
    _save_json(folder, 'config.json', cfg)
    write_code(folder, title)
    return folder


def export_bagging(folder, title, members, stage1_src, view, common):
    """Бэггинг моделей на остатках: общая первая ступень (stage1/) и вторые ступени моделей (members/, без своих stage1)."""
    import shutil
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(os.path.join(folder, 'members'))
    shutil.copytree(stage1_src, os.path.join(folder, 'stage1'))
    names = []
    for src in members:
        name = _member_name(src)
        shutil.copytree(src, os.path.join(folder, 'members', name), ignore=shutil.ignore_patterns('stage1', 'model.py', 'example.py'))
        names.append(name)
    cfg = dict(title=title, core='bagging', input='members', members=names, views=[view],
               **{k: common[k] for k in ('step', 'persist_steps', 'merge_steps', 'smooth_steps', 'trained_until')})
    _save_json(folder, 'config.json', cfg)
    write_code(folder, title)
    return folder


def load_folder(folder):
    """Загрузить модель через её собственный model.py (как это будет в проекте)."""
    import sys
    import importlib.util
    tag = '_'.join(os.path.abspath(folder).replace(os.sep, '/').split('/')[-3:])   # агрегат_метод_final — имя модуля
    spec = importlib.util.spec_from_file_location('model_' + ''.join(ch if ch.isalnum() else '_' for ch in tag),
                                                  os.path.join(folder, 'model.py'))
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True       # не оставлять __pycache__ в папке модели
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod.load(folder)


README = '''# Модели котлов для внедрения

Структура: `<агрегат>/<метод>/final/` — итоговая модель (учится на всех данных, для новых часов), `<агрегат>/<метод>/<ГГГГ-ММ>/` — модели месяцев (каждая обучена только на данных до начала месяца — для прошлого). Так их читает дашборд.

Каждая папка — отдельная модель, самодостаточная: её можно скопировать в проект целиком.

| Файл | Что в нём |
|---|---|
| `model.py` | код инференса именно этой модели (`load(папка)` -> модель, `модель.score(df)` -> результат) |
| `config.json` | метод, виды (сколько датчиков, медленный сдвиг), пороги, сглаживание, режим котла, дата обучения, качество на проверке |
| `sensors.csv` | датчики модели: роль (проверяется / условие / состояние механизма), обычная ошибка для каждого вида |
| `scaler.json` | нормировка входа: медиана и масштаб каждого числового датчика; текстовые датчики и их обычное состояние |
| `params.npz`, `*.joblib`, `*.json`, `net.keras` | параметры модели |
| `stage1/` | первая ступень (analogs или регрессия) у моделей на остатках |
| `members/` | вложенные модели у объединения и бэггинга |
| `example.py` | пример запуска |

**Вход** — готовая часовая таблица, как `hourly.parquet` от `opc_to_wide.py`, как есть: столбец `TimeValue` (или время в индексе)
и столбцы датчиков. Нужна история не меньше 4 суток до оцениваемых часов (пуск, сглаживание, окна). Недостающие датчики
считаются пропусками и перечислены в `r.missing`.

**Выход** `r = модель.score(df)` — таблицы pandas:
- `r.hourly` — по часам, столбцы (вид, поле): ошибка, сглаженная, порог, отношение, выше порога, тревога;
- `r.mode` — режим котла; модель оценивает только часы работы (не пуск и не останов);
- `r.sensor_errors[вид]` — час × датчик: ошибка датчика / его обычная ошибка (вклад датчика; 1 — граница нормы);
- `r.contrib_share[вид]` — доля датчика в сумме, %;
- `r.expected` — ожидание, разброс и факт по датчикам (analogs, AAKR, MSET, регрессия);
- `r.alarms` — эпизоды тревог с главными датчиками;
- `r.missing` — каких датчиков не хватило.

**Несколько моделей в одном проекте** — загружайте по пути к папке (см. `load_model` в `example.py`).
'''
