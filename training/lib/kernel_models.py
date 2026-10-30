"""
Модели AAKR и MSET: ожидание каждого датчика по похожим часам нормальной работы. Два режима: mode='per_sensor' (по
умолчанию) — у каждого датчика свои входы: условия работы, состояния механизмов и связанные датчики, без него самого;
mode='all' — классический: одно расстояние по всем датчикам сразу, сам датчик тоже входит (ноутбук 02 — «… (все датчики)»).
В monitoring_v3 related используется mode='all', linear=False, robust=False; у MSET full_memory=True,
lam=0 включает псевдообратную матрицу по всей сохранённой памяти. Для этого пути также нужен scipy.
Используются ноутбуком 02 (раздел 4.7, класс KernelDet) и экспортом k2_deploy (ядро «kernel»: код инференса копируется
в model.py папки модели, поэтому папка модели от этого файла не зависит). Нужны только numpy и pandas.

AAKR — автоассоциативная ядерная регрессия.

Ожидание датчика = среднее значений этого датчика в векторах памяти (нормальные часы), взвешенное ядром Гаусса по
расстоянию:  x̂_j = Σ K(d_i) m_ij / Σ K(d_i),  K(d) = exp(−d²/h²),  d² — средний квадрат разницы по входам.

Входы (режим per_sensor, по умолчанию): условия работы, состояния механизмов и до k самых связанных с датчиком других
датчиков — сам датчик в своём расстоянии не участвует, поэтому неисправный датчик не «подтягивает» ожидание к себе.
Режим all — классический: расстояние по всем датчикам сразу (одно на час), сам датчик тоже входит.
Устойчивость: второй проход без входов, которые сами сильно ушли от ожидания (неисправность одного датчика не сдвигает
ожидания соседей). Пропуски не заменяются: расстояние — по имеющимся входам.

MSET — Multivariate State Estimation Technique (с локальной памятью).

Память D — векторы нормы: равномерно по режимам (как у AAKR) плюс векторы с крайними значениями каждого датчика и
условия (0.5% и 99.5% — классический отбор «мин-макс», чтобы края нормы были в памяти). Для часа x и датчика j берутся
K ближайших векторов памяти D_k (по тем же входам, что у AAKR: условия, состояния механизмов, связанные датчики — без
самого датчика), и ожидание считается через нелинейный оператор сходства ⊗ с регуляризацией λ:
    x̂_j = D_k,j · w / Σw,   w = (D_kᵀ ⊗ D_k + λI)⁻¹ · (D_kᵀ ⊗ x),   s(a, b) = exp(−mean((a − b)²) / h²)
Отличие от AAKR: веса учитывают сходство векторов памяти между собой (близкие дубли не набирают лишний вес, ожидание
может выходить за выпуклую оболочку памяти). Локальная память вместо одной матрицы на всю память: (Dᵀ⊗D)⁻¹ по сотням
векторов в многомерном пространстве плохо обусловлена и даёт медленно плывущий остаток (долгие ложные тревоги).
Пропущенные и исключённые (устойчивый проход) входы не участвуют ни в x, ни в D_kᵀ ⊗ D_k — матрица считается для каждого
часа по его входам. Если система вырождена — ядерные веса, как у AAKR.
"""
import numpy as np


def robust_scale(A):
    """Медиана и разброс (межквартильный размах / 1.349) по столбцам; нулевой разброс -> std, иначе 1."""
    A = np.asarray(A, float)
    if A.ndim == 2 and A.shape[1] == 0:
        return np.zeros(0), np.ones(0)
    if A.shape[0] == 0:
        return np.zeros(A.shape[1]), np.ones(A.shape[1])
    q25, med, q75 = np.nanpercentile(A, [25, 50, 75], axis=0)
    sc = (q75 - q25) / 1.349
    std = np.nanstd(A, axis=0)
    sc = np.where(sc > 1e-9, sc, np.where(std > 1e-9, std, 1.0))
    return np.nan_to_num(med), np.nan_to_num(sc, nan=1.0)


class Linear:
    """Линейная часть: каждый датчик ≈ гребневая регрессия на условия работы и состояния механизмов. Ядерная модель
    работает с остатком — так она не смещается на краях и в редких сочетаниях режима (сезон × нагрузка)."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def _F(self, C, B):
        Cn = np.nan_to_num((np.asarray(C, float) - self.cm) / self.cs)
        return np.hstack([np.ones((len(Cn), 1)), Cn, np.asarray(B, float)])

    def fit(self, X, C, B):
        X = np.asarray(X, float)
        self.cm, self.cs = robust_scale(C)
        F = self._F(C, B)
        reg = self.lam * np.eye(F.shape[1])
        reg[0, 0] = 0
        self.beta = np.zeros((F.shape[1], X.shape[1]))
        ok = ~np.isnan(X)
        full = ok.all(axis=0)
        if full.any():
            self.beta[:, full] = np.linalg.solve(F.T @ F + reg, F.T @ X[:, full])
        for j in np.flatnonzero(~full):
            m = ok[:, j]
            if m.sum() > 3 * F.shape[1]:
                Fm = F[m]
                self.beta[:, j] = np.linalg.solve(Fm.T @ Fm + reg, Fm.T @ X[m, j])
            elif m.any():
                self.beta[0, j] = np.nanmedian(X[:, j])
        return self

    def predict(self, C, B):
        return self._F(C, B) @ self.beta


def strata(load, B, n_bins=10):
    """Номер «режима» каждой строки: квантиль нагрузки × сочетание состояний механизмов."""
    load = np.asarray(load, float)
    ok = np.isfinite(load)
    edges = np.unique(np.nanpercentile(load[ok], np.linspace(0, 100, n_bins + 1))) if ok.any() else np.array([0.0])
    lb = np.clip(np.searchsorted(edges, np.nan_to_num(load), side='right') - 1, 0, max(len(edges) - 2, 0))
    if B.shape[1]:
        _, pat = np.unique((B > 0.5).astype(np.int8), axis=0, return_inverse=True)
        pat = np.asarray(pat).ravel()
    else:
        pat = np.zeros(len(load), int)
    return lb * 10_000 + pat


def stratified_rows(cells, n, rng, min_per_cell=5):
    """n строк так, чтобы каждый режим был представлен (редкие — не меньше min_per_cell), остальное — пропорционально."""
    m = len(cells)
    if m <= n:
        return np.arange(m)
    uniq, inv, cnt = np.unique(cells, return_inverse=True, return_counts=True)
    take = np.minimum(cnt, np.maximum(min_per_cell, np.floor(cnt / m * n).astype(int)))
    while take.sum() > n:                              # слишком много редких режимов — урезаем самые крупные
        take[np.argmax(take)] -= 1
    rows = []
    for k in range(len(uniq)):
        idx = np.flatnonzero(inv == k)
        rows.append(np.sort(rng.choice(idx, take[k], replace=False)))
    return np.sort(np.concatenate(rows))


def find_neighbors(Xn, k=10, min_corr=0.3, max_rows=4000, seed=0):
    """Для каждого датчика — до k самых связанных с ним других датчиков (|корреляция| ≥ min_corr) по обучающим данным.
    Возвращает массив p × k (−1 — пусто)."""
    import pandas as pd
    n, p = Xn.shape
    rng = np.random.default_rng(seed)
    rows = np.arange(n) if n <= max_rows else np.sort(rng.choice(n, max_rows, replace=False))
    C = pd.DataFrame(Xn[rows]).corr(min_periods=100).abs().to_numpy(copy=True)
    np.fill_diagonal(C, -1)
    C = np.nan_to_num(C, nan=-1)
    out = np.full((p, k), -1, dtype=np.int32)
    for j in range(p):
        order = np.argsort(-C[j])[:k]
        good = order[C[j, order] >= min_corr]
        out[j, :len(good)] = good
    return out


class AAKR:
    name = 'AAKR'

    def __init__(self, memory=1500, neighbors=10, h=0.7, h_grid=(0.4, 0.7, 1.0, 1.5), cond_weight=1.0, state_weight=2.0,
                 mode='per_sensor', robust=True, robust_thr=3.0, linear=True, seed=0):
        self.memory, self.k, self.h, self.h_grid = memory, neighbors, h, tuple(h_grid)
        self.linear = linear
        self.cond_weight, self.state_weight, self.mode = cond_weight, state_weight, mode
        self.robust, self.robust_thr, self.seed = robust, robust_thr, seed

    # ------------------------------------------------------------------ обучение
    def _cond(self, C, B):
        Cn = (np.asarray(C, float) - self.cm) / self.cs * self.cond_weight
        return np.hstack([Cn, np.asarray(B, float) * self.state_weight]) if B.shape[1] else Cn

    def fit(self, X, C, B, nb=None):
        X, C, B = np.asarray(X, float), np.asarray(C, float), np.asarray(B, float)
        self.lin = Linear().fit(X, C, B) if self.linear else None
        if self.lin is not None:
            X = X - self.lin.predict(C, B)
        self.xm, self.xs = robust_scale(X)
        self.cm, self.cs = robust_scale(C)
        Xn = (X - self.xm) / self.xs
        self.nb = nb if nb is not None else (np.empty((X.shape[1],0),dtype=np.int32) if self.mode=='all' else find_neighbors(Xn,self.k,seed=self.seed))
        rng = np.random.default_rng(self.seed)
        rows = self._memory_rows(Xn, C, B, rng)
        self.Mc = self._cond(C[rows], B[rows]).astype(np.float32)
        self.Mx = Xn[rows].astype(np.float32)
        self.p = X.shape[1]
        return self

    def _memory_rows(self, Xn, C, B, rng):
        """Векторы памяти: равномерно по режимам (квантиль нагрузки × сочетание состояний). Нагрузка — первое условие;
        у классического варианта (без отдельных условий) — столбец датчиков strat_col (датчик работы котла)."""
        sc = getattr(self, 'strat_col', None)
        load = Xn[:, sc] if sc is not None else (C[:, 0] if C.shape[1] else np.zeros(len(Xn)))
        return stratified_rows(strata(load, B), self.memory, rng)

    # ------------------------------------------------------------------ оценка
    def predict(self, X, C, B, cols=None, h=None, chunk=32):
        """Ожидание (исходные единицы) и разброс (масштаб датчика) для строк; cols — только эти датчики."""
        h = self.h if h is None else h
        X, C, B = np.asarray(X, float), np.asarray(C, float), np.asarray(B, float)
        n, p = X.shape
        cols = np.arange(p) if cols is None else np.asarray(cols)
        E = np.full((n, p), np.nan)
        lin = self.lin.predict(C, B) if getattr(self, 'lin', None) is not None else 0.0
        Xn = ((X - lin - self.xm) / self.xs).astype(np.float32)
        A = self._cond(C, B).astype(np.float32)
        q = A.shape[1]
        for a in range(0, n, chunk):
            sl = slice(a, min(a + chunk, n))
            xq, aq = Xn[sl], A[sl]
            dc = ((aq[:, None, :] - self.Mc[None]) ** 2).sum(-1)                      # условия: строки × память
            if getattr(self,'full_memory',False):
                dc=((aq.astype(float)[:,None,:]-self.Mc.astype(float)[None])**2).sum(-1)
            T = (xq[:, None, :] - self.Mx[None]) ** 2                                   # строки × память × датчики
            V = ~np.isnan(T)
            T = np.nan_to_num(T, copy=False)
            Eb = self._estimate(xq, dc, T, V, q, cols, h, flagged=None)
            if self.robust:
                F = np.abs(xq - Eb) > self.robust_thr                                    # входы, сами ушедшие от ожидания
                F &= ~np.isnan(xq)
                if F.any():
                    Eb2 = self._estimate(xq, dc, T, V, q, cols, h, flagged=F)
                    Eb = np.where(np.isnan(Eb2), Eb, Eb2)
            E[sl] = Eb * self.xs + self.xm
        E = E + lin
        S = np.broadcast_to(self.xs, E.shape).copy()
        return E, S

    def _combine(self, d2, Mt0, ok, h, xq, cols, flagged):
        """Ожидания по расстояниям до памяти (строки × память × датчики): ядерное среднее."""
        d2 = d2 - d2.min(axis=1, keepdims=True)                                         # устойчивость exp
        W = np.exp(-d2 / (h * h))
        num = (W * Mt0[None]).sum(1)
        den = (W * ok[None]).sum(1)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

    def _weights(self, dc, ds, cnt, q, h):
        d2 = (dc + ds) / np.maximum(q + cnt, 1)
        d2 = d2 - d2.min(axis=1, keepdims=True)                                         # устойчивость exp
        return np.exp(-d2 / (h * h))

    def _wmean(self, w, m):
        ok = ~np.isnan(m)
        sw = w[:, ok].sum(1)
        return np.where(sw > 0, w[:, ok] @ m[ok] / np.where(sw > 0, sw, 1), np.nan)

    def _adj(self):
        """A[n, j] = 1, если датчик n — вход датчика j (матрица входов для расчёта всех датчиков одним умножением)."""
        if getattr(self, '_A', None) is None:
            A = np.zeros((self.p, self.p), np.float32)
            for j in range(self.p):
                idx = self.nb[j][self.nb[j] >= 0]
                A[idx, j] = 1.0
            self._A = A
        return self._A

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_A', None)
        d.pop('_full_factor_cache',None)
        return d

    def _estimate(self, xq, dc, T, V, q, cols, h, flagged):
        nrow, M, p = T.shape
        Eb = np.full((nrow, self.p), np.nan, dtype=np.float32)
        Mt = self.Mx[:, cols]
        ok = ~np.isnan(Mt)
        Mt0 = np.where(ok, Mt, 0.0).astype(np.float32)
        if self.mode == 'all':
            Vf = V if flagged is None else V & ~flagged[:, None, :]
            w = self._weights(dc, (T * Vf).sum(-1), Vf.sum(-1), q, h)                  # строки × память
            num, den = w @ Mt0, w @ ok.astype(np.float32)
            Eb[:, cols] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            return Eb
        if self.mode == 'all_other':
            # Sum once and subtract the target's own coordinate instead of
            # multiplying by a dense off-diagonal adjacency matrix.
            Vf=V if flagged is None else V & ~flagged[:,None,:]
            weighted=T*Vf
            ds=weighted.sum(-1,keepdims=True,dtype=np.float64)-weighted[:,:,cols]
            cnt=Vf.sum(-1,keepdims=True)-Vf[:,:,cols].astype(np.int32)
            d2=(dc[:,:,None]+ds)/np.maximum(q+cnt,1)
            Eb[:,cols]=self._combine(d2,Mt0,ok,h,xq,cols,flagged)
            return Eb
        A = self._adj()[:, cols]                                                       # датчики × оцениваемые
        if flagged is None:
            DS = (T.reshape(-1, p) @ A).reshape(nrow, M, -1)
            CNT = (V.reshape(-1, p).astype(np.float32) @ A).reshape(nrow, M, -1)
            rows = np.arange(nrow)
        else:
            need = (flagged.astype(np.float32) @ A) > 0                                  # у этого датчика в часе ушёл вход
            rows = np.flatnonzero(need.any(axis=1))
            if not len(rows):
                return Eb
            DS = np.empty((len(rows), M, len(cols)), np.float32)
            CNT = np.empty_like(DS)
            for r_, b in enumerate(rows):
                Ab = A * (~flagged[b])[:, None]
                DS[r_], CNT[r_] = T[b] @ Ab, V[b].astype(np.float32) @ Ab
        d2 = (dc[rows][:, :, None] + DS) / np.maximum(q + CNT, 1)                      # строки × память × датчики
        Eb[np.ix_(rows, cols)] = self._combine(d2, Mt0, ok, h, xq[rows], cols, None if flagged is None else flagged[rows])
        if flagged is not None:                                                        # пересчитаны только нужные пары
            keep = np.zeros((nrow, self.p), bool)
            keep[np.ix_(rows, cols)] = need[rows]
            Eb[~keep] = np.nan
        return Eb


class MSET(AAKR):
    name = 'MSET'

    def __init__(self, memory=1500, local=20, neighbors=10, h=0.7, h_grid=(0.4, 0.7, 1.0), lam=1.0, cond_weight=1.0,
                 state_weight=2.0, mode='per_sensor', robust=True, robust_thr=3.0, linear=True, seed=0):
        super().__init__(memory=memory, neighbors=neighbors, h=h, h_grid=h_grid, cond_weight=cond_weight,
                         state_weight=state_weight, mode=mode, robust=robust, robust_thr=robust_thr, linear=linear,
                         seed=seed)
        self.local, self.lam = local, lam

    def _estimate(self, xq, dc, T, V, q, cols, h, flagged):
        """mode='all' — классический MSET: K ближайших векторов памяти по одному расстоянию по всем датчикам (сам датчик
        тоже входит) и веса с учётом их сходства между собой; иначе — по датчикам (как AAKR, см. _local)."""
        if self.mode != 'all':
            return super()._estimate(xq, dc, T, V, q, cols, h, flagged)
        if getattr(self,'full_memory',False):
            return self._full_estimate(xq,dc,T,V,q,cols,h,flagged)
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            nrow, M, p = T.shape
            Vf = V if flagged is None else V & ~flagged[:, None, :]
            d2 = (dc + (T * Vf).sum(-1)) / np.maximum(q + Vf.sum(-1), 1)               # строки × память
            K = min(self.local, M)
            kk = np.argpartition(d2, K - 1, axis=1)[:, :K]                               # строки × K
            dk = np.take_along_axis(d2, kk, axis=1)
            s = np.exp(-(dk - dk.min(1, keepdims=True)) / (h * h)).astype(np.float32)   # сходство с часом
            Ck = self.Mc[kk]                                                             # строки × K × условия
            Xk = self.Mx[kk]                                                             # строки × K × датчики
            dist = ((Ck[:, :, None] - Ck[:, None]) ** 2).sum(-1)
            Tk = (Xk[:, :, None] - Xk[:, None]) ** 2
            Vk = ~np.isnan(Tk)
            dist = dist + np.where(Vk, Tk, 0.0).sum(-1)
            G = np.exp(-dist / np.maximum(q + Vk.sum(-1), 1) / (h * h)).astype(np.float32)   # сходство векторов между собой
            G += self.lam * np.eye(K, dtype=np.float32)
            w = spd_solve(G, s)
            sw = w.sum(-1)
            bad = ~np.isfinite(sw) | (np.abs(sw) < 1e-6)                                # вырождено — ядерные веса
            w = np.where(bad[:, None], s, w)
            Mt = Xk[:, :, cols]
            ok = ~np.isnan(Mt)
            num = (w[:, :, None] * np.where(ok, Mt, 0.0)).sum(1)
            den = (w[:, :, None] * ok).sum(1)
            Eb = np.full((nrow, self.p), np.nan, dtype=np.float32)
            Eb[:, cols] = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), np.nan)
            return Eb

    def _full_estimate(self,xq,dc,T,V,q,cols,h,flagged):
        """Classical MSET: D @ pinv(D.T ⊗ D) @ (D.T ⊗ x), no ridge or imputation."""
        from scipy.linalg import pinvh
        if self.lam!=0 or flagged is not None:raise ValueError('Classical MSET requires lam=0 and robust=False')
        if not (np.isfinite(self.Mx).all() and np.isfinite(self.Mc).all() and np.isfinite(xq).all()):
            raise ValueError('Classical MSET requires complete dictionary and query vectors')
        token=float(h)
        cache=self.__dict__.get('_full_factor_cache')
        if cache is None or cache[0]!=token:
            z=np.hstack([self.Mc,self.Mx]).astype(float)
            squares=(z*z).sum(-1)
            distance=np.maximum(squares[:,None]+squares[None,:]-2*(z@z.T),0.)/z.shape[1]
            gram=np.exp(-distance/(h*h))
            inverse,rank=pinvh(gram,rtol=1e-10,return_rank=True,check_finite=True)
            self.dictionary_rank=int(rank)
            self._full_factor_cache=(token,inverse)
        inverse=self._full_factor_cache[1]
        ds=((xq.astype(float)[:,None,:]-self.Mx.astype(float)[None])**2).sum(-1)
        rhs=np.exp(-(dc+ds)/(q+self.p)/(h*h))
        weights=rhs@inverse
        if not np.isfinite(weights).all():raise ValueError('Nonfinite full-memory MSET weights')
        out=np.full((len(xq),self.p),np.nan,dtype=np.float32)
        out[:,cols]=weights@self.Mx[:,cols]
        return out

    def _memory_rows(self, Xn, C, B, rng):
        base = super()._memory_rows(Xn, C, B, rng)
        ext = []
        for col in list(Xn.T) + list(self._cond(C, B).T):          # мин-макс: крайние значения каждого датчика и условия
            ok = np.flatnonzero(~np.isnan(col))
            if len(ok) < 20:
                continue
            lo, hi = np.percentile(col[ok], [0.5, 99.5])
            ext += [ok[np.argmin(np.abs(col[ok] - lo))], ok[np.argmin(np.abs(col[ok] - hi))]]
        return np.unique(np.r_[base, np.array(ext, int)])

    def _combine(self, d2, Mt0, ok, h, xq, cols, flagged):
        with np.errstate(invalid='ignore', over='ignore'):           # вырожденные случаи ловятся по сумме весов
            return self._local(d2, Mt0, ok, h, xq, cols, flagged)

    def _local(self, d2, Mt0, ok, h, xq, cols, flagged, block=48):
        nrow, M, nc = d2.shape
        K = min(self.local, M)
        # All-other inputs can be hundreds of channels. Avoid allocating a
        # rows x K x K x 48 x channels tensor (several GB on real boilers).
        block=min(block,max(1,2_000_000//max(1,nrow*K*K*self.nb.shape[1])))
        cols = np.asarray(cols)
        d2 = np.where(ok.T[None], d2.transpose(0, 2, 1), np.inf)    # строки × датчики × память; нет значения — не брать
        kk = np.argpartition(d2, K - 1, axis=2)[:, :, :K]
        dk = np.take_along_axis(d2, kk, axis=2).transpose(0, 2, 1)    # строки × K × датчики
        kk = kk.transpose(0, 2, 1)
        fin = np.isfinite(dk)
        s = np.where(fin, np.exp(-(dk - np.where(fin, dk, np.inf).min(1, keepdims=True)) / (h * h)), 0.0).astype(np.float32)
        avail = ~np.isnan(xq) if flagged is None else ~np.isnan(xq) & ~flagged
        MxP = np.hstack([self.Mx, np.full((len(self.Mx), 1), np.nan, np.float32)])      # индекс −1 -> пустой вход
        avP = np.hstack([avail, np.zeros((nrow, 1), bool)])
        q = self.Mc.shape[1]
        out = np.full((nrow, nc), np.nan, np.float32)
        for b0 in range(0, nc, block):
            bc = slice(b0, min(b0 + block, nc))
            kb, sb, nb = kk[:, :, bc], s[:, :, bc], self.nb[cols[bc]]                  # nb: датчики блока × k
            Ck = self.Mc[kb]                                                           # строки × K × блок × условия
            dist = ((Ck[:, :, None] - Ck[:, None]) ** 2).sum(-1)                       # строки × K × K × блок
            Xk = MxP[kb[..., None], nb[None, None]]                                    # строки × K × блок × k
            T = (Xk[:, :, None] - Xk[:, None]) ** 2
            V = ~np.isnan(T) & avP[:, nb][:, None, None]
            dist += np.where(V, T, 0.0).sum(-1)
            cnt = q + avP[:, nb].sum(-1)[:, None, None, :]                            # одна нормировка на час: G — ядро Гаусса
            G = np.exp(-dist / cnt / (h * h)).transpose(0, 3, 1, 2)                   # строки × блок × K × K
            G += self.lam * np.eye(K, dtype=np.float32)
            sv = sb.transpose(0, 2, 1)                                                 # строки × блок × K
            w = spd_solve(G.reshape(-1, K, K), sv.reshape(-1, K)).reshape(sv.shape)
            w = np.where(fin[:, :, bc].transpose(0, 2, 1), w, 0.0)
            sw = w.sum(-1)
            bad = ~np.isfinite(sw) | (np.abs(sw) < 1e-6)                              # вырождено — ядерные веса
            w = np.where(bad[..., None], sv, w)
            sw = np.where(bad, sv.sum(-1), sw)
            y = Mt0[kb, np.arange(bc.start, bc.stop)[None, None]].transpose(0, 2, 1)  # строки × блок × K
            out[:, bc] = np.where(np.abs(sw) > 1e-12, (w * y).sum(-1) / np.where(np.abs(sw) > 1e-12, sw, 1.0), np.nan)
        return out


def spd_solve(G, s):
    """Решение G·w = s для пачки малых симметричных положительно определённых матриц (Холецкий, векторно по пачке:
    np.linalg.solve на тысячах матриц 20×20 в десятки раз медленнее)."""
    with np.errstate(all='ignore'):                                 # невырожденность проверяет вызывающий (сумма весов)
        return _chol_solve(G, s)


def _chol_solve(G, s):
    b, K, _ = G.shape
    L = np.zeros_like(G)
    for j in range(K):
        d = np.sqrt(np.maximum(G[:, j, j] - (L[:, j, :j] ** 2).sum(1), 1e-10))
        L[:, j, j] = d
        if j + 1 < K:
            L[:, j + 1:, j] = (G[:, j + 1:, j] - np.einsum('bik,bk->bi', L[:, j + 1:, :j], L[:, j, :j])) / d[:, None]
    y = np.zeros_like(s)
    for i in range(K):
        y[:, i] = (s[:, i] - (L[:, i, :i] * y[:, :i]).sum(1)) / L[:, i, i]
    w = np.zeros_like(s)
    for i in range(K - 1, -1, -1):
        w[:, i] = (y[:, i] - (L[:, i + 1:, i] * w[:, i + 1:]).sum(1)) / L[:, i, i]
    return w
