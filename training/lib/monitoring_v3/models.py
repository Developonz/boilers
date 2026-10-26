"""Predictors with explicit inheritance of fitted state; no hidden monthly refit."""
from copy import deepcopy
import numpy as np
import pandas as pd
from .preparation import FrozenEncoder


def balanced_rows(features, limit):
    """Deterministic coverage of load ranges, units and categorical operating states."""
    if len(features) <= limit:
        return np.arange(len(features))
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(features[c])]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(features[c])]
    strata = features[cats].fillna('__MISSING__').astype(str).agg('|'.join, axis=1) if cats else pd.Series('', index=features.index)
    if numeric:
        x = pd.to_numeric(features[numeric[0]], errors='coerce')
        bins = pd.qcut(x, min(10, max(1, x.nunique())), duplicates='drop').astype(str) if x.nunique() > 1 else pd.Series('0', index=x.index)
        strata = strata+'|'+bins
    cells = [np.flatnonzero(strata.to_numpy() == s) for s in sorted(strata.unique())]
    chosen = []
    # Round-robin evenly spaced representatives avoids starving rare modes.
    sequences = [cell[np.linspace(0, len(cell)-1, min(len(cell), limit)).astype(int)] for cell in cells]
    for rank in range(limit):
        for seq in sequences:
            if rank < len(seq):
                chosen.append(seq[rank])
                if len(chosen) == limit:
                    return np.sort(chosen)
    return np.sort(chosen)


class LinearStats:
    """Per-target sufficient statistics with missing targets excluded."""
    def __init__(self, width, targets, ridge=1.):
        self.xx = np.zeros((targets, width, width))
        self.xy = np.zeros((width, targets))
        self.count = np.zeros(targets, int)
        self.ridge = ridge

    def update(self, features, y):
        for j in range(y.shape[1]):
            good = np.isfinite(y[:, j]) & np.isfinite(features).all(axis=1)
            x = features[good]
            self.xx[j] += x.T@x
            self.xy[:, j] += x.T@y[good, j]
            self.count[j] += int(good.sum())
        reg = np.eye(features.shape[1])*self.ridge
        reg[0, 0] = 1e-9
        self.beta = np.column_stack([np.linalg.solve(self.xx[j]+reg, self.xy[:, j]) for j in range(y.shape[1])])
        return self.beta


class NormalModel:
    def __init__(self, method, cfg):
        self.method, self.cfg = method, deepcopy(cfg)
        self.updates = 0

    def fit_initial(self, features, targets):
        self.encoder = FrozenEncoder.fit(features)
        self.targets = list(targets)
        self.counts = targets.count().to_numpy(copy=True)
        q = targets.quantile([.005, .25, .5, .75, .995])
        self.centers = q.loc[.5].fillna(0).to_numpy(float)
        sc = ((q.loc[.75]-q.loc[.25])/1.349).fillna(targets.std()).fillna(0).to_numpy(float)
        # Nonzero explicit reference for a first change in a formerly constant sensor.
        self.scale = np.maximum(sc, np.maximum(np.abs(self.centers)*.001, 1e-3))
        self.floor = np.maximum(self.scale*.1, 1e-6)
        keep = balanced_rows(features, self.cfg['memory_rows'])
        self.memory_f = features.iloc[keep].copy()
        self.memory_y = targets.iloc[keep].copy()
        if self.method in ('CatBoost', 'HistGradientBoosting'):
            self.predictors = {}
            self._fit_trees(features, targets, initial=True)
        else:
            c, y = self.encoder.numeric(features), targets.to_numpy(float)
            if self.method == 'analogs':
                self.stats = LinearStats(c.shape[1]+1, y.shape[1])
                self.stats.update(np.column_stack([np.ones(len(c)), c]), y)
            else:
                import kernel_models as KM
                klass = KM.AAKR if self.method == 'AAKR' else KM.MSET
                self.kernel = klass(memory=self.cfg['memory_rows'], neighbors=min(10, max(0, y.shape[1]-1)), seed=self.cfg['seed'])
                self.kernel.fit(y, c, np.empty((len(c), 0)))
                lin_f = self.kernel.lin._F(c, np.empty((len(c), 0)))
                self.stats = LinearStats(lin_f.shape[1], y.shape[1])
                self.kernel.lin.beta = self.stats.update(lin_f, y)
            self._rebuild_memory()
        return self

    def _fit_trees(self, features, targets, initial):
        from .threads import numerical_threads
        native = self.encoder.native(features)
        cats = list(self.encoder.categories)
        for j, col in enumerate(self.targets):
            good = targets[col].notna() & np.isfinite(targets[col])
            if good.sum() < self.cfg['min_sensor_rows']:
                continue
            y = targets.loc[good, col].to_numpy(float)
            if np.ptp(y) <= 1e-12:
                # No tree can learn variation that has never been observed.
                self.predictors.setdefault(col, ('constant', float(np.median(y))))
                continue
            if not native.loc[good].nunique(dropna=False).gt(1).any():
                self.predictors.setdefault(col,('reference',float(np.median(y))))
                continue
            previous = self.predictors.get(col)
            if self.method == 'HistGradientBoosting':
                if not initial:
                    continue  # Frozen comparison baseline, never described as incremental.
                from sklearn.ensemble import HistGradientBoostingRegressor
                with numerical_threads(self.cfg['threads']):
                    models = [HistGradientBoostingRegressor(loss='quantile', quantile=a,
                              max_iter=self.cfg['cb_initial_trees'], max_depth=self.cfg['cb_depth'],
                              early_stopping=False, random_state=self.cfg['seed']).fit(self.encoder.numeric(features.loc[good]), y)
                              for a in (.25, .5, .75)]
                self.predictors[col] = ('hist', models)
            else:
                from catboost import CatBoostRegressor
                old = previous[1] if previous and previous[0] == 'catboost' else None
                trees = self.cfg['cb_initial_trees'] if initial else self.cfg['cb_update_trees']
                if old is not None and old.tree_count_+trees > self.cfg['cb_max_trees']:
                    continue
                model = CatBoostRegressor(loss_function='MultiQuantile:alpha=0.25,0.5,0.75',
                    iterations=trees, depth=self.cfg['cb_depth'], learning_rate=self.cfg['learning_rate'],
                    random_seed=self.cfg['seed'], thread_count=self.cfg['threads'],
                    has_time=True, allow_writing_files=False, verbose=False, task_type='CPU')
                model.fit(native.loc[good], y, cat_features=cats, init_model=old)
                self.predictors[col] = ('catboost', model)

    def update(self, features, targets):
        """Input contains ONLY newly admitted rows, never the previous full training window."""
        if not len(features):
            return self
        if list(targets) != self.targets or list(features) != self.encoder.columns:
            raise ValueError('Schema changed: incremental update cannot reinterpret old features')
        if self.method == 'HistGradientBoosting':
            return self
        if self.method == 'CatBoost':
            keep = balanced_rows(self.memory_f, self.cfg['replay_rows'])
            # Real dates must be ordered across units too (MultiIndex: time, unit).
            combined_f = pd.concat([self.memory_f.iloc[keep], features]).sort_index(kind='stable')
            combined_y = pd.concat([self.memory_y.iloc[keep], targets]).reindex(combined_f.index)
            self._fit_trees(combined_f, combined_y, initial=False)
        else:
            c = self.encoder.numeric(features)
            if self.method == 'analogs':
                f = np.column_stack([np.ones(len(c)), c])
            else:
                f = self.kernel.lin._F(c, np.empty((len(c), 0)))
            beta = self.stats.update(f, targets.to_numpy(float))
            if self.method != 'analogs':
                self.kernel.lin.beta = beta
        self.counts += targets.count().to_numpy()
        self.memory_f = pd.concat([self.memory_f, features])
        self.memory_y = pd.concat([self.memory_y, targets])
        keep = balanced_rows(self.memory_f, self.cfg['memory_rows'])
        self.memory_f, self.memory_y = self.memory_f.iloc[keep], self.memory_y.iloc[keep]
        if self.method != 'CatBoost':
            self._rebuild_memory()
        self.updates += 1
        return self

    def _rebuild_memory(self):
        c = self.encoder.numeric(self.memory_f)
        y = self.memory_y.to_numpy(float)
        if self.method == 'analogs':
            from sklearn.neighbors import NearestNeighbors
            self.memory_c = c
            self.residual_memory = y-np.column_stack([np.ones(len(c)), c])@self.stats.beta
            self.nn = NearestNeighbors(n_neighbors=min(self.cfg['neighbors'], len(c))).fit(c)
        else:
            m = self.kernel
            residual = y-m.lin.predict(c, np.empty((len(c), 0)))
            m.Mx = ((residual-m.xm)/m.xs).astype(np.float32)
            m.Mc = m._cond(c, np.empty((len(c), 0))).astype(np.float32)

    def expected(self, features, actual):
        n, p = len(features), len(self.targets)
        med, spread = np.full((n, p), np.nan), np.full((n, p), np.nan)
        if self.method in ('CatBoost', 'HistGradientBoosting'):
            native = self.encoder.native(features)
            for j, col in enumerate(self.targets):
                item = self.predictors.get(col)
                if item is None:
                    continue
                kind, model = item
                if kind in ('constant','reference'):
                    med[:, j], spread[:, j] = model, self.scale[j]
                    continue
                pred = (model.predict(native,thread_count=self.cfg['threads']) if kind == 'catboost' else
                        np.column_stack([m.predict(self.encoder.numeric(features)) for m in model]))
                # Sorting repairs crossing; interval diagnostics are reported separately.
                pred = np.sort(pred, axis=1)
                med[:, j], spread[:, j] = pred[:, 1], np.maximum((pred[:, 2]-pred[:, 0])/1.349, self.floor[j])
        else:
            c = self.encoder.numeric(features)
            if self.method == 'analogs':
                for start in range(0, n, 128):
                    pos = slice(start, min(n, start+128))
                    _, nb = self.nn.kneighbors(c[pos])
                    with np.errstate(all='ignore'):
                        q = np.nanquantile(self.residual_memory[nb], [.25, .5, .75], axis=1)
                    med[pos] = q[1]+np.column_stack([np.ones(len(c[pos])), c[pos]])@self.stats.beta
                    spread[pos] = np.maximum((q[2]-q[0])/1.349, self.floor)
            else:
                med, spread = self.kernel.predict(actual.reindex(columns=self.targets).to_numpy(float), c, np.empty((n, 0)))
                spread = np.maximum(spread, self.floor)
        valid = self.counts >= self.cfg['min_sensor_rows']
        med[:, ~valid], spread[:, ~valid] = np.nan, np.nan
        return (pd.DataFrame(med, index=features.index, columns=self.targets),
                pd.DataFrame(spread, index=features.index, columns=self.targets))

    def errors(self, features, actual):
        med, spread = self.expected(features, actual)
        return actual.reindex(columns=self.targets).sub(med).div(spread).replace([np.inf, -np.inf], np.nan)

    def tree_counts(self):
        return {c: int(m.tree_count_) for c, (kind, m) in getattr(self, 'predictors', {}).items() if kind == 'catboost'}

    def interval_diagnostics(self, features, actual):
        """Raw quantile crossing and empirical central-50% coverage, evaluated out of sample."""
        rows=[]
        if self.method not in ('CatBoost','HistGradientBoosting'):
            return rows
        native=self.encoder.native(features)
        for col,(kind,model) in self.predictors.items():
            if kind in ('constant','reference'):
                continue
            pred=(model.predict(native,thread_count=self.cfg['threads']) if kind=='catboost' else
                  np.column_stack([m.predict(self.encoder.numeric(features)) for m in model]))
            good=actual[col].notna().to_numpy() & np.isfinite(pred).all(axis=1)
            if not good.any():
                continue
            crossing=np.any(np.diff(pred[good],axis=1)<0,axis=1)
            ordered=np.sort(pred[good],axis=1);y=actual[col].to_numpy()[good]
            rows.append(dict(sensor=col,observed=int(good.sum()),raw_crossing_fraction=float(crossing.mean()),
                central50_coverage=float(((y>=ordered[:,0])&(y<=ordered[:,2])).mean()),
                median_interval_width=float(np.median(ordered[:,2]-ordered[:,0])),
                median_absolute_error=float(np.median(abs(y-ordered[:,1])))))
        return rows

    def candidate(self, features, targets, validation_f, validation_y):
        candidate = deepcopy(self).update(features, targets)
        # Compare reconstruction in physical units as well: inflating uncertainty is no improvement.
        om, oscale = self.expected(validation_f, validation_y)
        nm, nscale = candidate.expected(validation_f, validation_y)
        actual=validation_y.reindex(columns=self.targets)
        old = actual.sub(om).div(oscale).replace([np.inf,-np.inf],np.nan).abs().median().median()
        new = actual.sub(nm).div(nscale).replace([np.inf,-np.inf],np.nan).abs().median().median()
        oe = validation_y.sub(om).abs().div(self.scale).median().median()
        ne = validation_y.sub(nm).abs().div(self.scale).median().median()
        accepted = bool(np.isfinite(ne) and (not np.isfinite(oe) or ne <= max(oe, 1e-6)*(1+self.cfg['validation_tolerance'])))
        return candidate if accepted else self, dict(accepted=accepted, old_error=float(old), new_error=float(new),
                                                    old_physical_error=float(oe), new_physical_error=float(ne),
                                                    trees=candidate.tree_counts())


class RelatedModel:
    """Independent predictors with explicit, reviewed sibling inputs, never self inputs."""
    def __init__(self, cfg, related):
        self.cfg, self.related = cfg, related
        self.method = 'CatBoost-related'
        self.updates = 0

    def _inputs(self, f, target):
        keep = [c for c in f if not c.startswith('sensor::')]
        siblings = self.related.get(target, [])
        if target in siblings or not siblings:
            raise ValueError(f'Related inputs for {target} must be nonempty and exclude the target')
        return f[keep+['sensor::'+c for c in siblings]]

    def fit_initial(self, features, targets):
        self.targets = list(targets)
        self.members = {c: NormalModel('CatBoost', self.cfg).fit_initial(self._inputs(features, c), targets[[c]]) for c in targets}
        self.scale = np.array([self.members[c].scale[0] for c in targets])
        self.memory_f = features.iloc[balanced_rows(features, self.cfg['memory_rows'])].copy()
        self.memory_y = targets.reindex(self.memory_f.index)
        return self

    def update(self, features, targets):
        for c, model in self.members.items():
            model.update(self._inputs(features, c), targets[[c]])
        self.updates += 1
        return self

    def expected(self, features, actual):
        predictions = [self.members[c].expected(self._inputs(features, c), actual[[c]]) for c in self.targets]
        return pd.concat([p[0] for p in predictions], axis=1), pd.concat([p[1] for p in predictions], axis=1)

    def errors(self, features, actual):
        med, spread = self.expected(features, actual)
        return actual[self.targets].sub(med).div(spread)

    def tree_counts(self):
        return {c: m.tree_counts().get(c, 0) for c, m in self.members.items()}

    candidate = NormalModel.candidate


class AnalogCorrection:
    """CatBoost predicts out-of-sample analog bias, never the current observed error."""
    method = 'CatBoostCorrection'

    def __init__(self, cfg):
        self.cfg, self.updates = cfg, 0

    def _correction_inputs(self, features):
        dummy = pd.DataFrame(np.nan, index=features.index, columns=self.targets)
        med, _ = self.base.expected(features, dummy)
        f = features.copy()
        distance, _ = self.base.nn.kneighbors(self.base.encoder.numeric(features))
        f['analog_distance'] = distance.mean(axis=1)
        for c in med:
            f['expected::'+c] = med[c]
        return f, med

    def fit_initial(self, features, targets):
        times = features.index.get_level_values(0)
        split = times[int(len(times)*.7)]
        first = times < split
        if np.sum(~first) < self.cfg['min_sensor_rows']:
            raise ValueError('Insufficient out-of-sample healthy analog residuals')
        self.targets = list(targets)
        self.base = NormalModel('analogs',self.cfg).fit_initial(features.loc[first],targets.loc[first])
        cf, med = self._correction_inputs(features.loc[~first])
        self.correction = NormalModel('CatBoost',self.cfg).fit_initial(cf,targets.loc[~first]-med)
        self.base.update(features.loc[~first],targets.loc[~first])
        self.scale = self.base.scale
        self.memory_f, self.memory_y = self.base.memory_f, self.base.memory_y
        return self

    def update(self, features, targets):
        cf, med = self._correction_inputs(features)
        # Labels are produced BEFORE admitting new rows to the analog memory.
        self.correction.update(cf,targets-med)
        self.base.update(features,targets)
        self.memory_f,self.memory_y = self.base.memory_f,self.base.memory_y
        self.updates += 1
        return self

    def expected(self, features, actual):
        cf, med = self._correction_inputs(features)
        bias, _ = self.correction.expected(cf,actual)
        _, spread = self.base.expected(features,actual)
        return med+bias, spread

    def errors(self, features, actual):
        med,spread = self.expected(features,actual)
        return (actual[self.targets]-med)/spread

    def tree_counts(self):
        return self.correction.tree_counts()

    candidate = NormalModel.candidate


class MultiTargetModel(NormalModel):
    """Shared trees; observed target coordinates only, frozen output scaling."""
    def fit_initial(self, features, targets):
        self.targets=list(targets);self.encoder=FrozenEncoder.fit(features)
        self.counts=targets.count().to_numpy(copy=True)
        self.scale=targets.std().fillna(0).clip(lower=1e-3).to_numpy(float)
        self.floor=self.scale*.1
        self._fit_shared(features,targets,None)
        rows=balanced_rows(features,self.cfg['memory_rows'])
        self.memory_f,self.memory_y=features.iloc[rows],targets.iloc[rows]
        return self

    def _fit_shared(self, f, y, old):
        from catboost import CatBoostRegressor
        n=self.cfg['cb_initial_trees'] if old is None else self.cfg['cb_update_trees']
        if old is not None and old.tree_count_+n > self.cfg['cb_max_trees']:
            self.shared=old;return
        self.shared=CatBoostRegressor(loss_function='MultiRMSEWithMissingValues',iterations=n,
            depth=self.cfg['cb_depth'],learning_rate=self.cfg['learning_rate'],has_time=True,
            random_seed=self.cfg['seed'],thread_count=self.cfg['threads'],allow_writing_files=False,verbose=False,
            allow_const_label=True)
        observed=y.notna().any(axis=1)
        self.shared.fit(self.encoder.native(f.loc[observed]),y.loc[observed]/self.scale,
                        cat_features=list(self.encoder.categories),init_model=old)

    def update(self, features, targets):
        rows=balanced_rows(self.memory_f,self.cfg['replay_rows'])
        f=pd.concat([self.memory_f.iloc[rows],features]).sort_index()
        y=pd.concat([self.memory_y.iloc[rows],targets]).reindex(f.index)
        self._fit_shared(f,y,self.shared)
        self.counts+=targets.count().to_numpy()
        f=pd.concat([self.memory_f,features]);y=pd.concat([self.memory_y,targets])
        rows=balanced_rows(f,self.cfg['memory_rows'])
        self.memory_f,self.memory_y=f.iloc[rows],y.iloc[rows]
        self.updates+=1
        return self

    def expected(self, features, actual):
        med=np.asarray(self.shared.predict(self.encoder.native(features),thread_count=self.cfg['threads'])).reshape(len(features),len(self.targets))*self.scale
        med[:,self.counts < self.cfg['min_sensor_rows']]=np.nan
        return pd.DataFrame(med,index=features.index,columns=self.targets),pd.DataFrame(
            np.broadcast_to(self.scale,med.shape),index=features.index,columns=self.targets)

    def tree_counts(self):
        return {'shared':self.shared.tree_count_}
