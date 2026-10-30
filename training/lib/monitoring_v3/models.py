"""Predictors with explicit inheritance of fitted state; no hidden monthly refit."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .preparation import FrozenEncoder
from .progress import phase,PredictionProgress,current_progress


def snapshot_model(model):
    """Copy mutable pipeline state, retaining immutable fitted CatBoost estimators.

    Every continuation fits a NEW estimator with init_model; it never calls fit
    on a retained estimator. Avoid CatBoost.__deepcopy__ and its native reload
    path before continuation (upstream issue catboost/catboost#2905).
    """
    from catboost import CatBoost
    memo={};seen=set()
    def preserve(value):
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,CatBoost):
            memo[id(value)]=value
        elif isinstance(value,dict):
            for child in value.values():preserve(child)
        elif isinstance(value,(list,tuple)):
            for child in value:preserve(child)
        elif isinstance(value,(NormalModel,RelatedModel,AnalogCorrection)):
            preserve(vars(value))
    preserve(model)
    return deepcopy(model,memo)


def balanced_rows(features, limit):
    """Deterministic coverage of load ranges, units and categorical operating states."""
    if len(features) <= limit:
        return np.arange(len(features))
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(features[c])]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(features[c])]
    # Feature construction explicitly identifies the physical load; sorted
    # pooled columns must not accidentally balance on a temperature instead.
    load=features.attrs.get('load_column')
    if load in numeric:numeric=[load]+[c for c in numeric if c!=load]
    strata = features[cats].astype('string').fillna('__MISSING__').astype(str).agg('|'.join, axis=1) if cats else pd.Series('', index=features.index)
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


def operating_support(model,features):
    memory=getattr(model,'memory_f',None)
    if memory is None or not len(memory):return pd.DataFrame(dict(regime_seen=False,load_extrapolation=True),index=features.index)
    cats=[c for c in memory if c in features and not pd.api.types.is_numeric_dtype(memory[c])]
    if cats:
        known=memory[cats].astype('string').fillna('__MISSING__').astype(str).agg('|'.join,axis=1)
        query=features[cats].astype('string').fillna('__MISSING__').astype(str).agg('|'.join,axis=1)
        seen=query.isin(set(known))
    else:seen=pd.Series(True,index=features.index)
    load=features.attrs.get('load_column')
    outside=pd.Series(False,index=features.index)
    if load in memory:
        x=pd.to_numeric(features[load],errors='coerce');past=pd.to_numeric(memory[load],errors='coerce')
        outside=~x.between(past.min(),past.max())
    conditions=[c for c in features.attrs.get('condition_columns',[]) if c in memory and pd.api.types.is_numeric_dtype(memory[c])]
    ranges={};constant={}
    for c in conditions:
        past=pd.to_numeric(memory[c],errors='coerce').dropna();query=pd.to_numeric(features[c],errors='coerce')
        if not len(past):continue
        ranges[c]=(~query.between(past.min(),past.max())).where(query.notna()).astype(float)
        if past.nunique()==1:constant[c]=query.sub(past.iloc[0]).abs().gt(1e-9).where(query.notna()).astype(float)
    return pd.DataFrame(dict(regime_seen=seen,load_extrapolation=outside,
        condition_extrapolation_fraction=pd.DataFrame(ranges,index=features.index).mean(axis=1),
        constant_condition_changed=pd.DataFrame(constant,index=features.index).max(axis=1)),index=features.index)


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

    def _numeric_features(self,features):
        return self.encoder.numeric(features)

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
        rules=json.loads(Path(self.cfg['sensor_rules']).read_text(encoding='utf-8')) if Path(self.cfg['sensor_rules']).exists() else {}
        effective=[{**rules.get('defaults',{}),**rules.get('sensors',{}).get(c,{})} for c in self.targets]
        limits=[max(float(rule.get('scale_floor',0)),float(rule.get('resolution',0))) for rule in effective]
        self.floor=np.maximum(self.floor,limits)
        self.engineering_floor=self.floor.copy()
        keep = balanced_rows(features, self.cfg['memory_rows'])
        self.memory_f = features.iloc[keep].copy()
        self.memory_y = targets.iloc[keep].copy()
        if self.method in ('CatBoost', 'HistGradientBoosting'):
            self.predictors = {}
            self._fit_trees(features, targets, initial=True)
        else:
            c, y = self._numeric_features(features), targets.to_numpy(float)
            if self.method == 'analogs':
                self.stats = LinearStats(c.shape[1]+1, y.shape[1])
                self.stats.update(np.column_stack([np.ones(len(c)), c]), y)
            else:
                import kernel_models as KM
                klass = KM.AAKR if self.method == 'AAKR' else KM.MSET
                sensor_mode=getattr(self,'sensor_input_mode',None)
                all_other=sensor_mode=='all_other'
                classic=sensor_mode=='all'
                p=y.shape[1]
                nb=np.array([[i for i in range(p) if i!=j] for j in range(p)],dtype=np.int32).reshape(p,max(0,p-1)) if all_other else None
                self.kernel = klass(memory=self.cfg['memory_rows'], neighbors=max(0,p-1) if all_other else min(10,max(0,p-1)),
                    mode=sensor_mode if (all_other or classic) else 'per_sensor',robust=not (all_other or classic),
                    linear=not classic,seed=self.cfg['seed'])
                if classic and self.method=='MSET':
                    self.kernel.full_memory=True;self.kernel.lam=0.
                self.kernel.fit(y, c, np.empty((len(c), 0)),nb=nb)
                if self.kernel.lin is not None:
                    lin_f = self.kernel.lin._F(c, np.empty((len(c), 0)))
                    self.stats = LinearStats(lin_f.shape[1], y.shape[1])
                    self.kernel.lin.beta = self.stats.update(lin_f, y)
                else:self.stats=None
            self._rebuild_memory()
        return self

    def _fit_trees(self, features, targets, initial):
        from .threads import numerical_threads
        from .progress import current_progress
        from .target_training import TargetProgress,current_store
        native = self.encoder.native(features)
        cats = list(self.encoder.categories)
        reporter=TargetProgress(current_progress(),self.method,len(self.targets),initial)
        reporter.inputs(native,cats,self.cfg)
        store=current_store() if initial else None
        scope=store.scope(self.method,native,targets,self.cfg) if store else None
        variation={}
        for j, col in enumerate(self.targets):
            good = targets[col].notna() & np.isfinite(targets[col])
            trees=self.cfg['cb_initial_trees'] if initial else self.cfg['cb_update_trees']
            reporter.start(j+1,col,int(good.sum()),trees)
            saved=store.restore(scope,col) if store else None
            if saved is not None:
                if saved['predictor'] is not None:self.predictors[col]=saved['predictor']
                reporter.done(saved['status'],cached=True)
                continue
            if good.sum() < self.cfg['min_sensor_rows']:
                if store:store.save(scope,col,None,'insufficient_data')
                reporter.done('insufficient_data')
                continue
            y = targets.loc[good, col].to_numpy(float)
            previous = self.predictors.get(col)
            if np.ptp(y) <= 1e-12 and not (previous and previous[0]=='catboost'):
                # No tree can learn variation that has never been observed.
                self.predictors.setdefault(col, ('constant', float(np.median(y))))
                if store:store.save(scope,col,self.predictors[col],'constant')
                reporter.done('constant')
                continue
            mask_key=np.packbits(good.to_numpy()).tobytes()
            if mask_key not in variation:
                # Usually load is already variable: do not scan every input of
                # a wide forecast matrix again for every target.
                variation[mask_key]=any(native.loc[good,c].nunique(dropna=False)>1 for c in native.columns)
            if not variation[mask_key]:
                self.predictors.setdefault(col,('reference',float(np.median(y))))
                if store:store.save(scope,col,self.predictors[col],'constant_inputs')
                reporter.done('constant_inputs')
                continue
            previous = self.predictors.get(col)
            if self.method == 'HistGradientBoosting':
                if not initial:
                    reporter.done('frozen_baseline')
                    continue  # Frozen comparison baseline, never described as incremental.
                from sklearn.ensemble import HistGradientBoostingRegressor
                with numerical_threads(self.cfg['threads']):
                    numeric=self.encoder.numeric(features.loc[good])
                    models = [HistGradientBoostingRegressor(loss='quantile', quantile=a,
                              max_iter=self.cfg['cb_initial_trees'], max_depth=self.cfg['cb_depth'],
                              early_stopping=False, random_state=self.cfg['seed']).fit(numeric, y)
                              for a in (.25, .5, .75)]
                self.predictors[col] = ('hist', models)
            else:
                from catboost import CatBoostRegressor
                old = previous[1] if previous and previous[0] == 'catboost' else None
                trees = self.cfg['cb_initial_trees'] if initial else self.cfg['cb_update_trees']
                if old is not None and old.tree_count_+trees > self.cfg['cb_max_trees']:
                    reporter.done('tree_capacity_reached')
                    continue
                model = CatBoostRegressor(loss_function='MultiQuantile:alpha=0.25,0.5,0.75',
                    iterations=trees, depth=self.cfg['cb_depth'], learning_rate=self.cfg['learning_rate'],
                    random_seed=self.cfg['seed'], thread_count=self.cfg['threads'],
                    has_time=True, allow_writing_files=False, verbose=False, task_type='CPU',allow_const_label=True,
                    boost_from_average=False if old is not None else None)
                callbacks={'callbacks':[reporter]} if reporter.progress else {}
                from .continuation import fit_continued
                fit_continued(model,native.loc[good],y,old,cat_features=cats,**callbacks)
                self.predictors[col] = ('catboost', model)
            if store:store.save(scope,col,self.predictors[col],'trained')
            reporter.done('trained')

    def update(self, features, targets):
        """Input contains ONLY newly admitted rows, never the previous full training window."""
        if not len(features):
            return self
        if list(targets) != self.targets or list(features) != self.encoder.columns:
            raise ValueError('Schema changed: incremental update cannot reinterpret old features')
        if self.method == 'HistGradientBoosting':
            return self
        if self.method == 'CatBoost':
            with phase('подготовка новых строк и старых примеров'):
                keep = balanced_rows(self.memory_f, self.cfg['replay_rows'])
                # Real dates must be ordered across units too (MultiIndex: time, unit).
                combined_f = pd.concat([self.memory_f.iloc[keep], features]).sort_index(kind='stable')
                combined_y = pd.concat([self.memory_y.iloc[keep], targets]).reindex(combined_f.index)
            self._fit_trees(combined_f, combined_y, initial=False)
        else:
            c = self._numeric_features(features)
            if self.method == 'analogs':
                f = np.column_stack([np.ones(len(c)), c])
            elif self.kernel.lin is not None:
                f = self.kernel.lin._F(c, np.empty((len(c), 0)))
            if self.stats is not None:
                beta = self.stats.update(f, targets.to_numpy(float))
                if self.method != 'analogs':self.kernel.lin.beta = beta
        with phase(f'обновление памяти примеров: лимит {self.cfg["memory_rows"]} строк'):
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
        c = self._numeric_features(self.memory_f)
        y = self.memory_y.to_numpy(float)
        if self.method == 'analogs':
            from sklearn.neighbors import NearestNeighbors
            self.memory_c = c
            self.residual_memory = y-np.column_stack([np.ones(len(c)), c])@self.stats.beta
            self.nn = NearestNeighbors(n_neighbors=min(self.cfg['neighbors'], len(c))).fit(c)
        else:
            m = self.kernel
            residual = y-(m.lin.predict(c, np.empty((len(c), 0))) if m.lin is not None else 0.)
            m.Mx = ((residual-m.xm)/m.xs).astype(np.float32)
            m.Mc = m._cond(c, np.empty((len(c), 0))).astype(np.float32)
            m.__dict__.pop('_full_factor_cache',None)

    def _tree_prediction_input(self, features):
        """One transient input per batch; never retain a native Pool in model state."""
        with phase(f'подготовка входов прогноза {self.method}: {len(features)} строк, {len(features.columns)} входов'):
            if self.method == 'CatBoost':
                from catboost import Pool
                return Pool(self.encoder.native(features),cat_features=list(self.encoder.categories),
                            thread_count=self.cfg['threads'])
            return self.encoder.numeric(features)

    def expected(self, features, actual, *, prediction_input=None,diagnostic_start=None):
        n, p = len(features), len(self.targets)
        med, spread = np.full((n, p), np.nan), np.full((n, p), np.nan)
        diagnostics=[]
        if self.method in ('CatBoost', 'HistGradientBoosting'):
            tree_input=prediction_input
            if n and tree_input is None and any(kind in ('catboost','hist') for kind,_ in self.predictors.values()):
                tree_input=self._tree_prediction_input(features)
            reporter=PredictionProgress(p,n)
            for j, col in enumerate(self.targets):
                if not n:break
                reporter.start(j+1,col)
                item = self.predictors.get(col)
                if item is None:
                    reporter.done('no_model')
                    continue
                kind, model = item
                if kind in ('constant','reference'):
                    med[:, j], spread[:, j] = model, max(self.scale[j],self.floor[j])
                    reporter.done(kind)
                    continue
                pred = (model.predict(tree_input,thread_count=self.cfg['threads']) if kind == 'catboost' else
                        np.column_stack([m.predict(tree_input) for m in model]))
                # Sorting repairs crossing; interval diagnostics are reported separately.
                good=actual[col].notna().to_numpy()&np.isfinite(pred).all(axis=1) if col in actual else np.zeros(n,bool)
                if diagnostic_start is not None:good&=features.index>=pd.Timestamp(diagnostic_start)
                crossing=np.any(np.diff(pred,axis=1)<0,axis=1)
                pred = np.sort(pred, axis=1)
                if good.any():
                    truth=actual[col].to_numpy()[good];ordered=pred[good]
                    diagnostics.append(dict(sensor=col,observed=int(good.sum()),
                        raw_crossing_fraction=float(crossing[good].mean()),
                        central50_coverage=float(((truth>=ordered[:,0])&(truth<=ordered[:,2])).mean()),
                        median_interval_width=float(np.median(ordered[:,2]-ordered[:,0])),
                        median_absolute_error=float(np.median(abs(truth-ordered[:,1])))))
                med[:, j], spread[:, j] = pred[:, 1], np.maximum((pred[:, 2]-pred[:, 0])/1.349, self.floor[j])
                reporter.done()
        else:
            c = self._numeric_features(features)
            if self.method == 'analogs':
                for start in range(0, n, 128):
                    pos = slice(start, min(n, start+128))
                    _, nb = self.nn.kneighbors(c[pos])
                    with np.errstate(all='ignore'):
                        q = np.nanquantile(self.residual_memory[nb], [.25, .5, .75], axis=1)
                    med[pos] = q[1]+np.column_stack([np.ones(len(c[pos])), c[pos]])@self.stats.beta
                    spread[pos] = np.maximum((q[2]-q[0])/1.349, self.floor)
            else:
                # Bound tensors by memory width for a 200--400 sensor related
                # model; MSET additionally bounds its local Gram blocks.
                chunk=(max(1,min(32,2_000_000//max(1,len(self.kernel.Mx)*len(self.targets))))
                       if getattr(self,'sensor_input_mode',None) in ('all_other','all') else 32)
                med, spread = self.kernel.predict(actual.reindex(columns=self.targets).to_numpy(float), c, np.empty((n, 0)),chunk=chunk)
                spread = np.maximum(spread, self.floor)
        valid = self.counts >= self.cfg['min_sensor_rows']
        med[:, ~valid], spread[:, ~valid] = np.nan, np.nan
        center=pd.DataFrame(med,index=features.index,columns=self.targets)
        center.attrs['prediction_diagnostics']=diagnostics
        return center,pd.DataFrame(spread,index=features.index,columns=self.targets)

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
        if not len(features):return rows
        tree_input=(self._tree_prediction_input(features)
                    if any(kind in ('catboost','hist') for kind,_ in self.predictors.values()) else None)
        reporter=PredictionProgress(len(self.predictors),len(features))
        for j,(col,(kind,model)) in enumerate(self.predictors.items(),1):
            if kind in ('constant','reference'):
                continue
            reporter.start(j,col)
            pred=(model.predict(tree_input,thread_count=self.cfg['threads']) if kind=='catboost' else
                  np.column_stack([m.predict(tree_input) for m in model]))
            reporter.done('interval_diagnostics')
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
        with phase('копирование предыдущей модели'):
            candidate = snapshot_model(self)
        with phase('дообучение кандидата'):
            candidate.update(features, targets)
        if not self.cfg.get('validate_candidate',True):
            progress=current_progress()
            if progress:
                progress.say('Обновление принято без проверки кандидата',status='candidate_validation_disabled',accepted=True)
            return candidate,dict(accepted=True,validation_status='disabled',trees=candidate.tree_counts())
        shared={}
        # Only ordinary CatBoost shares an input schema across both complete
        # models. Related/Correction/Multi keep their own prediction contracts.
        if (type(self) is NormalModel and self.method=='CatBoost' and len(validation_f) and
            any(kind=='catboost' for model in (self,candidate) for kind,_ in model.predictors.values())):
            with phase('подготовка общих проверочных входов'):
                shared['prediction_input']=self._tree_prediction_input(validation_f)
        # Compare reconstruction in physical units as well: inflating uncertainty is no improvement.
        with phase(f'проверочные прогнозы старой модели: {len(validation_f)} строк'):
            om, oscale = self.expected(validation_f, validation_y,**shared)
        with phase(f'проверочные прогнозы новой модели: {len(validation_f)} строк'):
            nm, nscale = candidate.expected(validation_f, validation_y,**shared)
        with phase('сравнение ошибок и решение об обновлении'):
            actual=validation_y.reindex(columns=self.targets)
            old = actual.sub(om).div(oscale).replace([np.inf,-np.inf],np.nan).abs().median().median()
            new = actual.sub(nm).div(nscale).replace([np.inf,-np.inf],np.nan).abs().median().median()
            oe = validation_y.sub(om).abs().div(self.scale).median().median()
            ne = validation_y.sub(nm).abs().div(self.scale).median().median()
            accepted = bool(np.isfinite(ne) and (not np.isfinite(oe) or ne <= max(oe, 1e-6)*(1+self.cfg['validation_tolerance'])))
            per_old=actual.sub(om).abs().div(self.scale).median()
            per_new=actual.sub(nm).abs().div(self.scale).median()
            critical=[c for c in self.cfg.get('critical_sensors',[]) if c in actual]
            failed=[c for c in critical if pd.notna(per_new[c]) and pd.notna(per_old[c]) and
                    per_new[c]>max(float(per_old[c]),1e-6)*(1+self.cfg['validation_tolerance'])]
            catastrophic=[c for c in actual if pd.notna(per_new[c]) and pd.notna(per_old[c]) and
                          per_new[c]>max(float(per_old[c])*3.,.5)]
            failed=sorted(set(failed+catastrophic))
            accepted=accepted and not failed
            progress=current_progress()
            if progress:
                progress.say(('Кандидат принят' if accepted else 'Кандидат отклонён')+
                    f': ошибка старой модели {oe:.6g}, новой {ne:.6g}',status='candidate_decision',
                    accepted=accepted,old_physical_error=float(oe),new_physical_error=float(ne))
        return candidate if accepted else self, dict(accepted=accepted, old_error=float(old), new_error=float(new),
                                                    old_physical_error=float(oe), new_physical_error=float(ne),
                                                    validation_status='checked',critical_regressions=failed,
                                                    target_errors={c:dict(old=float(per_old[c]),new=float(per_new[c])) for c in actual},
                                                    trees=candidate.tree_counts())


class RelatedKernelModel(NormalModel):
    """Classical autoassociative reconstruction with one complete sensor vector."""
    sensor_input_mode='all'

    def _numeric_features(self,features):
        # Fixed scaling/one-hot encoding, without median imputation or missing
        # indicator columns. Complete-vector checks run before this method.
        parts=[]
        for c in self.encoder.columns:
            if c in self.encoder.categories:
                values=features[c].astype('string')
                categories=self.encoder.categories[c]
                parts.extend(values.eq(v).to_numpy(float) for v in categories)
                parts.append((~values.isin(categories)).to_numpy(float))
            else:
                parts.append(((features[c]-self.encoder.medians[c])/self.encoder.scales[c]).to_numpy(float))
        return np.column_stack(parts) if parts else np.empty((len(features),0))

    @staticmethod
    def _conditions(features):
        return features.loc[:,[c for c in features if not c.startswith('sensor::')]]

    @classmethod
    def complete_vectors(cls,features,targets):
        f=cls._conditions(features)
        numeric=f.select_dtypes(include='number')
        return (f.notna().all(axis=1)&np.isfinite(numeric).all(axis=1)&np.isfinite(targets).all(axis=1))

    def fit_initial(self,features,targets):
        complete=self.complete_vectors(features,targets)
        if complete.sum()<self.cfg['min_sensor_rows']:
            raise ValueError(f'Classical {self.method}: {int(complete.sum())} complete training vectors; '
                             f"required {self.cfg['min_sensor_rows']}")
        self.vector_diagnostics=dict(input_rows=len(targets),complete_rows=int(complete.sum()),
                                     incomplete_rows=int((~complete).sum()))
        super().fit_initial(self._conditions(features.loc[complete]),targets.loc[complete])
        self.related_diagnostics=[dict(target=c,input='все проверяемые датчики, включая сам датчик',
            input_count=len(self.targets),includes_self=True,source='classical_all',
            estimator='full_memory_MSET' if self.method=='MSET' else 'AAKR') for c in self.targets]
        return self

    def update(self,features,targets):
        complete=self.complete_vectors(features,targets)
        self.vector_diagnostics=dict(input_rows=len(targets),complete_rows=int(complete.sum()),
                                     incomplete_rows=int((~complete).sum()))
        if not complete.any():return self
        return super().update(self._conditions(features.loc[complete]),targets.loc[complete])

    def expected(self,features,actual,**kwargs):
        complete=self.complete_vectors(features,actual.reindex(columns=self.targets))
        median=pd.DataFrame(np.nan,index=features.index,columns=self.targets);spread=median.copy()
        if complete.any():
            m,s=super().expected(self._conditions(features.loc[complete]),actual.loc[complete],**kwargs)
            median.loc[complete]=m;spread.loc[complete]=s
        return median,spread


class RelatedModel:
    """Frozen sibling inputs, selected on training data; explicit lists take priority."""
    def __init__(self, cfg, related):
        self.cfg, self.related = deepcopy(cfg), deepcopy(related)
        self.method = 'CatBoost-related'
        self.updates = 0

    def _inputs(self, f, target):
        keep = [c for c in f if not c.startswith('sensor::')]
        siblings = self.related.get(target, [])
        if target in siblings:
            raise ValueError(f'Related inputs for {target} must exclude the target')
        return f[keep+['sensor::'+c for c in siblings]]

    def fit_initial(self, features, targets):
        self.targets = list(targets)
        from .config import DEFAULTS
        policy={**DEFAULTS['related_policy'],**self.cfg.get('related_policy',{})}
        available={c.removeprefix('sensor::') for c in features if c.startswith('sensor::')}
        rows=np.linspace(0,len(features)-1,min(len(features),policy['sample_rows'])).astype(int)
        sample=targets.iloc[rows].replace([np.inf,-np.inf],np.nan)
        correlations=sample.corr(method='spearman',min_periods=self.cfg['min_sensor_rows']) if policy['mode']=='auto' else None
        self.related_diagnostics=[]
        for target in self.targets:
            if target in self.related:
                chosen=list(dict.fromkeys(self.related[target]));source='manual'
                if target in chosen or set(chosen)-available:
                    raise ValueError(f'Invalid related inputs for {target}: self input or unavailable sensor')
            elif policy['mode']=='manual':
                raise ValueError(f'No related_inputs for {target}; provide a list or use related_policy.mode="auto"')
            else:
                candidates=correlations[target].drop(labels=[target]).dropna()
                candidates=candidates.loc[candidates.index.isin(available)&candidates.abs().ge(policy['min_abs_correlation'])]
                chosen=sorted(candidates.index,key=lambda c:(-abs(candidates[c]),c))[:policy['max_inputs']]
                source='training_spearman' if chosen else 'conditions_fallback'
            self.related[target]=chosen
            if not chosen:
                self.related_diagnostics.append(dict(target=target,input='',source=source,correlation=np.nan))
            for sibling in chosen:
                self.related_diagnostics.append(dict(target=target,input=sibling,source=source,
                    correlation=float(correlations.loc[target,sibling]) if correlations is not None and sibling in correlations else np.nan))
        self.members = {c: NormalModel('CatBoost', self.cfg).fit_initial(self._inputs(features, c), targets[[c]]) for c in targets}
        self.scale = np.array([self.members[c].scale[0] for c in targets])
        self.memory_f = features.iloc[balanced_rows(features, self.cfg['memory_rows'])].copy()
        self.memory_y = targets.reindex(self.memory_f.index)
        return self

    def update(self, features, targets):
        for c, model in self.members.items():
            model.update(self._inputs(features, c), targets[[c]])
        f=pd.concat([self.memory_f,features]).sort_index()
        y=pd.concat([self.memory_y,targets]).reindex(f.index)
        keep=balanced_rows(f,self.cfg['memory_rows'])
        self.memory_f,self.memory_y=f.iloc[keep],y.iloc[keep]
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
        self.centers=self.base.centers.copy()
        cf, med = self._correction_inputs(features.loc[~first])
        self.correction = NormalModel('CatBoost',self.cfg).fit_initial(cf,targets.loc[~first]-med)
        # Keep the reference fixed: otherwise correction labels would describe
        # an older analog predictor than the one used when applying the model.
        self.scale = self.base.scale
        self.floor=self.base.floor.copy()
        self.engineering_floor=self.base.engineering_floor.copy()
        rows=balanced_rows(features,self.cfg['memory_rows'])
        self.memory_f, self.memory_y = features.iloc[rows], targets.iloc[rows]
        return self

    def update(self, features, targets):
        cf, med = self._correction_inputs(features)
        # Every generation learns residuals against the same independent base.
        self.correction.update(cf,targets-med)
        f=pd.concat([self.memory_f,features]);y=pd.concat([self.memory_y,targets])
        rows=balanced_rows(f,self.cfg['memory_rows'])
        self.memory_f,self.memory_y = f.iloc[rows],y.iloc[rows]
        self.updates += 1
        return self

    def expected(self, features, actual):
        cf, med = self._correction_inputs(features)
        bias, _ = self.correction.expected(cf,actual)
        _, spread = self.base.expected(features,actual)
        return med+bias, spread.clip(lower=pd.Series(self.floor,index=self.targets),axis=1)

    def errors(self, features, actual):
        med,spread = self.expected(features,actual)
        return (actual[self.targets]-med)/spread

    def tree_counts(self):
        return self.correction.tree_counts()

    candidate = NormalModel.candidate


class MultiTargetModel(NormalModel):
    """Shared trees; observed target coordinates only, frozen output scaling."""
    def fit_initial(self, features, targets):
        targets=targets.replace([np.inf,-np.inf],np.nan)
        self.targets=list(targets);self.encoder=FrozenEncoder.fit(features)
        self.counts=targets.count().to_numpy(copy=True)
        self.centers=targets.median().fillna(0).to_numpy(float)
        self.scale=targets.std().fillna(0).clip(lower=1e-3).to_numpy(float)
        self.floor=self.scale*.1
        rules=json.loads(Path(self.cfg['sensor_rules']).read_text(encoding='utf-8')) if Path(self.cfg['sensor_rules']).exists() else {}
        policies=[{**rules.get('defaults',{}),**rules.get('sensors',{}).get(c,{})} for c in self.targets]
        self.floor=np.maximum(self.floor,[max(float(p.get('scale_floor',0)),float(p.get('resolution',0))) for p in policies])
        self.engineering_floor=self.floor.copy()
        # CatBoost cannot initialise a multivariate bias for a completely empty
        # coordinate. Keep the public schema, but fit only observed outputs.
        self.shared_targets=[c for c in targets if targets[c].count()>=self.cfg['min_sensor_rows']]
        self.late_models={}
        self.shared=None
        self._fit_shared(features,targets,None)
        rows=balanced_rows(features,self.cfg['memory_rows'])
        self.memory_f,self.memory_y=features.iloc[rows],targets.iloc[rows]
        return self

    def _fit_shared(self, f, y, old):
        from catboost import CatBoostRegressor
        columns=getattr(self,'shared_targets',self.targets)  # old checkpoints
        if not columns:return
        y=y[columns].replace([np.inf,-np.inf],np.nan)
        observed=y.notna().any(axis=1)
        if not observed.any():
            self.shared=old;return
        n=self.cfg['cb_initial_trees'] if old is None else self.cfg['cb_update_trees']
        if old is not None and old.tree_count_+n > self.cfg['cb_max_trees']:
            self.shared=old;return
        self.shared=CatBoostRegressor(loss_function='MultiRMSEWithMissingValues',iterations=n,
            depth=self.cfg['cb_depth'],learning_rate=self.cfg['learning_rate'],has_time=True,
            random_seed=self.cfg['seed'],thread_count=self.cfg['threads'],allow_writing_files=False,verbose=False,
            allow_const_label=True,boost_from_average=False if old is not None else None)
        from .continuation import fit_continued
        scales=pd.Series(self.scale,index=self.targets).loc[columns]
        fit_continued(self.shared,self.encoder.native(f.loc[observed]),y.loc[observed].div(scales),old,
                        cat_features=list(self.encoder.categories))

    def update(self, features, targets):
        if list(targets)!=self.targets or list(features)!=self.encoder.columns:
            raise ValueError('Schema changed: incremental update cannot reinterpret old features')
        targets=targets.replace([np.inf,-np.inf],np.nan)
        if not hasattr(self,'late_models'):self.late_models={}
        rows=balanced_rows(self.memory_f,self.cfg['replay_rows'])
        f=pd.concat([self.memory_f.iloc[rows],features]).sort_index()
        y=pd.concat([self.memory_y.iloc[rows],targets]).reindex(f.index)
        self._fit_shared(f,y,self.shared)
        for c,model in self.late_models.items():model.update(features,targets[[c]])
        # A formerly absent sensor gets its first model and a physical scale
        # from real past observations, without rebuilding the shared ensemble.
        for c in self.targets:
            if c in getattr(self,'shared_targets',self.targets) or c in self.late_models:continue
            if y[c].count()<self.cfg['min_sensor_rows']:continue
            model=NormalModel('CatBoost',self.cfg).fit_initial(f,y[[c]])
            self.late_models[c]=model;j=self.targets.index(c)
            self.centers[j]=model.centers[0];self.scale[j]=model.scale[0]
            self.floor[j]=model.floor[0];self.engineering_floor[j]=model.engineering_floor[0]
        self.counts+=targets.count().to_numpy()
        f=pd.concat([self.memory_f,features]);y=pd.concat([self.memory_y,targets])
        rows=balanced_rows(f,self.cfg['memory_rows'])
        self.memory_f,self.memory_y=f.iloc[rows],y.iloc[rows]
        self.updates+=1
        return self

    def expected(self, features, actual):
        med=pd.DataFrame(np.nan,index=features.index,columns=self.targets)
        spread=med.copy()
        columns=getattr(self,'shared_targets',self.targets)
        if len(features) and self.shared is not None:
            scale=pd.Series(self.scale,index=self.targets).loc[columns]
            med[columns]=np.asarray(self.shared.predict(self.encoder.native(features),thread_count=self.cfg['threads'])).reshape(len(features),len(columns))*scale.to_numpy()
            spread[columns]=np.broadcast_to(pd.Series(np.maximum(self.scale,self.floor),index=self.targets).loc[columns],(len(features),len(columns)))
        for c,model in getattr(self,'late_models',{}).items():
            m,s=model.expected(features,actual[[c]])
            med[c]=m[c];spread[c]=np.maximum(s[c],self.floor[self.targets.index(c)])
        med.loc[:,self.counts < self.cfg['min_sensor_rows']]=np.nan
        return med,spread

    def tree_counts(self):
        counts={'shared':self.shared.tree_count_} if self.shared is not None else {}
        for c,model in getattr(self,'late_models',{}).items():counts.update(model.tree_counts())
        for c in self.targets:
            if c not in getattr(self,'shared_targets',self.targets) and c not in getattr(self,'late_models',{}):counts[c]=0
        return counts

    def target_availability(self):
        shared=getattr(self,'shared_targets',self.targets)
        return [dict(sensor=c,observed_rows=int(self.counts[j]),status=(
            'shared' if c in shared else 'individual_late' if c in getattr(self,'late_models',{}) else 'waiting_for_measurements'))
            for j,c in enumerate(self.targets)]
