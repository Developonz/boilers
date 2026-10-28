"""A separate, causal classifier on sensor windows; no detector outputs required.

This source is also embedded in 05_Классификатор_без_регрессора.ipynb so that
copying the notebook alone works with the previous monitoring library.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import warnings
import numpy as np
import pandas as pd


DIRECT_DEFAULTS = dict(
    window_hours=24, lags_hours=[12, 24], trends_hours=[6, 24],
    delta_hours=[1, 6], switch_hours=[6, 24], include_since_start=True,
    include_startup=True, minimum_numeric_coverage=.7, horizon_hours=168,
    min_lead_hours=0, train_step_hours=6, seed=42, iterations=400,
    depth=4, learning_rate=.05, early_stopping_rounds=50, threads=2,
    threshold_metric='f2', minimum_precision=None, event_weights=True,
    train_end=None, validation_end=None, restore=True,
)


def direct_options(overrides=None):
    options={**DIRECT_DEFAULTS,**(overrides or {})}
    extra=set(options)-set(DIRECT_DEFAULTS)
    if extra:raise ValueError('Неизвестные параметры: '+str(sorted(extra)))
    if options['window_hours']!=24:raise ValueError('Этот контракт признаков рассчитан на окно 24 часа.')
    for name in ('lags_hours','trends_hours','delta_hours','switch_hours'):
        if any(h<=0 or h>24 for h in options[name]):raise ValueError(name+': допустимы часы от 0 до 24.')
    for name in ('train_step_hours','iterations','depth','early_stopping_rounds','threads','horizon_hours'):
        if options[name]<=0:raise ValueError(name+' должен быть положительным.')
    if options['min_lead_hours']<0 or options['min_lead_hours']>=options['horizon_hours']:
        raise ValueError('min_lead_hours должен быть меньше горизонта и неотрицательным.')
    if options['threshold_metric'] not in ('f1','f2'):raise ValueError('threshold_metric: f1 или f2.')
    if options['minimum_precision'] is not None and not 0<=options['minimum_precision']<=1:
        raise ValueError('minimum_precision: число от 0 до 1 или None.')
    if not 0<options['minimum_numeric_coverage']<=1:raise ValueError('Неверная полнота измерений.')
    if bool(options['train_end'])!=bool(options['validation_end']):
        raise ValueError('Задайте обе границы train_end / validation_end или оставьте обе None.')
    return options


def _direct_category(series):
    def value(v):
        if pd.isna(v):return '__MISSING__'
        if isinstance(v,(float,np.floating)) and float(v).is_integer():return str(int(v))
        return str(v)
    return series.astype(object).map(value)


def _direct_slope(frame,hours,step):
    """OLS on actual elapsed hours; available observations only, not imputed zeros."""
    window=max(1,int(pd.Timedelta(hours=hours)/step))
    t=pd.Series((frame.index-frame.index[0])/pd.Timedelta('1h'),index=frame.index)
    count=frame.rolling(window,min_periods=1).count()
    tx=frame.mul(t,axis=0).rolling(window,min_periods=1).sum()
    sx=frame.rolling(window,min_periods=1).sum()
    st=frame.notna().astype(float).mul(t,axis=0).rolling(window,min_periods=1).sum()
    st2=frame.notna().astype(float).mul(t*t,axis=0).rolling(window,min_periods=1).sum()
    denominator=st2-st*st/count
    return ((tx-st*sx/count)/denominator.where(denominator>1e-8)).where(count>=2)


def direct_features(context,cfg,options):
    """Features for both fit and inference. No labels or future sensor values used."""
    options=direct_options(options)
    index=context['frame'].index
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError('История должна иметь уникальный отсортированный индекс времени.')
    step=pd.Timedelta(cfg['step'])
    for h in [*options['lags_hours'],*options['delta_hours'],*options['switch_hours'],*options['trends_hours']]:
        if pd.Timedelta(hours=h)%step:raise ValueError('Окна и лаги должны делиться на шаг данных.')
    roles=context['roles']
    numeric=[c for c,r in roles.items() if r in ('target','condition','load')]
    categories=[c for c,r in roles.items() if r=='state']
    if any(str(c).strip().casefold() in ('состояние','_тип') for c in numeric+categories):
        raise ValueError('Служебная разметка ремонта не может быть входом классификатора.')
    if not numeric:raise ValueError(context['unit']+': нет числовых датчиков.')
    data=context['cleaned'].reindex(columns=numeric).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    coverage=data.notna().mean(axis=1)
    modes=['работа','пуск'] if options['include_startup'] else ['работа']
    operation=context['regime']['mode'].isin(modes)
    testing=context['frame'].get('_Тип',pd.Series('',index=index)).isin(['ИСП','ЗРР'])
    evidence=context.get('regime_spec',{})
    auto_currents=any(a.get('source')=='early_past' and a.get('threshold') is not None
        and pd.notna(a.get('threshold')) for a in evidence.get('audit',[]))
    needs_reference=context.get('threshold_source')=='initial_past' or auto_currents
    reference_end=pd.Timestamp(context.get('preparation_reference_end',evidence.get('reference_end',index[0])))
    reference_ready=pd.Series(index>=reference_end if needs_reference else True,index=index)
    eligible=operation&~testing&coverage.ge(options['minimum_numeric_coverage'])&reference_ready
    contiguous=index.to_series().diff().eq(step)
    segment=(~eligible|~eligible.shift(fill_value=False)|~contiguous).cumsum()
    schema=[];blocks=[]
    def record(sensor,operation,kind='numeric'):
        prefix='num' if sensor in numeric else 'cat'
        name=f'{prefix}::{sensor}::{operation}'
        description={'value':'Текущее значение', 'min_24h':'Минимум доступных значений за 24 часа',
            'max_24h':'Максимум доступных значений за 24 часа', 'mean_24h':'Среднее за 24 часа',
            'std_24h':'Стандартное отклонение за 24 часа, ddof=0',
            'held_hours':'Наблюдаемое время без изменения; на начале участка нижняя оценка'}.get(operation)
        if description is None:
            hours=operation.rsplit('_',1)[-1]
            description=('Значение '+hours+' назад' if operation.startswith('lag_') else
                'Изменение за '+hours if operation.startswith('delta_') else
                'Наклон линейного тренда за '+hours+', единиц датчика в час' if operation.startswith('trend_') else
                'Количество известных переключений за '+hours)
        schema.append(dict(feature=name,sensor=sensor,operation=operation,description=description,kind=kind,
            units=context.get('declared_units',{}).get(sensor,'')))
        return name
    names={}
    for operation in ['value',*[f'lag_{h}h' for h in options['lags_hours']],
        'min_24h','max_24h','mean_24h','std_24h',
        *[f'trend_{h}h' for h in options['trends_hours']],
        *[f'delta_{h}h' for h in options['delta_hours']]]:
        names[operation]={c:record(c,operation) for c in numeric}
    cat_names={}
    for c in categories:
        cat_names[c]={'value':record(c,'value','category'),'held_hours':record(c,'held_hours')}
        for h in options['switch_hours']:cat_names[c][f'switches_{h}h']=record(c,f'switches_{h}h')
    # Iterate only eligible continuous runs, never thousands of individual stopped rows.
    for _,chunk in data.loc[eligible].groupby(segment.loc[eligible],sort=False):
        out=[];rolling=chunk.rolling(max(1,int(pd.Timedelta('24h')/step)),min_periods=1)
        out.append(chunk.rename(columns=names['value']))
        for h in options['lags_hours']:
            out.append(chunk.shift(int(pd.Timedelta(hours=h)/step)).rename(columns=names[f'lag_{h}h']))
        for statistic in ('min','max','mean'):
            out.append(getattr(rolling,statistic)().rename(columns=names[statistic+'_24h']))
        std=rolling.std(ddof=0).where(rolling.count()>=2)
        out.append(std.rename(columns=names['std_24h']))
        for h in options['trends_hours']:
            out.append(_direct_slope(chunk,h,step).rename(columns=names[f'trend_{h}h']))
        for h in options['delta_hours']:
            out.append(chunk.diff(int(pd.Timedelta(hours=h)/step)).rename(columns=names[f'delta_{h}h']))
        for c in categories:
            source=context['frame'][c].reindex(chunk.index)
            known=source.notna();values=_direct_category(source)
            previous=values.shift();previous_known=known.shift(fill_value=False)
            change=known&previous_known&values.ne(previous)
            consecutive=(~known|~previous_known|values.ne(previous)).cumsum()
            # Observed unchanged duration; at initial observation it is a lower bound.
            held=source.groupby(consecutive).cumcount()*step/pd.Timedelta('1h')
            fields={cat_names[c]['value']:values,cat_names[c]['held_hours']:held.where(known)}
            for h in options['switch_hours']:
                count=change.astype(float).rolling(max(1,int(pd.Timedelta(hours=h)/step)),min_periods=1).sum()
                fields[cat_names[c][f'switches_{h}h']]=count.where(known)
            out.append(pd.DataFrame(fields,index=chunk.index))
        blocks.append(pd.concat(out,axis=1))
    columns=[s['feature'] for s in schema]
    features=pd.concat(blocks).reindex(index) if blocks else pd.DataFrame(index=index,columns=columns)
    features['unit']=str(context['unit'])
    schema.append(dict(feature='unit',sensor='',operation='boiler_id',description='Идентификатор котла',kind='category',units=''))
    if options['include_since_start']:
        features['since_start_hours']=context['regime']['since_start_hours'].where(eligible)
        schema.append(dict(feature='since_start_hours',sensor='',operation='time_since_start',description='Время после пуска; при неизвестном начале NaN',kind='numeric',units='h'))
    for entry in schema:
        c=entry['feature']
        if entry['kind']=='category':features[c]=_direct_category(features[c])
        else:features[c]=pd.to_numeric(features[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
    meta=pd.DataFrame(dict(eligible=eligible,numeric_coverage=coverage,mode=context['regime']['mode'],
        preparation_ready=reference_ready),index=index)
    return features,meta,pd.DataFrame(schema)


def direct_dataset(contexts,cfg,options,progress=None):
    from monitoring_v3.interpretation import risk_labels
    options=direct_options(options);xs={};ms={};schemas=[];audits=[]
    for unit,context in contexts.items():
        if progress:progress(f'{unit}: подготовка признаков и исходов')
        x,meta,schema=direct_features(context,cfg,options)
        index=meta.index[meta.eligible]
        observed=context['cleaned'][context['load']].notna()
        as_of=context['frame'].index[-1]
        y,g,audit,available=risk_labels(index,context['events'],options['horizon_hours'],as_of,as_of,
            observed=observed,min_lead_hours=options['min_lead_hours'],return_audit=True,return_available_at=True)
        meta['label']=y.reindex(meta.index)
        meta['label_group']=(unit+'::'+g.astype(str)).reindex(meta.index)
        meta['label_known_at']=available.reindex(meta.index)
        meta['source_end']=as_of
        audit.update(unit=unit,total_rows=len(meta),eligible_rows=int(meta.eligible.sum()),
            numeric_sensors=len([c for c,r in context['roles'].items() if r in ('load','condition','target')]),
            categorical_sensors=len([c for c,r in context['roles'].items() if r=='state']),
            features=len(x.columns),preparation_reference_end=context.get('preparation_reference_end'),
            excluded_reference_rows=int((~meta.preparation_ready).sum()),
            positive_rows=int(y.eq(1).sum()),negative_rows=int(y.eq(0).sum()),
            positive_events=int(g.loc[y.eq(1)].nunique()),unlabelled_rows=int(y.isna().sum()))
        xs[unit]=x;ms[unit]=meta;schemas.append(schema);audits.append(audit)
    def combine(parts):
        result=pd.concat(parts,names=['unit','time']).swaplevel().sort_index(kind='stable')
        result.index.names=['time','unit'];return result
    features=combine(xs);meta=combine(ms)
    schema=pd.concat(schemas,ignore_index=True).drop_duplicates('feature')
    for name in schema.loc[schema.kind.eq('category'),'feature']:
        features[name]=_direct_category(features[name])
    return features,meta,schema,pd.DataFrame(audits)


def _direct_masks(meta,train_end,validation_end,options):
    t=pd.DatetimeIndex(meta.index.get_level_values('time'))
    good=meta.eligible&meta.label.notna()
    gap=pd.Timedelta(hours=options['window_hours'])
    known=pd.to_datetime(meta.label_known_at)
    masks=dict(
        train=good&(t<train_end-gap)&known.le(train_end),
        validation=good&(t>=train_end)&(t<validation_end-gap)&known.le(validation_end),
        test=good&(t>=validation_end),
    )
    # Even manual time boundaries cannot split one failure/background group.
    later=set(meta.loc[masks['test'],'label_group'])
    masks['validation']&=~meta.label_group.isin(later)
    later|=set(meta.loc[masks['validation'],'label_group'])
    masks['train']&=~meta.label_group.isin(later)
    return masks


def direct_split(meta,options):
    options=direct_options(options)
    t=pd.DatetimeIndex(meta.index.get_level_values('time'))
    positive=meta.loc[meta.eligible&meta.label.eq(1)]
    starts=pd.Series(positive.index.get_level_values('time'),index=positive.label_group).groupby(level=0).min().sort_values()
    if options['train_end']:
        a,b=pd.Timestamp(options['train_end']),pd.Timestamp(options['validation_end'])
        if a>=b:raise ValueError('train_end должен быть раньше validation_end.')
        masks=_direct_masks(meta,a,b,options)
        method='manual'
    else:
        candidates=pd.DatetimeIndex(sorted(starts.unique()))
        if len(candidates)>24:candidates=candidates[np.unique(np.linspace(0,len(candidates)-1,24).astype(int))]
        best=None
        for i,a in enumerate(candidates[1:-1],1):
            for b in candidates[i+1:]:
                masks0=_direct_masks(meta,a,b,options)
                counts=[];okay=True
                for name,mask in masks0.items():
                    if meta.loc[mask,'label'].nunique()!=2:okay=False;break
                    counts.append(meta.loc[mask&meta.label.eq(1),'label_group'].nunique())
                if not okay:continue
                fractions=np.array(counts)/max(len(starts),1)
                loss=float(np.square(fractions-[.6,.2,.2]).sum()+.05*(1-fractions.sum()))
                if best is None or loss<best[0]:best=(loss,a,b,masks0)
        if best is None:
            raise ValueError('Нельзя выделить независимые train/validation/test с обоими классами. '
                f'Доступно независимых аварий: {len(starts)}. Нужны аварии в трёх последовательных '
                'периодах с известными к границам исходами. Объедините котлы одной группы или задайте '
                'две границы вручную; случайное перемешивание часов не используется.')
        _,a,b,masks=best;method='chronological_event_counts'
    rows=[]
    for name,mask in masks.items():
        subset=meta.loc[mask];positive=subset.label.eq(1)
        rows.append(dict(partition=name,rows=len(subset),negative=int(subset.label.eq(0).sum()),
            positive=int(positive.sum()),positive_events=int(subset.loc[positive,'label_group'].nunique()),
            negative_groups=int(subset.loc[~positive,'label_group'].nunique()),
            first=t[mask].min() if mask.any() else pd.NaT,last=t[mask].max() if mask.any() else pd.NaT))
    summary=pd.DataFrame(rows)
    if any(meta.loc[mask,'label'].nunique()!=2 for mask in masks.values()):
        raise ValueError('В одном из заданных периодов нет обоих классов:\n'+summary.to_string(index=False))
    return masks,summary,dict(train_end=str(a),validation_end=str(b),method=method,
        window_gap_hours=options['window_hours'],unassigned_labelled_rows=int(
            (meta.label.notna()&~(masks['train']|masks['validation']|masks['test'])).sum()))


def _direct_stratified_positions(positions,groups,amount,rng):
    """Round-robin groups, rather than sampling an entire quiet month."""
    if amount>=len(positions):return np.asarray(positions,dtype=int)
    buckets=[]
    for _,indices in pd.Series(positions,index=np.asarray(groups)).groupby(level=0,sort=False):
        values=indices.to_numpy().copy();rng.shuffle(values);buckets.append(list(values))
    rng.shuffle(buckets);selected=[]
    while len(selected)<amount:
        remaining=[]
        for bucket in buckets:
            if bucket:selected.append(bucket.pop())
            if bucket:remaining.append(bucket)
            if len(selected)==amount:break
        buckets=remaining
    return np.sort(selected)


def direct_balance(meta,train_mask,options):
    options=direct_options(options);t=pd.DatetimeIndex(meta.index.get_level_values('time'))
    period=pd.Timedelta(hours=options['train_step_hours']).value
    sample=(t.as_unit('ns').asi8%period)==0
    positions=np.flatnonzero(train_mask.to_numpy()&sample)
    counts=meta.iloc[positions].label.value_counts()
    if set(counts.index)!={0.,1.}:raise ValueError('После шага обучения нет обоих классов.')
    amount=int(counts.min());rng=np.random.default_rng(options['seed']);selected=[]
    for label in (0,1):
        pos=positions[meta.iloc[positions].label.eq(label).to_numpy()]
        selected.extend(_direct_stratified_positions(pos,meta.iloc[pos].label_group,amount,rng))
    selected=np.sort(selected);subset=meta.iloc[selected]
    weights=np.ones(len(selected),dtype=float)
    if options['event_weights']:
        weights=1/subset.label_group.map(subset.label_group.value_counts()).to_numpy(float)
        totals=pd.Series(weights,index=subset.index).groupby(subset.label).sum()
        weights=weights/subset.label.map(totals).to_numpy(float)*len(weights)/2
    report=pd.DataFrame([dict(label=label,rows=int(subset.label.eq(label).sum()),
        independent_groups=int(subset.loc[subset.label.eq(label),'label_group'].nunique()),
        total_weight=float(weights[subset.label.eq(label).to_numpy()].sum())) for label in (0,1)])
    return selected,weights,report


def direct_threshold(y,scores,options):
    from sklearn.metrics import precision_recall_curve
    p,r,thresholds=precision_recall_curve(y,scores)
    table=pd.DataFrame(dict(threshold=thresholds,precision=p[:-1],recall=r[:-1]))
    for beta,name in ((1,'f1'),(2,'f2')):
        denominator=beta*beta*table.precision+table.recall
        table[name]=(1+beta*beta)*table.precision*table.recall/denominator.replace(0,np.nan)
        table[name]=table[name].fillna(0.)
    table['allowed']=True
    if options['minimum_precision'] is not None:
        table['allowed']=table.precision.ge(options['minimum_precision'])
    allowed=table.loc[table.allowed]
    if allowed.empty:
        return .5,table,'minimum_precision_not_reached: выбран фиксированный 0.5; ограничение не выполнено'
    chosen=allowed.sort_values([options['threshold_metric'],'precision','threshold'],ascending=False).iloc[0]
    return float(chosen.threshold),table,'selected_on_validation_'+options['threshold_metric']


def direct_metrics(y,scores,threshold):
    from sklearn.metrics import (precision_score,recall_score,f1_score,fbeta_score,
        average_precision_score,roc_auc_score,log_loss,brier_score_loss,confusion_matrix,accuracy_score)
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float);pred=scores>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(rows=len(y),positive=int(y.sum()),prevalence=float(y.mean()),threshold=float(threshold),
        precision=float(precision_score(y,pred,zero_division=0)) if pred.any() else None,
        recall=float(recall_score(y,pred,zero_division=0)) if y.any() else None,
        f1=float(f1_score(y,pred,zero_division=0)),f2=float(fbeta_score(y,pred,beta=2,zero_division=0)),
        average_precision=float(average_precision_score(y,scores)) if len(np.unique(y))==2 else None,
        roc_auc=float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
        logloss=float(log_loss(y,np.column_stack([1-scores,scores]),labels=[0,1])),
        brier=float(brier_score_loss(y,scores)),accuracy=float(accuracy_score(y,pred)),
        true_positive=int(tp),false_positive=int(fp),false_negative=int(fn),true_negative=int(tn),
        predicted_positive_fraction=float(pred.mean()))


def direct_native(features,columns,categorical):
    out={}
    for c in columns:
        values=features[c] if c in features else pd.Series(np.nan,index=features.index)
        out[c]=_direct_category(values) if c in categorical else pd.to_numeric(values,errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
    return pd.DataFrame(out,index=features.index)


def direct_fit(features,meta,schema,options,folder,verbose=100):
    from catboost import CatBoostClassifier,Pool
    from monitoring_v3.threads import numerical_threads
    import joblib
    options=direct_options(options);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    masks,partitions,boundaries=direct_split(meta,options)
    positions,weights,balance=direct_balance(meta,masks['train'],options)
    train=features.iloc[positions]
    varying=train.nunique(dropna=False).gt(1)&train.notna().any()
    columns=list(train.columns[varying]);dropped=list(train.columns[~varying])
    if not columns:raise ValueError('После подготовки нет изменяющихся входов.')
    categorical=[c for c in columns if c in set(schema.loc[schema.kind.eq('category'),'feature'])]
    signature=hashlib.sha256(json.dumps(dict(contract=1,options=options,boundaries=boundaries,
        versions={p:importlib.metadata.version(p) for p in ('catboost','pandas','numpy','scikit-learn')}),sort_keys=True).encode())
    for frame in (features,meta):signature.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    token=signature.hexdigest();cache=folder/'classifier.joblib'
    if options['restore'] and cache.exists():
        try:cached=joblib.load(cache)
        except Exception:cached={}
        if cached.get('fingerprint')==token and all(k in cached for k in ('metrics','predictions','importance')):
            print('Модель и проверка восстановлены: данные и настройки совпадают.',flush=True)
            _direct_save_reports(cached,folder)
            return cached,masks,partitions,balance
    model=CatBoostClassifier(iterations=int(options['iterations']),depth=int(options['depth']),
        learning_rate=options['learning_rate'],loss_function='Logloss',eval_metric='PRAUC',
        random_seed=options['seed'],thread_count=int(options['threads']),has_time=True,
        allow_writing_files=False)
    validation=features.loc[masks['validation']]
    train_pool=Pool(direct_native(train,columns,categorical),meta.iloc[positions].label.astype(int),
        cat_features=categorical,weight=weights)
    validation_pool=Pool(direct_native(validation,columns,categorical),
        meta.loc[masks['validation'],'label'].astype(int),cat_features=categorical)
    with numerical_threads(options['threads']):
        model.fit(train_pool,eval_set=validation_pool,use_best_model=True,
            early_stopping_rounds=int(options['early_stopping_rounds']),verbose=verbose)
    validation_scores=model.predict_proba(validation_pool,thread_count=options['threads'])[:,1]
    threshold,threshold_table,threshold_status=direct_threshold(
        meta.loc[masks['validation'],'label'].astype(int),validation_scores,options)
    # Persist weights and threshold before test plots/report rendering.
    model.save_model(str(folder/'classifier.cbm'))
    package=dict(model=model,columns=columns,categorical=categorical,options=options,
        schema=schema,boundaries=boundaries,threshold=threshold,threshold_status=threshold_status,
        dropped_columns=dropped,fingerprint=token,score_status='uncalibrated_after_balanced_training',
        available_at=boundaries['validation_end'],training_rows=len(positions),
        train_prevalence=float(meta.loc[masks['train'],'label'].mean()),
        split_summary=partitions,balance_summary=balance,threshold_table=threshold_table,
        validation_scores=validation_scores)
    joblib.dump(package,cache,compress=3)
    # Predict known test examples and the unlabelled tail without using their outcomes.
    times=pd.DatetimeIndex(meta.index.get_level_values('time'))
    inference=meta.eligible&(times>=pd.Timestamp(boundaries['validation_end']))
    test_scores=model.predict_proba(Pool(direct_native(features.loc[inference],columns,categorical),
        cat_features=categorical),thread_count=options['threads'])[:,1]
    predictions=meta.loc[inference].copy();predictions['score']=test_scores
    predictions['prediction_05']=test_scores>=.5;predictions['prediction_tuned']=test_scores>=threshold
    predictions['evaluation_eligible']=masks['test'].reindex(predictions.index).fillna(False)
    predictions['model_available_at']=pd.Timestamp(package['available_at'])
    rows=[]
    for name,labels,scores in [('validation',meta.loc[masks['validation'],'label'],validation_scores),
        ('test',predictions.loc[predictions.evaluation_eligible,'label'],predictions.loc[predictions.evaluation_eligible,'score'])]:
        for label,value in [('default_05',.5),('validation_threshold',threshold)]:
            rows.append(dict(partition=name,decision=label,**direct_metrics(labels,scores,value)))
    tested=predictions.loc[predictions.evaluation_eligible]
    baseline=direct_metrics(tested.label,np.full(len(tested),package['train_prevalence']),.5)
    rows.append(dict(partition='test',decision='constant_train_prevalence',**baseline))
    package.update(predictions=predictions,metrics=pd.DataFrame(rows),
        importance=pd.DataFrame(dict(feature=columns,importance=model.get_feature_importance(thread_count=options['threads']))).sort_values('importance',ascending=False),
        learning_curve=model.get_evals_result())
    joblib.dump(package,cache,compress=3)
    _direct_save_reports(package,folder)
    return package,masks,partitions,balance


def _direct_save_reports(package,folder):
    folder=Path(folder)
    for name,key in [('metrics','metrics'),('split_summary','split_summary'),('balance_summary','balance_summary'),
        ('threshold_search','threshold_table'),('feature_importance','importance')]:
        package[key].to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    package['schema'].to_csv(folder/'feature_schema.csv',index=False,encoding='utf-8-sig')
    package['predictions'].to_parquet(folder/'test_predictions.parquet')
    (folder/'settings.json').write_text(json.dumps(dict(options=package['options'],boundaries=package['boundaries'],
        threshold=package['threshold'],threshold_status=package['threshold_status'],
        feature_columns=package['columns'],categorical_columns=package['categorical'],
        dropped_columns=package['dropped_columns'],score_status=package['score_status'],
        available_at=package['available_at']),ensure_ascii=False,indent=2),encoding='utf-8')


def direct_predict(package,features):
    from catboost import Pool
    times=(pd.DatetimeIndex(features.index.get_level_values('time'))
        if isinstance(features.index,pd.MultiIndex) else pd.DatetimeIndex(features.index))
    if len(times) and times.min()<pd.Timestamp(package['available_at']):
        raise ValueError('Модель ещё не была доступна на запрошенную дату: '+package['available_at'])
    x=direct_native(features,package['columns'],package['categorical'])
    score=package['model'].predict_proba(Pool(x,cat_features=package['categorical']),
        thread_count=package['options']['threads'])[:,1]
    return pd.DataFrame(dict(score=score,prediction=score>=package['threshold']),index=features.index)


def direct_event_report(package,contexts):
    from monitoring_v3.event_registry import event_type_at
    rows=[];predictions=package['predictions'];left=pd.Timestamp(package['available_at'])
    horizon=pd.Timedelta(hours=package['options']['horizon_hours'])
    for unit,context in contexts.items():
        part=predictions.xs(unit,level='unit') if unit in predictions.index.get_level_values('unit') else pd.DataFrame()
        right=context['frame'].index[-1]
        for event in context['events'].itertuples():
            kind,known=event_type_at(event,right)
            if kind!='Авария' or pd.isna(event.stop) or not left<=event.stop<=right:continue
            window=part.loc[(part.index>=event.stop-horizon)&(part.index<event.stop)] if len(part) else part
            window=window.loc[window.evaluation_eligible&window.label.eq(1)] if len(window) else window
            warnings=window.loc[window.prediction_tuned] if len(window) else window
            rows.append(dict(unit=unit,event_id=event.event_id,stop=event.stop,
                scored_points=len(window),window_complete_in_test=event.stop-horizon>=left,
                detected=bool(len(warnings)),first_warning=warnings.index[0] if len(warnings) else pd.NaT,
                lead_hours=float((event.stop-warnings.index[0])/pd.Timedelta('1h')) if len(warnings) else np.nan))
    return pd.DataFrame(rows,columns=['unit','event_id','stop','scored_points','window_complete_in_test','detected','first_warning','lead_hours'])


def direct_standard_report(package,contexts,folder):
    """Use sklearn displays/report, plus explicit event coverage and time plots."""
    from IPython.display import display
    from sklearn.metrics import classification_report,ConfusionMatrixDisplay,PrecisionRecallDisplay,RocCurveDisplay
    import matplotlib.pyplot as plt
    folder=Path(folder);test=package['predictions'].loc[lambda d:d.evaluation_eligible]
    display(package['metrics'])
    for threshold in dict.fromkeys([.5,package['threshold']]):
        print(f'ТЕСТ · порог {threshold:.6f}')
        print(classification_report(test.label.astype(int),test.score.ge(threshold),
            labels=[0,1],target_names=[f'Нет аварии за {package["options"]["horizon_hours"]} ч',
                f'Авария за {package["options"]["horizon_hours"]} ч'],zero_division=0))
    fig,axes=plt.subplots(1,3,figsize=(17,4))
    PrecisionRecallDisplay.from_predictions(test.label,test.score,ax=axes[0],name='CatBoost')
    axes[0].axhline(test.label.mean(),ls='--',color='grey',label='Доля положительных точек');axes[0].legend()
    RocCurveDisplay.from_predictions(test.label,test.score,ax=axes[1]);axes[1].plot([0,1],[0,1],'--',color='grey')
    ConfusionMatrixDisplay.from_predictions(test.label.astype(int),test.prediction_tuned,labels=[0,1],ax=axes[2],colorbar=False)
    axes[2].set_title(f'Порог validation: {package["threshold"]:.3f}')
    fig.tight_layout();fig.savefig(folder/'test_metrics.png',dpi=130,bbox_inches='tight');display(fig);plt.close(fig)
    events=direct_event_report(package,contexts);events.to_csv(folder/'test_events.csv',index=False,encoding='utf-8-sig')
    if len(events):
        display(events)
        print(f'Аварий обнаружено: {int(events.detected.sum())}/{len(events)}; '
            f'без пригодных точек: {int(events.scored_points.eq(0).sum())}; '
            f'полное предаварийное окно в тесте: {int(events.window_complete_in_test.sum())}.')
    else:print('В выбранном тестовом периоде нет зарегистрированных аварийных остановов для событийного отчёта.')
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    curve=package['threshold_table'];axes[0].plot(curve.threshold,curve.precision,label='Precision')
    axes[0].plot(curve.threshold,curve.recall,label='Recall');axes[0].plot(curve.threshold,curve[package['options']['threshold_metric']],label=package['options']['threshold_metric'].upper())
    axes[0].axvline(package['threshold'],ls='--',color='black');axes[0].set_title('Подбор только на validation');axes[0].legend()
    for name,values in package['learning_curve'].items():
        if 'Logloss' in values:axes[1].plot(values['Logloss'],label=name)
    axes[1].set_title('Logloss по итерациям');axes[1].legend();fig.tight_layout()
    fig.savefig(folder/'validation_threshold.png',dpi=130,bbox_inches='tight');display(fig);plt.close(fig)
    display(package['importance'].head(30))
    for unit,context in contexts.items():
        if unit not in package['predictions'].index.get_level_values('unit'):continue
        part=package['predictions'].xs(unit,level='unit')
        fig,ax=plt.subplots(figsize=(15,4));ax.plot(part.index,part.score,lw=.7,label='Балл риска')
        ax.axhline(package['threshold'],color='red',ls='--',label='Подобранный порог')
        ax.axhline(.5,color='grey',ls=':',label='0.5')
        if len(events):
            for stop in events.loc[events.unit.eq(unit),'stop']:ax.axvline(stop,color='darkred',alpha=.6)
        unknown=part.loc[~part.evaluation_eligible]
        if len(unknown):ax.scatter(unknown.index,unknown.score,s=3,color='grey',label='Исход ещё неизвестен / исключён')
        ax.set_title(unit+' · отдельный тест и последующая история; балл не откалиброванная вероятность')
        ax.set_ylim(-.02,1.02);ax.legend();fig.tight_layout()
        fig.savefig(folder/(unit+'_risk.png'),dpi=130,bbox_inches='tight');display(fig);plt.close(fig)
    return events
