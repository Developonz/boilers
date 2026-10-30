"""A separate, causal classifier on sensor windows; no detector outputs required.

This source is also embedded in 05_Классификатор_без_регрессора.ipynb so that
copying the notebook alone works with the previous monitoring library.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import warnings
import base64
from html import escape
from io import BytesIO
import numpy as np
import pandas as pd


DIRECT_DEFAULTS = dict(
    window_hours=24, lags_hours=[12, 24], trends_hours=[6, 24],
    delta_hours=[1, 6], switch_hours=[6, 24], include_since_start=True,
    include_startup=True, minimum_numeric_coverage=.7, horizon_hours=168,
    min_lead_hours=0, train_step_hours=6, seed=42, iterations=400,
    depth=4, learning_rate=.05, early_stopping_rounds=50, threads=2,
    threshold_metric='f2', minimum_precision=None, balance_train=False, event_weights=False,
    train_end=None, validation_end=None, test_end=None, restore=True,
    allow_single_class_evaluation=False, macro_thresholds=True,
    alarm_selection=False, alarm_persist_hours=3, alarm_merge_hours=12,
    alarm_budget=2., alarm_background_fraction=.05, alarm_candidates=61,
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
    if options['threshold_metric'] not in ('f05','f1','f2'):raise ValueError('threshold_metric: f05, f1 или f2.')
    if options['minimum_precision'] is not None and not 0<=options['minimum_precision']<=1:
        raise ValueError('minimum_precision: число от 0 до 1 или None.')
    if options['alarm_persist_hours']<=0 or options['alarm_merge_hours']<0 or options['alarm_budget']<0:
        raise ValueError('Неверные параметры подтверждения/бюджета тревог.')
    if not 0<=options['alarm_background_fraction']<=1 or options['alarm_candidates']<3:
        raise ValueError('Неверная доля фоновой тревоги / число кандидатов.')
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


def direct_features(context,cfg,options,eligible_only=False):
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
        out.append(chunk.rename(columns=names['value']).astype('float32'))
        for h in options['lags_hours']:
            out.append(chunk.shift(int(pd.Timedelta(hours=h)/step)).rename(columns=names[f'lag_{h}h']).astype('float32'))
        for statistic in ('min','max','mean'):
            out.append(getattr(rolling,statistic)().rename(columns=names[statistic+'_24h']).astype('float32'))
        std=rolling.std(ddof=0).where(rolling.count()>=2)
        out.append(std.rename(columns=names['std_24h']).astype('float32'))
        for h in options['trends_hours']:
            out.append(_direct_slope(chunk,h,step).rename(columns=names[f'trend_{h}h']).astype('float32'))
        for h in options['delta_hours']:
            out.append(chunk.diff(int(pd.Timedelta(hours=h)/step)).rename(columns=names[f'delta_{h}h']).astype('float32'))
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
    output_index=index[eligible] if eligible_only else index
    features=pd.concat(blocks).reindex(output_index) if blocks else pd.DataFrame(index=output_index,columns=columns)
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
    if options.get('test_end'):masks['test']&=t<pd.Timestamp(options['test_end'])
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
    if meta.loc[masks['train'],'label'].nunique()!=2:
        raise ValueError('В обучении отсутствует один из классов:\n'+summary.to_string(index=False))
    if not masks['validation'].any():raise ValueError('Нет известных исходов в validation.')
    if not options['allow_single_class_evaluation'] and any(meta.loc[mask,'label'].nunique()!=2 for mask in masks.values()):
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
    selected=positions
    if options['balance_train']:
        amount=int(counts.min());rng=np.random.default_rng(options['seed']);selected=[]
        for label in (0,1):
            pos=positions[meta.iloc[positions].label.eq(label).to_numpy()]
            selected.extend(_direct_stratified_positions(pos,meta.iloc[pos].label_group,amount,rng))
    selected=np.sort(selected);subset=meta.iloc[selected]
    weights=np.ones(len(selected),dtype=float)
    if options['balance_train'] and options['event_weights']:
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
    for beta,name in ((.5,'f05'),(1,'f1'),(2,'f2')):
        denominator=beta*beta*table.precision+table.recall
        table[name]=(1+beta*beta)*table.precision*table.recall/denominator.replace(0,np.nan)
        table[name]=table[name].fillna(0.)
    table['allowed']=True
    table['selection_valid']=pd.Series(y).nunique()==2
    npos=int(np.asarray(y).sum());nneg=len(y)-npos
    table['tp']=np.rint(table.recall*npos).astype(int)
    # Explicit >= tie handling for leaf-constant CatBoost predictions.
    predicted=len(scores)-np.searchsorted(np.sort(np.asarray(scores)),table.threshold.to_numpy(),side='left')
    table['fp']=predicted-table.tp;table['fn']=npos-table.tp;table['tn']=nneg-table.fp
    if options['minimum_precision'] is not None:
        table['allowed']=table.precision.ge(options['minimum_precision'])
    allowed=table.loc[table.allowed&table.selection_valid]
    if allowed.empty:
        return .5,table,'validation_not_evaluable_or_constraint_unmet: фиксированный 0.5'
    chosen=allowed.sort_values([options['threshold_metric'],'precision','threshold'],ascending=False).iloc[0]
    return float(chosen.threshold),table,'selected_on_validation_'+options['threshold_metric']


def direct_thresholds(package):
    """Four decisions from the saved validation PR curve, including old artifacts."""
    table=package['threshold_table'].copy()
    for beta,name in ((.5,'f05'),(1,'f1'),(2,'f2')):
        denominator=beta*beta*table.precision+table.recall
        table[name]=((1+beta*beta)*table.precision*table.recall/denominator.replace(0,np.nan)).fillna(0.)
    if 'tp' not in table and 'split_summary' in package:
        val=package['split_summary'].loc[lambda d:d.partition.eq('validation')]
        if len(val):
            pos=int(val.positive.iloc[0]);neg=int(val.negative.iloc[0]);table['tp']=np.rint(table.recall*pos)
            table['fp']=(table.tp/table.precision.replace(0,np.nan)-table.tp).fillna(0).round()
            table['fn']=pos-table.tp;table['tn']=neg-table.fp
    if {'tp','tn','fp','fn'}.issubset(table):
        for beta,name in ((.5,'f05'),(1,'f1'),(2,'f2')):
            positive=(1+beta**2)*table.tp/((1+beta**2)*table.tp+beta**2*table.fn+table.fp).replace(0,np.nan)
            negative=(1+beta**2)*table.tn/((1+beta**2)*table.tn+beta**2*table.fp+table.fn).replace(0,np.nan)
            table['macro_'+name]=(positive.fillna(0)+negative.fillna(0))/2
    allowed=table.loc[table['allowed'].eq(True)] if 'allowed' in table else table
    if 'selection_valid' in allowed:allowed=allowed.loc[allowed.selection_valid]
    rules={'default_05':dict(threshold=.5,title='Стандартный порог 0.5',status='fixed')}
    for metric,title in [('f1','Максимум F1'),('f2','Максимум F2 · приоритет Recall'),
                         ('f05','Максимум F0.5 · приоритет Precision')]:
        row=allowed.sort_values([metric,'precision','threshold'],ascending=False).iloc[0] if len(allowed) else None
        rules['best_'+metric]=dict(threshold=float(row.threshold) if row is not None else .5,
            title=title,category='positive_class',status='validation_'+metric if row is not None else 'validation_constraint_unmet_fallback_05')
    if package.get('options',{}).get('macro_thresholds',False):
        for metric,title in [('macro_f1','Максимум macro F1 · оба класса'),('macro_f2','Максимум macro F2 · оба класса'),('macro_f05','Максимум macro F0.5 · оба класса')]:
            row=allowed.sort_values([metric,'precision','threshold'],ascending=False).iloc[0] if len(allowed) and metric in allowed else None
            rules['best_'+metric]=dict(threshold=float(row.threshold) if row is not None else .5,title=title,
                category='both_classes',status='validation_'+metric if row is not None else 'validation_not_evaluable_fallback_05')
    rules.update(package.get('alarm_thresholds',{}))
    return rules,table


def direct_metrics(y,scores,threshold,predictions=None):
    from sklearn.metrics import (precision_score,recall_score,f1_score,fbeta_score,
        average_precision_score,roc_auc_score,log_loss,brier_score_loss,confusion_matrix,accuracy_score)
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float)
    if not len(y):return dict(rows=0,status='no_evaluable_rows',threshold=float(threshold))
    pred=scores>=threshold if predictions is None else np.asarray(predictions,dtype=bool)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(rows=len(y),negative=int((y==0).sum()),positive=int(y.sum()),prevalence=float(y.mean()),threshold=float(threshold),
        precision=float(precision_score(y,pred,zero_division=0)) if pred.any() else None,
        recall=float(recall_score(y,pred,zero_division=0)) if y.any() else None,
        f05=float(fbeta_score(y,pred,beta=.5,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),f2=float(fbeta_score(y,pred,beta=2,zero_division=0)),
        macro_f05=float(fbeta_score(y,pred,beta=.5,average='macro',labels=[0,1],zero_division=0)),
        macro_f1=float(fbeta_score(y,pred,beta=1,average='macro',labels=[0,1],zero_division=0)),
        macro_f2=float(fbeta_score(y,pred,beta=2,average='macro',labels=[0,1],zero_division=0)),
        specificity=float(tn/(tn+fp)) if tn+fp else None,
        average_precision=float(average_precision_score(y,scores)) if len(np.unique(y))==2 else None,
        roc_auc=float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
        random_roc_auc=.5,random_average_precision=float(y.mean()),
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


def direct_alarm_evaluate(predictions,contexts,options,threshold,step='1h',as_of=None,start=None,end=None):
    """Causal confirmation and repeat suppression, including gaps/unknown outcomes."""
    from monitoring_v3.alarm_scores import alarm_stream
    from monitoring_v3.event_registry import event_type_at
    policy=dict(step=step,persist_hours=options['alarm_persist_hours'],merge_hours=options['alarm_merge_hours'])
    parts={};events=[];tp_groups=set();false=uncertain=positive_notifications=0;background=active_background=0
    for unit in predictions.index.get_level_values('unit').unique():
        p=predictions.xs(unit,level='unit').sort_index().copy()
        local_threshold=(threshold.xs(unit,level='unit').reindex(p.index)
            if isinstance(threshold,pd.Series) and isinstance(threshold.index,pd.MultiIndex) else threshold)
        binary=pd.Series(np.where(p.score.notna(),np.where(p.score.ge(local_threshold),2.,0.),np.nan),index=p.index)
        stream,_=alarm_stream(binary,policy)
        p['alarm_active']=stream.active;p['alarm_notification']=stream.notification
        known=p.label.notna()
        if as_of is not None:known&=pd.to_datetime(p.label_known_at).le(pd.Timestamp(as_of))
        label=p.label.where(known)
        bg=label.eq(0);positive=label.eq(1)&p.alarm_notification
        false+=int((bg&p.alarm_notification).sum());uncertain+=int((~known&p.alarm_notification).sum())
        positive_notifications+=int(positive.sum());tp_groups.update(p.loc[positive,'label_group'].astype(str))
        background+=int(bg.sum());active_background+=int((bg&p.alarm_active).sum())
        if unit in contexts:
            c=contexts[unit];date=pd.Timestamp(as_of) if as_of is not None else c['frame'].index[-1]
            left=pd.Timestamp(start) if start is not None else p.index.min()
            right=pd.Timestamp(end) if end is not None else p.index.max()+pd.Timedelta(step)
            for e in c['events'].itertuples():
                kind,available=event_type_at(e,date)
                if kind!='Авария' or pd.isna(e.stop) or not left<=e.stop<right or pd.isna(available) or available>date:continue
                window=(p.index>=e.stop-pd.Timedelta(hours=options['horizon_hours']))&(p.index<e.stop)
                belongs=p.label_group.astype(str).str.endswith(str(e.event_id))
                notices=p.loc[window&belongs&positive]
                observed=p.loc[window&belongs&known]
                events.append(dict(unit=unit,event_id=e.event_id,stop=e.stop,detected=bool(len(notices)),
                    scored_points=len(observed),full_window_in_period=e.stop-pd.Timedelta(hours=options['horizon_hours'])>=left,
                    first_warning=notices.index[0] if len(notices) else pd.NaT,
                    lead_hours=float((e.stop-notices.index[0])/pd.Timedelta('1h')) if len(notices) else np.nan))
        parts[unit]=p
    combined=pd.concat(parts,names=['unit','time']).swaplevel().sort_index() if parts else predictions.copy()
    combined.index.names=['time','unit'];event_table=pd.DataFrame(events)
    hours=background*float(pd.Timedelta(step)/pd.Timedelta('1h'))
    recall=float(event_table.detected.mean()) if len(event_table) else None
    precision=len(tp_groups)/(len(tp_groups)+false) if len(tp_groups)+false else None
    metrics=dict(events=len(event_table),detected_events=int(event_table.detected.sum()) if len(event_table) else 0,
        event_recall=recall,notification_precision=precision,false_notifications=false,
        true_notifications=len(tp_groups),duplicate_notifications=positive_notifications-len(tp_groups),
        uncertain_notifications=uncertain,background_hours=hours,
        false_alarms_per_720h=false*720/hours if hours else None,
        background_alarm_fraction=active_background/background if background else None,
        unscored_events=int(event_table.scored_points.eq(0).sum()) if len(event_table) else 0,
        median_lead_hours=float(event_table.lead_hours.median()) if len(event_table) and event_table.lead_hours.notna().any() else None)
    metrics['event_f1']=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.
    return metrics,combined,event_table


def direct_alarm_thresholds(predictions,contexts,options,step,as_of,start,end):
    scores=predictions.score.dropna().to_numpy()
    if not len(scores):return {},pd.DataFrame()
    candidates=np.unique(np.r_[np.quantile(scores,np.linspace(0,1,int(options['alarm_candidates']))),np.nextafter(scores.max(),np.inf)])
    rows=[]
    for threshold in candidates:
        metrics,_,_=direct_alarm_evaluate(predictions,contexts,options,float(threshold),step,as_of,start,end)
        rows.append(dict(threshold=float(threshold),**metrics))
    table=pd.DataFrame(rows)
    table['allowed']=(table.false_alarms_per_720h.le(options['alarm_budget'])&
        table.background_alarm_fraction.le(options['alarm_background_fraction']))
    rules={}
    for name,objective,title in [('alarm_budget_recall','event_recall','Эпизоды: максимум Recall при бюджете'),
        ('alarm_budget_f1','event_f1','Эпизоды: максимум F1 при бюджете')]:
        allowed=table.loc[table.allowed&table[objective].notna()&table.events.gt(0)]
        selected=allowed.sort_values([objective,'false_notifications','threshold'],ascending=[False,True,False]).iloc[0] if len(allowed) else None
        rules[name]=dict(threshold=float(selected.threshold) if selected is not None else float(candidates[-1]),
            title=title,category='alarm_policy',status=('selected_on_validation_alarm_budget' if selected.detected_events>0 else 'constraints_met_but_no_events_detected') if selected is not None else 'validation_budget_not_evaluable_no_notifications',
            validation=dict(selected.to_dict()) if selected is not None else {})
    return rules,table


def direct_fit(features,meta,schema,options,folder,verbose=100,contexts=None,cfg=None):
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
    signature=hashlib.sha256(json.dumps(dict(contract=3,options=options,boundaries=boundaries,
        alarm_step=(cfg or {}).get('step','1h'),
        event_registry={u:c['events'].to_json(date_format='iso') for u,c in (contexts or {}).items()} if options['alarm_selection'] else {},
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
    validation=features.loc[masks['validation']]
    model=CatBoostClassifier(iterations=int(options['iterations']),depth=int(options['depth']),
        learning_rate=options['learning_rate'],loss_function='Logloss',eval_metric=('PRAUC' if meta.loc[masks['validation'],'label'].nunique()==2 else 'Logloss'),
        random_seed=options['seed'],thread_count=int(options['threads']),has_time=True,
        allow_writing_files=False)
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
        dropped_columns=dropped,fingerprint=token,score_status=('uncalibrated_after_balanced_training'
            if options['balance_train'] else 'uncalibrated_natural_training'),
        available_at=boundaries['validation_end'],training_rows=len(positions),
        train_prevalence=float(meta.loc[masks['train'],'label'].mean()),
        split_summary=partitions,balance_summary=balance,threshold_table=threshold_table,
        validation_scores=validation_scores)
    times=pd.DatetimeIndex(meta.index.get_level_values('time'))
    vquery=meta.eligible&(times>=pd.Timestamp(boundaries['train_end']))&(times<pd.Timestamp(boundaries['validation_end'])-pd.Timedelta(hours=options['window_hours']))
    vp=meta.loc[vquery].copy()
    vp['score']=model.predict_proba(Pool(direct_native(features.loc[vquery],columns,categorical),cat_features=categorical),thread_count=options['threads'])[:,1]
    package['validation_predictions']=vp
    package['alarm_step']=(cfg or {}).get('step','1h')
    if options['alarm_selection'] and contexts is not None:
        package['alarm_thresholds'],package['alarm_search']=direct_alarm_thresholds(vp,contexts,options,package['alarm_step'],
            boundaries['validation_end'],boundaries['train_end'],boundaries['validation_end'])
    package['thresholds'],_=direct_thresholds(package)
    joblib.dump(package,cache,compress=3)
    # Predict known test examples and the unlabelled tail without using their outcomes.
    inference=meta.eligible&(times>=pd.Timestamp(boundaries['validation_end']))
    if options['test_end']:inference&=times<pd.Timestamp(options['test_end'])
    test_scores=(model.predict_proba(Pool(direct_native(features.loc[inference],columns,categorical),
        cat_features=categorical),thread_count=options['threads'])[:,1] if inference.any() else np.array([]))
    predictions=meta.loc[inference].copy();predictions['score']=test_scores
    predictions['prediction_05']=test_scores>=.5;predictions['prediction_tuned']=test_scores>=threshold
    predictions['evaluation_eligible']=masks['test'].reindex(predictions.index).fillna(False)
    predictions['model_available_at']=pd.Timestamp(package['available_at'])
    rows=[]
    for name,labels,scores in [('validation',meta.loc[masks['validation'],'label'],validation_scores),
        ('test',predictions.loc[predictions.evaluation_eligible,'label'],predictions.loc[predictions.evaluation_eligible,'score'])]:
        for label,rule in package['thresholds'].items():
            rows.append(dict(partition=name,decision=label,**direct_metrics(labels,scores,rule['threshold'])))
    tested=predictions.loc[predictions.evaluation_eligible]
    for name,values in [('constant_train_prevalence',np.full(len(tested),package['train_prevalence'])),
        ('random_uniform_guess',np.random.default_rng(options['seed']).random(len(tested))),('always_alarm',np.ones(len(tested)))]:
        rows.append(dict(partition='test',decision=name,**direct_metrics(tested.label,values,.5)))
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
    if 'alarm_search' in package:package['alarm_search'].to_csv(folder/'alarm_threshold_search.csv',index=False,encoding='utf-8-sig')
    (folder/'settings.json').write_text(json.dumps(dict(options=package['options'],boundaries=package['boundaries'],
        threshold=package['threshold'],threshold_status=package['threshold_status'],thresholds=package.get('thresholds'),
        feature_columns=package['columns'],categorical_columns=package['categorical'],
        dropped_columns=package['dropped_columns'],score_status=package['score_status'],
        available_at=package['available_at'],alarm_step=package.get('alarm_step','1h')),ensure_ascii=False,indent=2),encoding='utf-8')


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


def direct_event_report(package,contexts,threshold=None):
    from monitoring_v3.event_registry import event_type_at
    rows=[];predictions=package['predictions'];left=pd.Timestamp(package['available_at'])
    horizon=pd.Timedelta(hours=package['options']['horizon_hours'])
    for unit,context in contexts.items():
        part=predictions.xs(unit,level='unit') if unit in predictions.index.get_level_values('unit') else pd.DataFrame()
        right=context['frame'].index[-1]
        for event in context['events'].itertuples():
            kind,known=event_type_at(event,right)
            if kind!='Авария' or pd.isna(event.stop) or not left<=event.stop<=right:continue
            if package['options'].get('test_end') and event.stop>=pd.Timestamp(package['options']['test_end']):continue
            window=part.loc[(part.index>=event.stop-horizon)&(part.index<event.stop)] if len(part) else part
            window=window.loc[window.evaluation_eligible&window.label.eq(1)] if len(window) else window
            if len(window):window=window.loc[window.label_group.astype(str).str.endswith(str(event.event_id))]
            warnings=window.loc[window.score.ge(package['threshold'] if threshold is None else threshold)] if len(window) else window
            rows.append(dict(unit=unit,event_id=event.event_id,stop=event.stop,
                scored_points=len(window),window_complete_in_test=event.stop-horizon>=left,
                detected=bool(len(warnings)),first_warning=warnings.index[0] if len(warnings) else pd.NaT,
                lead_hours=float((event.stop-warnings.index[0])/pd.Timedelta('1h')) if len(warnings) else np.nan))
    return pd.DataFrame(rows,columns=['unit','event_id','stop','scored_points','window_complete_in_test','detected','first_warning','lead_hours'])


def direct_table_html(title,frame,opened=False):
    if frame is None or frame.empty:content='<p>Нет данных.</p>'
    else:content='<div style="overflow:auto;max-height:520px">'+frame.to_html(
        index=False,escape=True,na_rep='—',float_format=lambda v:f'{v:.5g}')+'</div>'
    return f'<details{" open" if opened else ""}><summary><b>{escape(title)}</b></summary>{content}</details>'


def _direct_figure_html(figure,path):
    """Always embed a real PNG, even with an Agg/non-inline matplotlib backend."""
    import matplotlib.pyplot as plt
    buffer=BytesIO()
    try:
        figure.savefig(buffer,format='png',dpi=120,bbox_inches='tight')
        data=buffer.getvalue();Path(path).write_bytes(data)
        return '<img style="max-width:100%;height:auto" src="data:image/png;base64,'+base64.b64encode(data).decode('ascii')+'" />'
    finally:plt.close(figure)


def direct_nonzero_importance(package):
    importance=package['importance'].copy()
    values=pd.to_numeric(importance.importance,errors='coerce')
    return importance.loc[values.notna()&values.ne(0)].sort_values('importance',ascending=False)


def direct_report_summary(package,contexts,model_name='model'):
    rules,_=direct_thresholds(package);rows=[];predictions=package['predictions']
    for unit in predictions.index.get_level_values('unit').unique():
        same=predictions.index.get_level_values('unit')==unit;part=predictions.loc[same]
        known=part.evaluation_eligible;test=part.loc[known]
        if test.empty:
            rows.append(dict(model=model_name,unit=unit,decision='no_evaluable_test',rows=0));continue
        for key,rule in rules.items():
            threshold=rule['threshold'];decision=None
            if rule.get('category')=='alarm_policy':
                event_metrics,stream,_=direct_alarm_evaluate(part,{unit:contexts[unit]} if unit in contexts else {},
                    package['options'],threshold,package.get('alarm_step','1h'),
                    start=package['available_at'],end=package['options'].get('test_end'))
                decision=stream.loc[test.index,'alarm_active'].to_numpy()
            else:
                events=direct_event_report(package,{unit:contexts[unit]} if unit in contexts else {},threshold)
                event_metrics=dict(events=len(events),detected_events=int(events.detected.sum()) if len(events) else 0,
                    event_recall=float(events.detected.mean()) if len(events) else None,
                    unscored_events=int(events.scored_points.eq(0).sum()) if len(events) else 0,
                    median_lead_hours=float(events.lead_hours.median()) if len(events) and events.lead_hours.notna().any() else None)
            rows.append(dict(model=model_name,unit=unit,decision=key,threshold_status=rule['status'],
                **direct_metrics(test.label,test.score,threshold,decision),**event_metrics))
        unit_seed=int(hashlib.sha256(str(unit).encode()).hexdigest()[:8],16)
        for label,values in [('random_uniform_guess',np.random.default_rng(int(package['options']['seed'])+unit_seed).random(len(test))),
            ('constant_train_prevalence',np.full(len(test),package['train_prevalence'])),('always_alarm',np.ones(len(test)))]:
            rows.append(dict(model=model_name,unit=unit,decision=label,**direct_metrics(test.label,values,.5)))
    return pd.DataFrame(rows)


def direct_report_html(package,contexts,folder,model_name=None):
    """Folded per-boiler test reports; thresholds are selected on validation only."""
    from sklearn.metrics import classification_report,ConfusionMatrixDisplay,PrecisionRecallDisplay,RocCurveDisplay
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model_name=model_name or folder.name
    rules,curve=direct_thresholds(package)
    predictions=package['predictions'];rows=[];blocks=[];event_tables=[]
    number=lambda v:'—' if v is None or not np.isfinite(v) else f'{v:.4f}'
    importance=direct_nonzero_importance(package)
    # Group-level optional diagnostics are rendered once, never a validation metrics table.
    fig,axes=plt.subplots(1,2,figsize=(13,3.5))
    for name in ('precision','recall','f05','f1','f2'):axes[0].plot(curve.threshold,curve[name],label=name)
    for rule in list(rules.values())[1:]:axes[0].axvline(rule['threshold'],ls=':',alpha=.5)
    axes[0].set_title('Подбор порогов: только validation');axes[0].legend(fontsize=8)
    for name,values in package.get('learning_curve',{}).items():
        if 'Logloss' in values:axes[1].plot(values['Logloss'],label=name)
    axes[1].set_title('Logloss по итерациям');axes[1].legend();fig.tight_layout()
    diagnostics=_direct_figure_html(fig,folder/'validation_threshold.png')
    importance_html=direct_table_html('Важность признаков: только ненулевые',importance)
    if len(importance):
        top=importance.head(25).sort_values('importance')
        fig,ax=plt.subplots(figsize=(12,8));ax.barh(top.feature,top.importance)
        ax.set_title('25 наиболее важных признаков · общая модель '+model_name)
        fig.tight_layout();importance_html+='<details><summary>График важности признаков</summary>'+_direct_figure_html(fig,folder/'feature_importance.png')+'</details>'
    groups=package.get('balance_summary',pd.DataFrame())
    counts={int(r.label):int(r.rows) for r in groups.itertuples()} if len(groups) else {}
    balanced=package['options'].get('balance_train',True)
    training_note=('Сохранённая модель обучалась с балансировкой. Для обучения без неё повторите пункт 5.'
        if balanced else 'Обучение без балансировки и классовых весов; validation и test также не балансируются.')
    training_note+=f' В фактическом train группы: класс 0 — {counts.get(0,"—")}, класс 1 — {counts.get(1,"—")}.'
    for unit in sorted(predictions.index.get_level_values('unit').unique()):
        part=predictions.xs(unit,level='unit')
        test=part.loc[part.evaluation_eligible]
        if test.empty:
            blocks.append(f'<details><summary><b>{escape(unit)} · {escape(model_name)}</b></summary><p>Нет тестовых точек с известным исходом.</p></details>')
            continue
        y=test.label.astype(int);score=test.score
        ranking=direct_metrics(y,score,.5)
        body=f'<p>{escape(training_note)}</p><p><b>Независимый test:</b> {len(test)} часов; '
        body+=f'класс 0: {int(y.eq(0).sum())}; класс 1: {int(y.eq(1).sum())}.</p>'
        body+=f'<p>Входов модели: {len(package["columns"])}; деревьев: {getattr(package.get("model"),"tree_count_","—")}; '
        body+='доступна с '+escape(str(package['available_at']))+'.</p>'
        body+=f'<p><b>ROC AUC:</b> {number(ranking["roc_auc"])} · случайный ориентир: 0.5000<br>'
        body+=f'<b>Average Precision:</b> {number(ranking["average_precision"])} · случайный ориентир: {y.mean():.4f} (доля положительных).</p>'
        body+='<p>Пороги подбираются на validation модели; данные этого теста в подборе не участвовали. Если подбор невозможен, причина и fallback подписаны под вариантом.</p>'
        if y.nunique()<2:body+='<p><b>В тесте один класс: ROC AUC и AP здесь не оцениваются.</b></p>'
        opened_category=None
        for key,rule in rules.items():
            category=rule.get('category','positive_class')
            extra=category in ('both_classes','alarm_policy')
            if opened_category and opened_category!=category:body+='</details>';opened_category=None
            if extra and opened_category!=category:
                title='Дополнительно: F по обоим классам (macro)' if category=='both_classes' else 'Дополнительно: устойчивые эпизоды и бюджет ложных тревог'
                body+='<details><summary><b>'+title+'</b></summary>';opened_category=category
            threshold=rule['threshold'];pred=score.ge(threshold)
            policy_events=None;policy_metrics=None
            if category=='alarm_policy':
                unit_predictions=predictions.loc[predictions.index.get_level_values('unit')==unit]
                policy_metrics,stream,policy_events=direct_alarm_evaluate(unit_predictions,{unit:contexts[unit]} if unit in contexts else {},
                    package['options'],threshold,package.get('alarm_step','1h'),start=package['available_at'],end=package['options'].get('test_end'))
                pred=stream.xs(unit,level='unit').alarm_active.reindex(test.index)
            report=classification_report(y,pred,labels=[0,1],target_names=['Нет аварии','Авария'],digits=3,zero_division=0)
            fig,ax=plt.subplots(figsize=(4.5,3.7))
            ConfusionMatrixDisplay.from_predictions(y,pred,labels=[0,1],display_labels=['Нет аварии','Авария'],ax=ax,colorbar=False)
            ax.set_title(f'{unit} · порог {threshold:.5f}');fig.tight_layout()
            matrix=_direct_figure_html(fig,folder/(unit+'__'+key+'_confusion.png'))
            body+=f'<h4>{escape(rule["title"])} · порог {threshold:.6f}</h4>'
            if any(word in rule['status'] for word in ('fallback','not_evaluable','no_events_detected')):
                body+='<p><b>Ограничение подбора:</b> '+escape(rule['status'])+'. Смотрите наличие обоих классов/аварий и допустимого фона в validation.</p>'
            body+='<div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">'
            body+='<pre style="overflow:auto">'+escape(report)+'</pre><div style="max-width:480px">'+matrix+'</div></div>'
            if policy_metrics is not None:
                rate=policy_metrics['false_alarms_per_720h'];fraction=policy_metrics['background_alarm_fraction']
                body+='<p><b>Эпизоды на test:</b> ложных уведомлений '+str(policy_metrics['false_notifications'])
                body+='; на 720 фоновых часов '+number(rate)+'; доля фона в тревоге '+number(fraction)+'. '
                body+='Ограничения validation: '+number(package['options']['alarm_budget'])+' / 720 ч и '
                body+=number(package['options']['alarm_background_fraction'])+' доли фона. На test они проверяются заново, а не гарантируются.</p>'
            events=(policy_events if policy_events is not None else direct_event_report(package,{unit:contexts[unit]} if unit in contexts else {},threshold))
            event_tables.append(events.assign(decision=key,threshold=threshold))
            metric=direct_metrics(y,score,threshold)
            metric.update(model=model_name,unit=unit,decision=key,
                events=len(events),detected_events=int(events.detected.sum()) if len(events) else 0,
                event_recall=float(events.detected.mean()) if len(events) else None,
                unscored_events=int(events.scored_points.eq(0).sum()) if len(events) else 0,
                median_lead_hours=float(events.lead_hours.median()) if len(events) and events.lead_hours.notna().any() else None)
            rows.append(metric)
            if len(events):
                body+=f'<p>Аварий обнаружено: {metric["detected_events"]}/{len(events)}; без пригодных точек: {metric["unscored_events"]}.</p>'
                body+=direct_table_html('Отдельные аварии · '+rule['title'],events)
        if opened_category:body+='</details>'
        # A deterministic fair-coin baseline and a constant training-prior baseline.
        unit_seed=int(hashlib.sha256(str(unit).encode()).hexdigest()[:8],16)
        random_scores=np.random.default_rng(int(package['options']['seed'])+unit_seed).random(len(test))
        for label,values in [('random_uniform_guess',random_scores),
            ('constant_train_prevalence',np.full(len(test),package['train_prevalence']))]:
            rows.append(dict(model=model_name,unit=unit,decision=label,**direct_metrics(y,values,.5)))
        body+='<p><b>Baseline:</b> в итоговой таблице есть случайное угадывание '
        body+='(равномерные случайные баллы, порог 0.5, фиксированный seed) и постоянный прогноз по доле аварий в train.</p>'
        if y.nunique()==2:
            fig,axes=plt.subplots(1,2,figsize=(12,3.8))
            PrecisionRecallDisplay.from_predictions(y,score,ax=axes[0],name='CatBoost')
            axes[0].axhline(y.mean(),ls='--',color='grey',label='Случайный ориентир');axes[0].legend()
            RocCurveDisplay.from_predictions(y,score,ax=axes[1],name='CatBoost')
            axes[1].plot([0,1],[0,1],'--',color='grey',label='Случайный ориентир');axes[1].legend()
            fig.tight_layout();body+=_direct_figure_html(fig,folder/(unit+'__test_metrics.png'))
        fig,ax=plt.subplots(figsize=(14,3.8));ax.plot(part.index,part.score,lw=.7,label='Балл риска')
        for key,rule in rules.items():
            if rule.get('category') not in ('both_classes','alarm_policy'):
                ax.axhline(rule['threshold'],ls='--',lw=1,alpha=.8,label=rule['title'])
        events=direct_event_report(package,{unit:contexts[unit]} if unit in contexts else {})
        for stop in events.stop:ax.axvline(stop,color='darkred',alpha=.6)
        unknown=part.loc[~part.evaluation_eligible]
        if len(unknown):ax.scatter(unknown.index,unknown.score,s=3,color='grey',label='Исход неизвестен / исключён')
        ax.set_ylim(-.02,1.02);ax.set_title(unit+' · test и последующая история · балл риска')
        ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        body+=_direct_figure_html(fig,folder/(unit+'_risk.png'))
        blocks.append(f'<details class="direct-boiler"><summary style="cursor:pointer;font-size:17px"><b>{escape(unit)} · {escape(model_name)} · '
            f'ROC AUC {number(ranking["roc_auc"])} · AP {number(ranking["average_precision"])}</b></summary>{body}</details>')
    blocks.append('<details><summary><b>Дополнительно · '+escape(model_name)+'</b></summary>'+
        '<details><summary>Validation: пороги и кривая обучения</summary>'+diagnostics+'</details>'+importance_html+'</details>')
    summary=direct_report_summary(package,contexts,model_name)
    if len(summary):
        first=['unit','model','decision','threshold','rows','negative','positive','precision','recall','f05','f1','f2',
            'roc_auc','random_roc_auc','average_precision','random_average_precision']
        summary=summary[[c for c in first if c in summary]+[c for c in summary if c not in first]]
    summary.to_csv(folder/'test_summary_by_boiler.csv',index=False,encoding='utf-8-sig')
    if event_tables:pd.concat(event_tables,ignore_index=True).to_csv(folder/'test_events.csv',index=False,encoding='utf-8-sig')
    importance.to_csv(folder/'feature_importance_nonzero.csv',index=False,encoding='utf-8-sig')
    html='<div class="direct-risk-report">'+''.join(blocks)+'</div>'
    (folder/'report.html').write_text('<meta charset="utf-8">'+html,encoding='utf-8')
    return html,summary


def direct_standard_report(package,contexts,folder,model_name=None):
    from IPython.display import display,HTML
    html,summary=direct_report_html(package,contexts,folder,model_name)
    display(HTML(html))
    return summary
