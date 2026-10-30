"""Sequential, disk-cached direct-risk experiments and rolling temporal checks."""
import gc
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

try:
    direct_options
except NameError:
    from direct_risk import (direct_options,direct_features,direct_fit,direct_thresholds,
        direct_report_summary,direct_standard_report,direct_metrics,direct_alarm_evaluate,
        direct_table_html,direct_predict,direct_report_html,direct_split,direct_plot_segments,_direct_figure_html)


def _experiment_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temporary.replace(path)


def trim_direct_contexts(contexts):
    """Free large legacy regression-only matrices after the source audit is saved."""
    for c in contexts.values():
        for field in ('features','quality','input_quality','changes'):
            c.pop(field,None)
    gc.collect()


def prepare_direct_cache(contexts,cfg,options,root,progress=print):
    """One float32 matrix per boiler, eligible hours only; never a fleet-wide union."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);entries={};summaries=[]
    feature_keys=('window_hours','lags_hours','trends_hours','delta_hours','switch_hours',
        'include_since_start','include_startup','minimum_numeric_coverage')
    for unit,c in contexts.items():
        signature=hashlib.sha256(json.dumps(dict(contract=4,options={k:options[k] for k in feature_keys},
            step=cfg['step'],roles=c['roles'],units=c.get('declared_units',{}),
            reference=str(c.get('preparation_reference_end')),threshold_source=c.get('threshold_source'),
            evidence=c.get('regime_spec',{})),sort_keys=True,default=str).encode())
        for frame in (c['cleaned'][[s for s,r in c['roles'].items() if r in ('load','condition','target')]],
                      c['frame'][[s for s,r in c['roles'].items() if r=='state']],c['regime'][['mode','since_start_hours']]):
            signature.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
        folder=root/unit/signature.hexdigest()[:24];folder.mkdir(parents=True,exist_ok=True)
        marker=folder/'complete.json'
        required=['features.parquet','metadata.parquet','schema.csv','profile.csv','preview.parquet','tail.parquet']
        if marker.exists() and all((folder/name).exists() for name in required):
            row=json.loads(marker.read_text(encoding='utf-8'));progress(unit+': признаки восстановлены из кеша')
        else:
            progress(unit+': подготовка признаков только пригодных часов')
            x,meta,schema=direct_features(c,cfg,options,eligible_only=True)
            profile=schema.merge(pd.DataFrame(dict(feature=x.columns,dtype=x.dtypes.astype(str).to_numpy(),
                missing_fraction=x.isna().mean().to_numpy(),unique_values=x.nunique(dropna=True).to_numpy())),on='feature')
            x.to_parquet(folder/'features.parquet');meta.to_parquet(folder/'metadata.parquet')
            schema.to_csv(folder/'schema.csv',index=False,encoding='utf-8-sig')
            profile.to_csv(folder/'profile.csv',index=False,encoding='utf-8-sig')
            x.head(5).to_parquet(folder/'preview.parquet')
            x.tail(1).to_parquet(folder/'tail.parquet')
            row=dict(unit=unit,source_rows=len(meta),eligible_rows=len(x),features=len(x.columns),
                numeric_sensors=sum(r in ('target','condition','load') for r in c['roles'].values()),
                categorical_sensors=sum(r=='state' for r in c['roles'].values()),
                matrix_mb=float(x.memory_usage(deep=True).sum()/1024**2),folder=str(folder.resolve()))
            _experiment_json(marker,row);del x,meta,schema,profile;gc.collect()
        entries[unit]={**row,'folder':str(folder.resolve())};summaries.append(entries[unit])
    return entries,pd.DataFrame(summaries)


def cached_direct_labels(context,entry,cfg,options):
    from monitoring_v3.interpretation import risk_labels
    c=context;folder=Path(entry['folder'])
    signature=hashlib.sha256(json.dumps(dict(horizon=options['horizon_hours'],lead=options['min_lead_hours'],
        testing=hashlib.sha256(pd.util.hash_pandas_object(c['frame'].get('_Тип',pd.Series('',index=c['frame'].index)),index=True).to_numpy().tobytes()).hexdigest(),
        events=c['events'].to_json(date_format='iso'),step=cfg['step']),sort_keys=True).encode()).hexdigest()[:20]
    target=folder/f'labels_{options["horizon_hours"]}h_{signature}.parquet'
    audit_path=target.with_suffix('.json')
    if target.exists() and audit_path.exists():return pd.read_parquet(target),json.loads(audit_path.read_text(encoding='utf-8'))
    meta=pd.read_parquet(folder/'metadata.parquet');source_rows=len(meta);meta=meta.loc[meta.eligible].copy()
    testing=c['frame'].get('_Тип',pd.Series('',index=c['frame'].index)).reindex(meta.index).isin(['ИСП','ЗРР'])
    as_of=c['frame'].index[-1]
    y,g,audit,known=risk_labels(meta.index[~testing],c['events'],options['horizon_hours'],as_of,as_of,
        observed=c['cleaned'][c['load']].notna(),min_lead_hours=options['min_lead_hours'],
        return_audit=True,return_available_at=True)
    meta['label']=y;meta['label_group']=c['unit']+'::'+g.astype(str)
    meta['label_known_at']=known;meta['source_end']=as_of
    audit.update(unit=c['unit'],horizon_hours=options['horizon_hours'],source_rows=source_rows,
        excluded_testing_rows=int(testing.sum()),
        eligible_rows=len(meta),positive_rows=int(y.eq(1).sum()),negative_rows=int(y.eq(0).sum()),
        positive_events=int(g.loc[y.eq(1)].nunique()))
    meta.to_parquet(target);_experiment_json(audit_path,audit)
    return meta,audit


def experiment_families(contexts,cfg,methods):
    if not methods or set(methods)-{'unit','group'}:raise ValueError('TRAINING_METHODS: список unit / group.')
    families=[]
    for method in dict.fromkeys(methods):
        if method=='unit':
            families.extend(dict(method='unit',name=u,units=[u]) for u in contexts)
        else:
            for group,units in cfg['groups'].items():
                members=[u for u in units if u in contexts]
                if not members:continue
                if len(members)==1 and 'unit' in methods:continue
                families.append(dict(method='group',name=group,units=members))
    return families


def experiment_columns(entries,units,variant):
    if variant not in ('numeric','numeric_states','full'):raise ValueError('FEATURE_VARIANTS: numeric, numeric_states, full.')
    schemas=[pd.read_csv(Path(entries[u]['folder'])/'schema.csv',keep_default_na=False) for u in units]
    common=set(schemas[0].feature)
    for schema in schemas[1:]:common&=set(schema.feature)
    selected=schemas[0].loc[schemas[0].feature.isin(common)].copy()
    for schema in schemas[1:]:
        other=schema.set_index('feature');own=selected.set_index('feature')
        for feature in selected.feature:
            if str(own.loc[feature,'units']).strip().casefold()!=str(other.loc[feature,'units']).strip().casefold():
                raise ValueError('Разные физические единицы общего признака: '+feature)
    if variant=='numeric':selected=selected.loc[selected.feature.str.startswith('num::')|selected.feature.eq('unit')]
    elif variant=='numeric_states':
        selected=selected.loc[~selected.feature.eq('since_start_hours')&~selected.feature.str.endswith('::held_hours')]
    if not selected.feature.str.startswith('num::').any():raise ValueError('Нет общих физических числовых каналов для группы.')
    return selected


def load_experiment_features(entries,units,variant,max_mb=2048):
    schema=experiment_columns(entries,units,variant);columns=list(schema.feature)
    ncat=int(schema.kind.eq('category').sum());nnum=len(schema)-ncat
    estimate=sum(entries[u]['eligible_rows'] for u in units)*(nnum*4+ncat*80+64)/1024**2
    if max_mb is not None and estimate>max_mb:
        raise ValueError(f'Оценка активной матрицы {estimate:.0f} МБ > MAX_FEATURE_MATRIX_MB={max_mb}. '
            'Сократите группу/набор признаков или измените лимит при достаточной памяти.')
    parts={}
    for unit in units:
        p=pd.read_parquet(Path(entries[unit]['folder'])/'features.parquet',columns=columns)
        p.index.name='time';parts[unit]=p
    x=pd.concat(parts,names=['unit','time']).swaplevel().sort_index(kind='stable');x.index.names=['time','unit']
    del parts;gc.collect()
    return x,schema,float(x.memory_usage(deep=True).sum()/1024**2)


def rolling_policy(rolling):
    defaults=dict(adaptive_validation=True,max_validation_days=730,validation_step_days=30,
        min_train_events=2,min_validation_events=2,min_validation_negative_groups=2)
    result={k:rolling.get(k,v) for k,v in defaults.items()}
    if not isinstance(result['adaptive_validation'],bool):raise ValueError('adaptive_validation должен быть bool')
    for key in defaults.keys()-{'adaptive_validation'}:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(key+' должен быть целым >= 1')
    if result['max_validation_days']<rolling.get('validation_days',90):
        raise ValueError('max_validation_days меньше начального validation_days')
    return result


def adaptive_direct_fold(meta,options,fold,rolling):
    """Expand validation into the past; never inspect test outcomes to choose dates."""
    policy=rolling_policy(rolling);b=pd.Timestamp(fold['validation_end']);initial=pd.Timestamp(fold['train_end'])
    t=meta.index.get_level_values('time');past=meta.loc[t<b].copy()
    starts=[initial]
    if policy['adaptive_validation']:
        limit=max(b-pd.Timedelta(days=policy['max_validation_days']),t.min())
        left=initial
        while left>limit:
            left=max(limit,left-pd.Timedelta(days=policy['validation_step_days']));starts.append(left)
    best=None;attempts=[]
    for a in starts:
        opts={**options,'train_end':str(a),'validation_end':str(b),'test_end':fold['test_end'],
              'allow_single_class_evaluation':True}
        try:
            _,summary,_=direct_split(past,opts)
            train=summary.loc[summary.partition.eq('train')].iloc[0]
            val=summary.loc[summary.partition.eq('validation')].iloc[0]
            audit=dict(train_rows=int(train.rows),train_positive=int(train.positive),train_negative=int(train.negative),
                train_events=int(train.positive_events),validation_rows=int(val.rows),validation_positive=int(val.positive),
                validation_negative=int(val.negative),validation_events=int(val.positive_events),
                validation_negative_groups=int(val.negative_groups))
            ready=(audit['train_events']>=policy['min_train_events'] and
                   audit['validation_events']>=policy['min_validation_events'] and
                   audit['validation_negative_groups']>=policy['min_validation_negative_groups'] and
                   audit['train_negative']>0 and audit['validation_negative']>0)
            attempt=dict(train_end=str(a),ready=ready,**audit);attempts.append(attempt)
            if best is None or (audit['validation_events'],audit['validation_negative'])>(best['validation_events'],best['validation_negative']):best=attempt
            if ready:
                return {**fold,'nominal_train_end':str(initial),'train_end':str(a),**audit,
                    'validation_days_actual':float((b-a)/pd.Timedelta('1D')),
                    'split_status':'ready' if a==initial else 'validation_expanded',
                    'split_attempts':len(attempts),'split_reason':''}
        except ValueError as exc:attempts.append(dict(train_end=str(a),ready=False,reason=str(exc)))
    chosen=best or {'train_end':str(initial)}
    return {**fold,**{k:v for k,v in chosen.items() if k!='ready'},'nominal_train_end':str(initial),
        'split_status':'insufficient_past_events','split_attempts':len(attempts),
        'split_reason':f'Нет прошлой выборки с train >= {policy["min_train_events"]} авариями и validation >= '
            f'{policy["min_validation_events"]} авариями / {policy["min_validation_negative_groups"]} группами фона. '
            'Тест не использовался; F-порог не подменяется на 0.5.'}


def rolling_direct_folds(contexts,cfg,rolling):
    allowed={'folds','validation_days','test_days','end',*rolling_policy(rolling)}
    if set(rolling)-allowed:raise ValueError('Неизвестные параметры ROLLING: '+str(set(rolling)-allowed))
    count=int(rolling.get('folds',3));validation=int(rolling.get('validation_days',90));test=int(rolling.get('test_days',90))
    if min(count,validation,test)<1:raise ValueError('Число периодов и длительности должны быть положительными.')
    end=pd.Timestamp(rolling['end']) if rolling.get('end') else max(c['frame'].index[-1] for c in contexts.values())+pd.Timedelta(cfg['step'])
    folds=[]
    for i in range(count):
        test_end=end-pd.Timedelta(days=test*(count-1-i));test_start=test_end-pd.Timedelta(days=test)
        validation_start=test_start-pd.Timedelta(days=validation)
        folds.append(dict(fold=f'fold_{i+1:02d}',train_end=str(validation_start),validation_end=str(test_start),test_end=str(test_end)))
    return folds


def run_direct_suite(contexts,cfg,options,entries,methods,variants,horizons,rolling,root,max_mb=2048,progress=print):
    """Only one active feature matrix, horizon label frame, and fitted model in RAM."""
    import joblib
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    marker=root/'evaluation_v3.json'
    if not marker.exists():
        import shutil
        backup=root/'previous_reports_before_v3'
        for name in ('experiment_index.csv','fold_metrics.csv','comparison_rolling.csv','rolling_alarm_metrics.csv','validation_choices.csv','selected_models.json'):
            old=root/name
            if old.exists():
                backup.mkdir(exist_ok=True)
                if not (backup/name).exists():shutil.copy2(old,backup/name)
        _experiment_json(marker,dict(contract=3,models_folder='experiments_v3',selection='past_validation_AP',features='telemetry_only_v1'))
    families=experiment_families(contexts,cfg,methods);folds=rolling_direct_folds(contexts,cfg,rolling)
    if not variants or not horizons:raise ValueError('Задайте непустые списки FEATURE_VARIANTS и HORIZONS.')
    if len(set(horizons))!=len(horizons) or any(h<=options['min_lead_hours'] for h in horizons):raise ValueError('Проверьте HORIZONS: уникальные положительные горизонты больше min_lead_hours.')
    records=[];metrics=[];audits=[];plans={}
    def save():
        pd.DataFrame(records).to_csv(root/'experiment_index.csv',index=False,encoding='utf-8-sig')
        (pd.concat(metrics,ignore_index=True) if metrics else pd.DataFrame(columns=['method','name','variant','horizon_hours','fold','unit','decision','rows'])).to_csv(root/'fold_metrics.csv',index=False,encoding='utf-8-sig')
        if audits:pd.DataFrame(audits).drop_duplicates(['unit','horizon_hours']).to_csv(root/'label_audit.csv',index=False,encoding='utf-8-sig')
    for family in families:
        for variant in dict.fromkeys(variants):
            progress(f"{family['method']} · {family['name']} · {variant}: загрузка одной матрицы")
            try:x,schema,mb=load_experiment_features(entries,family['units'],variant,max_mb)
            except ValueError as exc:
                records.append(dict(**{k:v for k,v in family.items() if k!='units'},variant=variant,status='feature_error',reason=str(exc)));save();continue
            progress(f'Активная матрица: {x.shape}; {mb:.1f} МБ (без суммирования горизонтов/моделей)')
            for horizon in horizons:
                base=direct_options({**options,'horizon_hours':horizon,'allow_single_class_evaluation':True})
                parts={}
                for unit in family['units']:
                    progress(f'{unit}: метки горизонта {horizon} ч (кешируются отдельно от признаков)')
                    parts[unit],audit=cached_direct_labels(contexts[unit],entries[unit],cfg,base);audits.append(audit)
                meta=pd.concat(parts,names=['unit','time']).swaplevel().sort_index(kind='stable').reindex(x.index)
                meta.index.names=['time','unit'];del parts
                experiment=f"{family['method']}__{family['name']}__{variant}__{horizon}h"
                for nominal in folds:
                    plan_key=(family['method'],family['name'],horizon,nominal['fold'])
                    if plan_key not in plans:plans[plan_key]=adaptive_direct_fold(meta,base,nominal,rolling)
                    fold=plans[plan_key]
                    folder=root/'experiments_v3'/experiment/fold['fold'];folder.mkdir(parents=True,exist_ok=True)
                    detail=dict(method=family['method'],name=family['name'],units=family['units'],variant=variant,
                        horizon_hours=horizon,fold=fold['fold'],train_end=fold['train_end'],validation_end=fold['validation_end'],test_end=fold['test_end'])
                    record=dict(**{k:v for k,v in detail.items() if k!='units'},experiment=experiment,
                        folder=str(folder.resolve()),matrix_mb=mb,
                        **{k:v for k,v in fold.items() if k not in detail})
                    progress(experiment+' · '+fold['fold'])
                    opts=direct_options({**base,**{k:fold[k] for k in ('train_end','validation_end','test_end')}})
                    t=meta.index.get_level_values('time')
                    if not (meta.eligible&(t>=pd.Timestamp(fold['validation_end']))&(t<pd.Timestamp(fold['test_end']))).any():
                        records.append(dict(**record,status='no_test_observations',reason='Нет рабочих измерений в этом периоде.'));save();continue
                    if fold['split_status']=='insufficient_past_events':
                        records.append(dict(**record,status='insufficient_past_events',reason=fold['split_reason']));save();continue
                    try:
                        package,_,_,_=direct_fit(x,meta,schema,opts,folder,verbose=100,
                            contexts={u:contexts[u] for u in family['units']},cfg=cfg,readiness=rolling_policy(rolling))
                        package['experiment']=detail;joblib.dump(package,folder/'classifier.joblib',compress=3)
                        summary=direct_report_summary(package,{u:contexts[u] for u in family['units']},experiment)
                        for key in ('method','name','variant','horizon_hours','fold','train_end','validation_end','test_end'):summary[key]=detail[key]
                        summary.to_csv(folder/'test_summary_by_boiler.csv',index=False,encoding='utf-8-sig')
                        metrics.append(summary)
                        selection=package['validation_selection']
                        actual=package['split_summary'].set_index('partition')
                        for partition in ('train','validation'):
                            record[partition+'_rows']=int(actual.loc[partition,'rows'])
                            record[partition+'_positive']=int(actual.loc[partition,'positive'])
                            record[partition+'_negative']=int(actual.loc[partition,'negative'])
                            record[partition+'_events']=int(actual.loc[partition,'positive_events'])
                        record['validation_negative_groups']=int(actual.loc['validation','negative_groups'])
                        tested=package['split_summary'].loc[lambda d:d.partition.eq('test')].iloc[0]
                        records.append(dict(**record,status='complete',features=len(package['columns']),trees=package['model'].tree_count_,
                            validation_average_precision=selection['average_precision'],validation_index_hash=selection['index_hash'],
                            test_rows=int(tested.rows),test_positive=int(tested.positive),test_negative=int(tested.negative),
                            test_events=int(tested.positive_events),
                            test_status=('evaluated' if package['predictions'].evaluation_eligible.any() else 'no_known_test_outcomes'),reason=''))
                        del package,summary;gc.collect()
                    except ValueError as exc:
                        records.append(dict(**record,status='insufficient_or_invalid_data',reason=str(exc)))
                        progress('Пропуск с явной причиной: '+str(exc))
                    save()
                del meta;gc.collect()
            del x,schema;gc.collect()
    save()
    select_validation_variants(pd.DataFrame(records),root)
    return pd.DataFrame(records),pd.concat(metrics,ignore_index=True) if metrics else pd.DataFrame()


def select_validation_variants(index,root=None):
    """Choose a feature set per family/horizon/time using only past validation AP."""
    needed={'status','validation_average_precision','validation_index_hash'}
    if not needed.issubset(index):return pd.DataFrame()
    good=index.loc[index.status.eq('complete')&index.validation_average_precision.notna()].copy()
    rows=[]
    for _,group in good.groupby(['method','name','horizon_hours','fold'],sort=False):
        if group.validation_index_hash.nunique()!=1:continue
        order={'numeric':0,'numeric_states':1,'full':2}
        winner=group.assign(_simplicity=group.variant.map(order)).sort_values(
            ['validation_average_precision','_simplicity'],ascending=[False,True]).iloc[0].drop(labels='_simplicity')
        rows.append({**winner.to_dict(),'selected_variant':winner['variant'],
            'selection_status':'selected_on_validation_AP' if len(group)>1 else 'only_available_candidate',
            'candidate_count':len(group)})
    result=pd.DataFrame(rows)
    if root is not None:
        result.to_csv(Path(root)/'validation_choices.csv',index=False,encoding='utf-8-sig')
        if len(result):
            latest=result.sort_values('validation_end').groupby(['method','name','horizon_hours'],sort=False).tail(1)
            _experiment_json(Path(root)/'selected_models.json',latest[[c for c in (
                'method','name','horizon_hours','selected_variant','folder','validation_end','validation_average_precision',
                'selection_status') if c in latest]].to_dict('records'))
        else:_experiment_json(Path(root)/'selected_models.json',[])
    return result


def aggregate_direct_folds(metrics):
    """Sum disjoint test confusions, but never pool scores of different models for AUC/AP."""
    if metrics.empty:return pd.DataFrame()
    keys=['method','name','variant','horizon_hours','unit','decision'];rows=[]
    good=metrics.loc[pd.to_numeric(metrics.rows,errors='coerce').gt(0)]
    for key,g in good.groupby(keys,dropna=False,sort=False):
        row=dict(zip(keys,key));row.update(folds=g.fold.nunique(),rows=int(g.rows.sum()),
            positive=int(g.positive.sum()),negative=int(g.negative.sum()))
        tp,fp,fn,tn=(float(g[c].sum()) for c in ('true_positive','false_positive','false_negative','true_negative'))
        row.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
            specificity=tn/(tn+fp) if tn+fp else None,predicted_positive_fraction=(tp+fp)/row['rows'])
        for beta,name in ((.5,'f05'),(1,'f1'),(2,'f2')):
            pos=(1+beta**2)*tp/((1+beta**2)*tp+beta**2*fn+fp) if tp+fp+fn else 0.
            neg=(1+beta**2)*tn/((1+beta**2)*tn+beta**2*fp+fn) if tn+fp+fn else 0.
            row[name]=pos;row['macro_'+name]=(pos+neg)/2
        for column in ('roc_auc','average_precision','random_average_precision'):
            values=pd.to_numeric(g[column],errors='coerce');row[column+'_mean']=float(values.mean())
            row[column+'_min']=float(values.min());row[column+'_max']=float(values.max())
        valid_ap=pd.to_numeric(g.average_precision,errors='coerce').notna()
        row['rankable_folds']=int(valid_ap.sum())
        row['random_average_precision_mean']=pd.to_numeric(g.loc[valid_ap,'random_average_precision'],errors='coerce').mean()
        row['ap_lift_mean']=(pd.to_numeric(g.loc[valid_ap,'average_precision'],errors='coerce')/
            pd.to_numeric(g.loc[valid_ap,'random_average_precision'],errors='coerce').replace(0,np.nan)).mean()
        row['event_recall_fold_mean']=pd.to_numeric(g.get('event_recall',pd.Series(dtype=float)),errors='coerce').mean()
        rows.append(row)
    return pd.DataFrame(rows)


def direct_rolling_alarm_summary(index,contexts,root):
    """Run the notification state once over all disjoint causal fold predictions.

    Thresholds may change at fold boundaries; the persistence/repeat state does
    not restart there. No optimisation is performed on these test predictions.
    """
    rows=[]
    complete=index.loc[index.status.eq('complete')]
    for experiment,group in complete.groupby('experiment',sort=False):
        pieces=[];options=None;rules={};step='1h';start=None;end=None
        for record in group.sort_values('validation_end').itertuples():
            folder=Path(record.folder)
            settings=json.loads((folder/'settings.json').read_text(encoding='utf-8'))
            options=settings['options'];step=settings.get('alarm_step','1h')
            p=pd.read_parquet(folder/'test_predictions.parquet')
            time=pd.DatetimeIndex(p.index.get_level_values('time'))
            if not pd.to_datetime(p.model_available_at).le(time).all():raise ValueError('Попытка использовать модель раньше её доступности.')
            for key,rule in (settings.get('thresholds') or {}).items():
                if rule.get('category')=='alarm_policy':
                    p['threshold::'+key]=rule['threshold'];rules[key]=rule
            pieces.append(p)
            a=pd.Timestamp(record.validation_end);b=pd.Timestamp(record.test_end)
            start=a if start is None else min(start,a);end=b if end is None else max(end,b)
        if not pieces:continue
        predictions=pd.concat(pieces).sort_index()
        if predictions.index.has_duplicates:raise ValueError('Тестовые периоды пересеклись: '+experiment)
        for key in rules:
            for unit in predictions.index.get_level_values('unit').unique():
                p=predictions.loc[predictions.index.get_level_values('unit')==unit]
                metric,_,events=direct_alarm_evaluate(p,{unit:contexts[unit]} if unit in contexts else {},options,
                    p['threshold::'+key],step=step,start=start,end=end)
                first=group.iloc[0]
                rows.append(dict(method=first.method,name=first['name'],variant=first.variant,horizon_hours=first.horizon_hours,
                    unit=unit,decision=key,**{'rolling_'+k:v for k,v in metric.items()}))
        del predictions,pieces;gc.collect()
    result=pd.DataFrame(rows)
    result.to_csv(Path(root)/'rolling_alarm_metrics.csv',index=False,encoding='utf-8-sig')
    return result


def render_direct_suite(index,contexts,root,report_folds='none'):
    """Read/render one saved model at a time; no training, no PACKAGES collection."""
    from IPython.display import display,HTML
    import joblib
    index=index.copy();root=Path(root)
    # A relocated output folder must not keep referring to another computer's drive.
    if 'folder' in index:
        for i,row in index.iterrows():
            if Path(str(row.folder)).exists():continue
            for sub in ('experiments_v2','experiments'):
                candidate=root/sub/str(row.experiment)/str(row.fold)
                if candidate.exists():index.loc[i,'folder']=str(candidate);break
    good=index.loc[index.status.eq('complete')].copy() if 'status' in index else pd.DataFrame()
    if good.empty:
        display(HTML(direct_table_html('Нет готовых моделей: причины каждого пропуска',index,opened=True)))
        return pd.DataFrame(),pd.DataFrame()
    choices=select_validation_variants(index,root)
    automatic=pd.DataFrame()
    if len(choices):
        automatic=choices.copy();automatic['source_variant']=automatic.variant;automatic['variant']='AUTO'
        automatic['experiment']=automatic.apply(lambda r:f'{r.method}__{r["name"]}__AUTO__{r.horizon_hours}h',axis=1)
        display(HTML('<p><b>AUTO:</b> набор признаков выбирается до очередного теста по AP на одинаковой прошлой validation. '
            'При равенстве предпочтение более простому набору: numeric, numeric_states, full. Тест в выборе не участвует.</p>'+
            direct_table_html('Какие признаки выбирались по прошлому',choices.drop(columns=['validation_index_hash'],errors='ignore'))))
    overall=[]
    all_records=pd.concat([automatic,good],ignore_index=True) if len(automatic) else good
    for experiment,group in all_records.groupby('experiment',sort=False):
        html,summary=combined_direct_report(group,contexts,root/'overall_reports'/experiment)
        display(HTML(html));overall.append(summary)
    # Individual-version reports are optional diagnostics, never the main view.
    detail=good
    if report_folds in ('none',None,False):detail=good.iloc[:0]
    elif report_folds=='latest':
        detail=good.sort_values('test_end').groupby('experiment',sort=False).tail(1)
    elif report_folds!='all':
        if not isinstance(report_folds,(list,tuple)):raise ValueError('REPORT_FOLDS: latest, all или список fold_01...')
        detail=good.loc[good.fold.isin(report_folds)]
    for row in detail.itertuples():
        folder=Path(row.folder);package=joblib.load(folder/'classifier.joblib')
        detail=package.get('experiment',{});units=detail.get('units',list(contexts))
        title=f'{row.method} · {row.name} · {row.variant} · {row.horizon_hours} ч · {row.fold}'
        from html import escape
        html,_=direct_report_html(package,{u:contexts[u] for u in units if u in contexts},folder,title)
        display(HTML('<details><summary><b>Диагностика отдельной версии: '+escape(title)+'</b></summary>'+html+'</details>'))
        del package,html;gc.collect()
    target=Path(root)/'fold_metrics.csv'
    metrics=pd.read_csv(target) if target.exists() else pd.DataFrame()
    aggregate=pd.concat(overall,ignore_index=True) if overall else pd.DataFrame()
    aggregate.to_csv(Path(root)/'comparison_rolling.csv',index=False,encoding='utf-8-sig')
    columns=[c for c in ('unit','method','name','variant','horizon_hours','decision','rows','precision','recall','f1','macro_f1',
        'roc_auc_mean','roc_auc_min','roc_auc_max','average_precision_mean','random_average_precision_mean',
        'rolling_events','rolling_detected_events','rolling_false_notifications','rolling_false_alarms_per_720h',
        'rolling_background_alarm_fraction','excluded_threshold_rows') if c in aggregate]
    display(HTML('<h3>Качество моделей за всю последовательную тестовую историю</h3>'+direct_table_html(
        f'Сводные метрики по котлам ({len(aggregate)} строк)',aggregate[columns],opened=len(aggregate)<=50)+
        direct_table_html('Все показатели модели, включая поддержку и матрицы ошибок',aggregate)))
    display(HTML(direct_table_html('Все отдельные периоды (в том числе macro и эпизоды)',metrics)))
    audit_cols=[c for c in ('method','name','variant','horizon_hours','fold','train_end','validation_end','test_end',
        'validation_days_actual','train_rows','train_events','validation_rows','validation_positive','validation_negative',
        'validation_events','test_rows','test_events','test_status','split_status','status','reason') if c in index]
    display(HTML(direct_table_html('Диагностика выборок: часы, аварии и причины пропусков',index[audit_cols])))
    display(HTML(direct_table_html('Статусы всех запусков: причины пропусков',index)))
    return aggregate,metrics


def combined_direct_report(records,contexts,folder):
    """Standard reports on the continuous held-out history; no averaged confusions."""
    from html import escape
    from sklearn.metrics import classification_report,ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    pieces=[];rules={};version_rows=[];importance=[];ranking=[]
    for row in records.sort_values('validation_end').itertuples():
        source=Path(row.folder);settings=json.loads((source/'settings.json').read_text(encoding='utf-8'))
        p=pd.read_parquet(source/'test_predictions.parquet');times=p.index.get_level_values('time')
        if not pd.to_datetime(p.model_available_at).le(times).all():raise ValueError('Модель применена до доступности')
        p['version']=str(row.fold)
        p['baseline_prevalence']=settings.get('train_prevalence',np.nan)
        for key,rule in (settings.get('thresholds') or {}).items():
            threshold=rule.get('threshold');threshold=float(threshold) if threshold is not None else np.inf
            available=rule.get('available',True) and not any(s in rule.get('status','') for s in ('fallback','not_evaluable','threshold_unavailable'))
            p['threshold::'+key]=threshold if available else np.inf;rules[key]=rule
        for unit in p.index.get_level_values('unit').unique():
            tested=p.xs(unit,level='unit').loc[lambda d:d.evaluation_eligible&d.label.notna()]
            values=direct_metrics(tested.label,tested.score,.5)
            ranking.append(dict(unit=unit,version=row.fold,**values))
        pieces.append(p)
        stats=pd.read_csv(source/'split_summary.csv')
        stats['version']=row.fold;stats['selected_variant']=getattr(row,'source_variant',row.variant)
        version_rows.append(stats)
        imp=source/'feature_importance.csv'
        if imp.exists():importance.append(pd.read_csv(imp))
    if not pieces:return '<p>Нет сохранённых тестовых оценок.</p>',pd.DataFrame()
    p=pd.concat(pieces).sort_index()
    if p.index.has_duplicates:raise ValueError('Тестовые часы пересекаются: '+str(folder))
    first=records.iloc[0];options=settings['options'];step=settings.get('alarm_step','1h')
    start=pd.Timestamp(records.validation_end.min());end=pd.Timestamp(records.test_end.max())
    rank=pd.DataFrame(ranking);summaries=[];blocks=[]
    title=f'{first.method} · {first["name"]} · {first.variant} · {first.horizon_hours} ч'
    num=lambda v:'—' if v is None or pd.isna(v) or not np.isfinite(v) else f'{v:.3f}'
    extra={'both_classes':'Дополнительные пороги: оба класса (macro)',
           'alarm_policy':'Подтверждённые тревоги и бюджет фона'}
    for unit in p.index.get_level_values('unit').unique():
        local=p.loc[p.index.get_level_values('unit')==unit].copy();d=local.xs(unit,level='unit')
        known=d.evaluation_eligible.eq(True)&d.label.notna()&d.score.notna()
        y_all=d.loc[known,'label'].astype(int)
        rr=rank.loc[rank.unit.eq(unit)].copy()
        valid_rank=rr.roc_auc.notna() if 'roc_auc' in rr else pd.Series(False,index=rr.index)
        ranks=dict(roc_auc_mean=rr.loc[valid_rank,'roc_auc'].mean() if valid_rank.any() else np.nan,
            roc_auc_min=rr.loc[valid_rank,'roc_auc'].min() if valid_rank.any() else np.nan,
            roc_auc_max=rr.loc[valid_rank,'roc_auc'].max() if valid_rank.any() else np.nan,
            average_precision_mean=rr.loc[valid_rank,'average_precision'].mean() if valid_rank.any() else np.nan,
            random_average_precision_mean=rr.loc[valid_rank,'random_average_precision'].mean() if valid_rank.any() else np.nan,
            rankable_versions=int(valid_rank.sum()),versions=len(rr))
        body=f'<p><b>Вся независимая тестовая история:</b> {start} → {end}; известных часов {len(y_all)}, '
        body+=f'положительных {int(y_all.sum())}, фоновых {int(y_all.eq(0).sum())}. Версий модели: {len(rr)}.</p>'
        support=pd.concat(version_rows,ignore_index=True)
        for partition,label in (('train','Обучение'),('validation','Подбор')):
            g=support.loc[support.partition.eq(partition)]
            body+=f'<p>{label}: {int(g.rows.min())}–{int(g.rows.max())} строк, '
            body+=f'{int(g.positive_events.min())}–{int(g.positive_events.max())} независимых аварий на версию '
            '(для всей модели/группы). Точные даты и состав скрыты в диагностике.</p>'
        body+=f'<p><b>ROC AUC:</b> среднее {num(ranks["roc_auc_mean"])}; диапазон {num(ranks["roc_auc_min"])}–{num(ranks["roc_auc_max"])}; случайно 0.5.<br>'
        body+=f'<b>Average Precision:</b> среднее {num(ranks["average_precision_mean"])}; случайный AP на тех же периодах {num(ranks["random_average_precision_mean"])}. '
        body+='Баллы разных версий не объединены для AUC/AP; отчёты ниже объединяют реальные ошибки.</p>'
        if not len(y_all):body+='<p>Нет часов с известным исходом; обученная модель пока не оценена.</p>'
        category_open=None
        for key,rule in rules.items():
            category=rule.get('category','positive_class')
            if category_open and category_open!=category:body+='</details>';category_open=None
            if category in extra and category_open!=category:
                body+='<details><summary><b>'+extra[category]+'</b></summary>';category_open=category
            threshold=d.get('threshold::'+key,pd.Series(np.inf,index=d.index))
            available=np.isfinite(threshold);take=known&available
            event_metrics={};events=pd.DataFrame()
            if category=='alarm_policy':
                event_metrics,stream,events=direct_alarm_evaluate(local,{unit:contexts[unit]} if unit in contexts else {},options,
                    local['threshold::'+key],step,start=start,end=end)
                pred=stream.xs(unit,level='unit').alarm_active.reindex(d.index)
            else:pred=d.score.ge(threshold)
            y=y_all.reindex(d.index[take]);selected_pred=pred.loc[take]
            body+='<h4>'+escape(rule['title'])+'</h4>'
            if not len(y):
                body+='<p>Нет пригодного порога на оцениваемых часах. Подстановка 0.5 не выполнялась.</p>'
                summaries.append(dict(method=first.method,name=first['name'],variant=first.variant,horizon_hours=first.horizon_hours,
                    unit=unit,decision=key,rows=0,excluded_threshold_rows=int(known.sum()),threshold_status='unavailable',**ranks));continue
            # With per-version thresholds, derive confusion using the actual
            # decisions; ranking always comes from the independent version table.
            metric=direct_metrics(y,d.loc[take,'score'],.5,selected_pred)
            metric.pop('roc_auc',None);metric.pop('average_precision',None);metric.pop('threshold',None)
            metric.update(excluded_threshold_rows=int((known&~available).sum()))
            report=classification_report(y,selected_pred,labels=[0,1],target_names=['Фон','Перед аварией'],digits=3,zero_division=0)
            fig,ax=plt.subplots(figsize=(4.5,3.7))
            ConfusionMatrixDisplay.from_predictions(y,selected_pred,labels=[0,1],display_labels=['Фон','Перед аварией'],ax=ax,colorbar=False)
            ax.set_title('Вся тестовая история');fig.tight_layout()
            body+='<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:center"><pre>'+escape(report)+'</pre>'
            body+=_direct_figure_html(fig,folder/(unit+'__'+key+'_confusion.png'))+'</div>'
            body+='<p>У каждой версии свой порог из прошлого; исключено часов без пригодного порога: '+str(metric['excluded_threshold_rows'])+'.</p>'
            if event_metrics:
                body+=f'<p><b>Уведомления за всю историю:</b> обнаружено {event_metrics["detected_events"]}/{event_metrics["events"]}; '
                body+=f'ложных {event_metrics["false_notifications"]}; на 720 ч фона {num(event_metrics["false_alarms_per_720h"])}; '
                body+=f'доля фона в тревоге {num(event_metrics["background_alarm_fraction"])}; без оценок {event_metrics["unscored_events"]} аварий.</p>'
                body+=direct_table_html('Все аварии и обнаружение',events)
            summaries.append(dict(method=first.method,name=first['name'],variant=first.variant,horizon_hours=first.horizon_hours,
                unit=unit,decision=key,**metric,**ranks,**{'rolling_'+k:v for k,v in event_metrics.items()}))
        if category_open:body+='</details>'
        seed=int(hashlib.sha256(str(unit).encode()).hexdigest()[:8],16)
        baselines={'always_alarm':np.ones(len(y_all)),'random_uniform_guess':np.random.default_rng(seed).random(len(y_all)),
                   'constant_train_prevalence':d.loc[known,'baseline_prevalence'].to_numpy()}
        for key,values in baselines.items():
            if not np.isfinite(values).all():continue
            metric=direct_metrics(y_all,values,.5);metric.pop('roc_auc',None);metric.pop('average_precision',None)
            summaries.append(dict(method=first.method,name=first['name'],variant=first.variant,horizon_hours=first.horizon_hours,
                unit=unit,decision=key,**metric))
        fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True)
        direct_plot_segments(axes[0],d.index,d.score,step,lw=.7,label='Балл риска')
        for key in ('default_05','best_f2'):
            col='threshold::'+key
            if col in d:direct_plot_segments(axes[0],d.index,d[col].where(np.isfinite(d[col])),step,lw=.9,ls='--',label=key)
        axes[0].set_ylim(-.02,1.02);axes[0].set_ylabel('Балл');axes[0].legend(fontsize=8)
        if 'threshold::alarm_budget_recall' in d:
            _,stream,events=direct_alarm_evaluate(local,{unit:contexts[unit]} if unit in contexts else {},options,
                local['threshold::alarm_budget_recall'],step,start=start,end=end)
            state=stream.xs(unit,level='unit').alarm_active.astype(float)
            direct_plot_segments(axes[1],state.index,state,step,lw=.8,label='Подтверждённая тревога')
            for event in events.itertuples():
                for ax in axes:ax.axvline(event.stop,color='red',alpha=.4)
        axes[1].set_ylabel('Тревога');axes[0].set_title(unit+' · '+title+' · вся тестовая история')
        for ax in axes:ax.grid(alpha=.2)
        fig.tight_layout();body+=_direct_figure_html(fig,folder/(unit+'_whole_history.png'))
        blocks.append('<details class="direct-boiler"><summary><b>'+escape(unit+' · '+title)+'</b></summary>'+body+'</details>')
    diagnostics=direct_table_html('Диагностика отдельных версий: состав выборок',pd.concat(version_rows,ignore_index=True))
    if importance:
        imp=pd.concat(importance).groupby('feature',as_index=False).importance.sum()
        imp['importance']=imp.importance/len(importance);imp=imp.loc[imp.importance.ne(0)].sort_values('importance',ascending=False)
        diagnostics+=direct_table_html('Ненулевая важность: среднее по версиям, отсутствие признака считается нулём',imp)
        if len(imp):
            top=imp.head(25).iloc[::-1];fig,ax=plt.subplots(figsize=(12,8))
            ax.barh(top.feature,top.importance);ax.set_title('Важность входов: среднее по версиям');fig.tight_layout()
            diagnostics+='<details><summary>График важности входов</summary>'+_direct_figure_html(fig,folder/'importance.png')+'</details>'
    html='<details><summary><b>'+escape(title+' · общий результат')+'</b></summary>'+''.join(blocks)+diagnostics+'</details>'
    result=pd.DataFrame(summaries);result.to_csv(folder/'overall_metrics.csv',index=False,encoding='utf-8-sig')
    (folder/'report.html').write_text('<meta charset="utf-8">'+html,encoding='utf-8')
    return html,result
