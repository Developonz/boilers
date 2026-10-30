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
        direct_table_html,direct_predict,direct_report_html)


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
        signature=hashlib.sha256(json.dumps(dict(contract=3,options={k:options[k] for k in feature_keys},
            step=cfg['step'],roles=c['roles'],units=c.get('declared_units',{}),
            reference=str(c.get('preparation_reference_end')),threshold_source=c.get('threshold_source'),
            evidence=c.get('regime_spec',{})),sort_keys=True,default=str).encode())
        for frame in (c['cleaned'],c['frame'][[s for s,r in c['roles'].items() if r=='state']+
            (['_Тип'] if '_Тип' in c['frame'] else [])],c['regime'][['mode','since_start_hours']]):
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
        events=c['events'].to_json(date_format='iso'),step=cfg['step']),sort_keys=True).encode()).hexdigest()[:20]
    target=folder/f'labels_{options["horizon_hours"]}h_{signature}.parquet'
    audit_path=target.with_suffix('.json')
    if target.exists() and audit_path.exists():return pd.read_parquet(target),json.loads(audit_path.read_text(encoding='utf-8'))
    meta=pd.read_parquet(folder/'metadata.parquet');source_rows=len(meta);meta=meta.loc[meta.eligible].copy()
    as_of=c['frame'].index[-1]
    y,g,audit,known=risk_labels(meta.index,c['events'],options['horizon_hours'],as_of,as_of,
        observed=c['cleaned'][c['load']].notna(),min_lead_hours=options['min_lead_hours'],
        return_audit=True,return_available_at=True)
    meta['label']=y;meta['label_group']=c['unit']+'::'+g.astype(str)
    meta['label_known_at']=known;meta['source_end']=as_of
    audit.update(unit=c['unit'],horizon_hours=options['horizon_hours'],source_rows=source_rows,
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


def rolling_direct_folds(contexts,cfg,rolling):
    allowed={'folds','validation_days','test_days','end'}
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
    families=experiment_families(contexts,cfg,methods);folds=rolling_direct_folds(contexts,cfg,rolling)
    if not variants or not horizons:raise ValueError('Задайте непустые списки FEATURE_VARIANTS и HORIZONS.')
    if len(set(horizons))!=len(horizons) or any(h<=options['min_lead_hours'] for h in horizons):raise ValueError('Проверьте HORIZONS: уникальные положительные горизонты больше min_lead_hours.')
    records=[];metrics=[];audits=[]
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
                for fold in folds:
                    folder=root/'experiments'/experiment/fold['fold'];folder.mkdir(parents=True,exist_ok=True)
                    detail=dict(method=family['method'],name=family['name'],units=family['units'],variant=variant,
                        horizon_hours=horizon,fold=fold['fold'],train_end=fold['train_end'],validation_end=fold['validation_end'],test_end=fold['test_end'])
                    record=dict(**{k:v for k,v in detail.items() if k!='units'},experiment=experiment,
                        folder=str(folder.resolve()),matrix_mb=mb)
                    progress(experiment+' · '+fold['fold'])
                    opts=direct_options({**base,**{k:fold[k] for k in ('train_end','validation_end','test_end')}})
                    t=meta.index.get_level_values('time')
                    if not (meta.eligible&(t>=pd.Timestamp(fold['validation_end']))&(t<pd.Timestamp(fold['test_end']))).any():
                        records.append(dict(**record,status='no_test_observations',reason='Нет рабочих измерений в этом периоде.'));save();continue
                    try:
                        package,_,_,_=direct_fit(x,meta,schema,opts,folder,verbose=100,
                            contexts={u:contexts[u] for u in family['units']},cfg=cfg)
                        package['experiment']=detail;joblib.dump(package,folder/'classifier.joblib',compress=3)
                        summary=direct_report_summary(package,{u:contexts[u] for u in family['units']},experiment)
                        for key in ('method','name','variant','horizon_hours','fold','train_end','validation_end','test_end'):summary[key]=detail[key]
                        summary.to_csv(folder/'test_summary_by_boiler.csv',index=False,encoding='utf-8-sig')
                        metrics.append(summary)
                        records.append(dict(**record,status='complete',features=len(package['columns']),trees=package['model'].tree_count_,reason=''))
                        del package,summary;gc.collect()
                    except ValueError as exc:
                        records.append(dict(**record,status='insufficient_or_invalid_data',reason=str(exc)))
                        progress('Пропуск с явной причиной: '+str(exc))
                    save()
                del meta;gc.collect()
            del x,schema;gc.collect()
    save()
    return pd.DataFrame(records),pd.concat(metrics,ignore_index=True) if metrics else pd.DataFrame()


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


def render_direct_suite(index,contexts,root,report_folds='latest'):
    """Read/render one saved model at a time; no training, no PACKAGES collection."""
    from IPython.display import display,HTML
    import joblib
    good=index.loc[index.status.eq('complete')].copy() if 'status' in index else pd.DataFrame()
    if good.empty:
        display(HTML(direct_table_html('Нет готовых моделей: причины каждого пропуска',index,opened=True)))
        return pd.DataFrame(),pd.DataFrame()
    if report_folds=='latest':
        good=good.sort_values('test_end').groupby('experiment',sort=False).tail(1)
    elif report_folds!='all':
        if not isinstance(report_folds,(list,tuple)):raise ValueError('REPORT_FOLDS: latest, all или список fold_01...')
        good=good.loc[good.fold.isin(report_folds)]
    for row in good.itertuples():
        folder=Path(row.folder);package=joblib.load(folder/'classifier.joblib')
        detail=package.get('experiment',{});units=detail.get('units',list(contexts))
        title=f'{row.method} · {row.name} · {row.variant} · {row.horizon_hours} ч · {row.fold}'
        from html import escape
        html,_=direct_report_html(package,{u:contexts[u] for u in units if u in contexts},folder,title)
        display(HTML('<details><summary><b>'+escape(title)+'</b></summary>'+html+'</details>'))
        del package,html;gc.collect()
    target=Path(root)/'fold_metrics.csv'
    metrics=pd.read_csv(target) if target.exists() else pd.DataFrame()
    aggregate=aggregate_direct_folds(metrics)
    alarm_summary=direct_rolling_alarm_summary(index,contexts,root)
    if len(alarm_summary) and len(aggregate):
        aggregate=aggregate.merge(alarm_summary,on=['method','name','variant','horizon_hours','unit','decision'],how='left')
    aggregate.to_csv(Path(root)/'comparison_rolling.csv',index=False,encoding='utf-8-sig')
    display(HTML('<h3>Сравнение способов, признаков и горизонтов по всем периодам</h3>'+direct_table_html(
        f'Сводные метрики по котлам ({len(aggregate)} строк)',aggregate,opened=len(aggregate)<=50)))
    display(HTML(direct_table_html('Все отдельные периоды (в том числе macro и эпизоды)',metrics)))
    display(HTML(direct_table_html('Эпизоды по всей последовательной истории: состояние не сбрасывается на границах моделей',alarm_summary)))
    display(HTML(direct_table_html('Статусы всех запусков: причины пропусков',index)))
    return aggregate,metrics
