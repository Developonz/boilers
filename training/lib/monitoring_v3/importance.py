"""Native model reliance; never interpret it as a causal explanation of an alarm."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

CONTRACT=1
COLUMNS=['target','feature','importance_percent','status','method','problem']


def native_importance(estimator,target,threads):
    names=list(estimator.feature_names_)
    values=np.asarray(estimator.get_feature_importance(type='PredictionValuesChange',thread_count=threads),float)
    if values.shape!=(len(names),) or not np.isfinite(values).all():
        raise ValueError('Invalid native feature importance for '+str(target))
    total=values.sum()
    values=values*100/total if total>0 else np.zeros_like(values)
    return pd.DataFrame(dict(target=target,feature=names,importance_percent=values,
        status='available',method='PredictionValuesChange'))


def _safe_native(estimator,target,threads):
    try:return native_importance(estimator,target,threads)
    except Exception as exc:
        # An auxiliary explanation must not discard a successfully trained model.
        return pd.DataFrame([dict(target=target,feature='',importance_percent=np.nan,status='importance_failed',
            method='PredictionValuesChange',problem=type(exc).__name__+': '+str(exc))])


def regressor_importance(model):
    threads=model.cfg['threads'];parts=[]
    if hasattr(model,'members'):
        parts=[regressor_importance(m) for m in model.members.values()]
    elif hasattr(model,'correction'):
        parts=[regressor_importance(model.correction)]
    elif hasattr(model,'shared'):
        parts=[_safe_native(model.shared,'__shared_targets__',threads)]
    elif hasattr(model,'predictors'):
        for target in model.targets:
            item=model.predictors.get(target)
            if item is not None and item[0]=='catboost':
                parts.append(_safe_native(item[1],target,threads))
            else:
                status='constant_output' if item is not None and item[0] in ('constant','reference') else 'unsupported_or_untrained'
                parts.append(pd.DataFrame([dict(target=target,feature='',importance_percent=np.nan,status=status,method='unavailable')]))
    else:
        parts=[pd.DataFrame([dict(target=c,feature='',importance_percent=np.nan,
            status='unsupported_model',method='unavailable') for c in model.targets])]
    return pd.concat(parts,ignore_index=True).reindex(columns=COLUMNS) if parts else pd.DataFrame(columns=COLUMNS)


def save_tables(table,folder,previous=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    table=table.sort_values(['target','importance_percent','feature'],ascending=[True,False,True],kind='stable')
    available=table.loc[table.status.eq('available')]
    targets=available.target.nunique()
    summary=available.groupby('feature').importance_percent.sum().div(max(targets,1)).rename('mean_importance_percent').reset_index()
    summary['targets_used']=targets;summary['targets_total']=table.target.nunique()
    summary=summary.sort_values(['mean_importance_percent','feature'],ascending=[False,True],kind='stable')
    delta=table.loc[table.status.eq('available'),['target','feature','importance_percent']].copy()
    if previous and (Path(previous)/'feature_importance.csv').exists():
        old=pd.read_csv(Path(previous)/'feature_importance.csv')
        old=old.loc[old.status.eq('available'),['target','feature','importance_percent']].rename(columns={'importance_percent':'previous_percent'})
        delta=delta.merge(old,on=['target','feature'],how='outer')
        # A new/removed input is explicit; absence of an entire target model is
        # not silently described as zero reliance of an otherwise fitted model.
        current_targets=set(table.loc[table.status.eq('available'),'target'])
        previous_targets=set(old.target)
        common=delta.target.isin(current_targets&previous_targets)
        delta.loc[common,['importance_percent','previous_percent']]=delta.loc[common,['importance_percent','previous_percent']].fillna(0.)
        delta['delta_percentage_points']=delta.importance_percent-delta.previous_percent
        delta['delta_percentage_points']=delta.delta_percentage_points.mask(delta.delta_percentage_points.abs().lt(1e-9),0.)
    else:
        delta['previous_percent']=np.nan;delta['delta_percentage_points']=np.nan
    table.to_csv(folder/'feature_importance.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(folder/'feature_importance_summary.csv',index=False,encoding='utf-8-sig')
    delta.to_csv(folder/'feature_importance_delta.csv',index=False,encoding='utf-8-sig')
    return table,summary,delta


def write_regressor_importance(model,folder,token,previous=None):
    folder=Path(folder);marker=folder/'importance.json'
    if marker.exists():
        meta=json.loads(marker.read_text(encoding='utf-8'))
        if meta.get('contract')==CONTRACT and meta.get('fingerprint')==token and all(
            (folder/file).exists() for file in ('feature_importance.csv','feature_importance_summary.csv','feature_importance_delta.csv')):
            return folder
    save_tables(regressor_importance(model),folder,previous)
    meta=dict(contract=CONTRACT,fingerprint=token,type='PredictionValuesChange',
        meaning='global model reliance, not direction or cause of a specific anomaly',
        previous_version=str(previous) if previous else None,
        aggregation='equal mean across available target models; constant/untrained targets excluded')
    marker.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return folder


def write_classifier_importance(model,folder,threads):
    folder=Path(folder)
    old=folder/'feature_importance.csv'
    previous=None
    if old.exists():
        import shutil,uuid
        previous=folder/'importance_history'/uuid.uuid4().hex;previous.mkdir(parents=True)
        shutil.copy2(old,previous/old.name)
    save_tables(_safe_native(model,'classifier',threads).reindex(columns=COLUMNS),folder,previous)
