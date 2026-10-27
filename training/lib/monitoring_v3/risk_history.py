"""Monthly out-of-time risk predictions; never apply the final head to its past."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from .classification import CONTRACT,train_classifier,probabilities
from .checkpoint_cache import semantic_config
from .risk_dataset import make_examples

HISTORY_VERSION=1


def _event_hash(context):
    return hashlib.sha256(json.dumps(context['events'].to_dict('records'),sort_keys=True,default=str,ensure_ascii=False).encode()).hexdigest()


def _config_hash(cfg):
    return hashlib.sha256(json.dumps(semantic_config(cfg),sort_keys=True,default=str,ensure_ascii=False).encode()).hexdigest()


def read_risk_timeline(directory,cfg=None,context=None):
    """Never show a risk file left over from different detector features."""
    directory=Path(directory);meta=directory/'risk_timeline.meta.json'
    target=directory/'risk_timeline.parquet';source=directory/'interpretation_features.parquet'
    if not (meta.exists() and target.exists() and source.exists()):return None
    information=json.loads(meta.read_text(encoding='utf-8'))
    if information.get('features_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():return None
    if cfg is not None and information.get('config_sha256')!=_config_hash(cfg):return None
    if context is not None and information.get('events_sha256')!=_event_hash(context):return None
    return pd.read_parquet(target)


def _combine(tables):
    out=pd.concat(tables,names=['unit','time']).swaplevel().sort_index(kind='stable')
    out.index.names=['time','unit']
    return out


def build_risk_history(cfg,items,contexts,as_of,progress=None,on_update=None):
    """Train independent rolling heads on labels available at each month start.

    Final working-head artifacts are neither read nor overwritten. Incomplete
    labels may still be predicted: follow-up is needed to train, not to infer.
    """
    if not items:return {}
    as_of=pd.Timestamp(as_of);parts={}
    digest=hashlib.sha256(json.dumps([HISTORY_VERSION,CONTRACT,semantic_config(cfg),str(as_of)],
        sort_keys=True,ensure_ascii=False,default=str).encode())
    for item in sorted(items,key=lambda i:(i['unit'],str(i['directory']))):
        unit=item['unit'];path=Path(item['directory'])/'interpretation_features.parquet'
        f=pd.read_parquet(path).loc[lambda d:d.index<=as_of]
        parts[unit]=f
        digest.update(unit.encode());digest.update(pd.util.hash_pandas_object(f,index=True).to_numpy().tobytes())
        digest.update(json.dumps(contexts[unit]['events'].to_dict('records'),sort_keys=True,default=str,ensure_ascii=False).encode())
        c=contexts[unit];observed=c.get('cleaned',c['frame'])[c['load']].notna()
        digest.update(pd.util.hash_pandas_object(observed,index=True).to_numpy().tobytes())
    key=digest.hexdigest()
    identity=hashlib.sha256(json.dumps(sorted((i.get('experiment',''),i['unit']) for i in items)).encode()).hexdigest()[:16]
    root=Path(cfg['output'])/'risk_history'/identity/key
    root.mkdir(parents=True,exist_ok=True)
    cache=root/'history.parquet';manifest=root/'complete.json'
    if cache.exists() and manifest.exists():
        history=pd.read_parquet(cache)
        if progress:progress.say('История риска восстановлена из отдельного кеша',status='risk_history_restored')
        if on_update:on_update(1,1)
    else:
        nonempty=[f.index.min() for f in parts.values() if len(f)]
        if not nonempty:return {}
        months=pd.date_range(min(nonempty).to_period('M').to_timestamp(),as_of,freq='MS')
        records=[];audits=[];done=0;total=len(months)*len(cfg['risk_horizons'])
        if on_update:on_update(0,total)
        for hours in cfg['risk_horizons']:
            previous=None
            for cutoff in months:
                end=cutoff+pd.offsets.MonthBegin(1)
                xs,ys,gs,ats={},{},{},{}
                for unit,f in parts.items():
                    x,y,g,audit,available=make_examples(f,contexts[unit],hours,cutoff,cfg,return_available_at=True)
                    xs[unit],ys[unit],gs[unit],ats[unit]=x,y,g,available
                x,y,g,available=map(_combine,(xs,ys,gs,ats))
                folder=root/f'risk_{hours}h'/f'{cutoff:%Y-%m}'
                label=f'Исторический риск {hours} ч · модель на {cutoff:%Y-%m-%d}'
                if progress:
                    with progress.stage(label):
                        status=train_classifier(x,y,g,cfg,folder,previous,horizon_hours=hours,task='risk',
                            label_available_at=available,evaluate=False)
                else:
                    status=train_classifier(x,y,g,cfg,folder,previous,horizon_hours=hours,task='risk',
                        label_available_at=available,evaluate=False)
                ready=status['status'] in ('trained','trained_uncalibrated')
                previous=joblib.load(folder/'interpreter.joblib') if ready else None
                audits.append({**status,'horizon_hours':hours,'cutoff':str(cutoff)})
                for unit,f in parts.items():
                    query=f.loc[(f.index>=cutoff)&(f.index<end)]
                    if query.empty:continue
                    valid=query.features_valid.fillna(False).astype(bool)
                    if cfg['risk_policy']['samples']=='episodes':valid&=query.episode_active.eq(1)
                    out=pd.DataFrame(dict(unit=unit,horizon_hours=hours,model_cutoff=cutoff,
                        probability=np.nan,raw_score=np.nan,status=status['status'],
                        probability_status=status.get('probability_status','unavailable')),index=query.index)
                    out.index.name='time'
                    out.loc[~valid,'status']='outside_eligible_risk_points'
                    if ready and valid.any():
                        raw,classes=probabilities(dict(previous,calibrator=None),query.loc[valid],threads=cfg['threads'])
                        calibrated,_=probabilities(previous,query.loc[valid],require_calibration=True,threads=cfg['threads'])
                        if '1.0' in classes:
                            col=classes.index('1.0')
                            out.loc[valid,'raw_score']=raw[:,col]
                            out.loc[valid,'probability']=calibrated[:,col]
                    records.append(out.reset_index())
                done+=1
                if on_update:on_update(done,total)
        history=pd.concat(records,ignore_index=True) if records else pd.DataFrame(
            columns=['time','unit','horizon_hours','model_cutoff','probability','raw_score','status','probability_status'])
        temporary=cache.with_suffix('.tmp.parquet');history.to_parquet(temporary,index=False);temporary.replace(cache)
        (root/'months.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        manifest.write_text(json.dumps(dict(version=HISTORY_VERSION,causal=True,fingerprint=key,
            rows=len(history),description='fit at month start; known labels only; next-month predictions')),encoding='utf-8')
    result={}
    for item in items:
        unit=item['unit'];directory=Path(item['directory'])
        output=history.loc[history.unit.eq(unit)].copy()
        output.to_parquet(directory/'risk_timeline.parquet',index=False)
        (directory/'risk_timeline.meta.json').write_text(json.dumps(dict(version=HISTORY_VERSION,
            fingerprint=key,as_of=str(as_of),causal=True,
            config_sha256=_config_hash(cfg),events_sha256=_event_hash(contexts[unit]),
            features_sha256=hashlib.sha256((directory/'interpretation_features.parquet').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
        result[unit]=output
    return result
