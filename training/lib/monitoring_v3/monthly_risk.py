"""Live monthly risk lifecycle shared with the detector scheduler."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from .classification import CONTRACT,train_classifier,probabilities,_metrics
from .checkpoint_cache import semantic_config
from .risk_dataset import make_examples,alarm_features
from .alarm_scores import alarm_stream
from .risk_history import _combine,_event_hash,_config_hash
from .risk_reporting import readiness_rows

MONTHLY_VERSION=1
READY=('trained','trained_uncalibrated')


def family_key(spec):
    return (spec['group'],spec['method'],spec['variant'],spec['horizon'],spec['scenario'],
            tuple(spec['sources']) if spec['scenario']=='transfer' else ())


def interpretation_options(options=None):
    result={'run':True,'attach':True,'as_of':None,**(options or {})}
    if set(result)-{'run','attach','as_of'}:raise ValueError('Unknown interpretation options')
    if not all(isinstance(result[k],bool) for k in ('run','attach')):raise ValueError('run/attach must be bool')
    if result['as_of'] is not None:
        value=pd.Timestamp(result['as_of'])
        if pd.isna(value) or value.tz is not None:raise ValueError('as_of must be a local timestamp')
        result['as_of']=value
    return result


def month_features(pieces,feature_parts,contexts,spec,beginning,end,cfg,dest):
    result={}
    for unit in spec['recipients']:
        if not feature_parts[unit]:continue
        index=contexts[unit]['frame'].index
        index=index[(index>=beginning)&(index<end)]
        f=pd.concat(feature_parts[unit]).sort_index().reindex(index)
        d=pd.concat(pieces[unit]['mean::instant']).sort_index().reindex(index)
        threshold=d['threshold_budget_'+str(cfg['alarm_budget'])]
        stream,_=alarm_stream(d.smoothed/threshold.where(np.isfinite(threshold)),cfg)
        f=alarm_features(f,stream,cfg)
        folder=Path(dest)/unit;folder.mkdir(exist_ok=True)
        f.to_parquet(folder/'interpretation_features.parquet')
        result[unit]=f
    return result


def _hash_frame(digest,frame):
    digest.update(str(list(frame) if isinstance(frame,pd.DataFrame) else frame.name).encode())
    digest.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())


def _write_json(path,value):
    from .artifacts import _file_operation
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    _file_operation(lambda:temp.replace(path))


class MonthlyRisk:
    def __init__(self,cfg,items,contexts,options=None):
        self.cfg=cfg;self.items=items;self.contexts=contexts;self.options=interpretation_options(options)
        description=sorted((i['experiment'],i['unit']) for i in items)
        identity=hashlib.sha256(json.dumps(description,ensure_ascii=False).encode()).hexdigest()[:20]
        self.root=Path(cfg['output'])/'monthly_risk'/identity
        self.previous={};self.chain={};self.predictions=[];self.statuses={};self.frames={};self.last_cutoff=None
        self.metrics=[]

    def step(self,cutoff,end,frames,progress):
        progress.set_detail('Подготовка помесячного риска',risk_done=0,risk_total=len(self.cfg['risk_horizons']))
        cutoff=pd.Timestamp(cutoff);self.frames.update(frames)
        fit_date=min(cutoff,self.options['as_of']) if self.options['as_of'] is not None else cutoff
        statuses={};records=[]
        self.metrics=[]
        past=pd.concat(self.predictions,ignore_index=True) if self.predictions else pd.DataFrame()
        for number,hours in enumerate(self.cfg['risk_horizons'],1):
            head=f'risk_{hours}h';xs,ys,gs,ats,audits={},{},{},{},[]
            for unit,f in self.frames.items():
                x,y,g,audit,available=make_examples(f,self.contexts[unit],hours,fit_date,self.cfg,return_available_at=True)
                xs[unit],ys[unit],gs[unit],ats[unit]=x,y,g,available;audits.append(audit)
            for item in self.items:
                if item['unit'] in self.frames:continue
                empty=pd.DataFrame({'features_valid':pd.Series(dtype=bool)},index=pd.DatetimeIndex([]))
                _,_,_,audit=make_examples(empty,self.contexts[item['unit']],hours,fit_date,self.cfg)
                audits.append(audit)
            if xs:
                x,y,g,available=map(_combine,(xs,ys,gs,ats))
                x=x.reindex(sorted(x.columns),axis=1)
            else:
                ix=pd.MultiIndex.from_arrays([pd.DatetimeIndex([]),[]],names=['time','unit'])
                x=pd.DataFrame(index=ix);y=pd.Series(index=ix,dtype=float);g=pd.Series(index=ix,dtype=str)
                available=pd.Series(index=ix,dtype='datetime64[ns]')
            digest=hashlib.sha256(json.dumps([MONTHLY_VERSION,CONTRACT,semantic_config(self.cfg),self.options['run'],str(fit_date),
                self.chain.get(hours,'')],sort_keys=True,ensure_ascii=False,default=str).encode())
            for value in (x,y,g,available):_hash_frame(digest,value)
            token=digest.hexdigest();folder=self.root/f'{cutoff:%Y-%m-%d}'/head;folder.mkdir(parents=True,exist_ok=True)
            marker=folder/'fit_signature.json';cached=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
            restored=cached.get('fingerprint')==token and (folder/'status.json').exists()
            if restored:
                status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
                restored=status['status'] not in READY or (folder/'interpreter.joblib').exists()
            label=f'Риск {hours} ч · {"+".join(i["unit"] for i in self.items)} · {cutoff:%Y-%m}'
            with progress.stage(label+' · '+('восстановление' if restored else 'обучение / продолжение')):
                if not restored:
                    if self.options['run']:
                        status=train_classifier(x,y,g,self.cfg,folder,self.previous.get(hours),horizon_hours=hours,
                            task='risk',label_available_at=available,evaluate=False)
                    else:
                        status=dict(status='not_run',probability_status='unavailable')
                    status.update(workflow='monthly_prequential_v1',available_at=str(fit_date),
                        model_cutoff=str(cutoff),scope=self.cfg['risk_policy']['scope'],
                        units=sorted({i['unit'] for i in self.items}),label_audit=audits,
                        shared_head=str((folder/'interpreter.joblib').resolve()))
                    _write_json(folder/'status.json',status)
                    _write_json(marker,dict(fingerprint=token,predecessor=self.chain.get(hours,'')))
                status={**status,'restored':restored}
                prequential={}
                if len(past):
                    previous_scores=past.loc[past.horizon_hours.eq(hours)].drop_duplicates(['time','unit'],keep='last').set_index(['time','unit'])
                    previous_scores=previous_scores.reindex(y.index)
                    for source in ('raw_score','probability'):
                        known=y.notna()&previous_scores[source].notna()
                        targets=y.loc[known];scores=previous_scores.loc[known,source].to_numpy(float)
                        metric=_metrics(targets,np.column_stack([1-scores,scores]),['0.0','1.0'],self.cfg['risk_policy']['confidence'])
                        metric.update(as_of=str(fit_date),source='monthly_out_of_time',
                            scope=self.cfg['risk_policy']['scope'],units='+'.join(sorted({i['unit'] for i in self.items})),
                            independent_positive_events=int(g.loc[known&y.eq(1)].nunique()))
                        prequential[source]=metric
                        self.metrics.append(dict(head=head,score_type=source,**metric))
                status['prequential']=prequential
                status['shared_head']=str((folder/'interpreter.joblib').resolve())
                _write_json(folder/'status.json',status)
                previous=joblib.load(folder/'interpreter.joblib') if status['status'] in READY else None
                self.previous[hours]=previous;self.chain[hours]=token
                statuses[head]=status
                for unit,f in self.frames.items():
                    query=f.loc[(f.index>=cutoff)&(f.index<end)]
                    if query.empty:continue
                    valid=query.features_valid.fillna(False).astype(bool)
                    if self.cfg['risk_policy']['samples']=='episodes':valid&=query.episode_active.eq(1)
                    out=pd.DataFrame(dict(unit=unit,horizon_hours=hours,model_cutoff=fit_date,
                        probability=np.nan,raw_score=np.nan,status=status['status'],
                        probability_status=status.get('probability_status','unavailable')),index=query.index)
                    out.index.name='time';out.loc[~valid,'status']='outside_eligible_risk_points'
                    if previous is not None and valid.any():
                        raw,classes=probabilities(dict(previous,calibrator=None),query.loc[valid],threads=self.cfg['threads'])
                        calibrated,cal_classes=probabilities(previous,query.loc[valid],require_calibration=True,threads=self.cfg['threads'])
                        if '1.0' in classes:out.loc[valid,'raw_score']=raw[:,classes.index('1.0')]
                        if '1.0' in cal_classes:out.loc[valid,'probability']=calibrated[:,cal_classes.index('1.0')]
                    records.append(out.reset_index())
            progress.set_detail(f'Риск: горизонты {number}/{len(self.cfg["risk_horizons"])}',
                risk_done=number,risk_total=len(self.cfg['risk_horizons']))
        self.statuses=statuses;self.last_cutoff=cutoff
        if records:self.predictions.append(pd.concat(records,ignore_index=True))
        history=pd.concat(self.predictions,ignore_index=True) if self.predictions else pd.DataFrame(
            columns=['time','unit','horizon_hours','model_cutoff','probability','raw_score','status','probability_status'])
        history=history.drop_duplicates(['unit','horizon_hours','time'],keep='last').sort_values(['time','unit','horizon_hours'])
        self.publish(cutoff,history)
        return statuses,{i['unit']:history.loc[history.unit.eq(i['unit'])].copy() for i in self.items}

    def publish(self,cutoff,history):
        from interpretation_training import attach_heads
        from .importance import write_classifier_importance
        for head,package in self.previous.items():
            if package is not None:
                folder=Path(self.statuses[f'risk_{head}h']['shared_head']).parent
                # Global importance is for this exact version, no extra model fit.
                write_classifier_importance(package['model'],folder,self.cfg['threads'])
        for item in self.items:
            directory=Path(item['directory']);directory.mkdir(parents=True,exist_ok=True)
            existing=directory/'interpretation_status.json'
            # A reviewed cause head from a later run must never be backdated
            # into these earlier monthly exports. Section 7 handles causes.
            status=dict(self.statuses)
            _write_json(existing,status)
            readiness_rows(self.statuses,self.cfg).to_csv(directory/'classifier_readiness.csv',index=False,encoding='utf-8-sig')
            metrics=pd.DataFrame(self.metrics) if self.metrics else pd.DataFrame(columns=['head','score_type','precision','recall','rows'])
            metrics.to_csv(directory/'risk_metrics.csv',index=False,encoding='utf-8-sig')
            output=history.loc[history.unit.eq(item['unit'])]
            output.to_parquet(directory/'risk_timeline.parquet',index=False)
            feature_path=directory/'interpretation_features.parquet'
            if feature_path.exists():
                _write_json(directory/'risk_timeline.meta.json',dict(version=MONTHLY_VERSION,workflow='monthly_prequential_v1',
                    as_of=str(cutoff),causal=True,config_sha256=_config_hash(self.cfg),
                    events_sha256=_event_hash(self.contexts[item['unit']]),
                    features_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest()))
            if self.options['attach']:
                for target in (directory/'model_history'/f'{cutoff:%Y-%m}',directory/'model'):
                    if target.exists():attach_heads(directory,status,cutoff,exported=target)


def saved_monthly_statuses(directory,as_of=None):
    path=Path(directory)/'interpretation_status.json'
    if not path.exists():return None
    statuses=json.loads(path.read_text(encoding='utf-8'))
    result={k:v for k,v in statuses.items() if k.startswith('risk_') and v.get('workflow')=='monthly_prequential_v1'}
    if not result:return None
    if as_of is not None and any(pd.Timestamp(v['available_at'])>pd.Timestamp(as_of) for v in result.values()):return None
    return result
