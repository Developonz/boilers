"""Reusable interpretation workflow for the notebook and optional command-line entry point."""
from pathlib import Path
import json
import pandas as pd
from monitoring_v3.data import load_sources
from monitoring_v3.interpretation import train_head, risk_labels


def train(cfg, directory, unit, as_of, attach=False, features_path=None, contexts=None, risk_statuses=None):
    import joblib
    directory = Path(directory)
    if contexts is None:
        contexts,_=load_sources(cfg)
    context=contexts[unit]
    if features_path and attach:
        raise ValueError('Ensemble heads require a matching ensemble inference artifact; do not attach to one detector')
    features=pd.read_parquet(features_path or directory/'interpretation_features.parquet')
    valid=features['features_valid'].fillna(False) if 'features_valid' in features else context['eligible'].reindex(features.index).fillna(False)
    features=features.loc[(features.index < as_of) & valid]
    statuses={}
    if risk_statuses is not None:
        statuses.update(risk_statuses)
    else:
        from monitoring_v3.risk_dataset import make_examples
        for hours in cfg['risk_horizons']:
            f,y,g,audit,available=make_examples(features,context,hours,as_of,cfg,return_available_at=True)
            dest=directory/'interpretation'/f'risk_{hours}h'
            previous=joblib.load(dest/'interpreter.joblib') if (dest/'interpreter.joblib').exists() else None
            result=train_head(f,y,g,cfg,dest,previous,horizon_hours=hours,task='risk',label_available_at=available)
            result['label_audit']=audit
            statuses[f'risk_{hours}h']=result
    labels=pd.read_csv(directory/'episode_labels.csv',parse_dates=['confirmed'])
    available=pd.to_datetime(labels.label_available_at,errors='coerce')
    labels=labels.loc[labels.label.notna() & labels.label.astype(str).str.strip().ne('') & available.le(as_of)].copy()
    if labels.empty:
        statuses['cause']={'status':'requires_reviewed_episode_labels'}
    else:
        conflict=labels.groupby('confirmed').label.nunique().gt(1)
        if conflict.any():
            statuses['cause']={'status':'conflicting_reviewed_labels','times':[str(t) for t in conflict.index[conflict]]}
            (directory/'interpretation_status.json').write_text(json.dumps(statuses,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            if attach:attach_heads(directory,statuses,as_of)
            return statuses
        labels=labels.sort_values('confirmed').drop_duplicates('confirmed')
        f=features.reindex(pd.DatetimeIndex(labels.confirmed)).drop(columns=['episode_group','episode_sample_due'],errors='ignore')
        y=pd.Series(labels.label.to_numpy(),index=f.index)
        # Overlapping/nearby incidents belong to one time block; embargo handles boundaries.
        values=labels.alarm_cluster_id.astype(str).to_numpy() if 'alarm_cluster_id' in labels else [str(t.to_period('W')) for t in f.index]
        g=pd.Series([f'{unit}::{value}' for value in values],index=f.index)
        dest=directory/'interpretation'/'cause'
        previous=joblib.load(dest/'interpreter.joblib') if (dest/'interpreter.joblib').exists() else None
        statuses['cause']=train_head(f,y,g,cfg,dest,previous,
            label_available_at=pd.Series(pd.to_datetime(labels.label_available_at).to_numpy(),index=f.index))
    (directory/'interpretation_status.json').write_text(json.dumps(statuses,ensure_ascii=False,indent=2),encoding='utf-8')
    if attach:
        attach_heads(directory,statuses,as_of)
    return statuses


def attach_heads(directory,statuses,as_of):
    import joblib,hashlib,shutil,uuid,tempfile
    from monitoring_v3.artifacts import publish_export
    from monitoring_v3.checkpoint_cache import STATE_CONTRACT
    from monitoring_v3.classification import CONTRACT
    exported=Path(directory)/'model'
    if not exported.is_dir():raise ValueError('No export to attach interpretation to')
    if not any(s.get('status') in ('trained','trained_uncalibrated') for s in statuses.values()) and not (
        (exported/'interpreter.joblib').exists() or any(exported.glob('risk_*h.joblib'))):
        return
    metadata=json.loads((exported/'config.json').read_text(encoding='utf-8')) if (exported/'config.json').exists() else {}
    ready=('trained','trained_uncalibrated')
    if any(s.get('status') in ready for s in statuses.values()) and metadata.get('state_contract')!=STATE_CONTRACT:
        raise ValueError('Rebuild the core model before attaching heads: incompatible preparation contract')
    with tempfile.TemporaryDirectory(prefix='.heads-',dir=Path(directory)) as temp:
        staged=Path(temp)/'model';shutil.copytree(exported,staged,ignore=shutil.ignore_patterns('__pycache__'))
        kinds=set(statuses)|{p.stem for p in staged.glob('risk_*h.joblib')}
        if (staged/'interpreter.joblib').exists():kinds.add('cause')
        attached=[]
        for kind in sorted(kinds):
            status=statuses.get(kind,{'status':'not_selected'})
            target=staged/('interpreter.joblib' if kind=='cause' else kind+'.joblib')
            importance_target=staged/(kind+'_feature_importance.csv')
            source=Path(status.get('shared_head',Path(directory)/'interpretation'/kind/'interpreter.joblib'))
            head=joblib.load(source) if status.get('status') in ready and source.exists() else None
            if head is None or head.get('contract')!=CONTRACT:
                if status.get('status') in ready:status['export_attachment']='missing_or_incompatible_head'
                for file in (target,importance_target):
                    if not file.exists():continue
                    history=Path(directory)/'interpretation_export_history';history.mkdir(exist_ok=True)
                    shutil.copy2(file,history/(uuid.uuid4().hex+'__'+file.name));file.unlink()
                continue
            head['available_at']=str(as_of);joblib.dump(head,target)
            source_importance=source.parent/'feature_importance.csv'
            if importance_target.exists():importance_target.unlink()
            if source_importance.exists():shutil.copy2(source_importance,importance_target)
            status['export_attachment']='attached'
            attached.append(dict(kind=kind,available_at=str(as_of),probability_status=head.get('report',{}).get('probability_status'),
                samples=head.get('risk_policy',{}).get('samples')))
        if metadata:
            metadata['interpretation_heads']=attached
            (staged/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        manifest={str(p.relative_to(staged)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(staged.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in str(p)}
        (staged/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        publish_export(staged,exported)
    (Path(directory)/'interpretation_status.json').write_text(json.dumps(statuses,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def train_risk_group(cfg,items,contexts,as_of):
    import hashlib,joblib
    from monitoring_v3.risk_dataset import make_examples
    parts=[]
    for item in items:
        unit=item['unit'];path=Path(item['directory'])/'interpretation_features.parquet'
        parts.append((unit,pd.read_parquet(path),contexts[unit]))
    description=sorted((i.get('experiment') or Path(i['directory']).parent.name,i['unit']) for i in items)
    key=hashlib.sha256(json.dumps(description,ensure_ascii=False).encode()).hexdigest()[:20]
    root=Path(cfg['output'])/'interpretation_groups'/key
    root.mkdir(parents=True,exist_ok=True)
    (root/'group_identity.json').write_text(json.dumps(description,ensure_ascii=False,indent=2),encoding='utf-8')
    statuses={}
    for hours in cfg['risk_horizons']:
        xs,ys,gs,ats,audits={},{},{},{},[]
        for unit,features,context in parts:
            f,y,g,audit,available=make_examples(features,context,hours,as_of,cfg,return_available_at=True)
            xs[unit]=f;ys[unit]=y;gs[unit]=g;ats[unit]=available;audits.append(audit)
        def combine(tables):
            out=pd.concat(tables,names=['unit','time']).swaplevel().sort_index(kind='stable')
            out.index.names=['time','unit'];return out
        x,y,g=combine(xs),combine(ys),combine(gs)
        available=combine(ats)
        x=x.reindex(sorted(x.columns),axis=1)
        directory=root/f'risk_{hours}h'
        previous=joblib.load(directory/'interpreter.joblib') if (directory/'interpreter.joblib').exists() else None
        status=train_head(x,y,g,cfg,directory,previous,horizon_hours=hours,task='risk',label_available_at=available)
        status.update(scope='group',units=sorted(xs),label_audit=audits,shared_head=str((directory/'interpreter.joblib').resolve()))
        (directory/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        statuses[f'risk_{hours}h']=status
    return statuses
