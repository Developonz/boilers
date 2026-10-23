"""Reusable interpretation workflow for the notebook and optional command-line entry point."""
from pathlib import Path
import json
import pandas as pd
from monitoring_v3.data import load_sources
from monitoring_v3.interpretation import train_head, risk_labels


def train(cfg, directory, unit, as_of, attach=False, features_path=None, contexts=None):
    import joblib
    directory = Path(directory)
    if contexts is None:
        contexts,_=load_sources(cfg)
    context=contexts[unit]
    if features_path and attach:
        raise ValueError('Ensemble heads require a matching ensemble inference artifact; do not attach to one detector')
    features=pd.read_parquet(features_path or directory/'interpretation_features.parquet')
    features=features.loc[(features.index < as_of) & context['eligible'].reindex(features.index).fillna(False)]
    statuses={}
    for hours in cfg['risk_horizons']:
        # Sampling removes redundant neighboring rows; independent-event gates are still mandatory.
        f=features.iloc[::max(1,int(pd.Timedelta('6h')/pd.Timedelta(cfg['step'])))]
        y,g=risk_labels(f.index,context['events'],hours,context['frame'].index[-1],as_of)
        dest=directory/'interpretation'/f'risk_{hours}h'
        previous=joblib.load(dest/'interpreter.joblib') if (dest/'interpreter.joblib').exists() else None
        statuses[f'risk_{hours}h']=train_head(f,y,g,cfg,dest,previous)
    labels=pd.read_csv(directory/'episode_labels.csv',parse_dates=['confirmed'])
    available=pd.to_datetime(labels.label_available_at,errors='coerce')
    labels=labels.loc[labels.label.notna() & labels.label.astype(str).str.strip().ne('') & available.le(as_of)].copy()
    if labels.empty:
        statuses['cause']={'status':'requires_reviewed_episode_labels'}
    else:
        labels=labels.sort_values('confirmed').drop_duplicates('confirmed')
        f=features.reindex(pd.DatetimeIndex(labels.confirmed))
        y=pd.Series(labels.label.to_numpy(),index=f.index)
        # Overlapping/nearby incidents belong to one time block; embargo handles boundaries.
        g=pd.Series([f'{unit}::{t.to_period("W")}' for t in f.index],index=f.index)
        dest=directory/'interpretation'/'cause'
        previous=joblib.load(dest/'interpreter.joblib') if (dest/'interpreter.joblib').exists() else None
        statuses['cause']=train_head(f,y,g,cfg,dest,previous)
    (directory/'interpretation_status.json').write_text(json.dumps(statuses,ensure_ascii=False,indent=2),encoding='utf-8')
    if attach:
        import shutil,hashlib
        exported=directory/'model'
        if not exported.is_dir():
            raise ValueError('No export to attach interpretation to')
        # Attaching is explicit because it changes the reviewed inference artifact.
        for kind in ('cause',*[f'risk_{h}h' for h in cfg['risk_horizons']]):
            source=directory/'interpretation'/kind/'interpreter.joblib'
            if source.exists():
                head=joblib.load(source);head['available_at']=str(as_of)
                joblib.dump(head,exported/('interpreter.joblib' if kind=='cause' else kind+'.joblib'))
        manifest={str(p.relative_to(exported)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(exported.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in str(p)}
        (exported/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return statuses

