"""Compare identical operating hours and bootstrap independent events/time blocks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .alarm_scores import evaluate
from .experiments import event_bootstrap


def common_comparison(root, contexts, cfg):
    root=Path(root);families={}
    for p in sorted((root/'experiments').glob('*/experiment.json')):
        spec=json.loads(p.read_text(encoding='utf-8'))
        for unit in spec['recipients']:
            directory=p.parent/unit
            if not directory.exists():
                continue
            for b in cfg['alarm_budgets']:
                for path in directory.glob('*__'+str(b)+'.parquet'):
                    # Compare like-for-like schemes. Missing rare-event evidence
                    # must not erase the continuous detectors' comparison period.
                    family=(spec['group'],spec['scenario'],'+'.join(spec['sources']),unit,path.stem.rsplit('__',1)[0],b)
                    families.setdefault(family,[]).append((p.parent.name,path,b))
    rows=[]
    for family,items in families.items():
        frames={str(p):pd.read_parquet(p) for _,p,_ in items}
        masks={key: d.score.notna() & np.isfinite(d.ratio) for key,d in frames.items()}
        shared=pd.concat(masks,axis=1).fillna(False).all(axis=1)
        for name,path,b in items:
            d=frames[str(path)].copy()
            d.loc[~shared.reindex(d.index).fillna(False),'score']=np.nan
            summary,events=evaluate(d,contexts[family[3]]['events'],cfg,start=cfg['first_test'])
            # Keep the original notification stream; no new notices at mask boundaries.
            ci=event_bootstrap(events)
            valid=d.score.notna() & np.isfinite(d.ratio) & (d.index>=pd.Timestamp(cfg['first_test']))
            hourly=pd.DataFrame(dict(active=d.active.where(valid),notification=d.notification.where(valid),valid=valid.astype(int)))
            blocks=hourly.resample('7D').agg(dict(active='sum',notification='sum',valid='sum'))
            blocks=blocks.loc[blocks.valid.gt(0)]
            rng=np.random.default_rng(cfg['seed']);rate_samples=[]
            step_h=float(pd.Timedelta(cfg['step'])/pd.Timedelta('1h'))
            if len(blocks)>=2:
                for _ in range(300):
                    sample=blocks.iloc[rng.integers(0,len(blocks),len(blocks))]
                    rate_samples.append(float(sample.notification.sum()*720/(sample.valid.sum()*step_h)))
            rows.append(dict(group=family[0],scenario=family[1],sources=family[2],unit=family[3],
                experiment=name,view=path.stem.rsplit('__',1)[0],budget=b,**summary,**{k:v for k,v in ci.items() if k!='events'},
                weekly_notification_rate_low=float(np.quantile(rate_samples,.025)) if rate_samples else np.nan,
                weekly_notification_rate_high=float(np.quantile(rate_samples,.975)) if rate_samples else np.nan))
    result=pd.DataFrame(rows)
    result.to_csv(root/'comparison_common_hours.csv',index=False,encoding='utf-8-sig')
    return result
