"""Optional investigations; outputs explicitly distinguish synthetic evidence from real events."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def inject(frame, sensor, start, kind, magnitude, hours=48, step='1h'):
    out=frame.copy();start=pd.Timestamp(start)
    n=max(1,int(pd.Timedelta(hours=hours)/pd.Timedelta(step)))
    pos=np.flatnonzero((frame.index >= start)&(frame.index < start+pd.Timedelta(hours=hours)))
    x=pd.to_numeric(frame[sensor],errors='coerce')
    if not len(pos):
        raise ValueError('No injection interval')
    if kind=='drift':
        out.iloc[pos,out.columns.get_loc(sensor)]=x.iloc[pos].to_numpy()+np.linspace(0,magnitude,len(pos))
    elif kind in ('step','balance'):
        out.iloc[pos,out.columns.get_loc(sensor)]=x.iloc[pos].to_numpy()+magnitude
    elif kind=='frozen':
        before=x.loc[x.index < start].dropna()
        if before.empty:
            raise ValueError('Frozen injection needs a previous measurement')
        out.iloc[pos,out.columns.get_loc(sensor)]=before.iloc[-1]
    elif kind=='delay':
        shift=max(1,int(round(magnitude)))
        out.iloc[pos,out.columns.get_loc(sensor)]=x.shift(shift).iloc[pos].to_numpy()
    else:
        raise ValueError('Unknown injection type')
    return out


def sensitivity(model, frame, sensor, start, cfg, dest):
    baseline=model.score(frame)
    training_scale=float(model.state['model'].scale[model.state['model'].targets.index(sensor)])
    rows=[]
    for kind in ('step','drift','frozen','delay','balance'):
        for severity in (1.,3.,6.):
            magnitude=severity*training_scale if kind!='delay' else severity
            altered=inject(frame,sensor,start,kind,magnitude,step=cfg['step'])
            result=model.score(altered)
            for view in result.hourly.columns.get_level_values(0).unique():
                part=result.hourly[view].loc[pd.Timestamp(start):pd.Timestamp(start)+pd.Timedelta('48h')]
                notices=part.index[part['уведомление']]
                original=baseline.hourly[view].reindex(part.index)
                rows.append(dict(kind=kind,severity=severity,view=view,synthetic=True,
                    delay_hours=float((notices[0]-pd.Timestamp(start))/pd.Timedelta('1h')) if len(notices) else np.nan,
                    peak_ratio=float(part['отношение'].max()),baseline_peak=float(original['отношение'].max())))
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'synthetic_sensitivity.csv',index=False,encoding='utf-8-sig')
    (dest/'scope.txt').write_text('Synthetic perturbations test code sensitivity only. Balance injection is a sensor offset, not a validated physical simulation. No production quality claim.',encoding='utf-8')
    return pd.DataFrame(rows)


def event_bootstrap(event_results, repeats=1000, seed=42):
    """Resample independent events, not their correlated hours."""
    if not len(event_results):
        return dict(events=0,recall_low=None,recall_high=None)
    rng=np.random.default_rng(seed);v=event_results.detected.astype(float).to_numpy()
    means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(repeats)])
    return dict(events=len(v),recall_low=float(np.quantile(means,.025)),recall_high=float(np.quantile(means,.975)))


def ensemble_features(root, unit):
    """Join prequential evidence from independent detectors without future label features."""
    root=Path(root);parts=[]
    for path in sorted(root.glob('experiments/*/'+unit+'/interpretation_features.parquet')):
        frame=pd.read_parquet(path)
        selected=frame.select_dtypes(include='number')
        selected=selected.add_prefix(path.parent.parent.name+'::')
        parts.append(selected)
    if not parts:
        raise ValueError('No prequential detector outputs')
    return pd.concat(parts,axis=1)


def survival_landmarks(features, events, observed_end, as_of):
    rows=[]
    known=events.loc[events.label_available_at.le(as_of)].sort_values('stop')
    for t in features.index:
        end=min(pd.Timestamp(observed_end),pd.Timestamp(as_of))
        if end <= t:
            continue
        next_events=known.loc[known.stop.gt(t)&known.stop.le(end)]
        e=next_events.iloc[0] if len(next_events) else None
        stop=e.stop if e is not None else end
        duration=float((stop-t)/pd.Timedelta('1h'))
        failure=e is not None and e.type=='Авария'
        rows.append(dict(time=t,duration_hours=duration,event=failure,
                         cohort=e.event_id if e is not None else 'right_censored',
                         label=duration if failure else -duration))
    return pd.DataFrame(rows).set_index('time') if rows else pd.DataFrame()


def survival_experiment(features, events, cfg, as_of, observed_end, dest):
    from catboost import CatBoostRegressor
    from .preparation import FrozenEncoder
    import joblib
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    # Weekly landmarks avoid treating every hour as a separate life.
    f=features.resample('7D').first().dropna(how='all')
    labels=survival_landmarks(f,events,observed_end,as_of)
    status=dict(status='insufficient_independent_events',assumption='planned stops are censored; censoring may be informative')
    if len(labels)>=20 and labels.loc[labels.event,'cohort'].nunique()>=5:
        boundary=labels.index[int(len(labels)*.7)]
        tr=(labels.index < boundary)&(labels.index+pd.to_timedelta(labels.duration_hours,unit='h') < boundary)
        te=labels.index >= boundary
        te &= ~labels.cohort.isin(set(labels.loc[tr,'cohort'])).to_numpy()
        if labels.loc[tr&labels.event,'cohort'].nunique()>=3 and labels.loc[te&labels.event,'cohort'].nunique()>=1:
            enc=FrozenEncoder.fit(f.reindex(labels.index[tr]))
            model=CatBoostRegressor(loss_function='Cox',iterations=cfg['cb_initial_trees'],depth=cfg['cb_depth'],
                thread_count=cfg['threads'],random_seed=cfg['seed'],has_time=True,allow_writing_files=False,verbose=False)
            # Equal total weight per operating spell/cohort.
            counts=labels.loc[tr,'cohort'].map(labels.loc[tr,'cohort'].value_counts())
            model.fit(enc.native(f.reindex(labels.index[tr])),labels.loc[tr,'label'],
                      cat_features=list(enc.categories),sample_weight=1/counts.to_numpy())
            predictions=model.predict(enc.native(f.reindex(labels.index[te])))
            test=labels.loc[te].assign(relative_risk_score=predictions)
            concordant,total=0.,0
            for i in range(len(test)):
                if not test.event.iloc[i]:
                    continue
                compare=test.duration_hours.to_numpy()>test.duration_hours.iloc[i]
                concordant+=float(np.sum(predictions[i]>predictions[compare])+.5*np.sum(predictions[i]==predictions[compare]))
                total+=int(compare.sum())
            status.update(status='trained_experimental',test_concordance=concordant/total if total else None,
                          train_cohorts=int(labels.loc[tr,'cohort'].nunique()),test_cohorts=int(test.cohort.nunique()))
            joblib.dump(dict(model=model,encoder=enc,status=status),dest/'survival.joblib')
            test.to_csv(dest/'heldout_landmarks.csv')
    (dest/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    return status
