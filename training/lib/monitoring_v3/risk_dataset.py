"""Causal decision points for continuous or alarm-conditional failure risk."""
import numpy as np
import pandas as pd
from .interpretation import risk_labels


def alarm_features(features,timeline,cfg):
    out=features.copy();active=timeline.active.reindex(out.index).fillna(False)
    step=pd.Timedelta(cfg['step'])
    contiguous=out.index.to_series().diff().eq(step)
    segment=(~active|~active.shift(fill_value=False)|~contiguous).cumsum()
    age=active.groupby(segment).cumcount()*step/pd.Timedelta('1h')
    out['episode_active']=active.astype(float)
    out['episode_age_hours']=age.clip(upper=48).where(active)
    out['episode_sample_due']=np.isclose(age.mod(cfg['risk_policy']['sample_hours']),0)&active
    starts=out.index.to_series().where(active & (~active.shift(fill_value=False)|~contiguous)).ffill()
    out['episode_group']=starts.dt.strftime('%Y-%m-%dT%H:%M:%S').where(active,'')
    return out


def make_examples(features,context,hours,as_of,cfg,return_available_at=False):
    f=features.loc[features.index<as_of].copy()
    valid=f['features_valid'].fillna(False).astype(bool) if 'features_valid' in f else context['eligible'].reindex(f.index).fillna(False)
    if cfg['risk_policy']['samples']=='episodes':
        active=f.get('episode_active',pd.Series(0.,index=f.index)).eq(1)
        age=f.get('episode_age_hours',pd.Series(np.nan,index=f.index))
        # Sampling is anchored at each confirmation, not at every sixth retained row.
        sample=f.get('episode_sample_due',pd.Series(False,index=f.index)).fillna(False)
        valid &= active & sample
    else:
        step=pd.Timedelta(hours=cfg['risk_policy']['sample_hours']).value
        valid &= (f.index.as_unit('ns').asi8%step)==0
    f=f.loc[valid]
    load=context.get('load')
    cleaned=context.get('cleaned',context['frame'])
    observed=cleaned[load].notna() if load in cleaned else pd.Series(True,index=context['frame'].index)
    y,g,audit,available=risk_labels(f.index,context['events'],hours,context['frame'].index[-1],as_of,
        observed=observed,min_lead_hours=cfg['min_lead_hours'],return_audit=True,return_available_at=True)
    unit=context.get('unit',str(features['unit'].iloc[0]) if len(features) and 'unit' in features else '')
    background='background::'+f.episode_group.astype(str) if cfg['risk_policy']['samples']=='episodes' and 'episode_group' in f else g
    g=g.where(y.ne(0),background)
    g=unit+'::'+g
    from .event_registry import event_type_at
    known_emergencies=[]
    for event in context['events'].itertuples():
        kind,date=event_type_at(event,as_of)
        detected=getattr(event,'stop_detected_at',event.stop)
        if (kind=='Авария' and pd.notna(event.stop) and event.stop<=pd.Timestamp(as_of)
            and pd.notna(date) and date<=pd.Timestamp(as_of)
            and (pd.isna(detected) or detected<=pd.Timestamp(as_of))):
            known_emergencies.append(str(event.event_id))
    represented=set(g.loc[y.eq(1)])
    absent=[event for event in known_emergencies if unit+'::'+event not in represented]
    # Identifiers are for grouping/traceability, never predictors.
    f=f.drop(columns=['episode_group','episode_sample_due'],errors='ignore')
    audit.update(candidate_rows=len(f),observed_label_rows=int(y.notna().sum()),
        positive_events=int(g.loc[y.eq(1)].nunique()),negative_groups=int(g.loc[y.eq(0)].nunique()),
        samples=cfg['risk_policy']['samples'],unit=unit)
    audit.update(registered_emergencies=len(known_emergencies),emergencies_without_positive_examples=len(absent),
        emergency_ids_without_examples=absent)
    return (f,y,g,audit,available) if return_available_at else (f,y,g,audit)
