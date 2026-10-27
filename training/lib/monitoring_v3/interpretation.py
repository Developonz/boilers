"""Episode evidence, reviewable labels, and abstaining CatBoost interpretation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .alarm_scores import alarm_stream,view_policy
from .preparation import FrozenEncoder,categorical_values


def episode_features(errors, context, cfg, groups=None):
    e = errors
    a=e.abs()
    f = {}
    f['mean_abs'] = a.mean(axis=1)
    f['max_abs'] = a.max(axis=1)
    f['signed_mean'] = e.mean(axis=1)
    f['coverage'] = e.notna().mean(axis=1)
    f['fraction_above_1'] = e.abs().gt(1).astype(float).where(e.notna()).mean(axis=1)
    valid=context['eligible'].reindex(e.index).fillna(False)&f['coverage'].ge(cfg['minimum_coverage'])
    contiguous=e.index.to_series().diff().eq(pd.Timedelta(cfg['step']))
    segment=(~valid|~valid.shift(fill_value=False)|~contiguous).cumsum()
    for h in (6, 24, 48):
        f[f'mean_abs_{h}h'] = f['mean_abs'].where(valid).groupby(segment).transform(lambda s:s.rolling(f'{h}h',min_periods=1).mean())
        f[f'persistent_{h}h'] = e.where(valid,axis=0).groupby(segment).transform(lambda s:s.rolling(f'{h}h',min_periods=1).mean()).abs().mean(axis=1)
        f[f'history_{h}h_coverage']=valid.astype(float).groupby(segment).transform(lambda s:s.rolling(f'{h}h',min_periods=1).sum())/(h*pd.Timedelta('1h')/pd.Timedelta(cfg['step']))
    import re
    groups=groups or {}
    families={}
    for c in e:
        family=groups.get(c) or next((key for pattern,key in [
            ('подшип|bearing','bearings'),('температур.*металл|metal','metal_temperature'),
            ('температур|temperature','temperature'),('давлен|pressure','pressure'),
            ('расход|flow','flow'),('уровень|level','level'),('небаланс|balance','balance')]
            if re.search(pattern,c,re.I)),'other')
        families.setdefault(family,[]).append(c)
    for group,cols in families.items():
        prefix='group::'+group+'::';part=e[cols]
        f[prefix+'mean_abs']=part.abs().mean(axis=1)
        f[prefix+'max_abs']=part.abs().max(axis=1)
        f[prefix+'fraction']=part.abs().gt(1).astype(float).where(part.notna()).mean(axis=1)
        f[prefix+'positive_fraction']=part.gt(1).astype(float).where(part.notna()).mean(axis=1)
        f[prefix+'negative_fraction']=part.lt(-1).astype(float).where(part.notna()).mean(axis=1)
        f[prefix+'growth_6h']=f[prefix+'mean_abs'].diff(max(1,int(pd.Timedelta('6h')/pd.Timedelta(cfg['step']))))
    for c in cfg.get('critical_sensors',[]):
        if c in e:f['critical::'+c]=e[c]
    f['growth_1h'] = f['mean_abs'].diff(max(1, int(pd.Timedelta('1h')/pd.Timedelta(cfg['step']))))
    for flag in ('missing','spike','frozen','masked'):
        f['quality_'+flag] = context['quality'].xs(flag, axis=1, level=1).mean(axis=1)
    load = context['frame'][context['load']]
    f['load'] = load
    f['load_delta'] = load.diff(max(1, int(pd.Timedelta('1h')/pd.Timedelta(cfg['step']))))
    f['since_start_hours'] = context['regime'].since_start_hours
    for c in context['regime']:
        if c.startswith('evidence::') or c in ('since_stop_hours','regime_status','evidence_conflict','run_age_known','observed_run_hours','recovering_after_unknown'):
            f['regime::'+c]=context['regime'][c]
    for c, r in context['roles'].items():
        if r == 'state':
            values = context['frame'][c].astype(object)
            f[c] = categorical_values(values)
            f[c+'::change'] = (values.notna() & values.shift().notna() & values.ne(values.shift())).astype(float)
        elif r=='condition':
            values=pd.to_numeric(context['cleaned'][c],errors='coerce')
            f['input::'+c]=values
            f['input::'+c+'::delta_1h']=values.diff(max(1,int(pd.Timedelta('1h')/pd.Timedelta(cfg['step']))))
    f['features_valid']=valid & f['history_24h_coverage'].ge(.5)
    f['unit'] = context['unit']
    return pd.DataFrame(f,index=e.index)


def write_catalog(timelines, features, context, cfg, dest):
    """Features are taken at confirmation, never at the future peak/end of the episode."""
    rows = []
    for view, timeline in timelines.items():
        _, episodes = alarm_stream(timeline.ratio, view_policy(cfg,view))
        for ep in episodes.itertuples():
            t = ep.confirmed
            evidence = features.loc[t]
            changes = [c for c in evidence.index if c.endswith('::change') and evidence[c] == 1]
            hints = []
            if changes or abs(evidence.get('load_delta', 0)) > max(context['threshold']*.1, 1e-6):
                hints.append('possible_mode_change')
            if evidence.get('quality_frozen', 0) > 0 or evidence.get('quality_missing', 0) > .1:
                hints.append('check_measurements')
            key = hashlib.sha256(f"{context['unit']}|{view}|{t.isoformat()}".encode()).hexdigest()[:20]
            rows.append(dict(episode_id=key, unit=context['unit'], view=view, start=ep.start,
                             confirmed=t, end=ep.end, peak=ep.peak, suggestions=';'.join(hints),
                             label='', label_available_at='', reviewer='', reason='',
                             evidence_time=t, label_source='unreviewed'))
    catalog = pd.DataFrame(rows, columns=['episode_id','unit','view','start','confirmed','end','peak',
                         'suggestions','label','label_available_at','reviewer','reason','evidence_time','label_source'])
    catalog['alarm_cluster_id']=''
    until=None;cluster=''
    for i,row in catalog.sort_values('confirmed',kind='stable').iterrows():
        if until is None or row.confirmed>=until+pd.Timedelta(hours=cfg['merge_hours']):
            cluster=hashlib.sha256(f'{context["unit"]}::{row.confirmed}'.encode()).hexdigest()[:20]
        catalog.at[i,'alarm_cluster_id']=cluster
        until=max(until,row.end) if until is not None else row.end
    dest = Path(dest)
    # Do not overwrite previously reviewed human labels on a resumed run.
    path = dest/'episode_labels.csv'
    if path.exists():
        previous=pd.read_csv(path)
        reviewed=previous.loc[previous.label.notna() & previous.label.astype(str).str.strip().ne('')].copy()
        if len(reviewed):
            history=dest/'reviewed_episode_history.csv'
            if history.exists():reviewed=pd.concat([pd.read_csv(history),reviewed],ignore_index=True)
            reviewed.drop_duplicates(['episode_id','label','label_available_at']).to_csv(history,index=False,encoding='utf-8-sig')
        old = previous.set_index('episode_id')
        for i, row in catalog.iterrows():
            if row.episode_id in old.index:
                for col in ('label','label_available_at','reviewer','reason','label_source'):
                    catalog.at[i,col] = old.at[row.episode_id,col]
    catalog.to_csv(path, index=False, encoding='utf-8-sig')


def risk_labels(index, events, horizon_hours, observed_end, as_of, observed=None, min_lead_hours=0, return_audit=False,return_available_at=False):
    """Negative labels require complete follow-up; intervening planned stops censor."""
    y = pd.Series(np.nan, index=index)
    group = pd.Series('', index=index, dtype=object)
    available_at=pd.Series(pd.NaT,index=index,dtype='datetime64[ns]')
    horizon = pd.Timedelta(hours=horizon_hours)
    audit={}
    def note(reason):audit[reason]=audit.get(reason,0)+1
    observed=observed.sort_index().fillna(False).astype(bool) if observed is not None else None
    if observed is not None:
        observed_times=observed.index.as_unit('ns').asi8
        missing=np.r_[0,np.cumsum(~observed.to_numpy())]
        inferred_step=observed.index.to_series().diff().dropna().min()
    for t in index:
        limit=min(t+horizon,pd.Timestamp(observed_end),pd.Timestamp(as_of))
        future = events.loc[events.stop.gt(t) & events.stop.le(limit)].sort_values('stop')
        # Once a known observed failure has occurred, its positive label does
        # not require waiting for the remainder of H after the failure. Only a
        # negative label needs the entire event-free follow-up horizon.
        if future.empty and t+horizon>limit:
            note('incomplete_future');continue
        end=future.stop.iloc[0] if len(future) else t+horizon
        uncertain=events.loc[events.stop.isna() & events.start.gt(t) & events.start.le(end)] if 'start' in events else events.iloc[:0]
        if len(uncertain):note('document_without_observed_stop');continue
        if observed is not None:
            left=np.searchsorted(observed_times,pd.Timestamp(t).value,side='right')
            right=np.searchsorted(observed_times,pd.Timestamp(end).value,side='right')
            expected=int(np.ceil((end-t)/inferred_step)) if pd.notna(inferred_step) else 1
            if right-left<expected or missing[right]!=missing[left]:note('unobserved_future');continue
        if future.empty:
            y.loc[t] = 0.; group.loc[t] = f'background::{t.to_period("W")}'
            available_at.loc[t]=t+horizon
            note('negative')
        else:
            event = future.iloc[0]
            detected=getattr(event,'stop_detected_at',pd.NaT)
            if pd.notna(detected) and detected>pd.Timestamp(as_of):note('stop_not_yet_confirmed');continue
            from .event_registry import event_type_at
            kind,available=event_type_at(event,as_of)
            if pd.isna(available) or available > pd.Timestamp(as_of) or kind != 'Авария':
                note('unknown_or_nonfailure_stop')
                continue
            if event.stop-t<pd.Timedelta(hours=min_lead_hours):note('too_late_to_warn');continue
            y.loc[t] = 1.; group.loc[t] = event.event_id
            available_at.loc[t]=max(pd.Timestamp(available),event.stop,pd.Timestamp(detected) if pd.notna(detected) else event.stop)
            note('positive')
    result=(y,group,audit) if return_audit else (y,group)
    return result+(available_at,) if return_available_at else result


def train_head(features, labels, groups, cfg, dest, previous=None, *, horizon_hours=0, task='cause',label_available_at=None):
    from .classification import train_classifier
    return train_classifier(features,labels,groups,cfg,dest,previous,horizon_hours=horizon_hours,task=task,label_available_at=label_available_at)


def predict_head(package, features):
    from .classification import probabilities
    p,classes=probabilities(package,features)
    i=p.argmax(axis=1);confidence=p.max(axis=1)
    calibrated=package.get('calibrator') is not None
    labels=np.asarray(classes)[i]
    valid=features.get('features_valid',pd.Series(True,index=features.index)).fillna(False)
    return pd.DataFrame(dict(label=np.where(calibrated & valid & (confidence>=package['threshold']),labels,'unknown'),
        confidence=np.where(calibrated & valid,confidence,np.nan),
        probability_status=package.get('report',{}).get('probability_status','unknown')),index=features.index)
