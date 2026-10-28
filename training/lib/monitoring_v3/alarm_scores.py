"""Missing-aware spatial scores and forward-only confirmed alarm episodes."""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def view_policy(cfg,view):
    return dict(cfg,persist_hours=float(pd.Timedelta(cfg['step'])/pd.Timedelta('1h')),
                merge_hours=0) if view.endswith('::event') else cfg


def sensor_scale(errors):
    q = errors.abs().quantile(.99)
    positive = q[q.gt(1e-9)]
    floor = float(positive.median()*.2) if len(positive) else 1.
    # A previously constant target still has an explicit engineering scale upstream.
    return q.where(q.gt(1e-9), 1.).clip(lower=max(floor, 1e-6))


def score_views(errors, scale, cfg, groups=None, event_sensors=(),event_changes=None):
    """Only target columns with real estimates take part in numerator AND denominator."""
    groups = groups or {}
    normalized = errors.div(scale.reindex(errors.columns), axis=1)
    rare=[c for c in event_sensors if c in normalized]
    continuous=normalized.drop(columns=rare)
    outputs, matrices = {}, {}
    step = pd.Timedelta(cfg['step'])
    contiguous=errors.index.to_series().diff().eq(step)
    for temporal, hours in [('instant', 0), ('slow48', cfg['slow_hours'])]:
        e = continuous
        if hours:
            # Do not carry evidence across a stop, unknown load or a gap in scoring.
            valid_row = e.notna().any(axis=1)
            seg = (~valid_row | ~valid_row.shift(fill_value=False) | ~contiguous).cumsum()
            minimum = max(1, int(math.ceil(hours*pd.Timedelta('1h')/step/2)))
            e = e.groupby(seg).transform(lambda x: x.rolling(f'{hours}h', min_periods=minimum).mean()).where(e.notna())
        a = e.abs()
        coverage = a.notna().mean(axis=1) if len(a.columns) else pd.Series(0.,index=a.index)
        good = coverage.ge(cfg['minimum_coverage'])
        out = dict(max=a.max(axis=1), mean=a.mean(axis=1), fraction=a.gt(1).astype(float).where(a.notna()).mean(axis=1))
        # Select the same descending top five without constructing a Series for
        # every timestamp. Missing channels stay out of the denominator.
        values=a.to_numpy(dtype=float,na_value=np.nan)
        count=np.minimum(5,a.notna().sum(axis=1).to_numpy())
        k=min(5,len(a.columns))
        top=np.sort(np.partition(np.where(np.isnan(values),-np.inf,values),
                                 len(a.columns)-k,axis=1)[:,-k:],axis=1)[:,::-1] if k else values
        total=np.where(top == -np.inf,0.,top).sum(axis=1)
        out['top5']=pd.Series(np.divide(total,count,out=np.full(len(a),np.nan),where=count>0),index=a.index)
        for group in sorted(set(groups.values()) - {''}):
            cols = [c for c in a if groups.get(c) == group]
            if not cols:continue
            out['group:'+group] = a[cols].mean(axis=1).where(a[cols].notna().mean(axis=1).ge(cfg['minimum_coverage']))
        for name, s in out.items():
            key = name+'::'+temporal
            cols=[c for c in a if groups.get(c)==name[len('group:'):]] if name.startswith('group:') else list(a)
            local_coverage=a[cols].notna().mean(axis=1) if cols else pd.Series(0.,index=a.index)
            valid = local_coverage.ge(cfg['minimum_coverage']) & s.notna()
            segments = (~valid | ~valid.shift(fill_value=False) | ~contiguous).cumsum()
            smooth = s.where(valid).groupby(segments).transform(
                lambda x: x.rolling(f"{cfg['smooth_hours']}h", min_periods=1).mean())
            outputs[key] = pd.DataFrame(dict(score=s.where(valid), smoothed=smooth.where(valid), coverage=local_coverage))
            matrices[key] = a[cols].where(valid)
    if rare:
        available=normalized[rare].notna()
        changes=event_changes.reindex(index=errors.index,columns=rare).astype(float).where(available) if event_changes is not None else pd.DataFrame(np.nan,index=errors.index,columns=rare)
        score=changes.sum(axis=1,min_count=1)
        outputs['rare_changes::event']=pd.DataFrame(dict(score=score,smoothed=score,coverage=available.mean(axis=1)))
        matrices['rare_changes::event']=changes
    return outputs, matrices


def _alarm_runs(values, times, contiguous, required, step_ns, merge_ns):
    """Array-only episode boundaries, shared by reporting and threshold search."""
    above=np.isfinite(values)&(values>1)
    starts=np.flatnonzero(above&(~np.r_[False,above[:-1]]|~contiguous))
    ends=np.flatnonzero(above&(~np.r_[above[1:],False]|~np.r_[contiguous[1:],False]))+1
    enough=ends-starts>=required
    starts,ends=starts[enough],ends[enough]
    confirmed=starts+required-1
    notify=np.ones(len(starts),dtype=bool)
    if len(starts)>1:
        notify[1:]=(times[confirmed[1:]]-(times[ends[:-1]-1]+step_ns))>=merge_ns
    return starts,ends,confirmed,notify


def alarm_stream(ratio, cfg):
    """No retrospective activation or bridging. merge_hours suppresses repeat notices only."""
    ix=ratio.index;step=pd.Timedelta(cfg['step']);n=len(ix)
    if not n:
        return pd.DataFrame(dict(ratio=ratio,active=[],notification=[]),index=ix),pd.DataFrame(
            columns=['start','confirmed','end','notification','peak'])
    values=ratio.to_numpy(dtype=float,na_value=np.nan)
    times=ix.as_unit('ns').asi8
    contiguous=np.r_[False,np.diff(times)==step.value]
    required=max(1,int(math.ceil(pd.Timedelta(hours=cfg['persist_hours'])/step)))
    starts,ends,confirmed,notify=_alarm_runs(values,times,contiguous,required,step.value,
                                          pd.Timedelta(hours=cfg['merge_hours']).value)
    changes=np.zeros(n+1,dtype=int)
    changes[confirmed]+=1;changes[ends]-=1
    active=np.cumsum(changes[:-1])>0
    notices=np.zeros(n,dtype=bool)
    notices[confirmed]=notify
    # Peaks start at confirmation, matching the original streaming algorithm.
    rows=[dict(start=ix[a],confirmed=ix[c],end=ix[b-1]+step,notification=bool(flag),
               peak=float(values[c:b].max())) for a,c,b,flag in zip(starts,confirmed,ends,notify)]
    return pd.DataFrame(dict(ratio=ratio,active=active,notification=notices),index=ix),pd.DataFrame(
        rows, columns=['start', 'confirmed', 'end', 'notification', 'peak'])


def calibrate_threshold(score, cfg, budget=None):
    """Past calibration only. Infinity means insufficient data, not 'healthy'."""
    budget = cfg['alarm_budget'] if budget is None else budget
    valid = score.notna() & np.isfinite(score)
    hours = valid.sum()*pd.Timedelta(cfg['step'])/pd.Timedelta('1h')
    if valid.sum() < cfg['calibration_rows']:
        return dict(threshold=np.inf, hours=float(hours), status='insufficient_calibration')
    values = score[valid]
    candidates = np.unique(np.r_[np.maximum(values.quantile(np.linspace(0, 1, 101)), 1e-9),
                                  max(float(values.max())+1e-9, 1e-9)])
    raw=score.to_numpy(dtype=float,na_value=np.nan)
    step=pd.Timedelta(cfg['step']);times=score.index.as_unit('ns').asi8
    contiguous=np.r_[False,np.diff(times)==step.value]
    required=max(1,int(math.ceil(pd.Timedelta(hours=cfg['persist_hours'])/step)))
    merge_ns=pd.Timedelta(hours=cfg['merge_hours']).value
    for threshold in candidates:
        _,ends,confirmed,notices=_alarm_runs(raw/threshold,times,contiguous,required,step.value,merge_ns)
        rate = np.count_nonzero(notices)*720/hours
        fraction = float(np.sum(ends-confirmed)/valid.sum())
        if rate <= budget and fraction <= cfg['max_background_fraction']:
            from scipy.stats import chi2
            count=int(np.count_nonzero(notices))
            return dict(threshold=float(threshold), hours=float(hours), status='calibrated',
                        alarm_rate=float(rate), alarm_fraction=fraction, budget=float(budget),
                        rate_resolution=float(720/hours),
                        evidence='limited' if hours<max(720,720/budget) else 'multi_month',
                        rate_low=float(.5*chi2.ppf(.025,2*count)*720/hours) if count else 0.,
                        rate_high=float(.5*chi2.ppf(.975,2*(count+1))*720/hours))
    raise AssertionError('Upper threshold must yield no alarms')


def _precision(timeline,positive,background,uncertain,valid,cfg,start,end):
    """Notification precision; unknown/neutral/unobserved outcomes are excluded."""
    ix=timeline.index;step=pd.Timedelta(cfg['step']);horizon=pd.Timedelta(days=cfg['pre_window_days'])
    observed=timeline.get('load_observed',pd.Series(True,index=ix)).fillna(False).to_numpy(bool)
    times=ix.as_unit('ns').asi8
    left=np.arange(len(ix))+1;right=np.searchsorted(times,times+horizon.value,side='right')
    bad=np.r_[0,np.cumsum(~observed)]
    expected=int(math.ceil(horizon/step))
    complete=pd.Series((times+horizon.value<=times[-1])&(right-left>=expected)&(bad[right]==bad[left]),index=ix)
    true=positive&~uncertain&valid;false=background&complete
    notices=timeline.notification.fillna(False)&valid
    tp=int((notices&true).sum());fp=int((notices&false).sum())
    active=timeline.active.fillna(False).to_numpy(bool)
    contiguous=np.r_[False,np.diff(times)==step.value]
    starts=np.flatnonzero(active&(~np.r_[False,active[:-1]]|~contiguous))
    ends=np.flatnonzero(active&(~np.r_[active[1:],False]|~np.r_[contiguous[1:],False]))+1
    etp=efp=excluded=0
    for a,b in zip(starts,ends):
        if not start<=ix[a]<end or not valid.iloc[a]:continue
        if true.iloc[a:b].any():etp+=1
        elif false.iloc[a:b].any():efp+=1
        else:excluded+=1
    return dict(precision=tp/(tp+fp) if tp+fp else np.nan,
        precision_true_notifications=tp,precision_false_notifications=fp,
        precision_excluded_notifications=int(notices.sum())-tp-fp,
        precision_incomplete_future_notifications=int((notices&background&~complete).sum()),
        episode_precision=etp/(etp+efp) if etp+efp else np.nan,
        precision_true_episodes=etp,precision_false_episodes=efp,precision_excluded_episodes=excluded)


def evaluate(timeline, events, cfg, start=None, end=None):
    ix = timeline.index
    step = pd.Timedelta(cfg['step'])
    start, end = pd.Timestamp(start or ix[0]), pd.Timestamp(end or ix[-1]+step)
    valid = timeline.score.notna() & np.isfinite(timeline.ratio) & (ix >= start) & (ix < end)
    positive = pd.Series(False, index=ix)
    background = pd.Series(True, index=ix)
    leads, details = [], []
    uncertain=pd.Series(False,index=ix)
    for e in events.itertuples():
        if e.type=='Неизвестно' and pd.notna(e.stop):
            uncertain|=(ix>=e.stop-pd.Timedelta(days=cfg['pre_window_days']))&(ix<e.stop)
        if e.type != 'Авария' or pd.isna(e.stop):
            continue
        left = e.stop-pd.Timedelta(days=cfg['pre_window_days'])
        right = e.stop-pd.Timedelta(hours=cfg['min_lead_hours'])
        window = (ix >= left) & (ix < right)
        positive |= window
        background &= ~((ix >= e.stop-pd.Timedelta(days=cfg['pre_window_days']+cfg['neutral_days'])) & (ix < e.stop))
        if not start <= e.stop < end:
            continue
        notifications = timeline.index[window & valid & timeline.notification]
        ongoing = bool((window & valid & timeline.active).any())
        caught = bool(len(notifications))
        lead = float((e.stop-notifications[0])/pd.Timedelta('1h')) if caught else np.nan
        if caught:
            leads.append(lead)
        details.append(dict(stop=e.stop, detected=caught, ongoing_alarm=ongoing and not caught,
                            event_id=e.event_id if hasattr(e,'event_id') else str(e.stop),
                            observable=bool((window&valid).any()),
                            lead_hours=lead, observed_hours=float((window & valid).sum()*step/pd.Timedelta('1h')),
                            possible_hours=float(np.sum(window & (ix >= start) & (ix < end))*step/pd.Timedelta('1h'))))
    background &= valid & ~uncertain & timeline.get('background_eligible', pd.Series(True,index=ix)).fillna(False)
    used = (positive | background) & valid
    labels = positive[used].astype(int)
    finite_ratio = timeline.ratio[used].replace([np.inf, -np.inf], np.nan)
    known = finite_ratio.notna()
    ap = float(average_precision_score(labels[known], finite_ratio[known])) if labels[known].nunique() == 2 else np.nan
    hours = float(background.sum()*step/pd.Timedelta('1h'))
    caught = sum(x['detected'] for x in details)
    summary = dict(events=len(details), detected=caught, recall=caught/len(details) if details else np.nan,
                   unanchored_events=int(((events.type=='Авария')&events.stop.isna()&events.start.ge(start)&events.start.lt(end)).sum()) if 'start' in events else 0,
                   observable_events=sum(x['observable'] for x in details),
                   unobserved_events=sum(not x['observable'] for x in details),
                   recall_observed=(sum(x['detected'] for x in details if x['observable'])/sum(x['observable'] for x in details)
                                    if any(x['observable'] for x in details) else np.nan),
                   uncertain_hours=float((uncertain&valid).sum()*step/pd.Timedelta('1h')),
                   uncertain_notifications=int((uncertain&valid&timeline.notification).sum()),
                   ongoing_events=sum(x['ongoing_alarm'] for x in details),
                   background_hours=hours, background_notifications=int((timeline.notification & background).sum()),
                   alarms_per_720h=float((timeline.notification & background).sum()*720/hours) if hours else np.nan,
                   background_fraction=float(timeline.active[background].mean()) if hours else np.nan,
                   lead_hours=float(np.median(leads)) if leads else np.nan, average_precision=ap,
                   random_precision=float(labels[known].mean()) if known.any() else np.nan,
                   calculated_hours=float(valid.sum()*step/pd.Timedelta('1h')),
                   unscored_hours=float(((ix >= start) & (ix < end) & ~valid).sum()*step/pd.Timedelta('1h')))
    period=(ix>=start)&(ix<end)
    eligible=timeline.get('eligible',pd.Series(True,index=ix)).eq(True)&period
    summary['working_unscored_hours']=float((eligible&~valid).sum()*step/pd.Timedelta('1h'))
    summary['non_working_hours']=float((period&~eligible).sum()*step/pd.Timedelta('1h'))
    summary['availability_operating']=float((eligible&valid).sum()/eligible.sum()) if eligible.any() else np.nan
    summary.update(_precision(timeline,positive,background,uncertain,valid,cfg,start,end))
    return summary, pd.DataFrame(details)
