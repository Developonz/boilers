"""Missing-aware spatial scores and forward-only confirmed alarm episodes."""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def sensor_scale(errors):
    q = errors.abs().quantile(.99)
    positive = q[q.gt(1e-9)]
    floor = float(positive.median()*.2) if len(positive) else 1.
    # A previously constant target still has an explicit engineering scale upstream.
    return q.where(q.gt(1e-9), 1.).clip(lower=max(floor, 1e-6))


def score_views(errors, scale, cfg, groups=None):
    """Only target columns with real estimates take part in numerator AND denominator."""
    groups = groups or {}
    normalized = errors.div(scale.reindex(errors.columns), axis=1)
    outputs, matrices = {}, {}
    step = pd.Timedelta(cfg['step'])
    for temporal, hours in [('instant', 0), ('slow48', cfg['slow_hours'])]:
        e = normalized
        if hours:
            # Do not carry evidence across a stop, unknown load or a gap in scoring.
            valid_row = e.notna().any(axis=1)
            seg = (~valid_row | ~valid_row.shift(fill_value=False)).cumsum()
            minimum = max(1, int(math.ceil(hours*pd.Timedelta('1h')/step/2)))
            e = e.groupby(seg).transform(lambda x: x.rolling(f'{hours}h', min_periods=minimum).mean()).where(e.notna())
        a = e.abs()
        coverage = a.notna().mean(axis=1)
        good = coverage.ge(cfg['minimum_coverage'])
        out = dict(max=a.max(axis=1), mean=a.mean(axis=1), fraction=a.gt(1).where(a.notna()).mean(axis=1))
        out['top5'] = a.apply(lambda r: r.nlargest(min(5, r.count())).mean(), axis=1)
        for group in sorted(set(groups.values()) - {''}):
            cols = [c for c in a if groups.get(c) == group]
            out['group:'+group] = a[cols].mean(axis=1).where(a[cols].notna().mean(axis=1).ge(cfg['minimum_coverage']))
        for name, s in out.items():
            key = name+'::'+temporal
            valid = good & s.notna()
            segments = (~valid | ~valid.shift(fill_value=False)).cumsum()
            smooth = s.where(valid).groupby(segments).transform(
                lambda x: x.rolling(f"{cfg['smooth_hours']}h", min_periods=1).mean())
            outputs[key] = pd.DataFrame(dict(score=s.where(valid), smoothed=smooth.where(valid), coverage=coverage))
            matrices[key] = a.where(good)
    return outputs, matrices


def alarm_stream(ratio, cfg):
    """No retrospective activation or bridging. merge_hours suppresses repeat notices only."""
    ix = ratio.index
    step = pd.Timedelta(cfg['step'])
    count, first, last_time, last_end = 0, None, None, None
    confirmed = False
    active, notices, rows = [], [], []
    current = None
    for t, v in ratio.items():
        contiguous = last_time is not None and t-last_time == step
        above = np.isfinite(v) and v > 1
        if not contiguous or not above:
            if current is not None:
                current['end'] = last_time+step
                rows.append(current)
                last_end = current['end']
                current = None
            count, first, confirmed = 0, None, False
        notice = False
        if above:
            first = first if count else t
            count += 1
            if not confirmed and count*step >= pd.Timedelta(hours=cfg['persist_hours']):
                confirmed = True
                notice = last_end is None or t-last_end >= pd.Timedelta(hours=cfg['merge_hours'])
                current = dict(start=first, confirmed=t, end=t+step, notification=notice, peak=float(v))
            if current is not None:
                current['peak'] = max(current['peak'], float(v))
        active.append(confirmed and above)
        notices.append(notice)
        last_time = t
    if current is not None:
        current['end'] = ix[-1]+step
        rows.append(current)
    return pd.DataFrame(dict(ratio=ratio, active=active, notification=notices), index=ix), pd.DataFrame(
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
    for threshold in candidates:
        stream, _ = alarm_stream(score/threshold, cfg)
        rate = stream.notification.sum()*720/hours
        fraction = float(stream.active[valid].mean())
        if rate <= budget and fraction <= cfg['max_background_fraction']:
            return dict(threshold=float(threshold), hours=float(hours), status='calibrated',
                        alarm_rate=float(rate), alarm_fraction=fraction, budget=float(budget))
    raise AssertionError('Upper threshold must yield no alarms')


def evaluate(timeline, events, cfg, start=None, end=None):
    ix = timeline.index
    step = pd.Timedelta(cfg['step'])
    start, end = pd.Timestamp(start or ix[0]), pd.Timestamp(end or ix[-1]+step)
    valid = timeline.score.notna() & np.isfinite(timeline.ratio) & (ix >= start) & (ix < end)
    positive = pd.Series(False, index=ix)
    background = pd.Series(True, index=ix)
    leads, details = [], []
    for e in events.itertuples():
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
                            lead_hours=lead, observed_hours=float((window & valid).sum()*step/pd.Timedelta('1h')),
                            possible_hours=float(np.sum(window & (ix >= start) & (ix < end))*step/pd.Timedelta('1h'))))
    background &= valid & timeline.get('background_eligible', pd.Series(True,index=ix)).fillna(False)
    used = (positive | background) & valid
    labels = positive[used].astype(int)
    finite_ratio = timeline.ratio[used].replace([np.inf, -np.inf], np.nan)
    known = finite_ratio.notna()
    ap = float(average_precision_score(labels[known], finite_ratio[known])) if labels[known].nunique() == 2 else np.nan
    hours = float(background.sum()*step/pd.Timedelta('1h'))
    caught = sum(x['detected'] for x in details)
    summary = dict(events=len(details), detected=caught, recall=caught/len(details) if details else np.nan,
                   ongoing_events=sum(x['ongoing_alarm'] for x in details),
                   background_hours=hours, background_notifications=int((timeline.notification & background).sum()),
                   alarms_per_720h=float((timeline.notification & background).sum()*720/hours) if hours else np.nan,
                   background_fraction=float(timeline.active[background].mean()) if hours else np.nan,
                   lead_hours=float(np.median(leads)) if leads else np.nan, average_precision=ap,
                   random_precision=float(labels[known].mean()) if known.any() else np.nan,
                   calculated_hours=float(valid.sum()*step/pd.Timedelta('1h')),
                   unscored_hours=float(((ix >= start) & (ix < end) & ~valid).sum()*step/pd.Timedelta('1h')))
    return summary, pd.DataFrame(details)
