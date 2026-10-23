"""Episode evidence, reviewable labels, and abstaining CatBoost interpretation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .alarm_scores import alarm_stream
from .preparation import FrozenEncoder


def episode_features(errors, context, cfg):
    e = errors
    f = pd.DataFrame(index=e.index)
    f['mean_abs'] = e.abs().mean(axis=1)
    f['max_abs'] = e.abs().max(axis=1)
    f['signed_mean'] = e.mean(axis=1)
    f['coverage'] = e.notna().mean(axis=1)
    f['fraction_above_1'] = e.abs().gt(1).where(e.notna()).mean(axis=1)
    for h in (6, 24, 48):
        f[f'mean_abs_{h}h'] = f.mean_abs.rolling(f'{h}h', min_periods=1).mean()
        f[f'persistent_{h}h'] = e.rolling(f'{h}h', min_periods=1).mean().abs().mean(axis=1)
    f['growth_1h'] = f.mean_abs.diff(max(1, int(pd.Timedelta('1h')/pd.Timedelta(cfg['step']))))
    for flag in ('missing','spike','frozen','masked'):
        f['quality_'+flag] = context['quality'].xs(flag, axis=1, level=1).mean(axis=1)
    load = context['frame'][context['load']]
    f['load'] = load
    f['load_delta'] = load.diff(max(1, int(pd.Timedelta('1h')/pd.Timedelta(cfg['step']))))
    f['since_start_hours'] = context['regime'].since_start_hours
    for c, r in context['roles'].items():
        if r == 'state':
            values = context['frame'][c].astype(object)
            f[c] = values.fillna('__MISSING__').astype(str)
            f[c+'::change'] = (values.notna() & values.shift().notna() & values.ne(values.shift())).astype(float)
    f['unit'] = context['unit']
    return f


def write_catalog(timelines, features, context, cfg, dest):
    """Features are taken at confirmation, never at the future peak/end of the episode."""
    rows = []
    for view, timeline in timelines.items():
        _, episodes = alarm_stream(timeline.ratio, cfg)
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
    dest = Path(dest)
    # Do not overwrite previously reviewed human labels on a resumed run.
    path = dest/'episode_labels.csv'
    if path.exists():
        old = pd.read_csv(path).set_index('episode_id')
        for i, row in catalog.iterrows():
            if row.episode_id in old.index:
                for col in ('label','label_available_at','reviewer','reason','label_source'):
                    catalog.at[i,col] = old.at[row.episode_id,col]
    catalog.to_csv(path, index=False, encoding='utf-8-sig')


def risk_labels(index, events, horizon_hours, observed_end, as_of):
    """Negative labels require complete follow-up; intervening planned stops censor."""
    y = pd.Series(np.nan, index=index)
    group = pd.Series('', index=index, dtype=object)
    horizon = pd.Timedelta(hours=horizon_hours)
    for t in index:
        if t+horizon > min(pd.Timestamp(observed_end), pd.Timestamp(as_of)):
            continue
        future = events.loc[events.stop.gt(t) & events.stop.le(t+horizon)].sort_values('stop')
        if future.empty:
            y.loc[t] = 0.; group.loc[t] = f'background::{t.to_period("W")}'
        else:
            event = future.iloc[0]
            if event.label_available_at > pd.Timestamp(as_of) or event.type != 'Авария':
                continue
            y.loc[t] = 1.; group.loc[t] = event.event_id
    return y, group


def _split(frame, embargo):
    times = pd.DatetimeIndex(sorted(frame.index.unique()))
    if len(times) < 30:
        return None
    a, b = times[int(len(times)*.6)], times[int(len(times)*.8)]
    return frame.index < a-embargo, (frame.index >= a) & (frame.index < b-embargo), frame.index >= b


def train_head(features, labels, groups, cfg, dest, previous=None):
    """Fit/gate/calibrate/test by time. Small event sets return an explicit missing-evidence status."""
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, average_precision_score
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    good = labels.notna() & features.notna().any(axis=1)
    x, y, g = features.loc[good].sort_index(), labels.loc[good].sort_index(), groups.loc[good].sort_index()
    split = _split(x, pd.Timedelta(hours=max(cfg['risk_horizons'])))
    status = dict(status='insufficient_independent_labels', rows=len(x), groups=int(g.nunique()))
    if split is None:
        (dest/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8'); return status
    tr, cal, te = split
    # No event (or weekly background block) may occur in two partitions.
    train_groups = set(g.loc[tr]); cal &= ~g.isin(train_groups).to_numpy()
    test_groups = set(g.loc[cal]); te &= ~g.isin(train_groups | test_groups).to_numpy()
    classes = sorted(y.loc[tr].astype(str).unique())
    enough = len(classes) >= 2
    for part in (tr, cal, te):
        enough &= len(y.loc[part]) >= 10 and set(y.loc[part].astype(str)) <= set(classes)
        enough &= all(g.loc[part & y.astype(str).eq(c)].nunique() >= (3 if part is tr else 1) for c in classes)
    if not enough:
        (dest/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8'); return status
    encoder = previous['encoder'] if previous else FrozenEncoder.fit(x.loc[tr])
    if previous and previous['classes'] != classes:
        raise ValueError('New interpreter classes require reviewed schema migration')
    ids = pd.Series([f'{t.isoformat()}::{label}' for t,label in zip(x.index,y.astype(str))],index=x.index)
    fit_rows = tr if previous is None else tr & ~ids.isin(previous.get('ingested',[])).to_numpy()
    if not np.any(fit_rows):
        return dict(status='no_new_labels')
    if previous and previous['model'].tree_count_+cfg['cb_update_trees']>cfg['cb_max_trees']:
        return dict(status='tree_capacity_reached',trees=previous['model'].tree_count_)
    model = CatBoostClassifier(iterations=cfg['cb_initial_trees'] if previous is None else cfg['cb_update_trees'],
                loss_function='MultiClass', depth=cfg['cb_depth'], random_seed=cfg['seed'],
                thread_count=cfg['threads'], has_time=True, allow_writing_files=False, verbose=False)
    # Keep old reviewed examples in the continuation batch.
    if previous:
        replay = previous['replay']
        fit_x = pd.concat([replay[0], x.loc[fit_rows]]).sort_index()
        fit_y = pd.concat([replay[1], y.loc[fit_rows].astype(str)]).reindex(fit_x.index)
    else:
        fit_x, fit_y = x.loc[tr], y.loc[tr].astype(str)
    model.fit(encoder.native(fit_x), fit_y, cat_features=list(encoder.categories), init_model=previous['model'] if previous else None)
    # Multinomial logistic calibration on a later period, test period remains untouched.
    raw = np.log(np.clip(model.predict_proba(encoder.native(x.loc[cal])), 1e-9, 1.))
    calibrator = LogisticRegression(C=1., max_iter=1000).fit(raw, y.loc[cal].astype(str))
    raw_test = np.log(np.clip(model.predict_proba(encoder.native(x.loc[te])), 1e-9, 1.))
    probabilities = calibrator.predict_proba(raw_test)
    status = dict(status='trained', train_groups=int(g.loc[tr].nunique()), calibration_groups=int(g.loc[cal].nunique()),
                  test_groups=int(g.loc[te].nunique()), test_logloss=float(log_loss(y.loc[te].astype(str), probabilities, labels=calibrator.classes_)),
                  test_average_precision=float(average_precision_score(y.loc[te].astype(str).eq('1.0'), probabilities[:,list(calibrator.classes_).index('1.0')])) if '1.0' in calibrator.classes_ else None,
                  evidence='chronological held-out episodes', abstain_below=.7)
    import joblib
    keep = np.linspace(0, len(fit_x)-1, min(cfg['replay_rows'], len(fit_x))).astype(int)
    package = dict(model=model, encoder=encoder, calibrator=calibrator, classes=classes,
                   trained_through=str(x.index[tr].max()), replay=(fit_x.iloc[keep],fit_y.iloc[keep]),
                   ingested=sorted(set(ids.loc[fit_rows]) | set(previous.get('ingested',[]) if previous else [])),
                   threshold=.7, report=status)
    joblib.dump(package, dest/'interpreter.joblib')
    (dest/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    return status


def predict_head(package, features):
    p = package['model'].predict_proba(package['encoder'].native(features))
    calibrated = package['calibrator'].predict_proba(np.log(np.clip(p, 1e-9, 1.)))
    i = calibrated.argmax(axis=1); confidence = calibrated.max(axis=1)
    labels = package['calibrator'].classes_[i]
    return pd.DataFrame(dict(label=np.where(confidence >= package['threshold'], labels, 'unknown'),
                             confidence=confidence), index=features.index)
