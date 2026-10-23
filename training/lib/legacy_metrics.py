"""Понятные метрики legacy; эксперимент близости хранится отдельно."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import k2_lib as K


def alarm_series(res, ns, multiplier=1.0):
    ix, step = ns['t_ix'], ns['STEP']
    smooth = res['score'] if res.get('smoothed') else K.rolling_mean(res['score'], ns['hs'](ns['SMOOTH_H']))
    ratio = (smooth.reindex(ix) / (res['thr'].reindex(ix) * multiplier)).where(ns['running'], 0.)
    eligible = np.asarray(ns['scorable'], bool) & (ix >= pd.Timestamp(ns['FIRST_TEST']))
    valid = eligible & np.isfinite(ratio.to_numpy()) & res['score'].reindex(ix).notna().to_numpy()
    above = pd.Series((ratio.to_numpy() > 1) & valid, index=ix)
    candidates = K.episodes(above)
    candidates = candidates[candidates.value.eq(True) & candidates.hours.ge(ns['PERSIST_H'])]
    merged = []
    for row in candidates.itertuples():
        bridge = (ix >= merged[-1][1]) & (ix < row.start) if merged else np.zeros(len(ix), bool)
        if (merged and row.start - merged[-1][1] < pd.Timedelta(hours=ns['MERGE_GAP_H'])
                and valid[bridge].all()):
            merged[-1][1] = row.end
        else:
            merged.append([row.start, row.end])
    episodes = pd.DataFrame(merged, columns=['start', 'end'])
    active = np.zeros(len(ix), bool)
    for row in episodes.itertuples():
        active |= (ix >= row.start) & (ix < row.end)
    active &= valid
    return ratio, episodes, active, valid


def event_windows(ns, days):
    ix = ns['t_ix']
    first, last = pd.Timestamp(ns['FIRST_TEST']), ix[-1] + ns['STEP']
    if ns.get('EVAL_END') is not None:
        last = min(last, pd.Timestamp(ns['EVAL_END']) + ns['STEP'])
    first = max(first, ns.get('_EVENT_START', first))
    last = min(last, ns.get('_EVENT_END', last))
    result = []
    for e in ns['fails'].itertuples():
        if first <= e.stop < last:
            mask = ((ix >= e.stop - pd.Timedelta(days=days)) &
                    (ix < e.stop - pd.Timedelta(hours=ns.get('MIN_LEAD_H', 3))))
            result.append((e, mask))
    return result


def evaluate(res, ns, days=None, multiplier=1., measurement_mask=None):
    days = ns['PRE_WINDOW_D'] if days is None else days
    ix = ns['t_ix']
    ratio, episodes, active, valid = alarm_series(res, ns, multiplier)
    if measurement_mask is not None:
        valid &= np.asarray(measurement_mask, bool)
    active &= valid
    pre = np.zeros(len(ix), bool)
    related = np.zeros(len(episodes), bool)
    event_rows, leads = [], []
    for e, window in event_windows(ns, days):
        pre |= window
        hits = []
        for j, alarm in enumerate(episodes.itertuples()):
            observed = window & active & (ix >= alarm.start) & (ix < alarm.end)
            hit = observed.any()
            if ns.get('ALARM_MATCH', 'overlap') == 'start':
                hit = hit and e.stop - pd.Timedelta(days=days) <= alarm.start < e.stop - pd.Timedelta(hours=ns.get('MIN_LEAD_H', 3))
            if hit:
                related[j] = True
                hits.append(alarm.start)
        caught = bool(hits)
        lead = float((e.stop - min(hits)) / pd.Timedelta('1h')) if caught else np.nan
        if caught:
            leads.append(lead)
        possible = window & np.asarray(ns['scorable'], bool) & (ix >= pd.Timestamp(ns['FIRST_TEST']))
        event_rows.append({'останов': e.stop, 'окно, сут': days, 'предупреждение': caught,
                           'начало тревоги': min(hits) if hits else pd.NaT,
                           'продолжающаяся': bool(hits and min(hits) < e.stop - pd.Timedelta(days=days)),
                           'упреждение, ч': lead, 'рабочих часов в окне': possible.sum() / ns['PER_H'],
                           'рассчитано часов': (valid & window).sum() / ns['PER_H']})
    # Same legacy neutral band, based on stops rather than silently changing it by scenario.
    dist = K.days_to_next(ix, ns['fails'].stop).to_numpy()
    bg = (valid & np.asarray(ns['mode'].eq(K.M_RUN)) & np.asarray(ns['state'].eq(K.ST_WORK)) &
          (dist > days + ns['NEUTRAL_D']))
    positive = pre & valid
    used = positive | bg
    labels, scores = positive[used].astype(int), ratio.to_numpy()[used]
    two = len(labels) and np.unique(labels).size == 2
    ap = float(average_precision_score(labels, scores)) if two else np.nan
    baseline = float(labels.mean()) if len(labels) else np.nan
    background_episodes = 0
    neutral_episodes = 0
    for j, alarm in enumerate(episodes.itertuples()):
        if related[j]:
            continue
        episode_mask = (ix >= alarm.start) & (ix < alarm.end)
        if (episode_mask & active & bg).any():
            background_episodes += 1
        elif (episode_mask & active).any():
            neutral_episodes += 1
    n_events, caught = len(event_rows), sum(r['предупреждение'] for r in event_rows)
    n_pre = int(related.sum())
    precision = n_pre / (n_pre + background_episodes) if n_pre + background_episodes else np.nan
    recall = caught / n_events if n_events else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and precision + recall else (0. if np.isfinite(precision + recall) else np.nan)
    bg_hours = bg.sum() / ns['PER_H']
    metrics = {'PR AUC': ap, 'PR AUC случайный уровень': baseline,
               'поймано аварий': f'{caught} из {n_events}', 'найдено': caught, 'всего остановов': n_events,
               'precision': precision, 'recall': recall, 'F1': f1,
               'тревог в фоне в месяц': background_episodes / (bg_hours / 720) if bg_hours else np.nan,
               '% фона в тревоге': float(active[bg].mean()) if bg.any() else np.nan,
               'упреждение, ч': float(np.median(leads)) if leads else np.nan,
               'полезных эпизодов': n_pre, 'фоновых эпизодов': background_episodes,
               'нейтральных эпизодов': neutral_episodes, 'часов фона': bg_hours,
               'рассчитано рабочих часов': valid.sum() / ns['PER_H']}
    # F1 combines event recall and episode precision, as in legacy; this is explicit in the report.
    return metrics, pd.DataFrame(event_rows), (ratio, episodes, active, valid)


def soft_average_precision(relevance, score):
    """Graded analogue of AP. Tied scores enter together; not sklearn sample_weight."""
    r, s = np.asarray(relevance, float), np.asarray(score, float)
    ok = np.isfinite(r) & np.isfinite(s)
    r, s = r[ok], s[ok]
    if not len(r) or r.sum() <= 0:
        return np.nan
    order = np.argsort(-s, kind='stable')
    r, s = r[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    cumulative = np.cumsum(r)[ends]
    precision = cumulative / (ends + 1)
    recall = cumulative / r.sum()
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def proximity_table(res, ns):
    ratio, _, _, valid = alarm_series(res, ns)
    ix = ns['t_ix']
    # No next recorded stop: future is unknown, not a zero-relevance example.
    events = ns['events'].sort_values('stop').drop_duplicates('stop')
    starts = pd.DatetimeIndex(events.stop)
    positions = starts.searchsorted(ix, side='right')
    known = positions < len(starts)
    distance = np.full(len(ix), np.nan)
    next_failure = np.zeros(len(ix), bool)
    if len(starts):
        distance[known] = (starts[positions[known]] - ix[known]) / pd.Timedelta('1D')
        next_failure[known] = events.type.to_numpy()[positions[known]] == K.EV_FAIL
    allowed = valid & known & (distance >= ns.get('MIN_LEAD_H', 3) / 24)
    rows = []
    for half_life in ns.get('PROXIMITY_HALF_LIVES_D', [1, 3, 7, 14]):
        relevance = np.where(next_failure[allowed], np.exp2(-distance[allowed] / half_life), 0.)
        rows.append({'полураспад веса, сут': half_life,
                     'PR близости (эксперимент)': soft_average_precision(relevance, ratio.to_numpy()[allowed]),
                     'базовый вес': float(relevance.mean()) if len(relevance) else np.nan,
                     'часов проверки': allowed.sum() / ns['PER_H']})
    return pd.DataFrame(rows)


def threshold_table(res, ns):
    rows = []
    split = ns.get('THRESHOLD_VALIDATION_END')
    ix = ns['t_ix']
    periods = [('весь период: диагностический', None)] if not split else [
        ('прошлый проверочный', ix < pd.Timestamp(split)), ('последующий контрольный', ix >= pd.Timestamp(split))]
    for multiplier in ns.get('THRESHOLD_MULTIPLIERS', [.5, .75, 1., 1.25, 1.5, 2.]):
        for label, mask in periods:
            # Separate event denominators by period, retaining the original full alarm stream.
            local = dict(ns)
            if mask is not None:
                if not np.any(mask):
                    continue
                first, last = ix[mask].min(), ix[mask].max()
                local['_EVENT_START'] = first
                local['_EVENT_END'] = last + ns['STEP']
            result, _, _ = evaluate(res, local, multiplier=multiplier, measurement_mask=mask)
            rows.append({'множитель порога': multiplier, 'период': label, **result})
    return pd.DataFrame(rows)
