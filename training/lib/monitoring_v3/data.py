"""Source audit and event registry. Repair labels never enter predictor features."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .preparation import regimes, run_threshold, clean, conditions, RUN


def read_rules(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Sensor rules not found: {p.resolve()}')
    return json.loads(p.read_text(encoding='utf-8'))


def registry(frame, mode, cfg):
    state = frame['Состояние'].fillna('Нет данных').astype(str)
    group = state.ne(state.shift()).cumsum()
    rows = []
    step = pd.Timedelta(cfg['step'])
    kinds = {'Аварийный ремонт': 'Авария', 'Плановый ремонт': 'План', 'Резерв': 'Резерв'}
    for _, s in state.groupby(group):
        value = s.iloc[0]
        if value not in kinds:
            continue
        start, end = s.index[0], s.index[-1]+step
        # Last shutdown before the repair mark, or first shutdown within 72h after it.
        on = mode['mode'].isin(['работа', 'пуск'])
        candidates = mode.index[(~on) & on.shift(fill_value=False) &
                                (mode.index >= start-pd.Timedelta('72h')) & (mode.index <= start+pd.Timedelta('72h'))]
        before = candidates[candidates <= start]
        stop = before[-1] if len(before) else candidates[0] if len(candidates) else pd.NaT
        rows.append(dict(event_id=f'{start.isoformat()}::{value}', start=start, stop=stop, end=end,
                         type=kinds[value], reason='', source='Состояние',
                         label_available_at=start, verified=False, open_ended=end >= frame.index[-1]+step,
                         issue='missing_load_stop' if pd.isna(stop) else 'reason_unverified'))
    return pd.DataFrame(rows, columns=['event_id','start','stop','end','type','reason','source',
                                      'label_available_at','verified','open_ended','issue'])


def prepare_unit(frame, roles, unit, cfg, rules, threshold=None):
    loads = [c for c, r in roles.items() if r == 'load']
    if len(loads) != 1:
        raise ValueError(f'{unit}: exactly one load channel required')
    load = loads[0]
    if threshold is None:
        threshold = cfg['run_thresholds'].get(unit)
    source = 'manual' if threshold is not None else 'initial_past'
    if threshold is None:
        threshold = run_threshold(frame[load], frame.get('Состояние'), cfg['first_test'], cfg['threshold_fraction'])
    mode = regimes(frame[load], threshold, cfg)
    targets = [c for c, r in roles.items() if r == 'target']
    cleaned, flags, changes = clean(frame, targets, rules, cfg['step'])
    features = conditions(cleaned, roles, mode, unit, cfg['step'])
    events = registry(frame, mode, cfg)
    tests = frame.get('_Тип', pd.Series('', index=frame.index)).isin(['ИСП', 'ЗРР'])
    eligible = mode.eligible & ~tests
    return dict(frame=frame, cleaned=cleaned, roles=roles, unit=unit, targets=targets,
                threshold=float(threshold), threshold_source=source, load=load, regime=mode,
                features=features, quality=flags, changes=changes, events=events, eligible=eligible)


def healthy(context, as_of, cfg):
    """Select only mature rows and repair labels already available at as_of."""
    frame, events = context['frame'], context['events']
    ix, as_of = frame.index, pd.Timestamp(as_of)
    stop = as_of-pd.Timedelta(days=cfg['admission_delay_days'])
    ok = context['eligible'] & (ix < stop) & frame['Состояние'].eq('Работа')
    for e in events.itertuples():
        if e.label_available_at >= as_of:
            continue
        if e.type == 'Авария':
            anchor = e.stop if not pd.isna(e.stop) and e.stop < as_of else e.start
            ok &= ~((ix >= anchor-pd.Timedelta(days=cfg['admission_delay_days'])) & (ix < anchor))
        # Do not use the future end of an ongoing repair.
        if not e.open_ended and e.end < as_of:
            ok &= ~((ix >= e.end) & (ix < e.end+pd.Timedelta(hours=cfg['post_repair_hours'])))
    return ok


def load_sources(cfg, output=None, strict=True):
    import k2_lib as K
    import legacy_group as G
    rules = read_rules(cfg['sensor_rules'])
    units = list(dict.fromkeys(u for group in cfg['groups'].values() for u in group))
    roles = {u: G.read_roles(cfg['sensor_file'], u) for u in units}
    for members in cfg['groups'].values():
        aligned, _ = G.align_load_roles({u: roles[u] for u in members})
        roles.update(aligned)
    contexts, issues = {}, []
    for unit in units:
        path = Path(cfg['data_paths'].get(unit) or Path(cfg['data_root'])/unit/'hourly.parquet')
        if not path.is_file():
            issues.append(dict(unit=unit, problem='missing_source', path=str(path.resolve())))
            continue
        try:
            raw = K.read_table(str(path))
            if K.TIME_COL not in raw:
                raise ValueError(f'Missing {K.TIME_COL}')
            time = pd.to_datetime(raw[K.TIME_COL], errors='coerce')
            invalid = int(time.isna().sum())
            duplicates = int(time.dropna().duplicated().sum())
            numeric_original = set(r['original'] for r in roles[unit] if r['role'] in ('target','condition','load'))
            raw = raw.copy()
            for c in numeric_original & set(raw):
                raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(',', '.', regex=False), errors='coerce')
            raw[K.TIME_COL] = time
            raw = raw.loc[time.notna()].sort_values(K.TIME_COL, kind='stable').set_index(K.TIME_COL)
            if raw.empty:
                raise ValueError('No valid timestamps')
            numeric = list(raw.select_dtypes(include='number'))
            categorical = [c for c in raw if c not in numeric]
            grid = raw[numeric].resample(cfg['step']).mean()
            if categorical:
                grid = grid.join(raw[categorical].resample(cfg['step']).last())
            if K.STATE_COL not in grid:
                grid[K.STATE_COL] = K.ST_NODATA
            mapped, channel_roles, declared_units, aliases, audit = G.map_frame(grid, roles[unit], unit)
            # Align a role only after source mapping; never bypass physical unit checks.
            load_cols = [c for c, r in channel_roles.items() if r == 'load']
            if len(load_cols) != 1:
                raise ValueError('Load channel missing or ambiguous')
            c = prepare_unit(mapped, channel_roles, unit, cfg, rules)
            c.update(path=str(path.resolve()), audit=pd.DataFrame(audit), declared_units=declared_units,
                     aliases=aliases, invalid_timestamps=invalid, duplicate_timestamps=duplicates,
                     raw_rows=len(raw), original_roles=roles[unit])
            contexts[unit] = c
            if output:
                audit_unit(c, cfg, Path(output)/unit)
        except (ValueError, KeyError) as exc:
            issues.append(dict(unit=unit, problem=str(exc), path=str(path.resolve())))
    if output:
        dest = Path(output); dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(issues, columns=['unit','problem','path']).to_csv(dest/'source_issues.csv', index=False, encoding='utf-8-sig')
    if strict and issues:
        raise ValueError('Source audit failed: '+json.dumps(issues, ensure_ascii=False))
    return contexts, issues


def audit_unit(context, cfg, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True, exist_ok=True)
    c = context
    c['audit'].to_csv(dest/'sensor_audit.csv', index=False, encoding='utf-8-sig')
    c['events'].to_csv(dest/'events_to_review.csv', index=False, encoding='utf-8-sig')
    c['regime'].assign(eligible=c['eligible'], healthy=healthy(c, cfg['first_test'], cfg)).to_parquet(dest/'regimes.parquet')
    c['quality'].to_parquet(dest/'quality.parquet')
    sensor = c['frame'][c['targets']].agg(['count','nunique','min','max']).T
    sensor['missing_fraction'] = c['frame'][c['targets']].isna().mean()
    sensor.to_csv(dest/'sensor_statistics.csv', encoding='utf-8-sig')
    step_h = float(pd.Timedelta(cfg['step'])/pd.Timedelta('1h'))
    audit = dict(unit=c['unit'], source=c['path'], rows=len(c['frame']), raw_rows=c['raw_rows'],
                 invalid_timestamps=c['invalid_timestamps'], duplicate_timestamps=c['duplicate_timestamps'],
                 load=c['load'], threshold=c['threshold'], threshold_source=c['threshold_source'],
                 threshold_history_end=cfg['first_test'],
                 modes_hours={str(k):float(v*step_h) for k,v in c['regime']['mode'].value_counts().items()},
                 work_label_but_stopped=int((c['frame']['Состояние'].eq('Работа') & c['regime']['mode'].eq('стоит')).sum()),
                 load_unknown=int(c['regime']['mode'].eq('неизвестно').sum()),
                 label_status='repair reasons and correctness require human review')
    (dest/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(c['frame'].index, c['frame'][c['load']], linewidth=.6, label=c['load'])
    axes[0].axhline(c['threshold'], color='red', label='Run threshold')
    axes[0].legend()
    modes = {'стоит':0, 'неизвестно':1, 'пуск':2, 'работа':3}
    axes[1].step(c['frame'].index, c['regime']['mode'].map(modes), where='post')
    axes[1].set_yticks(list(modes.values()), list(modes))
    for e in c['events'].itertuples():
        axes[0].axvspan(e.start, e.end, color='red' if e.type == 'Авария' else 'gray', alpha=.15)
    fig.tight_layout(); fig.savefig(dest/'load_and_modes.png', dpi=130); plt.close(fig)
