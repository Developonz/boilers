"""Source audit and event registry. Repair labels never enter predictor features."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .preparation import regimes, run_threshold, clean, conditions, RUN
from .regime_evidence import fit_evidence,evidence_features,reference_end
from .event_registry import build_registry,event_type_at
from .input_grid import normalize_grid


def read_rules(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Sensor rules not found: {p.resolve()}')
    rules=json.loads(p.read_text(encoding='utf-8'))
    allowed={'kind','mask_frozen','mask_spikes','window_hours','frozen_hours','spike_factor','min','max','group','scale_floor','resolution'}
    if not isinstance(rules,dict) or not isinstance(rules.get('defaults',{}),dict) or not isinstance(rules.get('sensors',{}),dict):
        raise ValueError('Sensor rules require defaults/sensors dictionaries')
    for name,rule in [('defaults',rules.get('defaults',{}))]+list(rules.get('sensors',{}).items()):
        if not isinstance(rule,dict) or set(rule)-allowed:raise ValueError('Unknown/invalid sensor rules for '+str(name))
        if rule.get('kind','unreviewed') not in ('unreviewed','continuous','discrete','rare'):raise ValueError('Invalid sensor kind: '+str(name))
        if 'group' in rule and not isinstance(rule['group'],str):raise ValueError('Sensor group must be a string: '+str(name))
        for key in ('mask_frozen','mask_spikes'):
            if key in rule and not isinstance(rule[key],bool):raise ValueError(key+' must be True or False: '+str(name))
        for key in ('window_hours','frozen_hours','spike_factor','scale_floor','resolution','min','max'):
            if key not in rule:continue
            value=rule[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('Invalid '+key+': '+str(name))
            if key in ('window_hours','frozen_hours','spike_factor') and value<=0:raise ValueError(key+' must be positive: '+str(name))
            if key in ('scale_floor','resolution') and value<0:raise ValueError(key+' must be nonnegative: '+str(name))
        if 'min' in rule and 'max' in rule and rule['min']>rule['max']:raise ValueError('min exceeds max: '+str(name))
    for name,rule in rules.get('sensors',{}).items():
        effective={**rules.get('defaults',{}),**rule}
        if 'min' in effective and 'max' in effective and effective['min']>effective['max']:raise ValueError('Effective min exceeds max: '+str(name))
    return rules


def registry(frame, mode, cfg, unit=''):
    return build_registry(frame,mode,cfg,unit)


def prepare_unit(frame, roles, unit, cfg, rules, threshold=None, regime_spec=None,build_events=True):
    if any(str(c).strip().casefold() in ('состояние','_тип') and r!='exclude' for c,r in roles.items()):
        raise ValueError('Repair metadata cannot be model inputs or targets')
    loads = [c for c, r in roles.items() if r == 'load']
    if len(loads) != 1:
        raise ValueError(f'{unit}: exactly one load channel required')
    load = loads[0]
    if threshold is None:
        threshold = cfg['run_thresholds'].get(unit)
    source = 'manual' if threshold is not None else 'initial_past'
    if threshold is None:
        threshold = run_threshold(frame[load], frame.get('Состояние'), reference_end(frame,cfg,load), cfg['threshold_fraction'])
    regime_spec=regime_spec if regime_spec is not None else fit_evidence(frame,roles,unit,cfg)
    targets = [c for c, r in roles.items() if r == 'target']
    checked=targets+[c for c,r in roles.items() if r in ('load','condition')]
    cleaned, all_flags, changes = clean(frame, checked, rules, cfg['step'])
    flags=all_flags.loc[:,all_flags.columns.get_level_values('sensor').isin(targets)]
    evidence=evidence_features(cleaned,regime_spec)
    mode = regimes(cleaned[load], threshold, cfg,evidence)
    features = conditions(cleaned, roles, mode, unit, cfg['step'])
    from .event_registry import empty_registry
    events = registry(frame, mode, cfg, unit) if build_events else empty_registry()
    tests = frame.get('_Тип', pd.Series('', index=frame.index)).isin(['ИСП', 'ЗРР'])
    eligible = mode.eligible & ~tests
    input_columns=[c for c,r in roles.items() if r in ('load','condition','state')]
    input_coverage=cleaned[input_columns].notna().mean(axis=1)
    return dict(frame=frame, cleaned=cleaned, roles=roles, unit=unit, targets=targets,
                threshold=float(threshold), threshold_source=source, load=load, regime=mode,
                features=features, quality=flags,input_quality=all_flags,changes=changes,events=events,eligible=eligible,
                regime_spec=regime_spec,regime_policy=cfg['regime_policy'],input_coverage=input_coverage,preparation_reference_end=pd.Timestamp(regime_spec['reference_end']))


def healthy(context, as_of, cfg):
    """Select only mature rows and repair labels already available at as_of."""
    frame, events = context['frame'], context['events']
    ix, as_of = frame.index, pd.Timestamp(as_of)
    stop = as_of-pd.Timedelta(days=cfg['admission_delay_days'])
    required=[c for c,r in context['roles'].items() if r in ('load','condition','state')]
    enough_inputs=context['cleaned'][required].notna().mean(axis=1).ge(cfg['minimum_coverage'])
    ok = context['eligible'] & (ix < stop) & frame['Состояние'].eq('Работа') & enough_inputs
    for e in events.itertuples():
        observed=getattr(e,'stop_detected_at',e.label_available_at)
        effective,available=event_type_at(e,as_of)
        known=pd.notna(available) and available<as_of
        if not known and (pd.isna(observed) or observed>=as_of):continue
        if effective in ('Авария','Неизвестно') or not known:
            anchor = e.stop if not pd.isna(e.stop) and e.stop < as_of else e.start
            ok &= ~((ix >= anchor-pd.Timedelta(days=cfg['admission_delay_days'])) & (ix < anchor))
        # Do not use the future end of an ongoing repair.
        if known and not e.open_ended and e.end < as_of:
            ok &= ~((ix >= e.end) & (ix < e.end+pd.Timedelta(hours=cfg['post_repair_hours'])))
    return ok


def load_sources(cfg, output=None, strict=True):
    import k2_lib as K
    import legacy_group as G
    rules = read_rules(cfg['sensor_rules'])
    units = list(dict.fromkeys(u for group in cfg['groups'].values() for u in group))
    roles = {u: G.read_roles(cfg['sensor_file'], u) for u in units}
    if any(r['role']!='exclude' and str(r['original']).strip().casefold() in ('состояние','_тип') for rows in roles.values() for r in rows):
        raise ValueError('Repair metadata cannot be assigned a predictor/target role')
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
            numeric_original = set(r['original'] for r in roles[unit] if r['role'] in ('target','condition','load'))
            state_original=[r['original'] for r in roles[unit] if r['role']=='state']
            grid,source_audit=normalize_grid(raw,cfg['step'],numeric_original,state_original)
            if K.STATE_COL not in grid:
                grid[K.STATE_COL] = K.ST_NODATA
            mapped, channel_roles, declared_units, aliases, audit = G.map_frame(grid, roles[unit], unit)
            additions={}
            for r in roles[unit]:
                if r['role']!='exclude' and r['key'] not in channel_roles:
                    additions[r['key']]=pd.Series(np.nan,index=mapped.index)
                    channel_roles[r['key']]=r['role'];declared_units[r['key']]=r['units']
            if additions:mapped=pd.concat([mapped,pd.DataFrame(additions,index=mapped.index)],axis=1)
            # Align a role only after source mapping; never bypass physical unit checks.
            load_cols = [c for c, r in channel_roles.items() if r == 'load']
            if len(load_cols) != 1:
                raise ValueError('Load channel missing or ambiguous')
            c = prepare_unit(mapped, channel_roles, unit, cfg, rules)
            c.update(path=str(path.resolve()), audit=pd.DataFrame(audit), declared_units=declared_units,
                     aliases=aliases,**source_audit,original_roles=roles[unit])
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
    pd.DataFrame(c['regime_spec']['audit']).to_csv(dest/'motor_current_audit.csv',index=False,encoding='utf-8-sig')
    c['regime'].assign(eligible=c['eligible'], healthy=healthy(c, cfg['first_test'], cfg)).to_parquet(dest/'regimes.parquet')
    c['quality'].to_parquet(dest/'quality.parquet')
    sensor = c['frame'][c['targets']].agg(['count','nunique','min','max']).T
    sensor['missing_fraction'] = c['frame'][c['targets']].isna().mean()
    sensor.to_csv(dest/'sensor_statistics.csv', encoding='utf-8-sig')
    step_h = float(pd.Timedelta(cfg['step'])/pd.Timedelta('1h'))
    audit = dict(unit=c['unit'], source=c['path'], rows=len(c['frame']), raw_rows=c['raw_rows'],
                 invalid_timestamps=c['invalid_timestamps'], duplicate_timestamps=c['duplicate_timestamps'],
                 load=c['load'], threshold=c['threshold'], threshold_source=c['threshold_source'],
                 threshold_history_end=str(c['preparation_reference_end']),
                 load_current_conflicts=int(c['regime']['evidence_conflict'].sum()),
                 modes_hours={str(k):float(v*step_h) for k,v in c['regime']['mode'].value_counts().items()},
                 work_label_but_stopped=int((c['frame']['Состояние'].eq('Работа') & c['regime']['mode'].eq('стоит')).sum()),
                 load_unknown=int(c['regime']['mode'].eq('неизвестно').sum()),
                 label_status='repair reasons and correctness require human review')
    (dest/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    from incremental_plots import plot_preparation
    fig=plot_preparation(c)
    fig.savefig(dest/'load_and_modes.png', dpi=130); plt.close(fig)
