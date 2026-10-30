"""One chronological lifecycle; legacy fast/refit settings are intentionally absent."""
from copy import deepcopy
from pathlib import Path
import json
import pandas as pd
import math
from numbers import Real


DEFAULTS = dict(
    groups={'БКЗ-500': ['К-5', 'К-6', 'К-7']}, data_root='opc_wide', data_paths={},
    sensor_file='sensor_roles.xlsx', sensor_rules='sensor_rules.json', output='runs/incremental',
    first_test='2025-03-01', step='1h', run_thresholds={}, startup_hours=24,
    load_hold_hours=3, threshold_fraction=.25, admission_delay_days=14, post_repair_hours=48,
    min_train_rows=500, calibration_rows=168, min_sensor_rows=48,
    memory_rows=1500, replay_rows=500, update_rows=2000, calibration_limit=4000,
    models=['analogs', 'AAKR', 'MSET', 'CatBoost'],
    feature_sets=['conditions'], horizons=[1, 6], neighbors=30,
    cb_initial_trees=150, cb_update_trees=20, cb_max_trees=1500, cb_depth=4,
    learning_rate=.05, threads=2, seed=42,
    minimum_coverage=.7, smooth_hours=6, slow_hours=48, persist_hours=3,
    merge_hours=12, alarm_budgets=[.5, 1., 2.], alarm_budget=1.,
    max_background_fraction=.05, pre_window_days=7, neutral_days=7, min_lead_hours=3,
    validation_tolerance=.25, validate_rows=168, validate_candidate=True, schema_version=1,
    scenarios=['local'], related_inputs={}, risk_horizons=[24, 168],
    related_policy=dict(mode='auto', max_inputs=6, min_abs_correlation=.2, sample_rows=5000),
    history_policy=dict(block_hours=168, calibration_every=3, validation_every=9,
                        bootstrap=True, reference_days=60),
    regime_policy=dict(stop_hours=2, off_fraction=.9, current_fraction=.15,
                       current_thresholds={}, current_pairs={}, oil_channels=[],unknown_recovery_hours=None),
    risk_policy=dict(scope='group', samples='episodes', sample_hours=6,
                     min_positive_events=2, min_negative_groups=2,
                     min_calibration_groups=2, confidence=.7),
    critical_sensors=[],
    event_overrides=[],
)


def load(path=None, overrides=None):
    cfg = deepcopy(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text(encoding='utf-8')))
    if overrides:
        cfg.update(overrides)
    for key in ('history_policy','regime_policy','risk_policy','related_policy'):
        if not isinstance(cfg[key],dict):raise ValueError(key+' must be a dictionary')
        extra=set(cfg[key])-set(DEFAULTS[key])
        if extra:raise ValueError(f'Unknown {key} keys: {sorted(extra)}')
        cfg[key]={**deepcopy(DEFAULTS[key]),**cfg[key]}
    for key,default in DEFAULTS.items():
        if type(default) in (int,float) and (isinstance(cfg[key],bool) or not isinstance(cfg[key],Real) or not math.isfinite(cfg[key])):
            raise ValueError(key+' must be a finite number')
    for policy in ('history_policy','regime_policy','risk_policy','related_policy'):
        for key,default in DEFAULTS[policy].items():
            value=cfg[policy][key]
            if type(default) in (int,float) and (isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value)):
                raise ValueError(policy+'.'+key+' must be a finite number')
    for key in ('horizons','risk_horizons','alarm_budgets'):
        if not isinstance(cfg[key],list) or not cfg[key] or any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or v<=0 for v in cfg[key]):
            raise ValueError(key+' must contain positive finite numbers')
    if not isinstance(cfg['run_thresholds'],dict) or any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or v<0 for v in cfg['run_thresholds'].values() if v is not None):
        raise ValueError('run_thresholds must contain finite nonnegative numbers')
    first=pd.Timestamp(cfg['first_test'])
    if pd.isna(first) or first.tz is not None:raise ValueError('first_test must be a valid local timestamp without timezone')
    cfg['alarm_budgets']=[float(b) for b in cfg['alarm_budgets']]
    cfg['alarm_budget']=float(cfg['alarm_budget'])
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    if not isinstance(cfg['validate_candidate'],bool):
        raise ValueError('validate_candidate must be True or False')
    related=cfg['related_policy']
    if related['mode'] not in ('auto','manual') or not 0<=related['min_abs_correlation']<=1:
        raise ValueError('Invalid related_policy mode/correlation')
    for key in ('max_inputs','sample_rows'):
        value=related[key]
        if isinstance(value,bool) or int(value)!=value or value<=0:
            raise ValueError('related_policy.'+key+' must be a positive integer')
        related[key]=int(value)
    if not isinstance(cfg['related_inputs'],dict) or any(
        not isinstance(v,list) or any(not isinstance(c,str) for c in v) for v in cfg['related_inputs'].values()):
        raise ValueError('related_inputs must map target names to lists of sensor names')
    step=pd.Timedelta(cfg['step'])
    if not pd.Timedelta(0)<step<=pd.Timedelta('1h') or pd.Timedelta('1h')%step!=pd.Timedelta(0):
        raise ValueError('step must divide one hour and be at most one hour')
    if not cfg['groups'] or any(not units or len(set(units))!=len(units) for units in cfg['groups'].values()):
        raise ValueError('groups must contain nonempty lists of distinct units')
    for name in list(cfg['groups'])+[u for units in cfg['groups'].values() for u in units]:
        if not name or '..' in name or any(c in name for c in '<>:"/\\|?*'):
            raise ValueError('Group/unit names must be safe path components')
    if not cfg['models'] or not cfg['feature_sets'] or not cfg['scenarios']:
        raise ValueError('Select at least one model, feature set and scenario')
    if any(cfg[key]<0 for key in ('startup_hours','load_hold_hours','post_repair_hours','merge_hours')):
        raise ValueError('Regime durations cannot be negative')
    if not cfg['horizons'] or any(h<=0 or pd.Timedelta(hours=h)%step!=pd.Timedelta(0) for h in cfg['horizons']):
        raise ValueError('Forecast horizons must be positive multiples of step')
    if not 0<cfg['threshold_fraction']<1 or not 0<cfg['max_background_fraction']<=1:
        raise ValueError('Invalid threshold/background fractions')
    for key in ('min_train_rows', 'calibration_rows', 'min_sensor_rows', 'memory_rows',
                'calibration_limit', 'cb_initial_trees', 'cb_update_trees', 'cb_max_trees',
                'threads', 'persist_hours', 'validate_rows','replay_rows','update_rows','neighbors'):
        if cfg[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 < cfg['minimum_coverage'] <= 1:
        raise ValueError('minimum_coverage must be in (0, 1]')
    if cfg['cb_max_trees']<cfg['cb_initial_trees']:raise ValueError('cb_max_trees must cover the initial trees')
    for key in ('min_train_rows','calibration_rows','min_sensor_rows','memory_rows','replay_rows','update_rows',
                'calibration_limit','validate_rows','cb_initial_trees','cb_update_trees','cb_max_trees','cb_depth','threads','neighbors'):
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],(int,float)) or int(cfg[key])!=cfg[key] or cfg[key]<=0:
            raise ValueError(key+' must be a positive integer')
        cfg[key]=int(cfg[key])
    if not 0<cfg['learning_rate']<=1:raise ValueError('learning_rate must be in (0, 1]')
    if cfg['admission_delay_days'] < cfg['pre_window_days']:
        raise ValueError('Admission delay must cover the failure prediction horizon')
    if not set(cfg['models']) <= {'analogs', 'AAKR', 'MSET', 'CatBoost', 'HistGradientBoosting', 'CatBoostCorrection', 'CatBoostMulti'}:
        raise ValueError('Unsupported model')
    if not set(cfg['feature_sets']) <= {'conditions', 'related', 'forecast'}:
        raise ValueError('Unsupported feature set')
    if not set(cfg['scenarios']) <= {'local', 'pooled', 'transfer'}:
        raise ValueError('Unsupported scenario')
    if cfg['alarm_budget'] not in cfg['alarm_budgets'] or any(x <= 0 for x in cfg['alarm_budgets']):
        raise ValueError('alarm_budget must be one of the positive alarm_budgets')
    if any(not pd.notna(h) or h<=0 for h in cfg['risk_horizons']) or not cfg['risk_horizons']:
        raise ValueError('risk_horizons must contain positive hours')
    cfg['risk_horizons']=[int(h) if float(h).is_integer() else float(h) for h in cfg['risk_horizons']]
    hp,rp,ip=cfg['history_policy'],cfg['regime_policy'],cfg['risk_policy']
    if hp['block_hours']<=0 or hp['calibration_every']<2 or hp['validation_every']<3 or hp['reference_days']<=0:
        raise ValueError('Invalid history_policy')
    for key in ('calibration_every','validation_every'):
        if isinstance(hp[key],bool) or int(hp[key])!=hp[key]:raise ValueError('history_policy.'+key+' must be an integer')
        hp[key]=int(hp[key])
    if not isinstance(hp['bootstrap'],bool):raise ValueError('history_policy.bootstrap must be True or False')
    if rp['stop_hours']<=0 or not 0<rp['off_fraction']<=1 or not 0<rp['current_fraction']<1:
        raise ValueError('Invalid regime_policy')
    if rp['unknown_recovery_hours'] is not None and (isinstance(rp['unknown_recovery_hours'],bool) or not isinstance(rp['unknown_recovery_hours'],Real) or not math.isfinite(rp['unknown_recovery_hours'])):
        raise ValueError('unknown_recovery_hours must be a finite number or None')
    if rp['unknown_recovery_hours'] is not None and rp['unknown_recovery_hours']<0:raise ValueError('unknown_recovery_hours must be nonnegative or None')
    if not isinstance(rp['current_thresholds'],dict) or not isinstance(rp['current_pairs'],dict):raise ValueError('Current settings must be dictionaries')
    if cfg['calibration_limit']<max(cfg['calibration_rows'],cfg['min_sensor_rows']):raise ValueError('calibration_limit must cover calibration_rows and min_sensor_rows')
    if ip['scope'] not in ('group','unit') or ip['samples'] not in ('episodes','continuous'):
        raise ValueError('Invalid risk scope/samples')
    if pd.Timedelta(hours=ip['sample_hours'])%step!=pd.Timedelta(0) or pd.Timedelta(hours=hp['block_hours'])%step!=pd.Timedelta(0):
        raise ValueError('Sampling/block durations must be multiples of step')
    if any(ip[k]<=0 for k in ('sample_hours','min_positive_events','min_negative_groups','min_calibration_groups')) or not 0<ip['confidence']<1:
        raise ValueError('Invalid risk_policy')
    for key in ('min_positive_events','min_negative_groups','min_calibration_groups'):
        if isinstance(ip[key],bool) or int(ip[key])!=ip[key]:raise ValueError('risk_policy.'+key+' must be an integer')
        ip[key]=int(ip[key])
    if cfg['min_lead_hours']<0 or cfg['validation_tolerance']<0 or cfg['smooth_hours']<=0 or cfg['slow_hours']<=0:
        raise ValueError('Lead/tolerance must be nonnegative; smoothing windows must be positive')
    if any(h<=cfg['min_lead_hours'] for h in cfg['risk_horizons']):
        raise ValueError('Risk horizons must exceed min_lead_hours')
    for event in cfg['event_overrides']:
        if not {'unit','stop','type','label_available_at'}<=set(event) or event['type'] not in ('Авария','План','Резерв','Неизвестно'):
            raise ValueError('event_overrides require unit, stop, type and label_available_at')
    return cfg
