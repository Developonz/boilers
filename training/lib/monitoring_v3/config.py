"""One chronological lifecycle; legacy fast/refit settings are intentionally absent."""
from copy import deepcopy
from pathlib import Path
import json
import pandas as pd


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
    validation_tolerance=.25, validate_rows=168, schema_version=1,
    scenarios=['local'], related_inputs={}, risk_horizons=[24, 168],
)


def load(path=None, overrides=None):
    cfg = deepcopy(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text(encoding='utf-8')))
    if overrides:
        cfg.update(overrides)
    cfg['alarm_budgets']=[float(b) for b in cfg['alarm_budgets']]
    cfg['alarm_budget']=float(cfg['alarm_budget'])
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
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
                'threads', 'persist_hours', 'validate_rows'):
        if cfg[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 < cfg['minimum_coverage'] <= 1:
        raise ValueError('minimum_coverage must be in (0, 1]')
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
    return cfg
