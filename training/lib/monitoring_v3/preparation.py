"""Shared forward-only regime, quality and feature preparation, used at deployment too."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

STOP, START, RUN, UNKNOWN = 'стоит', 'пуск', 'работа', 'неизвестно'


def run_threshold(load, state, before, fraction=.25):
    ref = pd.to_numeric(load, errors='coerce').where(load.index < pd.Timestamp(before))
    if state is not None:
        ref = ref.where(state.eq('Работа'))
    ref = ref[np.isfinite(ref) & ref.gt(0)]
    if not len(ref):
        raise ValueError('Cannot infer load threshold from confirmed past operating hours; set run_thresholds')
    return float(ref.median() * fraction)


def regimes(load, threshold, cfg):
    step = pd.Timedelta(cfg['step'])
    x = pd.to_numeric(load, errors='coerce').replace([np.inf, -np.inf], np.nan)
    last = pd.Series(x.index.where(x.notna()), index=x.index).ffill()
    age = (x.index.to_series() - last).dt.total_seconds() / 3600
    held = x.ffill().where(age.le(cfg['load_hold_hours']))
    on = held.gt(threshold)
    unknown = held.isna()
    segment = (on.ne(on.shift()) | unknown | unknown.shift(fill_value=True)).cumsum()
    since = segment.groupby(segment).cumcount() * step / pd.Timedelta('1h')
    mode = pd.Series(STOP, index=x.index, dtype=object)
    mode.loc[on] = RUN
    mode.loc[on & since.lt(cfg['startup_hours'])] = START
    mode.loc[unknown] = UNKNOWN
    return pd.DataFrame(dict(mode=mode, load_missing=x.isna(), load_age_hours=age,
                             since_start_hours=since.where(on), eligible=mode.eq(RUN)), index=x.index)


def clean(frame, targets, rules, step='1h'):
    """Suspicion is retained; masking requires an explicit per-sensor policy."""
    out, flags, features = frame.copy(), {}, {}
    per_h = float(pd.Timedelta('1h') / pd.Timedelta(step))
    for c in targets:
        rule = {**rules.get('defaults', {}), **rules.get('sensors', {}).get(c, {})}
        kind = rule.get('kind', 'unreviewed')
        if kind not in ('unreviewed', 'continuous', 'discrete', 'rare'):
            raise ValueError(f'Unknown kind for {c}: {kind}')
        x = pd.to_numeric(frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        delta = x.diff()
        same = x.notna() & x.eq(x.shift())
        group = (~same).cumsum()
        age = x.groupby(group).cumcount() / per_h
        window = max(2, int(rule.get('window_hours', 168) * per_h))
        history = delta.shift().rolling(window, min_periods=max(2, min(window, int(24*per_h))))
        scale = (history.quantile(.75) - history.quantile(.25)) / 1.349
        spike = delta.sub(history.median()).abs().gt(rule.get('spike_factor', 8.) * scale) & scale.gt(1e-9)
        freeze = x.notna() & age.ge(rule.get('frozen_hours', 24)) & (kind == 'continuous')
        invalid = x.isna()
        if 'min' in rule:
            invalid |= x.lt(rule['min'])
        if 'max' in rule:
            invalid |= x.gt(rule['max'])
        mask = invalid | (freeze & bool(rule.get('mask_frozen', False)))
        mask |= spike & bool(rule.get('mask_spikes', False)) & (kind == 'continuous')
        out[c] = x.mask(mask)
        for key, val in dict(missing=x.isna(), spike=spike.fillna(False),
                             frozen=freeze.fillna(False), masked=mask).items():
            flags[c, key] = val
        features[c+'::held_hours'] = age.where(x.notna())
        features[c+'::changed'] = (x.notna() & x.shift().notna() & ~same).astype(float)
        features[c+'::delta'] = delta
    flag_frame = pd.DataFrame(flags, index=frame.index)
    flag_frame.columns = pd.MultiIndex.from_tuples(flag_frame.columns, names=['sensor', 'quality'])
    return out, flag_frame, pd.DataFrame(features, index=frame.index)


def conditions(frame, roles, regime, unit, step='1h'):
    cols = [c for c, r in roles.items() if r in ('load', 'condition', 'state')]
    columns = {c:frame[c] for c in cols}
    columns['unit'] = str(unit)
    columns['since_start_hours'] = regime.since_start_hours.clip(upper=720)
    columns['load_missing'] = regime.load_missing.astype(float)
    n = max(1, int(round(pd.Timedelta('1h') / pd.Timedelta(step))))
    for c, role in roles.items():
        if role in ('condition', 'load'):
            columns[c+'::delta_1h'] = pd.to_numeric(frame[c], errors='coerce').diff(n)
    columns['month_sin'] = np.sin(2*np.pi*frame.index.month/12)
    columns['month_cos'] = np.cos(2*np.pi*frame.index.month/12)
    for c in [c for c, r in roles.items() if r == 'state']:
        value = frame[c].astype(object)
        group = value.ne(value.shift()).cumsum()
        columns[c+'::held_hours'] = value.groupby(group).cumcount() * pd.Timedelta(step)/pd.Timedelta('1h')
        columns[c] = value.fillna('__MISSING__').astype(str)
    return pd.DataFrame(columns,index=frame.index)


@dataclass
class FrozenEncoder:
    """Fixed native categories for CatBoost; fixed one-hot basis for numerical baselines."""
    columns: list
    categories: dict
    medians: dict
    scales: dict

    @classmethod
    def fit(cls, frame):
        cat = {c: sorted(frame[c].fillna('__MISSING__').astype(str).unique())
               for c in frame if not pd.api.types.is_numeric_dtype(frame[c])}
        numeric = frame.drop(columns=list(cat)).astype(float)
        q = numeric.quantile([.25, .5, .75])
        scales = ((q.loc[.75]-q.loc[.25])/1.349).replace(0, np.nan).fillna(numeric.std()).fillna(1.).clip(lower=1e-9)
        return cls(list(frame), cat, q.loc[.5].fillna(0).to_dict(), scales.to_dict())

    def native(self, frame):
        columns={}
        for c in self.columns:
            values=frame[c] if c in frame else pd.Series(np.nan,index=frame.index)
            columns[c] = (values.fillna('__MISSING__').astype(str) if c in self.categories else
                          pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan))
        return pd.DataFrame(columns,index=frame.index)

    def numeric(self, frame):
        f = self.native(frame)
        parts = []
        for c in self.columns:
            if c in self.categories:
                parts.extend([(f[c] == v).to_numpy(float) for v in self.categories[c]])
                parts.append((~f[c].isin(self.categories[c])).to_numpy(float))
            else:
                parts.extend([((f[c]-self.medians[c])/self.scales[c]).fillna(0).to_numpy(float),
                              f[c].isna().to_numpy(float)])
        return np.column_stack(parts)
