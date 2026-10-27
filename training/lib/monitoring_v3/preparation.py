"""Shared forward-only regime, quality and feature preparation, used at deployment too."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

STOP, START, RUN, UNKNOWN = 'стоит', 'пуск', 'работа', 'неизвестно'


def categorical_values(values):
    """Numeric state 1 must not turn into a new category '1.0' after a gap."""
    def text(value):
        if pd.isna(value):return '__MISSING__'
        if isinstance(value,(float,np.floating)) and np.isfinite(value) and float(value).is_integer():return str(int(value))
        return str(value)
    return values.astype(object).map(text)


def run_threshold(load, state, before, fraction=.25):
    ref = pd.to_numeric(load, errors='coerce').where(load.index < pd.Timestamp(before))
    if state is not None:
        ref = ref.where(state.eq('Работа'))
    ref = ref[np.isfinite(ref) & ref.gt(0)]
    if not len(ref):
        raise ValueError('Cannot infer load threshold from confirmed past operating hours; set run_thresholds')
    return float(ref.median() * fraction)


def regimes(load, threshold, cfg, evidence=None):
    step = pd.Timedelta(cfg['step'])
    x = pd.to_numeric(load, errors='coerce').replace([np.inf, -np.inf], np.nan)
    last = pd.Series(x.index.where(x.notna()), index=x.index).ffill()
    age = (x.index.to_series() - last).dt.total_seconds() / 3600
    held = x.ffill().where(age.le(cfg['load_hold_hours']))
    policy=cfg.get('regime_policy',{})
    off=threshold*policy.get('off_fraction',.9)
    required=max(1,int(np.ceil(pd.Timedelta(hours=policy.get('stop_hours',2))/step)))
    modes=[];ages=[];stops=[];stop_times=[];conflicts=[];known_ages=[];observed_ages=[];recoveries=[]
    running=False;started=None;low_start=None;low_count=0;ever_on=False
    age_known=True;recovering=False;recovery_start=None
    recovery_hours=policy.get('unknown_recovery_hours')
    if recovery_hours is None:recovery_hours=cfg['startup_hours']
    coal=evidence.coal_evidence.reindex(x.index) if evidence is not None else pd.Series(np.nan,index=x.index)
    for t,v,c,observed_value in zip(x.index,held.to_numpy(),coal.to_numpy(),x.to_numpy()):
        conflict=bool(np.isfinite(v) and v<=off and c==1)
        confirmed=False;stop_at=pd.NaT
        if not np.isfinite(v) or conflict:
            mode=UNKNOWN;age_run=np.nan;low_count=0;low_start=None
            if running:age_known=False;recovering=True;recovery_start=None
            # Unknown data are not evidence of either shutdown or a fresh start.
        elif v>threshold or (running and v>off):
            if not running:started=t;age_known=True;recovering=False;recovery_start=None
            running=True;ever_on=True;low_count=0;low_start=None
            age_run=float((t-started)/pd.Timedelta('1h')) if age_known else np.nan
            if recovering:
                if recovery_start is None:recovery_start=t
                recovering=(t-recovery_start)<pd.Timedelta(hours=recovery_hours)
                mode=UNKNOWN if recovering else RUN
            else:mode=START if age_known and age_run<cfg['startup_hours'] else RUN
        else:
            if not np.isfinite(observed_value):
                # Holding the last low sample is not a second measurement that
                # confirms a physical stop. A missing sample breaks confirmation.
                modes.append(UNKNOWN if running else STOP);ages.append(np.nan)
                stops.append(False);stop_times.append(pd.NaT);conflicts.append(conflict)
                if running:age_known=False;recovering=True;recovery_start=None
                known_ages.append(age_known);observed_ages.append(np.nan);recoveries.append(recovering)
                low_count=0;low_start=None
                continue
            low_start=t if low_count==0 else low_start;low_count+=1;age_run=np.nan
            if low_count>=required:
                confirmed=bool(running and ever_on);stop_at=low_start if confirmed else pd.NaT
                running=False;started=None;mode=STOP;age_known=True;recovering=False;recovery_start=None
            else:mode=UNKNOWN if running else STOP
        modes.append(mode);ages.append(age_run);stops.append(confirmed);stop_times.append(stop_at);conflicts.append(conflict)
        known_ages.append(age_known)
        observed_ages.append(float((t-recovery_start)/pd.Timedelta('1h')) if recovery_start is not None and mode!=STOP else age_run)
        recoveries.append(recovering)
    result=pd.DataFrame(dict(mode=modes,load_missing=x.isna(),load_age_hours=age,
        since_start_hours=ages,stop_confirmed=stops,physical_stop_at=pd.Series(stop_times,index=x.index,dtype='datetime64[ns]'),
        evidence_conflict=conflicts,run_age_known=known_ages,observed_run_hours=observed_ages,
        recovering_after_unknown=recoveries),index=x.index)
    result['eligible']=result['mode'].eq(RUN)
    if evidence is not None:result=pd.concat([result,evidence.add_prefix('evidence::')],axis=1)
    last=pd.Series(pd.NaT,index=x.index,dtype='datetime64[ns]')
    last.loc[result.stop_confirmed]=result.loc[result.stop_confirmed,'physical_stop_at']
    result['since_stop_hours']=(x.index.to_series()-last.ffill()).dt.total_seconds()/3600
    result['regime_status']=np.where(result.evidence_conflict,'load_current_conflict',
        np.where(result.recovering_after_unknown,'recovering_after_unknown',
        np.where(result['mode'].eq(UNKNOWN)&coal.eq(1),'probable_coal_operation',
        np.where(result['mode'].eq(UNKNOWN),'insufficient_evidence',np.where(~result.run_age_known,'run_age_unknown','load_confirmed')))))
    return result


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
        columns[c] = categorical_values(value)
    for c in regime:
        if c.startswith('evidence::') or c in ('since_stop_hours','evidence_conflict','regime_status','run_age_known','observed_run_hours','recovering_after_unknown'):
            columns['regime::'+c]=regime[c]
    result=pd.DataFrame(columns,index=frame.index)
    result.attrs['load_column']=next((c for c,r in roles.items() if r=='load'),None)
    result.attrs['condition_columns']=[c for c,r in roles.items() if r=='condition']
    return result


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
