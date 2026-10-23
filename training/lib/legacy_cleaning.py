"""Forward-only sensor filters. This source is embedded into exported model.py.

Only current/past sensor values and the supplied operating mask are used.
The operating mask itself remains the historical legacy regime definition.
"""
import numpy as np
import pandas as pd


def causal_clean_numeric(frame, running, columns, *, per_h=1, window_h=168,
                         min_history_h=24, spike_factor=8., frozen_h=24,
                         min_change_rate=.9):
    """Past-IQR jump filter and frozen-value detection without retroactive masking.

    A jump is evaluated against increments strictly BEFORE the current timestamp.
    A frozen segment is masked only after frozen_h has elapsed. Its eligibility
    is latched from the historical hourly change rate at the segment's beginning.
    No forward fill, backward shift or centered/full-dataset statistics are used.
    """
    per_h = max(1, int(per_h))
    window = max(2, int(round(window_h * per_h)))
    minimum = max(2, int(round(min_history_h * per_h)))
    if minimum > window:
        raise ValueError('CLEAN_MIN_HISTORY_H must not exceed CLEAN_WINDOW_H.')
    result = frame.copy()
    running = pd.Series(np.asarray(running, bool), index=frame.index)
    spikes, frozen = {}, {}
    for col in columns:
        value = pd.to_numeric(frame[col], errors='coerce').where(running)
        delta = value.diff()
        history = delta.shift(1).rolling(window, min_periods=minimum)
        scale = (history.quantile(.75) - history.quantile(.25)) / 1.349
        center = history.median()
        spike = ((delta - center).abs() > spike_factor * scale) & scale.gt(1e-9)
        # Hourly changes for hourly and sub-hourly grids; only preceding pairs
        # contribute to the eligibility of a newly starting constant segment.
        previous_hour = value.shift(per_h)
        pairs = value.notna() & previous_hour.notna()
        changes = value.ne(previous_hour) & pairs
        counts = pairs.astype(int).cumsum().shift(1).fillna(0)
        rate = changes.astype(int).cumsum().shift(1).div(counts.where(counts > 0))
        eligible = (counts.ge(48 * per_h) & rate.ge(min_change_rate)).fillna(False)
        segment_start = value.isna() | value.shift().isna() | value.ne(value.shift())
        segment = segment_start.cumsum()
        age = value.groupby(segment, sort=False).cumcount() + 1
        latched = eligible.where(segment_start).ffill().fillna(False).astype(bool)
        freeze = value.notna() & latched & age.ge(max(1, int(round(frozen_h * per_h))))
        spikes[col], frozen[col] = spike.fillna(False), freeze
        result[col] = result[col].mask(spikes[col] | frozen[col])
    return result, dict(spikes=pd.DataFrame(spikes, index=frame.index),
                        frozen=pd.DataFrame(frozen, index=frame.index))
