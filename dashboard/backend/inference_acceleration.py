"""Bounded reuse of deterministic, row-wise legacy inference.

Temporal preparation and alarm construction still run on each requested history.
Only identical *prepared* numeric vectors from the same loaded model are reused.
"""
from collections import OrderedDict
import hashlib
import inspect
import os
import sys
import threading
import time

import numpy as np
import pandas as pd


class RowCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.size_bytes = 0
        self.items = OrderedDict()
        self.lock = threading.Lock()
        self.next_token = 0

    def token(self):
        with self.lock:
            self.next_token += 1
            return self.next_token

    def get(self, key):
        with self.lock:
            item = self.items.get(key)
            if item is not None:
                self.items.move_to_end(key)
                return item[0]
        return None

    def put(self, key, values):
        # Count owned array storage, Python containers and LRU overhead.
        size = (sys.getsizeof(key) + sys.getsizeof(key[-1]) + 128 + sys.getsizeof(values)
                + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in values.items()))
        if size > self.max_bytes:
            return
        with self.lock:
            old = self.items.pop(key, None)
            if old is not None:
                self.size_bytes -= old[1]
            self.items[key] = (values, size)
            self.size_bytes += size
            while self.size_bytes > self.max_bytes:
                _, (_, removed) = self.items.popitem(last=False)
                self.size_bytes -= removed


_shared_cache = None
_shared_lock = threading.Lock()


def shared_cache():
    global _shared_cache
    with _shared_lock:
        if _shared_cache is None:
            mb = max(0, int(os.getenv('MODEL_ROW_CACHE_MB', '128')))
            _shared_cache = RowCache(mb * 1024 * 1024)
        return _shared_cache


class CachedPointwiseCore:
    def __init__(self, core, cache):
        self.wrapped = core
        self.cache = cache
        self.token = cache.token()
        self.rows_computed = self.rows_reused = 0
        self.compute_seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def errors(self, features):
        if features.ndim != 2 or not len(features) or not self.cache.max_bytes:
            return self.wrapped.errors(features)
        features = np.asarray(features)
        keys = [(self.token, features.dtype.str, features.shape[1], row.tobytes())
                for row in features]
        rows, missing = [], OrderedDict()
        for i, key in enumerate(keys):
            found = self.cache.get(key)
            rows.append(found)
            if found is None:
                missing.setdefault(key, i)
        if missing:
            positions = list(missing.values())
            # Preserve the export's C/F layout convention for the predictor.
            order = 'F' if features.flags.f_contiguous and not features.flags.c_contiguous else 'C'
            query = np.array(features[positions], order=order, copy=True)
            start = time.perf_counter()
            result = self.wrapped.errors(query)
            self.compute_seconds += time.perf_counter() - start
            self.rows_computed += len(positions)
            if any(not isinstance(v, np.ndarray) or v.ndim == 0 or len(v) != len(positions)
                   for v in result.values()):
                # Only row-shaped results are eligible for this optimization.
                return self.wrapped.errors(features)
            computed = {}
            for offset, key in enumerate(missing):
                values = {name: np.array(value[offset], copy=True) for name, value in result.items()}
                computed[key] = values
                self.cache.put(key, values)
            for i, key in enumerate(keys):
                if rows[i] is None:
                    rows[i] = computed[key]
        self.rows_reused += len(features) - len(missing)
        return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


_MAX_CLEAN_CELLS = 100_000


def fast_causal_clean_numeric(frame, running, columns, *, per_h=1, window_h=168,
                              min_history_h=24, spike_factor=8., frozen_h=24,
                              min_change_rate=.9):
    """Vectorized equivalent of the recognized legacy causal cleaner."""
    per_h = max(1, int(per_h))
    window = max(2, int(round(window_h * per_h)))
    minimum = max(2, int(round(min_history_h * per_h)))
    if minimum > window:
        raise ValueError('CLEAN_MIN_HISTORY_H must not exceed CLEAN_WINDOW_H.')
    columns = list(columns)
    result = frame.copy()
    if not columns:
        empty = pd.DataFrame(index=frame.index)
        return result, dict(spikes=empty, frozen=empty.copy())
    # Long histories must not allocate a full-history matrix for every
    # intermediate mask/rate. Work in bounded column blocks instead.
    if len(columns) > 1 and len(frame) * len(columns) > _MAX_CLEAN_CELLS:
        width = max(1, _MAX_CLEAN_CELLS // max(1, len(frame)))
        parts = dict(spikes=[], frozen=[])
        for start in range(0, len(columns), width):
            block = columns[start:start + width]
            clean, masks = fast_causal_clean_numeric(frame[block], running, block,
                per_h=per_h, window_h=window_h, min_history_h=min_history_h,
                spike_factor=spike_factor, frozen_h=frozen_h, min_change_rate=min_change_rate)
            result[block] = clean[block]
            for name in parts:
                parts[name].append(masks[name])
        return result, {name: pd.concat(values, axis=1) for name, values in parts.items()}
    running = pd.Series(np.asarray(running, bool), index=frame.index)
    value = frame[columns].apply(pd.to_numeric, errors='coerce').where(running, axis=0)
    delta = value.diff()
    history = delta.shift(1).rolling(window, min_periods=minimum)
    scale = (history.quantile(.75) - history.quantile(.25)) / 1.349
    center = history.median()
    spike = (((delta - center).abs() > spike_factor * scale) & scale.gt(1e-9)).fillna(False)
    previous_hour = value.shift(per_h)
    pairs = value.notna() & previous_hour.notna()
    changes = value.ne(previous_hour) & pairs
    counts = pairs.astype(int).cumsum().shift(1).fillna(0)
    rate = changes.astype(int).cumsum().shift(1).div(counts.where(counts > 0))
    eligible = (counts.ge(48 * per_h) & rate.ge(min_change_rate)).fillna(False)
    segment_start = value.isna() | value.shift().isna() | value.ne(value.shift())
    positions = np.arange(len(value))[:, None]
    starts = np.maximum.accumulate(np.where(segment_start.to_numpy(), positions, 0), axis=0)
    age = pd.DataFrame(positions - starts + 1, index=value.index, columns=value.columns)
    latched = eligible.where(segment_start).ffill().fillna(False).astype(bool)
    freeze = value.notna() & latched & age.ge(max(1, int(round(frozen_h * per_h))))
    result[columns] = result[columns].mask(spike | freeze)
    return result, dict(spikes=spike, frozen=freeze)


_CLEANER_HASH = '7161efb83ed88faee3b0fe96dbe053909fe701923c90cfa463e0e0c97679be0f'


def install_acceleration(model, module):
    """Support only known row-independent exports and the exact known cleaner."""
    if os.getenv('MODEL_ACCELERATION', '1').strip().lower() in {'0', 'false', 'off'}:
        return
    cleaner = getattr(module, 'causal_clean_numeric', None)
    if cleaner is not None and cleaner is not fast_causal_clean_numeric:
        try:
            source = inspect.getsource(cleaner).replace('\r\n', '\n').strip().encode()
            if hashlib.sha256(source).hexdigest() == _CLEANER_HASH:
                module.causal_clean_numeric = fast_causal_clean_numeric
        except (OSError, TypeError):
            pass
    seen = set()

    def visit(member):
        if id(member) in seen:
            return
        seen.add(id(member))
        cfg = getattr(member, 'cfg', {})
        core = getattr(member, 'core', None)
        if cfg.get('core') in {'kernel', 'analogs'} and core is not None:
            if not isinstance(core, CachedPointwiseCore):
                member.core = CachedPointwiseCore(core, shared_cache())
        stage = getattr(member, 'stage1', None)
        if stage is not None:
            visit(stage)
        for child in getattr(member, 'members', []):
            visit(child[0] if isinstance(child, tuple) else child)
    visit(model)


def cache_counters(model):
    seen, result = set(), dict(rows_computed=0, rows_reused=0, compute_seconds=0.0)

    def visit(member):
        if id(member) in seen:
            return
        seen.add(id(member))
        core = getattr(member, 'core', None)
        if isinstance(core, CachedPointwiseCore):
            for key in result:
                result[key] += getattr(core, key)
        stage = getattr(member, 'stage1', None)
        if stage is not None:
            visit(stage)
        for child in getattr(member, 'members', []):
            visit(child[0] if isinstance(child, tuple) else child)
    visit(getattr(model, 'core', model) if hasattr(model, 'display') else model)
    return result
