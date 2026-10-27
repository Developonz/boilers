# -*- coding: utf-8 -*-
"""
Модель К-2 для дашборда.

Цепочка расчёта на выбранный момент «at»:
  1. часовая таблица (hourly.parquet от opc_to_wide.py) — история модели до at включительно;
  2. единая с ноутбуком очистка: подозрения сохраняются; маскирование только по правилам канала;
  3. модель из папки k2_deploy (model.py этой папки) -> r = model.score(история);
  4. на момент at: режим котла, отношение к порогу выбранного вида, тревога, индекс здоровья, а по каждому
     датчику — ожидание и коридор нормы для текущего режима, отклонение («ошибка / обычная ошибка») и уровень.

Здесь нет ничего про веб: main.py превращает результат в ответы API.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd

from inference_acceleration import cache_counters, install_acceleration

TIME_COL = 'TimeValue'
FEATURE_RX = re.compile(r'__(min|max|std|n)$')      # признаки внутри часа (opc_to_wide.py) — дашборду не нужны
LABEL_COLS = ('Состояние',)                        # метка по заявкам на ремонт: в эксплуатации её нет, модель её не видит
COND_PREFIX = 'условие: '                          # производные условия модели (считаются из данных внутри model.py)
M_STOP, M_START, M_RUN, M_PRESTOP = 'стоит', 'пуск', 'работа', 'перед остановом'   # как в model.py

FROZEN_MIN_H = 24             # залипание: одно значение ≥ 24 ч подряд при работе котла ...
FROZEN_MIN_CHANGE_RATE = 0.9  # ... у датчика, который обычно меняется каждый час
SPIKE_K = 8                   # скачок: изменение туда и обратно больше 8 обычных часовых изменений
CLEAN_LOOKBACK_D = 3          # сколько суток до истории модели брать, чтобы увидеть начало залипания


class DataUnavailable(RuntimeError):
    """Нет файла данных, модели или нужных столбцов — сообщение показывается пользователю как есть."""


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def health_from_ratio(ratio: Optional[float], good: float, zero: float) -> Optional[int]:
    """Индекс здоровья 0…100 из отношения «сглаженная ошибка / порог модели»:
    ≤ good -> 100; порог (1) -> 50; ≥ zero -> 0; между точками — линейно. Ниже 50 — модель видит аномалию."""
    r = _finite(ratio)
    if r is None:
        return None
    if r <= good:
        value = 100.0
    elif r <= 1.0:
        value = 100.0 - 50.0 * (r - good) / (1.0 - good)
    elif r >= zero:
        value = 0.0
    else:
        value = 50.0 - 50.0 * (r - 1.0) / (zero - 1.0)
    return int(round(value))


# ============================================================================ данные

def resolve_hourly_path(base_dir: Path, configured: str) -> Path:
    """HOURLY_PATH (абсолютный или от папки дашборда), иначе data/hourly.parquet или первый hourly.parquet в data/."""
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else base_dir / path
    default = base_dir / 'data' / 'hourly.parquet'
    if default.exists():
        return default
    found = sorted((base_dir / 'data').glob('**/hourly.parquet'))
    return found[0] if found else default


def read_hourly(path: Path, numeric_columns=None, normalizer=None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Часовая таблица: время в индексе (сетка 1 ч), только сами датчики (без __min/__max/…, служебных и меток)."""
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    if TIME_COL not in names:
        raise DataUnavailable(f'В {path.name} нет столбца времени «{TIME_COL}»')
    cols = [c for c in names
            if c == TIME_COL or not (FEATURE_RX.search(c) or c.startswith('_') or c in LABEL_COLS)]
    df = pd.read_parquet(path, columns=cols)
    df.columns = [str(c).strip() for c in df.columns]        # пробелы в конце имён — как при обучении (k2_lib их обрезает)
    if normalizer is not None:
        df=normalizer(df)
        numeric=[c for c in df if pd.api.types.is_numeric_dtype(df[c])]
        return df,numeric,[c for c in df if c not in numeric]
    df = df.loc[:, ~df.columns.duplicated()]
    stamps = pd.to_datetime(df.pop(TIME_COL), errors='coerce')
    df.index = pd.DatetimeIndex(stamps).round('1h')
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep='last')].sort_index().asfreq('1h')
    df.index.name = TIME_COL
    # Roles/scaler, rather than the Parquet storage type, determine which
    # model inputs are numeric. Match legacy training: decimal commas are
    # accepted; OPC status text and infinities become missing observations.
    # Do not apply this conversion to categorical mechanism states.
    for c in set(numeric_columns or ()) & set(df.columns):
        values = df[c]
        if not pd.api.types.is_numeric_dtype(values):
            values = pd.to_numeric(values.astype(str).str.replace('\xa0', '', regex=False)
                                   .str.replace(' ', '', regex=False)
                                   .str.replace(',', '.', regex=False), errors='coerce')
        df[c] = values.astype('float64').replace([np.inf, -np.inf], np.nan)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text = [c for c in df.columns if c not in set(numeric)]
    for c in numeric:
        df[c] = df[c].astype('float64')
    for c in text:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df, numeric, text


# ============================================================================ модель

@dataclass
class LoadedModel:
    folder: Path
    model: Any                       # объект модели из model.py папки (score(df) -> Result)
    module: Any                      # сам model.py (detect_mode и константы — те же, что у модели)
    cfg: Dict[str, Any]
    view: Dict[str, Any]             # выбранный вид: name, top, slow_h, thr, качество на проверке
    roles: Dict[str, str]            # датчик -> роль из sensors.csv (проверяется / условие / …)
    q: Dict[str, float]              # датчик -> обычная ошибка для выбранного вида
    bools: List[Tuple[str, str]]     # текстовые датчики, которые модель видит: (датчик, обычное состояние)
    scaler_num: List[str]            # числовые входы модели (из scaler.json)

    @property
    def run_col(self) -> str:
        return self.cfg['run_col']

    @property
    def mills(self) -> List[str]:
        return list((self.cfg.get('cond_spec') or {}).get('mills') or {})

    @property
    def checked(self) -> List[str]:
        return [c for c, role in self.roles.items() if role == 'проверяется']

    @property
    def conditions(self) -> List[str]:
        return [c for c, role in self.roles.items() if role == 'условие' and not c.startswith(COND_PREFIX)]

    @property
    def inputs(self) -> set:
        """Числовые датчики, которые модель получает на вход (без производных условий), и датчик нагрузки."""
        cols = self.scaler_num or list(self.roles)
        return {c for c in cols if not c.startswith(COND_PREFIX)} | {self.run_col}

    @property
    def trained_until(self) -> Optional[pd.Timestamp]:
        value = self.cfg.get('trained_until')
        return pd.Timestamp(value) if value else None


def find_model_dir(root: Path) -> Path:
    """Папка модели: сама MODEL_DIR (если в ней config.json) или вложенная папка с config.json
    (при нескольких — analogs_plus, иначе первая по имени)."""
    if (root / 'config.json').exists():
        return root
    found = sorted(p.parent for p in root.glob('*/config.json'))
    if not found:
        raise DataUnavailable(
            f'Модель не найдена: положите папку модели из deploy/К-2_hourly/ (с config.json и model.py) в {root}')
    preferred = [p for p in found if p.name == 'analogs_plus']
    return (preferred or found)[0]


def load_model(root: Path, view_name: str, model_unit: str = '') -> LoadedModel:
    try:
        return _load_model(root, view_name, model_unit)
    except DataUnavailable:
        raise
    except Exception as exc:
        raise DataUnavailable(f'Не удалось загрузить модель из {root}: {exc}') from exc


def _load_model(root: Path, view_name: str, model_unit: str = '') -> LoadedModel:
    folder = find_model_dir(root)
    if not (folder / 'model.py').exists():
        raise DataUnavailable(f'В папке модели {folder} нет model.py (её создаёт k2_deploy.py)')
    spec = importlib.util.spec_from_file_location('k2_model_' + re.sub(r'\W', '_', folder.name), folder / 'model.py')
    module = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True        # не оставлять __pycache__ в папке модели
    try:
        spec.loader.exec_module(module)
        model = module.load(str(folder))
    finally:
        sys.dont_write_bytecode = old
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if str(cfg.get('version', '')).startswith('02-v4'):
        return load_v4_model(folder, model, cfg, view_name, model_unit)
    if cfg.get('core') != 'incremental':
        install_acceleration(model, module)
    views = cfg.get('views') or []
    if not views:
        raise DataUnavailable(f'В {folder / "config.json"} нет видов (views)')
    view = next((v for v in views if v.get('name') == view_name), None)
    if view is None:
        raise DataUnavailable(f'Нет вида «{view_name}». Доступны: {[v["name"] for v in views]}')
    roles: Dict[str, str] = {}
    q: Dict[str, float] = {}
    sens_file = folder / 'sensors.csv'
    if sens_file.exists():
        sens = pd.read_csv(sens_file, encoding='utf-8-sig')
        roles = dict(zip(sens['датчик'], sens.get('роль', pd.Series('проверяется', index=sens.index))))
        qcol = f'обычная ошибка: {view["name"]}'
        if qcol in sens:
            q = dict(zip(sens['датчик'], pd.to_numeric(sens[qcol], errors='coerce')))
    bools: List[Tuple[str, str]] = []
    scaler_num: List[str] = []
    scaler_file = folder / 'scaler.json'
    if scaler_file.exists():
        scaler = json.loads(scaler_file.read_text(encoding='utf-8'))
        bools = [(str(c), str(m)) for c, m in scaler.get('bools', [])]
        scaler_num = list(scaler.get('num', []))
    # Keep the exported core in canonical names; expose raw boiler names to the UI.
    if cfg.get('input_aliases'):
        display = {common: raw for raw, common in cfg['input_aliases'].items()}
        rename = lambda c: display.get(c, c)
        model = LegacyAdapter(model, module, display)
        cfg['run_col'] = rename(cfg['run_col'])
        cfg['blank_cols'] = [rename(c) for c in cfg.get('blank_cols', [])]
        if cfg.get('cond_spec'):
            cfg['cond_spec']['mills'] = {rename(c): v for c, v in cfg['cond_spec'].get('mills', {}).items()}
        roles = {rename(c): v for c, v in roles.items()}
        q = {rename(c): v for c, v in q.items()}
        scaler_num = [rename(c) for c in scaler_num]
        bools = [(rename(c), v) for c, v in bools]
    return LoadedModel(folder=folder, model=model, module=module, cfg=cfg, view=view, roles=roles, q=q,
                       bools=bools, scaler_num=scaler_num)


class LegacyAdapter:
    """Use the export unchanged; translate only result labels back to raw sensor names."""
    def __init__(self, core, module, display):
        self.core, self.module, self.display = core, module, display
        self._input_capture = None
        enabled = os.getenv('MODEL_ACCELERATION', '1').strip().lower() not in {'0', 'false', 'off'}
        if (enabled and core.cfg.get('causal_cleaning') and core.cfg.get('input') != 'residuals'
                and callable(getattr(core, '_inputs', None))):
            self._input_capture = ContextVar('legacy_prepared_inputs', default=None)
            original = core._inputs

            def observed_inputs(frame, rows, ok):
                result = original(frame, rows, ok)
                collector = self._input_capture.get()
                if collector is not None:
                    collector.update(index=frame.index, missing=result[1])
                return result

            core._inputs = observed_inputs

    def score(self, raw,**kwargs):
        captured = {}
        token = self._input_capture.set(captured) if self._input_capture else None
        try:
            result = self.core.score(raw,**kwargs)
        finally:
            if self._input_capture:
                self._input_capture.reset(token)
        # Report invalid sensor inputs as unscored, not as zero deviation.
        # _inputs is the export's own preparation, including its causal filters.
        if (self.core.cfg.get('causal_cleaning') and self.core.cfg.get('input') != 'residuals'
                and self.core.cfg.get('core') != 'incremental'):
            if captured:
                index, missing = captured['index'], captured['missing']
            else:
                canonical = self.module.prepare(raw.rename(columns=self.core.cfg.get('input_aliases', {})))
                ok = result.mode.isin([M_RUN, M_PRESTOP]).to_numpy()
                _, missing = self.core._inputs(canonical, ok, ok)
                index = canonical.index
            valid = pd.DataFrame(~missing, index=index, columns=self.core.scaler.num)
            result.sensor_errors = {v: frame.where(valid.reindex(index=frame.index, columns=frame.columns))
                                    for v, frame in result.sensor_errors.items()}
        rename = lambda c: self.display.get(c, c)
        result.sensor_errors = {v: frame.rename(columns=rename) for v, frame in result.sensor_errors.items()}
        result.contrib_share = {v: frame.rename(columns=rename) for v, frame in result.contrib_share.items()}
        if result.expected is not None:
            result.expected = result.expected.rename(columns=rename, level=0)
        result.missing = [rename(c) for c in result.missing]
        return result


def causal_mode(series, threshold, startup, prestop=0, min_stop=0, ffill=0):
    """V4 mode display uses only values available at each timestamp."""
    running = series.notna() & series.gt(threshold)
    groups = running.ne(running.shift()).cumsum()
    times = series.index.to_series()
    age = times - times.groupby(groups).transform('first')
    return pd.Series(np.where(~running, M_STOP,
        np.where(age < pd.Timedelta(hours=startup), M_START, M_RUN)), index=series.index)


class V4Adapter:
    """Translate v4 results to the dashboard API without independent cleanup."""
    def __init__(self, core, unit, display):
        self.core, self.unit, self.display = core, unit, display
        self.module = sys.modules[core.__class__.__module__]

    def score(self, raw):
        views, signed = self.core.score_raw(raw, self.unit)
        rename = {c: self.core.schemas[self.unit]['names'].get(self.module.canonical_name(c, self.unit), c)
                  for c in raw.columns}
        canonical = raw.rename(columns=rename)
        display = lambda key: self.display.get(key, key)
        actual = canonical.reindex(columns=self.core.targets).apply(pd.to_numeric, errors='coerce')
        expected = actual - signed * self.core.q
        norm = pd.DataFrame(1.0, index=actual.index, columns=actual.columns).where(expected.notna())
        fields = {}
        for key in self.core.targets:
            for label, source in [('факт', actual), ('ожидание', expected), ('разброс', norm)]:
                fields[(display(key), label)] = source[key]
        expected_table = pd.DataFrame(fields)
        hourly, errors, shares, episodes = {}, {}, {}, []
        for view, frame in views.items():
            stream = self.module.alarm_stream(frame.ratio, self.core.config)
            hourly[(view, 'отношение')] = frame.ratio
            hourly[(view, 'тревога')] = stream.active
            values = signed.copy()
            slow_h = self.core.config['views'][view]['slow_h']
            if slow_h:
                values = values.apply(lambda col: self.module.contiguous_transform(col, slow_h))
            values = values.abs().rename(columns=display)
            errors[view] = values
            shares[view] = values.div(values.sum(axis=1).replace(0, np.nan), axis=0) * 100
            for ep in self.module.alarm_episodes(stream).itertuples():
                episodes.append((view, ep.start, ep.end, ep.hours, ep.peak))
        load = canonical[self.core.load]
        threshold = self.core.run_thresholds.get(self.unit, self.core.default_run_threshold)
        mode = causal_mode(load, threshold, self.core.config['startup_h'])
        missing = [display(k) for k in self.core.roles if k not in canonical]
        return SimpleNamespace(hourly=pd.DataFrame(hourly), mode=mode, sensor_errors=errors,
            expected=expected_table, contrib_share=shares,
            alarms=pd.DataFrame(episodes, columns=['вид', 'начало', 'конец', 'часов', 'пик']), missing=missing)


def load_v4_model(folder, core, config, view_name, model_unit=''):
    unit = model_unit or os.environ.get('MODEL_UNIT', '').strip()
    if not unit and len(core.schemas) == 1:
        unit = next(iter(core.schemas))
    if unit not in core.schemas:
        raise DataUnavailable('Для модели v4 задайте MODEL_UNIT в backend/.env, например К-7. '
                              f'Схемы модели: {list(core.schemas)}')
    if core.config['data_kind'] != 'hourly':
        raise DataUnavailable('Этот дашборд читает часовую сетку. Для instant используйте API модели v4 напрямую.')
    display = core.schemas[unit].get('display', {})
    views = [dict(name=name, top=spec['top'], slow_h=spec['slow_h'],
                  thr=core.thresholds[name]['threshold']) for name, spec in core.config['views'].items()]
    view = next((v for v in views if v['name'] == view_name), None)
    if view is None:
        raise DataUnavailable(f'Нет вида «{view_name}». Доступны: {[v["name"] for v in views]}')
    cfg = dict(config, views=views, core=core.method, title=f'{core.method}, {unit}, v4',
        run_col=display.get(core.load, core.load),
        run_thr=core.run_thresholds.get(unit, core.default_run_threshold),
        startup_steps=core.config['startup_h'], smooth_steps=core.config['smooth_h'],
        persist_steps=core.config['persist_h'], prestop_steps=0, min_stop_steps=0,
        ffill_steps=0, version='02-v4', unit=unit)
    roles = {display.get(k, k): ('проверяется' if role == 'target' else 'условие') for k, role in core.roles.items()}
    q = {display.get(k, k): float(core.q[i]) for i, k in enumerate(core.targets)}
    bools = [(display.get(c, c), ' / '.join(categories)) for c, categories in core.encoder.categories.items()]
    numeric = [display.get(k, k) for k, role in core.roles.items() if role != 'state']
    return LoadedModel(folder=folder, model=V4Adapter(core, unit, display),
        module=SimpleNamespace(detect_mode=causal_mode), cfg=cfg, view=view, roles=roles,
        q=q, bools=bools, scaler_num=numeric)


# ============================================================================ результат

@dataclass
class SensorEval:
    """Оценка одного числового датчика на момент at."""
    column: str
    role: str                              # checked — проверяется моделью; condition — условие режима; none — не в модели
    evaluated: bool = False
    reason: str = ''                       # почему не оценивается
    level: str = 'N'                       # A / B / V / G; N — не оценивается
    ratio: Optional[float] = None          # ошибка / обычная ошибка, среднее за smooth_h (по нему уровень)
    ratio_now: Optional[float] = None      # то же за последний час
    fact: Optional[float] = None           # значение, которое видела модель (после очистки)
    expected: Optional[float] = None       # ожидание для текущего режима
    spread: Optional[float] = None         # разброс аналогов
    low: Optional[float] = None            # коридор нормы режима: ожидание ± обычная ошибка × разброс
    high: Optional[float] = None
    share: Optional[float] = None          # доля датчика в отклонении котла, %
    main_in_alarm: bool = False            # среди главных датчиков идущей тревоги


@dataclass
class Evaluation:
    at: pd.Timestamp
    lm: LoadedModel                        # модель, которая считала этот момент (итоговая или модель месяца)
    hist: pd.DataFrame                     # сырые данные истории модели (до at включительно)
    window: pd.DataFrame                   # сырые данные окна показа
    numeric: List[str]
    text: List[str]
    mode: pd.Series                        # режим котла по часам истории
    ratio: pd.Series                       # сглаженная ошибка / порог (1 — порог)
    alarm: pd.Series                       # тревога модели (выше порога persist ч подряд)
    health: pd.Series                      # индекс здоровья по часам (NaN — не оценивается)
    errors: Optional[pd.DataFrame]         # ошибка / обычная по часам и датчикам (невалидные значения — NaN)
    errors_smooth: Optional[pd.DataFrame]
    expected: Optional[pd.DataFrame]       # (датчик, ожидание/разброс/факт) по оценённым часам
    episodes: List[Dict[str, Any]]
    sensors: Dict[str, SensorEval]
    frozen_hours: Dict[str, int]           # залипшие на at датчики -> часов подряд одно значение
    missing: List[str]                     # нужных модели датчиков нет в данных
    mode_now: str
    ratio_now: Optional[float]
    alarm_now: bool
    health_now: Optional[int]
    not_scored_reason: str = ''            # почему час at не оценён моделью
    alarm_sensors: List[Tuple[str, float]] = field(default_factory=list)
    interpretation: Dict[str, Any] = field(default_factory=dict)


# ============================================================================ сервис

class K2Service:
    """Данные + модель + расчёт на момент; результаты кэшируются по моменту."""

    def __init__(self, base_dir: Path, hourly_path: str, model_dir: str, view: str, history_days: Optional[int],
                 window_hours: int, health_good: float, health_zero: float, levels: Tuple[float, float, float],
                 months_dir: str = '', cache_size: int = 8, model_unit: str = ''):
        self.base_dir = base_dir
        self.hourly_cfg = hourly_path
        model_root = Path(model_dir)
        self.model_root = model_root if model_root.is_absolute() else base_dir / model_root
        self.months_cfg = months_dir
        self.view_name = view
        self.model_unit = model_unit
        self.history_h = None if history_days is None else max(int(history_days) * 24, int(window_hours) + 96)
        self.window_h = max(1, int(window_hours))
        self.health_good, self.health_zero = float(health_good), float(health_zero)
        self.levels = tuple(float(x) for x in levels)
        # A full-history result includes large per-sensor tables. Retain only
        # the latest such result per boiler, rather than eight full copies.
        self.cache_size = min(cache_size, 1) if self.history_h is None else cache_size
        self._lock = threading.RLock()
        self._data: Optional[Tuple[pd.DataFrame, List[str], List[str]]] = None
        self._data_key: Optional[Tuple[str, float]] = None
        self._model: Optional[LoadedModel] = None
        self._months: Optional[Dict[pd.Period, Path]] = None
        self._months_root: Optional[Path] = None
        self._month_models: 'OrderedDict[Path, LoadedModel]' = OrderedDict()
        self._stats: Optional[Dict[str, Any]] = None
        self._cache: 'OrderedDict[pd.Timestamp, Evaluation]' = OrderedDict()
        self.last_performance: Dict[str, Any] = {}

    # ---------------------------------------------------------------- данные и модель
    @property
    def hourly_path(self) -> Path:
        return resolve_hourly_path(self.base_dir, self.hourly_cfg)

    def data(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        path = self.hourly_path
        if not path.exists():
            raise DataUnavailable(
                f'Не найден файл часовых данных {path}. Положите hourly.parquet (от opc_to_wide.py) в папку data '
                'или задайте HOURLY_PATH в backend/.env')
        key = (str(path), path.stat().st_mtime)
        with self._lock:
            if self._data is None or self._data_key != key:
                try:
                    lm = self._model
                    columns = (set(lm.scaler_num) | {lm.run_col} | set(lm.mills)) if lm else None
                    core=lm.model.core if lm and isinstance(lm.model,LegacyAdapter) else lm.model if lm else None
                    normalizer=getattr(core,'normalize_input',None) if lm and lm.cfg.get('core')=='incremental' else None
                    self._data = read_hourly(path, numeric_columns=columns,normalizer=normalizer)
                    if self._data[0].empty:
                        raise DataUnavailable(f'В {path} нет часовых данных.')
                except DataUnavailable:
                    raise
                except Exception as exc:
                    raise DataUnavailable(f'Не удалось прочитать {path}: {exc}') from exc
                self._data_key = key
                self._stats = None
                self._cache.clear()
            return self._data

    def model(self) -> LoadedModel:
        """Итоговая модель (обучена до trained_until) — для моментов после этой даты, как в эксплуатации."""
        with self._lock:
            if self._model is None:
                self._model = load_model(self.model_root, self.view_name, self.model_unit)
                # data_range() can read telemetry before the model is needed.
                # Re-read it once with the loaded model's numeric schema.
                self._data_key = None
                self._months = None
                self._month_models.clear()
                self._stats = None
                self._cache.clear()
            return self._model

    def months(self) -> Dict[pd.Period, Path]:
        """Модели месяцев (как онлайн-оценка в ноутбуке 02: модель месяца обучена только на данных до его начала):
        папки ГГГГ-ММ в MODEL_MONTHS_DIR или в «<папка модели>_history» рядом с итоговой моделью."""
        with self._lock:
            if self._months is None:
                if self.months_cfg:
                    root = Path(self.months_cfg)
                    root = root if root.is_absolute() else self.base_dir / root
                else:
                    lm = self.model()
                    root = (lm.folder.parent if lm.folder.name == 'final' else
                            lm.folder.parent / f'{lm.folder.name}_history')
                found: Dict[pd.Period, Path] = {}
                if root.is_dir():
                    for p in sorted(root.iterdir()):
                        if (p / 'config.json').exists() and (p / 'model.py').exists():
                            try:
                                found[pd.Period(p.name, freq='M')] = p
                            except ValueError:
                                continue
                self._months, self._months_root = found, root
            return self._months

    def model_for(self, at: pd.Timestamp) -> LoadedModel:
        """Модель, которой оценивается момент: итоговая — после её даты обучения; раньше — модель месяца момента.
        Итоговая модель «помнит» часы, на которых училась, и в прошлом показывала бы их как норму."""
        lm = self.model()
        until = lm.trained_until
        available=lm.cfg.get('available_at')
        if (available and at>=pd.Timestamp(available)) or (not available and until is not None and at>until):
            return lm
        path = self.months().get(at.to_period('M'))
        if path is None:
            raise DataUnavailable(f'Нет исторической модели для {at:%Y-%m}. '
                                  'Оценка итоговой моделью на её обучающей истории недопустима.')
        with self._lock:
            if path in self._month_models:
                self._month_models.move_to_end(path)
            else:
                self._month_models[path] = load_model(path, self.view_name, self.model_unit)
                while len(self._month_models) > 3:
                    self._month_models.popitem(last=False)
            return self._month_models[path]

    def data_range(self) -> Dict[str, Optional[str]]:
        df, _, _ = self.data()
        start, end = df.index[0], df.index[-1]
        min_at = min(start + pd.Timedelta(hours=96), end)
        return {'start': start.isoformat(), 'end': end.isoformat(), 'minAsOf': min_at.isoformat()}

    def clamp(self, at: Optional[pd.Timestamp]) -> pd.Timestamp:
        df, _, _ = self.data()
        end = df.index[-1]
        if at is None:
            return end
        low = min(df.index[0] + pd.Timedelta(hours=96), end)
        return max(low, min(pd.Timestamp(at).floor('h'), end))

    def _sensor_stats(self) -> Dict[str, Any]:
        """Свойства датчиков по всему файлу (как в ноутбуке): доля часов с изменением и обычное часовое изменение."""
        if self._stats is not None:
            return self._stats
        lm = self.model()
        df, numeric, _ = self.data()
        if lm.run_col not in df.columns:
            raise DataUnavailable(f'В данных нет датчика работы котла «{lm.run_col}»')
        numeric = [c for c in numeric if c in lm.inputs]
        mode = self._mode(df)
        run = mode.ne(M_STOP).to_numpy()
        A = np.where(run[:, None], df[numeric].to_numpy(float), np.nan)
        prev = np.vstack([np.full((1, A.shape[1]), np.nan), A[:-1]])
        pairs = ~np.isnan(A) & ~np.isnan(prev)
        n_pairs = pairs.sum(axis=0)
        changed = ((A != prev) & pairs).sum(axis=0)
        rate = np.where(n_pairs > 48, changed / np.maximum(n_pairs, 1), np.nan)
        d1 = pd.DataFrame(A - prev, columns=numeric)
        scale = 1.4826 * (d1 - d1.median()).abs().median()
        skip = {lm.run_col, *lm.mills, *lm.conditions}      # условия модель получает как есть (так же при обучении)
        cleaned = [c for c in lm.scaler_num if c in set(numeric) and not c.startswith(COND_PREFIX) and c not in skip]
        self._stats = {'rate': pd.Series(rate, index=numeric), 'scale': scale, 'cleaned': cleaned}
        return self._stats

    def _mode(self, df: pd.DataFrame) -> pd.Series:
        lm = self.model()
        cfg = lm.cfg
        if cfg.get('core') == 'incremental':
            core = lm.model.core if isinstance(lm.model, LegacyAdapter) else lm.model
            return core.operating_mode(df)
        return lm.module.detect_mode(df[cfg['run_col']].astype(float), cfg['run_thr'], cfg['startup_steps'],
                                     cfg['prestop_steps'], cfg['min_stop_steps'], cfg['ffill_steps'])

    # ---------------------------------------------------------------- расчёт
    def evaluate(self, at: Optional[pd.Timestamp] = None) -> Evaluation:
        start = time.perf_counter()
        with self._lock:
            lm = self.model()
            df, numeric, text = self.data()
            at = self.clamp(at)
            if at in self._cache:
                self._cache.move_to_end(at)
                self.last_performance = dict(evaluation_cached=True, total_seconds=time.perf_counter() - start)
                return self._cache[at]
            ev = self._evaluate(df, numeric, text, lm, at)
            self.last_performance['total_seconds'] = time.perf_counter() - start
            self._cache[at] = ev
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return ev

    def _clean(self, sl: pd.DataFrame, numeric: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Как в ноутбуке 02: залипания и одиночные скачки датчиков модели -> пропуски.
        Возвращает очищенную копию и залипшие на последний час датчики (все числовые) с длиной залипания."""
        st = self._sensor_stats()
        run = self._mode(sl).ne(M_STOP).to_numpy()
        A = np.where(run[:, None], sl[numeric].to_numpy(float), np.nan)
        rate = st['rate'].reindex(numeric).to_numpy(float)
        frozen = np.zeros(A.shape, bool)
        frozen_now: Dict[str, int] = {}
        for j, c in enumerate(numeric):
            if not (rate[j] >= FROZEN_MIN_CHANGE_RATE):
                continue
            a = A[:, j]
            new = np.ones(len(a), bool)
            new[1:] = ~(a[1:] == a[:-1])
            rid = np.cumsum(new) - 1
            length = np.bincount(rid)[rid]
            frozen[:, j] = ~np.isnan(a) & (length >= FROZEN_MIN_H)
            if len(a) and frozen[-1, j]:
                frozen_now[c] = int(length[-1])
        clean = sl.copy()
        cols = st['cleaned']
        if cols:
            pos = [numeric.index(c) for c in cols]
            B = A[:, pos]
            d1 = np.full_like(B, np.nan)
            d1[1:] = B[1:] - B[:-1]
            d2 = np.full_like(B, np.nan)
            d2[:-1] = B[1:] - B[:-1]
            sc = SPIKE_K * st['scale'].reindex(cols).to_numpy(float)
            with np.errstate(invalid='ignore'):
                spikes = (np.abs(d1) > sc) & (np.abs(d2) > sc) & (np.sign(d1) != np.sign(d2))
            mask = frozen[:, pos] | spikes
            values = sl[cols].to_numpy(float)
            values[mask] = np.nan
            clean[cols] = values
        return clean, frozen_now

    def _evaluate(self, df: pd.DataFrame, numeric: List[str], text: List[str], final: LoadedModel,
                  at: pd.Timestamp) -> Evaluation:
        # показываются только датчики, которые модель получает на вход; оценивает момент модель его месяца
        numeric = [c for c in numeric if c in final.inputs]
        usual = dict(final.bools)
        text = [c for c in text if c in usual]
        lm = self.model_for(at)
        cfg, view = lm.cfg, lm.view
        vname = view['name']
        h0 = df.index[0] if self.history_h is None else at - pd.Timedelta(hours=self.history_h - 1)
        sl = df.loc[h0 - pd.Timedelta(days=CLEAN_LOOKBACK_D):at]
        if cfg.get('version') == '02-v4' or cfg.get('causal_cleaning'):
            clean, frozen_now = sl.copy(), {}
        else:
            clean, frozen_now = self._clean(sl, numeric)
        hist = df.loc[h0:at]
        incremental=cfg.get('core')=='incremental'
        # Preparation needs the complete causal prefix; the runtime bounds costly
        # predictions to the requested display period plus rolling warmup.
        model_in = df.loc[:at] if incremental else clean.loc[h0:at]
        counters = cache_counters(lm.model)
        score_start = time.perf_counter()
        try:
            r = lm.model.score(model_in,period_start=h0) if incremental else lm.model.score(model_in)
        except Exception as exc:  # noqa: BLE001 — сообщение модели показываем как есть
            raise DataUnavailable(f'Не удалось посчитать момент {at:%d.%m.%Y %H:%M}: {exc}') from exc
        after = cache_counters(lm.model)
        self.last_performance = dict(evaluation_cached=False, model_folder=str(lm.folder),
            history_rows=len(model_in), score_seconds=time.perf_counter() - score_start,
            **{key: after[key] - counters[key] for key in after})

        hourly = r.hourly
        if vname not in hourly.columns.get_level_values(0):
            raise DataUnavailable(f'В расчёте нет вида «{vname}» (MODEL_VIEW)')
        mode = r.mode.reindex(hist.index)
        ratio = hourly[(vname, 'отношение')].astype(float).reindex(hist.index)
        alarm = hourly[(vname, 'тревога')].reindex(hist.index).fillna(False).astype(bool)
        scored_mode = mode.isin([M_RUN, M_PRESTOP])
        health = pd.Series([health_from_ratio(x, self.health_good, self.health_zero) if ok else None
                            for x, ok in zip(ratio.to_numpy(), scored_mode.to_numpy())], index=hist.index, dtype=float)

        mode_now = str(mode.iloc[-1])
        ratio_now = _finite(ratio.iloc[-1]) if mode_now in (M_RUN, M_PRESTOP) else None
        alarm_now = bool(alarm.iloc[-1]) and ratio_now is not None
        health_now = health_from_ratio(ratio_now, self.health_good, self.health_zero)

        # ---- ошибки датчиков: только валидные значения (есть факт и ожидание), сглаживание как у модели
        M = r.sensor_errors.get(vname)
        exp = r.expected
        shares = r.contrib_share.get(vname)
        checked = [c for c in lm.checked if M is not None and c in M.columns]
        errors = errors_smooth = None
        if M is not None and checked:
            if exp is not None:
                fact_e = exp.xs('факт', axis=1, level=1).reindex(index=M.index, columns=checked)
                exp_e = exp.xs('ожидание', axis=1, level=1).reindex(index=M.index, columns=checked)
                spr_e = exp.xs('разброс', axis=1, level=1).reindex(index=M.index, columns=checked)
                valid = fact_e.notna() & exp_e.notna() & spr_e.notna()
            else:
                valid = model_in.reindex(index=M.index, columns=checked).notna()
            errors = M[checked].astype(float).where(valid)
            if incremental:
                contiguous=errors.index.to_series().diff().eq(pd.Timedelta(cfg.get('step','1h')))
                smoothed={}
                for column in errors:
                    valid_column=errors[column].notna()
                    segments=(~valid_column|~valid_column.shift(fill_value=False)|~contiguous).cumsum()
                    smoothed[column]=errors[column].groupby(segments).transform(
                        lambda values:values.rolling(pd.Timedelta(hours=float(cfg.get('smooth_steps',6))),min_periods=1).mean()).where(valid_column)
                errors_smooth=pd.DataFrame(smoothed,index=errors.index)
            else:errors_smooth = errors.rolling(f'{int(cfg.get("smooth_steps", 6))}h', min_periods=1).mean()

        scored_now = errors is not None and at in errors.index
        if cfg.get('normal_history_status')=='unsafe_retracted':
            not_scored='обучающая норма пересмотрена; оценки недоступны до восстановления модели'
        elif mode_now == M_STOP:
            not_scored = 'котёл стоит'
        elif mode_now == M_START:
            not_scored = f'пуск котла (первые {cfg.get("startup_steps", "")} ч после включения)'
        elif not scored_now or ratio_now is None:
            not_scored = 'недостаточно данных, незнакомый режим или прогрев расчёта'
        else:
            not_scored = ''

        # ---- эпизоды тревог выбранного вида
        episodes: List[Dict[str, Any]] = []
        al = r.alarms
        if al is not None and len(al):
            for row in al[al['вид'] == vname].itertuples(index=False):
                start, end = pd.Timestamp(row[1]), pd.Timestamp(row[2])
                episodes.append({'start': start, 'end': end, 'hours': float(row[3]), 'peak': _finite(row[4]),
                                 'active': bool(start <= at < end)})

        alarm_sensors: List[Tuple[str, float]] = []
        if alarm_now and errors is not None:
            ep = next((e for e in episodes if e['active']), None)
            if ep is not None:
                part = errors.loc[(errors.index >= ep['start']) & (errors.index <= at)]
                if len(part):
                    top = part.quantile(0.9).dropna().nlargest(3)
                    alarm_sensors = [(c, float(v)) for c, v in top.items() if v > 0]

        # ---- оценка каждого числового датчика на at
        sensors: Dict[str, SensorEval] = {}
        checked_set = set(checked)
        cond_set = set(lm.conditions) | {lm.run_col}
        main_set = {c for c, _ in alarm_sensors}
        blank_cols = set(cfg.get('blank_cols') or [])              # модель месяца их не проверяет (мало данных при обучении)
        t1, t2, t3 = self.levels
        for c in numeric:
            if c in checked_set:
                se = SensorEval(c, 'checked')
            elif c in cond_set:
                sensors[c] = SensorEval(c, 'condition', reason='режимный параметр: задаёт режим работы котла')
                continue
            else:
                sensors[c] = SensorEval(c, 'none', reason='нет оценки')
                continue
            sensors[c] = se
            if exp is not None and at in exp.index and (c, 'ожидание') in exp.columns:
                se.fact = _finite(exp.at[at, (c, 'факт')])
                se.expected = _finite(exp.at[at, (c, 'ожидание')])
                se.spread = _finite(exp.at[at, (c, 'разброс')])
            if se.expected is not None and se.spread is not None:
                q = _finite(lm.q.get(c))
                if q is not None:
                    se.low, se.high = se.expected - q * se.spread, se.expected + q * se.spread
            if not_scored:
                se.reason = not_scored
                continue
            if c in blank_cols:
                se.reason = 'мало данных по датчику за год — норма пока не определена'
                continue
            if c in frozen_now and c in self._sensor_stats()['cleaned']:
                se.reason = f'значение не меняется {frozen_now[c]} ч подряд — возможен отказ датчика'
                continue
            now = _finite(errors.at[at, c]) if errors is not None else None
            if now is None:
                if pd.isna(hist.at[at, c]):
                    se.reason = 'нет данных за этот час'
                elif exp is not None and se.expected is None:
                    se.reason = 'нетипичный режим — нормы для него нет'
                else:
                    se.reason = 'значение исключено моделью: скачок, залипание или недоступный вход'
                continue
            se.evaluated = True
            se.ratio_now = now
            se.ratio = _finite(errors_smooth.at[at, c])
            x = se.ratio if se.ratio is not None else now
            se.level = 'A' if x >= t3 else 'B' if x >= t2 else 'V' if x >= t1 else 'G'
            if shares is not None and at in shares.index and c in shares.columns:
                se.share = _finite(shares.at[at, c])
            se.main_in_alarm = c in main_set

        window = hist.loc[at - pd.Timedelta(hours=self.window_h - 1):at, numeric + text]
        return Evaluation(at=at, lm=lm, hist=hist, window=window, numeric=numeric, text=text, mode=mode, ratio=ratio,
                          alarm=alarm, health=health, errors=errors, errors_smooth=errors_smooth, expected=exp,
                          episodes=episodes, sensors=sensors, frozen_hours=frozen_now, missing=list(r.missing),
                          mode_now=mode_now, ratio_now=ratio_now, alarm_now=alarm_now, health_now=health_now,
                          not_scored_reason=not_scored, alarm_sensors=alarm_sensors,
                          interpretation=(r.interpretation.loc[at].where(pd.notna(r.interpretation.loc[at]), None).to_dict()
                                          if hasattr(r,'interpretation') and at in r.interpretation.index else {}))

    # ---------------------------------------------------------------- описание модели для API
    def model_info(self, lm=None) -> Dict[str, Any]:
        lm = lm or self.model()
        cfg, view = lm.cfg, lm.view
        quality = view.get('качество на проверке') or {}
        return {
            'folder': lm.folder.name,
            'title': cfg.get('title'),
            'core': cfg.get('core'),
            'view': view.get('name'),
            'views': [v.get('name') for v in cfg.get('views', [])],
            'topSensors': int(view.get('top', 1)),
            'threshold': _finite(view.get('thr')),
            'trainedUntil': cfg.get('trained_until'),
            'availableAt':cfg.get('available_at'),
            'calibrationStart':cfg.get('calibration_start'),
            'calibrationEnd':cfg.get('calibration_end'),
            'normalHistoryStatus':cfg.get('normal_history_status','ready'),
            'operatingThresholdSource':cfg.get('operating_threshold_source'),
            'interpretationHeads':cfg.get('interpretation_heads',[]),
            'created': cfg.get('created'),
            'smoothHours': cfg.get('smooth_steps'),
            'persistHours': cfg.get('persist_steps'),
            'startupHours': cfg.get('startup_steps'),
            'runColumn': lm.run_col,
            'checkedSensors': len(lm.checked),
            'conditionSensors': lm.conditions,
            'mechanismStates': [{'column': c, 'usual': m} for c, m in lm.bools],
            'quality': {k: (v if isinstance(v, (str, int)) or v is None else _finite(v)) for k, v in quality.items()},
            'healthAnchors': {'good': self.health_good, 'zero': self.health_zero},
            'levelThresholds': {'V': self.levels[0], 'B': self.levels[1], 'A': self.levels[2]},
            'historyHours': self.history_h,
            'windowHours': self.window_h,
            'monthModels': [str(m) for m in sorted(self.months())],
            'monthModelsDir': str(self._months_root) if self._months_root else None,
        }
