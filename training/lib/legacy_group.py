"""Excel schema and pooled training adapter for the ORIGINAL notebook 02.

Detector classes, preprocessing transforms, OOF calibration and alarm/metric
functions are taken from the old notebook/k2_lib, not boiler_monitoring v4.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import types
import time
import unicodedata

import joblib
try:
    from joblib.externals import cloudpickle  # Older joblib bundles it.
except ImportError:
    import cloudpickle  # Recent distributions use the existing dependency.
import numpy as np
import pandas as pd
import k2_lib as K
from legacy_cleaning import causal_clean_numeric

VERSION = 'legacy-roles-groups-2.2'
IMBALANCE_RX = re.compile(r'\bнебаланс\s+пар\s*/\s*вода\b', re.IGNORECASE)


def canonical_name(name, unit=None):
    text = unicodedata.normalize('NFKC', str(name))
    text = re.sub('[\u200b\u200c\u200d\u2060\ufeff]', '', text)
    text = ' '.join(text.replace('\xa0', ' ').split()).lower().replace('ё', 'е')
    # Only the physical parameter phrase identifies this sensor; its formula
    # and averaging suffix may differ. Startup imbalance keeps its own name.
    if is_imbalance_name(text):
        return 'небаланс пар/вода'
    text = re.sub(r'^[кk][аa]?\s*[-–]?\s*\d+\s*', '', text)
    for short, full in [(r'[тt]|темп\.?|т-ра', 'температура'),
                        (r'f|расх\.?', 'расход'), (r'[рp]|давл\.?', 'давление')]:
        text = re.sub(rf'^(?:{short})(?=\s|$)', full, text)
    text = re.sub(r'n\s*[сc]\s*[тt]', 'nст', text)
    if 'небаланс' in text:
        text = re.sub(r'пар\s*/\s*вода', 'пар/вода', text)
    number = re.search(r'\d+', str(unit))
    if number:
        text = re.sub(rf'\b(дв|дс|дрг|мв|ммт|псу|вгд)\s*-?\s*{number[0]}(?=[а-яa-z\s-]|$)', r'\1-#', text)
    return ' '.join(text.split())


def is_imbalance_name(name):
    text = unicodedata.normalize('NFKC', str(name)).lower()
    text = re.sub('[\u200b\u200c\u200d\u2060\ufeff]', '', text)
    return bool(IMBALANCE_RX.search(text))


def read_roles(path, unit):
    import openpyxl
    number = re.search(r'\d+', unit)[0]
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        sheets = [s for s in wb if re.search(r'\d+', s.title) and re.search(r'\d+', s.title)[0] == number]
        if len(sheets) != 1:
            raise ValueError(f'{unit}: нужен один лист Excel, найдено {[s.title for s in sheets]}')
        rows = iter(sheets[0].values)
        header = {str(v).strip().lower(): i for i, v in enumerate(next(rows)) if v is not None}
        if not {'col', 'роль'} <= set(header):
            raise ValueError('В Excel нужны столбцы col и роль.')
        result = []
        aliases = {'проверять': 'target', 'условие': 'condition', 'состояние': 'state',
                   'нагрузка': 'load', 'исключить': 'exclude'}
        for values in rows:
            def get(key, default=None):
                return values[header[key]] if key in header and header[key] < len(values) else default
            original = get('col')
            if not original:
                continue
            role = str(get('роль') or 'проверять').strip().lower()
            role = aliases.get(role, role)
            if role not in {'target', 'condition', 'state', 'load', 'exclude'}:
                raise ValueError(f'{unit}: неизвестная роль {role}: {original}')
            declared_key = str(get('общий параметр') or original).strip()
            result.append(dict(original=str(original).strip(), role=role,
                key=canonical_name(declared_key, unit), declared_key=declared_key,
                units=str(get('единица') or '').strip()))
        return result
    finally:
        wb.close()


def align_load_roles(rows_by_unit, enabled=True):
    """Align explicitly declared steam-load channels, keeping physical headers in the audit."""
    loads = {unit: [r for r in rows if r['role'] == 'load'] for unit, rows in rows_by_unit.items()}
    if any(len(rows) != 1 for rows in loads.values()):
        raise ValueError('У каждого котла должна быть ровно одна роль «нагрузка»: ' +
                         '; '.join(f'{u}: {len(rows)}' for u, rows in loads.items()))
    keys = {rows[0]['key'] for rows in loads.values()}
    if len(keys) == 1:
        return rows_by_unit, False
    detail = '; '.join(f'{u}: {rows[0]["original"]}' for u, rows in loads.items())
    steam_flow = all('расход' in canonical_name(rows[0]['original'], unit) and
                     'пар' in canonical_name(rows[0]['original'], unit) for unit, rows in loads.items())
    units = {rows[0]['units'].strip().lower() for rows in loads.values() if rows[0]['units'].strip()}
    if not enabled or not steam_flow or len(units) > 1:
        raise ValueError('Не сопоставлены нагрузки группы. Проверьте общий параметр и единицы в Excel. ' + detail)
    result = {unit: [dict(r, key='нагрузка котла') if r['role'] == 'load' else dict(r) for r in rows]
              for unit, rows in rows_by_unit.items()}
    return result, True


def map_frame(raw, rows, unit):
    exact, normalized = {}, {}
    # Search physical headers directly: no boiler prefix, full-name equality,
    # or Excel formula spelling participates in imbalance identification.
    imbalance_columns = [column for column in raw if is_imbalance_name(column)]
    for column in raw:
        exact.setdefault(' '.join(str(column).split()), []).append(column)
        normalized.setdefault(canonical_name(column, unit), []).append(column)
    mapping, roles, units, audit = {}, {}, {}, []
    for row in rows:
        key = canonical_name(row['key'], unit)
        row = dict(row, key=key)
        if is_imbalance_name(row['original']) or is_imbalance_name(key):
            hits = imbalance_columns
        else:
            hits = (exact.get(' '.join(row['original'].split()), []) or
                    normalized.get(canonical_name(row['original'], unit), []) or normalized.get(key, []))
        status = 'excluded' if row['role'] == 'exclude' else 'matched' if hits else 'missing'
        audit.append(dict(unit=unit, **row, source=' | '.join(hits), status=status))
        if status != 'matched':
            continue
        if len(hits) != 1:
            raise ValueError(f'{unit}: неоднозначное имя {row["original"]}: {hits}')
        if row['key'] in roles:
            previous = next((k for k, v in mapping.items() if v == row['key']), None)
            if previous == hits[0] and roles[row['key']] == row['role']:
                audit[-1]['status'] = 'alias_duplicate'; continue
            raise ValueError(f'{unit}: общий параметр повторяется: {row["key"]}')
        if hits[0] in mapping:
            raise ValueError(f'{unit}: одному датчику назначено несколько общих параметров: {hits[0]}')
        mapping[hits[0]] = row['key']; roles[row['key']] = row['role']; units[row['key']] = row['units']
    # Construct the frame once: replacing hundreds of columns separately leaves
    # one pandas block per column and makes even metadata insertion expensive.
    columns = {}
    originals = {key: original for original, key in mapping.items()}
    for key, role in roles.items():
        values = raw[originals[key]]
        if role == 'state':
            columns[key] = values.astype('category')
        else:
            values = pd.to_numeric(values.astype(str).str.replace(',', '.', regex=False), errors='coerce')
            columns[key] = values.replace([np.inf, -np.inf], np.nan)
    for key in (K.STATE_COL, '_Тип'):
        if key in raw:
            columns[key] = raw[key]
    out = pd.DataFrame(columns, index=raw.index)
    if K.STATE_COL not in out:
        raise ValueError(f'{unit}: нет столбца {K.STATE_COL}')
    return out, roles, units, mapping, audit


def _clone(function, namespace):
    return types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


class _TrainingFunctions:
    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, key):
        return getattr(K, key)

    def time_blocks(self, index, count):
        # The original three chronological blocks are formed IN EACH boiler.
        positions = self.engine.pool_index.get_indexer(index)
        blocks = np.zeros(len(index), dtype=int)
        for left, right in self.engine.bounds.values():
            selected = np.flatnonzero((positions >= left) & (positions < right))
            blocks[selected] = K.time_blocks(index[selected], count)
        return blocks


class _DeployFunctions:
    """Add raw-name aliases to emitted legacy loaders without changing cores."""
    def __init__(self, module, engine):
        self.module, self.engine = module, engine

    def __getattr__(self, name):
        original = getattr(self.module, name)
        if name not in ('export_model', 'export_bagging', 'export_union'):
            return original
        def export(*args, **kwargs):
            result = original(*args, **kwargs)
            folder = Path(args[0] if args else kwargs['folder'])
            if name == 'export_model':
                core = args[2] if len(args) > 2 else kwargs['core']
                det = args[3] if len(args) > 3 else kwargs['det']
                prep = args[4] if len(args) > 4 else kwargs['prep']
                self._restore_memory_precision(folder, core, det)
                self._record_feature_order(folder, det, prep)
                views = args[5] if len(args) > 5 else kwargs['views']
                cfg_path = folder / 'config.json'
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
                cfg['legacy_q_dtypes'] = {v['name']: str(np.asarray(v['q']).dtype) for v in views}
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
                stage1 = kwargs.get('stage1')
                if stage1:
                    self._restore_memory_precision(folder / 'stage1', stage1[2], stage1[1])
                    self._record_feature_order(folder / 'stage1', stage1[1], stage1[0])
            for code in folder.rglob('model.py'):
                cfg_path = code.with_name('config.json')
                if not cfg_path.exists():
                    continue
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
                cfg.update(input_aliases=self.engine.contexts[self.engine.target]['input_aliases'],
                           training_units=self.engine.units, group=self.engine.name, legacy_roles_version=VERSION,
                           training_update=self.engine.ns.get('TRAIN_UPDATE', 'refit'))
                cfg['causal_cleaning'] = dict(columns=self.engine.targets, per_h=self.engine.ns['PER_H'],
                    window_h=self.engine.ns.get('CLEAN_WINDOW_H', 168), min_history_h=self.engine.ns.get('CLEAN_MIN_HISTORY_H', 24),
                    spike_factor=self.engine.ns.get('CLEAN_SPIKE_FACTOR', 8.), frozen_h=self.engine.ns.get('CLEAN_FROZEN_H', 24))
                if name == 'export_model':
                    cfg['calibration_strategy'] = getattr(det, '_calibration_strategy', 'refit_oof')
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
                text = code.read_text(encoding='utf-8')
                # Preserve the original operation order too: cancellation in
                # float32 can amplify algebraically equivalent PCA/Mahal code.
                text = text.replace('self.mean, self.comps, self.n = p[\'mean\'], p[\'comps\'], int(p[\'n\'])',
                    "self.mean, self.comps, self.n = p['mean'], p['comps'], int(p['n'])\n        import joblib\n        self.pca = joblib.load(os.path.join(folder, 'legacy_pca.joblib'))")
                text = text.replace('rec = (F - self.mean) @ self.comps.T @ self.comps + self.mean',
                                    'rec = self.pca.inverse_transform(self.pca.transform(F))')
                text = text.replace('C = D * (D @ self.P) / F.shape[1]\n        return dict(E=C.sum(axis=1)[:, None], C=C)',
                    "M = D @ self.P\n        C = D * M / F.shape[1]\n        return dict(E=(np.einsum('ij,ij->i', M, D) / F.shape[1])[:, None], C=C)")
                # Match the old notebook's actual feature precision: derive
                # load changes from raw values, then cast physical inputs to f32.
                precision = """        if cfg['input'] != 'residuals':
            d = d.copy()
            physical = [c for c in self.scaler.num if c in d and not c.startswith('условие: ')]
            d[physical] = d[physical].astype('float32')
"""
                marker = '        blank = [c for c in (cfg.get(\'blank_cols\') or [])'
                if precision not in text:
                    text = text.replace(marker, precision + marker)
                cleaning_code = """        if cfg['input'] != 'residuals' and cfg.get('causal_cleaning'):
            spec = dict(cfg['causal_cleaning'])
            spec['columns'] = [c for c in spec['columns'] if c in d]
            modes = detect_mode(df[cfg['run_col']].astype(float), cfg['run_thr'], cfg['startup_steps'],
                                cfg['prestop_steps'], cfg['min_stop_steps'], cfg['ffill_steps'])
            d, _ = causal_clean_numeric(d, modes.ne(M_STOP), **spec)
"""
                if cleaning_code not in text:
                    text = text.replace(marker, cleaning_code + marker)
                if 'def causal_clean_numeric(' not in text:
                    text += '\n\n' + Path(__file__).with_name('legacy_cleaning.py').read_text(encoding='utf-8')
                text = text.replace('F, _ = self.scaler.transform(Z)', "F, _ = self.scaler.transform(Z.astype('float32'))")
                # CSV loads q as float64. Keep the fitted dtype: dividing f32
                # errors by f64 changes rounding before the old score average.
                marker = "        self.qc = sens['обычный вклад']"
                restore_q = "        for name, dtype in self.cfg.get('legacy_q_dtypes', {}).items():\n            self.q[name] = self.q[name].astype(dtype)\n"
                if restore_q not in text:
                    text = text.replace(marker, restore_q + marker)
                text = text.replace('out = self.core.errors(F[pos])',
                    "out = self.core.errors(np.array(F[pos], order=self.cfg.get('feature_order', 'C'), copy=True))")
                needle = 'df = df if prepared else prepare(df)'
                replacement = "df = df.rename(columns=self.cfg.get('input_aliases', {}))\n        " + needle
                if replacement not in text:
                    text = text.replace(needle, replacement)
                code.write_text(text, encoding='utf-8')
            return result
        return export

    def _record_feature_order(self, folder, det, prep):
        if not hasattr(self.engine, 'ns') or prep is None or getattr(det, 'L', 0):
            return
        ns = self.engine.ns
        rows = np.zeros(len(ns['t_ix']), bool)
        rows[-min(128, len(rows)):] = True
        values = ns['inputs'](det, prep, rows)[0]
        path = Path(folder) / 'config.json'
        cfg = json.loads(path.read_text(encoding='utf-8'))
        cfg['feature_order'] = 'F' if values.flags.f_contiguous and not values.flags.c_contiguous else 'C'
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')

    @staticmethod
    def _restore_memory_precision(folder, core, det):
        if core == 'pca':
            joblib.dump(det.pca, Path(folder) / 'legacy_pca.joblib', compress=3)
            return
        if core not in ('analogs', 'analogs_boost'):
            return
        path = Path(folder) / 'params.npz'
        with np.load(path) as archive:
            params = {key: archive[key] for key in archive.files}
        # Legacy exporter downcasts Xtr; preserving fitted precision changes no
        # model, but prevents amplified roundoff for almost deterministic sensors.
        params['Xtr'] = det.am.Xtr
        np.savez_compressed(path, **params)


class GroupTraining:
    def __init__(self, ns):
        self.ns, self.target = ns, ns['UNIT']
        self.generated = ns.get('_RUN_GENERATED_MODELS', set())
        self.fit_audit = []
        self.operation_audit = []
        self._successful = {}
        self._fast_states = {}
        groups = ns.get('GROUPS', {})
        selected = ns.get('GROUP_NAME')
        if selected is None and groups:
            matches = [g for g, units in groups.items() if self.target in units]
            if len(matches) != 1:
                raise ValueError('Укажите GROUP_NAME: котёл должен входить ровно в одну выбранную группу.')
            selected = matches[0]
        if selected is not None and selected not in groups:
            raise ValueError(f'Нет группы {selected}')
        self.name = selected or f'отдельно_{self.target}'
        if re.search(r'[<>:"/\\|?*]', self.name) or self.name.strip() in ('', '.', '..'):
            raise ValueError('GROUP_NAME должен быть допустимым именем папки Windows.')
        self.schema_units = list(groups[selected]) if selected is not None else [self.target]
        if not self.schema_units or len(set(self.schema_units)) != len(self.schema_units) or self.target not in self.schema_units:
            raise ValueError('В группе нужны разные котлы, включая UNIT.')
        self.units = list(ns.get('TRAIN_UNITS', self.schema_units))
        if not self.units or len(set(self.units)) != len(self.units) or not set(self.units) <= set(self.schema_units):
            raise ValueError('TRAIN_UNITS: нужны неповторяющиеся котлы выбранной группы.')
        self.contexts, records = {}, []
        schema = {}
        role_rows, aligned = align_load_roles(
            {u: read_roles(ns['SENSOR_LIST_FILE'], u) for u in self.schema_units},
            ns.get('ALIGN_LOAD_ROLE', True))
        if aligned and ns.get('VERBOSE', False):
            print('Нагрузки сопоставлены по роли расхода пара; исходные точки измерения сохранены в sensor_audit.csv.', flush=True)
        for unit in self.schema_units:
            c = dict(ns, UNIT=unit)
            hourly = ns.get('DATA_PATHS', {}).get(unit) or (ns.get('HOURLY_PATH') if unit == self.target else None)
            hourly = hourly or str(Path(ns.get('DATA_ROOT', 'opc_wide')) / unit / 'hourly.parquet')
            instant = ns.get('INSTANT_PATHS', {}).get(unit) or (ns.get('INSTANT_PATH') if unit == self.target else None)
            instant = instant or str(Path(ns.get('DATA_ROOT', 'opc_wide')) / unit / 'instant.parquet')
            raw, source_types, _ = K.load_grid(ns['DATA_KIND'], hourly, instant, ns['INSTANT_STEP'], 'cache')
            # The old hourly loader drops columns classified as entirely empty.
            # Preserve their known schema: "present but no data" is not "absent".
            for column in source_types.get('empty', []):
                if column not in raw:
                    raw[column] = np.nan
            rows = role_rows[unit]
            df, roles, declared_units, mapping, audit = map_frame(raw, rows, unit)
            schema[unit] = declared_units; records.extend(audit)
            c.update(df=df, roles=roles, input_aliases=mapping, sensor_audit=audit,
                     source_imbalance_columns=[str(k) for k in raw if 'небаланс' in canonical_name(k)],
                     HOURLY_PATH=hourly, INSTANT_PATH=instant)
            self.contexts[unit] = c
        self.root = Path(ns.get('LEGACY_OUTPUT_DIR', 'anomaly_results_legacy')) / self.name
        if ns.get('EXPERIMENT_KEY'):
            self.root /= ns['EXPERIMENT_KEY']
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared_root = Path(ns.get('LEGACY_OUTPUT_DIR', 'anomaly_results_legacy')) / self.name / '_shared'
        self.audit = pd.DataFrame(records)
        common = set.intersection(*(set(c['roles']) for c in self.contexts.values()))
        self.audit['in_group'] = self.audit.key.isin(common) & self.audit.status.eq('matched')
        self.audit.to_csv(self.root / 'sensor_audit.csv', index=False, encoding='utf-8-sig')
        imbalance_rows = [r for r in records if is_imbalance_name(r['original']) or is_imbalance_name(r['key']) or is_imbalance_name(r['source'])]
        pd.DataFrame(imbalance_rows, columns=list(self.audit.columns[:-1])).to_csv(
            self.root / 'imbalance_audit.csv', index=False, encoding='utf-8-sig')
        for key in common:
            if len({c['roles'][key] for c in self.contexts.values()}) != 1:
                raise ValueError(f'Разные роли общего параметра {key}')
            declared = {schema[u][key] for u in self.schema_units if schema[u][key]}
            if len(declared) > 1:
                raise ValueError(f'Разные единицы общего параметра {key}: {declared}')
        imbalance_keys = []
        details = []
        for unit, c in self.contexts.items():
            rows = [r for r in imbalance_rows if r['unit'] == unit]
            matched = {r['key'] for r in rows if r['status'] in ('matched', 'alias_duplicate')}
            imbalance_keys.append(matched)
            details.append(f'{unit}: ' + ('; '.join(
                f"Excel={r['original']!r}, общий ключ={r['key']!r}, найдено={r['source']!r}, статус={r['status']}"
                for r in rows) if rows else 'в Excel нет строки небаланса') +
                f"; заголовки данных={c['source_imbalance_columns']!r}")
        common_imbalance = set.intersection(*imbalance_keys) & common
        if len(common_imbalance) != 1:
            raise ValueError('Небаланс не сопоставлен однозначно для всей группы.\n' +
                '\n'.join(details) + f'\nПодробности: {self.root / "imbalance_audit.csv"}')
        first = self.contexts[self.units[0]]['roles']
        self.targets = sorted(k for k in common if first[k] == 'target')
        self.conditions = sorted(k for k in common if first[k] in ('condition', 'load'))
        self.states = sorted(k for k in common if first[k] == 'state')
        loads = [k for k in common if first[k] == 'load']
        if len(loads) != 1 or len(self.targets) < 2:
            raise ValueError(f'Общий набор {self.name}: нагрузок={len(loads)}, проверяемых датчиков={len(self.targets)}. '
                             'Нужны одна нагрузка и минимум два проверяемых; подробности в sensor_audit.csv.')
        self.load = loads[0]
        for unit, c in self.contexts.items():
            if [k for k, r in c['roles'].items() if r == 'load'] != loads:
                raise ValueError(f'{unit}: назначьте одну и ту же физическую нагрузку в Excel.')
            self._prepare(c)
        ns.update(self.contexts[self.target])
        ns['RES_DIR'] = str(self.root / self.target)
        Path(ns['RES_DIR']).mkdir(parents=True, exist_ok=True)
        ns['MODELS_DIR'] = str(Path(ns.get('LEGACY_MODELS_DIR', 'models_legacy')) / self.name / ns.get('EXPERIMENT_KEY', ''))
        self._make_pool_index()
        data_hash = hashlib.sha256()
        for unit in self.units:
            c = self.contexts[unit]
            data_hash.update(unit.encode('utf-8'))
            data_hash.update(pd.util.hash_pandas_object(c['df'], index=True).values.tobytes())
        notebook_path = Path(ns.get('__file__', Path(__file__).resolve().parents[1] / '02_Обучение.ipynb'))
        if ns.get('LEGACY_CORE_FILE'):
            model_code = Path(ns['LEGACY_CORE_FILE']).read_text(encoding='utf-8')
        else:
            code = json.loads(notebook_path.read_text(encoding='utf-8'))['cells']
            model_code = ''.join(''.join(code[i]['source']) for i in (3, 9, 11, 13, 15, 17, 19, 21, 23))
        signature_data = dict(version=VERSION, units=self.units, data=data_hash.hexdigest(),
            notebook_models=hashlib.sha256(model_code.encode('utf-8')).hexdigest(),
            roles={u: c['roles'] for u, c in self.contexts.items()},
            sources={f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()
                     for f in ('legacy_group.py', 'legacy_cleaning.py', 'k2_lib.py', 'kernel_models.py')},
            settings={k: ns.get(k) for k in ('TRAIN_MODE', 'TRAIN_HOURS', 'TRAIN_MAX_DAYS', 'TRAIN_DAYS',
                'ALARM_Q', 'SMOOTH_H', 'PERSIST_H', 'PRE_WINDOW_D', 'NEUTRAL_D', 'MIN_TRAIN_COVER',
                'USE_NEURAL', 'DATA_KIND', 'INSTANT_STEP', 'AAKR_H', 'MSET_H',
                'RESIDUAL_AE_EPOCHS', 'RESIDUAL_LSTM_EPOCHS', 'RESIDUAL_LSTM_STEPS',
                'RESIDUAL_MAX_TRAIN', 'CLEAN_WINDOW_H', 'CLEAN_MIN_HISTORY_H', 'CLEAN_SPIKE_FACTOR', 'CLEAN_FROZEN_H')},
            regimes={u: (self.contexts[u]['run_thr'], self.contexts[u]['startup_h'], self.contexts[u]['COND_SPEC']) for u in self.units})
        self.compat_signature = None
        # Old releases used full-dataset statistics and future samples in cleaning.
        # Their fitted models/scores are not compatible with the causal filters.
        fast_start = ns.get('FIRST_TEST')
        if ns.get('FIRST_TEST_AUTO'):
            fast_start = ns.get('eda', {}).get('first_test') or K.first_test_month(ns['t_ix'], ns['normal'], ns['MIN_NORMAL_H'] * ns['PER_H'])
        self.signature = _hash(dict(signature_data, update=ns.get('TRAIN_UPDATE', 'refit'),
                                    fast_start=str(fast_start) if ns.get('TRAIN_UPDATE') == 'fast' else None,
                                    import_legacy_cache=ns.get('IMPORT_LEGACY_CACHE', True),
                                    fast_min_cal_h=ns.get('FAST_MIN_CAL_H', 72)))
        ns['GROUP_ENGINE'] = self

    def _prepare(self, c):
        df, unit = c['df'], c['UNIT']
        step = pd.Timedelta('1h' if c['DATA_KIND'] == 'hourly' else c['INSTANT_STEP'])
        per_h = max(1, int(round(pd.Timedelta('1h') / step)))
        hs = lambda h: max(1, int(round(h * per_h)))
        state = df[K.STATE_COL].astype(object).map(lambda v: v.strip() if isinstance(v, str) and v.strip() else None).ffill(limit=hs(3)).fillna(K.ST_NODATA)
        eda, eda_file = {}, None
        path = Path('eda_results') / unit / 'eda_decisions.json'
        if path.exists():
            eda = json.loads(path.read_text(encoding='utf-8')); eda_file = str(path)
        run_thr = c.get('RUN_THRESHOLDS', {}).get(unit)
        if run_thr is None:
            eda_load = eda.get('run_col')
            same_load = eda_load is None or canonical_name(eda_load, unit) == canonical_name(self.load, unit)
            run_thr = (eda.get('run_thr') if same_load else None) or K.run_threshold(df[self.load].astype(float), state, .25)
        startup_h = int(eda.get('startup_h', 24))
        mode, _, _ = K.detect_mode(df[self.load].astype(float), run_thr, hs(startup_h), hs(3), hs(2), ffill_h=hs(3))
        running = mode.ne(K.M_STOP)
        test_h = (df['_Тип'].astype(str).isin(['ИСП', 'ЗРР']) if '_Тип' in df else pd.Series(False, index=df.index)).to_numpy()
        scorable = running & mode.ne(K.M_START) & ~pd.Series(test_h, index=df.index)
        events = K.repair_events(state, 24)
        def stop_time(e):
            if not running.get(e.start, False):
                on = running[:e.start]; on = on[on]
                return on.index[-1] + step if len(on) else e.start
            after = running[e.start:e.start + pd.Timedelta('72h')]
            return after.index[~after.to_numpy()][0] if (~after).any() else e.start
        events['stop'] = [stop_time(e) for e in events.itertuples()]
        after_rep = np.zeros(len(df), bool)
        for e in events.itertuples():
            after_rep |= (df.index >= e.end) & (df.index < e.end + pd.Timedelta('48h'))
        normal = mode.eq(K.M_RUN) & state.eq(K.ST_WORK) & ~after_rep & ~test_h
        columns = self.targets + self.conditions
        X = df[columns].astype('float32')
        X, cleaning = causal_clean_numeric(X, running, self.targets, per_h=per_h,
            window_h=c.get('CLEAN_WINDOW_H', 168), min_history_h=c.get('CLEAN_MIN_HISTORY_H', 24),
            spike_factor=c.get('CLEAN_SPIKE_FACTOR', 8.), frozen_h=c.get('CLEAN_FROZEN_H', 24))
        spec = dict(run_col=self.load, run_thr=float(run_thr), per_h=per_h,
                    extra={k: k for k in self.conditions if k != self.load}, mills={})
        if c.get('RX_MILLS'):
            mills = K.condition_spec(df, self.load, run_thr, per_h, {}, K.unit_rx(c['RX_MILLS'], unit), normal, columns)
            spec['mills'] = mills['mills']
        CF = K.condition_frame(df, spec)
        cond_cols = K.condition_columns(spec)
        XC = pd.concat([X[self.targets], df[self.conditions].astype('float32'), CF], axis=1)
        T = df[self.states] if self.states else None
        c.update(STEP=step, PER_H=per_h, hs=hs, KIND_TAG=c['DATA_KIND'] if c['DATA_KIND'] == 'hourly' else f'instant_{c["INSTANT_STEP"]}',
            state=state, eda=eda, eda_file=eda_file, test_h=test_h, run_col=self.load, run_thr=run_thr, cleaning=cleaning,
            startup_h=startup_h, mode=mode, running=running, scorable=scorable, events=events,
            fails=events[events.type.eq(K.EV_FAIL)], plans=events[events.type.eq(K.EV_PLAN)], reserves=events[events.type.eq(K.EV_RES)],
            normal=normal, TARGETS=self.targets, COND_RAW=self.conditions, STATE_COLS=self.states,
            COND_SPEC=spec, COND_SPEC_REG=spec, COND_COLS=cond_cols, COND_COLS_REG=cond_cols,
            X=X, X_all=X, X_cond=XC, X_reg=XC.copy(), MISS=X.isna().to_numpy(), MISS_all=X.isna().to_numpy(),
            MISS_cond=XC.isna().to_numpy(), MISS_reg=XC.isna().to_numpy(), T=T, sens=columns, cand=columns,
            num_all=columns, txt_cols=self.states, enough=X.notna().mean(axis=1).ge(.7),
            prep_all=K.Prep(cols=self.states).fit(X.loc[normal], T.loc[normal] if T is not None else None),
            t_ix=df.index, RESID={}, SUBSET_DATA={}, types={'num': columns, 'txt': self.states})
        c['SENSOR_LIST_INFO'] = dict(listed=len(c['roles']), found=len(columns) + len(self.states), missing=[])

    def _make_pool_index(self):
        self.bounds, parts, actual = {}, [], []
        cursor, moment = 0, pd.Timestamp('1900-01-01')
        for unit in self.units:
            c = self.contexts[unit]
            if c['STEP'] != self.contexts[self.target]['STEP']:
                raise ValueError('У котлов группы должны совпадать шаги телеметрии.')
            n = len(c['df']); self.bounds[unit] = (cursor, cursor + n); cursor += n
            index = pd.date_range(moment, periods=n, freq=c['STEP'])
            if n:
                moment = index[-1] + pd.Timedelta(days=365)
            parts.append(index); actual.append(c['df'].index)
        self.pool_index = parts[0].append(parts[1:])
        self.actual_index = actual[0].append(actual[1:])

    def _stack(self, key):
        frames = [self.contexts[u][key] for u in self.units]
        if frames[0] is None:
            return None
        if isinstance(frames[0], np.ndarray):
            return np.concatenate(frames)
        result = pd.concat(frames, ignore_index=True)
        result.index = self.pool_index
        return result

    def install(self, ns):
        self.original_train = ns['train_one']
        self.helpers = {k: ns[k] for k in ('data_of', 'inputs', 'blank_cols', 'model_errors')}
        self.original_masks = ns['masks']
        for c in self.contexts.values():
            c['masks'] = _clone(self.original_masks, c)
        evaluation_hash = hashlib.sha256(pd.util.hash_pandas_object(self.contexts[self.target]['df'], index=True).values.tobytes()).hexdigest()
        ns['SIG'] = dict(ns['SIG'], group_signature=self.signature, evaluation_hash=evaluation_hash,
                         training_units=self.units, roles=True)
        ns['train_one'] = self.train_one
        ns['D'] = _DeployFunctions(ns['D'], self)
        if ns.get('FIRST_TEST_AUTO'):
            # Same old rule; target determines the start of its own report.
            ns['FIRST_TEST'] = ns.get('eda', {}).get('first_test') or K.first_test_month(ns['t_ix'], ns['normal'], ns['MIN_NORMAL_H'] * ns['PER_H'])

    def _pool_namespace(self, residual=False):
        pool = dict(self.ns)
        for key in ('df', 'X', 'X_all', 'X_cond', 'X_reg', 'MISS', 'MISS_all', 'MISS_cond', 'MISS_reg', 'T', 'normal', 'enough', 'scorable'):
            pool[key] = self._stack(key)
        pool.update(t_ix=self.pool_index, K=_TrainingFunctions(self), SUBSET_DATA={})
        if residual:
            frames = [self.contexts[u]['RESID']['X'] for u in self.units]
            X = pd.concat(frames, ignore_index=True); X.index = self.pool_index
            pool['RESID'] = dict(X=X, M=np.concatenate([self.contexts[u]['RESID']['M'] for u in self.units]))
        pool['masks'] = lambda m0: np.concatenate([self.contexts[u]['masks'](m0) for u in self.units])
        for key, func in self.helpers.items():
            pool[key] = _clone(func, pool)
        original_inputs = pool['inputs']
        def inputs(det, prep, rows, train=False, max_rows=None):
            length = getattr(det, 'L', 0)
            if length:
                rows = rows.copy()
                for left, right in self.bounds.values():
                    rows[left:min(right, left + length - 1)] = False
            if not np.any(rows):
                p = len(prep.names)
                shape = (0, length, p) if length else (0, p)
                mask_shape = (0, length, len(prep.num)) if length else (0, len(prep.num))
                return np.empty(shape), pd.DatetimeIndex([]), np.empty(mask_shape, bool)
            return original_inputs(det, prep, rows, train, max_rows)
        pool['inputs'] = inputs
        return pool

    residual_kind = None

    def _request_key(self, signature, det, m0, views, oof_rows):
        return _hash(dict(signature=signature, cutoff=str(m0), views=views, oof_rows=oof_rows,
                          kind=self.residual_kind if getattr(det, 'on_residuals', False) else None,
                          detector=type(det).__name__, parameters=det.__dict__))

    def _cached_raw(self, det, m0, views, oof_rows):
        for signature in (self.signature, self.compat_signature if self.ns.get('IMPORT_LEGACY_CACHE', True) else None):
            if not signature:
                continue
            key = self._request_key(signature, det, m0, views, oof_rows)
            path = self.shared_root / '_trained' / signature[:16] / (key + '.pkl')
            if path.exists() and (not self.ns['RECOMPUTE'] or key in self.generated):
                with path.open('rb') as f:
                    result = cloudpickle.load(f)
                result['cache_reused'] = True
                result.setdefault('update_mode', 'legacy_oof_cache')
                result['det']._calibration_strategy = result['update_mode']
                return result
        return None

    def _selection_hash(self, m0):
        digest = hashlib.sha256()
        for unit in self.units:
            digest.update(np.asarray(self.contexts[unit]['masks'](m0), bool).tobytes())
        return digest.hexdigest()

    def _fast_token(self, det, views, oof_rows):
        return _hash(dict(detector=type(det).__name__, parameters=det.__dict__, views=views, oof_rows=oof_rows))

    def _remember_fast(self, det, raw, requested, views, oof_rows):
        if (self.ns.get('TRAIN_UPDATE') == 'fast' and not getattr(det, 'on_residuals', False)
                and type(det).__name__ in ('AnalogCondDet', 'KernelDet', 'BoostDet', 'TwinsDet')):
            self._fast_states[self._fast_token(det, views, oof_rows)] = (pd.Timestamp(requested), raw)

    def _previous_fast(self, det, m0, views, oof_rows):
        known = self._fast_states.get(self._fast_token(det, views, oof_rows))
        if known and known[0] >= pd.Timestamp(m0):
            known = None
        # A resumed run or an export may start without the in-memory predecessor.
        for cutoff in reversed(self.ns.get('months', [])):
            if cutoff >= pd.Timestamp(m0):
                continue
            if known and cutoff <= known[0]:
                return known
            raw = self._cached_raw(det, cutoff, views, oof_rows)
            if raw is not None:
                return pd.Timestamp(cutoff), raw
        return known

    def _fast_train(self, make, det0, m0, views, oof_rows):
        # A prequential model depends on its predecessors. Build the same chronological
        # chain even if the first evaluated boiler had a missing/stopped month.
        # This is part of model updating, never triggered by export_history.
        moment = pd.Timestamp(m0)
        preceding = moment.to_period('M').to_timestamp()
        if preceding == moment:
            preceding -= pd.offsets.MonthBegin(1)
        start = pd.Timestamp(self.ns['FIRST_TEST']).to_period('M').to_timestamp()
        if start <= preceding < moment:
            self._train_raw(make, preceding, views, oof_rows)
        previous = self._previous_fast(det0, m0, views, oof_rows)
        if previous is None:
            return None  # Bootstrap uses the original three OOF folds.
        cutoff, raw = previous
        selection = self._selection_hash(m0)
        old_selection = raw.get('training_selection_hash') or self._selection_hash(pd.Timestamp(raw['trained_cutoff']))
        if selection == old_selection:
            return dict(raw, update_mode='unchanged_training_reuse', cache_reused=True,
                        calibration_until=str(m0), training_selection_hash=selection,
                        update_error_rows=0, fit_calls=0)
        pool = self._pool_namespace(False)
        train = pool['masks'](m0)
        if train.sum() < self.ns['hs'](500):
            return None
        Xd, _, Td = pool['data_of'](det0)
        cover = Xd.loc[train].notna().mean()
        blank = list(cover.index[cover < self.ns['MIN_TRAIN_COVER']])
        if blank != raw['blank']:
            return None  # Changed sensor availability needs a fresh OOF bootstrap.
        # Existing errors are retained only for hours still in the current healthy window.
        parts = []
        for error, index in raw['parts']:
            positions = self.pool_index.get_indexer(index)
            keep = positions >= 0
            keep[keep] &= train[positions[keep]]
            if keep.any():
                parts.append((error[keep], index[keep]))
        # Predict BEFORE fitting the updated model. No new hour calibrates itself in-sample.
        new_rows = train & (self.actual_index >= cutoff) & (self.actual_index < pd.Timestamp(m0))
        predicted = 0
        for left, right in self.bounds.values():
            rows = np.zeros(len(train), bool)
            rows[left:right] = new_rows[left:right]
            if rows.any():
                error, index = pool['model_errors'](raw['det'], raw['prep'], rows)
                if error is not None:
                    parts.append((error, index))
                    predicted += len(index)
        if sum(len(index) for _, index in parts) < self.ns['hs'](self.ns.get('FAST_MIN_CAL_H', 72)):
            return None
        cal = {v['name']: K.calibrate(parts, v['top'], v.get('slow', 0), self.ns['ALARM_Q']) for v in views}
        prep = K.Prep(cols=self.ns['STATE_COLS']).fit(Xd.loc[train], Td.loc[train] if Td is not None else None)
        det = make()
        det.blank = blank
        det.fit(pool['inputs'](det, prep, train, train=True)[0], prep)
        return dict(prep=prep, det=det, cal=cal, parts=parts, qc=None, blank=blank,
                    update_mode='fast_prequential', previous_cutoff=str(cutoff),
                    calibration_until=str(m0), update_error_rows=predicted,
                    training_selection_hash=selection, fit_calls=1)

    def _train_raw(self, make, m0, views, oof_rows):
        det = make()
        key = self._request_key(self.signature, det, m0, views, oof_rows)
        path = self.shared_root / '_trained' / self.signature[:16] / (key + '.pkl')
        result = self._cached_raw(det, m0, views, oof_rows)
        if result is not None:
            self._remember_fast(det, result, m0, views, oof_rows)
            return result
        started = time.perf_counter()
        if self.ns.get('VERBOSE', False):
            print(f'  {type(det).__name__} {getattr(det, "kind", "")}, {pd.Timestamp(m0):%Y-%m-%d}: '
                  + ', '.join(f'{u}: {self.contexts[u]["masks"](m0).sum() / self.contexts[u]["PER_H"]:.0f} ч' for u in self.units), flush=True)
        supported = (self.ns.get('TRAIN_UPDATE') == 'fast' and not getattr(det, 'on_residuals', False)
                     and type(det).__name__ in ('AnalogCondDet', 'KernelDet', 'BoostDet', 'TwinsDet'))
        result = self._fast_train(make, det, m0, views, oof_rows) if supported else None
        if result is None:
            pool = self._pool_namespace(getattr(det, 'on_residuals', False))
            result = _clone(self.original_train, pool)(make, m0, views, oof_rows)
            if result is not None:
                result.update(update_mode='oof_bootstrap' if supported else 'refit_oof', fit_calls=self.ns['N_FOLDS'] + 1)
        if result is not None:
            result['training_units'] = self.units
            result['training_rows_by_unit'] = {u: int(self.contexts[u]['masks'](m0).sum()) for u in self.units}
            if result.get('update_mode') != 'unchanged_training_reuse':
                result['trained_cutoff'] = str(m0)
                result['cache_reused'] = False
            result['training_selection_hash'] = self._selection_hash(m0)
            result['calibration_until'] = str(m0)
            result['update_seconds'] = time.perf_counter() - started
            result['det']._calibration_strategy = result['update_mode']
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as f:
                cloudpickle.dump(result, f)
            os.replace(tmp, path)
            self.generated.add(key)
            path.with_suffix('.json').write_text(json.dumps(dict(cutoff=str(m0), method=type(det).__name__,
                residual_kind=self.residual_kind if getattr(det, 'on_residuals', False) else None,
                parameters=det.__dict__, views=views, oof_rows=oof_rows,
                training_rows_by_unit=result['training_rows_by_unit'],
                update_mode=result['update_mode'], fit_calls=result.get('fit_calls'),
                update_seconds=result['update_seconds'], update_error_rows=result.get('update_error_rows'),
                blank=result['blank'], key=key), ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            self._remember_fast(det, result, m0, views, oof_rows)
            self.operation_audit.append(dict(cutoff=str(m0), mode=result['update_mode'], fit_calls=result.get('fit_calls', 0)))
            if self.ns.get('VERBOSE', False):
                print(f'    {result["update_mode"]}: fit={result.get("fit_calls", "—")}, '
                      f'новых калибровочных часов={result.get("update_error_rows", "—")}, '
                      f'{result["update_seconds"]:.1f} с', flush=True)
        return result

    def _unit_parts(self, parts, unit):
        if unit not in self.bounds:
            return []
        left, right = self.bounds[unit]
        result = []
        for error, index in parts:
            positions = self.pool_index.get_indexer(index)
            keep = (positions >= left) & (positions < right)
            if keep.any():
                result.append((error[keep], self.actual_index[positions[keep]]))
        return result

    def _for_target(self, raw):
        if raw is None:
            return None
        return dict(raw, parts=self._unit_parts(raw['parts'], self.target), group_parts=raw['parts'])

    def train_one(self, make, m0, views, oof_rows='часть'):
        det = make()
        token = _hash(dict(detector=type(det).__name__, parameters=det.__dict__, views=views,
                           oof_rows=oof_rows, residual=self.residual_kind if getattr(det, 'on_residuals', False) else None))
        raw = self._train_raw(make, m0, views, oof_rows)
        status = 'cache' if raw is not None and raw.get('cache_reused') else 'fit' if raw is not None else 'insufficient_training'
        if raw is not None:
            self._successful[token] = {pd.Timestamp(m0): raw}
        elif self.ns.get('REUSE_PREVIOUS_MODEL', True) and not getattr(det, 'on_residuals', False):
            candidates = {k: v for k, v in self._successful.get(token, {}).items() if k < pd.Timestamp(m0)}
            if not candidates and hasattr(self, 'shared_root'):
                stored = []
                for metadata in (self.shared_root / '_trained' / self.signature[:16]).glob('*.json'):
                    record = json.loads(metadata.read_text(encoding='utf-8'))
                    if (pd.Timestamp(record['cutoff']) < pd.Timestamp(m0) and
                        record['method'] == type(det).__name__ and
                        _hash(record['parameters']) == _hash(det.__dict__) and
                        record.get('views') == views and record.get('oof_rows') == oof_rows and
                        record.get('residual_kind') is None):
                        stored.append((pd.Timestamp(record['cutoff']), metadata.with_suffix('.pkl')))
                if stored:
                    cutoff, saved = max(stored, key=lambda item: item[0])
                    with saved.open('rb') as f:
                        candidates[cutoff] = cloudpickle.load(f)
            if candidates:
                raw = candidates[max(candidates)]
                status = 'previous_model'
        self.fit_audit.append(dict(cutoff=str(m0), detector=type(det).__name__, status=status,
                                  update_mode=raw.get('update_mode') if raw else None,
                                  update_seconds=raw.get('update_seconds') if raw else None,
                                  fit_calls=(0 if status in ('cache', 'previous_model') else raw.get('fit_calls')) if raw else None,
                                  new_calibration_rows=raw.get('update_error_rows') if raw else None,
                                  trained_cutoff=raw.get('trained_cutoff') if raw else None,
                                  training_rows_by_unit=raw.get('training_rows_by_unit') if raw else None))
        return self._for_target(raw)

    def model_for_export(self, make, m0, views):
        det = make()
        raw = self._cached_raw(det, m0, views, 'часть')
        if raw is None and self.ns.get('HISTORY_MISSING', 'previous') == 'previous':
            previous = self._previous_fast(det, m0, views, 'часть')
            raw = previous[1] if previous else None
            if raw is not None and self.ns.get('VERBOSE', False):
                print(f'  Экспорт {pd.Timestamp(m0):%Y-%m}: предыдущая модель от {raw["trained_cutoff"]}; без обучения.', flush=True)
        return self._for_target(raw)

    def install_residual_stage(self, ns):
        self.residual_stage_audit = []
        generated = ns.get('_RUN_GENERATED_RESIDUALS', set())
        def stage1(kind, m0, L=0):
            self.residual_kind = kind
            st = ns['STAGES'][kind]
            raw = self._train_raw(st['make'], m0, [dict(name='z', top=1)], None)
            if raw is None:
                return None
            key = _hash(dict(signature=self.signature, kind=kind, cutoff=str(m0), length=L,
                             detector=type(raw['det']).__name__))
            residual_data = hashlib.sha256()
            for c in self.contexts.values():
                residual_data.update(pd.util.hash_pandas_object(c['X_cond'], index=True).values.tobytes())
            key = _hash(dict(model=key, evaluation_data=residual_data.hexdigest()))
            path = self.shared_root / '_residuals' / self.signature[:16] / (key + '.joblib')
            reused = path.exists() and (not ns['RECOMPUTE'] or key in generated)
            if reused:
                matrices = joblib.load(path)
                for unit, matrix in matrices.items():
                    self.contexts[unit]['RESID'].update(X=matrix, M=self.contexts[unit]['MISS_cond'][:, :len(self.targets)])
                ns['RESID'] = self.contexts[self.target]['RESID']
                self.residual_stage_audit.append(dict(base=kind, cutoff=str(m0), length=L, reused=True))
                return self._for_target(raw)
            for unit, c in self.contexts.items():
                c['RESID']['M'] = c['MISS_cond'][:, :len(self.targets)]
                local = dict(c)
                for helper_name, function in self.helpers.items():
                    local[helper_name] = _clone(function, local)
                rows = c['scorable'].to_numpy() & c['enough'].to_numpy() & (c['t_ix'] >= m0 - L * c['STEP']) & (c['t_ix'] < m0 + pd.offsets.MonthBegin(1))
                error, index = local['model_errors'](raw['det'], raw['prep'], rows) if rows.any() else (None, None)
                Z = np.full((len(c['t_ix']), len(self.targets)), np.nan, np.float32)
                pieces = ([(error, index)] if error is not None else []) + self._unit_parts(raw['parts'], unit)
                for e, ix in pieces:
                    Z[c['t_ix'].get_indexer(ix)] = e[:, :len(self.targets)]
                c['RESID']['X'] = pd.DataFrame(Z, index=c['t_ix'], columns=self.targets)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            joblib.dump({u: c['RESID']['X'] for u, c in self.contexts.items()}, tmp, compress=3)
            os.replace(tmp, path)
            generated.add(key)
            self.residual_stage_audit.append(dict(base=kind, cutoff=str(m0), length=L, reused=False))
            ns['RESID'] = self.contexts[self.target]['RESID']
            return self._for_target(raw)
        ns['stage1'] = stage1
        old_bag = ns['bag_threshold']
        def bag_threshold(members):
            return old_bag([(dict(tm, parts=tm.get('group_parts', tm['parts'])), view) for tm, view in members])
        ns['bag_threshold'] = bag_threshold

    def summary(self):
        records = []
        for path in (self.shared_root / '_trained' / self.signature[:16]).glob('*.json'):
            record = json.loads(path.read_text(encoding='utf-8'))
            for unit, count in record['training_rows_by_unit'].items():
                records.append(dict(cutoff=record['cutoff'], method=record['method'], unit=unit,
                    residual_kind=record['residual_kind'], train_hours=count / self.contexts[unit]['PER_H'],
                    blank=' | '.join(record['blank']), model_key=record['key']))
        result = pd.DataFrame(records)
        result.to_csv(self.root / 'training_audit.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(getattr(self, 'residual_stage_audit', []),
                     columns=['base', 'cutoff', 'length', 'reused']).to_csv(
                         self.root / 'residual_stage_audit.csv', index=False, encoding='utf-8-sig')
        return result


def prepare_group(namespace):
    engine = GroupTraining(namespace)
    if namespace.get('VERBOSE', False):
        print(f'{VERSION}: обучение {engine.units}; оценка: {engine.target}.')
        print(f'Проверяемых: {len(engine.targets)}; условий: {len(engine.conditions)}; состояний: {len(engine.states)}.')
        print('Аудит датчиков:', engine.root / 'sensor_audit.csv')
    return engine
