"""Пакетный запуск старых моделей и отчёт внутри ноутбука, без установки пакета."""
from __future__ import annotations

import base64
import copy
import gc
from functools import partial
from html import escape
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from legacy_defaults import DEFAULTS
import legacy_metrics as metrics

VERSION = 'legacy-fleet-2.2'
MAIN = ['analogs', 'AAKR', 'MSET']
OTHER = ['Регрессия', 'Близнецы', 'AAKR (все датчики)', 'MSET (все датчики)',
         'Остатки analogs', 'Остатки AAKR', 'Остатки MSET', 'Остатки регрессии']
SCENARIO_NAMES = {'local_common': 'Индивидуальная · общие датчики',
                  'local_full': 'Индивидуальная · свои датчики', 'pooled': 'Общая модель',
                  'transfer': 'Перенос с другого котла'}


def defaults():
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(GROUPS={'БКЗ-500': ['К-5', 'К-6', 'К-7']}, SELECT_GROUPS=None,
               SCENARIOS=['local', 'pooled', 'transfer'], LOCAL_SENSOR_SETS=['common', 'full'],
               TRANSFER_DONORS=None, MAIN_METHODS=MAIN.copy(), OTHER_METHODS=[],
               MIN_LEAD_H=3, MERGE_GAP_H=12, ALARM_MATCH='overlap',
               THRESHOLD_MULTIPLIERS=[.5, .75, 1., 1.25, 1.5, 2.], THRESHOLD_VALIDATION_END=None,
               PROXIMITY_HALF_LIVES_D=[1, 3, 7, 14], REUSE_PREVIOUS_MODEL=True,
               CONTINUE_ON_ERROR=True, EVAL_END=None, MAX_PLOTS_PER_METHOD=3,
               EXPORT=True, RECOMPUTE=False, TRAIN_UPDATE='fast', FAST_MIN_CAL_H=72,
               IMPORT_LEGACY_CACHE=False, ALIGN_LOAD_ROLE=True, HISTORY_MISSING='previous',
               RESUME_EXPERIMENTS=True, SHOW_PROGRESS_RESULTS=False, VERBOSE=False,
               CLEAN_WINDOW_H=168, CLEAN_MIN_HISTORY_H=24, CLEAN_SPIKE_FACTOR=8., CLEAN_FROZEN_H=24,
               LEGACY_OUTPUT_DIR='anomaly_results_legacy', LEGACY_MODELS_DIR='models_legacy')
    return cfg


def configuration(namespace):
    cfg = defaults()
    cfg.update({k: v for k, v in namespace.items() if k in cfg})
    path = os.environ.get('LEGACY_CONFIG_JSON') or os.environ.get('BOILER_CONFIG_JSON')
    if path:
        overrides = json.loads(Path(path).read_text(encoding='utf-8'))
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError(f'Неизвестные параметры: {sorted(unknown)}')
        cfg.update(overrides)
    if cfg['TRAIN_UPDATE'] not in ('fast', 'refit'):
        raise ValueError('TRAIN_UPDATE: fast (обновление калибровки и одна модель) или refit (прежние три OOF-блока).')
    if cfg['HISTORY_MISSING'] not in ('previous', 'skip'):
        raise ValueError('HISTORY_MISSING: previous или skip; экспорт не обучает пропущенные месяцы.')
    for name in ('CLEAN_WINDOW_H', 'CLEAN_MIN_HISTORY_H', 'CLEAN_SPIKE_FACTOR', 'CLEAN_FROZEN_H'):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f'{name}: нужно положительное конечное число.')
    if cfg['CLEAN_MIN_HISTORY_H'] > cfg['CLEAN_WINDOW_H']:
        raise ValueError('CLEAN_MIN_HISTORY_H не может быть больше CLEAN_WINDOW_H.')
    return cfg


def _safe(value):
    value = str(value)
    if not value.strip() or value in ('.', '..') or re.search(r'[<>:"/\\|?*]', value) or value.endswith((' ', '.')):
        raise ValueError(f'Недопустимое имя папки: {value!r}')
    return value


def plan(cfg):
    groups = cfg['GROUPS']
    if not isinstance(groups, dict) or not groups:
        raise ValueError('Задайте GROUPS: название группы -> список котлов. UNIT не нужен.')
    selected = list(groups) if cfg['SELECT_GROUPS'] is None else cfg['SELECT_GROUPS']
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(groups):
        raise ValueError('SELECT_GROUPS должен содержать существующие неповторяющиеся группы.')
    scenarios = cfg['SCENARIOS']
    if not scenarios or set(scenarios) - {'local', 'pooled', 'transfer'}:
        raise ValueError('SCENARIOS: local, pooled, transfer.')
    if not cfg['LOCAL_SENSOR_SETS'] or set(cfg['LOCAL_SENSOR_SETS']) - {'common', 'full'}:
        raise ValueError('LOCAL_SENSOR_SETS: common и/или full.')
    if not set(cfg['MAIN_METHODS']) <= set(MAIN) or not set(cfg['OTHER_METHODS']) <= set(OTHER):
        raise ValueError(f'MAIN_METHODS: {MAIN}; OTHER_METHODS: {OTHER}')
    if not cfg['MAIN_METHODS'] and not cfg['OTHER_METHODS']:
        raise ValueError('Выберите хотя бы один метод.')
    for name in ('THRESHOLD_MULTIPLIERS', 'PROXIMITY_HALF_LIVES_D', 'WINDOWS_D'):
        if not cfg[name] or any(not np.isfinite(v) or v <= 0 for v in cfg[name]):
            raise ValueError(f'{name}: нужны положительные конечные числа.')
    rows = []
    for group in selected:
        _safe(group)
        units = groups[group]
        if not units or len(units) != len(set(units)):
            raise ValueError(f'{group}: нужны неповторяющиеся котлы.')
        for unit in units:
            _safe(unit)
        for scenario in dict.fromkeys(scenarios):
            if scenario in ('pooled', 'transfer') and len(units) == 1:
                rows.append(dict(group=group, scenario=scenario, target=units[0], donors='',
                                 sensor_set='common', status='неприменимо: в группе один котёл', key=''))
                continue
            for target in units:
                if scenario == 'local':
                    sets = list(dict.fromkeys(cfg['LOCAL_SENSOR_SETS'])) if len(units) > 1 else ['full']
                    options = [(f'local_{s}', [target], s) for s in sets]
                elif scenario == 'pooled':
                    options = [('pooled', units, 'common')]
                else:
                    donors = cfg['TRANSFER_DONORS']
                    allowed = (donors.get(group, units) if isinstance(donors, dict) else donors) or units
                    if set(allowed) - set(units):
                        raise ValueError(f'{group}: TRANSFER_DONORS содержит котлы вне группы.')
                    options = [('transfer', [u], 'common') for u in allowed if u != target]
                for kind, sources, sensors in options:
                    key = _safe(kind + '__' + '_'.join(sources) + '__v22__' + cfg['TRAIN_UPDATE'])
                    rows.append(dict(group=group, scenario=kind, target=target, donors=', '.join(sources),
                                     sensor_set=sensors, status='запланировано', key=key))
    return pd.DataFrame(rows)


def core_sections():
    path = Path(__file__).with_name('legacy_core.py')
    parts = re.split(r'^# %% (\w+)\s*$', path.read_text(encoding='utf-8'), flags=re.M)
    return path, dict(zip(parts[1::2], parts[2::2]))


def preflight(cfg):
    jobs = plan(cfg)
    roles = Path(cfg['SENSOR_LIST_FILE'])
    if not roles.is_file():
        raise FileNotFoundError(f'Нет таблицы ролей: {roles.resolve()}')
    import legacy_group as LG
    rows = []
    for group in jobs.group.unique():
        if not ((jobs.group == group) & (jobs.sensor_set == 'common') & jobs.status.eq('запланировано')).any():
            continue
        role_rows, aligned = LG.align_load_roles({u: LG.read_roles(roles, u) for u in cfg['GROUPS'][group]}, cfg['ALIGN_LOAD_ROLE'])
        common = set.intersection(*(set(r['key'] for r in rr if r['role'] != 'exclude') for rr in role_rows.values()))
        targets = [k for k in common if all(any(r['key'] == k and r['role'] == 'target' for r in rr) for rr in role_rows.values())]
        if len(cfg['GROUPS'][group]) > 1 and len(targets) < 2:
            raise ValueError(f'{group}: общих проверяемых датчиков по Excel {len(targets)}; проверьте общие параметры.')
        print(f'{group}: общих проверяемых по Excel {len(targets)}; сопоставление нагрузки по роли: {aligned}.')
    units = dict.fromkeys(u for group in jobs.group.unique() for u in cfg['GROUPS'][group])
    for unit in units:
        instant = cfg['DATA_KIND'] == 'instant'
        paths = cfg['INSTANT_PATHS'] if instant else cfg['DATA_PATHS']
        path = Path(paths.get(unit) or Path(cfg['DATA_ROOT']) / unit / ('instant.parquet' if instant else 'hourly.parquet'))
        if not path.exists():
            raise FileNotFoundError(f'{unit}: нет данных {path.resolve()}')
        selected = LG.read_roles(roles, unit)
        if not any(r['role'] == 'load' for r in selected):
            raise ValueError(f'{unit}: в Excel не назначена нагрузка.')
        rows.append(dict(котёл=unit, источник=str(path.resolve()), строк_Excel=len(selected)))
    return pd.DataFrame(rows)


def _views(name):
    return [dict(name=f'{name}: один датчик вне нормы', top=1),
            dict(name=f'{name}: несколько датчиков', top=5),
            dict(name=f'{name}: медленный сдвиг (2 сут)', top=1, slow=48)]


def _prepare(cfg, job):
    path, sections = core_sections()
    ns = dict(cfg, __name__='legacy_job_runtime', __file__=str(Path(__file__).resolve().parents[1] / '02_Обучение.ipynb'),
              UNIT=job.target, GROUP_NAME=job.group, TRAIN_UNITS=job.donors.split(', '),
              EXPERIMENT_KEY=job.key, LEGACY_CORE_FILE=str(path), RUN_OPTIONAL_SECTIONS=False,
              RUN_ONLY=cfg['MAIN_METHODS'] + cfg['OTHER_METHODS'])
    # "full" has its own schema. Common/local and transfer share the whole group schema.
    if job.sensor_set == 'full':
        ns['GROUPS'] = {job.group: [job.target]}
    requested_residual = any(n.startswith('Остатки ') for n in cfg['OTHER_METHODS'])
    aliases = {'Остатки analogs': 'analogs', 'Остатки AAKR': 'AAKR',
               'Остатки MSET': 'MSET', 'Остатки регрессии': 'regression'}
    ns['RESIDUAL_BASES'] = list(dict.fromkeys(list(cfg['RESIDUAL_BASES']) + [aliases[n] for n in cfg['OTHER_METHODS'] if n in aliases]))
    ns['USE_NEURAL'] = cfg['USE_NEURAL'] and requested_residual
    if not requested_residual:
        ns['RESIDUAL_METHODS'] = []
    exec(compile(sections['setup'], str(path) + ':setup', 'exec'), ns)
    ns['GROUP_ENGINE'] = ns['LG'].prepare_group(ns)
    if cfg['EVAL_END']:
        # Keep complete source history for training; cap the evaluated months and output below.
        ns['EVAL_END'] = pd.Timestamp(cfg['EVAL_END'])
    if ns['FIRST_TEST_AUTO']:
        ns['FIRST_TEST'] = ns['eda'].get('first_test') or ns['K'].first_test_month(ns['t_ix'], ns['normal'], ns['MIN_NORMAL_H'] * ns['PER_H'])
    if not ns['FIRST_TEST']:
        raise ValueError('Не определён FIRST_TEST.')
    exec(compile(sections['training'], str(path) + ':training', 'exec'), ns)
    ns['SIG'] = dict(ns['SIG'], fleet_version=VERSION, evaluation_end=str(cfg['EVAL_END']),
                     score_code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     export_code=hashlib.sha256(Path(__file__).with_name('k2_deploy.py').read_bytes()).hexdigest())
    ns['SIG']['training_update'] = cfg['TRAIN_UPDATE']
    if cfg['EVAL_END']:
        ns['months'] = ns['months'][ns['months'] <= ns['EVAL_END']]
    # The old display/statistical-rank code is not executed.
    ns['show'] = lambda name: None
    ns['MET'] = {}
    ns['QUALITY_OF'] = lambda name: None
    exec(compile(sections['residual'], str(path) + ':residual', 'exec'), ns)
    _install_score_cache(ns)
    return ns


def _install_score_cache(ns):
    """Read exact release-2.0 results when their full original signature matches."""
    def cached(name, compute, sig=None):
        sig = sig or ns['SIG']
        filename = name.replace(' ', '_') + '.joblib'
        current = Path(ns['OUT']) / filename
        if not ns['RECOMPUTE']:
            if current.exists():
                saved = joblib.load(current)
                if saved.get('sig') == sig:
                    ns['_LAST_SCORE_SOURCE'] = 'кэш'
                    if ns.get('VERBOSE'):
                        print(f'{name}: готовые оценки из кэша.', flush=True)
                    return saved['views']
            engine = ns['GROUP_ENGINE']
            if ns['IMPORT_LEGACY_CACHE'] and engine.compat_signature:
                manifest = json.loads(Path(__file__).with_name('cache_compat_v2.json').read_text(encoding='utf-8'))
                old_sig = dict(sig, group_signature=engine.compat_signature,
                               fleet_version=manifest['fleet_version'], score_code=manifest['fleet_hash'])
                old_sig.pop('training_update', None)
                old_key = ns['EXPERIMENT_KEY'].rsplit('__', 1)[0]
                previous = Path(ns['LEGACY_OUTPUT_DIR']) / engine.name / old_key / ns['UNIT'] / ns['KIND_TAG'] / filename
                if previous.exists():
                    saved = joblib.load(previous)
                    if saved.get('sig') == old_sig:
                        print(f'{name}: сохранён результат legacy 2.0; повторного расчёта нет.', flush=True)
                        return {v: dict(r, score_origin='legacy2.0_cache') for v, r in saved['views'].items()}
        started = time.perf_counter()
        out = compute()
        current.parent.mkdir(parents=True, exist_ok=True)
        temporary = current.with_suffix('.tmp')
        joblib.dump(dict(sig=sig, views=out), temporary, compress=3)
        os.replace(temporary, current)
        ns['_LAST_SCORE_SOURCE'] = 'расчёт'
        if ns.get('VERBOSE'):
            print(f'{name}: {(time.perf_counter() - started) / 60:.1f} мин', flush=True)
        return out
    ns['cached'] = cached


def _online(ns, name, make, views):
    def compute():
        out = ns['new_out'](views)
        slow = max(v.get('slow', 0) for v in views)
        ix = ns['t_ix']
        model_month = pd.Series(pd.NaT, index=ix, dtype='datetime64[ns]')
        reason = pd.Series('до начала проверки', index=ix, dtype=object)
        online = ix >= pd.Timestamp(ns['FIRST_TEST'])
        reason.loc[online] = 'нет модели'
        reason.loc[online & ~ns['running'].to_numpy()] = 'котёл остановлен'
        reason.loc[online & ns['mode'].eq(ns['M_START']).to_numpy()] = 'пуск'
        reason.loc[online & ns['test_h']] = 'испытания / ЗРР'
        reason.loc[online & ns['scorable'].to_numpy() & ~ns['enough'].to_numpy()] = 'недостаточно датчиков'
        for m0 in ns['months']:
            started = time.perf_counter()
            in_m = (ix >= m0) & (ix < m0 + pd.offsets.MonthBegin(1))
            te = in_m & ns['scorable'].to_numpy() & ns['enough'].to_numpy()
            if ns.get('EVAL_END') is not None:
                te &= ix <= pd.Timestamp(ns['EVAL_END'])
            if not te.any():
                continue
            tm = ns['train_one'](make, m0, views)
            if tm is None:
                continue
            rows = te | ((ix >= m0 - pd.Timedelta(hours=slow)) & (ix < m0) & ns['scorable'].to_numpy() & ns['enough'].to_numpy())
            error, error_ix = ns['model_errors'](tm['det'], tm['prep'], rows)
            if error is None:
                continue
            ns['fill'](out, tm, error, error_ix, m0, in_m, views, getattr(tm['det'], 'last_contrib', None))
            reason.loc[te] = 'разогрев окна / нет конечного результата'
            model_month.loc[te] = pd.Timestamp(tm['trained_cutoff'])
            if ns.get('VERBOSE'):
                print(f'    {name} · {m0:%Y-%m}: {int(te.sum()) / ns["PER_H"]:.0f} ч, '
                      f'модель {tm["trained_cutoff"]}, {tm.get("update_mode", "cache")}, '
                      f'{time.perf_counter() - started:.1f} с', flush=True)
        for v in views:
            item = out[v['name']]
            item['model_cutoff'] = model_month.copy()
            item['reason'] = reason.copy()
            item['reason'].loc[item['score'].notna()] = 'рассчитано'
        return out
    return ns['cached'](name, compute, ns['sig_of'](make()))


def _run_methods(ns):
    ns['run_online'] = lambda name, make, views: _online(ns, name, make, views)
    # Metrics/export metadata use the same understandable evaluation as the report.
    def show(name):
        ns['MET'][name] = metrics.evaluate(ns['RES'][name], ns)[0]
    ns['show'] = show
    ns['QUALITY_OF'] = lambda name: ns['MET'].get(name)
    # Schema and donor set define one final cutoff, independent of the target report.
    ns['M_NEXT'] = max(c['df'].index[-1] for c in ns['GROUP_ENGINE'].contexts.values()) + ns['STEP']
    if 'analogs' in ns['RUN_ONLY']:
        ns['run_and_show']('analogs', lambda: ns['K'].AnalogCondDet(ns['COND_COLS']), _views('analogs'))
    if 'Регрессия' in ns['RUN_ONLY']:
        ns['run_and_show']('Регрессия', ns['regression'], _views('Регрессия'))
    for name, kind, classic in [('AAKR', 'aakr', False), ('MSET', 'mset', False),
                                ('AAKR (все датчики)', 'aakr', True), ('MSET (все датчики)', 'mset', True)]:
        if name in ns['RUN_ONLY']:
            ns['run_and_show'](name, lambda kind=kind, classic=classic: ns['KernelDet'](ns['COND_COLS'], kind, classic=classic), _views(name))
    if 'Близнецы' in ns['RUN_ONLY']:
        ns['run_and_show']('Близнецы', lambda: ns['K'].TwinsDet(), [dict(name='Близнецы: расхождение дублей', top=1)])
    for kind in ns['SELECTED_RESIDUAL_BASES']:
        if ns['STAGES'][kind]['title'] in ns['RUN_ONLY'] and ns['SECOND']:
            ns['run_two_stage'](kind)
    return ns['RES']


def _steps(per_h, hours):
    return max(1, int(round(hours * per_h)))


def _report_context(ns):
    names = ['t_ix', 'STEP', 'PER_H', 'FIRST_TEST', 'scorable', 'running', 'mode', 'state',
             'fails', 'events', 'SMOOTH_H', 'PERSIST_H', 'MERGE_GAP_H', 'MIN_LEAD_H',
             'ALARM_MATCH', 'PRE_WINDOW_D', 'NEUTRAL_D', 'WINDOWS_D', 'THRESHOLD_MULTIPLIERS',
             'THRESHOLD_VALIDATION_END', 'PROXIMITY_HALF_LIVES_D']
    context = {k: ns[k] for k in names}
    per_h = ns['PER_H']
    context['hs'] = partial(_steps, per_h)
    if ns.get('EVAL_END'):
        context['EVAL_END'] = ns['EVAL_END']
        context['scorable'] = context['scorable'] & (context['t_ix'] <= pd.Timestamp(ns['EVAL_END']))
    return context


def _contributions(ns, views):
    rows = []
    ix = ns['t_ix']
    distance = ns['K'].days_to_next(ix, ns['fails'].stop).to_numpy()
    bg = ((distance > ns['PRE_WINDOW_D'] + ns['NEUTRAL_D']) & ns['scorable'].to_numpy() &
          (ix >= pd.Timestamp(ns['FIRST_TEST'])))
    for view, res in views.items():
        matrix = res.get('contrib')
        if matrix is None:
            continue
        cols = res.get('cols', ns['sens'])
        frame = pd.DataFrame(np.asarray(matrix, dtype='float32'), index=ix, columns=cols)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        usual = frame.loc[bg].quantile(.9) if bg.any() else pd.Series(np.nan, index=cols)
        for event, window in metrics.event_windows(ns, ns['PRE_WINDOW_D']):
            selected = window & ns['scorable'].to_numpy()
            if not selected.any():
                continue
            top = frame.loc[selected].quantile(.9).dropna().nlargest(10)
            for sensor, value in top.items():
                rows.append(dict(view=view, останов=event.stop, датчик=sensor,
                                 отклонение=float(value), обычный_фон=float(usual[sensor])))
    return pd.DataFrame(rows)


def run(cfg):
    jobs = plan(cfg)
    output = Path(cfg['LEGACY_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    result = dict(config=copy.deepcopy(cfg), plan=jobs, records=[], errors=[], status=[])
    if cfg.get('IMPORT_LEGACY_CACHE'):
        print('Очистка изменена: кэш 2.0/2.1 не используется. Новый расчёт — в папках __v22__.', flush=True)
    run_cfg = dict(cfg, _RUN_GENERATED_MODELS=set(), _RUN_GENERATED_RESIDUALS=set())
    runnable = jobs[jobs.status.eq('запланировано')]
    for i, job in enumerate(runnable.itertuples(index=False), 1):
        started = time.time()
        print(f'\n[{i}/{len(runnable)}] {job.group} · {SCENARIO_NAMES[job.scenario]} · {job.target}; обучение: {job.donors}', flush=True)
        meta = {k: getattr(job, k) for k in ['group', 'scenario', 'target', 'donors', 'sensor_set', 'key']}
        checkpoint = output / job.group / job.key / job.target / 'report.joblib'
        checkpoint_key = _checkpoint_key(cfg, job.group)
        try:
            if cfg['RESUME_EXPERIMENTS'] and not cfg['RECOMPUTE'] and checkpoint.exists():
                record = joblib.load(checkpoint)
                exports_available = not cfg['EXPORT'] or all((Path(p) / 'model.py').exists() for p in record['exports'])
                if record.get('checkpoint_key') == checkpoint_key and exports_available:
                    result['records'].append(record)
                    result['status'].append(dict(**meta, status='из сохранённого отчёта', seconds=round(time.time() - started, 2)))
                    _save_summary(result)
                    if cfg['SHOW_PROGRESS_RESULTS']:
                        _show_progress(record)
                    continue
            ns = _prepare(run_cfg, job)
            views = _run_methods(ns)
            if not views:
                raise ValueError('Не получено ни одного вида оценки; проверьте выбор методов.')
            context = _report_context(ns)
            audit = ns['GROUP_ENGINE'].summary()
            fit_audit = pd.DataFrame(ns['GROUP_ENGINE'].fit_audit)
            contributions = _contributions(ns, views)
            compact_views = {name: {k: v for k, v in res.items() if k != 'contrib'} for name, res in views.items()}
            record = dict(meta=meta, context=context, views=compact_views, training=audit, fits=fit_audit,
                          checkpoint_key=checkpoint_key,
                          contributions=contributions,
                          sensor_audit=ns['GROUP_ENGINE'].audit,
                          targets=list(ns['GROUP_ENGINE'].targets), conditions=list(ns['GROUP_ENGINE'].conditions),
                          states=list(ns['GROUP_ENGINE'].states), exports=list(ns['EXPORTED']))
            # No detector/namespace is retained by the report. Scores also support later threshold sweeps.
            result['records'].append(record)
            usable = any(res['score'].notna().any() for res in views.values())
            result['status'].append(dict(**meta, status='готово' if usable else 'нет результата: недостаточно обучения', seconds=round(time.time() - started, 2)))
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            joblib.dump(record, temporary, compress=3)
            os.replace(temporary, checkpoint)
            if cfg['SHOW_PROGRESS_RESULTS']:
                _show_progress(record)
            if ns.get('HAS_TF'):
                ns['tf'].keras.backend.clear_session()
            del ns
            del views
            gc.collect()
        except Exception as exc:
            error = dict(**meta, error=repr(exc), traceback=traceback.format_exc())
            result['errors'].append(error)
            result['status'].append(dict(**meta, status='ошибка', seconds=round(time.time() - started, 2)))
            print(error['traceback'], flush=True)
            if not cfg['CONTINUE_ON_ERROR']:
                raise
        _save_summary(result)
    result['tables'] = tables(result)
    # One resumable report artifact; the primary result is displayed in the notebook.
    joblib.dump(result, output / 'fleet_report.joblib', compress=3)
    print(f'\nЗавершено: {len(result["records"])} экспериментов; ошибок: {len(result["errors"])}. '
          'Таблицы и графики — в следующих ячейках.', flush=True)
    return result


def _checkpoint_key(cfg, group):
    settings = {k: v for k, v in cfg.items() if k not in ('RECOMPUTE', 'SHOW_PROGRESS_RESULTS', 'RESUME_EXPERIMENTS')}
    files = []
    for unit in cfg['GROUPS'][group]:
        instant = cfg['DATA_KIND'] == 'instant'
        paths = cfg['INSTANT_PATHS'] if instant else cfg['DATA_PATHS']
        path = Path(paths.get(unit) or Path(cfg['DATA_ROOT']) / unit / ('instant.parquet' if instant else 'hourly.parquet'))
        sources = sorted(path.rglob('*')) if path.is_dir() else [path]
        files.extend((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in sources if p.is_file())
    payload = dict(settings=settings, files=files,
                   excel=hashlib.sha256(Path(cfg['SENSOR_LIST_FILE']).read_bytes()).hexdigest(),
                   code={p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                         for p in ('legacy_fleet.py', 'legacy_group.py', 'legacy_core.py', 'legacy_metrics.py')})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def _show_progress(record):
    rows = []
    for view, res in record['views'].items():
        row = metrics.evaluate(res, record['context'])[0]
        rows.append(dict(view=view, **{k: row[k] for k in ('PR AUC', 'PR AUC случайный уровень', 'поймано аварий', 'precision', 'recall', 'F1', '% фона в тревоге', 'полезных эпизодов', 'фоновых эпизодов', 'нейтральных эпизодов', 'часов фона', 'рассчитано рабочих часов')}))
    display(HTML('<p><b>Готовый эксперимент:</b> ' + escape(f"{record['meta']['group']} · {record['meta']['target']} · {SCENARIO_NAMES[record['meta']['scenario']]}") + '</p>' + _table(pd.DataFrame(rows))))
    first = next(iter(record['views']), None)
    if first:
        display(HTML(_figure(record, first)))


def load_report(cfg):
    """Restore partial results after interruption without training any model."""
    jobs = plan(cfg)
    result = dict(config=copy.deepcopy(cfg), plan=jobs, records=[], errors=[], status=[])
    for job in jobs[jobs.status.eq('запланировано')].itertuples(index=False):
        path = Path(cfg['LEGACY_OUTPUT_DIR']) / job.group / job.key / job.target / 'report.joblib'
        if not path.exists():
            continue
        record = joblib.load(path)
        if record.get('checkpoint_key') == _checkpoint_key(cfg, job.group):
            result['records'].append(record)
            result['status'].append(dict(**record['meta'], status='восстановлено'))
    result['tables'] = tables(result)
    return result


def _save_summary(result):
    output = Path(result['config']['LEGACY_OUTPUT_DIR'])
    status = {'version': VERSION, 'experiments': result['status'], 'errors': result['errors']}
    path = output / 'run_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def tables(result):
    main_rows, threshold_rows, proximity_rows, event_rows, window_rows, hour_rows = [], [], [], [], [], []
    for record in result['records']:
        ns, meta = record['context'], record['meta']
        for view, res in record['views'].items():
            method = view.split(':', 1)[0]
            category = 'основные' if method in MAIN else 'прочие'
            key = dict(**meta, method=method, category=category, view=view)
            key['источник оценок'] = res.get('score_origin', result['config']['TRAIN_UPDATE'])
            basic, events, (_, _, _, valid) = metrics.evaluate(res, ns)
            main_rows.append(dict(**key, **basic))
            event_rows.extend(dict(**key, **r) for r in events.to_dict('records'))
            threshold_rows.extend(dict(**key, **r) for r in metrics.threshold_table(res, ns).to_dict('records'))
            proximity_rows.extend(dict(**key, **r) for r in metrics.proximity_table(res, ns).to_dict('records'))
            for days in ns['WINDOWS_D']:
                window_rows.append(dict(**key, **{'окно, сут': days}, **metrics.evaluate(res, ns, days=days)[0]))
            reason = res.get('reason')
            if reason is None:
                reason = pd.Series(np.where(valid, 'рассчитано', 'нет результата'), index=ns['t_ix'])
            for label, count in reason.loc[ns['t_ix'] >= pd.Timestamp(ns['FIRST_TEST'])].value_counts().items():
                hour_rows.append(dict(**key, причина=label, часов=count / ns['PER_H']))
    return {name: pd.DataFrame(rows) for name, rows in [('main', main_rows), ('thresholds', threshold_rows),
            ('proximity', proximity_rows), ('events', event_rows), ('windows', window_rows), ('hours', hour_rows)]}


def paired_comparison(result):
    rows = []
    records = result['records']
    for a in records:
        if a['meta']['scenario'] != 'local_common':
            continue
        for b in records:
            if b['meta']['group'] != a['meta']['group'] or b['meta']['target'] != a['meta']['target'] or b['meta']['scenario'] not in ('pooled', 'transfer'):
                continue
            for view in sorted(set(a['views']) & set(b['views'])):
                ia, ib = a['context']['t_ix'], b['context']['t_ix']
                va = metrics.alarm_series(a['views'][view], a['context'])[3]
                vb = metrics.alarm_series(b['views'][view], b['context'])[3]
                common = ia[va].intersection(ib[vb])
                for record in (a, b):
                    ns = record['context']
                    values = metrics.evaluate(record['views'][view], ns, measurement_mask=ns['t_ix'].isin(common))[0]
                    rows.append(dict(group=a['meta']['group'], target=a['meta']['target'], view=view,
                                     сравнение=b['meta']['key'], scenario=record['meta']['scenario'],
                                     donors=record['meta']['donors'], **values))
    return pd.DataFrame(rows)


def _table(frame):
    if frame.empty:
        return '<p>Нет строк для этого раздела.</p>'
    return frame.to_html(index=False, border=0, escape=True, na_rep='—', float_format=lambda x: f'{x:.3f}',
                         formatters={'% фона в тревоге': lambda x: f'{100*x:.1f}' if np.isfinite(x) else '—'})


def _model_usage(record):
    rows = []
    ns = record['context']
    for view, res in record['views'].items():
        cutoffs = res.get('model_cutoff')
        if cutoffs is None:
            continue
        valid = metrics.alarm_series(res, ns)[3]
        usage = pd.DataFrame({'месяц проверки': ns['t_ix'].to_period('M').astype(str),
                              'модель на дату': cutoffs.to_numpy()}, index=ns['t_ix']).loc[valid]
        for (month, cutoff), part in usage.dropna().groupby(['месяц проверки', 'модель на дату']):
            rows.append({'view': view, 'месяц проверки': month, 'модель на дату': cutoff,
                         'предыдущая модель': pd.Timestamp(cutoff) < pd.Timestamp(month),
                         'часов применения': len(part) / ns['PER_H']})
    return pd.DataFrame(rows)


def _figure(record, view, start=None, end=None):
    from legacy_plots import render_png
    data = render_png(record, view, start, end)
    return '<img alt="' + escape(view, quote=True) + '" src="data:image/png;base64,' + base64.b64encode(data).decode() + '" style="max-width:100%">'


def show_period(result, *, group, target, view, start=None, end=None, scenario=None, donors=None):
    """Display a selected period without training or rebuilding alarms after cropping."""
    records = [r for r in result['records'] if r['meta']['group'] == group and r['meta']['target'] == target
               and view in r['views'] and (scenario is None or r['meta']['scenario'] == scenario)
               and (donors is None or r['meta']['donors'] == donors)]
    if len(records) != 1:
        raise ValueError(f'Найдено экспериментов: {len(records)}. Уточните scenario и donors по RESULT["plan"].')
    display(HTML(_figure(records[0], view, start, end)))


def show_overview(result):
    display(HTML('<h3>Выполненные сценарии</h3>' + _table(pd.DataFrame(result['status']))))
    skipped = result['plan'][~result['plan'].status.eq('запланировано')]
    if len(skipped):
        display(HTML('<p>Неприменимые сценарии:</p>' + _table(skipped[['group', 'scenario', 'target', 'status']])))
    table = result['tables']['main']
    if table.empty:
        display(HTML('<p>Результатов нет. Ошибки перечислены выше.</p>'))
        return
    columns = ['group', 'scenario', 'target', 'donors', 'view', 'источник оценок', 'PR AUC', 'PR AUC случайный уровень', 'поймано аварий', 'precision', 'recall', 'F1',
               'тревог в фоне в месяц', '% фона в тревоге', 'упреждение, ч', 'полезных эпизодов', 'фоновых эпизодов', 'нейтральных эпизодов', 'часов фона', 'рассчитано рабочих часов']
    for category in ['основные', 'прочие']:
        display(HTML(f'<h3>{category.capitalize()} методы</h3>' + _table(table[table.category.eq(category)][columns])))


def show_details(result):
    table = result['tables']['main']
    fragments = ['<div id="legacy-top"><h3>Навигация по результатам</h3>']
    for group in dict.fromkeys(r['meta']['group'] for r in result['records']):
        anchor = 'group-' + hashlib.sha256(group.encode()).hexdigest()[:10]
        fragments.append(f'<a href="#{anchor}">{escape(group)}</a> &nbsp; ')
    for group in dict.fromkeys(r['meta']['group'] for r in result['records']):
        anchor = 'group-' + hashlib.sha256(group.encode()).hexdigest()[:10]
        fragments.append(f'<h3 id="{anchor}">{escape(group)}</h3>')
        group_records = [r for r in result['records'] if r['meta']['group'] == group]
        for scenario in dict.fromkeys(r['meta']['scenario'] for r in group_records):
            fragments.append(f'<details open><summary><b>{escape(SCENARIO_NAMES[scenario])}</b></summary>')
            for record in group_records:
                meta = record['meta']
                if meta['scenario'] != scenario:
                    continue
                fragments.append(f'<details><summary>{escape(meta["target"])} · обучение: {escape(meta["donors"])}</summary>')
                fragments.append(f'<p>Проверяемых датчиков: {len(record["targets"])}; условий: {len(record["conditions"])}; состояний: {len(record["states"])}.</p>')
                selected = table[table.group.eq(group) & table.scenario.eq(scenario) & table.target.eq(meta['target']) & table.donors.eq(meta['donors'])]
                columns = ['view', 'PR AUC', 'PR AUC случайный уровень', 'поймано аварий', 'precision', 'recall', 'F1', '% фона в тревоге', 'тревог в фоне в месяц', 'упреждение, ч', 'полезных эпизодов', 'фоновых эпизодов', 'нейтральных эпизодов', 'часов фона', 'рассчитано рабочих часов']
                for category in ['основные', 'прочие']:
                    part = selected[selected.category.eq(category)]
                    if part.empty:
                        continue
                    fragments.append(f'<details open><summary>{category.capitalize()} методы</summary>' + _table(part[columns]))
                    for method in part.method.unique():
                        fragments.append(f'<details><summary>Графики: {escape(method)}</summary>')
                        for view in part[part.method.eq(method)].view.head(result['config']['MAX_PLOTS_PER_METHOD']):
                            fragments.append(_figure(record, view))
                        fragments.append('</details>')
                    fragments.append('</details>')
                events = result['tables']['events']
                if len(events):
                    events = events[events.group.eq(group) & events.scenario.eq(scenario) & events.target.eq(meta['target']) & events.donors.eq(meta['donors'])]
                    fragments.append('<details><summary>Каждый останов</summary>' + _table(events.drop(columns=['key', 'sensor_set', 'category', 'method'], errors='ignore')) + '</details>')
                fragments.append('<details><summary>Какая модель применялась в каждом месяце</summary>' + _table(_model_usage(record)) + '</details>')
                fragments.append('<details><summary>Обучение / кэш / предыдущая модель</summary>' + _table(record['fits']) + _table(record['training']) + '</details>')
                if len(record.get('contributions', [])):
                    fragments.append('<details><summary>Датчики перед каждым остановом (90-й процентиль отклонения, × обычной ошибки)</summary>' + _table(record['contributions']) + '</details>')
                fragments.append('</details>')
            fragments.append('</details>')
        fragments.append('<a href="#legacy-top">К навигации ↑</a>')
    fragments.append('</div>')
    display(HTML(''.join(fragments)))


def show_table(result, name):
    frame = result['tables'][name]
    if frame.empty:
        display(HTML('<p>Нет результатов.</p>')); return
    # Group/scenario/boiler navigation remains available without widgets, network or a running kernel.
    fragments = []
    for (group, scenario, unit), part in frame.groupby(['group', 'scenario', 'target'], sort=False):
        fragments.append(f'<details><summary>{escape(group)} · {escape(SCENARIO_NAMES[scenario])} · {escape(unit)}</summary>')
        fragments.append(_table(part.drop(columns=['key', 'sensor_set', 'category', 'group', 'scenario', 'target'], errors='ignore')))
        fragments.append('</details>')
    display(HTML(''.join(fragments)))
