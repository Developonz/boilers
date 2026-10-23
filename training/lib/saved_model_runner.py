"""Apply frozen exports to another boiler. No training or calibration calls."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import re
import sys
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def discover_models(root):
    records = []
    root = Path(root)
    if root.is_dir():
        for path in sorted(root.rglob('config.json')):
            if any(p in {'_archive', 'stage1', 'members'} for p in path.relative_to(root).parts):
                continue
            if not path.with_name('model.py').is_file():
                continue
            cfg = json.loads(path.read_text(encoding='utf-8'))
            records.append(dict(folder=str(path.parent.resolve()), method=cfg.get('method', cfg.get('title')),
                                trained_until=cfg.get('trained_until'), core=cfg.get('core', 'actual')))
    return pd.DataFrame(records, columns=['folder', 'method', 'trained_until', 'core'])


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _runtime(folder, flavor):
    """Use the export's own code and restore the caller's module namespace."""
    names = ['kernel_models', 'boiler_monitoring'] if flavor == 'actual' else []
    tag = '_saved_boiler_' + uuid.uuid4().hex
    names.append(tag)
    previous = {name: sys.modules.get(name) for name in names}
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if flavor == 'actual':
            for name in names[:-1]:
                path = folder / (name + '.py')
                if not path.is_file():
                    raise FileNotFoundError(f'В экспорте нет {path.name}; нужна полная папка модели.')
                _module(name, path)
        loader = _module(tag, folder / 'model.py')
        model = loader.load(str(folder))
        yield model, sys.modules.get('boiler_monitoring') if flavor == 'actual' else loader
    finally:
        sys.dont_write_bytecode = old_flag
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _legacy_nodes(model):
    yield model
    if hasattr(model, 'stage1'):
        yield from _legacy_nodes(model.stage1)
    for member in getattr(model, 'members', []):
        yield from _legacy_nodes(member[0] if isinstance(member, tuple) else member)


def _requirements(model, flavor):
    if flavor == 'actual':
        return dict(model.roles)
    roles = {}
    for node in _legacy_nodes(model):
        cfg = node.cfg
        scaler = getattr(node, 'scaler', None)
        for col in getattr(scaler, 'num', []):
            if not col.startswith('условие: '):
                roles.setdefault(col, 'target')
        for col, _ in getattr(scaler, 'bools', []):
            roles[col] = 'state'
        spec = cfg.get('cond_spec') or {}
        for col in (spec.get('extra') or {}).values():
            roles[col] = 'condition'
        for col in (spec.get('mills') or {}):
            roles[col] = 'condition'
        if cfg.get('run_col'):
            roles[cfg['run_col']] = 'load'
        for col in cfg.get('presence_cols', []):
            roles.setdefault(col, 'target')
        for col in getattr(node, 'cols', []) if cfg.get('core') == 'rare' else []:
            roles[col] = 'state'
    return roles


def _group(cfg, sources, unit, groups, group_name):
    known = dict(cfg.get('config', {}).get('groups', {}))
    known.update(groups or {})
    origin = 'parameters_or_saved_config'
    if not known and cfg.get('group') and len(cfg.get('training_units', [])) > 1:
        known[cfg['group']] = cfg['training_units']
        origin = 'legacy_training_units'
    matches = [name for name, members in known.items() if unit in members and set(sources).issubset(members)]
    selected = group_name or (matches[0] if len(matches) == 1 else None)
    if selected not in matches:
        raise ValueError(f'Задайте GROUPS/GROUP_NAME: получатель {unit} и источники {sources} '
                         f'должны входить в одну группу. Подходящие группы: {matches}.')
    if len(set(known[selected])) != len(known[selected]):
        raise ValueError('В группе повторяются котлы.')
    return selected, list(known[selected]), origin


def _read_grid(path, step):
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix.lower() in ('.parquet', '.pq') else pd.read_csv(path, sep=None, engine='python')
    raw.columns = [str(c).strip() for c in raw]
    if not raw.columns.is_unique:
        raise ValueError('Повторяющиеся имена столбцов после удаления пробелов.')
    if 'TimeValue' in raw:
        raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop('TimeValue'), errors='raise'))
    elif not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError('Нужен TimeValue или DatetimeIndex; числовой индекс не считается временем.')
    if raw.index.tz is not None or raw.index.hasnans or raw.index.has_duplicates:
        raise ValueError('Нужно локальное время без timezone, пропусков времени и дублей.')
    raw = raw.sort_index()
    raw.index = raw.index.round(step)
    if raw.index.has_duplicates:
        raise ValueError('Округление времени создало дубли.')
    if raw.empty:
        raise ValueError('Датасет пуст.')
    return raw.asfreq(step)


def _map(raw, required, unit, rows, canonical, aliases):
    """Prefer the recipient's exact Excel name; never select an ambiguous alias."""
    normalized = {}
    for col in raw:
        normalized.setdefault(canonical(col, unit), []).append(col)
    output = pd.DataFrame(index=raw.index)
    audit = []
    used = {}
    for key, role in required.items():
        candidates = [r for r in rows if canonical(r['key'], unit) == canonical(key, unit)]
        if not candidates and role == 'load' and key == 'нагрузка котла':
            candidates = [r for r in rows if r['role'] == 'load' and
                          'расход' in canonical(r['original'], unit) and 'пар' in canonical(r['original'], unit)]
            if len(candidates) != 1:
                raise ValueError(f'{unit}: для общей нагрузки нужна одна роль расхода пара в Excel.')
        exact = list(dict.fromkeys(r['original'] for r in candidates if r['original'] in raw))
        hits = exact
        if not hits:
            for row in candidates:
                hits.extend(normalized.get(canonical(row['original'], unit), []))
            hits = list(dict.fromkeys(hits))
        if not hits:
            hits = list(dict.fromkeys([c for c in raw if aliases.get(canonical(c, unit)) == key] +
                                      normalized.get(canonical(key, unit), [])))
        if len(hits) > 1:
            raise ValueError(f'{unit}: неоднозначный датчик {key}: {hits}. Уточните Excel/названия.')
        source = hits[0] if hits else None
        if source in used:
            raise ValueError(f'Один столбец {source} сопоставлен с {used[source]} и {key}.')
        if source is not None:
            used[source] = key
        values = raw[source] if source is not None else pd.Series(np.nan, index=raw.index)
        if role == 'state':
            values = values.astype(object).where(values.notna(), None)
        else:
            values = pd.to_numeric(values.astype(str).str.replace(',', '.', regex=False), errors='coerce')
            values = values.replace([np.inf, -np.inf], np.nan)
        output[key] = values
        coverage = float(values.notna().mean())
        audit.append(dict(unit=unit, key=key, role=role, source=source,
                          status='missing' if source is None else 'empty' if coverage == 0 else 'matched',
                          coverage=coverage, excel_role=' | '.join(sorted(set(r['role'] for r in candidates)))))
    for col in ('Состояние', '_Тип', '_test'):
        if col in raw:
            output[col] = raw[col]
    if '_Тип' in output and '_test' not in output:
        output['_test'] = output['_Тип'].astype(str).isin(['ИСП', 'ЗРР'])
    return output, pd.DataFrame(audit)


def _period(raw, step, cutoff, start, end, allow_before_cutoff, warmup_h):
    automatic_start = start is None
    start = pd.Timestamp(start) if start is not None else raw.index[0]
    end = pd.Timestamp(end) if end is not None else raw.index[-1] + step
    if start.tz is not None or end.tz is not None or pd.isna(start) or pd.isna(end):
        raise ValueError('Границы периода должны быть корректным локальным временем без timezone.')
    if start != start.round(step) or end != end.round(step):
        raise ValueError(f'Границы периода должны попадать на сетку модели: {step}.')
    if not allow_before_cutoff and start < cutoff:
        if automatic_start:
            start = max(start, cutoff)
        else:
            raise ValueError('Период начинается раньше границы обучения. Для исторического переноса '
                             'разрешите ALLOW_BEFORE_CUTOFF=True; отчёт отметит временное наложение.')
    if start >= end or not ((raw.index >= start) & (raw.index < end)).any():
        raise ValueError('Нет данных в выбранном периоде. Для проверки прошлого выберите месячную '
                         'модель или ALLOW_BEFORE_CUTOFF=True.')
    if warmup_h is not None and (not np.isfinite(warmup_h) or warmup_h < 0):
        raise ValueError('WARMUP_HOURS должен быть >= 0 либо None (вся доступная предыстория).')
    left = raw.index[0] if warmup_h is None else start - pd.Timedelta(hours=warmup_h)
    return raw.loc[(raw.index >= left) & (raw.index < end)], start, end


def _fingerprint(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def run(model_dir, data_path, target_unit, *, flavor, sensor_file='sensor_roles.xlsx',
        groups=None, group_name=None, period_start=None, period_end=None,
        allow_before_cutoff=False, require_unseen_unit=False, allow_missing_targets=False,
        warmup_hours=None, output_dir='saved_model_checks', threads=2):
    folder = Path(model_dir).resolve()
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    detected = 'legacy' if 'core' in cfg else 'actual' if 'method' in cfg else None
    if flavor != detected:
        raise ValueError(f'Формат модели {detected}, открыт ноутбук {flavor}.')
    output_root = Path(output_dir).resolve()
    if output_root == folder or output_root.is_relative_to(folder):
        raise ValueError('Результаты нельзя сохранять внутрь папки обученной модели.')
    before = _fingerprint(folder)
    with _runtime(folder, flavor) as (model, runtime), threadpool_limits(limits=threads):
        sources = list(model.sources) if flavor == 'actual' else list(cfg.get('training_units', []))
        if not sources:
            raise ValueError('В модели нет списка котлов обучения; выберите экспорт с training_units/sources.')
        selected, members, group_origin = _group(cfg, sources, target_unit, groups, group_name)
        if require_unseen_unit and target_unit in sources:
            raise ValueError(f'{target_unit} входил в обучение. Для строгого переноса выберите другого получателя/модель.')
        step = model.step if flavor == 'actual' else pd.Timedelta(cfg['step'])
        cutoff = pd.Timestamp(model.cutoff) if flavor == 'actual' else pd.Timestamp(cfg['trained_until']) + step
        raw = _read_grid(data_path, step)
        raw, start, end = _period(raw, step, cutoff, period_start, period_end, allow_before_cutoff, warmup_hours)
        if flavor == 'actual':
            canonical = runtime.canonical_name
            rows = runtime.read_sensor_book(sensor_file, target_unit).to_dict('records') if sensor_file else []
            aliases = getattr(model, 'schemas', {}).get(target_unit, {}).get('names', {})
        else:
            import legacy_group as LG
            canonical = LG.canonical_name
            rows = LG.read_roles(sensor_file, target_unit) if sensor_file else []
            aliases = {canonical(k, target_unit): v for node in _legacy_nodes(model)
                       for k, v in node.cfg.get('input_aliases', {}).items()}
        frame, audit = _map(raw, _requirements(model, flavor), target_unit, rows, canonical, aliases)
        tag = re.sub(r'[^\w.-]+', '_', target_unit)
        dest = output_root / (tag + '_' + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        dest.mkdir(parents=True, exist_ok=False)
        audit.to_csv(dest / 'sensor_audit.csv', index=False, encoding='utf-8-sig')
        bad = audit.loc[audit.status.isin(['missing', 'empty']) &
                        (~audit.role.eq('target') | (not allow_missing_targets))]
        if len(bad):
            raise ValueError(f'Не хватает входов модели: {bad.key.tolist()}. Аудит: {dest / "sensor_audit.csv"}')
        numbers = []
        for source in audit.source.dropna():
            match = re.match(r'^(?:КА|KA|KА|КA|К|K)[\s._-]*(\d+)\b', source, re.IGNORECASE)
            if match:
                numbers.append(int(match[1]))
        expected_number = re.search(r'\d+', target_unit)
        if numbers and expected_number and set(numbers) != {int(expected_number[0])}:
            raise ValueError(f'В сопоставленных заголовках котлы {sorted(set(numbers))}, выбран {target_unit}. Проверьте DATA_PATH.')
        frames, contributions = {}, {}
        episodes = []
        if flavor == 'actual':
            scored, matrix = model.score(frame, target_unit)
            for view, scores in scored.items():
                stream = runtime.alarm_stream(scores.ratio, model.config)
                frames[view] = scores.assign(smoothed=scores.score, alarm=stream.active, notification=stream.notification)
                contributions[view] = matrix
                for row in runtime.alarm_episodes(stream).to_dict('records'):
                    episodes.append(dict(view=view, **row))
        else:
            result = model.score(frame, prepared=True)
            for view in result.hourly.columns.get_level_values(0).unique():
                data = result.hourly[view]
                local = pd.DataFrame(dict(score=data['ошибка'], smoothed=data['сглаженная'],
                    threshold=data['порог'], ratio=data['отношение'], alarm=data['тревога'],
                    eligible=result.mode.isin(['работа', 'перед остановом']), ood=np.nan), index=data.index)
                local['notification'] = False
                for row in result.alarms.loc[result.alarms['вид'].eq(view)].to_dict('records'):
                    left, right = pd.Timestamp(row['начало']), pd.Timestamp(row['конец'])
                    if left in local.index:
                        local.loc[left, 'notification'] = True
                    episodes.append(dict(view=view, start=left, end=right, hours=row['часов'], peak=row['пик']))
                frames[view] = local
                matrix = result.sensor_errors.get(view)
                contributions[view] = matrix if matrix is not None else pd.DataFrame(index=frame.index)
        episode_table = pd.DataFrame(episodes, columns=['view','start','end','hours','peak'])
        episode_table = episode_table.loc[episode_table.start.lt(end) & episode_table.end.gt(start)].copy()
        episode_table['started_before_period'] = episode_table.start.lt(start)
        episode_table['ends_after_period'] = episode_table.end.gt(end)
        episode_table['hours_in_period'] = (episode_table.end.clip(upper=end) - episode_table.start.clip(lower=start)) / pd.Timedelta('1h')
        summary = []
        for view in frames:
            frames[view] = frames[view].loc[(frames[view].index >= start) & (frames[view].index < end)].copy()
            data = frames[view]
            contributions[view] = contributions[view].reindex(data.index)
            observed = data.score.notna()
            possible = data.eligible.fillna(False)
            hours = observed.sum() * step / pd.Timedelta('1h')
            alarm_h = (data.alarm & observed).sum() * step / pd.Timedelta('1h')
            summary.append(dict(view=view, evaluated_hours=hours,
                availability=float((observed & possible).sum()/possible.sum()) if possible.any() else np.nan,
                alarm_hours=alarm_h, alarm_fraction_of_scored=alarm_h/hours if hours else np.nan,
                notifications_in_period=int(data.notification.sum()),
                episodes_overlapping_period=int(episode_table.view.eq(view).sum())))
        summary = pd.DataFrame(summary)
        timeline = pd.concat(frames, names=['view','time']).reset_index()
        timeline.insert(0,'unit',target_unit)
        timeline.to_parquet(dest/'timeline.parquet',index=False)
        timeline.to_csv(dest/'timeline.csv',index=False,encoding='utf-8-sig')
        summary.to_csv(dest/'summary.csv',index=False,encoding='utf-8-sig')
        episode_table.to_csv(dest/'alarms.csv',index=False,encoding='utf-8-sig')
        for i,(view,matrix) in enumerate(contributions.items()):
            matrix.rename_axis('time').to_parquet(dest/f'sensor_errors_{i:02}.parquet')
        after = _fingerprint(folder)
        if before != after:
            raise RuntimeError('Содержимое папки модели изменилось во время проверки.')
        metadata = dict(format=flavor, model_dir=str(folder), data_path=str(Path(data_path).resolve()),
            method=cfg.get('method', cfg.get('title')),
            target_unit=target_unit, sources=sources, recipient_in_training=target_unit in sources,
            group=selected, group_members=members, group_origin=group_origin,
            cutoff=str(cutoff), period_start=str(start), period_end_exclusive=str(end),
            evaluation_before_cutoff=bool(start < cutoff), input_start=str(frame.index[0]),
            state_labels_present='Состояние' in frame, step=str(step), model_files_unchanged=True,
            recipient_verified_by_headers=bool(numbers),
            run_threshold_used=float(model.run_thresholds.get(target_unit, model.default_run_threshold)) if flavor == 'actual'
                else [node.cfg['run_thr'] for node in _legacy_nodes(model) if 'run_thr' in node.cfg],
            model_sha256=before, views=list(frames),
            note='Веса, нормировка и пороги сохранённой модели не изменены. Статистика тревог не является оценкой ложных тревог по заявкам.')
        (dest/'run.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return dict(root=dest, metadata=metadata, audit=audit, scores=frames,
                    contributions=contributions, summary=summary, alarms=episode_table)


def plot(result, view=None, top=10):
    import matplotlib.pyplot as plt
    view = view or ('top5' if 'top5' in result['scores'] else next(iter(result['scores'])))
    if view not in result['scores']:
        raise ValueError(f'Нет вида {view!r}; доступны {list(result["scores"])}')
    scores = result['scores'][view]
    fig, axes = plt.subplots(2,1,figsize=(15,8),gridspec_kw={'height_ratios':[2,1]})
    axes[0].plot(scores.index,scores.ratio,label='отношение к порогу')
    axes[0].axhline(1,color='red',linestyle='--',label='порог')
    axes[0].fill_between(scores.index,0,1,where=scores.alarm.astype(bool),
                         transform=axes[0].get_xaxis_transform(),color='red',alpha=.12,label='тревога')
    axes[0].set_title(f'{result["metadata"]["target_unit"]} · {view}')
    axes[0].legend(); axes[0].grid(alpha=.2)
    matrix = result['contributions'][view].abs()
    peaks = matrix.max().dropna().nlargest(top).sort_values()
    if len(peaks):
        axes[1].barh(peaks.index,peaks.values)
    axes[1].set_title('Наибольшие нормированные ошибки датчиков за период')
    fig.tight_layout()
    fig.savefig(result['root']/'overview.png',dpi=140,bbox_inches='tight')
    return fig
