"""Resumable chronological experiments with honest current-version calibration."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import os
import sys
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from .config import load
from .data import load_sources, healthy, read_rules
from .models import NormalModel, RelatedModel, AnalogCorrection, MultiTargetModel, balanced_rows
from .alarm_scores import sensor_scale, score_views, calibrate_threshold, alarm_stream, evaluate
from .progress import Progress
from .artifacts import export_ready, publish_export, record_export
from .threads import numerical_threads


def atomic_dump(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    joblib.dump(value, tmp, compress=3)
    os.replace(tmp, path)


def dataset(context, variant, horizon, cfg):
    f = context['features'].copy()
    y = context['cleaned'][context['targets']].copy()
    eligible = context['eligible'].copy()
    if variant == 'related':
        f=pd.concat([f,y.add_prefix('sensor::')],axis=1)
    elif variant == 'forecast':
        shift = max(1, int(round(pd.Timedelta(hours=horizon)/pd.Timedelta(cfg['step']))))
        f = f.shift(shift)
        lagged={}
        for c in context['targets']:
            lagged[c+'::past_value'] = y[c].shift(shift)
            lagged[c+'::past_delta'] = y[c].diff().shift(shift)
            lagged[c+'::past_held_hours'] = context['changes'][c+'::held_hours'].shift(shift)
        f=pd.concat([f,pd.DataFrame(lagged,index=f.index)],axis=1)
        # A forecast must originate in an observed operating interval.
        eligible &= context['eligible'].shift(shift, fill_value=False)
        eligible &= context['eligible'].rolling(shift+1, min_periods=shift+1).sum().eq(shift+1)
    y = y.where(eligible, np.nan)
    return f, y


def align_contexts(contexts, units):
    """A pooled/transfer experiment has a fixed physical schema and declared units."""
    common = set.intersection(*(set(contexts[u]['roles']) for u in units))
    for col in common:
        roles = {contexts[u]['roles'][col] for u in units}
        declared = {contexts[u]['declared_units'].get(col) for u in units} - {'', None}
        if len(roles) != 1 or len(declared) > 1:
            raise ValueError(f'Incompatible physical schema: {col}, roles={roles}, units={declared}')
    out = {}
    for u in units:
        # Frames are read-only here; copy only dictionaries/lists we change.
        c = dict(contexts[u])
        c['roles'] = {k:v for k,v in c['roles'].items() if k in common}
        c['targets'] = [k for k in c['targets'] if k in common]
        if not c['targets']:
            raise ValueError('No shared target sensors')
        base = [k for k,v in contexts[u]['roles'].items() if v in ('load','condition','state') and k not in common]
        drop = [k for k in c['features'] if any(k == b or k.startswith(b+'::') for b in base)]
        c['features'] = c['features'].drop(columns=drop)
        c['features'] = c['features'].reindex(sorted(c['features']), axis=1)
        c['targets'] = sorted(c['targets'])
        out[u] = c
    return out


def experiments(cfg):
    for group, members in cfg['groups'].items():
        for scenario in cfg['scenarios']:
            sources = [[u] for u in members] if scenario in ('local','transfer') else [members]
            for donors in sources:
                targets = donors if scenario == 'local' else [u for u in members if u not in donors] if scenario == 'transfer' else members
                if not targets:
                    continue
                for method in cfg['models']:
                    for variant in cfg['feature_sets']:
                        if variant != 'conditions' and method != 'CatBoost':
                            continue
                        for horizon in (cfg['horizons'] if variant == 'forecast' else [0]):
                            yield dict(group=group, scenario=scenario, sources=donors, recipients=targets,
                                       method=method, variant=variant, horizon=horizon)


def fingerprint(contexts, cutoff, cfg, spec):
    digest = hashlib.sha256(json.dumps(dict(cfg=cfg, spec=spec), sort_keys=True, ensure_ascii=False).encode())
    for p in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(p.read_bytes())
    digest.update((Path(__file__).parent.parent/'kernel_models.py').read_bytes())
    for library in ('numpy','pandas','scipy','scikit-learn','catboost','joblib'):
        digest.update((library+importlib.metadata.version(library)).encode())
    digest.update(Path(cfg['sensor_rules']).read_bytes())
    for u in spec['sources']:
        past = contexts[u]['frame'].loc[lambda d:d.index < cutoff]
        digest.update(u.encode()); digest.update(pd.util.hash_pandas_object(past, index=True).to_numpy().tobytes())
        digest.update(json.dumps(contexts[u]['roles'], sort_keys=True).encode())
        digest.update(json.dumps(contexts[u]['aliases'], sort_keys=True).encode())
        digest.update(json.dumps(contexts[u]['declared_units'], sort_keys=True).encode())
    return digest.hexdigest()


def _pooled(tables):
    f = pd.concat(tables, names=['unit', 'time']).swaplevel().sort_index(kind='stable')
    f.index.names = ['time','unit']
    return f


def update_at(contexts, spec, cutoff, cfg, previous=None, progress=None):
    tag=f'{spec["method"]} · {"+".join(spec["sources"])} · {pd.Timestamp(cutoff):%Y-%m}'
    phase=lambda text:progress.stage(tag+' · '+text) if progress else nullcontext()
    fparts, yparts = {}, {}
    for u in spec['sources']:
        c = contexts[u]
        f, y = dataset(c, spec['variant'], spec['horizon'], cfg)
        mask = healthy(c, cutoff, cfg) & y.notna().mean(axis=1).ge(cfg['minimum_coverage'])
        fparts[u], yparts[u] = f.loc[mask], y.loc[mask]
    f, y = _pooled(fparts), _pooled(yparts)
    # Whole timestamps, including every unit at that timestamp, stay in one partition.
    mature_end = pd.Timestamp(cutoff)-pd.Timedelta(days=cfg['admission_delay_days'])
    warmup = int(np.ceil(pd.Timedelta(hours=cfg['slow_hours']+cfg['smooth_hours'])/pd.Timedelta(cfg['step'])))
    cal_start = mature_end-(cfg['calibration_rows']+warmup)*pd.Timedelta(cfg['step'])
    val_start = cal_start-cfg['validate_rows']*pd.Timedelta(cfg['step'])
    times = f.index.get_level_values('time')
    train, val, cal = times < val_start, (times >= val_start) & (times < cal_start), times >= cal_start
    if train.sum() < cfg['min_train_rows'] or val.sum() < cfg['min_sensor_rows'] or cal.sum() < cfg['min_sensor_rows']:
        return None, dict(status='insufficient_partition_data', cutoff=str(cutoff),
                          train=int(train.sum()), validation=int(val.sum()), calibration=int(cal.sum()))
    if previous is None:
        model = (RelatedModel(cfg, cfg['related_inputs']) if spec['variant'] == 'related' else
                 AnalogCorrection(cfg) if spec['method'] == 'CatBoostCorrection' else
                 MultiTargetModel(spec['method'],cfg) if spec['method'] == 'CatBoostMulti' else NormalModel(spec['method'], cfg))
        with phase(f'первоначальное обучение: {int(train.sum())} строк, {len(y.columns)} целей'):
            model.fit_initial(f.loc[train], y.loc[train])
            reference = deepcopy(model)
        ingested = set(f.index[train].tolist())
        audit = dict(status='initial', accepted=True, new_rows=int(train.sum()))
    else:
        model, reference, ingested = previous['model'], previous['reference'], set(previous['ingested'])
        new = train & np.array([key not in ingested for key in f.index])
        nf, ny = f.loc[new], y.loc[new]
        if len(nf) > cfg['update_rows']:
            keep = balanced_rows(nf, cfg['update_rows']); nf, ny = nf.iloc[keep], ny.iloc[keep]
        if not len(nf):
            audit = dict(status='no_new_admitted_rows', accepted=False, new_rows=0)
        elif spec['method'] == 'HistGradientBoosting':
            audit = dict(status='frozen_comparison_baseline', accepted=False, new_rows=0)
        elif model.tree_counts() and all(n+cfg['cb_update_trees']>cfg['cb_max_trees'] for n in model.tree_counts().values()):
            audit = dict(status='tree_capacity_reached',accepted=False,new_rows=0,trees=model.tree_counts())
        else:
            # Gate on unseen recent validation AND representation of old operating modes.
            old = reference.memory_f.iloc[balanced_rows(reference.memory_f, min(100, cfg['replay_rows']))]
            old_y = reference.memory_y.reindex(old.index)
            vf, vy = pd.concat([f.loc[val], old]).sort_index(), pd.concat([y.loc[val], old_y]).sort_index()
            with phase(f'обновление и проверка кандидата: {len(nf)} новых строк'):
                model, audit = model.candidate(nf, ny, vf, vy)
            audit.update(status='updated' if audit['accepted'] else 'candidate_rejected', new_rows=len(nf))
            if audit['accepted']:
                ingested.update(nf.index.tolist())
    errors, ref_errors = {}, {}
    for u in spec['sources']:
        cf, cy = f.xs(u, level='unit'), y.xs(u, level='unit')
        mask = (cf.index >= cal_start) & (cf.index < mature_end)
        # Reindex gaps before smoothing: noncontiguous healthy hours must not join episodes.
        full_index = pd.date_range(cal_start, mature_end, freq=cfg['step'], inclusive='left')[-cfg['calibration_limit']:]
        with phase(f'{u} · ошибки калибровки'):
            e = model.errors(cf.loc[mask], cy.loc[mask])
            reference_error = reference.errors(cf.loc[mask], cy.loc[mask])
        errors[u] = e.reindex(full_index)
        ref_errors[u] = reference_error.reindex(full_index)
    scale = sensor_scale(pd.concat(errors.values()))
    ref_scale = sensor_scale(pd.concat(ref_errors.values()))
    rules = read_rules(cfg['sensor_rules'])
    groups = {col:rules.get('sensors', {}).get(col, {}).get('group', '') for col in model.targets}
    thresholds = {}
    for u, e in errors.items():
        with phase(f'{u} · подбор порогов тревог'):
            views, _ = score_views(e, scale, cfg, groups)
            thresholds[u] = {view:{str(b):calibrate_threshold(data.smoothed, cfg, b) for b in cfg['alarm_budgets']}
                             for view, data in views.items()}
    audit.update(cutoff=str(cutoff), calibration_start=str(cal_start), mature_end=str(mature_end),
                 train_rows=int(train.sum()), validation_rows=int(val.sum()), calibration_rows=int(cal.sum()),
                 inherited_from=str(previous['cutoff']) if previous else None, trees=model.tree_counts(),
                 incremental_updates=model.updates)
    return dict(model=model, reference=reference, ingested=ingested, cutoff=pd.Timestamp(cutoff),
                scale=scale, reference_scale=ref_scale, thresholds=thresholds, groups=groups, audit=audit), audit


def score_context(state, context, spec, cfg):
    f, y = dataset(context, spec['variant'], spec['horizon'], cfg)
    e = state['model'].errors(f, y).where(context['eligible'], np.nan)
    re = state['reference'].errors(f, y).where(context['eligible'], np.nan)
    views, matrices = score_views(e, state['scale'], cfg, state['groups'])
    default = state['thresholds'].get(context['unit'])
    if default is None:
        # Strict transfer: no recipient data are used to fit thresholds or sensor scales.
        default = {}
        for view in next(iter(state['thresholds'].values())):
            default[view] = {}
            for b in cfg['alarm_budgets']:
                donors = [t[view][str(b)] for t in state['thresholds'].values()]
                threshold = max(d['threshold'] for d in donors)
                default[view][str(b)] = dict(threshold=threshold, status='donor_threshold', hours=sum(d['hours'] for d in donors))
    for view, d in views.items():
        d['threshold'] = default[view][str(cfg['alarm_budget'])]['threshold']
        d['ratio'] = d.smoothed/d.threshold.where(np.isfinite(d.threshold))
        d['eligible'] = context['eligible']
        d['background_eligible'] = context['frame']['Состояние'].eq('Работа')
    # Rich second-stage inputs use signed errors and quality, not repair labels.
    from .interpretation import episode_features
    interpretation_features = episode_features(e, context, cfg)
    interpretation_features['reference_mean'] = re.abs().mean(axis=1)
    interpretation_features['adaptive_mean'] = e.abs().mean(axis=1)
    interpretation_features['reference_gap'] = interpretation_features.reference_mean-interpretation_features.adaptive_mean
    return views, matrices, e, interpretation_features


def scoring_context(context, spec, cfg, start, end):
    """Keep only a month and the history required by lags and rolling features.

    Conditions, regimes and quality were already prepared on the full source, so
    their ages/categorical histories are retained rather than restarted here.
    """
    hours=max(cfg['slow_hours']+cfg['smooth_hours'],48.,1.)+spec['horizon']
    left=pd.Timestamp(start)-pd.Timedelta(hours=hours)-pd.Timedelta(cfg['step'])
    take=(context['frame'].index>=left)&(context['frame'].index<pd.Timestamp(end))
    return {key:(value.loc[take] if isinstance(value,(pd.DataFrame,pd.Series)) and
                 value.index.equals(context['frame'].index) else value) for key,value in context.items()}


def score_fingerprint(context, end, model_token):
    digest=hashlib.sha256(model_token.encode())
    frame=context['frame'].loc[lambda f:f.index<end]
    digest.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    for key in ('roles','aliases','declared_units'):
        digest.update(json.dumps(context.get(key,{}),sort_keys=True,ensure_ascii=False).encode())
    return digest.hexdigest()


def score_month(state, context, spec, cfg, cutoff, end):
    c=scoring_context(context,spec,cfg,cutoff,end)
    views,_,errors,features=score_context(state,c,spec,cfg)
    mask=(errors.index>=cutoff)&(errors.index<end)
    diagnostics=[]
    if hasattr(state['model'],'interval_diagnostics'):
        f,y=dataset(c,spec['variant'],spec['horizon'],cfg)
        diagnostics=[dict(month=str(cutoff),model_cutoff=str(state['cutoff']),**row)
                     for row in state['model'].interval_diagnostics(f.loc[mask],y.loc[mask])]
    thresholds=state['thresholds'].get(context['unit'])
    if thresholds is None:
        thresholds={view:{str(b):{'threshold':max(d[view][str(b)]['threshold']
            for d in state['thresholds'].values())} for b in cfg['alarm_budgets']}
            for view in views}
    result={}
    for view,scores in views.items():
        scores=scores.loc[mask].copy()
        scores['model_cutoff']=state['cutoff']
        for budget in cfg['alarm_budgets']:
            scores['threshold_budget_'+str(budget)]=thresholds[view][str(budget)]['threshold']
        result[view]=scores
    return dict(views=result,errors=errors.loc[mask],features=features.loc[mask],diagnostics=diagnostics)


def run(cfg):
    cfg=load(overrides=cfg)
    # Large numbers of small BLAS solves can stall with an unrestricted thread
    # pool. Apply the configured limit to the entire numerical lifecycle.
    with numerical_threads(cfg['threads']):
        return _run(cfg)


def _run(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    progress=Progress(root)
    with progress.stage('проверка зависимостей текущего Python'):
        missing=[]
        for name in ('numpy','pandas','scipy','scikit-learn','catboost','joblib','threadpoolctl','openpyxl','pyarrow','matplotlib'):
            try:
                importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(name)
        if missing:
            raise RuntimeError('В текущем ядре '+sys.executable+' отсутствуют пакеты: '+', '.join(missing)+
                               '. Установите зависимости в этом ядре: %pip install -r requirements.txt')
    with progress.stage('загрузка данных и аудит'):
        contexts,_=load_sources(cfg,root/'audit')
    specs=list(experiments(cfg))
    progress.say(f'Запланировано экспериментов: {len(specs)}. Модели сохраняются после каждого месячного среза.')
    summaries,statuses=[],[]
    for number,spec in enumerate(specs,1):
        name = '__'.join([spec['group'], spec['scenario'], '+'.join(spec['sources']), spec['method'], spec['variant'], str(spec['horizon'])])
        progress.say(f'Эксперимент {number}/{len(specs)}: {name}',experiment=name)
        if any(c in name for c in '<>:"/\\|?*') or '..' in name:
            raise ValueError('Experiment names must be safe path components')
        dest = root/'experiments'/name; dest.mkdir(parents=True, exist_ok=True)
        (dest/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
        members = list(dict.fromkeys(spec['sources']+spec['recipients']))
        selected = align_contexts(contexts, members) if spec['scenario'] != 'local' else {u:contexts[u] for u in members}
        if spec['variant'] == 'related' and not all(cfg['related_inputs'].get(c) for c in next(iter(selected.values()))['targets']):
            statuses.append(dict(experiment=name, status='requires_reviewed_related_inputs')); continue
        first = pd.Timestamp(cfg['first_test'])
        # Recipients cannot cause models to be trained beyond donor observation time.
        last = min(max(selected[u]['frame'].index[-1] for u in spec['recipients']),
                   min(selected[u]['frame'].index[-1] for u in spec['sources']))
        months = [first]+list(pd.date_range(first+pd.offsets.MonthBegin(1), last, freq='MS'))
        state = None
        pieces = {u:{} for u in spec['recipients']}
        error_parts, feature_parts, audit_rows = {u:[] for u in pieces}, {u:[] for u in pieces}, []
        prediction_diagnostics={u:[] for u in pieces}
        for month_number,cutoff in enumerate(months,1):
            tag=f'{name} · месяц {month_number}/{len(months)} · {cutoff:%Y-%m}'
            with progress.stage(tag+' · проверка сохранённого состояния'):
                token = fingerprint(selected, cutoff, cfg, spec)
            path = dest/'checkpoints'/f'{cutoff:%Y-%m-%d}.joblib'
            if path.exists():
                with progress.stage(tag+' · загрузка checkpoint'):
                    saved = joblib.load(path)
                    if saved.get('fingerprint') != token:
                        raise ValueError(f'Changed past data, configuration or code: use a new output directory ({path})')
                    state = saved['state']; audit = dict(state['audit'], restored=True)
            else:
                new_state, audit = update_at(selected, spec, cutoff, cfg, state,progress)
                if new_state is not None:
                    state = new_state
                    with progress.stage(tag+' · сохранение checkpoint'):
                        atomic_dump(dict(fingerprint=token, state=state), path)
            audit=dict(audit)
            audit_rows.append(audit)
            progress.say(tag+' · модель: '+audit['status'],status=audit['status'])
            if state is None:
                progress.say('Модель пока не получена: '+json.dumps(audit,ensure_ascii=False,default=str))
                pd.DataFrame(audit_rows).to_json(dest/'updates.json',orient='records',indent=2,force_ascii=False,date_format='iso')
                continue
            end = cutoff+pd.offsets.MonthBegin(1)
            for u in spec['recipients']:
                # Truncate inputs before scoring: no accidental future-dependent features.
                c = selected[u]
                from .runtime import export
                history=dest/u/'model_history'/f'{cutoff:%Y-%m}'
                current=dest/u/'model'
                with progress.stage(tag+f' · {u} · экспорт модели для дашборда'):
                    if not export_ready(history,token):
                        export(state,c,spec,cfg,history,fingerprint=token)
                    if not export_ready(current,token):
                        publish_export(history,current)
                    record_export(root,name,u,spec,state,cutoff,current,history)
                progress.say('Готовая модель для дашборда: '+str(current.resolve()),model_dir=str(current.resolve()))
                cached=dest/'scored'/f'{cutoff:%Y-%m-%d}'/(u+'.joblib')
                with progress.stage(tag+f' · {u} · оценки месяца и признаки'):
                    score_token=score_fingerprint(c,end,token)
                    saved=joblib.load(cached) if cached.exists() else None
                    if saved and saved.get('fingerprint')==score_token:
                        month=saved['month']
                        progress.say('Восстановлены сохранённые оценки месяца: '+str(cached))
                    else:
                        month=score_month(state,c,spec,cfg,cutoff,end)
                        atomic_dump(dict(fingerprint=score_token,month=month),cached)
                prediction_diagnostics[u].extend(month['diagnostics'])
                for view,scores in month['views'].items():
                    pieces[u].setdefault(view, []).append(scores)
                error_parts[u].append(month['errors']);feature_parts[u].append(month['features'])
            pd.DataFrame(audit_rows).to_json(dest/'updates.json',orient='records',indent=2,force_ascii=False,date_format='iso')
            progress.say(tag+' · расчёт месяца завершён',status='month_complete')
        pd.DataFrame(audit_rows).to_json(dest/'updates.json', orient='records', indent=2, force_ascii=False, date_format='iso')
        if state is None:
            statuses.append(dict(experiment=name, status='insufficient_history')); continue
        from .interpretation import write_catalog
        for u in spec['recipients']:
            ud = dest/u; ud.mkdir(exist_ok=True)
            timeline_features = pd.concat(feature_parts[u]).sort_index()
            residuals = pd.concat(error_parts[u]).sort_index()
            residuals.to_parquet(ud/'prequential_errors.parquet')
            pd.DataFrame(prediction_diagnostics[u]).to_csv(ud/'prediction_diagnostics.csv',index=False,encoding='utf-8-sig')
            timeline_features.to_parquet(ud/'interpretation_features.parquet')
            with progress.stage(name+f' · {u} · тревоги, метрики и отчёты'):
                timelines = {}
                for view, parts in pieces[u].items():
                    data = pd.concat(parts).sort_index()
                    # Persistence remains continuous across month boundaries.
                    for b in cfg['alarm_budgets']:
                        threshold = data['threshold_budget_'+str(b)]
                        ratio = data.smoothed/threshold.where(np.isfinite(threshold))
                        stream, episodes = alarm_stream(ratio, cfg)
                        scored = data.assign(ratio=ratio, active=stream.active, notification=stream.notification)
                        metrics, events = evaluate(scored, selected[u]['events'], cfg, start=first)
                        summaries.append(dict(experiment=name, unit=u, view=view, budget=b, **metrics))
                        slug = view.replace(':','_')+'__'+str(b)
                        scored.to_parquet(ud/(slug+'.parquet'))
                        episodes.to_csv(ud/(slug+'_alarms.csv'), index=False, encoding='utf-8-sig')
                        events.to_csv(ud/(slug+'_events.csv'), index=False, encoding='utf-8-sig')
                        if b == cfg['alarm_budget']:
                            timelines[view] = scored
                write_catalog(timelines, timeline_features, selected[u], cfg, ud)
            statuses.append(dict(experiment=name, unit=u, status='complete', model=str(ud/'model')))
        pd.DataFrame(summaries).to_csv(root/'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(statuses).to_csv(root/'status.csv', index=False, encoding='utf-8-sig')
    from .comparison import common_comparison
    with progress.stage('итоговое сравнение на общих часах'):
        common_comparison(root,contexts,cfg)
    progress.say('Расчёт завершён. Список готовых моделей: '+str((root/'models.csv').resolve()),status='complete')
    return dict(summary=pd.DataFrame(summaries), status=pd.DataFrame(statuses))
