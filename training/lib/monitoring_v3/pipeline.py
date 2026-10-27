"""Resumable chronological experiments with honest current-version calibration."""
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
from .models import NormalModel, RelatedModel, AnalogCorrection, MultiTargetModel, balanced_rows, snapshot_model,operating_support
from .alarm_scores import sensor_scale, score_views, calibrate_threshold, alarm_stream, evaluate,view_policy
from .progress import Progress
from .artifacts import export_ready, publish_export, record_export
from .threads import numerical_threads
from .target_training import initial_targets
from .checkpoint_cache import signatures,checkpoint_compatible,backup_checkpoint
from .partitioning import partition_history


def atomic_dump(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    joblib.dump(value, tmp, compress=3)
    os.replace(tmp, path)


def write_update_reports(dest,rows):
    pd.DataFrame(rows).to_json(dest/'updates.json',orient='records',indent=2,force_ascii=False,date_format='iso')
    calibration=[];targets=[];sensor_calibration=[];admission=[]
    for row in rows:
        admission.extend(dict(month=row.get('cutoff'),unit=unit,status=row.get('status'),**values)
            for unit,values in row.get('admission_by_unit',{}).items())
        for unit,views in row.get('calibration_by_unit',{}).items():
            for view,budgets in views.items():
                for budget,details in budgets.items():
                    calibration.append({**details,'month':row.get('cutoff'),'unit':unit,'view':view,'budget':budget,
                                        'calibration_start':row.get('calibration_start')})
        for sensor,details in row.get('target_errors',{}).items():
            targets.append(dict(month=row.get('cutoff'),sensor=sensor,accepted=row.get('accepted'),**details))
        sensor_calibration.extend(dict(month=row.get('cutoff'),**d) for d in row.get('calibration_sensor_diagnostics',[]))
    pd.DataFrame(calibration).to_csv(dest/'calibration_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(targets).to_csv(dest/'target_update_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sensor_calibration).to_csv(dest/'calibration_sensor_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(admission).to_csv(dest/'admission_diagnostics.csv',index=False,encoding='utf-8-sig')


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
    f.attrs['load_column']=context['load']
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


def transfer_preparation(contexts,spec,cfg):
    if spec['scenario']!='transfer':return contexts
    from .data import prepare_unit
    source=contexts[spec['sources'][0]];rules=read_rules(cfg['sensor_rules']);result=dict(contexts)
    for unit in spec['recipients']:
        original=contexts[unit]
        prepared=prepare_unit(original['frame'],original['roles'],unit,cfg,rules,
            threshold=source['threshold'],regime_spec=source['regime_spec'],build_events=False)
        prepared['events']=original['events']
        prepared['threshold_source']='donor:'+source['unit']
        prepared['features']=prepared['features'].reindex(sorted(prepared['features']),axis=1)
        result[unit]={**original,**prepared}
    return result


def _pooled(tables):
    f = pd.concat(tables, names=['unit', 'time']).swaplevel().sort_index(kind='stable')
    f.index.names = ['time','unit']
    loads={d.attrs.get('load_column') for d in tables.values()}-{None}
    if len(loads)==1:f.attrs['load_column']=next(iter(loads))
    f.attrs['condition_columns']=sorted({c for d in tables.values() for c in d.attrs.get('condition_columns',[]) if c in f})
    return f


def update_at(contexts, spec, cutoff, cfg, previous=None, progress=None):
    tag=f'{spec["method"]} · {"+".join(spec["sources"])} · {pd.Timestamp(cutoff):%Y-%m}'
    phase=lambda text:progress.stage(tag+' · '+text) if progress else nullcontext()
    fparts, yparts,admission = {}, {},{}
    for u in spec['sources']:
        c = contexts[u]
        f, y = dataset(c, spec['variant'], spec['horizon'], cfg)
        mask = healthy(c, cutoff, cfg) & y.notna().mean(axis=1).ge(cfg['minimum_coverage'])
        past=c['frame'].index<pd.Timestamp(cutoff)
        required=[name for name,role in c['roles'].items() if role in ('load','condition','state')]
        admission[u]=dict(source_past_rows=int(past.sum()),operating_rows=int((past&c['eligible']).sum()),
            enough_target_rows=int((past&c['eligible']&y.notna().mean(axis=1).ge(cfg['minimum_coverage'])).sum()),
            missing_input_rows=int((past&c['eligible']&c['cleaned'][required].notna().mean(axis=1).lt(cfg['minimum_coverage'])).sum()),
            admitted_normal_rows=int(mask.sum()))
        fparts[u], yparts[u] = f.loc[mask], y.loc[mask]
    f, y = _pooled(fparts), _pooled(yparts)
    original_previous=previous
    retracted=set(previous.get('ingested',()))-set(f.index) if previous else set()
    if retracted:previous=None
    def unavailable(audit):
        audit.update(retracted_normal_rows=len(retracted),previous_retained=original_previous is not None)
        if retracted:
            audit['unavailable_reason']=audit['status'];audit['status']='unsafe_retracted_normal_history'
            retained={**original_previous,'cutoff':pd.Timestamp(cutoff),'unsafe_normal':True,
                'audit':{**original_previous['audit'],**audit}}
            return retained,audit
        return None,audit
    # Whole timestamps, including every unit at that timestamp, stay in one partition.
    mature_end = pd.Timestamp(cutoff)-pd.Timedelta(days=cfg['admission_delay_days'])
    times = f.index.get_level_values('time')
    train,val,cal,reserved_cal,reserved_val,partition_audit=partition_history(f,cfg,original_previous)
    if train.sum()<cfg['min_train_rows'] or cal.sum()<cfg['calibration_rows']:
        reason='insufficient_training_data' if train.sum()<cfg['min_train_rows'] else 'insufficient_unseen_calibration'
        return unavailable(dict(status=reason,cutoff=str(cutoff),**partition_audit,
                         required_train=cfg['min_train_rows'],required_calibration=cfg['calibration_rows'],
                         admission_by_unit=admission))
    cal_start=times[cal].min()
    if previous is None:
        model = (RelatedModel(cfg, cfg['related_inputs']) if spec['variant'] == 'related' else
                 AnalogCorrection(cfg) if spec['method'] == 'CatBoostCorrection' else
                 MultiTargetModel(spec['method'],cfg) if spec['method'] == 'CatBoostMulti' else NormalModel(spec['method'], cfg))
        with phase(f'первоначальное обучение: {int(train.sum())} строк, {len(y.columns)} целей'):
            model.fit_initial(f.loc[train], y.loc[train])
            reference = snapshot_model(model)
        ingested = set(f.index[train].tolist())
        audit = dict(status='rebuilt_after_normal_retraction' if retracted else 'initial', accepted=True,
            new_rows=int(train.sum()),retracted_normal_rows=len(retracted))
    else:
        model, reference, ingested = snapshot_model(previous['model']), snapshot_model(previous['reference']), set(previous['ingested'])
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
        elif cfg['validate_candidate'] and val.sum()<cfg['min_sensor_rows']:
            audit=dict(status='insufficient_unseen_validation',accepted=False,new_rows=0,
                       waiting_new_rows=len(nf),required_validation=cfg['min_sensor_rows'],previous_retained=True)
        else:
            vf=vy=None
            if cfg['validate_candidate']:
                # Gate on unseen recent validation AND representation of old operating modes.
                old = reference.memory_f.iloc[balanced_rows(reference.memory_f, min(100, cfg['replay_rows']))]
                old_y = reference.memory_y.reindex(old.index)
                vf, vy = pd.concat([f.loc[val], old]).sort_index(), pd.concat([y.loc[val], old_y]).sort_index()
            label='обновление и проверка кандидата' if cfg['validate_candidate'] else 'обновление без проверки кандидата'
            with phase(f'{label}: {len(nf)} новых строк'):
                model, audit = model.candidate(nf, ny, vf, vy)
            audit.update(status='updated' if audit['accepted'] else 'candidate_rejected', new_rows=len(nf))
            if audit['accepted']:
                ingested.update(nf.index.tolist())
    errors, ref_errors = {}, {}
    with phase('ошибки калибровки · обновляемая модель · вся отложенная история'):
        all_errors=calibration_errors(model,f.loc[cal],y.loc[cal])
    with phase('ошибки калибровки · исходная эталонная модель · вся отложенная история'):
        all_reference_errors=calibration_errors(reference,f.loc[cal],y.loc[cal])
    for u in spec['sources']:
        source=all_errors.index.get_level_values('unit')==u
        e=all_errors.loc[source].droplevel('unit')
        reference_error=all_reference_errors.loc[source].droplevel('unit')
        # Smoothing and episodes explicitly reset at timestamp gaps. Do not
        # allocate every shutdown hour in a potentially many-year sparse history.
        errors[u] = e
        ref_errors[u] = reference_error
    scale = sensor_scale(pd.concat(errors.values()))
    ref_scale = sensor_scale(pd.concat(ref_errors.values()))
    rules = read_rules(cfg['sensor_rules'])
    groups = {col:rules.get('sensors', {}).get(col, {}).get('group', '') for col in model.targets}
    event_sensors=[c for c in model.targets if rules.get('sensors',{}).get(c,{}).get('kind') in ('rare','discrete')]
    thresholds = {}
    for u, e in errors.items():
        with phase(f'{u} · подбор порогов тревог'):
            changed=pd.DataFrame({c:contexts[u]['changes'][c+'::changed'] for c in event_sensors})
            views, _ = score_views(e, scale, cfg, groups,event_sensors,changed)
            thresholds[u] = {view:{str(b):(dict(threshold=.5,status='observed_change',hours=float(data.score.notna().sum())) if view.endswith('::event') else calibrate_threshold(data.smoothed, cfg, b)) for b in cfg['alarm_budgets']}
                             for view, data in views.items()}
    ready=[u for u,t in thresholds.items() if np.isfinite(t['mean::instant'][str(cfg['alarm_budget'])]['threshold'])]
    if not ready:
        return unavailable(dict(status='insufficient_valid_calibration_scores',cutoff=str(cutoff),
            **partition_audit,previous_retained=previous is not None,
            calibration_status={u:t['mean::instant'][str(cfg['alarm_budget'])] for u,t in thresholds.items()}))
    # A recipient without calibration cannot make a pooled source with valid
    # history unusable. Borrowing is explicit and always uses finite donor values.
    for u in thresholds:
        for view in thresholds[u]:
            for budget in cfg['alarm_budgets']:
                key=str(budget)
                if np.isfinite(thresholds[u][view][key]['threshold']):continue
                donors=[thresholds[v][view][key] for v in ready if np.isfinite(thresholds[v][view][key]['threshold'])]
                if donors:thresholds[u][view][key]=dict(threshold=max(d['threshold'] for d in donors),
                    status='borrowed_donor_calibration',hours=sum(d['hours'] for d in donors),donors=ready)
    audit.update(cutoff=str(cutoff), calibration_start=str(cal_start), mature_end=str(mature_end),
                 train_rows=int(train.sum()), validation_rows=int(val.sum()), calibration_rows=int(cal.sum()),
                 inherited_from=str(original_previous['cutoff']) if original_previous else None, trees=model.tree_counts(),
                 incremental_updates=model.updates,**{k:v for k,v in partition_audit.items() if k not in ('train','validation','calibration')},
                 last_train_time=str(max(k[0] for k in ingested)),calibration_ready_units=ready,
                 validation_recent_rows=int(val.sum()),calibration_by_unit=thresholds,admission_by_unit=admission)
    audit['calibration_sensor_diagnostics']=all_errors.attrs.get('calibration_sensor_diagnostics',[])
    audit['model_fitted_at']=str(cutoff) if previous is None or audit.get('accepted') else previous['audit'].get('model_fitted_at',str(previous['cutoff']))
    audit['calibration_last_time']=str(times[cal].max())
    frozen=original_previous.get('frozen') if original_previous else None
    if frozen is None:
        frozen_model=snapshot_model(model)
        frozen=dict(model=frozen_model,reference=frozen_model,scale=scale.copy(),reference_scale=scale.copy(),
                    cutoff=pd.Timestamp(cutoff),thresholds=thresholds,groups=groups,event_sensors=event_sensors,audit=dict(audit))
    return dict(model=model, reference=reference, ingested=ingested, cutoff=pd.Timestamp(cutoff),frozen=frozen,
                reserved_calibration=reserved_cal,reserved_validation=reserved_val,
                scale=scale, reference_scale=ref_scale, thresholds=thresholds, groups=groups,event_sensors=event_sensors, audit=audit), audit


def calibration_errors(model,features,actual):
    """An out-of-training measured floor complements explicit instrument scales."""
    if not len(features):return pd.DataFrame(index=features.index,columns=model.targets,dtype=float)
    if isinstance(model,RelatedModel):
        parts=[calibration_errors(m,model._inputs(features,c),actual[[c]]) for c,m in model.members.items()]
        result=pd.concat(parts,axis=1)
        result.attrs['calibration_sensor_diagnostics']=[r for p in parts for r in p.attrs.get('calibration_sensor_diagnostics',[])]
        return result
    if hasattr(model,'floor'):
        model.floor=getattr(model,'engineering_floor',model.floor).copy()
    med,spread=model.expected(features,actual)
    if hasattr(model,'floor'):
        measured=actual.reindex(columns=model.targets).sub(med).abs().median().to_numpy()/0.67448975
        floor=np.maximum(getattr(model,'engineering_floor',model.floor),np.nan_to_num(measured,nan=0.))
        model.floor=floor
        spread=spread.clip(lower=pd.Series(floor,index=model.targets),axis=1)
    actual=actual.reindex(columns=model.targets)
    error=actual.sub(med)
    result=error.div(spread).replace([np.inf,-np.inf],np.nan)
    diagnostics=[]
    units=features.index.get_level_values('unit').unique() if isinstance(features.index,pd.MultiIndex) else ['']
    centers=pd.Series(getattr(model,'centers',np.full(len(model.targets),np.nan)),index=model.targets)
    for unit in units:
        mask=features.index.get_level_values('unit')==unit if isinstance(features.index,pd.MultiIndex) else np.ones(len(features),bool)
        for sensor in model.targets:
            absolute=float(error.loc[mask,sensor].abs().median())
            baseline=float(actual.loc[mask,sensor].sub(centers[sensor]).abs().median())
            diagnostics.append(dict(unit=unit,sensor=sensor,observed=int(actual.loc[mask,sensor].notna().sum()),
                median_absolute_error=absolute,median_spread=float(spread.loc[mask,sensor].median()),
                constant_baseline_error=baseline,skill_vs_constant=(1-absolute/baseline) if baseline>1e-9 else None,
                status=('no_observations' if not actual.loc[mask,sensor].notna().any() else
                        'no_prediction' if not np.isfinite(absolute) else
                        'weak_predictability' if baseline>1e-9 and absolute>=baseline else 'constant_or_predictable')))
    result.attrs['calibration_sensor_diagnostics']=diagnostics
    return result


def score_context(state, context, spec, cfg, return_expected=False,with_features=True):
    f, y = dataset(context, spec['variant'], spec['horizon'], cfg)
    options={'diagnostic_start':context.get('diagnostic_start')} if type(state['model']) is NormalModel else {}
    med,spread=state['model'].expected(f,y,**options)
    inputs=[c for c,r in context['roles'].items() if r in ('load','condition','state')]
    input_coverage=context['cleaned'][inputs].notna().mean(axis=1) if inputs else pd.Series(0.,index=f.index)
    usable=context['eligible']&input_coverage.ge(cfg['minimum_coverage'])&(not state.get('unsafe_normal',False))
    e = y.reindex(columns=state['model'].targets).sub(med).div(spread).where(usable,np.nan)
    re = e if state['reference'] is state['model'] else state['reference'].errors(f, y).where(usable, np.nan)
    rare=state.get('event_sensors',[])
    changed=pd.DataFrame({c:context['changes'][c+'::changed'] for c in rare})
    views, matrices = score_views(e, state['scale'], cfg, state['groups'],rare,changed)
    default = state['thresholds'].get(context['unit'])
    if default is None:
        # Strict transfer: no recipient data are used to fit thresholds or sensor scales.
        default = {}
        for view in next(iter(state['thresholds'].values())):
            default[view] = {}
            for b in cfg['alarm_budgets']:
                donors = [t[view][str(b)] for t in state['thresholds'].values() if np.isfinite(t[view][str(b)]['threshold'])]
                threshold = max((d['threshold'] for d in donors),default=np.inf)
                default[view][str(b)] = dict(threshold=threshold, status='donor_threshold', hours=sum(d['hours'] for d in donors))
    for view, d in views.items():
        d['threshold'] = default[view][str(cfg['alarm_budget'])]['threshold']
        d['ratio'] = d.smoothed/d.threshold.where(np.isfinite(d.threshold))
        d['eligible'] = context['eligible']
        d['background_eligible'] = context['frame']['Состояние'].eq('Работа')
    if not with_features:
        base=(views,matrices,e,None)
        return base+(med,spread) if return_expected else base
    # Classifiers and alarms see the SAME normalized residual contract.
    from .interpretation import episode_features
    interpretation_features = episode_features(e.div(state['scale'],axis=1).drop(columns=rare), context, cfg,state['groups'])
    for c in rare:
        interpretation_features['event::'+c+'::changed']=changed[c]
        interpretation_features['event::'+c+'::value']=context['cleaned'][c]
    interpretation_features['reference_mean'] = re.div(state['reference_scale'],axis=1).drop(columns=rare).abs().mean(axis=1)
    interpretation_features['adaptive_mean'] = e.div(state['scale'],axis=1).drop(columns=rare).abs().mean(axis=1)
    interpretation_features['reference_gap'] = interpretation_features.reference_mean-interpretation_features.adaptive_mean
    interpretation_features['input_coverage']=input_coverage
    support=operating_support(state['model'],f)
    interpretation_features=pd.concat([interpretation_features,support],axis=1)
    interpretation_features['model_age_days']=(f.index-pd.Timestamp(state['audit'].get('model_fitted_at',state['cutoff']))).total_seconds()/86400
    interpretation_features['calibration_age_days']=(f.index-pd.Timestamp(state['audit'].get('calibration_last_time',state['audit']['mature_end']))).total_seconds()/86400
    base=(views,matrices,e,interpretation_features)
    return base+(med,spread) if return_expected else base


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
    c['diagnostic_start']=pd.Timestamp(cutoff)
    views,_,errors,features,med,spread=score_context(state,c,spec,cfg,return_expected=True)
    mask=(errors.index>=cutoff)&(errors.index<end)
    diagnostics=[]
    if hasattr(state['model'],'interval_diagnostics'):
        diagnostics=[dict(month=str(cutoff),model_cutoff=str(state['cutoff']),**row)
                     for row in med.attrs.get('prediction_diagnostics',[])]
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
    frozen={}
    if state.get('frozen') is not None:
        baseline=state['frozen']
        fv,_,_,_=score_context(baseline,c,spec,cfg,with_features=False)
        bt=baseline['thresholds'].get(context['unit'])
        for view,frame in fv.items():
            frame=frame.loc[mask].copy();frame['model_cutoff']=baseline['cutoff']
            for budget in cfg['alarm_budgets']:
                threshold=bt[view][str(budget)]['threshold'] if bt is not None else max(
                    (d[view][str(budget)]['threshold'] for d in baseline['thresholds'].values()
                     if np.isfinite(d[view][str(budget)]['threshold'])),default=np.inf)
                frame['threshold_budget_'+str(budget)]=threshold
            frozen[view]=frame
    return dict(views=result,frozen_views=frozen,errors=errors.loc[mask],features=features.loc[mask],diagnostics=diagnostics)


def run(cfg,live_report=None):
    cfg=load(overrides=cfg)
    # Large numbers of small BLAS solves can stall with an unrestricted thread
    # pool. Apply the configured limit to the entire numerical lifecycle.
    from .notebook_live import report_scope
    with numerical_threads(cfg['threads']),report_scope(live_report):
        return _run(cfg)


def _run(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    progress=Progress(root)
    from .notebook_live import current_report
    report=current_report();general_view=progress.view
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
    summaries,statuses,frozen_summaries,period_summaries=[],[],[],[]
    for number,spec in enumerate(specs,1):
        progress.view=general_view
        name = '__'.join([spec['group'], spec['scenario'], '+'.join(spec['sources']), spec['method'], spec['variant'], str(spec['horizon'])])
        progress.say(f'Эксперимент {number}/{len(specs)}: {name}',experiment=name)
        if any(c in name for c in '<>:"/\\|?*') or '..' in name:
            raise ValueError('Experiment names must be safe path components')
        dest = root/'experiments'/name; dest.mkdir(parents=True, exist_ok=True)
        (dest/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
        members = list(dict.fromkeys(spec['sources']+spec['recipients']))
        selected = align_contexts(contexts, members) if spec['scenario'] != 'local' else {u:contexts[u] for u in members}
        selected=transfer_preparation(selected,spec,cfg)
        if spec['variant'] == 'related' and not all(cfg['related_inputs'].get(c) for c in next(iter(selected.values()))['targets']):
            statuses.append(dict(experiment=name, status='requires_reviewed_related_inputs')); continue
        first = pd.Timestamp(cfg['first_test'])
        # Recipients cannot cause models to be trained beyond donor observation time.
        last = min(max(selected[u]['frame'].index[-1] for u in spec['recipients']),
                   min(selected[u]['frame'].index[-1] for u in spec['sources']))
        beginning=first
        if cfg['history_policy']['bootstrap']:
            reference=max(selected[u]['preparation_reference_end'] for u in spec['sources'])
            beginning=reference.to_period('M').to_timestamp()
            if beginning<reference:beginning+=pd.offsets.MonthBegin(1)
            beginning=min(first,beginning)
        months = [beginning]+list(pd.date_range(beginning+pd.offsets.MonthBegin(1), last, freq='MS'))
        if report:report.safely('start',name,spec,selected,months,progress)
        progress.say(f'Исторические признаки: с {beginning:%Y-%m-%d}; основной отчёт: с {first:%Y-%m-%d}')
        state = None
        pieces = {u:{} for u in spec['recipients']}
        frozen_pieces={u:{} for u in spec['recipients']}
        error_parts, feature_parts, audit_rows = {u:[] for u in pieces}, {u:[] for u in pieces}, []
        prediction_diagnostics={u:[] for u in pieces}
        for month_number,cutoff in enumerate(months,1):
            tag=f'{name} · месяц {month_number}/{len(months)} · {cutoff:%Y-%m}'
            with progress.stage(tag+' · проверка сохранённого состояния'):
                token,state_token = signatures(selected, cutoff, cfg, spec)
            path = dest/'checkpoints'/f'{cutoff:%Y-%m-%d}.joblib'
            restored=False;replace_old=False
            if path.exists():
                with progress.stage(tag+' · загрузка checkpoint'):
                    saved = joblib.load(path)
                    if checkpoint_compatible(saved,token,state_token,selected,cutoff,cfg,spec):
                        state = saved['state']; audit = dict(state['audit'], restored=True)
                        restored=True
                        if saved.get('fingerprint')!=token or saved.get('compatibility')!=state_token:
                            atomic_dump(dict(fingerprint=token,compatibility=state_token,state=state),path)
                            progress.say('Совместимый checkpoint восстановлен после обновления кода: '+str(path),
                                         status='checkpoint_migrated')
                    else:
                        replace_old=True
                        progress.say('Данные или параметры checkpoint изменились: пересчёт этого среза; '
                                     'старые модели будут сохранены. '+str(path),status='checkpoint_recompute')
            if not restored:
                recovery=initial_targets(dest/'partial'/f'{cutoff:%Y-%m-%d}',token,progress=progress)
                with recovery:
                    new_state, audit = update_at(selected, spec, cutoff, cfg, state,progress)
                if new_state is not None:
                    state = new_state
                    with progress.stage(tag+' · сохранение checkpoint'):
                        if replace_old:backup_checkpoint(path)
                        atomic_dump(dict(fingerprint=token,compatibility=state_token,state=state), path)
            audit=dict(audit)
            audit_rows.append(audit)
            progress.say(tag+' · модель: '+audit['status'],status=audit['status'])
            if state is None:
                progress.say('Модель пока не получена: '+json.dumps(audit,ensure_ascii=False,default=str))
                write_update_reports(dest,audit_rows)
                if report:report.safely('month',name,spec,month_number,pieces,selected,progress)
                continue
            end = cutoff+pd.offsets.MonthBegin(1)
            from .importance import write_regressor_importance
            importance_dir=dest/'feature_importance'/f'{cutoff:%Y-%m-%d}'
            older=sorted(p for p in importance_dir.parent.glob('*') if p.is_dir() and p.name<importance_dir.name)
            with progress.stage(tag+' · важность входных признаков'):
                write_regressor_importance(state['model'],importance_dir,token,older[-1] if older else None)
                import shutil
                for file in importance_dir.glob('feature_importance*.csv'):shutil.copy2(file,dest/file.name)
            for u in spec['recipients']:
                # Truncate inputs before scoring: no accidental future-dependent features.
                c = selected[u]
                from .runtime import export
                history=dest/u/'model_history'/f'{cutoff:%Y-%m}'
                current=dest/u/'model'
                export_token=hashlib.sha256((token+json.dumps({key:c.get(key) for key in
                    ('roles','aliases','declared_units','threshold','regime_spec')},sort_keys=True,ensure_ascii=False,default=str)).encode()).hexdigest()
                with progress.stage(tag+f' · {u} · экспорт модели для дашборда'):
                    if not export_ready(history,export_token):
                        export(state,c,spec,cfg,history,fingerprint=export_token,importance_dir=importance_dir)
                    if not export_ready(current,export_token):
                        publish_export(history,current)
                    record_export(root,name,u,spec,state,cutoff,current,history)
                progress.say('Готовая модель для дашборда: '+str(current.resolve()),model_dir=str(current.resolve()))
                cached=dest/'scored'/f'{cutoff:%Y-%m-%d}'/(u+'.joblib')
                with progress.stage(tag+f' · {u} · оценки месяца и признаки'):
                    score_token=score_fingerprint(c,end,state_token)
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
                for view,scores in month.get('frozen_views',{}).items():
                    frozen_pieces[u].setdefault(view,[]).append(scores)
                error_parts[u].append(month['errors']);feature_parts[u].append(month['features'])
            write_update_reports(dest,audit_rows)
            progress.say(tag+' · расчёт месяца завершён',status='month_complete')
            if report:
                with progress.stage(tag+' · обновление графика'):
                    report.safely('month',name,spec,month_number,pieces,selected,progress)
        write_update_reports(dest,audit_rows)
        if state is None:
            if report:report.safely('finish',name,spec,progress,available=False)
            statuses.append(dict(experiment=name, status='insufficient_history')); continue
        from .interpretation import write_catalog
        for u in spec['recipients']:
            ud = dest/u; ud.mkdir(exist_ok=True)
            full_index=selected[u]['frame'].index
            full_index=full_index[(full_index>=beginning)&(full_index<months[-1]+pd.offsets.MonthBegin(1))]
            if not len(full_index):
                statuses.append(dict(experiment=name,unit=u,status='no_recipient_observations_in_period'));continue
            timeline_features = pd.concat(feature_parts[u]).sort_index().reindex(full_index)
            residuals = pd.concat(error_parts[u]).sort_index().reindex(full_index)
            residuals.to_parquet(ud/'prequential_errors.parquet')
            pd.DataFrame(prediction_diagnostics[u]).to_csv(ud/'prediction_diagnostics.csv',index=False,encoding='utf-8-sig')
            with progress.stage(name+f' · {u} · тревоги, метрики и отчёты'):
                timelines = {}
                for view, parts in pieces[u].items():
                    data = pd.concat(parts).sort_index().reindex(full_index)
                    data['eligible']=selected[u]['eligible'].reindex(full_index).fillna(False)
                    data['background_eligible']=selected[u]['frame']['Состояние'].reindex(full_index).eq('Работа')
                    # Persistence remains continuous across month boundaries.
                    for b in cfg['alarm_budgets']:
                        threshold = data['threshold_budget_'+str(b)]
                        ratio = data.smoothed/threshold.where(np.isfinite(threshold))
                        stream, episodes = alarm_stream(ratio, view_policy(cfg,view))
                        scored = data.assign(ratio=ratio, active=stream.active, notification=stream.notification)
                        metrics, events = evaluate(scored, selected[u]['events'], cfg, start=first)
                        summaries.append(dict(experiment=name, unit=u, view=view, budget=b, **metrics))
                        period_summaries.extend(period_metrics(scored,selected[u]['events'],cfg,first,name,u,view,b,'adaptive'))
                        slug = view.replace(':','_')+'__'+str(b)
                        scored.to_parquet(ud/(slug+'.parquet'))
                        episodes.to_csv(ud/(slug+'_alarms.csv'), index=False, encoding='utf-8-sig')
                        events.to_csv(ud/(slug+'_events.csv'), index=False, encoding='utf-8-sig')
                        if b == cfg['alarm_budget']:
                            timelines[view] = scored
                from .risk_dataset import alarm_features
                timeline_features=alarm_features(timeline_features,timelines['mean::instant'],cfg)
                timeline_features.to_parquet(ud/'interpretation_features.parquet')
                write_catalog(timelines, timeline_features, selected[u], cfg, ud)
                frozen_folder=ud/'frozen';frozen_folder.mkdir(exist_ok=True)
                for view,parts in frozen_pieces[u].items():
                    data=pd.concat(parts).sort_index().reindex(full_index)
                    data['eligible']=selected[u]['eligible'].reindex(full_index).fillna(False)
                    data['background_eligible']=selected[u]['frame']['Состояние'].reindex(full_index).eq('Работа')
                    for budget in cfg['alarm_budgets']:
                        threshold=data['threshold_budget_'+str(budget)]
                        ratio=data.smoothed/threshold.where(np.isfinite(threshold))
                        stream,episodes=alarm_stream(ratio,view_policy(cfg,view))
                        scored=data.assign(ratio=ratio,active=stream.active,notification=stream.notification)
                        metrics,_=evaluate(scored,selected[u]['events'],cfg,start=first)
                        frozen_summaries.append(dict(experiment=name,unit=u,view=view,budget=budget,policy='frozen_initial',**metrics))
                        period_summaries.extend(period_metrics(scored,selected[u]['events'],cfg,first,name,u,view,budget,'frozen_initial'))
                        slug=view.replace(':','_')+'__'+str(budget)
                        scored.to_parquet(frozen_folder/(slug+'.parquet'))
                        episodes.to_csv(frozen_folder/(slug+'_alarms.csv'),index=False,encoding='utf-8-sig')
            statuses.append(dict(experiment=name, unit=u, status='complete', model=str(ud/'model')))
        pd.DataFrame(summaries).to_csv(root/'comparison.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(frozen_summaries).to_csv(root/'comparison_frozen.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(period_summaries).to_csv(root/'comparison_periods.csv',index=False,encoding='utf-8-sig')
        if report:report.safely('finish',name,spec,progress)
    pd.DataFrame(statuses).to_csv(root/'status.csv', index=False, encoding='utf-8-sig')
    from .comparison import common_comparison
    progress.view=general_view
    with progress.stage('итоговое сравнение на общих часах'):
        common_comparison(root,contexts,cfg)
    progress.say('Расчёт завершён. Список готовых моделей: '+str((root/'models.csv').resolve()),status='complete')
    return dict(summary=pd.DataFrame(summaries), status=pd.DataFrame(statuses))


def period_metrics(timeline,events,cfg,first,experiment,unit,view,budget,policy):
    rows=[];last=timeline.index[-1]+pd.Timedelta(cfg['step'])
    for months in (1,3,6,None):
        end=min(last,first+pd.DateOffset(months=months)) if months else last
        if end<=first:continue
        summary,_=evaluate(timeline,events,cfg,start=first,end=end)
        rows.append(dict(experiment=experiment,unit=unit,view=view,budget=budget,policy=policy,
                         period_months=months if months else 'all',period_start=str(first),period_end=str(end),**summary))
    return rows
