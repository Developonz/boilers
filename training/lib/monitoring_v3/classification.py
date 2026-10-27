"""Chronological, grouped classifiers with separate fit/calibration/test readiness."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import uuid
import numpy as np
import pandas as pd
from .preparation import FrozenEncoder
from .progress import phase

CONTRACT=3


def probabilities(package,features,require_calibration=False,threads=None):
    options={'thread_count':threads} if threads is not None else {}
    raw=np.asarray(package['model'].predict_proba(package['encoder'].native(features),**options))
    calibrator=package.get('calibrator')
    if calibrator is not None:
        return calibrator.predict_proba(np.log(np.clip(raw,1e-9,1.))),list(calibrator.classes_)
    return (np.full_like(raw,np.nan) if require_calibration else raw),list(package['model'].classes_)


def _time(frame):
    return pd.DatetimeIndex(frame.index.get_level_values(0))


def _parts(x,y,g,hours,policy):
    times=_time(x);positive=y.astype(str).eq('1.0')
    starts=pd.Series(times[positive],index=g.loc[positive].to_numpy()).groupby(level=0).min().sort_values()
    # With a small event set, lack of a test event is a limitation of evaluation,
    # not a prohibition on training. Reserve the latest events where possible.
    if len(starts)>=policy['min_positive_events']+policy['min_calibration_groups']+1:
        a=starts.iloc[-policy['min_calibration_groups']-1];b=starts.iloc[-1]
    elif len(starts)>policy['min_positive_events']:
        n=min(policy['min_calibration_groups'],len(starts)-policy['min_positive_events'])
        a=starts.iloc[-n]
        b=max(starts.iloc[-1]+pd.Timedelta(hours=hours+48),times[int(len(times)*.8)])
        if b>=times.max():b=None
    elif len(starts):
        a=b=None
    else:
        unique=pd.DatetimeIndex(sorted(times.unique()))
        a=unique[int(len(unique)*.7)] if len(unique)>=30 else None
        b=unique[int(len(unique)*.9)] if len(unique)>=60 else None
    embargo=pd.Timedelta(hours=hours+48 if hours else 48)
    tr=np.asarray(times<a-embargo) if a is not None else np.ones(len(x),bool)
    cal=np.asarray(times>=a) if a is not None else np.zeros(len(x),bool)
    te=np.zeros(len(x),bool)
    if b is not None:
        cal&=np.asarray(times<b-embargo);te=np.asarray(times>=b)
    train_groups=set(g.loc[tr]);cal&=~g.isin(train_groups).to_numpy()
    te&=~g.isin(train_groups|set(g.loc[cal])).to_numpy()
    return tr,cal,te


def _describe(x,y,g,mask):
    subset=y.loc[mask].astype(str)
    return dict(rows=len(subset),classes=subset.value_counts().to_dict(),
        groups={c:int(g.loc[mask & y.astype(str).eq(c)].nunique()) for c in subset.unique()},
        first=str(_time(x.loc[mask]).min()) if mask.any() else None,
        last=str(_time(x.loc[mask]).max()) if mask.any() else None,
        event_ids=sorted(set(g.loc[mask & y.astype(str).eq('1.0')].astype(str))))


def _deployment_parts(x,y,g,policy,previous):
    """Permanent group holdout, while new non-held-out history keeps arriving.

    This is calibration of the present deployment model, NOT a claim of a
    chronological test. Historical quality is measured by fresh causal folds.
    """
    reserved=set(previous.get('calibration_group_ids',())) if previous else set()
    trained=set(previous.get('trained_group_ids',())) if previous else set()
    for label in sorted(y.astype(str).unique()):
        mask=y.astype(str).eq(label)
        order=pd.Series(_time(x)[mask],index=g.loc[mask]).groupby(level=0).min().sort_values()
        minimum=policy['min_positive_events'] if label=='1.0' else policy['min_negative_groups']
        already=set(order.index)&reserved
        possible=[key for key in order.index if key not in reserved and key not in trained]
        # Leave real independent training groups for every class. Once reserved,
        # a group never becomes a training group of this continuing head.
        amount=min(max(0,policy['min_calibration_groups']-len(already)),max(0,len(order)-len(already)-minimum))
        choices=possible[minimum:] if previous is None else possible
        if len(choices)<amount:choices=possible[-amount:] if amount else []
        reserved.update(choices[:amount])
        eligible=possible[minimum:] if previous is None else possible
        for key in eligible:
            if len(set(order.index)-reserved)<=minimum:break
            if int(hashlib.sha256(str(key).encode()).hexdigest()[:8],16)%5==0:
                reserved.add(key)
    cal=g.isin(reserved).to_numpy();tr=~cal
    return tr,cal,reserved


def _weights(y,g):
    # One event cannot acquire arbitrary influence from hundreds of nearby hours.
    w=1/g.map(g.value_counts()).to_numpy(float)
    labels=y.astype(str)
    totals=pd.Series(w,index=y.index).groupby(labels).sum()
    return w/labels.map(totals).to_numpy(float)*len(w)/len(totals)


def _replay_indices(y,g,limit,policy):
    labels=y.astype(str);classes=sorted(labels.unique())
    minimum=max(policy['min_positive_events'],policy['min_negative_groups'])
    limit=min(len(y),max(limit,len(classes)*minimum))
    strata=pd.DataFrame({'label':labels.to_numpy(),'group':g.to_numpy()}).groupby(['label','group'],sort=False).indices
    chosen=[]
    for label in classes:
        groups=sorted((positions for (kind,_),positions in strata.items() if kind==label),key=lambda a:a[0])
        count=min(len(groups),max(1,limit//len(classes)))
        for i in np.linspace(0,len(groups)-1,count).astype(int):
            positions=groups[i];chosen.append(int(positions[len(positions)//2]))
    chosen=np.unique(chosen)
    remaining=np.setdiff1d(np.arange(len(y)),chosen)
    amount=min(limit-len(chosen),len(remaining))
    if amount:chosen=np.r_[chosen,remaining[np.linspace(0,len(remaining)-1,amount).astype(int)]]
    return np.sort(chosen)


def _calibration_gap(x,train,cal,hours):
    """A different episode ID alone does not separate overlapping time windows."""
    units=x['unit'].astype(str) if 'unit' in x else pd.Series('',index=x.index)
    times=_time(x).as_unit('ns').asi8;gap=pd.Timedelta(hours=hours+48).value
    result=cal.copy()
    for unit in units.unique():
        same=units.eq(unit).to_numpy();fitted=np.sort(times[train&same])
        rows=np.flatnonzero(cal&same)
        if not len(fitted) or not len(rows):continue
        pos=np.searchsorted(fitted,times[rows]);left=np.maximum(0,pos-1);right=np.minimum(len(fitted)-1,pos)
        distance=np.minimum(np.abs(times[rows]-fitted[left]),np.abs(times[rows]-fitted[right]))
        result[rows[distance<=gap]]=False
    return result


def _model(cfg,iterations):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=iterations,loss_function='MultiClass',depth=cfg['cb_depth'],
        learning_rate=cfg['learning_rate'],random_seed=cfg['seed'],thread_count=cfg['threads'],
        has_time=True,allow_writing_files=False,verbose=False)


def _metrics(y,p,classes):
    from sklearn.metrics import log_loss,average_precision_score,brier_score_loss
    if not len(y):return dict(status='no_test_rows')
    labels=y.astype(str)
    result=dict(rows=len(y),classes=labels.value_counts().to_dict())
    if not set(labels)<=set(classes):return dict(**result,status='unseen_test_class')
    result['logloss']=float(log_loss(labels,p,labels=classes))
    if '1.0' in classes:
        score=p[:,classes.index('1.0')];target=labels.eq('1.0')
        result['brier']=float(brier_score_loss(target,score))
        result['average_precision']=float(average_precision_score(target,score)) if target.nunique()==2 else None
        result['recall_evaluable']=bool(target.any())
    result['status']='evaluated' if labels.nunique()>1 else 'single_class_test'
    return result


def _save_head(package,dest,archive=True):
    import joblib
    target=Path(dest)/'interpreter.joblib';tmp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    joblib.dump(package,tmp,compress=3)
    if archive and target.exists():
        backups=Path(dest)/'previous_versions';backups.mkdir(exist_ok=True)
        shutil.copy2(target,backups/(uuid.uuid4().hex+'.joblib'))
    os.replace(tmp,target)


def _temporal_check(x,y,g,available,train,test,deadline,cfg):
    selected=train&available.le(deadline).to_numpy()
    detail=dict(train_before=str(deadline),training_rows=int(selected.sum()),
        late_labels_excluded=int((train&~selected).sum()),probability_status='raw_model_scores',
        max_training_label_available_at=str(available.loc[selected].max()) if selected.any() else None)
    if y.loc[selected].nunique()<2:return dict(detail,status='training_requires_two_classes')
    counts={c:int(g.loc[selected&y.astype(str).eq(c)].nunique()) for c in y.loc[selected].astype(str).unique()}
    detail['training_groups_by_class']=counts
    if any(n<(cfg['risk_policy']['min_positive_events'] if c=='1.0' else cfg['risk_policy']['min_negative_groups']) for c,n in counts.items()):
        return dict(detail,status='insufficient_training_events')
    if not test.any():return dict(detail,status='no_test_rows')
    enc=FrozenEncoder.fit(x.loc[selected]);native=enc.native(x.loc[selected])
    if not any(native[c].nunique(dropna=False)>1 for c in native):return dict(detail,status='no_variable_inputs')
    try:
        model=_model(cfg,cfg['cb_initial_trees'])
        model.fit(native,y.loc[selected].astype(str),cat_features=list(enc.categories),sample_weight=_weights(y.loc[selected],g.loc[selected]))
        p=model.predict_proba(enc.native(x.loc[test]))
        return dict(detail,**_metrics(y.loc[test],p,list(model.classes_)))
    except Exception as exc:
        return dict(detail,status='evaluation_failed',problem=type(exc).__name__+': '+str(exc))


def train_classifier(features,labels,groups,cfg,dest,previous=None,*,horizon_hours=0,task='cause',label_available_at=None,evaluate=True):
    from .threads import numerical_threads
    with numerical_threads(cfg['threads']):
        return _train_classifier(features,labels,groups,cfg,dest,previous,horizon_hours=horizon_hours,task=task,label_available_at=label_available_at,evaluate=evaluate)


def _train_classifier(features,labels,groups,cfg,dest,previous=None,*,horizon_hours=0,task='cause',label_available_at=None,evaluate=True):
    import joblib
    from sklearn.linear_model import LogisticRegression
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    good=labels.notna()&features.notna().any(axis=1)
    x=features.loc[good].sort_index();y=labels.reindex(x.index);g=groups.reindex(x.index).astype(str)
    available=(pd.to_datetime(label_available_at.reindex(x.index)) if label_available_at is not None else
        pd.Series(_time(x)+pd.Timedelta(hours=horizon_hours),index=x.index))
    known=available.notna();x=x.loc[known];y=y.loc[known];g=g.loc[known];available=available.loc[known]
    policy=cfg['risk_policy'];report=dict(task=task,rows=len(x),groups=int(g.nunique()),
        horizon_hours=horizon_hours,contract=CONTRACT,probability_status='unavailable',test_status='unavailable')
    def finish(status):
        report['status']=status
        (dest/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        return report
    if not len(x):return finish('no_observed_labelled_examples')
    eval_tr,eval_cal,eval_te=_parts(x,y,g,horizon_hours,policy)
    ids=pd.Series([f'{idx!s}::{label}' for idx,label in zip(x.index,y.astype(str))],index=x.index)
    signature=hashlib.sha256(json.dumps([CONTRACT,task,horizon_hours,list(x)],ensure_ascii=False).encode()).hexdigest()
    if previous and previous.get('feature_contract')!=signature:
        previous=None;report['migration']='new_feature_or_label_contract'
    fingerprints=pd.util.hash_pandas_object(x,index=True).astype(str)
    row_hash={str(idx):hashlib.sha256((str(value)+'::'+str(label)+'::'+str(group)+'::'+str(date)).encode()).hexdigest()
              for idx,value,label,group,date in zip(x.index,fingerprints,y.astype(str),g,available)}
    if previous and any(key not in row_hash or row_hash[key]!=value for key,value in previous.get('row_fingerprints',{}).items()):
        previous=None;report['migration']='historical_features_labels_or_groups_revised'
    if previous and previous.get('classes')!=sorted(y.astype(str).unique()):
        previous=None;report['migration']='class_schema_changed'
    tr,cal,reserved=_deployment_parts(x,y,g,policy,previous)
    # Protect the independent training minima first. Then purge neighbouring
    # training windows around the held-out groups, rather than making every
    # positive calibration window unusable because of nearby negative hours.
    protected=ids.isin(previous.get('ingested',())).to_numpy() if previous else np.zeros(len(x),bool)
    if previous is None:
        for label in sorted(y.astype(str).unique()):
            selected=y.astype(str).eq(label)
            order=pd.Series(_time(x)[selected],index=g.loc[selected]).groupby(level=0).min().sort_values()
            minimum=policy['min_positive_events'] if label=='1.0' else policy['min_negative_groups']
            protected|=g.isin(set(order.index[:minimum])).to_numpy()
    reserved_rows=int(cal.sum());cal=_calibration_gap(x,protected,cal,horizon_hours)
    report['calibration_overlap_excluded']=reserved_rows-int(cal.sum())
    before=int(tr.sum());tr=_calibration_gap(x,cal,tr,horizon_hours)
    report['training_overlap_excluded']=before-int(tr.sum())
    report['partitions']={name:_describe(x,y,g,mask) for name,mask in [('deployment_train',tr),('deployment_calibration',cal)]}
    eval_deadline=_time(x.loc[eval_cal|eval_te]).min() if (eval_cal|eval_te).any() else None
    known_eval_train=eval_tr&available.le(eval_deadline).to_numpy() if eval_deadline is not None else eval_tr
    report['evaluation_partitions']={name:_describe(x,y,g,mask) for name,mask in [('train',known_eval_train),('calibration',eval_cal),('test',eval_te)]}
    classes=sorted(y.loc[tr].astype(str).unique())
    if len(classes)<2:return finish('training_requires_two_classes')
    minimum={c:policy['min_positive_events'] if c=='1.0' else policy['min_negative_groups'] for c in classes}
    missing={c:minimum[c]-g.loc[tr & y.astype(str).eq(c)].nunique() for c in classes}
    if any(v>0 for v in missing.values()):
        report['missing_training_groups']={c:int(v) for c,v in missing.items() if v>0}
        return finish('insufficient_training_events')
    if previous and previous.get('classes')!=classes:
        previous=None;report['migration']='class_schema_changed'
    encoder=previous['encoder'] if previous else FrozenEncoder.fit(x.loc[tr])
    fit=tr & ~ids.isin(previous.get('ingested',())).to_numpy() if previous else tr
    if previous and not fit.any():
        model=previous['model'];report['fit_status']='no_new_labels'
    elif previous and previous['model'].tree_count_+cfg['cb_update_trees']>cfg['cb_max_trees']:
        model=previous['model'];fit=np.zeros(len(x),bool)
        report['fit_status']='tree_capacity_reached_previous_retained'
    else:
        xf,yf,gf=x.loc[fit],y.loc[fit].astype(str),g.loc[fit]
        if previous:
            replay=previous['replay'];xf=pd.concat([replay[0],xf]).sort_index()
            yf=pd.concat([replay[1],yf]).reindex(xf.index)
            gf=pd.concat([replay[2],gf]).reindex(xf.index)
        model=_model(cfg,cfg['cb_update_trees'] if previous else cfg['cb_initial_trees'])
        native=encoder.native(xf)
        if not any(native[c].nunique(dropna=False)>1 for c in native):return finish('no_variable_inputs')
        with phase(f'обучение рабочей головы: {len(xf)} строк, {len(xf.columns)} входов'):
            from .continuation import fit_continued
            fit_continued(model,native,yf,previous['model'] if previous else None,cat_features=list(encoder.categories),
                      sample_weight=_weights(yf,gf))
    calibrator=None
    cal_groups={c:g.loc[cal & y.astype(str).eq(c)].nunique() for c in classes}
    if all(cal_groups[c]>=policy['min_calibration_groups'] for c in classes) and set(y.loc[cal].astype(str))==set(classes):
        raw=np.log(np.clip(model.predict_proba(encoder.native(x.loc[cal])),1e-9,1.))
        # Group reservation is stratified. Correct its class prior back to the
        # observed decision-point prevalence; never balance calibration classes.
        priors=y.astype(str).value_counts()/y.loc[cal].astype(str).value_counts()
        weight=y.loc[cal].astype(str).map(priors).to_numpy(float)
        weight=weight/weight.mean()
        calibrator=LogisticRegression(C=1.,max_iter=1000).fit(raw,y.loc[cal].astype(str),sample_weight=weight)
        report['probability_status']='calibrated_group_holdout'
        report['calibration_prior_correction']={c:float(priors[c]) for c in classes}
    else:
        report['probability_status']='uncalibrated_score_only'
        report['calibration_groups_by_class']={c:int(v) for c,v in cal_groups.items()}
    actual_fit=fit|ids.isin(previous.get('ingested',()) if previous else ()).to_numpy()
    report['partitions']['deployment_train']=_describe(x,y,g,actual_fit)
    report['pending_training_rows']=int((tr&~actual_fit).sum())
    through=_time(x.loc[fit]).max() if fit.any() else pd.Timestamp(previous['trained_through'])
    if previous:through=max(through,pd.Timestamp(previous['trained_through']))
    package=dict(contract=CONTRACT,feature_contract=signature,model=model,encoder=encoder,
        calibration_group_ids=sorted(reserved),trained_group_ids=sorted(set(g.loc[fit])|set(previous.get('trained_group_ids',()) if previous else ())),
        row_fingerprints={**(previous.get('row_fingerprints',{}) if previous else {}),
                          **{str(idx):row_hash[str(idx)] for idx in x.index[fit]}},
        calibrator=calibrator,classes=classes,threshold=policy['confidence'],task=task,
        risk_policy=dict(policy),
        horizon_hours=horizon_hours,trained_through=str(through),
        training_units=sorted(features['unit'].dropna().astype(str).unique()) if 'unit' in features else [],
        ingested=sorted(set(ids.loc[fit])|set(previous.get('ingested',()) if previous else ())))
    keep=_replay_indices(y.loc[actual_fit],g.loc[actual_fit],cfg['replay_rows'],policy)
    package['replay']=(x.loc[actual_fit].iloc[keep],y.loc[actual_fit].astype(str).iloc[keep],g.loc[actual_fit].iloc[keep])
    report['status']='trained' if calibrator is not None else 'trained_uncalibrated'
    report['max_training_label_available_at']=str(available.loc[actual_fit].max())
    report['replay_rows']=len(keep);report['replay_classes']=package['replay'][1].value_counts().to_dict()
    report['test_status']='pending';package['report']=report
    with phase('сохранение рабочей головы до дополнительных проверок'):_save_head(package,dest)
    finish(report['status'])
    if not evaluate:
        report['test_status']='not_requested_for_causal_monthly_replay'
        package['report']=report
        _save_head(package,dest,archive=False)
        finish(report['status'])
        return report
    if eval_te.any():
        # The deployment model knows newer examples and must NEVER report its
        # own fit on a historical test as if it were a prediction from the past.
        deadline=_time(x.loc[eval_cal|eval_te]).min()
        with phase('независимая временная проверка классификатора'):
            report['test']=_temporal_check(x,y,g,available,eval_tr,eval_te,deadline,cfg)
        report['test']['model']='independent_chronological_evaluation_model'
        report['test_status']=report['test']['status']
    else:report['test_status']='no_test_rows'
    # At most three causal forward checks; test examples do not train their fold.
    if task=='risk':
        starts=pd.Series(_time(x)[y.astype(str).eq('1.0')],index=g.loc[y.astype(str).eq('1.0')]).groupby(level=0).min().sort_values()
        folds=[]
        for boundary in starts.iloc[policy['min_positive_events']:].tail(3):
            before=np.asarray(_time(x)<boundary-pd.Timedelta(hours=horizon_hours+48))
            after=np.asarray(_time(x)>=boundary)&~g.isin(set(g.loc[before])).to_numpy()
            usable=before&available.le(boundary).to_numpy()
            if y.loc[usable].nunique()<2 or any(g.loc[usable & y.astype(str).eq(c)].nunique()<minimum[c] for c in classes):
                folds.append(dict(status='insufficient_labels_known_at_boundary',test_from=str(boundary),
                    late_labels_excluded=int((before&~usable).sum())));continue
            with phase(f'временная проверка риска: обучение до {boundary:%Y-%m-%d}'):
                entry=_temporal_check(x,y,g,available,before,after,boundary,cfg)
            entry.update(feature_train_before=str(boundary-pd.Timedelta(hours=horizon_hours+48)),test_from=str(boundary),
                         train_events=int(g.loc[usable & y.astype(str).eq('1.0')].nunique()))
            folds.append(entry)
        report['forward_checks']=folds
    report['train_groups']=int(g.loc[actual_fit].nunique());report['calibration_groups']=int(g.loc[cal].nunique());report['test_groups']=int(g.loc[eval_te].nunique())
    report['status']='trained' if calibrator is not None else 'trained_uncalibrated'
    report['evidence']='chronological grouped observations; see partitions and forward_checks'
    from .importance import write_classifier_importance
    with phase('важность признаков классификатора'):
        write_classifier_importance(model,dest,cfg['threads'])
    package['report']=report
    _save_head(package,dest,archive=False)
    return finish(report['status'])
