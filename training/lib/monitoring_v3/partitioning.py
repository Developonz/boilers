"""Persistent out-of-training blocks; empty recent calendar windows never gate fit."""
import numpy as np
import pandas as pd


def partition_history(features, cfg, previous=None):
    ix=features.index
    times=pd.DatetimeIndex(ix.get_level_values('time'))
    hp=cfg['history_policy']
    block=times.as_unit('ns').asi8//pd.Timedelta(hours=hp['block_hours']).value
    ingested=set(previous.get('ingested',())) if previous else set()
    reserved_cal=set(previous.get('reserved_calibration',())) if previous else set()
    reserved_val=set(previous.get('reserved_validation',())) if previous else set()
    seen=np.array([key in ingested for key in ix],dtype=bool)
    order=np.argsort(times.as_unit('ns').asi8,kind='stable')
    ordered=times.as_unit('ns').asi8[order]
    blocks=np.split(order,np.flatnonzero(np.diff(ordered))+1) if len(order) else []
    seen_time=pd.Index(times).isin({key[0] for key in ingested})
    # Promotion/rebalancing must move complete timestamps of the pooled model,
    # even when the requested row count is not divisible by the number of units.
    def groups(mask,latest=False):
        return [rows for rows in (reversed(blocks) if latest else blocks) if len(rows) and mask[rows].all()]
    def move(source,target,need,keep=0,latest=False):
        moved=0;remaining=int(source.sum())
        for rows in groups(source&~seen,latest):
            if moved>=need:break
            if remaining-len(rows)<keep:continue
            source[rows]=False;target[rows]=True;moved+=len(rows);remaining-=len(rows)
        return moved
    cal=((block%hp['calibration_every'])==hp['calibration_every']-1)&~seen_time
    cal|=pd.Index(times).isin({key[0] for key in reserved_cal})
    val=((block%hp['validation_every'])==1)&~cal&~seen_time if cfg['validate_candidate'] and previous else np.zeros(len(ix),bool)
    val|=pd.Index(times).isin({key[0] for key in reserved_val})&~cal
    train=~(cal|val)
    needed=max(cfg['calibration_rows'],cfg['min_sensor_rows'])
    committed=pd.Index(times).isin({key[0] for key in reserved_cal})
    # If all operating campaigns happened to land on scheduled holdout weeks,
    # rebalance BEFORE the first fit. Previously reserved/ingested rows never move.
    shortage=max(0,cfg['min_train_rows']-int(train.sum()))
    releasable=cal&~committed&~seen
    if shortage:
        for rows in groups(releasable):
            if train.sum()>=cfg['min_train_rows']:break
            if cal.sum()-len(rows)<needed:continue
            cal[rows]=False;train[rows]=True
    if previous and not (train&~seen).any() and releasable.any():
        goal=max(1,int(releasable.sum())*2//3);moved=0
        for rows in groups(releasable):
            if moved>=goal:break
            if cal.sum()-len(rows)<needed:continue
            cal[rows]=False;train[rows]=True;moved+=len(rows)
    # Promote unused rows when scheduled holdout blocks fell in a shutdown. This
    # manifest persists: a reserved calibration hour can NEVER train a later model.
    deficit=max(0,needed-int(cal.sum()))
    if deficit:move(train,cal,deficit,keep=cfg['min_train_rows'] if previous is None else 1,latest=True)
    validation_target=max(cfg['min_sensor_rows'],cfg['validate_rows'])
    if previous and cfg['validate_candidate'] and val.sum()<validation_target:
        # Always leave at least one new row to update; a genuinely unavailable
        # validation set is reported separately rather than called "no history".
        move(train,val,validation_target-int(val.sum()),keep=int((train&seen).sum())+1,latest=True)
    reserved_cal.update(ix[cal].tolist());reserved_val.update(ix[val].tolist())
    selected_val=np.flatnonzero(val)
    if len(selected_val)>validation_target:
        chosen=groups(val,latest=True);val[:]=False
        for rows in chosen:
            if val.sum()>=validation_target:break
            val[rows]=True
    # Keep complete time ordering and explicit gaps when scoring these rows.
    selected=np.flatnonzero(cal)
    if len(selected)>cfg['calibration_limit']:
        chosen=groups(cal,latest=True);cal[:]=False
        for rows in chosen:
            if cal.sum()+len(rows)>cfg['calibration_limit'] and cal.sum()>=needed:break
            cal[rows]=True
    audit=dict(train=int(train.sum()),validation=int(val.sum()),calibration=int(cal.sum()),
               available_healthy_rows=len(ix),new_train_rows=int((train&~seen).sum()),
               partition_policy='persistent_past_blocks_v2',
               validation_required=bool(previous and cfg['validate_candidate']))
    return train,val,cal,reserved_cal,reserved_val,audit
