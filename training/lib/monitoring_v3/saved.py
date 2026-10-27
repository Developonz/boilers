"""Frozen model application to a recipient, with schema and artifact integrity checks."""
from pathlib import Path
import json
import uuid
import numpy as np
import pandas as pd


def run_saved(model_dir, data_path, target_unit, *, sensor_file, period_start=None, period_end=None,
              allow_before_cutoff=False, require_unseen_unit=False, warmup_hours=None,
              output_dir='saved_model_checks', allow_missing_targets=False, threads=2):
    import legacy_group as G
    from saved_model_runner import _read_grid, _map, _fingerprint, _runtime
    from .threads import numerical_threads
    folder = Path(model_dir).resolve()
    output = Path(output_dir).resolve()
    if output == folder or output.is_relative_to(folder):
        raise ValueError('Outputs must not be written into the model export')
    before = _fingerprint(folder)
    metadata = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if require_unseen_unit and target_unit in metadata['training_units']:
        raise ValueError('Recipient participated in model training')
    with _runtime(folder, 'incremental') as (model, _), numerical_threads(threads):
        import k2_lib as K
        rows = G.read_roles(sensor_file, target_unit)
        raw=model.normalize_input(K.read_table(str(data_path)),roles=rows)
        # Align the recipient's load role by physical steam-flow role, with unit validation.
        training = [{**r, 'role':model.roles.get(r['key'], r['role'])} for r in rows]
        recipient_load = [r for r in training if r['role'] == 'load']
        if len(recipient_load) != 1:
            raise ValueError('Exactly one recipient load channel required')
        row=recipient_load[0]
        expected_unit=model.declared_units.get(model.cfg['run_col'],'')
        if expected_unit and row.get('units') and expected_unit.strip().lower()!=row['units'].strip().lower():
            raise ValueError('Recipient load units differ from the training units')
        if row['key'] != model.cfg['run_col']:
            names=[G.canonical_name(row['original']),G.canonical_name(model.source_load_name)]
            if not all('расход' in n and 'пар' in n for n in names):
                raise ValueError('Different load channels require explicit physical-role agreement')
            training = [dict(r,key=model.cfg['run_col']) if r['role']=='load' else r for r in training]
        frame, audit = _map(raw, model.roles, target_unit, training, G.canonical_name, {})
        missing = audit.status.isin(['missing','empty'])
        if (missing & (~audit.role.eq('target') | (not allow_missing_targets))).any():
            raise ValueError('Required recipient channels are missing: '+str(audit.loc[missing,'key'].tolist()))
        start = pd.Timestamp(period_start) if period_start else model.state['cutoff']
        end = pd.Timestamp(period_end) if period_end else frame.index[-1]+pd.Timedelta(model.config['step'])
        if start < model.state['cutoff'] and not allow_before_cutoff:
            raise ValueError('Evaluation before model cutoff requires allow_before_cutoff=True')
        if start >= end or start > frame.index[-1]:
            raise ValueError('No evaluation interval after the model cutoff')
        frame = frame.loc[frame.index < end]
        if warmup_hours is not None:
            frame = frame.loc[frame.index >= start-pd.Timedelta(hours=warmup_hours)]
        # Strict transfer uses donor operating thresholds. Recipient identity is a feature, never a fit.
        model.unit = target_unit
        result = model.score(frame, prepared=True)
        dest = output/(target_unit+'_'+uuid.uuid4().hex[:10]); dest.mkdir(parents=True)
        audit.to_csv(dest/'sensor_audit.csv', index=False, encoding='utf-8-sig')
        frames, matrices, alarms, summaries = {}, {}, [], []
        for view in result.hourly.columns.get_level_values(0).unique():
            d = result.hourly[view]
            local = pd.DataFrame(dict(score=d['ошибка'],smoothed=d['сглаженная'],threshold=d['порог'],
                ratio=d['отношение'],alarm=d['тревога'],notification=d['уведомление'],
                eligible=result.mode.eq('работа'),coverage=d['покрытие']),index=d.index)
            local = local.loc[(local.index >= start)&(local.index < end)]
            frames[view], matrices[view] = local, result.sensor_errors[view].reindex(local.index)
            ep = result.alarms.loc[result.alarms['вид'].eq(view)]
            for r in ep.to_dict('records'):
                if r['начало'] < end and r['конец'] > start:
                    alarms.append(dict(view=view,start=r['начало'],end=r['конец'],hours=r['часов'],peak=r['пик']))
            hours = float(local.score.notna().sum()*pd.Timedelta(model.config['step'])/pd.Timedelta('1h'))
            summaries.append(dict(view=view,evaluated_hours=hours,availability=float(local.score.notna().mean()),
                                  notifications_in_period=int(local.notification.sum()),
                                  alarm_hours=float(local.alarm.sum()*pd.Timedelta(model.config['step'])/pd.Timedelta('1h'))))
        summary = pd.DataFrame(summaries)
        summary.to_csv(dest/'summary.csv',index=False,encoding='utf-8-sig')
        pd.concat(frames,names=['view','time']).to_parquet(dest/'timeline.parquet')
        episode_table = pd.DataFrame(alarms,columns=['view','start','end','hours','peak'])
        episode_table.to_csv(dest/'alarms.csv',index=False,encoding='utf-8-sig')
        result.interpretation.loc[(result.interpretation.index >= start)&(result.interpretation.index < end)].to_csv(dest/'interpretation.csv')
        after = _fingerprint(folder)
        if before != after:
            raise RuntimeError('Inference modified the model export')
        info = dict(format='incremental',sources=metadata['training_units'],target_unit=target_unit,
                    period_start=str(start),period_end_exclusive=str(end),model_files_unchanged=True,
                    threshold_policy='saved donor threshold; no recipient calibration',model_sha256=before)
        (dest/'run.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        return dict(root=dest,metadata=info,audit=audit,scores=frames,contributions=matrices,summary=summary,alarms=episode_table)
