"""Human-readable readiness and causal risk annotations for confirmed alarms."""
import numpy as np
import pandas as pd
from .alarm_scores import alarm_stream,view_policy

REASONS={
    'no_observed_labelled_examples':'нет примеров с известным исходом',
    'training_requires_two_classes':'в обучении отсутствует один из классов',
    'insufficient_training_events':'не хватает независимых событий в обучении',
    'trained_uncalibrated':'модель обучена; для процентов не хватает калибровочных событий',
    'trained':'модель обучена, калибровка вероятности выполнена',
    'outside_eligible_risk_points':'нет пригодного рабочего окна/активной тревоги',
    'no_variable_inputs':'входы не меняются: обучение невозможно',
    'not_run':'классификатор ещё не запущен',
    'requires_reviewed_episode_labels':'для причин нужны подтверждённые метки эпизодов',
    'history_disabled':'помесячная проверка риска выключена',
    'historical_risk_not_computed':'история риска ещё не рассчитана',
    'outside_risk_history':'эта тревога вне рассчитанной истории риска',
    'requires_monthly_training':'нужно выполнить раздел 3 с помесячным обучением риска',
}


def readiness_rows(statuses,cfg):
    rows=[]
    for head,status in statuses.items():
        if not head.startswith('risk_'):continue
        audits=status.get('label_audit',[])
        if isinstance(audits,dict):audits=[audits]
        partitions=status.get('partitions',{})
        tr=partitions.get('deployment_train',{}).get('groups',{})
        cal=partitions.get('deployment_calibration',{}).get('groups',{})
        policy=cfg['risk_policy']
        measured=status.get('prequential',{})
        rows.append(dict(голова=head,статус=status.get('status','not_run'),
            область=status.get('scope',cfg['risk_policy']['scope']),котлы=', '.join(status.get('units',[])),
            обучение=status.get('fit_status','ожидание'),деревьев=status.get('trees',0),
            новых_обученных_строк=status.get('new_training_rows',0),
            precision_балла=measured.get('raw_score',{}).get('precision'),
            precision_вероятности=measured.get('probability',{}).get('precision'),
            пояснение=REASONS.get(status.get('status'),'см. подробный статус'),
            аварийных_групп_всего=sum(a.get('positive_events',0) for a in audits),
            зарегистрировано_аварий=sum(a.get('registered_emergencies',0) for a in audits),
            аварий_без_положительных_примеров=sum(a.get('emergencies_without_positive_examples',0) for a in audits),
            фоновых_групп_всего=sum(a.get('negative_groups',0) for a in audits),
            аварий_обучение=tr.get('1.0',0),фон_обучение=tr.get('0.0',0),
            аварий_калибровка=cal.get('1.0',0),фон_калибровка=cal.get('0.0',0),
            не_хватает_аварий_обучение=max(0,policy['min_positive_events']-tr.get('1.0',0)),
            не_хватает_фона_обучение=max(0,policy['min_negative_groups']-tr.get('0.0',0)),
            не_хватает_аварий_калибровка=max(0,policy['min_calibration_groups']-cal.get('1.0',0)),
            не_хватает_фона_калибровка=max(0,policy['min_calibration_groups']-cal.get('0.0',0)),
            исключено_неполное_будущее=sum(a.get('incomplete_future',0) for a in audits),
            исключено_пропуски_будущего=sum(a.get('unobserved_future',0) for a in audits),
            исключено_неаварийный_неизвестный_останов=sum(a.get('unknown_or_nonfailure_stop',0) for a in audits),
            исключено_слишком_поздно=sum(a.get('too_late_to_warn',0) for a in audits),
            историческая_проверка=status.get('test_status','unavailable')))
    return pd.DataFrame(rows)


def alarm_risk_table(timeline,risk,cfg,view='mean::instant',statuses=None):
    _,episodes=alarm_stream(timeline.ratio,view_policy(cfg,view))
    rows=[];risk=risk.copy() if risk is not None else pd.DataFrame()
    if len(risk):
        risk['time']=pd.to_datetime(risk.time)
        risk['model_cutoff']=pd.to_datetime(risk.model_cutoff)
        risk=risk.loc[risk.model_cutoff.le(risk.time)].sort_values('time')
    for number,ep in enumerate(episodes.itertuples(),1):
        row=dict(alarm=number,start=ep.start,confirmed=ep.confirmed,end=ep.end)
        texts=[]
        for hours in cfg['risk_horizons']:
            part=risk.loc[risk.horizon_hours.eq(hours)&risk.time.ge(ep.confirmed)&risk.time.lt(ep.end)] if len(risk) else risk
            prefix=f'risk_{hours}h';first=None
            if len(part):
                first=part.iloc[0]
                probability=part.loc[part.probability.notna()&part.probability_status.eq('calibrated_group_holdout')]
                raw=part.loc[part.raw_score.notna()]
            else:probability=raw=part
            row[prefix+'_at_confirmation']=(float(first.probability) if first is not None and first.time==ep.confirmed
                and first.probability_status=='calibrated_group_holdout' else np.nan)
            row[prefix+'_score_at_confirmation']=float(first.raw_score) if first is not None and first.time==ep.confirmed else np.nan
            if len(raw):
                row[prefix+'_first_score_time']=raw.iloc[0].time
                row[prefix+'_max_score']=float(raw.raw_score.max())
            if len(probability):
                p=probability.iloc[0]
                row[prefix+'_first_probability_time']=p.time
                row[prefix+'_first_probability']=float(p.probability)
                row[prefix+'_max_probability']=float(probability.probability.max())
                later=f'с {p.time:%d.%m %H:%M}: ' if p.time>ep.confirmed else ''
                text=f'{hours} ч: {later}{p.probability:.0%} → max {probability.probability.max():.0%}'
                if len(raw) and raw.iloc[0].time<p.time:text+=f' (раньше балл {raw.iloc[0].raw_score:.2f})'
            elif len(raw):
                r=raw.iloc[0]
                later=f'с {r.time:%d.%m %H:%M}: ' if r.time>ep.confirmed else ''
                text=f'{hours} ч: {later}балл {r.raw_score:.2f} → max {raw.raw_score.max():.2f}'
            else:
                reason=str(first.status) if first is not None else ('historical_risk_not_computed' if risk.empty else 'outside_risk_history')
                text=f'{hours} ч: нет оценки ({REASONS.get(reason,reason)})'
                row[prefix+'_unavailable_reason']=reason
            row[prefix+'_text']=text;texts.append(text)
        row['annotation']='\n'.join(texts);rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['alarm','start','confirmed','end','annotation'])
