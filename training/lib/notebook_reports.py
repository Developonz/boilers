"""Notebook actions and embedded reports; no shell commands or separate processes."""
from copy import deepcopy
from html import escape
from io import BytesIO
from pathlib import Path
import json
import math
import os
from numbers import Real
import pandas as pd


def read_report(path):
    """An empty optional report is a valid result, including a CRLF-only CSV."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def show_table(title, frame, opened=False):
    from IPython.display import display, HTML
    content = frame.to_html(index=False, na_rep='—', float_format=lambda v:f'{v:.3f}', escape=True)
    display(HTML(f'<details{" open" if opened else ""}><summary><b>{escape(title)}</b> '
                 f'({len(frame)} строк)</summary>{content}</details>'))


def configure(cfg, rule_overrides=None):
    """Persist only effective notebook rules; keep base files and model semantics intact."""
    from monitoring_v3.config import load
    from monitoring_v3.data import read_rules
    cfg = load(overrides=cfg)
    if rule_overrides and any(rule_overrides.get(k) for k in ('defaults','sensors')):
        rules = deepcopy(read_rules(cfg['sensor_rules']))
        rules['defaults'] = {**rules.get('defaults',{}), **rule_overrides.get('defaults',{})}
        rules.setdefault('sensors',{})
        for sensor, options in rule_overrides.get('sensors',{}).items():
            rules['sensors'][sensor] = {**rules['sensors'].get(sensor,{}), **options}
        dest = Path(cfg['output'])/'notebook_settings'/'sensor_rules.json'
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(rules,ensure_ascii=False,indent=2),encoding='utf-8')
        cfg['sensor_rules'] = str(dest.resolve())
    return cfg


def notebook_config(settings=None, rule_overrides=None):
    """Reload the base file and apply notebook overrides before each preparation."""
    from monitoring_v3.config import load
    cfg = load(os.environ.get('BOILER_CONFIG_JSON', 'incremental_settings.json'))
    cfg.update(settings or {})
    thresholds = cfg.get('run_thresholds', {})
    if not isinstance(thresholds, dict):
        raise ValueError('run_thresholds должен быть словарём: {"К-5": 140.0}')
    for unit, value in thresholds.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not math.isfinite(float(value))):
            raise ValueError(f'run_thresholds[{unit!r}]: укажите конечное число, например 140.0')
    return configure(cfg, rule_overrides)


def check_preparation_thresholds(contexts, cfg):
    """Never present an old context as if it used the currently selected threshold."""
    from monitoring_v3.preparation import run_threshold
    for unit, c in contexts.items():
        selected = cfg['run_thresholds'].get(unit)
        expected = (float(selected) if selected is not None else
                    run_threshold(c['frame'][c['load']], c['frame'].get('Состояние'),
                                  c.get('preparation_reference_end',cfg['first_test']), cfg['threshold_fraction']))
        if not math.isclose(float(c['threshold']), expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f'{unit}: текущий порог работы {expected:g}, но данные подготовлены '
                             f'с порогом {c["threshold"]:g}. Повторите ячейку «Данные, режимы '
                             'и события до обучения».')


def show_preparation(contexts, issues, cfg):
    from IPython.display import display, Image
    from monitoring_v3.data import healthy
    from incremental_plots import plot_preparation
    import matplotlib.pyplot as plt
    check_preparation_thresholds(contexts, cfg)
    show_table('Проблемы источников',pd.DataFrame(issues,columns=['unit','problem','path']).rename(
        columns={'unit':'котёл','problem':'проблема','path':'путь'}),bool(issues))
    active={unit for units in cfg['groups'].values() for unit in units}
    unused=[{'котёл в run_thresholds':unit,'порог':value,'причина':'котла нет в groups'}
            for unit,value in cfg['run_thresholds'].items() if unit not in active]
    if unused:
        show_table('Неиспользуемые пороги: проверьте имена котлов',pd.DataFrame(unused),True)
    step_h = float(pd.Timedelta(cfg['step'])/pd.Timedelta('1h'))
    summary, modes, events = [], [], []
    for unit,c in contexts.items():
        summary.append({'котёл':unit,'датчик нагрузки':c['load'],'единица':c['declared_units'].get(c['load'],''),
            'задан в run_thresholds':cfg['run_thresholds'].get(unit) if cfg['run_thresholds'].get(unit) is not None else 'авто',
            'порог работы':c['threshold'],'источник порога':c['threshold_source'],
            'начало истории':c['frame'].index[0],'конец истории':c['frame'].index[-1],
            'ошибочных дат':c['invalid_timestamps'],'дублей времени':c['duplicate_timestamps'],
            'допущено к диагностике, ч':c['eligible'].sum()*step_h,
            'норма для первого обучения, ч':healthy(c,cfg['first_test'],cfg).sum()*step_h,
            'отмечена работа при нулевой нагрузке, ч':(
                c['frame']['Состояние'].eq('Работа') & c['regime']['mode'].eq('стоит')).sum()*step_h})
        for mode,count in c['regime']['mode'].value_counts().items():
            modes.append({'котёл':unit,'режим':mode,'часов':count*step_h})
        e=c['events'].copy();e.insert(0,'котёл',unit);events.append(e)
    show_table('Источники и настройки определения работы',pd.DataFrame(summary),True)
    show_table('Часы по режимам',pd.DataFrame(modes),True)
    event_table=pd.concat(events,ignore_index=True) if events else pd.DataFrame()
    show_table('Реестр остановов, ремонтов и резерва',event_table.rename(columns={
        'event_id':'идентификатор','start':'начало отметки','stop':'останов по нагрузке','end':'конец отметки',
        'type':'тип','reason':'причина','source':'источник','label_available_at':'метка доступна с',
        'verified':'подтверждено','open_ended':'наблюдение не завершено','issue':'требует проверки'}),True)
    for unit,c in contexts.items():
        show_table(unit+' · сопоставление датчиков',c['audit'])
        show_table(unit+' · токи приводов: группы и пороги',pd.DataFrame(c.get('regime_spec',{}).get('audit',[])))
        numeric=c['frame'][c['targets']]
        stats=numeric.agg(['count','nunique','min','max']).T.rename_axis('датчик').reset_index()
        stats['доля пропусков']=numeric.isna().mean().to_numpy()
        for flag in ('spike','frozen','masked'):
            stats[flag+', ч']=c['quality'].xs(flag,axis=1,level=1).sum().reindex(numeric.columns).to_numpy()*step_h
        show_table(unit+' · доступность и качество измерений',stats)
        figure=plot_preparation(c)
        buffer=BytesIO()
        try:
            figure.savefig(buffer,format='png',dpi=130)
            display(Image(data=buffer.getvalue()))
        finally:
            plt.close(figure)
    return event_table


def experiment_directories(cfg):
    rows=[]
    for meta in sorted((Path(cfg['output'])/'experiments').glob('*/experiment.json')):
        spec=json.loads(meta.read_text(encoding='utf-8'))
        for unit in spec['recipients']:
            directory=meta.parent/unit
            if (directory/'prequential_errors.parquet').exists():
                rows.append(dict(experiment=meta.parent.name,unit=unit,method=spec['method'],
                                 scenario=spec['scenario'],variant=spec['variant'],directory=directory))
    return rows


def show_results(cfg, result):
    show_table('Состояние расчёта',result['status'],True)
    exports=Path(cfg['output'])/'models.csv'
    if exports.exists():
        show_table('Готовые модели для дашборда: model_dir → MODEL_DIR',read_report(exports),True)
    show_table('Качество моделей и схем тревог',result['summary'],True)
    for file,title in [('comparison_frozen.csv','Замороженная первая модель: будущий период'),
                       ('comparison_periods.csv','Дообучение и замороженная модель: 1, 3, 6 месяцев и вся история')]:
        path=Path(cfg['output'])/file
        if path.exists():show_table(title,read_report(path))
    common=Path(cfg['output'])/'comparison_common_hours.csv'
    if common.exists() and common.stat().st_size>1:
        show_table('Сравнение на общих рассчитанных часах',read_report(common))
    for item in experiment_directories(cfg):
        directory=item['directory'];name=item['experiment']+' · '+item['unit']
        updates=directory.parent/'updates.json'
        importance=directory.parent/'feature_importance_summary.csv'
        if importance.exists():
            show_table(name+' · важность входов: средняя по обученным целям, первые 25',read_report(importance).head(25))
        if updates.exists():
            table=pd.DataFrame(json.loads(updates.read_text(encoding='utf-8')))
            show_table(name+' · месячные обновления',table.drop(columns=['trees','target_errors','calibration_by_unit','calibration_sensor_diagnostics','admission_by_unit'],errors='ignore'))
        for filename,title in [('admission_diagnostics.csv','допуск истории по месяцам: где отсеялись часы'),
                               ('calibration_diagnostics.csv','пороги: история, точность бюджета и неопределённость'),
                               ('calibration_sensor_diagnostics.csv','ошибки датчиков на отложенной норме и сравнение с константой'),
                               ('target_update_diagnostics.csv','проверка обновления по отдельным датчикам')]:
            report=directory.parent/filename
            if report.exists():show_table(name+' · '+title,read_report(report))
        for filename,title in [('prediction_diagnostics.csv','качество прогноза'),
                               ('episode_labels.csv','эпизоды и метки смысла'),
                               ('reviewed_episode_history.csv','история подтверждённых смысловых меток')]:
            path=directory/filename
            if path.exists() and path.stat().st_size>1:
                show_table(name+' · '+title,read_report(path))
    return pd.DataFrame(experiment_directories(cfg))


def show_detail(cfg, selection,contexts=None):
    from IPython.display import display, Image
    import matplotlib.pyplot as plt
    from incremental_plots import plot_timeline
    candidates=[item for item in experiment_directories(cfg)
                if (not selection.get('experiment') or item['experiment']==selection['experiment'])
                and (not selection.get('unit') or item['unit']==selection['unit'])]
    if not candidates:
        print('Нет рассчитанного эксперимента для выбранных котла и модели.')
        return None
    directory=candidates[0]['directory']
    view=selection.get('view','mean::instant')
    budget=float(selection.get('budget') or cfg['alarm_budget'])
    slug=view.replace(':','_')+'__'+str(budget)
    if not (directory/(slug+'.parquet')).exists():
        raise ValueError(f'Нет вида {view} с бюджетом {budget} в {directory.name}')
    if selection.get('style','legacy')=='legacy':
        from monitoring_v3.training_plot import plot_training_timeline
        from monitoring_v3.risk_history import read_risk_timeline
        if contexts is None:
            from monitoring_v3.data import load_sources
            contexts,_=load_sources(cfg)
        timeline=pd.read_parquet(directory/(slug+'.parquet'))
        status_path=directory/'interpretation_status.json'
        statuses=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        risk=read_risk_timeline(directory,cfg,contexts[candidates[0]['unit']])
        figure,alarm_table=plot_training_timeline(timeline,contexts[candidates[0]['unit']],cfg,
            candidates[0]['method']+' · '+candidates[0]['unit'],risk,statuses,view,
            selection.get('start'),selection.get('end'))
        show_table('Риск на тревогах: оценка при подтверждении и последующее развитие',alarm_table)
    else:
        figure=plot_timeline(directory,view,budget,selection.get('start'),selection.get('end'))
    buffer=BytesIO();figure.savefig(buffer,format='png',dpi=130);plt.close(figure)
    display(Image(data=buffer.getvalue()))
    for suffix,title,fields in [('_events.csv','Обнаружение отдельных аварий',['stop']),
                                 ('_alarms.csv','Эпизоды тревоги',['start','confirmed','end'])]:
        path=directory/(slug+suffix)
        frame=read_report(path)
        if len(frame) and (selection.get('start') or selection.get('end')):
            left=pd.Timestamp(selection['start']) if selection.get('start') else pd.Timestamp.min
            right=pd.Timestamp(selection['end']) if selection.get('end') else pd.Timestamp.max
            if suffix=='_events.csv':
                times=pd.to_datetime(frame['stop']);frame=frame.loc[times.ge(left)&times.le(right)]
            else:
                frame=frame.loc[pd.to_datetime(frame['end']).gt(left)&pd.to_datetime(frame['start']).le(right)]
        show_table(title,frame,True)
    versions=sorted(p for p in (directory.parent/'feature_importance').glob('*') if p.is_dir())
    selected_month=selection.get('importance_month')
    choices=[p for p in versions if p.name.startswith(f'{pd.Timestamp(selected_month):%Y-%m}')] if selected_month else versions
    version=choices[-1] if choices else directory.parent
    if selected_month and (not choices or not (version/'feature_importance.csv').exists()):
        raise ValueError('Нет сохранённой важности признаков для среза '+str(selected_month))
    show_feature_importance(version/'feature_importance.csv',target=selection.get('importance_sensor'),
        title='Регрессор · модель для среза '+version.name)
    return directory


def show_feature_importance(path,target=None,title='Важность признаков'):
    from IPython.display import display,Image
    import matplotlib.pyplot as plt
    path=Path(path)
    if not path.exists():return
    table=read_report(path)
    if table.empty:return
    available=table.loc[table.status.eq('available')]
    if available.empty:
        show_table(title+' · значение недоступно для этой модели',table)
        return
    if target is None:target=available.target.iloc[0]
    chosen=available.loc[available.target.eq(target)].head(20)
    if chosen.empty:
        raise ValueError('Нет важности для цели '+str(target)+'. Доступны: '+str(sorted(available.target.unique())))
    show_table(title+' · '+str(target)+' · первые 20',chosen,True)
    ranked=chosen.iloc[::-1]
    figure,axis=plt.subplots(figsize=(13,max(4,len(ranked)*.3)))
    labels=[str(name) if len(str(name))<95 else str(name)[:92]+'…' for name in ranked.feature]
    axis.barh(range(len(ranked)),ranked.importance_percent,color='#357abd')
    axis.set_yticks(range(len(ranked)),labels,fontsize=8)
    axis.set_xlabel('Важность PredictionValuesChange, % (показана часть входов)')
    axis.set_title(title+' · '+str(target));figure.tight_layout()
    buffer=BytesIO()
    try:
        figure.savefig(buffer,format='png',dpi=120);display(Image(data=buffer.getvalue()))
    finally:plt.close(figure)


def episode_catalog(cfg):
    frames=[]
    for item in experiment_directories(cfg):
        path=item['directory']/'episode_labels.csv'
        if path.exists():
            f=read_report(path)
            if f.empty:
                continue
            f.insert(0,'experiment',item['experiment']);f.insert(1,'recipient',item['unit'])
            frames.append(f)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def apply_episode_labels(cfg, edits):
    """Explicit notebook review; no autogenerated cause labels."""
    catalog=episode_catalog(cfg)
    pending={}
    for edit in edits:
        required={'experiment','unit','episode_id','label','label_available_at','reviewer'}
        if required-set(edit):
            raise ValueError('Недостаточно полей метки: '+str(sorted(required-set(edit))))
        match=catalog.loc[catalog.experiment.eq(edit['experiment']) & catalog.recipient.eq(edit['unit']) &
                          catalog.episode_id.eq(edit['episode_id'])]
        if len(match)!=1:
            raise ValueError('Эпизод не найден однозначно: '+str(edit['episode_id']))
        if any(pd.isna(edit[k]) or not str(edit[k]).strip() for k in ('label','reviewer')) or pd.isna(pd.to_datetime(edit['label_available_at'])):
            raise ValueError('Укажите смысл эпизода, специалиста и дату доступности метки')
        path=next(i['directory'] for i in experiment_directories(cfg)
                  if i['experiment']==edit['experiment'] and i['unit']==edit['unit'])/'episode_labels.csv'
        f=pending.setdefault(path,pd.read_csv(path))
        mask=f.episode_id.eq(edit['episode_id'])
        for col in ('label','label_available_at','reviewer','reason'):
            f[col]=f[col].astype(object)
            f.loc[mask,col]=edit.get(col,'')
        f['label_source']=f['label_source'].astype(object)
        f.loc[mask,'label_source']='reviewed_in_notebook'
    # Validate every requested edit before modifying any file.
    for path,frame in pending.items():
        frame.to_csv(path,index=False,encoding='utf-8-sig')
    return episode_catalog(cfg)


def run_interpretation(cfg, contexts, enabled=True, as_of=None, attach=False,live_report=None,risk_history=False,reuse_monthly=True):
    if reuse_monthly:
        return _monthly_interpretation_report(cfg,contexts,enabled,as_of,attach,live_report)
    from interpretation_training import train,train_risk_group
    from monitoring_v3.progress import Progress
    rows=[]
    items=experiment_directories(cfg);shared={};histories={};readiness=[]
    from monitoring_v3.risk_reporting import readiness_rows
    from monitoring_v3.risk_history import build_risk_history
    progress=Progress(Path(cfg['output'])) if enabled else None
    if enabled and cfg['risk_policy']['scope']=='group':
        families={}
        for item in items:
            spec=json.loads((item['directory'].parent/'experiment.json').read_text(encoding='utf-8'))
            key=(spec.get('group',''),item['method'],item['variant'],spec.get('horizon',0),item['scenario'],
                 '+'.join(spec.get('sources',[])) if item['scenario']=='transfer' else '')
            families.setdefault(key,[]).append(item)
        for key,members in families.items():
            if live_report:live_report.safely('select_items',members,progress)
            date=pd.Timestamp(as_of) if as_of else max(contexts[i['unit']]['frame'].index[-1] for i in members)
            with progress.stage('Классификаторы риска · группа '+', '.join(sorted({i['unit'] for i in members}))):
                result=train_risk_group(cfg,members,contexts,date)
            for item in members:shared[str(item['directory'])]=(result,date)
            if risk_history:
                historical=build_risk_history(cfg,members,contexts,date,progress,
                    on_update=lambda done,total:progress.set_detail('История риска: '+str(done)+'/'+str(total),risk_done=done,risk_total=total))
                for item in members:histories[str(item['directory'])]=historical.get(item['unit'])
    for item in items:
        if enabled:
            if live_report:live_report.safely('select_items',[item],progress)
            date=pd.Timestamp(as_of) if as_of else contexts[item['unit']]['frame'].index[-1]
            result=shared.get(str(item['directory']))
            if result:date=result[1]
            with progress.stage('Интерпретация · '+item['experiment']+' · '+item['unit']):
                statuses=train(cfg,item['directory'],item['unit'],date,attach=attach,contexts=contexts,
                               risk_statuses=result[0] if result else None)
            if risk_history and not result:
                historical=build_risk_history(cfg,[item],contexts,date,progress,
                    on_update=lambda done,total:progress.set_detail('История риска: '+str(done)+'/'+str(total),risk_done=done,risk_total=total))
                histories[str(item['directory'])]=historical.get(item['unit'])
        else:
            path=item['directory']/'interpretation_status.json'
            statuses=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'all':{'status':'not_run'}}
        for head,status in statuses.items():
            rows.append(dict(experiment=item['experiment'],unit=item['unit'],head=head,**status))
        ready=readiness_rows(statuses,cfg)
        if len(ready):
            ready.insert(0,'котёл',item['unit']);ready.insert(0,'эксперимент',item['experiment']);readiness.append(ready)
            ready.to_csv(item['directory']/'classifier_readiness.csv',index=False,encoding='utf-8-sig')
        if live_report:
            live_report.safely('risk',item,statuses,histories.get(str(item['directory'])),contexts[item['unit']])
    result=pd.DataFrame(rows)
    if progress:
        from monitoring_v3.progress_display import notebook_display
        progress.view=notebook_display()
        progress.say('Интерпретация завершена',status='complete')
    if readiness:
        table=pd.concat(readiness,ignore_index=True)
        table.to_csv(Path(cfg['output'])/'classifier_readiness.csv',index=False,encoding='utf-8-sig')
        show_table('Готовность риска: независимые группы и чего не хватает',table,True)
    show_table('Интерпретация смысла и предаварийного риска',result,True)
    shown=set()
    directories={(i['experiment'],i['unit']):i['directory'] for i in items}
    for row in rows:
        if row.get('status') not in ('trained','trained_uncalibrated'):continue
        folder=Path(row['shared_head']).parent if row.get('shared_head') else directories[(row['experiment'],row['unit'])]/'interpretation'/row['head']
        path=folder/'feature_importance.csv'
        if path not in shown:
            show_feature_importance(path,title=row['unit']+' · '+row['head']);shown.add(path)
    return result


def _monthly_interpretation_report(cfg,contexts,enabled,as_of,attach,live_report):
    """Section 7 reads the risk trained in section 3; only reviewed causes fit here."""
    from interpretation_training import train
    from monitoring_v3.monthly_risk import saved_monthly_statuses
    from monitoring_v3.risk_reporting import readiness_rows
    from monitoring_v3.risk_history import read_risk_timeline
    from monitoring_v3.progress import Progress
    items=experiment_directories(cfg);rows=[];ready=[];shown=set()
    progress=Progress(Path(cfg['output'])) if enabled else None
    for item in items:
        if as_of is not None:date=pd.Timestamp(as_of)
        elif cfg['risk_policy']['scope']=='group':
            spec=json.loads((item['directory'].parent/'experiment.json').read_text(encoding='utf-8'))
            units=cfg['groups'].get(spec.get('group'),[item['unit']])
            date=max(contexts[u]['frame'].index[-1] for u in units)
        else:date=contexts[item['unit']]['frame'].index[-1]
        statuses=saved_monthly_statuses(item['directory'],date)
        if statuses is None:
            statuses={f'risk_{h}h':{'status':'requires_monthly_training',
                'detail':'Выполните раздел 3: риск обучается вместе с детекторами по месяцам.'} for h in cfg['risk_horizons']}
        if enabled:
            with progress.stage('Проверенные причины · '+item['experiment']+' · '+item['unit']):
                statuses=train(cfg,item['directory'],item['unit'],date,attach=attach,contexts=contexts,risk_statuses=statuses)
        for head,status in statuses.items():rows.append(dict(experiment=item['experiment'],unit=item['unit'],head=head,**status))
        for head,status in statuses.items():
            if status.get('status') not in ('trained','trained_uncalibrated'):continue
            folder=Path(status['shared_head']).parent if status.get('shared_head') else item['directory']/'interpretation'/head
            path=folder/'feature_importance.csv'
            if path.exists() and path not in shown:
                show_feature_importance(path,title=head+' · версия на '+str(status.get('available_at',date)))
                shown.add(path)
        table=readiness_rows(statuses,cfg)
        if len(table):
            table.insert(0,'котёл',item['unit']);table.insert(0,'эксперимент',item['experiment']);ready.append(table)
        if live_report:
            history=read_risk_timeline(item['directory'],cfg,contexts[item['unit']])
            live_report.safely('set_monthly_risk',item,statuses,history)
    if progress:progress.say('Риск прочитан из раздела 3; разбор причин завершён',status='complete')
    result=pd.DataFrame(rows)
    if ready:show_table('Готовность помесячного риска из раздела 3',pd.concat(ready,ignore_index=True),True)
    show_table('Классификаторы: помесячный риск и подтверждённые причины',result,True)
    return result


def run_research(cfg, contexts, options, detail_dir=None):
    from monitoring_v3.experiments import ensemble_features,survival_experiment,sensitivity
    rows=[]
    for unit,context in contexts.items():
        if not any(i['unit']==unit for i in experiment_directories(cfg)):
            continue
        features=None
        if options.get('ensemble') or options.get('survival'):
            features=ensemble_features(Path(cfg['output']),unit)
            directory=Path(cfg['output'])/'investigations'/unit/'ensemble'
            directory.mkdir(parents=True,exist_ok=True)
            features.to_parquet(directory/'ensemble_features.parquet')
            rows.append(dict(unit=unit,task='ensemble',status='ready',rows=len(features),features=len(features.columns)))
            show_table(unit+' · совместные признаки детекторов, последние 168 строк',
                       features.tail(168).rename_axis('time').reset_index())
        if options.get('survival'):
            f=features.loc[context['eligible'].reindex(features.index).fillna(False)]
            date=pd.Timestamp(options['as_of']) if options.get('as_of') else context['frame'].index[-1]
            status=survival_experiment(f,context['events'],cfg,date,context['frame'].index[-1],
                                       Path(cfg['output'])/'investigations'/unit/'survival',observed=context['cleaned'][context['load']].notna())
            rows.append(dict(unit=unit,task='survival',**status))
    if options.get('sensitivity'):
        if not detail_dir:
            raise ValueError('Выберите рассчитанный эксперимент в DETAIL')
        directory=Path(detail_dir)
        unit=next(i['unit'] for i in experiment_directories(cfg) if i['directory']==directory)
        from saved_model_runner import _runtime
        with _runtime(directory/'model','incremental') as (model,_):
            sensor=options.get('sensor')
            if sensor not in model.cols:
                raise ValueError('Выберите sensor из списка проверяемых каналов: '+str(model.cols))
            if not options.get('start'):
                raise ValueError('Укажите start для искусственного нарушения')
            start=pd.Timestamp(options['start'])
            if start<model.state['cutoff']:
                raise ValueError('Искусственное нарушение должно начинаться после даты модели')
            table=sensitivity(model,contexts[unit]['frame'],sensor,start,cfg,
                              Path(cfg['output'])/'investigations'/unit/'sensitivity')
            show_table('Чувствительность к искусственным нарушениям',table,True)
            rows.append(dict(unit=unit,task='sensitivity',status='synthetic_only',rows=len(table)))
    result=pd.DataFrame(rows)
    show_table('Дополнительные исследования',result,True)
    if not rows:
        print('Дополнительные исследования выключены в RESEARCH; основные отчёты уже выведены выше.')
    return result
