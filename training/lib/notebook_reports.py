"""Notebook actions and embedded reports; no shell commands or separate processes."""
from copy import deepcopy
from html import escape
from io import BytesIO
from pathlib import Path
import json
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


def show_preparation(contexts, issues, cfg):
    from IPython.display import display, Image
    from monitoring_v3.data import healthy
    show_table('Проблемы источников',pd.DataFrame(issues,columns=['unit','problem','path']).rename(
        columns={'unit':'котёл','problem':'проблема','path':'путь'}),bool(issues))
    step_h = float(pd.Timedelta(cfg['step'])/pd.Timedelta('1h'))
    summary, modes, events = [], [], []
    for unit,c in contexts.items():
        summary.append({'котёл':unit,'датчик нагрузки':c['load'],'единица':c['declared_units'].get(c['load'],''),
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
        numeric=c['frame'][c['targets']]
        stats=numeric.agg(['count','nunique','min','max']).T.rename_axis('датчик').reset_index()
        stats['доля пропусков']=numeric.isna().mean().to_numpy()
        for flag in ('spike','frozen','masked'):
            stats[flag+', ч']=c['quality'].xs(flag,axis=1,level=1).sum().reindex(numeric.columns).to_numpy()*step_h
        show_table(unit+' · доступность и качество измерений',stats)
        path=Path(cfg['output'])/'audit'/unit/'load_and_modes.png'
        if path.exists():
            display(Image(filename=str(path)))
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
    show_table('Качество моделей и схем тревог',result['summary'],True)
    common=Path(cfg['output'])/'comparison_common_hours.csv'
    if common.exists() and common.stat().st_size>1:
        show_table('Сравнение на общих рассчитанных часах',read_report(common))
    for item in experiment_directories(cfg):
        directory=item['directory'];name=item['experiment']+' · '+item['unit']
        updates=directory.parent/'updates.json'
        if updates.exists():
            show_table(name+' · месячные обновления',pd.DataFrame(json.loads(updates.read_text(encoding='utf-8'))))
        for filename,title in [('prediction_diagnostics.csv','качество прогноза'),
                               ('episode_labels.csv','эпизоды и метки смысла')]:
            path=directory/filename
            if path.exists() and path.stat().st_size>1:
                show_table(name+' · '+title,read_report(path))
    return pd.DataFrame(experiment_directories(cfg))


def show_detail(cfg, selection):
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
    return directory


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


def run_interpretation(cfg, contexts, enabled=True, as_of=None, attach=False):
    from interpretation_training import train
    rows=[]
    for item in experiment_directories(cfg):
        if enabled:
            date=pd.Timestamp(as_of) if as_of else contexts[item['unit']]['frame'].index[-1]
            statuses=train(cfg,item['directory'],item['unit'],date,attach=attach,contexts=contexts)
        else:
            path=item['directory']/'interpretation_status.json'
            statuses=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'all':{'status':'not_run'}}
        for head,status in statuses.items():
            rows.append(dict(experiment=item['experiment'],unit=item['unit'],head=head,**status))
    result=pd.DataFrame(rows)
    show_table('Интерпретация смысла и предаварийного риска',result,True)
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
                                       Path(cfg['output'])/'investigations'/unit/'survival')
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
