"""Pandas / matplotlib / sklearn reports for detector notebook 02; no model fitting."""
from pathlib import Path
from html import escape
from io import BytesIO
import base64
import json
import numpy as np
import pandas as pd


def read_csv(path):
    path=Path(path)
    if not path.exists():return pd.DataFrame()
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def details(title,body,opened=False):
    return '<details'+(' open' if opened else '')+'><summary><b>'+escape(str(title))+'</b></summary>'+body+'</details>'


def table(title,frame,opened=False):
    body=('<p>Нет записей.</p>' if frame.empty else
          '<div style="overflow:auto;max-height:520px">'+frame.to_html(index=False,escape=True,na_rep='—',max_rows=None,max_cols=None)+'</div>')
    return details(title,body,opened)


def figure_html(fig,path):
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    buffer=BytesIO()
    try:
        fig.savefig(buffer,format='png',dpi=110,bbox_inches='tight')
        payload=buffer.getvalue();path.write_bytes(payload)
        return '<img style="max-width:100%;height:auto" src="data:image/png;base64,'+base64.b64encode(payload).decode()+'">'
    finally:plt.close(fig)


def show_html(body):
    from IPython.display import HTML,display
    display(HTML(body))


def preparation_report(contexts,issues,cfg):
    from incremental_plots import plot_preparation
    from monitoring_v3.pipeline import dataset
    from monitoring_v3.data import healthy
    from notebook_reports import check_preparation_thresholds
    check_preparation_thresholds(contexts,cfg)
    root=Path(cfg['output'])/'notebook_report';root.mkdir(parents=True,exist_ok=True)
    summary=[];events=[]
    for unit,c in contexts.items():
        frame=c['cleaned'];ix=frame.index
        summary.append(dict(unit=unit,start=ix[0],end=ix[-1],rows=len(ix),
            sensors=len(c['targets']),condition_features=c['features'].shape[1],
            eligible_rows=int(c['eligible'].sum()),healthy_before_test=int(healthy(c,cfg['first_test'],cfg).sum()),
            rows_in_test=int((ix>=pd.Timestamp(cfg['first_test'])).sum()),load=c['load'],run_threshold=c['threshold']))
        e=c['events'].copy();e.insert(0,'unit',unit);events.append(e)
    summary=pd.DataFrame(summary)
    summary.to_csv(root/'data_summary.csv',index=False,encoding='utf-8-sig')
    body=table('Источники и доступные данные',summary,True)
    if issues:body+=table('Проблемы источников',pd.DataFrame(issues),True)
    body+='<p>healthy_before_test — допустимая норма до начала проверки, до разделения на обучение и калибровку. '
    body+='Число строк конкретного месячного обучения показано далее в журнале отбора. '
    body+='Таблицы ниже разворачиваются стрелкой; признаки не обрезаются многоточием.</p>'
    for unit,c in contexts.items():
        d=c['cleaned'];cols=list(dict.fromkeys(c['targets']+[k for k,r in c['roles'].items() if r in ('condition','load')]))
        numeric=d[cols].apply(pd.to_numeric,errors='coerce')
        stats=numeric.describe().T.rename_axis('sensor').reset_index()
        stats['missing_fraction']=numeric.isna().mean().to_numpy()
        stats['unique_values']=numeric.nunique().to_numpy()
        for flag in ('spike','frozen','masked'):
            stats[flag+'_rows']=c['quality'].xs(flag,axis=1,level=1).sum().reindex(cols).to_numpy()
        schema=pd.DataFrame([dict(sensor=k,role=v,unit=c.get('declared_units',{}).get(k,''),dtype=str(d[k].dtype),
                                 unique_values=int(d[k].nunique()),missing_fraction=float(d[k].isna().mean()))
                             for k,v in c['roles'].items() if k in d])
        part=table('Назначение и типы каналов',schema)+table('Статистика и качество измерений',stats)
        part+=table('Пример исходных данных',c['frame'].head(5).reset_index())
        part+=table('Пример после подготовки',d.head(5).reset_index())
        part+=table('Режимы: число строк',c['regime']['mode'].value_counts().rename_axis('mode').reset_index(name='rows'),True)
        part+=table('Реестр остановов и ремонтов',c['events'])
        part+=table('Токи: сопоставление и пороги',pd.DataFrame(c.get('regime_spec',{}).get('audit',[])))
        if 'audit' in c:part+=table('Сопоставление датчиков',c['audit'])
        for variant in cfg['feature_sets']:
            horizons=cfg['horizons'] if variant=='forecast' else [0]
            for horizon in horizons:
                x,y=dataset(c,variant,horizon,cfg)
                names=pd.DataFrame({'feature':x.columns,'dtype':x.dtypes.astype(str).to_numpy(),
                                    'missing_fraction':x.isna().mean().to_numpy()})
                preview=y.index[y.notna().any(axis=1)][:5]
                part+=details(f'{variant}, горизонт {horizon} ч: {x.shape[1]} входных признаков; {y.shape[1]} целей',
                    '<p>Для CatBoost/related это общий набор кандидатов: собственный датчик исключается из входов его модели. '
                    'В pooled/transfer набор дополнительно согласуется между котлами.</p>'+
                    table('Все признаки',names)+table('Пример X: первые допустимые строки',x.loc[preview].reset_index())+
                    table('Пример y для этих же строк',y.loc[preview].reset_index()))
                del x,y
        if 'related' in cfg['feature_sets'] and set(cfg['models'])&{'AAKR','MSET'}:
            x,y=dataset(c,'related',0,cfg,method='AAKR')
            from monitoring_v3.models import RelatedKernelModel
            from monitoring_v3.data import healthy
            complete=RelatedKernelModel.complete_vectors(x,c['cleaned'][c['targets']])
            admitted_complete=int((complete&healthy(c,cfg['first_test'],cfg)).sum())
            part+=details('Классические AAKR/MSET: исходный вектор и полные измерения',
                '<p>Используются исходные измерения условий, состояний и всех проверяемых датчиков. '
                f'Полных векторов до разделения на норму/обучение: {int(complete.sum())} из {len(complete)}. '
                f'Полных допущенных нормальных векторов до first_test: {admitted_complete}. '
                'Вычисляемые режимные признаки не входят. Пропущенные значения не подставляются.</p>'+
                table('Состав вектора',pd.DataFrame({'channel':list(x)+list(y),
                    'role':['condition/state']*len(x)+['target']*len(y)})))
        part+=figure_html(plot_preparation(c),root/unit/'preparation.png')
        body+=details(unit+' · подготовка данных',part)
    event_frame=pd.concat(events,ignore_index=True) if events else pd.DataFrame()
    body+=table('Все события',event_frame)
    (root/'preparation.html').write_text(body,encoding='utf-8');show_html(body)
    return summary


def saved_experiments(cfg):
    from monitoring_v3.pipeline import experiments
    planned=list(experiments(cfg));rows=[]
    for meta in sorted((Path(cfg['output'])/'experiments').glob('*/experiment.json')):
        spec=json.loads(meta.read_text(encoding='utf-8'))
        if not any(spec==s for s in planned):continue
        for unit in spec['recipients']:
            rows.append(dict(experiment=meta.parent.name,unit=unit,method=spec['method'],variant=spec['variant'],
                             scenario=spec['scenario'],directory=meta.parent/unit,spec=spec))
    return rows


def load_timeline(item,view,budget,cfg,context):
    """Read finalized output, or reconstruct an explicitly provisional interrupted run.

    Only cached scores with the current checkpoint/data fingerprint are admitted.
    No fitting or retrospective prediction using the latest model is performed.
    """
    from monitoring_v3.alarm_scores import alarm_stream,view_policy
    import joblib
    directory=Path(item['directory']);spec=item['spec'];unit=item['unit']
    slug=view.replace(':','_')+'__'+str(float(budget))
    path=directory/(slug+'.parquet')
    candidates_on_disk=(list((directory.parent/'scored').glob('*/'+unit+'.joblib'))+
                        list((directory.parent/'scored').glob('*/'+unit+'.detector.joblib')))
    if path.exists() and not any(p.stat().st_mtime_ns>path.stat().st_mtime_ns for p in candidates_on_disk):
        return pd.read_parquet(path),False
    from monitoring_v3.pipeline import score_fingerprint
    from monitoring_v3.checkpoint_cache import semantic_config
    # Checkpoint compatibility is deliberately required for partial results.
    # Full exports have already been written, and remain useful independently.
    parts=[]
    for checkpoint in sorted((directory.parent/'checkpoints').glob('*.joblib')):
        folder=directory.parent/'scored'/checkpoint.stem
        candidates=[p for p in (folder/(unit+'.detector.joblib'),folder/(unit+'.joblib')) if p.exists()]
        if not candidates:continue
        end=pd.Timestamp(checkpoint.stem)+pd.offsets.MonthBegin(1)
        month=None
        for cached in sorted(candidates,key=lambda p:p.stat().st_mtime_ns,reverse=True):
            value=joblib.load(cached)
            token=value.get('model_token');old_cfg=value.get('report_config')
            if token is None or old_cfg is None:
                # Compatibility only: modern score caches carry these small
                # signatures, so viewing them never loads 226 native models.
                saved_state=joblib.load(checkpoint)
                token=saved_state.get('compatibility')
                old_cfg=getattr(saved_state['state']['model'],'cfg',{})
                del saved_state
            if token is None or semantic_config(old_cfg)!=semantic_config(cfg):continue
            expected=score_fingerprint(context,end,token)
            if value.get('fingerprint')==expected:month=value;break
        if month is None:continue
        scores=month['month']['views'].get(view)
        if scores is not None:parts.append(scores)
    if not parts:return pd.DataFrame(),True
    d=pd.concat(parts).sort_index();d=d.loc[~d.index.duplicated(keep='last')]
    first=min(pd.Timestamp(cfg['first_test']),d.index[0])
    ix=context['frame'].index
    ix=ix[(ix>=first)&(ix<=d.index[-1])];d=d.reindex(ix)
    threshold='threshold_budget_'+str(float(budget))
    if threshold not in d:raise ValueError('Нет сохранённого порога '+threshold)
    d['ratio']=d.smoothed/d[threshold].where(np.isfinite(d[threshold]))
    stream,_=alarm_stream(d.ratio,view_policy(cfg,view))
    d['active']=stream.active;d['notification']=stream.notification
    d['eligible']=context['eligible'].reindex(ix).fillna(False)
    d['background_eligible']=context['frame']['Состояние'].reindex(ix).eq('Работа')
    d['load_observed']=context['cleaned'][context['load']].reindex(ix).notna()
    return d,True


def hourly_examples(timeline,events,cfg,start,end=None):
    """Exactly the hourly ranking population used by alarm_scores.evaluate.

    Positive = an eligible score in the configured pre-failure window; negative
    = eligible observed background. Neutral/unknown intervals are excluded.
    This is not a classifier ground truth for arbitrary types of anomaly.
    """
    ix=timeline.index
    end=pd.Timestamp(end) if end is not None else ix[-1]+pd.Timedelta(cfg['step'])
    valid=timeline.score.notna()&np.isfinite(timeline.ratio)&(ix>=pd.Timestamp(start))&(ix<end)
    positive=pd.Series(False,index=ix);background=pd.Series(True,index=ix);uncertain=positive.copy()
    for e in events.itertuples():
        if pd.isna(e.stop):continue
        if e.type=='Неизвестно':uncertain|=(ix>=e.stop-pd.Timedelta(days=cfg['pre_window_days']))&(ix<e.stop)
        if e.type!='Авария':continue
        positive|=(ix>=e.stop-pd.Timedelta(days=cfg['pre_window_days']))&(ix<e.stop-pd.Timedelta(hours=cfg['min_lead_hours']))
        background&=~((ix>=e.stop-pd.Timedelta(days=cfg['pre_window_days']+cfg['neutral_days']))&(ix<e.stop))
    background&=valid&~uncertain&timeline.get('background_eligible',pd.Series(True,index=ix)).fillna(False)
    take=(positive|background)&valid
    return positive.loc[take].astype(int),timeline.loc[take,'ratio'],timeline.loc[take,'active'].fillna(False).astype(int)


def timeline_figure(timeline,context,cfg,title,start,end,budget=None):
    import matplotlib.pyplot as plt
    d=timeline.loc[start:end]
    fig,axes=plt.subplots(3,1,figsize=(14,7),sharex=True,gridspec_kw={'height_ratios':[3,1,1]})
    if d.empty:
        axes[0].text(.5,.5,'В выбранном периоде ещё нет результатов',ha='center',transform=axes[0].transAxes)
        return fig
    axes[0].plot(d.index,d.ratio,lw=.7,label='Отклонение / действующий порог')
    axes[0].axhline(1,color='red',ls='--',label='Порог тревоги')
    active=d.active.fillna(False).to_numpy(bool)
    axes[0].fill_between(d.index,0,1,where=active,transform=axes[0].get_xaxis_transform(),color='red',alpha=.12,label='Тревога')
    notices=d.loc[d.notification.fillna(False)]
    axes[0].scatter(notices.index,notices.ratio,s=12,color='red',label='Уведомление')
    axes[0].set_yscale('symlog',linthresh=1);axes[0].set_ylabel('Отклонение / порог');axes[0].set_title(title)
    key='threshold_budget_'+str(float(budget if budget is not None else cfg['alarm_budget']))
    threshold=d[key] if key in d else d['smoothed'].div(d.ratio).replace([np.inf,-np.inf],np.nan)
    axes[1].plot(d.index,d.smoothed,lw=.7,label='Сглаженное отклонение')
    axes[1].plot(d.index,threshold,lw=1,color='red',ls='--',label='Фактический порог этой схемы')
    axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylabel('Балл')
    mode=context['regime']['mode'].reindex(d.index)
    axes[2].step(d.index,mode.map({'стоит':0,'неизвестно':1,'пуск':2,'работа':3}),where='post',lw=.8)
    axes[2].set_yticks([0,1,2,3],['стоит','неизвестно','пуск','работа']);axes[2].set_ylabel('Режим')
    for e in context['events'].itertuples():
        if pd.isna(e.start) or pd.isna(e.end) or e.start>d.index[-1] or e.end<d.index[0]:continue
        for ax in axes:ax.axvspan(e.start,e.end,color='red' if e.type=='Авария' else 'gray',alpha=.12)
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left');axes[1].legend(fontsize=8)
    fig.tight_layout();return fig


def regression_table(directory,context=None,start=None,end=None):
    d=read_csv(Path(directory)/'prediction_diagnostics.csv')
    if 'month' in d:
        when=pd.to_datetime(d.month)
        if start is not None:d=d.loc[when.ge(pd.Timestamp(start).to_period('M').to_timestamp())]
        if end is not None:d=d.loc[pd.to_datetime(d.month).le(pd.Timestamp(end))]
    predictions=Path(directory)/'prequential_predictions.parquet'
    errors=Path(directory)/'prequential_errors.parquet'
    if context is not None and predictions.exists() and errors.exists():
        from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
        predicted=pd.read_parquet(predictions).loc[start:end]
        valid=pd.read_parquet(errors).reindex(predicted.index).notna()
        rows=[]
        for sensor in predicted:
            if sensor not in context['cleaned']:continue
            a=context['cleaned'][sensor].reindex(predicted.index)
            p=predicted[sensor]
            take=valid[sensor]&np.isfinite(a)&np.isfinite(p)
            a,p=a.loc[take].astype(float),p.loc[take].astype(float)
            if not len(a):continue
            rows.append(dict(sensor=sensor,rows=len(a),MAE=float(mean_absolute_error(a,p)),
                RMSE=float(np.sqrt(mean_squared_error(a,p))),
                R2=float(r2_score(a,p)) if len(a)>1 and a.nunique()>1 else np.nan))
        return pd.DataFrame(rows),d
    sufficient=['regression_rows','absolute_error_sum','squared_error_sum','target_sum','target_squared_sum']
    if d.empty or not set(sufficient).issubset(d):return pd.DataFrame(),d
    d=d.loc[d.regression_rows.gt(0)].copy()
    grouped=d.groupby('sensor')[sufficient].sum()
    n=grouped.regression_rows
    out=pd.DataFrame({'sensor':grouped.index,'rows':n.to_numpy(),
        'MAE':(grouped.absolute_error_sum/n).to_numpy(),
        'RMSE':np.sqrt(grouped.squared_error_sum/n).to_numpy()})
    total=grouped.target_squared_sum-grouped.target_sum**2/n
    out['R2']=(1-grouped.squared_error_sum/total.where(total.gt(1e-10))).to_numpy()
    return out,d


def classification_html(y,score,pred,path,prediction_title='Подтверждённая тревога, почасовая оценка'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (classification_report,ConfusionMatrixDisplay,roc_auc_score,
                                average_precision_score,RocCurveDisplay,PrecisionRecallDisplay)
    if not len(y):return '<p>Нет часов с известной оценочной меткой.</p>',{}
    two=y.nunique()==2
    roc=float(roc_auc_score(y,score)) if two else np.nan
    ap=float(average_precision_score(y,score)) if two else np.nan
    chance=float(y.mean())
    metrics={'hourly_roc_auc':roc,'hourly_average_precision':ap,'hourly_random_ap':chance,
             'hourly_rows':len(y),'hourly_positive_rows':int(y.sum())}
    from sklearn.metrics import precision_recall_fscore_support
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=[1],zero_division=0)
    metrics.update(hourly_precision=float(precision[0]),hourly_recall=float(recall[0]),hourly_f1=float(f1[0]))
    fmt=lambda v:'—' if not np.isfinite(v) else f'{v:.4f}'
    body=f'<p><b>ROC AUC: {fmt(roc)}; Average Precision: {fmt(ap)}; случайный AP: {chance:.4f}; случайный ROC AUC: 0.5.</b></p>'
    if not two:body+='<p>В выбранном периоде один класс: ROC AUC и AP не оцениваем.</p>'
    body+='<pre>'+escape(classification_report(y,pred,labels=[0,1],target_names=['фон','перед аварией'],digits=3,zero_division=0))+'</pre>'
    fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay.from_predictions(y,pred,labels=[0,1],
        display_labels=['фон','перед аварией'],ax=ax,colorbar=False,cmap='Blues',values_format='d')
    ax.set_title(prediction_title)
    body+=figure_html(fig,Path(path)/'confusion_matrix.png')
    if two:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        RocCurveDisplay.from_predictions(y,score,ax=axes[0]);axes[0].plot([0,1],[0,1],'--',color='gray',label='Случайно')
        PrecisionRecallDisplay.from_predictions(y,score,ax=axes[1]);axes[1].axhline(chance,color='gray',ls='--',label='Случайный AP')
        for ax in axes:ax.legend(fontsize=8)
        body+=details('ROC и Precision–Recall',figure_html(fig,Path(path)/'ranking_curves.png'))
    return body,metrics


def prediction_plot(directory,context,start,end,sensors=None):
    import matplotlib.pyplot as plt
    path=Path(directory)/'prequential_predictions.parquet'
    if not path.exists():return ('<p>Файл почасовых прогнозов не сохранён. Расчёт мог быть пропущен, '
        'прерван или выполнен прежней версией; это сообщение не подтверждает наличие готовой модели.</p>')
    predicted=pd.read_parquet(path).loc[start:end]
    chosen=[s for s in (sensors or list(predicted.columns[:3])) if s in predicted and s in context['cleaned']]
    if not chosen or predicted.empty:return '<p>Нет прогнозов для выбранных датчиков и периода.</p>'
    fig,axes=plt.subplots(len(chosen),1,figsize=(14,3*len(chosen)),squeeze=False,sharex=True)
    for ax,sensor in zip(axes[:,0],chosen):
        actual=context['cleaned'][sensor].reindex(predicted.index)
        ax.plot(actual.index,actual,lw=.7,label='Измерение')
        ax.plot(predicted.index,predicted[sensor],lw=.8,label='Прогноз модели, доступной в этот месяц')
        ax.set_title(sensor);ax.set_ylabel(context.get('declared_units',{}).get(sensor,''));ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();return figure_html(fig,Path(directory)/'notebook_report'/'actual_prediction.png')


def render_detector_results(cfg,contexts,options=None):
    """Rebuild all inline output from saved monthly / final results, no RESULT needed."""
    from monitoring_v3.alarm_scores import evaluate,view_policy
    options={'views':['mean::instant','top5::instant','max::instant','fraction::instant','mean::slow48'],
             'budgets':None,'start':None,'end':None,'units':None,'experiments':None,'sensors':None,**(options or {})}
    root=Path(cfg['output']);(root/'notebook_report').mkdir(parents=True,exist_ok=True)
    items=saved_experiments(cfg);rows=[];regression_rows=[]
    statuses=read_csv(root/'status.csv')
    if len(statuses):
        show_html(table('Статусы экспериментов (сохраняются и при незавершённом запуске)',statuses,len(statuses)<=12))
        failed=statuses.loc[statuses.status.isin(['failed','aborted','interrupted'])] if 'status' in statuses else pd.DataFrame()
        if len(failed):show_html(table('Ошибки и прерванные эксперименты',failed,True))
    for item in items:
        if options['units'] and item['unit'] not in options['units']:continue
        if options['experiments'] and item['experiment'] not in options['experiments']:continue
        directory=Path(item['directory']);directory.mkdir(parents=True,exist_ok=True)
        c=contexts[item['unit']];report_dir=directory/'notebook_report';report_dir.mkdir(exist_ok=True)
        if item['scenario']!='local':
            from monitoring_v3.pipeline import align_contexts,transfer_preparation
            members=list(dict.fromkeys(item['spec']['sources']+item['spec']['recipients']))
            c=transfer_preparation(align_contexts(contexts,members),item['spec'],cfg)[item['unit']]
        title=item['method']+' · '+item['unit']+' · '+item['variant']+' · '+item['scenario']
        part='<p><b>'+escape(title)+'</b></p>'
        model_dir=directory/'model'
        if all((model_dir/name).is_file() for name in ('state.joblib','config.json','manifest.json')):
            part+='<p>Сохранённый экспорт модели для дашборда: <code>'+escape(str(model_dir.resolve()))+'</code></p>'
        else:
            part+='<p><b>Готовый экспорт модели для дашборда не найден.</b></p>'
        status=statuses.loc[statuses.experiment.eq(item['experiment'])] if 'experiment' in statuses else pd.DataFrame()
        if 'unit' in status:status=status.loc[status.unit.eq(item['unit'])|status.unit.isna()]
        if status.empty and (directory.parent/'status.json').exists():
            status=pd.DataFrame(json.loads((directory.parent/'status.json').read_text(encoding='utf-8')))
            if 'unit' in status:status=status.loc[status.unit.eq(item['unit'])]
        if len(status):part+=table('Статус и причина',status,True)
        error=directory.parent/'error.txt'
        if error.exists():part+=details('Трейсбэк последней ошибки (предыдущей попытки, если расчёт уже возобновлён)',
                                        '<pre>'+escape(error.read_text(encoding='utf-8'))+'</pre>')
        updates=directory.parent/'updates.json'
        if updates.exists():
            update_frame=pd.DataFrame(json.loads(updates.read_text(encoding='utf-8')))
            skip=['trees','target_errors','calibration_by_unit','calibration_sensor_diagnostics','admission_by_unit','complete_vector_missingness']
            part+=table('Все месяцы: обучение, обновление, причины пропуска',update_frame.drop(columns=skip,errors='ignore'))
        for filename,label in [('admission_diagnostics.csv','Отбор часов: куда ушла история'),
                               ('calibration_diagnostics.csv','Калибровка: периоды и пороги'),
                               ('target_update_diagnostics.csv','Принятие обновлений по датчикам'),
                               ('related_inputs.csv','Связи related: выбранные входы и источник выбора'),
                               ('complete_vector_missingness.csv','Классические AAKR/MSET: какие каналы исключают полные векторы'),
                               ('target_availability.csv','CatBoostMulti: обученные датчики и ожидание измерений')]:
            frame=read_csv(directory.parent/filename)
            if 'unit' in frame:frame=frame.loc[frame.unit.eq(item['unit'])]
            if len(frame):part+=table(label,frame)
        regression,diagnostics=regression_table(directory,c,options['start'] or cfg['first_test'],options['end'])
        if len(regression):
            regression['unit']=regression.sensor.map(c.get('declared_units',{})).fillna('')
            regression_rows.append(regression.rename(columns={'unit':'measurement_unit'}).assign(
                experiment=item['experiment'],unit=item['unit'],variant=item['variant'],method=item['method']))
            regression.to_csv(report_dir/'regression_metrics.csv',index=False,encoding='utf-8-sig')
            part+=table('Регрессия на последующих месяцах: MAE, RMSE, R² по датчикам',regression)
            part+='<p>MAE и RMSE — в единицах датчика, только на часах с допустимой диагностикой и известным измерением. Разные датчики в одну физическую ошибку не усредняем. '
            part+='R² сравнивает с постоянным средним; отрицательное значение означает хуже этой константы на периоде оценки.</p>'
        elif len(diagnostics):part+='<p>Этот сохранённый расчёт содержит старые диагностические метрики, но не MAE/RMSE/R².</p>'
        if len(diagnostics):part+=table('Регрессия: все датчики и месяцы',diagnostics.drop(columns=[
            'absolute_error_sum','squared_error_sum','target_sum','target_squared_sum'],errors='ignore'))
        importance=read_csv(directory.parent/'feature_importance_summary.csv')
        if len(importance) and 'mean_importance_percent' in importance:
            importance=importance.loc[pd.to_numeric(importance.mean_importance_percent,errors='coerce').fillna(0).ne(0)]
            part+=table('Важность: только ненулевые входы, среднее по моделям датчиков',importance)
            if len(importance):
                import matplotlib.pyplot as plt
                top=importance.nlargest(20,'mean_importance_percent').iloc[::-1]
                fig,ax=plt.subplots(figsize=(12,max(4,.28*len(top))))
                ax.barh(top.feature,top.mean_importance_percent);ax.set_xlabel('Средняя важность, %');fig.tight_layout()
                part+=details('20 наиболее важных входов',figure_html(fig,report_dir/'importance.png'))
        detailed_views=options['views']
        # Completed output includes every available scheme; partial output uses
        # the requested schemes only and is explicitly marked below.
        views=list(detailed_views or [])
        if detailed_views is None:
            def decode(stem):
                name,mode=stem.rsplit('__',1)[0].rsplit('__',1)
                return (name.replace('group_','group:',1) if name.startswith('group_') else name)+'::'+mode
            views=[decode(p.stem) for p in directory.glob('*__*.parquet')]
            views=list(dict.fromkeys(views)) or ['mean::instant']
        # Summary covers ALL saved schemes, even when detailed figures are limited.
        summary=read_csv(root/'comparison.csv')
        existing=summary.loc[summary.experiment.eq(item['experiment'])&summary.unit.eq(item['unit'])] if {'experiment','unit'}.issubset(summary) else pd.DataFrame()
        all_pairs={(str(row.view),float(row.budget)) for row in existing.itertuples()}
        budgets=options['budgets'] if options['budgets'] is not None else [float(cfg['alarm_budget'])]
        pairs=sorted(all_pairs|{(v,float(b)) for v in views for b in budgets})
        timeline_count=0
        for view,budget in pairs:
            timeline,partial=load_timeline(item,view,budget,cfg,c)
            if timeline.empty:continue
            left=pd.Timestamp(options['start'] or cfg['first_test']);right=pd.Timestamp(options['end']) if options['end'] else None
            if not len(timeline.loc[left:right]):continue
            timeline_count+=1
            policy=view_policy(cfg,view)
            # Figure ends are inclusive pandas timestamps; evaluation is half-open.
            eval_end=right+pd.Timedelta(cfg['step']) if right is not None else None
            metrics,events=evaluate(timeline,c['events'],policy,start=left,end=eval_end)
            y,score,pred=hourly_examples(timeline,c['events'],policy,left,eval_end)
            from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support
            two=y.nunique()==2
            hp,hr,hf,_=precision_recall_fscore_support(y,pred,labels=[1],zero_division=0) if len(y) else ([np.nan],[np.nan],[np.nan],None)
            hm=dict(hourly_roc_auc=float(roc_auc_score(y,score)) if two else np.nan,
                    hourly_average_precision=float(average_precision_score(y,score)) if two else np.nan,
                    hourly_random_ap=float(y.mean()) if len(y) else np.nan,
                    hourly_rows=len(y),hourly_positive_rows=int(y.sum()),hourly_precision=float(hp[0]),hourly_recall=float(hr[0]),hourly_f1=float(hf[0]))
            rows.append(dict(experiment=item['experiment'],unit=item['unit'],method=item['method'],variant=item['variant'],
                view=view,budget=budget,partial=partial,**metrics,**hm))
            if view not in views or budget not in budgets:continue
            slug=view.replace(':','_')+'__'+str(budget);out=report_dir/slug
            report,hm=classification_html(y,score,pred,out)
            subset=pd.DataFrame([dict(view=view,budget=budget,**metrics)])
            if partial:report='<p><b>Промежуточный результат: эксперимент был остановлен; используются только готовые месячные оценки.</b></p>'+report
            report=table('Аварии и уведомления (отдельно от почасового classification report)',subset,True)+report
            report+=figure_html(timeline_figure(timeline,c,cfg,title+' · '+view,left,right,budget),out/'timeline.png')
            report+=table('Какие аварии обнаружены',events)
            from monitoring_v3.alarm_scores import alarm_stream
            _,episodes=alarm_stream(timeline.ratio,policy)
            report+=table('Все эпизоды тревоги',episodes)
            if len(y):pd.DataFrame({'label':y,'score':score,'predicted_alarm':pred}).to_parquet(out/'hourly_evaluation.parquet')
            part+=details(view+f' · бюджет {budget}/720 ч',report,view=='mean::instant')
        if not timeline_count:
            part+='<p><b>Нет сохранённых оценок тревог в выбранном периоде.</b> Смотрите статус и журнал месячных решений выше.</p>'
            if item['variant']=='related' and status.empty and not (model_dir/'state.joblib').exists() and not cfg.get('related_inputs'):
                part+='<p>Пустой related_inputs в прежней реализации приводил к пропуску related до обучения. '
                part+='После обновления кода связи выбираются автоматически только по обучающей истории; этот эксперимент нужно запустить.</p>'
        part+=details('Измерения и прогнозы датчиков (модель своего месяца)',prediction_plot(directory,c,
            options['start'] or cfg['first_test'],options['end'],options['sensors']))
        final=details(title,part)
        (report_dir/'report.html').write_text(final,encoding='utf-8');show_html(final)
    result=pd.DataFrame(rows)
    if result.empty:
        show_html('<p>Готовых оценок выбранных экспериментов нет. Проверьте output, выбор моделей и журнал месячных обновлений выше.</p>')
    else:
        result.to_csv(root/'notebook_report'/'all_metrics.csv',index=False,encoding='utf-8-sig')
        preferred=['unit','method','variant','view','budget','partial','events','detected','recall','precision','average_precision','random_precision',
                   'hourly_roc_auc','hourly_precision','hourly_recall','hourly_f1','alarms_per_720h','background_fraction','lead_hours','availability_operating']
        show_html(table('Итоговая таблица: котлы, модели и все рассчитанные схемы',result.reindex(columns=preferred),len(result)<=30)+
                  table('Все метрики, включая числители и исключённые часы/события',result))
    if regression_rows:
        combined=pd.concat(regression_rows,ignore_index=True)
        combined.to_csv(root/'notebook_report'/'regressor_sensor_metrics.csv',index=False,encoding='utf-8-sig')
        show_html(table('Все котлы и датчики: MAE, RMSE, R² за выбранный период',combined))
    return result


def saved_classifier_report(options,output):
    """Inspect saved out-of-time 05 predictions. Never load or fit a model."""
    import matplotlib.pyplot as plt
    root=Path(options.get('root','runs/direct_classifier_comparison'))
    explicit=options.get('file')
    paths=[Path(explicit)] if explicit else sorted(root.rglob('test_predictions.parquet')) if root.exists() else []
    paths=[p for p in paths if p.exists()]
    if options.get('folds','latest')=='latest' and not explicit:
        latest={}
        for p in paths:
            key=p.parent.parent if p.parent.name.startswith('fold_') else p.parent
            if key not in latest or p.parent.name>latest[key].parent.name:latest[key]=p
        paths=list(latest.values())
    if not paths:
        show_html('<p>Сохранённых test_predictions.parquet из 05 не найдено. Обучение здесь не запускается. '
                  'Укажите CLASSIFIER_VIEW.root (выходной каталог 05) либо CLASSIFIER_VIEW.file (конкретный файл).</p>')
        return pd.DataFrame()
    summary=[];out_root=Path(output)/'notebook_report'/'classifier_inspection'
    for number,path in enumerate(paths,1):
        d=pd.read_parquet(path)
        if not {'score','label','evaluation_eligible'}.issubset(d):
            show_html('<p>Неподходящий формат: '+escape(str(path))+'</p>');continue
        settings_path=path.parent/'settings.json'
        settings=json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
        rule_name=options.get('threshold_rule','best_f2')
        rules=settings.get('thresholds') or {}
        threshold=float(rules.get(rule_name,{}).get('threshold',.5))
        fallback=rule_name not in rules
        rule=rules.get(rule_name,{})
        if rule_name!='default_05' and (fallback or not np.isfinite(threshold) or not rule.get('available',True)
                or any(s in rule.get('status','') for s in ('fallback','threshold_unavailable','not_evaluable'))):
            show_html('<p>'+escape(str(path))+' · выбранный порог '+escape(rule_name)+
                ' не подобран. Для просмотра фиксированного baseline укажите threshold_rule="default_05". Подстановка 0.5 не выполняется.</p>')
            continue
        units=d.index.get_level_values('unit').unique() if isinstance(d.index,pd.MultiIndex) and 'unit' in d.index.names else ['all']
        for unit in units:
            if options.get('units') and unit not in options['units']:continue
            part=d.xs(unit,level='unit').copy() if unit!='all' else d.copy()
            if isinstance(part.index,pd.MultiIndex):
                part.index=pd.DatetimeIndex(part.index.get_level_values('time'))
            else:part.index=pd.to_datetime(part.index)
            part=part.sort_index().loc[options.get('start'):options.get('end')]
            if part.empty:continue
            if 'model_available_at' in part:
                causal=pd.to_datetime(part.model_available_at).le(part.index)
            elif settings.get('available_at'):
                causal=pd.Series(part.index>=pd.Timestamp(settings['available_at']),index=part.index)
            else:
                show_html('<p>'+escape(str(path))+' · не указана дата доступности модели: независимость оценок не подтверждена. '
                          'Используйте тестовый экспорт актуального 05.</p>');continue
            if not causal.all():
                show_html('<p>'+escape(str(path))+' · исключены оценки до даты доступности модели: '+str(int((~causal).sum()))+'</p>')
                part=part.loc[causal]
            eligible=part.evaluation_eligible.eq(True)&part.label.isin([0,1])&np.isfinite(part.score)
            tested=part.loc[eligible];y=tested.label.astype(int);score=tested.score
            pred=score.ge(threshold).astype(int)
            out=out_root/f'model_{number:03d}'/str(unit)
            html,metrics=classification_html(y,score,pred,out,'Классификатор 05: выбранный порог')
            html='<p>Источник: <code>'+escape(str(path.resolve()))+'</code></p>'+html
            html+='<p>Используются сохранённые test-оценки. Обучение и подбор порога здесь не выполняются. '
            html+='Положительный класс — предаварийные часы горизонта этой модели; остальные виды аномалий он не размечает.</p>'
            status='fallback 0.5: такого правила нет в сохранённых настройках' if fallback else str(rules[rule_name].get('status','сохранённый validation-порог'))
            html+='<p>Порог '+escape(rule_name)+f' = {threshold:.4f}; '+escape(status)+'.</p>'
            fig,axes=plt.subplots(2,1,figsize=(14,6),gridspec_kw={'height_ratios':[2,1]})
            grid=pd.date_range(part.index[0],part.index[-1],freq=settings.get('alarm_step','1h'))
            axes[0].plot(grid,part.score.reindex(grid),lw=.8,label='Балл классификатора')
            axes[0].axhline(threshold,color='red',ls='--',label='Сохранённый порог')
            axes[0].fill_between(part.index,0,1,where=(part.label.eq(1)&part.evaluation_eligible.eq(True)).to_numpy(bool),
                transform=axes[0].get_xaxis_transform(),alpha=.15,color='red',label='Размеченные предаварийные часы')
            axes[0].set_ylabel('Балл риска');axes[0].legend(fontsize=8)
            for label,title,color in ((0,'Фон','#3498db'),(1,'Перед аварией','#e74c3c')):
                values=score.loc[y.eq(label)]
                if len(values):axes[1].hist(values,bins=np.linspace(0,1,31),density=True,alpha=.5,label=title,color=color)
            axes[1].axvline(threshold,color='red',ls='--');axes[1].set_xlabel('Балл риска');axes[1].set_ylabel('Плотность');axes[1].legend(fontsize=8)
            fig.tight_layout();html+=figure_html(fig,out/'normal_vs_pre_failure.png')
            html+=table('Сохранённые оценки всех часов выбранного участка',part.reset_index())
            show_html(details(str(unit)+' · классификатор · '+str(path.parent.relative_to(root)) if path.is_relative_to(root) else str(unit)+' · '+str(path.parent),html))
            summary.append(dict(unit=unit,source=str(path),threshold_rule=rule_name,threshold=threshold,**metrics))
    result=pd.DataFrame(summary)
    if len(result):
        out_root.mkdir(parents=True,exist_ok=True);result.to_csv(out_root/'metrics.csv',index=False,encoding='utf-8-sig')
        show_html(table('Классификаторы 05: просмотр готовых тестовых оценок',result,len(result)<=20))
    return result
