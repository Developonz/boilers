"""Persistent per-model progress and plot slots, independent of model settings."""
from contextlib import contextmanager
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
import json
import math
from numbers import Real
import pandas as pd
from .progress_display import NotebookDisplay,notebook_display

_report=ContextVar('live_training_report',default=None)
DEFAULT_OPTIONS=dict(enabled=True,update_every_months=1,risk_history=True,
                     view='mean::instant',budget=None,start=None,end=None)


def current_report():return _report.get()


@contextmanager
def report_scope(report):
    token=_report.set(report)
    try:yield
    finally:_report.reset(token)


class Panel:
    def __init__(self,title,directory,context,report):
        self.title=title;self.directory=Path(directory);self.report=report
        # Plots need only modes and events. Do not retain the raw 200+ channel
        # telemetry of every completed experiment in notebook display objects.
        self.context=({'regime':context['regime'][['mode']],'events':context['events']}
                      if 'regime' in context and 'events' in context else context)
        self.timeline=None;self.risk=None;self.statuses={};self.done=0;self.total=0
        self.view=None;self.chart=None;self.table=None;self.snapshot={}
        if notebook_display() is not None:
            from IPython.display import HTML,display
            self.view=NotebookDisplay(display,HTML)
            self.update(dict(phase='Ожидание',status='running'),force=True)
            self.chart=display(HTML('<p>График появится после первого рассчитанного месяца проверки.</p>'),display_id=True)
            self.table=display(HTML('<p>Классификаторы риска ещё не рассчитаны.</p>'),display_id=True)

    def update(self,snapshot,force=False):
        self.snapshot=dict(snapshot)
        if self.view:
            self.view.update(dict(snapshot,title=self.title,month_done=self.done,month_total=self.total),force=force)

    def draw(self):
        if self.timeline is None:return
        from .training_plot import plot_training_timeline
        from IPython.display import Image,HTML
        import matplotlib.pyplot as plt
        options=self.report.options
        figure,table=plot_training_timeline(self.timeline,self.context,self.report.cfg,self.title,
            self.risk,self.statuses,view=options['view'],start=options['start'],end=options['end'])
        buffer=BytesIO()
        try:
            figure.savefig(buffer,format='png',dpi=105,bbox_inches='tight')
            self.directory.mkdir(parents=True,exist_ok=True)
            (self.directory/'training_plot.png').write_bytes(buffer.getvalue())
            table.to_csv(self.directory/'alarm_risk.csv',index=False,encoding='utf-8-sig')
            if self.chart:self.chart.update(Image(data=buffer.getvalue()))
            from .risk_reporting import readiness_rows
            ready=readiness_rows(self.statuses,self.report.cfg)
            content=('<p><b>Исторический риск:</b> отдельные модели на начало каждого месяца. '
                'Максимум внутри тревоги — последующее развитие, не прогноз при её начале.</p>'
                if self.risk is not None else '<p>Исторический риск пока не рассчитан. «Нет оценки» не означает низкий риск.</p>')
            if not options['risk_history']:content='<p>Исторические подписи риска выключены в LIVE.risk_history. Итоговые головы показаны ниже.</p>'
            if len(ready):content+='<details open><summary>Готовность итоговых классификаторов</summary>'+ready.to_html(index=False,escape=True)+'</details>'
            content+='<details><summary>Все тревоги и результаты риска</summary>'+table.to_html(index=False,escape=True,na_rep='—')+'</details>'
            if self.table:self.table.update(HTML(content))
        finally:plt.close(figure)


class Fanout:
    def __init__(self,panels):self.panels=panels
    def update(self,snapshot,force=False):
        for panel in self.panels:panel.update(snapshot,force)


class LiveTrainingReport:
    def __init__(self,cfg,options=None):
        self.cfg=cfg;self.options={**DEFAULT_OPTIONS,**(options or {})};self.panels={};self.warnings=[]
        unknown=set(self.options)-set(DEFAULT_OPTIONS)
        if unknown:raise ValueError('Неизвестные настройки LIVE: '+str(sorted(unknown)))
        for key in ('enabled','risk_history'):
            if not isinstance(self.options[key],bool):raise ValueError('LIVE.'+key+' должен быть bool')
        interval=self.options['update_every_months']
        if isinstance(interval,bool) or not isinstance(interval,int) or interval<1:raise ValueError('LIVE.update_every_months должен быть целым >= 1')
        if not isinstance(self.options['view'],str) or not self.options['view']:raise ValueError('LIVE.view должен быть названием схемы')
        view=self.options['view']
        standard={name+'::'+mode for name in ('mean','max','top5','fraction') for mode in ('instant','slow48')}
        if view not in standard|{'rare_changes::event'} and not (view.startswith('group:') and view.rsplit('::',1)[-1] in ('instant','slow48')):
            raise ValueError('LIVE.view: неизвестная схема '+view)
        budget=self.options['budget']
        if budget is not None and (isinstance(budget,bool) or not isinstance(budget,Real) or not math.isfinite(budget) or budget not in cfg['alarm_budgets']):
            raise ValueError('LIVE.budget должен входить в SETTINGS.alarm_budgets либо быть None')
        dates={k:pd.Timestamp(self.options[k]) for k in ('start','end') if self.options[k] is not None}
        if any(pd.isna(v) or v.tz is not None for v in dates.values()):raise ValueError('LIVE.start/end: нужна локальная дата без timezone')
        if dates.get('start',pd.Timestamp(cfg['first_test']))>dates.get('end',pd.Timestamp.max):raise ValueError('LIVE.end раньше начала графика')

    def safely(self,method,*args,**kwargs):
        if not self.options['enabled']:return
        try:return getattr(self,method)(*args,**kwargs)
        except Exception as exc:
            message=f'{method}: {type(exc).__name__}: {exc}'
            self.warnings.append(message)
            root=Path(self.cfg['output']);root.mkdir(parents=True,exist_ok=True)
            (root/'live_report_warnings.json').write_text(json.dumps(self.warnings,ensure_ascii=False,indent=2),encoding='utf-8')
            print('График/индикатор недоступен; обучение продолжается. '+message,flush=True)

    def start(self,name,spec,contexts,months,progress):
        panels=[]
        for unit in spec['recipients']:
            horizon=f' · {spec["horizon"]} ч' if spec.get('horizon') else ''
            donor=' · донор '+', '.join(spec.get('sources',[])) if spec['scenario']=='transfer' else ''
            panel=Panel(f'{spec["method"]} · {unit} · {spec["variant"]}{horizon} · {spec["scenario"]}{donor}',
                Path(self.cfg['output'])/'experiments'/name/unit,contexts[unit],self)
            panel.total=len(months);self.panels[(name,unit)]=panel;panels.append(panel)
        self.bind(progress,panels)

    def bind(self,progress,panels):
        if panels and all(p.view is not None for p in panels):progress.view=Fanout(panels)

    def month(self,name,spec,number,pieces,contexts,progress):
        from .alarm_scores import alarm_stream,view_policy
        import numpy as np
        view=self.options['view'];budget=float(self.options['budget'] or self.cfg['alarm_budget'])
        for unit in spec['recipients']:
            panel=self.panels[(name,unit)];panel.done=number
            parts=pieces[unit].get(view,[])
            if parts:
                d=pd.concat(parts).sort_index()
                # Include pre-model gaps within this elapsed history.
                ix=contexts[unit]['frame'].index
                ix=ix[(ix>=min(pd.Timestamp(self.cfg['first_test']),d.index[0]))&(ix<=d.index[-1])]
                d=d.reindex(ix)
                threshold=d['threshold_budget_'+str(budget)]
                d['ratio']=d.smoothed/threshold.where(np.isfinite(threshold))
                stream,_=alarm_stream(d.ratio,view_policy(self.cfg,view))
                d['active']=stream.active;d['notification']=stream.notification
                d['eligible']=contexts[unit]['eligible'].reindex(ix).fillna(False)
                d['background_eligible']=contexts[unit]['frame']['Состояние'].reindex(ix).eq('Работа')
                panel.timeline=d
                if number%self.options['update_every_months']==0 or number==panel.total:panel.draw()
        progress.set_detail(progress.detail)

    def finish(self,name,spec,progress,available=True):
        for unit in spec['recipients']:
            p=self.panels[(name,unit)];p.done=p.total
            label='Детектор рассчитан; риск ожидает интерпретации' if available else 'Модель не построена: недостаточно пригодной истории'
            p.update(dict(p.snapshot,phase=label,status='finished'),force=True)

    def select_items(self,items,progress):
        panels=[self.panels[(i['experiment'],i['unit'])] for i in items if (i['experiment'],i['unit']) in self.panels]
        self.bind(progress,panels)

    def risk(self,item,statuses,risk,context=None):
        panel=self.panels.get((item['experiment'],item['unit']))
        if panel is None:
            directory=Path(item['directory'])
            path=directory/(self.options['view'].replace(':','_')+'__'+str(float(self.options['budget'] or self.cfg['alarm_budget']))+'.parquet')
            if not path.exists():return
            if context is None:return
            panel=Panel(item['method']+' · '+item['unit']+' · '+item['variant'],directory,context,self)
            panel.timeline=pd.read_parquet(path);self.panels[(item['experiment'],item['unit'])]=panel
        panel.risk=risk;panel.statuses=statuses;panel.draw()
        panel.update(dict(panel.snapshot,phase='Детектор и интерпретация завершены',status='finished'),force=True)
