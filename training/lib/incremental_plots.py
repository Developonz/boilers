"""Plots of current preparation and saved timelines; no fitting or threshold changes."""
from pathlib import Path
import numpy as np
import pandas as pd


def plot_preparation(context):
    """Draw the actual prepared load and threshold, without reading a cached PNG."""
    import matplotlib.pyplot as plt
    c=context
    units=c.get('declared_units',{}).get(c['load'],'')
    value=f'{c["threshold"]:g}'
    threshold_label=f'Порог работы = {value}'+(f' {units}' if units else '')
    source={'manual':'задан вручную','initial_past':'рассчитан автоматически'}.get(
        c['threshold_source'],c['threshold_source'])
    fig,axes=plt.subplots(2,1,figsize=(15,6),sharex=True)
    axes[0].plot(c['frame'].index,c['frame'][c['load']],linewidth=.6,label=c['load'])
    axes[0].axhline(c['threshold'],color='red',label=threshold_label)
    axes[0].set_title(f'{c["unit"]} · {threshold_label} · {source}')
    axes[0].set_ylabel('Нагрузка'+(f', {units}' if units else ''))
    axes[0].legend()
    modes={'стоит':0,'неизвестно':1,'пуск':2,'работа':3}
    axes[1].step(c['frame'].index,c['regime']['mode'].map(modes),where='post')
    axes[1].set_yticks(list(modes.values()),list(modes))
    for event in c['events'].itertuples():
        axes[0].axvspan(event.start,event.end,color='red' if event.type=='Авария' else 'gray',alpha=.15)
    fig.tight_layout()
    return fig


def plot_timeline(directory, view='mean::instant', budget=1., start=None, end=None):
    import matplotlib.pyplot as plt
    directory=Path(directory)
    slug=view.replace(':','_')+'__'+str(float(budget))
    d=pd.read_parquet(directory/(slug+'.parquet'))
    d=d.loc[start:end]
    if d.empty:
        raise ValueError('No timeline rows in this interval')
    fig,axes=plt.subplots(3,1,figsize=(15,9),sharex=True,gridspec_kw={'height_ratios':[3,1,2]})
    axes[0].plot(d.index,d.ratio,linewidth=.7,label='Отклонение / порог')
    axes[0].axhline(1,color='red',linestyle='--',label='Порог тревоги')
    axes[0].fill_between(d.index,0,1,where=d.active.fillna(False).to_numpy(bool),
                         transform=axes[0].get_xaxis_transform(),color='red',alpha=.12,label='Подтверждённая тревога')
    notice=d.loc[d.notification.fillna(False)]
    axes[0].scatter(notice.index,notice.ratio,color='red',s=18,label='Уведомление',zorder=3)
    axes[0].set_yscale('symlog',linthresh=1)
    axes[0].set_title(f'{directory.name} · {directory.parent.name} · {view} · бюджет {budget}/720 ч')
    axes[0].legend(loc='upper left',fontsize=8)
    axes[1].plot(d.index,d.coverage*100,label='Доля доступных проверяемых датчиков')
    axes[1].set_ylim(-3,103);axes[1].set_ylabel('Покрытие, %');axes[1].legend(fontsize=8)
    errors=pd.read_parquet(directory/'prequential_errors.parquet').loc[d.index]
    cols=errors.abs().quantile(.9).nlargest(min(4,len(errors.columns))).index
    for c in cols:
        axes[2].plot(errors.index,errors[c],linewidth=.6,label=c)
    axes[2].set_ylabel('Знаковый остаток');axes[2].legend(loc='upper left',fontsize=8)
    audit=directory.parents[2]/'audit'/directory.name
    events_path=audit/'events_to_review.csv'
    if events_path.exists():
        events=pd.read_csv(events_path,parse_dates=['start','end'])
        for event in events.itertuples():
            if event.start>d.index[-1] or event.end<d.index[0]:continue
            for ax in axes:
                ax.axvspan(event.start,event.end,color='red' if event.type=='Авария' else 'gray',alpha=.1)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    return fig
