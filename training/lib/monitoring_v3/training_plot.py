"""The legacy two-panel presentation, using current causal scores and risk."""
import numpy as np
import pandas as pd
from .alarm_scores import evaluate
from .risk_reporting import alarm_risk_table


def plot_training_timeline(timeline,context,cfg,title='',risk=None,statuses=None,
                           view='mean::instant',start=None,end=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from legacy_plots import _score_scale
    import k2_lib as K
    table=alarm_risk_table(timeline,risk,cfg,view,statuses)
    left=pd.Timestamp(start or cfg['first_test'])
    d=timeline.loc[left: end]
    if 'active' not in d or 'notification' not in d:
        from .alarm_scores import alarm_stream,view_policy
        stream,_=alarm_stream(timeline.ratio,view_policy(cfg,view))
        d=d.assign(active=stream.active.reindex(d.index),notification=stream.notification.reindex(d.index))
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(15,5.3),sharex=True,gridspec_kw={'height_ratios':[4,1]})
    if d.empty:
        ax.text(.5,.5,'Период проверки ещё не рассчитан',transform=ax.transAxes,ha='center')
        ax.set_title(title);fig.tight_layout();return fig,table
    right=d.index[-1]+pd.Timedelta(cfg['step']);left=max(left,d.index[0])
    modes=context['regime']['mode'].reindex(d.index)
    running=modes.isin(['работа','пуск']);stopped=modes.eq('стоит')
    ratio=d.ratio.where(np.isfinite(d.ratio)).mask(stopped,0.)
    daily=ratio.resample('1D').max()
    def spans(mask):
        values=mask.to_numpy(bool);cont=d.index.to_series().diff().eq(pd.Timedelta(cfg['step'])).to_numpy()
        starts=np.flatnonzero(values&(~np.r_[False,values[:-1]]|~cont))
        ends=np.flatnonzero(values&(~np.r_[values[1:],False]|~np.r_[cont[1:],False]))
        return [(d.index[a],d.index[b]+pd.Timedelta(cfg['step'])) for a,b in zip(starts,ends)]
    for a,b in spans(stopped):ax.axvspan(a,b,color='#bdc3c7',alpha=.35,lw=0)
    colors={'Авария':'#e74c3c','План':'#3498db','Резерв':'#f39c12','Неизвестно':'#8e44ad'}
    for event in context['events'].itertuples():
        if pd.isna(event.start) or event.start>=right or event.end<=left:continue
        ax.axvspan(event.start,event.end,color=colors.get(event.type,'#8e44ad'),alpha=.25,lw=0)
    ax.plot(daily.index,daily,color='#2c3e50',lw=1)
    ax.axhline(1,color='#c0392b',ls='--',lw=1.2);_score_scale(ax,daily)
    ax.set_ylabel('ошибка / порог')
    metrics,_=evaluate(d,context['events'],cfg,start=left,end=right)
    number=lambda v: f'{v:.3f}' if np.isfinite(v) else '—'
    ax.set_title(f'{title}\n{view}: PR AUC {number(metrics["average_precision"])} '
        f'(случайно {number(metrics["random_precision"])}), поймано аварий '
        f'{metrics["detected"]}/{metrics["events"]}, тревог в фоне '
        f'{number(metrics["alarms_per_720h"])}/720 ч, доля фона в тревоге '
        f'{number(100*metrics["background_fraction"])}%',fontsize=10)
    ax2.fill_between(d.index,1,2,where=running.to_numpy(),color='#27ae60',lw=0,step='post')
    ax2.fill_between(d.index,1,2,where=stopped.to_numpy(),color='#bdc3c7',lw=0,step='post')
    unknown=~(running|stopped)
    ax2.fill_between(d.index,1,2,where=unknown.to_numpy(),facecolor='none',edgecolor='#8e44ad',hatch='///',lw=0,step='post')
    missing=running&~np.isfinite(d.ratio)
    for a,b in spans(missing):ax.axvspan(a,b,facecolor='none',edgecolor='#95a5a6',hatch='///',lw=0,alpha=.3)
    visible=table.loc[table.end.gt(left)&table.confirmed.lt(right)] if len(table) else table
    for i,alarm in enumerate(visible.itertuples()):
        ax2.axvspan(alarm.confirmed,alarm.end,ymin=0,ymax=.5,color='#c0392b',lw=0)
        # All alarms retain an ID. Long explanations are in the full table;
        # compact labels use staggered lanes and rotation to limit overlap.
        pieces=[]
        for h in cfg['risk_horizons']:
            value=getattr(alarm,f'risk_{h}h_at_confirmation',np.nan)
            raw=getattr(alarm,f'risk_{h}h_score_at_confirmation',np.nan)
            maximum=getattr(alarm,f'risk_{h}h_max_probability',np.nan)
            if np.isfinite(value):text=f'{h}ч {value:.0%}→{maximum:.0%}'
            elif np.isfinite(raw):
                peak=getattr(alarm,f'risk_{h}h_max_score',raw)
                text=f'{h}ч балл {raw:.2f}→{peak:.2f}'
                if np.isfinite(maximum):text+=f'; позже {maximum:.0%}'
            elif np.isfinite(maximum):text=f'{h}ч нет→позже {maximum:.0%}'
            else:text=f'{h}ч нет оценки'
            pieces.append(text)
        label=f'#{alarm.alarm} '+ ' / '.join(pieces)
        ax.text(max(alarm.confirmed,left),.97-(i%3)*.26,label,transform=ax.get_xaxis_transform(),
            rotation=70,ha='right',va='top',fontsize=6.5,
            bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1),clip_on=True)
    ax2.set_ylim(0,2);ax2.set_yticks([.5,1.5],['тревога','котёл'],fontsize=9)
    ax2.grid(False);ax2.set_xlim(left,right);K.date_axis(ax2)
    handles=[Line2D([0],[0],color='#2c3e50',label='ошибка / порог (максимум за сутки; 0 — стоит)'),
        Line2D([0],[0],color='#c0392b',ls='--',label='порог'),
        Patch(color=colors['Авария'],alpha=.4,label='аварийный ремонт'),
        Patch(color=colors['План'],alpha=.4,label='плановый ремонт'),
        Patch(color=colors['Резерв'],alpha=.4,label='резерв'),
        Patch(color='#bdc3c7',label='котёл стоит'),Patch(color='#27ae60',label='котёл работает'),
        Patch(color='#c0392b',label='тревога'),Patch(facecolor='none',edgecolor='#8e44ad',hatch='///',label='режим неизвестен'),
        Patch(facecolor='none',edgecolor='#95a5a6',hatch='///',label='работа без оценки')]
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.08),fontsize=8)
    fig.text(.01,-.12,'Риск: оценка при подтверждении → максимум ПОСЛЕ подтверждения. '
        '«Балл» не является вероятностью. Даты и причины отсутствия — в таблице тревог.',fontsize=8)
    fig.tight_layout();return fig,table
