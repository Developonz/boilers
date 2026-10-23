"""Original legacy notebook plot; only the score-axis scaling is retained from 2.2."""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import k2_lib as K
import legacy_metrics as metrics


PLOT_VERSION = 'legacy-original-2panel-2.2.1'

EV_COL = {K.EV_FAIL: '#e74c3c', K.EV_PLAN: '#3498db', K.EV_RES: '#f39c12'}


def _score_scale(ax, values):
    """Keep the current scale: linear to 1, compressed above 1; no peak clipping."""
    peak = float(values.max()) if values.notna().any() else 1.
    ax.set_yscale('symlog', linthresh=1., linscale=1.)
    ax.set_ylim(0, max(1.5, peak * 1.15))
    ticks = [0., .5, 1., 2., 5.] + [10.**p for p in range(1, max(1, int(np.ceil(np.log10(max(peak, 1.))))) + 1)]
    ticks = sorted(set(t for t in ticks if t <= ax.get_ylim()[1]))
    ax.set_yticks(ticks if len(ticks) < 12 else [0., 1.] + ticks[3::max(1, len(ticks)//8)])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))


def plot_record(record, view, start=None, end=None):
    """Original two-panel layout. Cropping never rebuilds alarms or full-period metrics."""
    ns, res = record['context'], record['views'][view]
    ix, step = ns['t_ix'], ns['STEP']
    left = max(pd.Timestamp(ns['FIRST_TEST']), ix[0], pd.Timestamp(start) if start is not None else ix[0])
    right = min(ix[-1] + step, pd.Timestamp(end) if end is not None else ix[-1] + step)
    if ns.get('EVAL_END') is not None:
        right = min(right, pd.Timestamp(ns['EVAL_END']) + step)
    visible = (ix >= left) & (ix < right)
    if left >= right or not visible.any():
        raise ValueError('В выбранном периоде нет часов проверки.')
    ratio, episodes, _, valid = metrics.alarm_series(res, ns)
    # The original plot displayed zero while stopped and a daily maximum.
    # Hours with no working score remain missing rather than being painted normal.
    running = ns['running'].reindex(ix).astype(bool)
    daily = ratio.where(valid | ~running.to_numpy()).loc[visible].resample('1D').max()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 4.4), sharex=True,
                                 gridspec_kw={'height_ratios': [4, 1]})
    stops = K.episodes(running)
    for period in stops[stops.value.eq(False)].itertuples():
        ax.axvspan(period.start, period.end, color='#bdc3c7', alpha=.35, lw=0)
    for event in ns['events'].itertuples():
        ax.axvspan(event.start, event.end, color=EV_COL[event.type], alpha=.25, lw=0)
    ax.plot(daily.index, daily, color='#2c3e50', lw=1)
    ax.axhline(1, color='#c0392b', ls='--', lw=1.2)
    _score_scale(ax, daily)
    ax.set_ylabel('ошибка / порог')
    m = metrics.evaluate(res, ns)[0]
    ax.set_title(f'{view}: PR AUC {m["PR AUC"]:.3f} (случайно {m["PR AUC случайный уровень"]:.3f}), '
                 f'поймано {m["поймано аварий"]}, тревог в фоне {m["тревог в фоне в месяц"]:.1f}/мес, '
                 f'в тревоге {m["% фона в тревоге"]:.0%} времени фона', fontsize=11)
    ax2.fill_between(running.index, 1, 2, where=running.to_numpy(), color='#27ae60', lw=0, step='post')
    ax2.fill_between(running.index, 1, 2, where=~running.to_numpy(), color='#bdc3c7', lw=0, step='post')
    for alarm in episodes.itertuples():
        ax2.axvspan(alarm.start, alarm.end, ymin=0, ymax=.5, color='#c0392b', lw=0)
    ax2.set_ylim(0, 2)
    ax2.set_yticks([.5, 1.5]); ax2.set_yticklabels(['тревога', 'котёл'], fontsize=9)
    ax2.grid(False); ax2.set_xlim(left, right)
    K.date_axis(ax2)
    handles = [Line2D([0], [0], color='#2c3e50', lw=1.5,
                      label='ошибка / порог (максимум за сутки; 0 — котёл стоит)'),
               Line2D([0], [0], color='#c0392b', ls='--', lw=1.2, label='порог'),
               Patch(color=EV_COL[K.EV_FAIL], alpha=.4, label='аварийный ремонт'),
               Patch(color=EV_COL[K.EV_PLAN], alpha=.4, label='плановый ремонт'),
               Patch(color=EV_COL[K.EV_RES], alpha=.4, label='резерв'),
               Patch(color='#bdc3c7', label='котёл стоит'),
               Patch(color='#27ae60', label='котёл работает'), Patch(color='#c0392b', label='тревога')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.13), fontsize=9)
    return fig


def render_png(record, view, start=None, end=None):
    fig = plot_record(record, view, start, end)
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=105, bbox_inches='tight'); plt.close(fig)
    return stream.getvalue()
