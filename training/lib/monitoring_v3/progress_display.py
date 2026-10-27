"""One bounded, throttled notebook output; optional outside a Jupyter kernel."""
from html import escape
import time


def duration(seconds):
    seconds=max(0.,float(seconds))
    return f'{int(seconds//60)} мин {int(seconds%60):02d} с' if seconds>=60 else f'{seconds:.1f} с'


def render(snapshot):
    status=snapshot.get('status','running')
    color='#dc2626' if status=='failed' else '#16a34a' if status=='finished' else '#2563eb'
    total=snapshot.get('target_count',0)
    count=snapshot.get('completed',0)
    iteration=snapshot.get('iteration',0);iterations=snapshot.get('iterations',0)
    position=min(float(total),count) if total else None
    value=f' value="{position:.6f}" max="{int(total)}"' if total else ' max="1"'
    count_text=f'Датчики: {count}/{total} · {100*position/total:.1f}%' if total else 'Выполняется этап без счётчика датчиков'
    if status=='finished':value=' value="1" max="1"';count_text='Расчёт завершён'
    trees=f' · деревья {iteration}/{iterations}' if iterations else ''
    eta=(f' · ориентир на остальные цели: {duration(snapshot["eta"])}'
         if snapshot.get('eta') is not None and total else '')
    target=snapshot.get('target','')
    current=(f'<div>Текущий датчик {snapshot.get("target_index",0)}/{total}: '
             f'<b>{escape(str(target))}</b>{trees} · '
             f'{duration(snapshot.get("operation_seconds",0))}</div>' if target else '')
    recent=snapshot.get('recent',[])
    history=('<details><summary>Последние завершённые этапы</summary><ul>'+''.join(
        f'<li>{escape(str(item))}</li>' for item in recent)+'</ul></details>' if recent else '')
    note=escape(str(snapshot.get('notice','')))
    title=escape(str(snapshot.get('title','Обучение и проверка моделей')))
    months=snapshot.get('month_total',0)
    overall=('' if not months else f'<div>Месячные срезы: {snapshot.get("month_done",0)}/{months} · '
        f'{100*snapshot.get("month_done",0)/months:.1f}%</div><progress value="{snapshot.get("month_done",0)}" '
        f'max="{months}" aria-label="Месячные срезы" style="width:100%;height:14px"></progress>')
    risk_total=snapshot.get('risk_total',0)
    if risk_total:
        done=snapshot.get('risk_done',0)
        overall+=f'<div>Исторические срезы риска: {done}/{risk_total} · {100*done/risk_total:.1f}%</div>'
        overall+=f'<progress value="{done}" max="{risk_total}" aria-label="История риска" style="width:100%;height:14px"></progress>'
    return (
        '<div style="font:14px/1.5 system-ui,sans-serif;max-width:1100px;'
        'padding:12px 16px;border:1px solid #94a3b8;border-radius:8px;overflow-wrap:anywhere">'
        f'<b style="color:{color}">{title}</b>{overall}'
        f'<div><b>Этап:</b> {escape(str(snapshot.get("phase","Подготовка")))}</div>'
        f'<progress{value} aria-label="Ход текущего этапа" style="width:100%;height:20px;accent-color:{color}"></progress>'
        f'<div>{count_text}{eta} · этап: {duration(snapshot.get("phase_seconds",0))}'
        f' · всего: {duration(snapshot.get("elapsed_seconds",0))}</div>'
        f'{current}<div>{escape(str(snapshot.get("detail","")))}</div>'
        f'<div style="margin-top:6px">{note}</div>{history}</div>'
    )


class NotebookDisplay:
    def __init__(self, display, html, refresh_seconds=.4):
        self.display=display;self.html=html;self.refresh_seconds=refresh_seconds
        self.handle=None;self.updated=-float('inf')

    def update(self, snapshot, force=False):
        now=time.perf_counter()
        if self.handle is not None and not force and now-self.updated<self.refresh_seconds:return
        content=self.html(render(snapshot))
        if self.handle is None:
            self.handle=self.display(content,display_id=True)
            if self.handle is None:raise RuntimeError('Notebook display handle is unavailable')
        else:self.handle.update(content)
        self.updated=now


def notebook_display():
    try:
        from IPython import get_ipython
        shell=get_ipython()
        if shell is None or getattr(shell,'kernel',None) is None:return None
        from IPython.display import HTML,display
        return NotebookDisplay(display,HTML)
    except ImportError:
        return None
