"""Local phase timings and periodic output during long numerical operations."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event, Lock, Thread
import json
import time
import uuid
from .progress_display import notebook_display

_active=ContextVar('training_progress',default=None)
_phase=ContextVar('training_phase',default='')


def current_progress():
    return _active.get()


@contextmanager
def phase(label):
    """A timed substage; keep model code usable without a pipeline logger."""
    progress=current_progress()
    if progress is None:
        yield
    else:
        parent=_phase.get()
        with progress.stage(parent+' · '+label if parent else label):
            yield


class PredictionProgress:
    def __init__(self, count, rows):
        self.progress=current_progress();self.count=count;self.rows=rows

    def start(self, index, target):
        self.index=index;self.target=target;self.started=time.perf_counter()
        self.label=f'прогноз · датчик {index}/{self.count} «{target}»'
        if self.progress:
            self.progress.set_detail(self.label)
            self.progress.say(self.label+f' · начало, {self.rows} строк',status='prediction_started',
                target=target,target_index=index,target_count=self.count,rows=self.rows,display=index==1)

    def done(self, result='predicted'):
        elapsed=time.perf_counter()-self.started
        if self.progress:
            self.progress.say(self.label+f' · {result} · {elapsed:.3f} с',status='prediction_complete',
                target=self.target,target_index=self.index,target_count=self.count,
                result=result,prediction_seconds=round(elapsed,6),
                display=self.index%25==0 or self.index==self.count or elapsed>=1.)
            self.progress.set_detail(f'Прогноз обработан для {self.index}/{self.count} датчиков')


class Progress:
    def __init__(self, root, interval=15):
        self.path=Path(root)/'progress.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.interval=interval
        self.started=time.perf_counter()
        self.run_id=uuid.uuid4().hex
        self.lock=Lock()
        self.detail=''
        self.visible_stage=None
        self.view=notebook_display()
        self.live=dict(phase='Подготовка',phase_started=self.started,status='running',
                       target_count=0,completed=0,detail='',notice='',recent=[])

    def _render_locked(self, force=False):
        if self.view is None:return
        now=time.perf_counter()
        snapshot=dict(self.live,elapsed_seconds=now-self.started,
                      phase_seconds=now-self.live['phase_started'],
                      operation_seconds=now-self.live.get('operation_started',now))
        snapshot['recent']=list(self.live['recent'])
        try:self.view.update(snapshot,force=force)
        except Exception:
            # A notebook renderer must never stop numerical work or its journal.
            self.view=None
            print('Обновляемый индикатор недоступен; продолжение с текстовым журналом.',flush=True)

    def _observe_locked(self, message, fields):
        status=fields.get('status');now=time.perf_counter()
        if status=='started':
            self.live.update(phase=fields['phase'],phase_started=now,status='running',detail='',
                             target_count=0,completed=0,target=None,iteration=0,iterations=0,eta=None)
        elif status in ('target_started','prediction_started'):
            self.live.update(status='running',target=fields['target'],target_index=fields['target_index'],
                target_count=fields['target_count'],completed=fields['target_index']-1,
                iteration=0,iterations=fields.get('iterations',0),operation_started=now,detail=message)
        elif status in ('target_complete','prediction_complete'):
            self.live.update(completed=fields['target_index'],iteration=0,iterations=0,target=None,detail=message,
                             eta=fields.get('estimated_remaining_seconds'))
        elif status=='complete' and fields.get('phase'):
            self.live['recent'].append(f'{fields["phase"]} — {now-self.live["phase_started"]:.2f} с')
            self.live['recent']=self.live['recent'][-5:]
        elif status=='complete':
            self.live.update(status='finished',phase='Расчёт завершён',target=None,detail='',notice=message)
        elif status=='failed':
            self.live.setdefault('failure',message)
            self.live.update(status='failed',notice=self.live['failure'])
        elif status not in ('running',):
            self.live['notice']=message

    def set_detail(self, text, **fields):
        with self.lock:
            self.detail=text
            self.live.update(detail=text,**fields)
            self._render_locked()

    def say(self, message, *, display=True, _stage=None, **fields):
        with self.lock:
            if _stage is not None and self.visible_stage is not _stage:return
            if _phase.get():fields.setdefault('phase',_phase.get())
            record=dict(run=self.run_id,elapsed_seconds=round(time.perf_counter()-self.started,3),
                        message=message,**fields)
            with self.path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            self._observe_locked(message,fields)
            force=(fields.get('status') in ('started','complete','failed','month_complete') or
                   fields.get('status') in ('target_complete','prediction_complete') and
                   fields.get('target_index')==fields.get('target_count'))
            self._render_locked(force=force)
            if display and (self.view is None or fields.get('status')=='failed'):print(message,flush=True)

    @contextmanager
    def stage(self, label):
        started=time.perf_counter()
        token=_active.set(self)
        phase_token=_phase.set(label)
        marker=object()
        with self.lock:
            parent=self.visible_stage
            parent_phase=self.live['phase'];parent_started=self.live['phase_started']
            if parent is None:self.live.pop('failure',None)
            self.visible_stage=marker;self.detail=''
        self.say('Начало: '+label,phase=label,status='started')
        stop=Event()
        def heartbeat():
            last_log=started
            while not stop.wait(min(self.interval,.5) if self.view is not None else self.interval):
                with self.lock:
                    if self.visible_stage is not marker:continue
                    detail=self.detail
                    if time.perf_counter()-last_log<self.interval:
                        self._render_locked()
                        continue
                suffix=' · '+detail if detail else ''
                self.say(f'Выполняется: {label}{suffix} · {time.perf_counter()-started:.0f} с',
                         phase=label,status='running',_stage=marker)
                last_log=time.perf_counter()
        thread=Thread(target=heartbeat,daemon=True);thread.start()
        def finish():
            stop.set();thread.join()
        try:
            yield
        except BaseException as exc:
            finish()
            with self.lock:
                detail=self.detail
            suffix=' · '+detail if detail else ''
            self.say(f'Прервано: {label}{suffix} · {type(exc).__name__}: {exc}',phase=label,status='failed')
            raise
        else:
            finish()
            self.say(f'Готово: {label} · {time.perf_counter()-started:.1f} с',phase=label,status='complete')
        finally:
            finish()
            with self.lock:
                self.visible_stage=parent;self.detail=''
                if parent is not None:
                    self.live.update(phase=parent_phase,phase_started=parent_started,
                                     target=None,target_count=0,completed=0,iteration=0,iterations=0,eta=None,detail='')
                    self._render_locked(force=True)
            _phase.reset(phase_token)
            _active.reset(token)
