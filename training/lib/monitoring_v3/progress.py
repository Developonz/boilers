"""Local phase timings and periodic output during long numerical operations."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event, Lock, Thread
import json
import time
import uuid

_active=ContextVar('training_progress',default=None)


def current_progress():
    return _active.get()


class Progress:
    def __init__(self, root, interval=15):
        self.path=Path(root)/'progress.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.interval=interval
        self.started=time.perf_counter()
        self.run_id=uuid.uuid4().hex
        self.lock=Lock()
        self.detail=''

    def set_detail(self, text):
        with self.lock:
            self.detail=text

    def say(self, message, **fields):
        with self.lock:
            print(message,flush=True)
            record=dict(run=self.run_id,elapsed_seconds=round(time.perf_counter()-self.started,3),
                        message=message,**fields)
            with self.path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')

    @contextmanager
    def stage(self, label):
        started=time.perf_counter()
        token=_active.set(self)
        self.set_detail('')
        self.say('Начало: '+label,phase=label,status='started')
        stop=Event()
        def heartbeat():
            while not stop.wait(self.interval):
                with self.lock:
                    detail=self.detail
                suffix=' · '+detail if detail else ''
                self.say(f'Выполняется: {label}{suffix} · {time.perf_counter()-started:.0f} с',
                         phase=label,status='running')
        thread=Thread(target=heartbeat,daemon=True);thread.start()
        def finish():
            stop.set();thread.join()
        try:
            yield
        except BaseException as exc:
            finish()
            self.say(f'Прервано: {label} · {type(exc).__name__}: {exc}',phase=label,status='failed')
            raise
        else:
            finish()
            self.say(f'Готово: {label} · {time.perf_counter()-started:.1f} с',phase=label,status='complete')
        finally:
            finish()
            self.set_detail('')
            _active.reset(token)
