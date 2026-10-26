"""Per-target diagnostics and durable initial-fit recovery, outside model state."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
import time
import sys
import uuid
import joblib
import numpy as np
import pandas as pd
from .artifacts import _file_operation

_store=ContextVar('initial_target_store',default=None)
_last_failed_fit=None


@lru_cache(maxsize=1)
def _versions():
    return {name:importlib.metadata.version(name) for name in ('numpy','pandas','catboost','scikit-learn','joblib')}


def training_scope(method, native, targets, cfg):
    params={key:cfg[key] for key in ('cb_initial_trees','cb_depth','learning_rate','seed','threads','min_sensor_rows')}
    description=[method,list(native.columns),list(targets.columns),params,_versions()]
    digest=hashlib.sha256(json.dumps(description,ensure_ascii=False,sort_keys=True).encode())
    for frame in (native,targets):
        digest.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    return digest.hexdigest()[:24]


class TargetStore:
    def __init__(self, folder, token):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.token=token
        self.rescue=None
        path=self.folder/'fingerprint.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))['fingerprint']!=token:
            raise ValueError('Initial-target cache is incompatible; use a new output directory: '+str(path))
        if not path.exists():
            temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                temporary.write_text(json.dumps({'fingerprint':token}),encoding='utf-8')
                _file_operation(lambda:os.replace(temporary,path))
            finally:
                if temporary.exists():_file_operation(temporary.unlink)

    def scope(self, method, native, targets, cfg):
        return training_scope(method,native,targets,cfg)

    def path(self, scope, target):
        key=hashlib.sha256(str(target).encode()).hexdigest()[:24]
        return self.folder/scope/(key+'.joblib')

    def restore(self, scope, target):
        path=self.path(scope,target)
        if not path.exists():
            rescued=self.folder/'rescued_targets.joblib'
            if self.rescue is None:
                self.rescue=joblib.load(rescued) if rescued.exists() else {}
            if self.rescue and self.rescue.get('scope')!=scope:
                raise ValueError('Спасённые модели не соответствуют данным или параметрам обучения. '
                                 'Сохраните прежние данные/параметры либо выберите другую output.')
            if self.rescue.get('scope')==scope and target in self.rescue.get('predictors',{}):
                predictor=self.rescue['predictors'][target]
                status='trained' if predictor[0]=='catboost' else predictor[0]
                self.save(scope,target,predictor,status)
                return dict(fingerprint=self.token,target=target,predictor=predictor,status=status)
            return None
        data=joblib.load(path)
        if data['fingerprint']!=self.token or data['target']!=target:
            raise ValueError('Initial-target cache does not match the current input: '+str(path))
        return data

    def save(self, scope, target, predictor, status):
        path=self.path(scope,target);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            joblib.dump(dict(fingerprint=self.token,target=target,predictor=predictor,status=status),temporary,compress=3)
            _file_operation(lambda:os.replace(temporary,path))
        finally:
            if temporary.exists():_file_operation(temporary.unlink)


@contextmanager
def initial_targets(folder, token):
    marker=_store.set(TargetStore(folder,token))
    try:yield
    finally:_store.reset(marker)


def current_store():
    return _store.get()


def rescue_initial(new_output, exception=None):
    """After interrupting an old notebook, salvage completed CatBoost targets.

    Call before restarting the kernel. No raw training frame is written; its hash
    gates later reuse together with model parameters and installed library versions.
    """
    global _last_failed_fit
    error=exception or getattr(sys,'last_exc',None) or getattr(sys,'last_value',None) or _last_failed_fit
    if error is None:raise ValueError('Нет сохранённого исключения. Сначала прервите ячейку, не перезапуская ядро.')
    _last_failed_fit=error
    frame=error.__traceback__;fit=None;monthly=None
    while frame:
        local=frame.tb_frame.f_locals
        if frame.tb_frame.f_code.co_name=='_fit_trees' and local.get('initial'):
            model=local.get('self')
            if getattr(model,'method',None)=='CatBoost':fit=local
        if frame.tb_frame.f_code.co_name=='update_at' and local.get('previous') is None:monthly=local
        frame=frame.tb_next
    if fit is None or monthly is None:
        raise ValueError('В traceback нет незавершённого первого обучения CatBoost. После перезапуска ядра память восстановить нельзя.')
    model=fit['self'];spec=monthly['spec'];cutoff=pd.Timestamp(monthly['cutoff'])
    if spec['method']!='CatBoost' or spec['variant'] not in ('conditions','forecast'):
        raise ValueError('Спасение старого запуска поддерживает обычный CatBoost conditions/forecast.')
    for name,predictor in model.predictors.items():
        kind,value=predictor
        if kind=='catboost':
            if (value.tree_count_!=model.cfg['cb_initial_trees'] or
                value.get_param('loss_function')!='MultiQuantile:alpha=0.25,0.5,0.75'):
                raise ValueError('Неожиданные параметры модели: '+str(name))
        elif kind not in ('constant','reference'):
            raise ValueError('Неожиданный тип модели: '+str(kind))
    if not model.predictors:
        raise ValueError('Пока нет ни одной завершённой модели датчика для сохранения.')
    scope=training_scope(model.method,model.encoder.native(fit['features']),fit['targets'],model.cfg)
    name='__'.join([spec['group'],spec['scenario'],'+'.join(spec['sources']),spec['method'],spec['variant'],str(spec['horizon'])])
    if '..' in name or any(c in name for c in '<>:"/\\|?*'):raise ValueError('Недопустимое имя эксперимента')
    folder=Path(new_output).resolve()/'experiments'/name/'partial'/f'{cutoff:%Y-%m-%d}'
    folder.mkdir(parents=True,exist_ok=True);path=folder/'rescued_targets.joblib'
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        joblib.dump(dict(scope=scope,predictors=model.predictors),temporary,compress=3)
        _file_operation(lambda:os.replace(temporary,path))
    finally:
        if temporary.exists():_file_operation(temporary.unlink)
    result=dict(saved_targets=len(model.predictors),total_targets=len(model.targets),path=str(path))
    print('Сохранено завершённых целей:',result['saved_targets'],'из',result['total_targets'])
    print('Новая output:',str(Path(new_output).resolve()))
    print('После перезапуска ядра используйте эту output и те же данные/параметры.')
    return result


class TargetProgress:
    def __init__(self, progress, method, count, initial):
        self.progress=progress;self.method=method;self.count=count;self.initial=initial
        self.times=[];self.started=0.;self.label='';self.index=0;self.iterations=0

    def inputs(self, native, cats, cfg):
        if not self.progress:return
        cardinality={col:int(native[col].nunique(dropna=False)) for col in cats}
        top=sorted(cardinality.items(),key=lambda item:item[1],reverse=True)[:5]
        self.progress.say(f'{self.method}: входов {len(native.columns)}, числовых {len(native.columns)-len(cats)}, '
            f'категориальных {len(cats)}, строк {len(native)}, потоков {cfg["threads"]} '
            f'из {os.cpu_count()} логических CPU, глубина {cfg["cb_depth"]}',
            status='fit_inputs',features=len(native.columns),categorical_features=len(cats),rows=len(native),
            threads=cfg['threads'],logical_cpus=os.cpu_count(),largest_categories=top)
        if top:
            self.progress.say('Наибольшее число категорий: '+', '.join(f'{name}: {n}' for name,n in top),
                              status='category_cardinality')

    def start(self, index, name, rows, trees):
        self.index=index;self.name=name;self.iterations=trees;self.started=time.perf_counter()
        self.label=f'{self.method} · цель {index}/{self.count} «{name}»'
        if self.progress:
            self.progress.set_detail(self.label+' · подготовка / начало fit')
            self.progress.say(self.label+f' · начало, {rows} пригодных строк, до {trees} деревьев',
                              status='target_started',target=name,target_index=index,target_count=self.count,
                              rows=int(rows),iterations=trees)

    def after_iteration(self, info):
        if self.progress:
            self.progress.set_detail(self.label+f' · деревья {info.iteration}/{self.iterations}',
                                     iteration=int(info.iteration),iterations=self.iterations)
        return True

    def done(self, status, cached=False):
        elapsed=time.perf_counter()-self.started
        if status=='trained' and not cached:self.times.append(elapsed)
        eta=(float(np.mean(self.times[-10:]))*(self.count-self.index) if len(self.times)>=3 else None)
        if self.progress:
            suffix=f' · ориентир на оставшиеся цели: {eta/60:.1f} мин' if eta is not None else ''
            outcome=('восстановлено: ' if cached else '')+status
            self.progress.say(self.label+f' · {outcome} · {elapsed:.2f} с'+suffix,
                status='target_complete',target=self.name,target_index=self.index,target_count=self.count,
                result=status,restored=cached,target_seconds=round(elapsed,3),estimated_remaining_seconds=eta)
            self.progress.set_detail(f'Обработано целей {self.index}/{self.count}; последняя «{self.name}»: {status}')
