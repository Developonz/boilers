"""Portable inference. Its public Result interface also serves the existing dashboard."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import importlib.metadata
import sys
import joblib
import numpy as np
import pandas as pd
from .data import prepare_unit
from .pipeline import score_context
from .preparation import regimes
from .alarm_scores import alarm_stream


def export(state, context, spec, cfg, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    package = dict(state=state, spec=spec, config=cfg, roles=context['roles'], unit=context['unit'],
                   threshold=context['threshold'], rules=json.loads(Path(cfg['sensor_rules']).read_text(encoding='utf-8')),
                   declared_units=context.get('declared_units',{}),
                   source_load_name=next((r['original'] for r in context.get('original_roles',[]) if r['role']=='load'),context['load']))
    joblib.dump(package, folder/'state.joblib', compress=3)
    code = folder/'monitoring_v3'; code.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, code/source.name)
    shutil.copy2(Path(__file__).parent.parent/'kernel_models.py', folder/'kernel_models.py')
    libraries=('numpy','pandas','scipy','scikit-learn','catboost','joblib','threadpoolctl','openpyxl','pyarrow','matplotlib')
    versions={name:importlib.metadata.version(name) for name in libraries}
    (folder/'environment.json').write_text(json.dumps(dict(python=sys.version,libraries=versions),indent=2),encoding='utf-8')
    (folder/'requirements.txt').write_text('\n'.join(name+'=='+version for name,version in versions.items())+'\n',encoding='utf-8')
    loader = '''from pathlib import Path
import hashlib, json, sys

def load(folder=None):
    root = Path(folder or Path(__file__).parent).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Export file changed: '+name)
    # Prevent silently applying a different cached implementation in a long-lived notebook.
    for name in ('monitoring_v3', 'kernel_models'):
        for key, module in list(sys.modules.items()):
            if (key == name or key.startswith(name+'.')) and getattr(module, '__file__', None):
                source = Path(module.__file__)
                relative = Path('kernel_models.py') if name == 'kernel_models' else Path('monitoring_v3')/source.name
                copy = root/relative
                if copy.exists() and source.exists() and source.read_bytes() != copy.read_bytes():
                    raise RuntimeError('Different runtime already imported; use a fresh Python process: '+key)
    sys.path.insert(0, str(root))
    try:
        from monitoring_v3.runtime import Model
        return Model(root)
    finally:
        sys.path.remove(str(root))
'''
    (folder/'model.py').write_text(loader, encoding='utf-8')
    thresholds = state['thresholds'].get(context['unit']) or next(iter(state['thresholds'].values()))
    views = [dict(name=v, top=1, slow_h=cfg['slow_hours'] if v.endswith('slow48') else 0,
                  thr=float(t[str(cfg['alarm_budget'])]['threshold'])) for v,t in thresholds.items()]
    metadata = dict(version='monitoring-3.0', core='incremental', input='conditions', title=spec['method'],
                    run_col=context['load'], run_thr=context['threshold'], causal_cleaning=True,
                    trained_until=str(state['cutoff']-pd.Timedelta(cfg['step'])), step=cfg['step'],
                    startup_steps=cfg['startup_hours'], smooth_steps=cfg['smooth_hours'],
                    persist_steps=cfg['persist_hours'], views=views,
                    input_aliases=context.get('aliases', {}), training_units=spec['sources'],
                    group=spec['group'], cond_spec=dict(mills={}),
                    training_update='incremental', calibration_strategy='current_version_heldout',
                    label_status='causes require reviewed episodes')
    (folder/'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')
    rows = []
    for c,r in context['roles'].items():
        row = {'датчик':c, 'роль':'проверяется' if r == 'target' else 'условие' if r in ('condition','load') else 'состояние механизма'}
        for v in thresholds:
            row['обычная ошибка: '+v] = state['scale'].get(c, np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder/'sensors.csv', index=False, encoding='utf-8-sig')
    (folder/'scaler.json').write_text(json.dumps(dict(num=[c for c,r in context['roles'].items() if r != 'state'],
                                                    bools=[]), ensure_ascii=False), encoding='utf-8')
    manifest = {str(p.relative_to(folder)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in str(p)}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return folder


class Model:
    def __init__(self, folder):
        self.folder = Path(folder)
        package = joblib.load(self.folder/'state.joblib')
        self.__dict__.update(package)
        self.cfg = json.loads((self.folder/'config.json').read_text(encoding='utf-8'))
        self.cols = self.state['model'].targets
        self.q = {v['name']:self.state['scale'].to_numpy() for v in self.cfg['views']}

    def needed(self):
        return list(self.roles)

    def operating_mode(self, frame):
        raw=frame.rename(columns=self.cfg.get('input_aliases',{}))
        return regimes(raw[self.cfg['run_col']],self.threshold,self.config)['mode']

    def score(self, frame, prepared=False):
        raw = frame.rename(columns=self.cfg.get('input_aliases', {})).copy()
        if not isinstance(raw.index, pd.DatetimeIndex):
            raw.index = pd.to_datetime(raw.index)
        if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
            raise ValueError('Inference requires unique, increasing timestamps')
        raw = raw.asfreq(self.config['step'])
        missing = [c for c in self.roles if c not in raw]
        for c in missing:
            raw[c] = np.nan
        # Repair labels are evaluation metadata; predictions cannot depend on them.
        raw['Состояние'] = 'Нет данных'
        context = prepare_unit(raw, self.roles, self.unit, self.config, self.rules, threshold=self.threshold)
        views, matrices, errors, features = score_context(self.state, context, self.spec, self.config)
        hourly, shares, episodes = {}, {}, []
        for name, data in views.items():
            stream, eps = alarm_stream(data.ratio, self.config)
            hourly[name, 'ошибка'] = data.score
            hourly[name, 'сглаженная'] = data.smoothed
            hourly[name, 'порог'] = data.threshold
            hourly[name, 'отношение'] = data.ratio
            hourly[name, 'выше порога'] = data.ratio.gt(1)
            hourly[name, 'тревога'] = stream.active
            hourly[name, 'уведомление'] = stream.notification
            hourly[name, 'покрытие'] = data.coverage
            a = matrices[name]
            shares[name] = a.div(a.sum(axis=1).replace(0,np.nan), axis=0)*100
            for e in eps.itertuples():
                episodes.append({'вид':name, 'начало':e.confirmed, 'конец':e.end,
                                 'часов':float((e.end-e.confirmed)/pd.Timedelta('1h')), 'пик':e.peak,
                                 'главные датчики':'', 'первое превышение':e.start})
        from .pipeline import dataset
        f,y = dataset(context, self.spec['variant'], self.spec['horizon'], self.config)
        med, spread = self.state['model'].expected(f, y)
        expected = pd.concat({'ожидание':med, 'разброс':spread, 'факт':y}, axis=1).swaplevel(0,1,axis=1).sort_index(axis=1)
        interpretation = pd.DataFrame(dict(label='unknown', confidence=np.nan), index=raw.index)
        head = self.folder/'interpreter.joblib'
        if head.exists():
            from .interpretation import predict_head
            package=joblib.load(head)
            interpretation = predict_head(package, features)
            if package.get('available_at'):
                interpretation.loc[interpretation.index < pd.Timestamp(package['available_at']),:]=['unknown',np.nan]
        for hours in self.config['risk_horizons']:
            risk_head=self.folder/f'risk_{hours}h.joblib'
            if risk_head.exists():
                from .interpretation import predict_head
                package=joblib.load(risk_head)
                p=package['model'].predict_proba(package['encoder'].native(features))
                probabilities=package['calibrator'].predict_proba(np.log(np.clip(p,1e-9,1.)))
                classes=list(package['calibrator'].classes_)
                interpretation[f'risk_{hours}h']=probabilities[:,classes.index('1.0')] if '1.0' in classes else np.nan
                if package.get('available_at'):
                    interpretation.loc[interpretation.index < pd.Timestamp(package['available_at']),f'risk_{hours}h']=np.nan
        interpretation.loc[~context['eligible'],'label']='unknown'
        return SimpleNamespace(hourly=pd.DataFrame(hourly, index=raw.index), mode=context['regime']['mode'],
                               sensor_errors=matrices, contrib_share=shares, expected=expected,
                               alarms=pd.DataFrame(episodes, columns=['вид','начало','конец','часов','пик','главные датчики','первое превышение']),
                               missing=missing, interpretation=interpretation, quality=context['quality'])
