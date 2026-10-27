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
from .pipeline import score_context,scoring_context
from .preparation import regimes
from .alarm_scores import alarm_stream,view_policy
from .checkpoint_cache import STATE_CONTRACT

EXPORT_VERSION='monitoring-3.2'


def export(state, context, spec, cfg, folder, fingerprint=None,importance_dir=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    package = dict(state=state, spec=spec, config=cfg, roles=context['roles'], unit=context['unit'],
                   regime_spec=context.get('regime_spec'),
                   threshold=context['threshold'], rules=json.loads(Path(cfg['sensor_rules']).read_text(encoding='utf-8')),
                   declared_units=context.get('declared_units',{}),
                   source_load_name=next((r['original'] for r in context.get('original_roles',[]) if r['role']=='load'),context['load']))
    joblib.dump(package, folder/'state.joblib', compress=3)
    if importance_dir:
        for file in Path(importance_dir).glob('feature_importance*.csv'):shutil.copy2(file,folder/file.name)
    code = folder/'monitoring_v3'; code.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, code/source.name)
    compatibility=Path(__file__).with_name('cache_compatibility.json')
    if compatibility.exists():shutil.copy2(compatibility,code/compatibility.name)
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
                  thr=float(t[str(cfg['alarm_budget'])]['threshold']) if np.isfinite(t[str(cfg['alarm_budget'])]['threshold']) else None,
                  kind='observed_change' if v.endswith('::event') else 'model_deviation') for v,t in thresholds.items()]
    metadata = dict(version=EXPORT_VERSION, core='incremental', input=spec['variant'], title=spec['method'],
                    run_col=context['load'], run_thr=context['threshold'], causal_cleaning=True,
                    trained_until=state['audit'].get('last_train_time',str(state['cutoff']-pd.Timedelta(cfg['step']))), step=cfg['step'],
                    available_at=str(state['cutoff']),last_train_time=state['audit'].get('last_train_time'),
                    calibration_start=state['audit'].get('calibration_start'),calibration_end=state['audit'].get('mature_end'),
                    preparation_history='full_causal_prefix',duration_units='hours',state_contract=STATE_CONTRACT,
                    operating_threshold_source=context.get('threshold_source'),
                    transfer_preparation='donor' if spec['scenario']=='transfer' else 'site',
                    normal_history_status='unsafe_retracted' if state.get('unsafe_normal') else 'ready',
                    startup_steps=cfg['startup_hours'], smooth_steps=cfg['smooth_hours'],
                    persist_steps=cfg['persist_hours'], views=views,
                    input_aliases=context.get('aliases', {}), training_units=spec['sources'],
                    group=spec['group'], cond_spec=dict(mills={}),
                    training_update='incremental', calibration_strategy='current_version_heldout',
                    label_status='causes require reviewed episodes')
    if fingerprint is not None:
        metadata['checkpoint_fingerprint']=fingerprint
    (folder/'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
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

    def normalize_input(self,frame,roles=None):
        from .input_grid import normalize_grid
        inverse={v:k for k,v in self.cfg.get('input_aliases',{}).items()}
        numeric=[inverse.get(c,c) for c,r in self.roles.items() if r!='state'] if roles is None else [r['original'] for r in roles if r['role'] in ('load','condition','target')]
        states=[inverse.get(c,c) for c,r in self.roles.items() if r=='state'] if roles is None else [r['original'] for r in roles if r['role']=='state']
        return normalize_grid(frame,self.config['step'],numeric,states)[0]

    def operating_mode(self, frame):
        raw=frame.rename(columns=self.cfg.get('input_aliases',{}))
        from .preparation import clean
        from .regime_evidence import evidence_features
        numeric=[c for c,r in self.roles.items() if r in ('condition','load')]
        raw=raw.reindex(columns=list(dict.fromkeys(list(raw)+list(self.roles))))
        cleaned,_,_=clean(raw,numeric,self.rules,self.config['step'])
        evidence=evidence_features(cleaned,self.regime_spec) if self.regime_spec else None
        return regimes(cleaned[self.cfg['run_col']],self.threshold,self.config,evidence)['mode']

    def score(self, frame, prepared=False,period_start=None):
        raw = frame.rename(columns=self.cfg.get('input_aliases', {})).copy()
        if not isinstance(raw.index, pd.DatetimeIndex):
            raw.index = pd.to_datetime(raw.index)
        if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
            raise ValueError('Inference requires unique, increasing timestamps')
        raw = raw.asfreq(self.config['step'])
        missing = [c for c in self.roles if c not in raw]
        if missing:raw=pd.concat([raw,pd.DataFrame(np.nan,index=raw.index,columns=missing)],axis=1)
        # Repair labels are evaluation metadata; predictions cannot depend on them.
        raw['Состояние'] = 'Нет данных'
        hashes=pd.util.hash_pandas_object(raw,index=True).to_numpy()
        cache=getattr(self,'_prepared_cache',None)
        if cache is not None and len(hashes)<=len(cache[0]) and np.array_equal(hashes,cache[0][:len(hashes)]):
            context=cache[1]
        else:
            context=prepare_unit(raw,self.roles,self.unit,self.config,self.rules,threshold=self.threshold,
                regime_spec=self.regime_spec,build_events=False)
            self._prepared_cache=(hashes,context)
        start=pd.Timestamp(period_start) if period_start is not None else raw.index[0]
        context=scoring_context(context,self.spec,self.config,start,raw.index[-1]+pd.Timedelta(self.config['step']))
        raw=context['frame']
        views,matrices,errors,features,med,spread=score_context(self.state,context,self.spec,self.config,return_expected=True)
        hourly, shares, episodes = {}, {}, []
        for name, data in views.items():
            stream, eps = alarm_stream(data.ratio, view_policy(self.config,name))
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
        from .risk_dataset import alarm_features
        primary,_=alarm_stream(views['mean::instant'].ratio,self.config)
        features=alarm_features(features,primary,self.config).drop(columns=['episode_group'],errors='ignore')
        expected = pd.concat({'ожидание':med, 'разброс':spread, 'факт':y}, axis=1).swaplevel(0,1,axis=1).sort_index(axis=1)
        interpretation = pd.DataFrame(dict(label='unknown', confidence=np.nan), index=raw.index)
        head = self.folder/'interpreter.joblib'
        if head.exists():
            from .interpretation import predict_head
            package=joblib.load(head)
            interpretation = predict_head(package, features)
            if package.get('available_at'):
                interpretation.loc[interpretation.index < pd.Timestamp(package['available_at']),['label','confidence']]=['unknown',np.nan]
        for risk_head in sorted(self.folder.glob('risk_*h.joblib')):
            hours=risk_head.stem[len('risk_'):-1]
            if risk_head.exists():
                from .interpretation import predict_head
                package=joblib.load(risk_head)
                from .classification import probabilities as head_probabilities
                probability,classes=head_probabilities(package,features,require_calibration=True)
                raw_score,raw_classes=head_probabilities(dict(package,calibrator=None),features)
                interpretation[f'risk_{hours}h']=probability[:,classes.index('1.0')] if '1.0' in classes else np.nan
                interpretation[f'risk_{hours}h_score']=raw_score[:,raw_classes.index('1.0')] if '1.0' in raw_classes else np.nan
                interpretation[f'risk_{hours}h_status']=package.get('report',{}).get('probability_status','unknown')
                usable=context['eligible']&features.features_valid.fillna(False)
                if package.get('risk_policy',self.config['risk_policy'])['samples']=='episodes':usable&=features.episode_active.eq(1)
                interpretation.loc[~usable,[f'risk_{hours}h',f'risk_{hours}h_score']]=np.nan
                if package.get('available_at'):
                    interpretation.loc[interpretation.index < pd.Timestamp(package['available_at']),f'risk_{hours}h']=np.nan
                    interpretation.loc[interpretation.index < pd.Timestamp(package['available_at']),f'risk_{hours}h_score']=np.nan
        interpretation.loc[~context['eligible'],'label']='unknown'
        unusable=~context['eligible']|~features.features_valid.fillna(False)
        interpretation.loc[unusable,'confidence']=np.nan
        interpretation.loc[unusable,'label']='unknown'
        return SimpleNamespace(hourly=pd.DataFrame(hourly, index=raw.index), mode=context['regime']['mode'],
                               sensor_errors=matrices, contrib_share=shares, expected=expected,
                               alarms=pd.DataFrame(episodes, columns=['вид','начало','конец','часов','пик','главные датчики','первое превышение']),
                               missing=missing, interpretation=interpretation, quality=context['quality'])
