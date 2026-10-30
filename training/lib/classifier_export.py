"""Export trained 05 classifiers with frozen preparation and native weights."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import uuid
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    return value


def _json(path,value):
    Path(path).write_text(json.dumps(_plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def freeze_preparation(contexts,cfg):
    units={}
    for unit,c in contexts.items():
        units[unit]=dict(roles=c['roles'],aliases=c.get('aliases',{}),declared_units=c.get('declared_units',{}),
            threshold=c['threshold'],threshold_source=c['threshold_source'],load=c['load'],
            regime_spec=c['regime_spec'],reference_end=str(c['preparation_reference_end']),history_start=str(c['frame'].index[0]))
    first=next(iter(contexts.values()))
    if not all('preparation_rules' in c and 'preparation_config' in c for c in contexts.values()):
        raise ValueError('Повторите подготовку данных: отсутствуют зафиксированные правила подготовки')
    rules=first['preparation_rules'];preparation=first['preparation_config']
    if any(c['preparation_rules']!=rules or c['preparation_config']!=preparation for c in contexts.values()):
        raise ValueError('В группе различаются настройки подготовки')
    return _plain(dict(units=units,sensor_rules=rules,preparation=preparation))


def export_classifier(package,contexts,cfg,folder):
    if package.get('feature_contract')!='telemetry_only_v1' or 'deployment_preparation' not in package:
        raise ValueError('Модель из прежнего контракта. Выполните обучение в обновлённом 05 перед экспортом.')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    frozen=package['deployment_preparation'];units=frozen['units'];rules=frozen['sensor_rules']
    if any(pd.Timestamp(c['reference_end'])>pd.Timestamp(package['available_at']) for c in units.values()):
        raise ValueError('Подготовка использовала сведения позже даты доступности модели')
    effective=[{**rules.get('defaults',{}),**rules.get('sensors',{}).get(c,{})}
               for spec in units.values() for c,r in spec['roles'].items() if r in ('target','load','condition')]
    if any(rule.get('mask_spikes',False) and rule.get('kind','unreviewed')=='continuous' for rule in effective):
        raise ValueError('Для дашборда скачки должны учитываться: отключите mask_spikes и повторите 05.')
    runtime=folder/'runtime';runtime.mkdir(exist_ok=True)
    (runtime/'__init__.py').write_text('from .inference import Model\n',encoding='utf-8')
    for name in ('preparation.py','regime_evidence.py','input_grid.py'):shutil.copy2(ROOT/'monitoring_v3'/name,runtime/name)
    shutil.copy2(ROOT/'classifier_runtime.py',runtime/'inference.py')
    source=(ROOT/'direct_risk.py').read_text(encoding='utf-8');tree=ast.parse(source)
    wanted={'direct_options','_direct_category','_direct_slope','direct_features','direct_native'}
    pieces=['import numpy as np\nimport pandas as pd\n']
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in wanted or isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DIRECT_DEFAULTS' for t in node.targets):
            pieces.append(ast.get_source_segment(source,node))
    (runtime/'feature_functions.py').write_text('\n\n'.join(pieces)+'\n',encoding='utf-8')
    package['model'].save_model(str(folder/'classifier.cbm'))
    package['schema'].to_csv(folder/'feature_schema.csv',index=False,encoding='utf-8-sig')
    if 'metrics' in package:package['metrics'].to_csv(folder/'metrics.csv',index=False,encoding='utf-8-sig')
    config=dict(version='direct-risk-1',feature_contract=package['feature_contract'],columns=package['columns'],
        categorical=package['categorical'],options=package['options'],units=units,sensor_rules=rules,
        preparation=frozen['preparation'],
        available_at=package['available_at'],threshold=package.get('threshold'),threshold_status=package.get('threshold_status'),
        thresholds=package.get('thresholds',{}),score_status=package['score_status'],experiment=package.get('experiment',{}),
        model_fingerprint=package.get('fingerprint'),validation_selection=package.get('validation_selection',{}))
    _json(folder/'config.json',config)
    versions={p:importlib.metadata.version(p) for p in ('catboost','numpy','pandas')}
    _json(folder/'environment.json',versions)
    (folder/'requirements.txt').write_text('\n'.join(k+'=='+v for k,v in versions.items())+'\n',encoding='utf-8')
    manifest={str(p.relative_to(folder)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    _json(folder/'manifest.json',manifest)
    return folder


def export_direct_suite(root,contexts=None,cfg=None,output=None,method='unit',variant='AUTO',horizons=(168,)):
    import joblib
    if method not in ('unit','group'):raise ValueError('EXPORT_METHOD: unit или group')
    if variant not in ('AUTO','numeric','numeric_states','full'):raise ValueError('Неверный EXPORT_VARIANT')
    root=Path(root);output=Path(output or root/'dashboard_export');rows=[]
    name='validation_choices.csv' if variant=='AUTO' else 'experiment_index.csv'
    index=pd.read_csv(root/name)
    if 'status' in index:index=index.loc[index.status.eq('complete')]
    index=index.loc[index.method.eq(method)&index.horizon_hours.isin(horizons)]
    if variant!='AUTO':index=index.loc[index.variant.eq(variant)]
    if index.empty:raise ValueError('Нет выбранных обученных моделей для экспорта: '+name)
    for item in index.sort_values('validation_end').itertuples():
        package=joblib.load(Path(item.folder)/'classifier.joblib')
        family=str(item.name)
        if any(c in family for c in '<>:"/\\|?*') or '..' in family:raise ValueError('Unsafe model family')
        dest=output/method/family/(str(int(item.horizon_hours))+'h');dest.mkdir(parents=True,exist_ok=True)
        identity=str(package.get('fingerprint',''))+str(item.variant)
        version=pd.Timestamp(package['available_at']).strftime('%Y%m%dT%H%M%S')+'_'+hashlib.sha256(identity.encode()).hexdigest()[:10]
        final=dest/version
        temporary=dest/(version+'.'+uuid.uuid4().hex+'.tmp')
        try:
            selected=package.get('deployment_preparation',{}).get('units',{})
            if not selected:raise ValueError('Нужно обучение в обновлённом 05: нет сохранённой подготовки')
            export_classifier(package,selected,cfg,temporary)
            if final.exists():
                if (final/'manifest.json').read_bytes()!=(temporary/'manifest.json').read_bytes():
                    version+='_'+hashlib.sha256((temporary/'manifest.json').read_bytes()).hexdigest()[:8];final=dest/version
            if not final.exists():os.replace(temporary,final)
            old=json.loads((dest/'index.json').read_text(encoding='utf-8')) if (dest/'index.json').exists() else []
            records=[v for v in old if v['folder']!=version]+[dict(folder=version,available_at=package['available_at'],units=list(selected),variant=item.variant)]
            _json(dest/'index.json.tmp',sorted(records,key=lambda v:v['available_at']))
            os.replace(dest/'index.json.tmp',dest/'index.json')
            rows.append(dict(method=method,family=family,horizon_hours=int(item.horizon_hours),version=version,
                available_at=package['available_at'],model_dir=str(dest.resolve()),units=', '.join(selected),status='exported'))
        finally:
            if temporary.exists():
                if temporary.resolve().parent!=dest.resolve():raise ValueError('Export cleanup outside target directory')
                shutil.rmtree(temporary)
    return pd.DataFrame(rows)
