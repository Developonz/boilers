"""Compatibility of learned state, separate from incidental implementation changes."""
from functools import lru_cache
from pathlib import Path
import base64
import hashlib
import importlib.metadata
import json
import shutil
import uuid
import zlib
import pandas as pd
from .config import DEFAULTS

# Bump when training/preparation/calibration semantics or state schema change.
STATE_CONTRACT=4
LIBRARIES=('numpy','pandas','scipy','scikit-learn','catboost','joblib')


def semantic_config(cfg):
    result={**DEFAULTS,**cfg}
    # Selection and destination do not change one concrete experiment's model.
    result = {k:v for k,v in result.items() if k not in
            ('output','models','feature_sets','horizons','scenarios','groups','data_root','data_paths','sensor_file','sensor_rules')}
    # Related links are frozen inside each fitted model. Preserve existing
    # conditions checkpoints when the default automatic selector is introduced.
    if result.get('related_policy')==DEFAULTS['related_policy']:
        result.pop('related_policy',None)
    return result


def _data_chunks(contexts,cutoff,cfg,spec):
    for library in LIBRARIES:
        yield (library+importlib.metadata.version(library)).encode()
    yield Path(cfg['sensor_rules']).read_bytes()
    for unit in spec['sources']:
        context=contexts[unit]
        past=context['frame'].loc[lambda d:d.index<cutoff]
        yield unit.encode()
        yield pd.util.hash_pandas_object(past,index=True).to_numpy().tobytes()
        for key in ('roles','aliases','declared_units'):
            yield json.dumps(context[key],sort_keys=True).encode()


def signatures(contexts,cutoff,cfg,spec):
    """Hash data once, producing implementation and stable state signatures."""
    implementation=hashlib.sha256(json.dumps(dict(cfg=cfg,spec=spec),sort_keys=True,ensure_ascii=False).encode())
    for source in sorted(Path(__file__).parent.glob('*.py')):implementation.update(source.read_bytes())
    implementation.update((Path(__file__).parent.parent/'kernel_models.py').read_bytes())
    state_description=dict(contract=STATE_CONTRACT,cfg=semantic_config(cfg),spec=spec)
    if spec['variant']=='related' and spec['method'] in ('AAKR','MSET'):
        state_description['kernel_related_contract']='classical_all_pinv_complete_v2'
    state=hashlib.sha256(json.dumps(state_description,sort_keys=True,ensure_ascii=False).encode())
    for chunk in _data_chunks(contexts,cutoff,cfg,spec):
        implementation.update(chunk);state.update(chunk)
    return implementation.hexdigest(),state.hexdigest()


@lru_cache(maxsize=1)
def _legacy_code():
    path=Path(__file__).with_name('cache_compatibility.json')
    if not path.exists():return ()
    registry=json.loads(path.read_text(encoding='utf-8'))
    if registry['contract']!=STATE_CONTRACT:return ()
    result=[];seen=set()
    for item in registry['versions']:
        code=zlib.decompress(base64.b64decode(item['code_zlib_base64']))
        if hashlib.sha256(code).hexdigest()!=item['code_sha256']:
            raise ValueError('Damaged cache compatibility registry')
        lf=code.replace(b'\r\n',b'\n')
        for variant in (code,lf,lf.replace(b'\n',b'\r\n')):
            key=hashlib.sha256(variant).digest()
            if key not in seen:result.append((item['id'],variant));seen.add(key)
    return tuple(result)


def legacy_compatible(saved,contexts,cutoff,cfg,spec):
    old_cfg=getattr(saved['state']['model'],'cfg',None)
    if old_cfg is None or semantic_config(old_cfg)!=semantic_config(cfg):return False
    description=json.dumps(dict(cfg=old_cfg,spec=spec),sort_keys=True,ensure_ascii=False).encode()
    candidates=[]
    for name,code in _legacy_code():
        digest=hashlib.sha256(description);digest.update(code);candidates.append(digest)
    if not candidates:return False
    for chunk in _data_chunks(contexts,cutoff,cfg,spec):
        for digest in candidates:digest.update(chunk)
    return any(digest.hexdigest()==saved.get('fingerprint') for digest in candidates)


def checkpoint_compatible(saved,token,state_token,contexts,cutoff,cfg,spec):
    if saved.get('compatibility') is not None:
        return saved['compatibility']==state_token
    return saved.get('fingerprint')==token or legacy_compatible(saved,contexts,cutoff,cfg,spec)


def backup_checkpoint(path):
    """Copy before replacing; preserve the old model on any failed recomputation."""
    path=Path(path).resolve();parent=path.parent
    folder=parent/'previous_versions'
    if folder.is_symlink() or folder.resolve().parent!=parent:
        raise ValueError('Checkpoint backup must stay inside its checkpoint directory')
    folder.mkdir(exist_ok=True)
    backup=folder/(path.stem+'-'+uuid.uuid4().hex+'.joblib')
    shutil.copy2(path,backup)
    return backup
