"""Publish completed exports and maintain a list usable before the run finishes."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import uuid
import pandas as pd


def _file_operation(action):
    """Windows scanners/readers can briefly hold a newly written directory."""
    for attempt in range(8):
        try:
            return action()
        except PermissionError:
            if attempt==7:
                raise
            time.sleep(min(.05*2**attempt,1.))


def export_ready(folder, token):
    folder=Path(folder)
    try:
        if json.loads((folder/'config.json').read_text(encoding='utf-8')).get('checkpoint_fingerprint')!=token:
            return False
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        return bool(manifest) and all((folder/name).is_file() and
            hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest for name,digest in manifest.items())
    except (OSError,ValueError):
        return False


def publish_export(source, destination):
    """Keep the previous complete folder until the replacement is fully copied."""
    source,destination=Path(source).resolve(),Path(destination).absolute()
    parent=destination.parent.resolve()
    if destination.is_symlink() or destination.resolve().parent!=parent:
        raise ValueError('Export destination must be a regular child directory')
    temporary=parent/('.'+destination.name+'.publishing-'+uuid.uuid4().hex)
    backup=parent/('.'+destination.name+'.previous-'+uuid.uuid4().hex)
    def remove_owned(path):
        if path.parent.resolve()!=parent or path.is_symlink():
            raise ValueError('Refusing to remove a path outside the export parent')
        if path.exists():
            _file_operation(lambda:shutil.rmtree(path) if path.exists() else None)
    try:
        shutil.copytree(source,temporary,ignore=shutil.ignore_patterns('__pycache__'))
        if destination.exists():
            _file_operation(lambda:os.replace(destination,backup))
        _file_operation(lambda:os.replace(temporary,destination))
    except BaseException:
        # This also covers interruption immediately after moving the old folder,
        # before the promotion call has started.
        if backup.exists() and not destination.exists():
            _file_operation(lambda:os.replace(backup,destination))
        raise
    else:
        remove_owned(backup)
    finally:
        remove_owned(temporary)


def record_export(root, name, unit, spec, state, month, current, history):
    path=Path(root)/'models.csv'
    frame=pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not frame.empty:
        frame=frame.loc[~(frame.experiment.eq(name)&frame.unit.eq(unit))]
    row=dict(experiment=name,unit=unit,method=spec['method'],scenario=spec['scenario'],
             variant=spec['variant'],evaluation_month=str(month),model_cutoff=str(state['cutoff']),
             model_dir=str(Path(current).resolve()),history_dir=str(Path(history).resolve()),status='ready')
    frame=pd.concat([frame,pd.DataFrame([row])],ignore_index=True)
    temporary=path.with_suffix('.csv.tmp');frame.to_csv(temporary,index=False,encoding='utf-8-sig')
    os.replace(temporary,path)
