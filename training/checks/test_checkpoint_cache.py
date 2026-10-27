"""Upgrade known versions in place; never silently reuse changed training data."""
from contextlib import redirect_stdout
from pathlib import Path
import io,json,subprocess,sys,tempfile,unittest,zipfile
from unittest.mock import patch
import joblib
import pandas as pd
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from monitoring_v3.checkpoint_cache import signatures
from monitoring_v3.data import load_sources
from monitoring_v3.models import NormalModel
from monitoring_v3.pipeline import run,experiments,fingerprint
from monitoring_v3.target_training import initial_targets
from test_runtime_performance import fixture
from test_target_training import sample


class CheckpointCacheTests(unittest.TestCase):
    def test_legacy_partial_header_and_payload_reuse_actual_training_signature(self):
        x,y,cfg=sample()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            with initial_targets(tmp,'old-implementation'):NormalModel('CatBoost',cfg).fit_initial(x,y)
            for path in Path(tmp).glob('*/*.joblib'):
                data=joblib.load(path);data.pop('contract');joblib.dump(data,path)
            with initial_targets(tmp,'new-implementation'),patch.object(CatBoostRegressor,'fit',side_effect=AssertionError('Unnecessary initial fit')):
                NormalModel('CatBoost',cfg).fit_initial(x,y)
            changed=dict(cfg,cb_depth=cfg['cb_depth']+1);calls=[];original=CatBoostRegressor.fit
            def count(model,*args,**kwargs):calls.append(1);return original(model,*args,**kwargs)
            with initial_targets(tmp,'newer-implementation'),patch.object(CatBoostRegressor,'fit',new=count):
                NormalModel('CatBoost',changed).fit_initial(x,y)
            self.assertEqual(len(calls),2)
            self.assertEqual(len(list(Path(tmp).glob('*/*.joblib'))),4)

    def test_state_signature_ignores_ui_code_but_not_data_parameters_or_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(tmp);contexts,_=load_sources(cfg);spec=next(experiments(cfg));cutoff=pd.Timestamp(cfg['first_test'])
            token,state=signatures(contexts,cutoff,cfg,spec)
            self.assertEqual(token,fingerprint(contexts,cutoff,cfg,spec))
            original=Path.read_bytes
            def ui_change(path):return original(path)+(b'\n# UI-only change' if path.name=='progress.py' else b'')
            with patch.object(Path,'read_bytes',new=ui_change):other_token,other_state=signatures(contexts,cutoff,cfg,spec)
            self.assertNotEqual(token,other_token);self.assertEqual(state,other_state)
            self.assertNotEqual(state,signatures(contexts,cutoff,dict(cfg,cb_depth=cfg['cb_depth']+1),spec)[1])
            Path(cfg['sensor_rules']).write_text('{"defaults":{}}',encoding='utf-8')
            self.assertNotEqual(state,signatures(contexts,cutoff,cfg,spec)[1])
            Path(cfg['sensor_rules']).write_text('{}',encoding='utf-8')
            contexts['К-1']['frame'].loc[contexts['К-1']['frame'].index[0],'load']+=1.
            self.assertNotEqual(state,signatures(contexts,cutoff,cfg,spec)[1])

    def test_finished_checkpoint_survives_implementation_change_without_refitting(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);run(cfg)
            paths=list(Path(cfg['output']).glob('experiments/*/checkpoints/*.joblib'))
            for path in paths:
                data=joblib.load(path);data['fingerprint']='previous-ui-build';joblib.dump(data,path)
            with patch.object(NormalModel,'fit_initial',side_effect=AssertionError('Initial fit repeated')),patch.object(
                    NormalModel,'candidate',side_effect=AssertionError('Monthly fit repeated')):
                result=run(cfg)
            self.assertTrue(result['status'].status.eq('complete').all())
            self.assertTrue(all(joblib.load(p)['fingerprint']!='previous-ui-build' for p in paths))

    def test_previous_contract_rebuilds_in_same_output_and_preserves_original_models(self):
        archive=ROOT/'runs/prediction_fix/training_update_validation.zip'
        if not archive.exists():self.skipTest('Historical release archive unavailable')
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);old=Path(tmp)/'old';old.mkdir()
            with zipfile.ZipFile(archive) as package:package.extractall(old)
            legacy={k:v for k,v in cfg.items() if k not in ('history_policy','regime_policy','risk_policy','critical_sensors','event_overrides')}
            config=Path(tmp)/'old_cfg.json';config.write_text(json.dumps(legacy,ensure_ascii=False),encoding='utf-8')
            program='\n'.join(['import sys,json',f'sys.path[:0]={repr([str(old/"training/lib"),str(ROOT/"lib")])}',
                'from monitoring_v3.pipeline import run',f'run(json.load(open({str(config)!r},encoding="utf-8")))'])
            script=Path(tmp)/'legacy_run.py';script.write_text(program,encoding='utf-8')
            child=subprocess.run([sys.executable,'-X','utf8',str(script)],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8',timeout=90)
            self.assertEqual(child.returncode,0,child.stderr[-3000:])
            paths=list(Path(cfg['output']).glob('experiments/*/checkpoints/*.joblib'))
            self.assertTrue(paths);self.assertTrue(all('compatibility' not in joblib.load(p) for p in paths))
            old_bytes={p.read_bytes() for p in paths};calls=[];original=NormalModel.fit_initial
            def fit(model,*args,**kwargs):calls.append(1);return original(model,*args,**kwargs)
            with patch.object(NormalModel,'fit_initial',new=fit):
                result=run(cfg)
            self.assertTrue(result['status'].status.eq('complete').all())
            self.assertTrue(calls)
            self.assertTrue(all('compatibility' in joblib.load(p) for p in paths))
            backups=list(Path(cfg['output']).glob('experiments/*/checkpoints/previous_versions/*.joblib'))
            self.assertTrue(old_bytes<={p.read_bytes() for p in backups})

    def test_incompatible_checkpoint_recomputes_and_preserves_previous_files(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);run(cfg)
            paths=list(Path(cfg['output']).glob('experiments/*/checkpoints/*.joblib'))
            original={p:p.read_bytes() for p in paths};changed=dict(cfg,cb_depth=cfg['cb_depth']+1)
            with patch('monitoring_v3.pipeline.update_at',side_effect=RuntimeError('rebuild interrupted')):
                with self.assertRaisesRegex(RuntimeError,'rebuild interrupted'):run(changed)
            self.assertTrue(all(p.read_bytes()==data for p,data in original.items()))
            result=run(changed)
            self.assertTrue(result['status'].status.eq('complete').all())
            backups=list(Path(cfg['output']).glob('experiments/*/checkpoints/previous_versions/*.joblib'))
            self.assertEqual(len(backups),len(paths))
            self.assertEqual({p.read_bytes() for p in backups},set(original.values()))


if __name__=='__main__':unittest.main()
