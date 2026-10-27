"""Target-level recovery and diagnostics on interrupted initial CatBoost fits."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from monitoring_v3.config import load
from monitoring_v3.models import NormalModel
from monitoring_v3.preparation import FrozenEncoder
from monitoring_v3.progress import Progress
from monitoring_v3.target_training import initial_targets,rescue_initial
from catboost import CatBoostRegressor


def sample():
    ix=pd.date_range('2024-01-01',periods=150,freq='h');rng=np.random.default_rng(77)
    x=pd.DataFrame({'load':rng.uniform(100,200,len(ix)),'state':np.where(np.arange(len(ix))%3,'on','off')},index=ix)
    y=pd.DataFrame({'temperature':.3*x.load+rng.normal(0,.1,len(ix)),
                    'pressure':.05*x.load+rng.normal(0,.1,len(ix))},index=ix)
    cfg=load(overrides=dict(cb_initial_trees=6,cb_depth=2,min_sensor_rows=10,memory_rows=200,threads=1))
    return x,y,cfg


class TargetTrainingTests(unittest.TestCase):
    def test_completed_target_survives_interrupt_without_retraining(self):
        x,y,cfg=sample();original=CatBoostRegressor.fit;calls=[]
        def interrupt_second(model,*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise KeyboardInterrupt('interrupted on second target')
            return original(model,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);progress=Progress(root)
            with self.assertRaises(KeyboardInterrupt),initial_targets(root/'partial','token'),progress.stage('initial'):
                with patch.object(CatBoostRegressor,'fit',new=interrupt_second):
                    NormalModel('CatBoost',cfg).fit_initial(x,y)
            self.assertEqual(len(list((root/'partial').glob('*/*.joblib'))),1)
            resumed=[]
            def count_fit(model,*args,**kwargs):
                resumed.append(1);return original(model,*args,**kwargs)
            with initial_targets(root/'partial','token'),progress.stage('resume'):
                with patch.object(CatBoostRegressor,'fit',new=count_fit):
                    model=NormalModel('CatBoost',cfg).fit_initial(x,y)
            self.assertEqual(len(resumed),1)
            expected=NormalModel('CatBoost',cfg).fit_initial(x,y)
            pd.testing.assert_frame_equal(model.expected(x,y)[0],expected.expected(x,y)[0])
            records=[json.loads(line) for line in (root/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r.get('restored') and r.get('target')=='temperature' for r in records))
            info=next(r for r in records if r['status']=='fit_inputs')
            self.assertEqual(info['features'],2);self.assertEqual(info['largest_categories'],[['state',2]])

    def test_input_or_schema_changes_do_not_reuse_target_cache(self):
        x,y,cfg=sample();original=CatBoostRegressor.fit
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            with initial_targets(tmp,'token'):NormalModel('CatBoost',cfg).fit_initial(x,y)
            changed=y+10;calls=[]
            def count(model,*args,**kwargs):calls.append(1);return original(model,*args,**kwargs)
            with initial_targets(tmp,'token'),patch.object(CatBoostRegressor,'fit',new=count):
                NormalModel('CatBoost',cfg).fit_initial(x,changed)
            self.assertEqual(len(calls),2)
            with initial_targets(tmp,'different-token'),patch.object(CatBoostRegressor,'fit',side_effect=AssertionError('Compatible targets refitted')):
                NormalModel('CatBoost',cfg).fit_initial(x,y)

    def test_native_conversion_matches_previous_contract(self):
        x,_,_=sample();x['extra']=np.nan;x.loc[x.index[0],'load']=np.inf
        encoder=FrozenEncoder.fit(x.replace(np.inf,np.nan));later=x.drop(columns='extra')
        expected=later.reindex(columns=encoder.columns).copy()
        for col in encoder.columns:
            expected[col]=(expected[col].fillna('__MISSING__').astype(str) if col in encoder.categories else
                           pd.to_numeric(expected[col],errors='coerce').replace([np.inf,-np.inf],np.nan))
        pd.testing.assert_frame_equal(encoder.native(later),expected)

    def test_variation_check_short_circuits_and_reuses_row_mask(self):
        x,y,cfg=sample();cfg['memory_rows']=200
        x=pd.concat([x,pd.DataFrame({f'extra{i}':1. for i in range(100)},index=x.index)],axis=1)
        original=pd.Series.nunique;scanned=[]
        def count(series,*args,**kwargs):
            scanned.append(series.name);return original(series,*args,**kwargs)
        with patch.object(pd.Series,'nunique',new=count):NormalModel('CatBoost',cfg).fit_initial(x,y)
        self.assertEqual(scanned,['load'])

    def test_old_interrupted_pipeline_can_rescue_to_new_output(self):
        from test_runtime_performance import fixture
        from monitoring_v3.pipeline import run
        original=CatBoostRegressor.fit;calls=[]
        def interrupt(model,*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise KeyboardInterrupt('old initial fit interrupted')
            return original(model,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);new=Path(tmp)/'rescued'
            try:
                with patch.object(CatBoostRegressor,'fit',new=interrupt):run(cfg)
            except KeyboardInterrupt as error:
                result=rescue_initial(new,exception=error)
            else:self.fail('Expected an interrupted initial fit')
            self.assertEqual(result['saved_targets'],1)
            self.assertTrue(Path(result['path']).exists())
            cfg['output']=str(new);fresh=[]
            def count_initial(model,*args,**kwargs):
                if kwargs.get('init_model') is None:fresh.append(1)
                return original(model,*args,**kwargs)
            with patch.object(CatBoostRegressor,'fit',new=count_initial):complete=run(cfg)
            self.assertTrue(complete['status'].status.eq('complete').all())
            self.assertEqual(len(fresh),1)


if __name__=='__main__':unittest.main()
