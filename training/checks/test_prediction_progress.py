"""Shared prediction input preserves results and exposes monthly validation stalls."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
import io
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,Pool

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from monitoring_v3.models import NormalModel
from monitoring_v3.progress import Progress
from monitoring_v3.threads import numerical_threads
from test_target_training import sample


def direct_expected(model,features):
    """Prior prediction contract: each estimator receives the DataFrame separately."""
    native=model.encoder.native(features);shape=(len(features),len(model.targets))
    med=np.full(shape,np.nan);spread=np.full(shape,np.nan)
    for j,col in enumerate(model.targets):
        item=model.predictors.get(col)
        if item is None:continue
        kind,estimator=item
        if kind in ('constant','reference'):
            med[:,j]=estimator;spread[:,j]=model.scale[j];continue
        raw=(estimator.predict(native,thread_count=model.cfg['threads']) if kind=='catboost' else
             np.column_stack([m.predict(model.encoder.numeric(features)) for m in estimator]))
        pred=np.sort(raw,axis=1)
        med[:,j]=pred[:,1];spread[:,j]=np.maximum((pred[:,2]-pred[:,0])/1.349,model.floor[j])
    valid=model.counts>=model.cfg['min_sensor_rows'];med[:,~valid]=np.nan;spread[:,~valid]=np.nan
    return tuple(pd.DataFrame(a,index=features.index,columns=model.targets) for a in (med,spread))


class PredictionProgressTests(unittest.TestCase):
    def test_disabled_validation_updates_without_prediction_and_preserves_previous(self):
        x,y,cfg=sample();cfg['validate_candidate']=False
        model=NormalModel('CatBoost',cfg).fit_initial(x.iloc[:110],y.iloc[:110])
        before=model.tree_counts().copy()
        with patch.object(NormalModel,'expected',side_effect=AssertionError('Validation must not predict')):
            updated,audit=model.candidate(x.iloc[110:130],y.iloc[110:130],None,None)
        self.assertTrue(audit['accepted']);self.assertEqual(audit['validation_status'],'disabled')
        self.assertEqual(model.tree_counts(),before)
        self.assertEqual(updated.updates,model.updates+1)
        self.assertEqual(updated.tree_counts(),{key:n+cfg['cb_update_trees'] for key,n in before.items()})
        self.assertNotIn('old_error',audit)

    def test_validation_switch_defaults_on_and_rejects_strings(self):
        from monitoring_v3.config import load
        self.assertTrue(load()['validate_candidate'])
        self.assertFalse(load(overrides={'validate_candidate':False})['validate_candidate'])
        with self.assertRaisesRegex(ValueError,'validate_candidate'):
            load(overrides={'validate_candidate':'False'})

    def test_pipeline_without_validation_keeps_calibration_and_exports(self):
        from test_runtime_performance import fixture
        from monitoring_v3.pipeline import run
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['validate_candidate']=False
            result=run(cfg)
            records=[json.loads(line) for line in (Path(cfg['output'])/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(result['status'].status.eq('complete').all())
            self.assertTrue(any(r.get('status')=='candidate_validation_disabled' for r in records))
            self.assertFalse(any('проверочные прогнозы' in r.get('phase','') for r in records))
            self.assertTrue(any('подбор порогов' in r.get('phase','') for r in records))
            self.assertTrue(list(Path(cfg['output']).glob('experiments/*/*/model/state.joblib')))

    def test_shared_pool_matches_dataframe_with_missing_and_unseen_values(self):
        x,y,cfg=sample();y['constant']=3.;y['rare']=np.nan;y.loc[y.index[:5],'rare']=1.
        x.loc[x.index[::7],'load']=np.nan
        model=NormalModel('CatBoost',cfg).fit_initial(x.iloc[:110],y.iloc[:110])
        later=x.iloc[110:].copy();later.loc[later.index[::3],'state']='unseen'
        later.loc[later.index[1],'state']=None
        expected=direct_expected(model,later)
        with patch('catboost.Pool',wraps=Pool) as constructor:
            got=model.expected(later,y.iloc[110:])
        self.assertEqual(constructor.call_count,1)
        for a,b in zip(got,expected):pd.testing.assert_frame_equal(a,b,check_exact=True)
        with patch('catboost.Pool',wraps=Pool) as constructor:
            diagnostics=model.interval_diagnostics(later,y.iloc[110:])
        self.assertEqual(constructor.call_count,1);self.assertEqual(len(diagnostics),2)
        with patch('catboost.Pool',wraps=Pool) as constructor:
            empty=model.expected(later.iloc[:0],y.iloc[:0])
        self.assertEqual(constructor.call_count,0);self.assertEqual(empty[0].shape,(0,4))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.joblib';joblib.dump(model,path);loaded=joblib.load(path)
            for a,b in zip(loaded.expected(later,y.iloc[110:]),expected):
                pd.testing.assert_frame_equal(a,b,check_exact=True)

    def test_candidate_shares_one_pool_and_keeps_acceptance_contract(self):
        x,y,cfg=sample();old=NormalModel('CatBoost',cfg).fit_initial(x.iloc[:110],y.iloc[:110])
        updated=deepcopy(old).update(x.iloc[110:130],y.iloc[110:130])
        vf,vy=x.iloc[130:],y.iloc[130:]
        om,oscale=direct_expected(old,vf);nm,nscale=direct_expected(updated,vf)
        values=dict(old_error=vy.sub(om).div(oscale).abs().median().median(),
                    new_error=vy.sub(nm).div(nscale).abs().median().median(),
                    old_physical_error=vy.sub(om).abs().div(old.scale).median().median(),
                    new_physical_error=vy.sub(nm).abs().div(old.scale).median().median())
        accepted=bool(values['new_physical_error']<=max(values['old_physical_error'],1e-6)*(1+cfg['validation_tolerance']))
        original=CatBoostRegressor.predict;inputs=[]
        def observe(estimator,data,*args,**kwargs):
            inputs.append(data);return original(estimator,data,*args,**kwargs)
        with patch('catboost.Pool',wraps=Pool) as constructor,patch.object(CatBoostRegressor,'predict',new=observe):
            result,audit=old.candidate(x.iloc[110:130],y.iloc[110:130],vf,vy)
        self.assertEqual(constructor.call_count,1);self.assertEqual(len(inputs),4)
        self.assertTrue(all(data is inputs[0] for data in inputs));self.assertIsInstance(inputs[0],Pool)
        self.assertEqual(audit['accepted'],accepted)
        for key,value in values.items():self.assertAlmostEqual(audit[key],value,places=12)
        expected=updated if accepted else old
        self.assertEqual(result.tree_counts(),expected.tree_counts())
        for a,b in zip(result.expected(vf,vy),direct_expected(expected,vf)):
            pd.testing.assert_frame_equal(a,b,check_exact=True)

    def test_monthly_substages_report_sensor_and_suppress_parent_heartbeat(self):
        x,y,cfg=sample();model=NormalModel('CatBoost',cfg).fit_initial(x.iloc[:110],y.iloc[:110])
        original=CatBoostRegressor.predict;delayed=[]
        def slow_first(estimator,*args,**kwargs):
            if not delayed:delayed.append(True);time.sleep(.06)
            return original(estimator,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            progress=Progress(tmp,interval=.005)
            with progress.stage('monthly'),patch.object(CatBoostRegressor,'predict',new=slow_first):
                model.candidate(x.iloc[110:130],y.iloc[110:130],x.iloc[130:],y.iloc[130:])
            records=[json.loads(line) for line in (Path(tmp)/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
        starts=[r['phase'] for r in records if r['status']=='started']
        labels=['копирование предыдущей модели','дообучение кандидата','обновление памяти примеров',
                'подготовка общих проверочных входов','проверочные прогнозы старой модели',
                'проверочные прогнозы новой модели','сравнение ошибок и решение']
        positions=[next(i for i,label in enumerate(starts) if text in label) for text in labels]
        self.assertEqual(positions,sorted(positions))
        predictions=[r for r in records if r['status']=='prediction_complete']
        self.assertEqual(len(predictions),4)
        self.assertTrue(all('prediction_seconds' in r and r['phase'].startswith('monthly · проверочные прогнозы') for r in predictions))
        running=[r for r in records if r['status']=='running' and 'проверочные прогнозы старой модели' in r['phase']]
        self.assertTrue(any('датчик 1/2' in r['message'] for r in running))
        begin=next(i for i,r in enumerate(records) if r['status']=='prediction_started')
        end=next(i for i,r in enumerate(records[begin:],begin) if r['status']=='prediction_complete')
        self.assertFalse(any(r.get('phase')=='monthly' and r['status']=='running' for r in records[begin:end]))
        self.assertEqual(sum(r['status']=='candidate_decision' for r in records),1)

    def test_prediction_failure_identifies_stage_and_target(self):
        x,y,cfg=sample();model=NormalModel('CatBoost',cfg).fit_initial(x.iloc[:110],y.iloc[:110])
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            progress=Progress(tmp)
            with self.assertRaisesRegex(RuntimeError,'predict failed'),progress.stage('monthly'):
                with patch.object(CatBoostRegressor,'predict',side_effect=RuntimeError('predict failed')):
                    model.candidate(x.iloc[110:130],y.iloc[110:130],x.iloc[130:],y.iloc[130:])
            records=[json.loads(line) for line in (Path(tmp)/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
        start=next(r for r in records if r['status']=='prediction_started')
        self.assertEqual(start['target'],'temperature');self.assertIn('старой модели',start['phase'])
        self.assertTrue(any(r['status']=='failed' and 'старой модели' in r['phase'] for r in records))
        self.assertFalse(any(r['status']=='candidate_decision' for r in records))

    def test_constant_predictors_do_not_construct_pool(self):
        x,y,cfg=sample();y[:]=7.;model=NormalModel('CatBoost',cfg).fit_initial(x,y)
        with patch('catboost.Pool',wraps=Pool) as constructor:
            model.expected(x,y);model.interval_diagnostics(x,y)
        self.assertEqual(constructor.call_count,0)

    def test_hist_numeric_inputs_are_prepared_once_for_all_quantiles(self):
        x,y,cfg=sample()
        with numerical_threads(1):
            model=NormalModel('HistGradientBoosting',cfg).fit_initial(x,y)
            expected=direct_expected(model,x)
            with patch.object(model.encoder,'numeric',wraps=model.encoder.numeric) as numeric:
                got=model.expected(x,y)
            self.assertEqual(numeric.call_count,1)
            for a,b in zip(got,expected):pd.testing.assert_frame_equal(a,b,check_exact=True)


if __name__=='__main__':unittest.main()
