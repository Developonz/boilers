"""Do not deepcopy fitted native CatBoost objects before monthly continuation."""
from contextlib import redirect_stdout
from pathlib import Path
import io,json,sys,tempfile,unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoost

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from monitoring_v3.models import NormalModel,RelatedModel,AnalogCorrection,MultiTargetModel,snapshot_model
from monitoring_v3.pipeline import run
from test_target_training import sample
from test_runtime_performance import fixture


class CatBoostSnapshotTests(unittest.TestCase):
    def test_snapshot_isolates_mutable_state_without_copying_native_estimators(self):
        x,y,cfg=sample();model=NormalModel('CatBoost',cfg).fit_initial(x.iloc[:110],y.iloc[:110])
        before=model.expected(x.iloc[130:],y.iloc[130:]);counts=model.counts.copy()
        original=model.predictors['temperature'][1];trees=model.tree_counts().copy()
        with patch.object(CatBoost,'__deepcopy__',side_effect=AssertionError('Native deepcopy forbidden')):
            frozen=snapshot_model(model)
            self.assertIs(frozen.predictors['temperature'][1],original)
            updated,audit=model.candidate(x.iloc[110:130],y.iloc[110:130],x.iloc[130:],y.iloc[130:])
            # Exercise the actual continuation even if the acceptance gate rejects it.
            candidate=snapshot_model(model).update(x.iloc[110:130],y.iloc[110:130])
        self.assertIsNot(candidate.predictors['temperature'][1],original)
        self.assertEqual(model.tree_counts(),trees);np.testing.assert_array_equal(model.counts,counts)
        for a,b in zip(frozen.expected(x.iloc[130:],y.iloc[130:]),before):pd.testing.assert_frame_equal(a,b,check_exact=True)
        for a,b in zip(model.expected(x.iloc[130:],y.iloc[130:]),before):pd.testing.assert_frame_equal(a,b,check_exact=True)
        frozen.memory_f.iloc[0,0]=-999
        self.assertNotEqual(model.memory_f.iloc[0,0],-999)
        self.assertIsNot(frozen.encoder,model.encoder)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snapshot.joblib';joblib.dump(candidate,path)
            loaded=joblib.load(path)
            for a,b in zip(loaded.expected(x.iloc[130:],y.iloc[130:]),candidate.expected(x.iloc[130:],y.iloc[130:])):
                pd.testing.assert_frame_equal(a,b,check_exact=True)

    def test_nested_related_correction_and_shared_models_preserve_only_native_objects(self):
        x,y,cfg=sample();normal=NormalModel('CatBoost',cfg).fit_initial(x,y)
        related=RelatedModel(cfg,{});related.members={'temperature':normal}
        correction=AnalogCorrection(cfg);correction.base=related;correction.correction=normal
        multi=MultiTargetModel('CatBoostMulti',cfg);multi.shared=normal.predictors['temperature'][1]
        for model in (related,correction,multi):
            with self.subTest(kind=type(model).__name__),patch.object(CatBoost,'__deepcopy__',side_effect=AssertionError('Native deepcopy forbidden')):
                snapshot=snapshot_model(model)
            self.assertIsNot(snapshot,model);self.assertIsNot(snapshot.cfg,model.cfg)
            if model is multi:self.assertIs(snapshot.shared,multi.shared)
            elif model is related:self.assertIs(snapshot.members['temperature'].predictors['temperature'][1],multi.shared)
            else:self.assertIs(snapshot.correction.predictors['temperature'][1],multi.shared)

    def test_monthly_pipeline_never_deepcopies_catboost_and_keeps_frozen_reference(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['validate_candidate']=False
            with patch.object(CatBoost,'__deepcopy__',side_effect=AssertionError('Native deepcopy forbidden')):
                result=run(cfg)
            self.assertTrue(result['status'].status.eq('complete').all())
            paths=sorted(Path(cfg['output']).glob('experiments/*/checkpoints/*.joblib'))
            first,last=(joblib.load(path)['state'] for path in (paths[0],paths[-1]))
            for col in first['model'].predictors:
                self.assertIs(first['model'].predictors[col][1],first['reference'].predictors[col][1])
            self.assertEqual(first['reference'].tree_counts(),last['reference'].tree_counts())
            self.assertTrue(all(n==cfg['cb_initial_trees']+cfg['cb_update_trees'] for n in last['model'].tree_counts().values()))
            records=[json.loads(line) for line in (Path(cfg['output'])/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r.get('status')=='prediction_complete' and 'ошибки калибровки' in r.get('phase','') for r in records))
            self.assertTrue(any('ошибки калибровки · обновляемая модель' in r.get('phase','') for r in records))
            self.assertTrue(any('ошибки калибровки · исходная эталонная модель' in r.get('phase','') for r in records))


if __name__=='__main__':unittest.main()
