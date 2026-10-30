"""Regression checks for the failures reproduced in the user's executed 02."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'checks')]
from test_target_training import sample
from test_runtime_performance import fixture
from monitoring_v3.models import MultiTargetModel,RelatedModel,snapshot_model
from monitoring_v3.config import load
from monitoring_v3.pipeline import run
from monitoring_v3.data import load_sources
import detector_notebook as DR


class DetectorRecoveryTests(unittest.TestCase):
    def test_multitarget_pipeline_exports_and_scores_a_sensor_that_appeared_later(self):
        import joblib
        from monitoring_v3.runtime import Model
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg.update(models=['CatBoostMulti'],minimum_coverage=.5,validate_candidate=False)
            path=Path(tmp)/'source/К-1/hourly.parquet'
            raw=pd.read_parquet(path);raw.loc[:1599,'pressure']=np.nan;raw.to_parquet(path,index=False)
            result=run(cfg,train_classifiers=False)
            self.assertTrue(result['status'].status.eq('complete').all(),result['status'].to_string())
            directory=next((Path(cfg['output'])/'experiments').glob('*'))/'К-1/model'
            loaded=Model(directory)
            self.assertIn('pressure',loaded.state['model'].late_models)
            self.assertTrue((directory/'target_availability.csv').is_file())
            context,_=load_sources(cfg)
            f=context['К-1']['features'].iloc[-10:]
            actual=context['К-1']['cleaned'].loc[f.index,loaded.state['model'].targets]
            self.assertTrue(np.isfinite(loaded.state['model'].expected(f,actual)[0]['pressure']).all())

    def test_multitarget_absent_output_can_start_later_without_refitting_shared_model(self):
        x,y,cfg=sample();cfg['cb_update_trees']=2
        y['late']=np.nan
        model=MultiTargetModel('CatBoostMulti',cfg).fit_initial(x,y)
        self.assertEqual(model.shared_targets,['temperature','pressure'])
        before=model.expected(x,y)[0]
        self.assertTrue(before['late'].isna().all())
        candidate=snapshot_model(model);old_trees=model.shared.tree_count_
        ix=pd.date_range(x.index[-1]+pd.Timedelta('1h'),periods=30,freq='h')
        new_x=x.iloc[:30].set_axis(ix)
        new_y=y.iloc[:30].set_axis(ix);new_y['late']=new_x.load*.7
        new_y['pressure']=np.nan  # no labels for an existing coordinate this month
        candidate.update(new_x,new_y)
        self.assertEqual(model.shared.tree_count_,old_trees)
        self.assertEqual(candidate.shared.tree_count_,old_trees+2)
        self.assertIn('late',candidate.late_models)
        self.assertTrue(np.isfinite(candidate.expected(new_x,new_y)[0]['late']).all())
        pd.testing.assert_frame_equal(before,model.expected(x,y)[0])
        self.assertEqual(candidate.target_availability()[-1]['status'],'individual_late')

    def test_related_auto_selects_training_only_and_manual_lists_override(self):
        x,y,cfg=sample();f=pd.concat([x,y.add_prefix('sensor::')],axis=1)
        cfg['related_policy']['min_abs_correlation']=.5
        model=RelatedModel(cfg,{'pressure':[]}).fit_initial(f,y)
        self.assertEqual(model.related['temperature'],['pressure'])
        self.assertEqual(model.related['pressure'],[])
        self.assertNotIn('sensor::temperature',model._inputs(f,'temperature'))
        frozen=dict(model.related)
        model.update(f.iloc[:20].set_axis(pd.date_range('2025-01-01',periods=20,freq='h')),
                     y.iloc[:20].set_axis(pd.date_range('2025-01-01',periods=20,freq='h')))
        self.assertEqual(model.related,frozen)

    def test_failure_is_recorded_and_other_experiments_continue(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp)
            cfg=load(overrides={**cfg,'models':['CatBoostMulti','CatBoost'],
                               'feature_sets':['conditions','related'],'related_inputs':{}})
            with patch.object(MultiTargetModel,'fit_initial',side_effect=ValueError('deliberate failed model')):
                result=run(cfg,train_classifiers=False)
            statuses=pd.read_csv(Path(cfg['output'])/'status.csv')
            self.assertEqual(len(statuses),3)
            self.assertEqual((statuses.status=='failed').sum(),1)
            self.assertEqual((statuses.status=='complete').sum(),2)
            self.assertTrue(statuses.loc[statuses.status.eq('failed'),'reason'].str.contains('deliberate').all())
            related=next((Path(cfg['output'])/'experiments').glob('*__related__0'))
            self.assertTrue((related/'related_inputs.csv').is_file())
            self.assertTrue((related/'К-1/model/related_inputs.json').is_file())
            self.assertTrue((related/'status.json').is_file())
            self.assertEqual((result['status'].status=='failed').sum(),1)

    def test_keyboard_interrupt_leaves_status_and_traceable_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['models']=['CatBoostMulti']
            with patch.object(MultiTargetModel,'fit_initial',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):run(cfg,train_classifiers=False)
            status=pd.read_csv(Path(cfg['output'])/'status.csv')
            self.assertTrue(status.status.eq('interrupted').all())

    def test_report_does_not_claim_nonexistent_export_or_legacy_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(tmp);cfg['feature_sets']=['related']
            from monitoring_v3.pipeline import experiments
            spec=next(experiments(cfg));name='__'.join([spec['group'],spec['scenario'],'+'.join(spec['sources']),spec['method'],spec['variant'],'0'])
            folder=Path(cfg['output'])/'experiments'/name;folder.mkdir(parents=True)
            (folder/'experiment.json').write_text(json.dumps(spec),encoding='utf-8')
            contexts,issues=load_sources(cfg);self.assertFalse(issues)
            html=[]
            with patch.object(DR,'show_html',side_effect=html.append):
                result=DR.render_detector_results(cfg,contexts,{'views':['mean::instant']})
            output=''.join(html)
            self.assertTrue(result.empty)
            self.assertIn('Готовый экспорт модели для дашборда не найден',output)
            self.assertIn('Пустой related_inputs',output)
            self.assertNotIn('Старый расчёт не сохранял',output)


if __name__=='__main__':unittest.main()
