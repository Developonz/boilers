"""Focused contracts for notebook 02 and detector-only saved reports."""
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
from test_runtime_performance import fixture
from monitoring_v3.config import load
from monitoring_v3.data import load_sources
from monitoring_v3.pipeline import run
from monitoring_v3.alarm_scores import evaluate
import detector_notebook as DR


class DetectorNotebookTests(unittest.TestCase):
    def test_classifier_inspection_reads_only_saved_test_scores(self):
        import matplotlib
        matplotlib.use('Agg')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);saved=root/'classifier/experiment/fold_01';saved.mkdir(parents=True)
            ix=pd.MultiIndex.from_product([pd.date_range('2025-01-01',periods=100,freq='h'),['К-1']],names=['time','unit'])
            y=np.arange(100)%3==0
            p=pd.DataFrame({'label':y.astype(int),'score':np.where(y,.8,.1),'evaluation_eligible':True,
                            'model_available_at':pd.Timestamp('2025-01-01')},index=ix)
            p.to_parquet(saved/'test_predictions.parquet')
            (saved/'settings.json').write_text(json.dumps({'thresholds':{'best_f2':{'threshold':.6,'status':'validation_f2'}}}))
            html=[]
            with patch.object(DR,'show_html',side_effect=html.append),patch('joblib.load',side_effect=AssertionError('must not load a model')):
                metrics=DR.saved_classifier_report({'root':str(root/'classifier')},root/'output')
            self.assertEqual(metrics.iloc[0].threshold,.6)
            self.assertEqual(metrics.iloc[0].hourly_roc_auc,1.)
            self.assertEqual(metrics.iloc[0].hourly_precision,1.)
            self.assertIn('data:image/png;base64,',''.join(html))
            self.assertTrue(list((root/'output').rglob('normal_vs_pre_failure.png')))

    def test_notebook_runs_without_classifiers_and_report_survives_interruption(self):
        import matplotlib
        matplotlib.use('Agg')
        from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(tmp)
            cfg=load(overrides={**cfg,'feature_sets':['conditions','related'],
                'related_inputs':{'temperature':['pressure'],'pressure':['temperature']}})
            source=Path(tmp)/'source/К-1/hourly.parquet'
            data=pd.read_parquet(source)
            # Known failure with a pre-failure drift, and both labels in test.
            data.loc[2150:2173,'load']=0
            data.loc[2150:2173,'Состояние']='Аварийный ремонт'
            data.loc[2100:2149,'pressure']+=3
            data.to_parquet(source,index=False)
            contexts,issues=load_sources(cfg);self.assertFalse(issues)
            notebook=json.loads((ROOT/'02_Обучение.ipynb').read_text(encoding='utf-8'))
            ns={}
            for i,c in enumerate(notebook['cells']):
                if c['cell_type']=='code':compile(''.join(c['source']),f'cell-{i}','exec')
            # Exercise the actual embedded presentation code, not just the module.
            exec(''.join(notebook['cells'][5]['source']).split('DR.show_html')[0],ns)
            report=ns['DR'];captured=[]
            with redirect_stdout(io.StringIO()),patch('monitoring_v3.monthly_risk.MonthlyRisk',side_effect=AssertionError('risk lifecycle called')),\
                 patch('monitoring_v3.monthly_risk.month_features',side_effect=AssertionError('risk features called')),\
                 patch('monitoring_v3.interpretation.episode_features',side_effect=AssertionError('episode features called')):
                result=run(cfg,train_classifiers=False)
            self.assertTrue(result['status'].status.eq('complete').all())
            self.assertTrue(result['interpretation'].empty)
            self.assertFalse((Path(cfg['output'])/'monthly_risk').exists())
            self.assertFalse(list(Path(cfg['output']).glob('experiments/*/*/interpretation_features.parquet')))
            items=report.saved_experiments(cfg);self.assertEqual({i['variant'] for i in items},{'conditions','related'})
            with patch.object(report,'show_html',side_effect=captured.append):
                report.preparation_report(contexts,[],cfg)
                summary=report.render_detector_results(cfg,contexts,{'views':['mean::instant'],'sensors':['pressure']})
            self.assertFalse(summary.empty)
            self.assertEqual(set(summary.variant),{'conditions','related'})
            html=''.join(captured)
            self.assertIn('data:image/png;base64,',html)
            self.assertIn('precision',html)
            self.assertIn('MAE',html)
            directory=Path(items[0]['directory'])
            predictions=pd.read_parquet(directory/'prequential_predictions.parquet')
            residuals=pd.read_parquet(directory/'prequential_errors.parquet')
            agg,_=report.regression_table(directory)
            for row in agg.itertuples():
                p=predictions[row.sensor].where(residuals[row.sensor].notna()).dropna()
                a=contexts['К-1']['cleaned'][row.sensor].reindex(p.index)
                self.assertEqual(len(p),row.rows)
                self.assertAlmostEqual(row.MAE,mean_absolute_error(a,p),places=8)
                self.assertAlmostEqual(row.RMSE,np.sqrt(mean_squared_error(a,p)),places=8)
                self.assertAlmostEqual(row.R2,r2_score(a,p),places=7)
            timeline,partial=report.load_timeline(items[0],'mean::instant',cfg['alarm_budget'],cfg,contexts['К-1'])
            y,s,p=report.hourly_examples(timeline,contexts['К-1']['events'],cfg,cfg['first_test'])
            metric,_=evaluate(timeline,contexts['К-1']['events'],cfg,start=cfg['first_test'])
            self.assertAlmostEqual(y.mean(),metric['random_precision'])
            self.assertTrue((p.index==s.index).all())
            # Delete final view in a temporary fixture only: recovery must use
            # saved predictions, never fit / predict with the newest model.
            path=directory/('mean__instant__'+str(float(cfg['alarm_budget']))+'.parquet')
            path.unlink()
            recovered,partial=report.load_timeline(items[0],'mean::instant',cfg['alarm_budget'],cfg,contexts['К-1'])
            self.assertTrue(partial)
            pd.testing.assert_series_equal(timeline.ratio,recovered.ratio,check_freq=False)
            self.assertTrue(list((directory/'notebook_report').glob('*/confusion_matrix.png')))
            # Missing months are not fabricated as healthy observations.
            self.assertTrue(recovered.ratio.isna().any())


if __name__=='__main__':unittest.main()
