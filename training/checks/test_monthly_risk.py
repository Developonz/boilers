"""The risk heads advance with detector months in section 3, not a final full fit."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import shutil
import sys
import tempfile
import warnings
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'checks')]
from monitoring_v3.config import load
from monitoring_v3.monthly_risk import MonthlyRisk,saved_monthly_statuses
from monitoring_v3.progress import Progress
from monitoring_v3.pipeline import run,update_at
from test_runtime_performance import fixture


def two_boilers(tmp):
    cfg=fixture(tmp);root=Path(tmp)
    folder=root/'source'/'К-2';folder.mkdir()
    shutil.copy2(root/'source'/'К-1/hourly.parquet',folder/'hourly.parquet')
    wb=openpyxl.load_workbook(root/'roles.xlsx');wb.copy_worksheet(wb['К-1']).title='К-2';wb.save(root/'roles.xlsx')
    cfg['groups']={'test':['К-1','К-2']};cfg['models']=['analogs'];cfg['risk_horizons']=[24]
    return cfg


class MonthlyRiskTests(unittest.TestCase):
    def test_pooled_history_does_not_end_with_the_shortest_boiler(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=two_boilers(tmp);cfg['scenarios']=['pooled']
            path=Path(tmp)/'source'/'К-1/hourly.parquet'
            pd.read_parquet(path).iloc[:1800].to_parquet(path,index=False)
            result=run(cfg)
            self.assertTrue(result['status'].status.eq('complete').all())
            history=next(Path(cfg['output']).glob('experiments/*/К-2/model_history'))
            self.assertTrue((history/'2024-04').exists())
            timeline=pd.read_parquet(history.parent/'mean__instant__1.0.parquet')
            self.assertGreaterEqual(timeline.index.max(),pd.Timestamp('2024-04-01'))

    def test_model_registry_retries_transient_windows_lock(self):
        from monitoring_v3.artifacts import record_export
        import os
        original=os.replace;calls=[]
        def replace(source,target):
            calls.append(1)
            if len(calls)<3:raise PermissionError('temporary reader')
            return original(source,target)
        with tempfile.TemporaryDirectory() as tmp,patch('monitoring_v3.artifacts.os.replace',side_effect=replace),patch('monitoring_v3.artifacts.time.sleep'):
            record_export(tmp,'e','K',dict(method='CatBoost',scenario='local',variant='conditions'),
                dict(cutoff='2024-05-01'),'2024-05-01',Path(tmp)/'model',Path(tmp)/'history')
            self.assertEqual(pd.read_csv(Path(tmp)/'models.csv').unit.tolist(),['K'])
            self.assertEqual(len(calls),3)

    def test_categorical_object_booleans_do_not_warn_or_change_labels(self):
        from monitoring_v3.preparation import FrozenEncoder
        frame=pd.DataFrame({'state':pd.Series([True,False,None],dtype=object),'load':[1.,2.,3.]})
        with warnings.catch_warnings():
            warnings.simplefilter('error',FutureWarning)
            encoder=FrozenEncoder.fit(frame)
            result=encoder.native(frame)
        self.assertEqual(result.state.tolist(),['True','False','__MISSING__'])
        from monitoring_v3.notebook_live import LiveReport,LiveTrainingReport
        self.assertIs(LiveReport,LiveTrainingReport)
        book=json.loads((ROOT/'02_Обучение.ipynb').read_text(encoding='utf-8'))
        training=next(''.join(c['source']) for c in book['cells'] if 'RESULT = LF.run(CFG' in ''.join(c['source']))
        self.assertNotIn('LIVE_REPORT',training)
        self.assertIn('live_options=',training)

    def test_precision_counts_notifications_and_excludes_unknown_future(self):
        from monitoring_v3.alarm_scores import evaluate,alarm_stream
        from monitoring_v3.classification import _metrics
        cfg=load(overrides={'pre_window_days':1,'neutral_days':0})
        ix=pd.date_range('2024-01-01',periods=240,freq='h')
        ratio=pd.Series(.3,index=ix)
        for a in (8,78,108,233):ratio.iloc[a:a+5]=2.
        stream,_=alarm_stream(ratio,cfg)
        timeline=stream.assign(score=ratio,eligible=True,background_eligible=True,load_observed=True)
        events=pd.DataFrame([dict(stop=ix[100],start=ix[100],type='Авария',event_id='failure')])
        metrics,_=evaluate(timeline,events,cfg)
        self.assertAlmostEqual(metrics['precision'],1/3)
        self.assertEqual(metrics['precision_true_notifications'],1)
        self.assertEqual(metrics['precision_false_notifications'],2)
        self.assertEqual(metrics['precision_excluded_notifications'],1)
        self.assertEqual(metrics['recall'],1.)
        timeline.loc[ix[20],'load_observed']=False
        changed,_=evaluate(timeline,events,cfg)
        self.assertAlmostEqual(changed['precision'],.5)
        events=pd.concat([events,pd.DataFrame([dict(stop=ix[132],start=ix[132],type='Неизвестно',event_id='unknown')])])
        censored,_=evaluate(stream.assign(score=ratio,background_eligible=True),events,cfg)
        self.assertAlmostEqual(censored['precision'],.5)
        values=_metrics(pd.Series([1.,0.,1.]),np.array([[.1,.9],[.2,.8],[.8,.2]]),['0.0','1.0'])
        self.assertAlmostEqual(values['precision'],.5);self.assertAlmostEqual(values['recall'],.5)

    def test_native_heads_continue_with_known_labels_and_restore_every_month(self):
        ix=pd.date_range('2024-01-01',periods=155*24,freq='h');sample=ix[::6]
        rng=np.random.default_rng(14)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=load(overrides=dict(output=tmp,risk_horizons=[24],threads=1,cb_initial_trees=4,cb_update_trees=2,
                cb_depth=2,risk_policy={'samples':'continuous'}))
            items=[];contexts={};frames={}
            for number,unit in enumerate(('K1','K2')):
                directory=Path(tmp)/'experiments'/'local'/unit;directory.mkdir(parents=True)
                items.append(dict(experiment=unit,unit=unit,directory=directory))
                events=pd.DataFrame([dict(stop=ix[day*24],start=ix[day*24],type='Авария',
                    label_available_at=ix[day*24]+pd.Timedelta(days=delay),event_id=f'{unit}-{day}')
                    for day,delay in [(15+number*5,0),(70+number*5,8),(110+number*5,0)]])
                contexts[unit]=dict(unit=unit,load='load',frame=pd.DataFrame({'load':100.},index=ix),events=events,
                    eligible=pd.Series(True,index=ix))
                frames[unit]=pd.DataFrame(dict(unit=unit,features_valid=True,mean_abs=rng.normal(size=len(sample))),index=sample)
            session=MonthlyRisk(cfg,items,contexts,{'attach':False})
            months=pd.date_range('2024-02-01','2024-06-01',freq='MS');counts=[]
            for cutoff in months:
                end=cutoff+pd.offsets.MonthBegin(1)
                inputs={u:f.loc[f.index<end] for u,f in frames.items()}
                for item in items:inputs[item['unit']].to_parquet(item['directory']/'interpretation_features.parquet')
                status,history=session.step(cutoff,end,inputs,Progress(tmp))
                head=status['risk_24h'];counts.append(head.get('trees',0))
                if head['status'] in ('trained','trained_uncalibrated'):
                    self.assertLessEqual(pd.Timestamp(head['max_training_label_available_at']),cutoff)
                    package=joblib.load(head['shared_head'])
                    self.assertLess(pd.Timestamp(package['trained_through']),cutoff)
            self.assertTrue(all(b>a for a,b in zip(counts,counts[1:])),counts)
            self.assertEqual(counts[0],4)
            self.assertEqual(counts[-1],4+4*2)
            self.assertEqual(status['risk_24h']['fit_status'],'incremental')
            for h in history.values():self.assertTrue((pd.to_datetime(h.model_cutoff)<=pd.to_datetime(h.time)).all())
            restored=MonthlyRisk(cfg,items,contexts,{'attach':False})
            with patch('monitoring_v3.monthly_risk.train_classifier',side_effect=AssertionError('must restore')):
                for cutoff in months:
                    end=cutoff+pd.offsets.MonthBegin(1)
                    result,replayed=restored.step(cutoff,end,{u:f.loc[f.index<end] for u,f in frames.items()},Progress(tmp))
                    self.assertTrue(result['risk_24h']['restored'])
            for unit in history:pd.testing.assert_frame_equal(history[unit],replayed[unit])

    def test_detectors_advance_together_then_risk_and_section7_does_not_fit_risk(self):
        from monitoring_v3.data import load_sources
        from notebook_reports import run_interpretation
        from monitoring_v3.runtime import Model
        order=[];original=MonthlyRisk.step
        def detector(contexts,spec,cutoff,*args,**kwargs):
            order.append(('detector',pd.Timestamp(cutoff),spec['sources'][0]))
            return update_at(contexts,spec,cutoff,*args,**kwargs)
        def risk(session,cutoff,*args,**kwargs):
            order.append(('risk',pd.Timestamp(cutoff),'+'.join(i['unit'] for i in session.items)))
            return original(session,cutoff,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=two_boilers(tmp)
            with patch('monitoring_v3.pipeline.update_at',side_effect=detector),patch.object(MonthlyRisk,'step',new=risk):
                result=run(cfg)
            self.assertTrue(result['status'].status.eq('complete').all())
            for date in sorted({row[1] for row in order}):
                self.assertEqual([r[0] for r in order if r[1]==date],['detector','detector','risk'])
            directories=list(Path(cfg['output']).glob('experiments/*/К-*'))
            for directory in directories:
                self.assertIsNotNone(saved_monthly_statuses(directory))
                self.assertTrue((directory/'risk_timeline.parquet').exists())
                self.assertTrue((directory/'classifier_readiness.csv').exists())
            contexts=load_sources(cfg)[0]
            with (patch('interpretation_training.train_risk_group',side_effect=AssertionError('no final full fit')),
                 patch('monitoring_v3.classification.train_classifier',side_effect=AssertionError('no risk fit in section 7')),
                 patch('notebook_reports.show_table')):
                heads=run_interpretation(cfg,contexts,attach=False)
            self.assertTrue(heads.loc[heads['head'].str.startswith('risk_'),'workflow'].eq('monthly_prequential_v1').all())
            with (patch('monitoring_v3.pipeline.update_at',side_effect=AssertionError('restore detectors')),
                 patch('monitoring_v3.monthly_risk.train_classifier',side_effect=AssertionError('restore classifiers'))):
                repeated=run(cfg)
            self.assertTrue(repeated['status'].status.eq('complete').all())

    def test_attachment_preserves_actual_head_availability_in_month_exports(self):
        from interpretation_training import attach_heads
        from monitoring_v3.classification import CONTRACT
        from monitoring_v3.checkpoint_cache import STATE_CONTRACT
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);exported=directory/'model_history'/'2024-05';exported.mkdir(parents=True)
            (exported/'config.json').write_text(json.dumps({'state_contract':STATE_CONTRACT}),encoding='utf-8')
            source=directory/'head.joblib';joblib.dump(dict(contract=CONTRACT,report={},risk_policy={}),source)
            statuses={'risk_24h':dict(status='trained_uncalibrated',shared_head=str(source),available_at='2024-05-01')}
            attach_heads(directory,statuses,'2024-06-01',exported=exported)
            self.assertEqual(joblib.load(exported/'risk_24h.joblib')['available_at'],'2024-05-01')


if __name__=='__main__':unittest.main()
