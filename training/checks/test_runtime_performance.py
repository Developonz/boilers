"""Regression contracts for vectorization, bounded scoring and durable exports."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from monitoring_v3.config import load
from monitoring_v3.data import load_sources
from monitoring_v3.alarm_scores import alarm_stream,score_views,calibrate_threshold
from monitoring_v3.pipeline import run,update_at,score_context,score_month,scoring_context,score_fingerprint
from monitoring_v3.models import NormalModel
from monitoring_v3.artifacts import export_ready,publish_export
from monitoring_v3.runtime import Model
from monitoring_v3.progress import Progress
import legacy_group


def scalar_alarm(ratio,cfg):
    """Pre-optimization streaming contract; deliberately independent of run detection."""
    step=pd.Timedelta(cfg['step']);count=0;first=last_time=last_end=current=None
    confirmed=False;active=[];notices=[];rows=[]
    for t,v in ratio.items():
        contiguous=last_time is not None and t-last_time==step
        above=np.isfinite(v) and v>1
        if not contiguous or not above:
            if current is not None:
                current['end']=last_time+step;rows.append(current);last_end=current['end'];current=None
            count=0;first=None;confirmed=False
        notice=False
        if above:
            first=first if count else t;count+=1
            if not confirmed and count*step>=pd.Timedelta(hours=cfg['persist_hours']):
                confirmed=True;notice=last_end is None or t-last_end>=pd.Timedelta(hours=cfg['merge_hours'])
                current=dict(start=first,confirmed=t,end=t+step,notification=notice,peak=float(v))
            if current is not None:current['peak']=max(current['peak'],float(v))
        active.append(confirmed and above);notices.append(notice);last_time=t
    if current is not None:
        current['end']=ratio.index[-1]+step;rows.append(current)
    return pd.DataFrame(dict(ratio=ratio,active=active,notification=notices),index=ratio.index),pd.DataFrame(
        rows,columns=['start','confirmed','end','notification','peak'])


def scalar_calibration(score,cfg,budget):
    valid=score.notna()&np.isfinite(score)
    hours=valid.sum()*pd.Timedelta(cfg['step'])/pd.Timedelta('1h')
    values=score[valid]
    candidates=np.unique(np.r_[np.maximum(values.quantile(np.linspace(0,1,101)),1e-9),
                                max(float(values.max())+1e-9,1e-9)])
    for threshold in candidates:
        stream,_=scalar_alarm(score/threshold,cfg)
        rate=stream.notification.sum()*720/hours
        fraction=float(stream.active[valid].mean())
        if rate<=budget and fraction<=cfg['max_background_fraction']:
            return dict(threshold=float(threshold),hours=float(hours),status='calibrated',
                        alarm_rate=float(rate),alarm_fraction=fraction,budget=float(budget))
    raise AssertionError('No threshold')


def fixture(root):
    root=Path(root);folder=root/'source'/'К-1';folder.mkdir(parents=True)
    rng=np.random.default_rng(45);ix=pd.date_range('2024-01-01',periods=2400,freq='h');t=np.arange(len(ix))
    power=180+40*np.sin(t/50);power[1400:1424]=0
    state=np.where((t>=1400)&(t<1424),'Плановый ремонт','Работа')
    f=pd.DataFrame({'TimeValue':ix,'Состояние':state,'load':power,
        'temperature':230+.3*power+rng.normal(0,.2,len(t)),
        'pressure':10+.05*power+rng.normal(0,.1,len(t)),
        'mechanism':np.where(t%100<60,'on','off')})
    f.loc[1490:1515,'pressure']=np.nan
    f.to_parquet(folder/'hourly.parquet',index=False)
    wb=openpyxl.Workbook();sheet=wb.active;sheet.title='К-1'
    sheet.append(['col','роль','общий параметр','единица'])
    for row in [('load','нагрузка','load','т/ч'),('temperature','проверять','temperature','C'),
                ('pressure','проверять','pressure','МПа'),('mechanism','состояние','mechanism','')]:sheet.append(row)
    wb.save(root/'roles.xlsx');(root/'rules.json').write_text('{}',encoding='utf-8')
    return load(overrides=dict(groups={'test':['К-1']},data_root=str(root/'source'),
        sensor_file=str(root/'roles.xlsx'),sensor_rules=str(root/'rules.json'),output=str(root/'out'),
        first_test='2024-03-01',models=['CatBoost'],feature_sets=['conditions'],scenarios=['local'],
        startup_hours=2,min_train_rows=80,min_sensor_rows=20,calibration_rows=24,validate_rows=24,
        cb_initial_trees=4,cb_update_trees=2,cb_depth=2,memory_rows=60,replay_rows=20,threads=1))


class RuntimePerformanceTests(unittest.TestCase):
    def test_vectorized_alarms_match_streaming_contract(self):
        rng=np.random.default_rng(71)
        for step in ('1h','30min','15min'):
            for persist in (.5,1,3,7):
                for merge in (0,2,12):
                    cfg=load(overrides=dict(step=step,persist_hours=persist,merge_hours=merge))
                    ix=pd.date_range('2024-01-01',periods=300,freq=step).delete([15,70,71])
                    values=rng.choice([np.nan,np.inf,-np.inf,.2,1.,1.1,2.,5.],len(ix))
                    values[80:125]=2.;values[160:190]=3.
                    ratio=pd.Series(values,index=ix)
                    got=alarm_stream(ratio,cfg);expected=scalar_alarm(ratio,cfg)
                    for a,b in zip(got,expected):pd.testing.assert_frame_equal(a,b)

    def test_top_five_matches_pandas_with_missing_channels(self):
        rng=np.random.default_rng(72)
        for width in (1,3,5,7,150):
            ix=pd.date_range('2024-01-01',periods=100,freq='h')
            e=pd.DataFrame(rng.normal(size=(100,width)),index=ix)
            e=e.mask(rng.random(e.shape)<.4);e.iloc[10]=np.nan
            cfg=load(overrides={'minimum_coverage':.1})
            scale=pd.Series(1.,index=e.columns)
            got=score_views(e,scale,cfg)[0]['top5::instant'].score
            expected=e.abs().apply(lambda row:row.nlargest(min(5,row.count())).mean(),axis=1)
            expected=expected.where(e.notna().mean(axis=1)>=cfg['minimum_coverage'])
            pd.testing.assert_series_equal(got,expected,check_names=False)

    def test_calibration_is_unchanged(self):
        rng=np.random.default_rng(73);ix=pd.date_range('2024-01-01',periods=400,freq='h')
        score=pd.Series(rng.lognormal(size=400),index=ix);score.iloc[80:90]=np.nan
        cfg=load()
        for budget in cfg['alarm_budgets']:
            actual=calibrate_threshold(score,cfg,budget)
            expected=scalar_calibration(score,cfg,budget)
            self.assertEqual(actual,expected)

    def test_wide_mapping_and_forecast_do_not_fragment_frames(self):
        from monitoring_v3.preparation import conditions,regimes
        from monitoring_v3.pipeline import dataset
        ix=pd.date_range('2024-01-01',periods=80,freq='h')
        raw=pd.DataFrame({f's{i}':np.arange(80,dtype=float) for i in range(220)},index=ix)
        raw=pd.concat([raw,pd.DataFrame({'Состояние':'Работа','_Тип':''},index=ix)],axis=1)
        rows=[dict(original=f's{i}',key=f's{i}',role='load' if i==0 else 'target',units='') for i in range(220)]
        with warnings.catch_warnings():
            warnings.simplefilter('error',pd.errors.PerformanceWarning)
            frame,roles,*_=legacy_group.map_frame(raw,rows,'К-1')
            cfg=load();regime=regimes(frame.s0,20,cfg)
            features=conditions(frame,roles,regime,'К-1')
            targets=[c for c,r in roles.items() if r=='target']
            context=dict(features=features,cleaned=frame,targets=targets,eligible=regime.eligible,
                         changes=pd.DataFrame({c+'::held_hours':0. for c in targets},index=ix))
            dataset(context,'forecast',6,cfg)
        self.assertEqual(len(frame.columns),222)
        self.assertEqual(frame['Состояние'].iloc[0],'Работа')

    def test_month_buffer_matches_full_history_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(tmp);contexts,_=load_sources(cfg);c=contexts['К-1']
            start=pd.Timestamp(cfg['first_test']);end=start+pd.offsets.MonthBegin(1)
            for variant,horizon in [('conditions',0),('forecast',1),('forecast',6)]:
                spec=dict(method='CatBoost',variant=variant,horizon=horizon,sources=['К-1'],recipients=['К-1'])
                state,_=update_at(contexts,spec,start,cfg)
                full={k:(v.loc[v.index<end] if isinstance(v,(pd.Series,pd.DataFrame)) and v.index.equals(c['frame'].index)
                         else v) for k,v in c.items()}
                views,_,errors,features=score_context(state,full,spec,cfg)
                bounded=score_month(state,c,spec,cfg,start,end)
                pd.testing.assert_frame_equal(bounded['errors'],errors.loc[start:end-pd.Timedelta(cfg['step'])])
                pd.testing.assert_frame_equal(bounded['features'],features.loc[start:end-pd.Timedelta(cfg['step'])])
                for name,frame in views.items():
                    got=bounded['views'][name][frame.columns]
                    pd.testing.assert_frame_equal(got,frame.loc[start:end-pd.Timedelta(cfg['step'])])
                self.assertLess(len(scoring_context(c,spec,cfg,start,end)['frame']),len(full['frame']))

    def test_score_cache_detects_partial_month_changes_only(self):
        ix=pd.date_range('2024-01-01',periods=20,freq='h');frame=pd.DataFrame({'x':1.},index=ix)
        c={'frame':frame};end=ix[10]
        token=score_fingerprint(c,end,'model')
        later=frame.copy();later.iloc[15,0]=2.
        self.assertEqual(token,score_fingerprint({'frame':later},end,'model'))
        later.iloc[5,0]=2.
        self.assertNotEqual(token,score_fingerprint({'frame':later},end,'model'))

    def test_export_survives_scoring_failure_and_resume_uses_caches(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);root=Path(cfg['output'])
            with patch('monitoring_v3.pipeline.score_month',side_effect=RuntimeError('simulated scoring failure')):
                with self.assertRaisesRegex(RuntimeError,'simulated scoring failure'):run(cfg)
            index=pd.read_csv(root/'models.csv');export=Path(index.iloc[0].model_dir)
            self.assertTrue((export/'model.py').exists());self.assertTrue((export/'manifest.json').exists())
            model=Model(export);contexts,_=load_sources(cfg)
            self.assertEqual(len(model.score(contexts['К-1']['frame']).hourly),2400)
            with patch.object(NormalModel,'fit_initial',side_effect=AssertionError('initial fit repeated')):
                first=run(cfg)
            with patch('monitoring_v3.pipeline.score_month',side_effect=AssertionError('scoring cache ignored')):
                second=run(cfg)
            pd.testing.assert_frame_equal(first['summary'],second['summary'])
            self.assertTrue((root/'progress.jsonl').exists())

    def test_failed_publication_preserves_previous_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'history';source.mkdir();(source/'model').write_text('new')
            dest=root/'model';dest.mkdir();(dest/'model').write_text('previous')
            with patch('monitoring_v3.artifacts.shutil.copytree',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):publish_export(source,dest)
            self.assertEqual((dest/'model').read_text(),'previous')

    def test_failed_swap_restores_previous_export(self):
        import os
        replace=os.replace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'history';source.mkdir();(source/'model').write_text('new')
            dest=root/'model';dest.mkdir();(dest/'model').write_text('previous')
            def fail_swap(a,b):
                if '.publishing-' in str(a):raise OSError('swap failed')
                return replace(a,b)
            with patch('monitoring_v3.artifacts.os.replace',side_effect=fail_swap):
                with self.assertRaisesRegex(OSError,'swap failed'):publish_export(source,dest)
            self.assertEqual((dest/'model').read_text(),'previous')
            self.assertFalse(any(root.glob('.model.*')))

    def test_long_phase_emits_heartbeat_and_finishes(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            progress=Progress(tmp,interval=.01)
            with progress.stage('test phase'):
                Event().wait(.08)
            records=[json.loads(line) for line in (Path(tmp)/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(records[0]['status'],'started')
            self.assertTrue(any(row['status']=='running' for row in records))
            self.assertEqual(records[-1]['status'],'complete')

    def test_transient_windows_lock_does_not_abort_publication(self):
        import os
        replace=os.replace;attempts=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'history';source.mkdir();(source/'model').write_text('new')
            dest=root/'model';dest.mkdir();(dest/'model').write_text('previous')
            def temporary_lock(a,b):
                if '.publishing-' in str(a):
                    attempts.append(1)
                    if len(attempts)<3:raise PermissionError('scanner holds folder')
                return replace(a,b)
            with patch('monitoring_v3.artifacts.os.replace',side_effect=temporary_lock):publish_export(source,dest)
            self.assertEqual((dest/'model').read_text(),'new')
            self.assertEqual(len(attempts),3)
            self.assertFalse(any(root.glob('.model.*')))

    def test_interrupt_after_moving_old_export_restores_it(self):
        import os
        replace=os.replace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'history';source.mkdir();(source/'model').write_text('new')
            dest=root/'model';dest.mkdir();(dest/'model').write_text('previous')
            def interrupt_after_move(a,b):
                result=replace(a,b)
                if Path(a)==dest:raise KeyboardInterrupt('interrupted after moving old export')
                return result
            with patch('monitoring_v3.artifacts.os.replace',side_effect=interrupt_after_move):
                with self.assertRaises(KeyboardInterrupt):publish_export(source,dest)
            self.assertEqual((dest/'model').read_text(),'previous')
            self.assertFalse(any(root.glob('.model.*')))


if __name__=='__main__':unittest.main()
