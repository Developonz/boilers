"""Causal historical risk, legacy presentation and independent notebook slots."""
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
from monitoring_v3.config import load
from monitoring_v3.risk_reporting import alarm_risk_table,readiness_rows
from monitoring_v3.risk_history import build_risk_history
from monitoring_v3.training_plot import plot_training_timeline
from monitoring_v3.notebook_live import LiveTrainingReport
from monitoring_v3.classification import train_classifier
from monitoring_v3.event_registry import empty_registry


def plot_sample():
    cfg=load(overrides=dict(first_test='2024-01-01',persist_hours=3,risk_horizons=[24,168]))
    ix=pd.date_range('2024-01-01',periods=120,freq='h')
    ratio=np.r_[np.zeros(10),np.full(15,2.),np.zeros(30),np.full(15,3.),np.zeros(50)]
    d=pd.DataFrame(dict(ratio=ratio,score=ratio,eligible=True,background_eligible=True),index=ix)
    mode=pd.Series('работа',index=ix);mode.iloc[30:35]='стоит';mode.iloc[40:45]='неизвестно'
    d.loc[ix[40:48],'ratio']=np.nan
    context=dict(regime=pd.DataFrame({'mode':mode}),events=empty_registry())
    return cfg,d,context


class LiveTrainingTests(unittest.TestCase):
    def test_alarm_labels_distinguish_probability_score_and_late_availability(self):
        cfg,d,_=plot_sample();ix=d.index
        risk=pd.DataFrame([
            dict(time=ix[12],horizon_hours=24,model_cutoff=ix[0],probability=.6,raw_score=.9,status='trained',probability_status='calibrated_group_holdout'),
            dict(time=ix[18],horizon_hours=24,model_cutoff=ix[0],probability=.8,raw_score=.95,status='trained',probability_status='calibrated_group_holdout'),
            dict(time=ix[12],horizon_hours=168,model_cutoff=ix[0],probability=np.nan,raw_score=.75,status='trained_uncalibrated',probability_status='uncalibrated_score_only'),
            dict(time=ix[60],horizon_hours=24,model_cutoff=ix[0],probability=.7,raw_score=.8,status='trained',probability_status='calibrated_group_holdout'),
            # A malformed file claiming to use a future model must be ignored.
            dict(time=ix[57],horizon_hours=24,model_cutoff=ix[80],probability=.99,raw_score=.99,status='trained',probability_status='calibrated_group_holdout'),
        ])
        table=alarm_risk_table(d,risk,cfg)
        self.assertEqual(len(table),2)
        self.assertAlmostEqual(table.iloc[0].risk_24h_at_confirmation,.6)
        self.assertIn('60% → max 80%',table.iloc[0].risk_24h_text)
        self.assertIn('балл 0.75',table.iloc[0].risk_168h_text)
        self.assertNotIn('75%',table.iloc[0].risk_168h_text)
        self.assertTrue(pd.isna(table.iloc[1].risk_24h_at_confirmation))
        self.assertIn('с 03.01 12:00',table.iloc[1].risk_24h_text)

    def test_original_two_panel_style_and_unknown_hours_are_preserved(self):
        import matplotlib.pyplot as plt
        cfg,d,context=plot_sample()
        fig,table=plot_training_timeline(d,context,cfg,'CatBoost · К-5')
        try:
            self.assertEqual(len(fig.axes),2)
            self.assertEqual(fig.axes[0].get_yscale(),'symlog')
            self.assertEqual(fig.axes[1].get_yticklabels()[0].get_text(),'тревога')
            daily=fig.axes[0].lines[0].get_ydata()
            self.assertEqual(len(daily),5)
            self.assertTrue(any('#1' in t.get_text() for t in fig.axes[0].texts))
            self.assertIn('нет оценки',fig.axes[0].texts[0].get_text())
            buffer=io.BytesIO();fig.savefig(buffer,format='png',bbox_inches='tight')
            self.assertGreater(len(buffer.getvalue()),10000)
        finally:plt.close(fig)

    def test_readiness_counts_groups_and_censoring_not_hour_count(self):
        cfg=load();statuses={'risk_24h':dict(status='trained_uncalibrated',
            label_audit=[dict(positive_events=3,negative_groups=20,incomplete_future=30)],
            partitions={'deployment_train':{'groups':{'1.0':2,'0.0':15}},
                        'deployment_calibration':{'groups':{'1.0':1,'0.0':2}}})}
        row=readiness_rows(statuses,cfg).iloc[0]
        self.assertEqual(row['не_хватает_аварий_обучение'],0)
        self.assertEqual(row['не_хватает_аварий_калибровка'],1)
        self.assertEqual(row['исключено_неполное_будущее'],30)

    def test_registered_emergency_without_an_alarm_is_reported(self):
        from monitoring_v3.risk_dataset import make_examples
        ix=pd.date_range('2024-01-01',periods=100,freq='h')
        frame=pd.DataFrame({'load':100.},index=ix)
        events=pd.DataFrame([dict(stop=ix[60],type='Авария',label_available_at=ix[60],event_id='failure')])
        context=dict(frame=frame,load='load',unit='K',events=events)
        f=pd.DataFrame(dict(unit='K',features_valid=True,episode_active=0.,episode_sample_due=False,episode_group=''),index=ix)
        _,_,_,audit=make_examples(f,context,24,ix[-1],load())
        self.assertEqual(audit['registered_emergencies'],1)
        self.assertEqual(audit['positive_events'],0)
        self.assertEqual(audit['emergency_ids_without_examples'],['failure'])

    def test_separate_model_slots_do_not_overwrite_previous_model(self):
        class Handle:
            def __init__(self):self.updates=[]
            def update(self,x):self.updates.append(x)
        handles=[]
        def display(*args,**kwargs):
            h=Handle();handles.append(h);return h
        class Progress:
            view=None
        with tempfile.TemporaryDirectory() as tmp,patch('monitoring_v3.notebook_live.notebook_display',return_value=object()),patch('IPython.display.display',side_effect=display):
            cfg=load(overrides={'output':tmp});report=LiveTrainingReport(cfg)
            spec=dict(method='CatBoost',variant='conditions',scenario='local',recipients=['К-5'])
            report.start('a',spec,{'К-5':{}},[1,2],Progress())
            first=report.panels[('a','К-5')];first.chart.update('first chart')
            old_count=len(first.view.handle.updates)
            report.start('b',dict(spec,recipients=['К-6']),{'К-6':{}},[1],Progress())
            second=report.panels[('b','К-6')];second.update({'phase':'second'},force=True)
            self.assertEqual(len(handles),6)
            self.assertIsNot(first.chart,second.chart)
            self.assertEqual(first.chart.updates,['first chart'])
            self.assertEqual(len(first.view.handle.updates),old_count)

    def test_monthly_risk_uses_known_labels_and_caches_without_final_head(self):
        ix=pd.date_range('2024-01-01',periods=230*24,freq='h');sample=ix[::6]
        rng=np.random.default_rng(91)
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load(overrides=dict(output=tmp,first_test='2024-02-01',risk_horizons=[24],
                cb_initial_trees=4,cb_update_trees=2,cb_depth=2,threads=1))
            events=pd.DataFrame([dict(stop=ix[day*24],start=ix[day*24],type='Авария',
                label_available_at=ix[day*24]+pd.Timedelta(days=late),event_id=str(day)) for day,late in [(20,0),(60,40),(110,0),(160,0),(210,0)]])
            frame=pd.DataFrame({'load':100.},index=ix)
            context=dict(frame=frame,load='load',unit='K',eligible=pd.Series(True,index=ix),events=events)
            f=pd.DataFrame(dict(unit='K',mean_abs=rng.normal(size=len(sample)),features_valid=True,
                episode_active=1.,episode_age_hours=0.,episode_sample_due=True,
                episode_group=[str(t.to_period('W')) for t in sample]),index=sample)
            directory=Path(tmp)/'K';directory.mkdir();f.to_parquet(directory/'interpretation_features.parquet')
            items=[dict(experiment='test',unit='K',directory=directory)]
            result=build_risk_history(cfg,items,{'K':context},ix[-1])['K']
            self.assertTrue((pd.to_datetime(result.model_cutoff)<=pd.to_datetime(result.time)).all())
            self.assertTrue(result.loc[result.time<pd.Timestamp('2024-05-01'),'raw_score'].isna().all())
            self.assertTrue(result.raw_score.notna().any())
            for path in Path(tmp).glob('risk_history/*/*/risk_24h/*/status.json'):
                status=json.loads(path.read_text(encoding='utf-8'))
                if 'max_training_label_available_at' in status:
                    self.assertLessEqual(pd.Timestamp(status['max_training_label_available_at']),pd.Timestamp(path.parent.name))
            with patch('monitoring_v3.risk_history.train_classifier',side_effect=AssertionError('must restore cache')):
                restored=build_risk_history(cfg,items,{'K':context},ix[-1])['K']
            pd.testing.assert_frame_equal(result,restored)
            self.assertFalse((directory/'model').exists())
            from monitoring_v3.risk_history import read_risk_timeline
            self.assertIsNotNone(read_risk_timeline(directory,cfg,context))
            changed=dict(context,events=events.assign(type='План'))
            self.assertIsNone(read_risk_timeline(directory,cfg,changed))
            f.assign(mean_abs=0.).to_parquet(directory/'interpretation_features.parquet')
            self.assertIsNone(read_risk_timeline(directory,cfg,context))


if __name__=='__main__':unittest.main()
