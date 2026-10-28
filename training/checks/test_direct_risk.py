"""Focused checks for the standalone sensor-window classifier."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib')]
from monitoring_v3.config import load
from monitoring_v3.data import prepare_unit
from direct_risk import (direct_options,direct_features,direct_dataset,direct_split,
    direct_balance,direct_fit,direct_predict,direct_event_report,direct_threshold)


def context(rows=120,failures=()):
    ix=pd.date_range('2024-01-01',periods=rows,freq='h')
    t=np.arange(rows);rng=np.random.default_rng(17)
    temperature=100+.002*t+rng.normal(0,.1,rows)
    load0=np.full(rows,200.);state=np.full(rows,'Работа',dtype=object)
    for point in failures:
        start=max(0,point-168)
        temperature[start:point]+=np.linspace(0,10,point-start)
        load0[point:point+24]=0;state[point:point+24]='Аварийный ремонт'
    frame=pd.DataFrame(dict(load=load0,temperature=temperature,
        state=np.where(t%48<36,'on','off'),Состояние=state),index=ix)
    cfg=load(overrides=dict(groups={'test':['К-1']},run_thresholds={'К-1':50},startup_hours=2,
        first_test='2024-03-01',min_sensor_rows=10,threads=1))
    roles={'load':'load','temperature':'target','state':'state'}
    c=prepare_unit(frame,roles,'К-1',cfg,{})
    c['declared_units']={'load':'t/h','temperature':'C','state':''}
    return c,cfg


class DirectRiskTests(unittest.TestCase):
    def test_exact_features_and_no_future_dependency(self):
        c,cfg=context();options=direct_options()
        c['cleaned']['temperature']=np.arange(120)*2.+5
        x,meta,schema=direct_features(c,cfg,options)
        self.assertEqual(len(x.columns),2*11+4+2)
        self.assertFalse(any(name.startswith('group::') for name in x))
        self.assertAlmostEqual(x.iloc[40]['num::temperature::trend_6h'],2.,places=4)
        self.assertAlmostEqual(x.iloc[40]['num::temperature::trend_24h'],2.,places=4)
        self.assertAlmostEqual(x.iloc[40]['num::temperature::std_24h'],np.std(np.arange(24)*2),places=4)
        self.assertEqual(x.iloc[40]['num::temperature::lag_24h'],37.)
        self.assertEqual(x.iloc[40]['num::temperature::delta_6h'],12.)
        altered={**c,'cleaned':c['cleaned'].copy(),'frame':c['frame'].copy()}
        altered['cleaned'].loc[altered['cleaned'].index[61]:,['load','temperature']]=10000
        altered['frame'].loc[altered['frame'].index[61]:,'state']='future'
        future,_,_=direct_features(altered,cfg,options)
        pd.testing.assert_frame_equal(x.iloc[:61],future.iloc[:61])
        self.assertFalse(np.isinf(x.select_dtypes(include='number').to_numpy()).any())

    def test_windows_do_not_bridge_stop_and_partial_history_is_retained(self):
        c,cfg=context(160,failures=[70]);x,meta,_=direct_features(c,cfg,direct_options())
        self.assertFalse(meta.iloc[80].eligible)
        self.assertTrue(meta.iloc[94].eligible)
        self.assertTrue(pd.isna(x.iloc[94]['num::temperature::lag_12h']))
        self.assertTrue(pd.notna(x.iloc[94]['num::temperature::mean_24h']))
        self.assertTrue(meta.iloc[0].eligible)
        self.assertTrue(pd.isna(x.iloc[0]['num::temperature::std_24h']))

    def test_missing_category_is_not_switch_and_resets_held_age(self):
        c,cfg=context();c['frame'].loc[c['frame'].index[10],'state']=None
        x,_,_=direct_features(c,cfg,direct_options())
        self.assertEqual(x.iloc[10]['cat::state::value'],'__MISSING__')
        self.assertTrue(pd.isna(x.iloc[10]['cat::state::held_hours']))
        self.assertEqual(x.iloc[11]['cat::state::held_hours'],0)
        self.assertEqual(x.iloc[11]['cat::state::switches_24h'],0)

    def test_inferred_preparation_does_not_reach_back_into_its_reference_period(self):
        c,cfg=context();c['threshold_source']='initial_past'
        c['preparation_reference_end']=c['frame'].index[48]
        _,meta,_=direct_features(c,cfg,direct_options())
        self.assertFalse(meta.iloc[:48].eligible.any())
        self.assertTrue(meta.iloc[48].eligible)

    def test_split_purges_known_future_and_groups_and_balances_only_train(self):
        c,cfg=context(24*320,failures=[24*d for d in (50,100,150,200,250,300)])
        options=direct_options();x,m,s,a=direct_dataset({'К-1':c},cfg,options)
        masks,summary,boundaries=direct_split(m,options)
        groups={name:set(m.loc[mask,'label_group']) for name,mask in masks.items()}
        self.assertFalse(groups['train']&groups['validation'])
        self.assertFalse(groups['train']&groups['test'])
        self.assertFalse(groups['validation']&groups['test'])
        self.assertTrue(m.loc[masks['train'],'label_known_at'].le(pd.Timestamp(boundaries['train_end'])).all())
        self.assertTrue(m.loc[masks['validation'],'label_known_at'].le(pd.Timestamp(boundaries['validation_end'])).all())
        times=m.index.get_level_values('time')
        self.assertLess(times[masks['train']].max()+pd.Timedelta('24h'),times[masks['validation']].min())
        selected,weights,balanced=direct_balance(m,masks['train'],options)
        self.assertEqual(balanced.rows.iloc[0],balanced.rows.iloc[1])
        self.assertAlmostEqual(balanced.total_weight.iloc[0],balanced.total_weight.iloc[1])
        self.assertTrue(m.iloc[selected].index.is_unique)
        self.assertTrue((m.iloc[selected].index.get_level_values('time').hour%6==0).all())
        self.assertTrue(m.loc[masks['test'],'label'].mean()<.5)

    def test_threshold_is_validation_optimum(self):
        y=pd.Series([0,0,0,0,1,1]);scores=np.array([.01,.02,.03,.06,.08,.1])
        threshold,table,status=direct_threshold(y,scores,direct_options())
        self.assertEqual(threshold,.08)
        self.assertIn('validation',status)

    def test_entire_pre_failure_week_is_positive_and_unknown_tail_is_not_negative(self):
        c,cfg=context(500,failures=[250]);options=direct_options()
        _,meta,_,_=direct_dataset({'К-1':c},cfg,options)
        positive=meta.loc[meta.label.eq(1)]
        self.assertEqual(len(positive),168)
        self.assertEqual(positive.index.get_level_values('time').min(),c['frame'].index[250]-pd.Timedelta('168h'))
        self.assertEqual(positive.index.get_level_values('time').max(),c['frame'].index[249])
        self.assertTrue(meta.iloc[-100:].label.isna().all())
        frame=c['frame'].copy();frame['Состояние']=frame['Состояние'].replace('Аварийный ремонт','Плановый ремонт')
        planned=prepare_unit(frame,c['roles'],'К-1',cfg,{})
        _,planned_meta,_,_=direct_dataset({'К-1':planned},cfg,options)
        self.assertFalse(planned_meta.label.eq(1).any())
        self.assertTrue(planned_meta.iloc[82:250].label.isna().all())

    def test_native_fit_save_restore_and_inference(self):
        c,cfg=context(24*320,failures=[24*d for d in (50,100,150,200,250,300)])
        options=direct_options({'iterations':12,'early_stopping_rounds':4,'depth':2,'threads':1})
        x,m,s,a=direct_dataset({'К-1':c},cfg,options)
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            package,masks,summary,balanced=direct_fit(x,m,s,options,folder,verbose=False)
            self.assertTrue((Path(folder)/'classifier.cbm').exists())
            self.assertTrue((Path(folder)/'test_predictions.parquet').exists())
            self.assertEqual(len(package['metrics']),5)
            tested=package['predictions'].loc[lambda d:d.evaluation_eligible]
            predicted=direct_predict(package,x.loc[tested.index])
            np.testing.assert_allclose(predicted.score,tested.score)
            self.assertGreater(len(direct_event_report(package,{'К-1':c})),0)
            with patch('catboost.CatBoostClassifier.fit',side_effect=AssertionError('must restore')):
                restored,_,_,_=direct_fit(x,m,s,options,folder,verbose=False)
            self.assertEqual(package['fingerprint'],restored['fingerprint'])


if __name__=='__main__':unittest.main()
