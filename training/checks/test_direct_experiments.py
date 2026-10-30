"""Rolling evaluation, disk memory boundaries, macro-F and alarm-policy checks."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'checks')]
from test_direct_risk import context
from direct_risk import (direct_options,direct_dataset,direct_features,direct_threshold,
    direct_thresholds,direct_alarm_thresholds,direct_alarm_evaluate)
from direct_experiments import (prepare_direct_cache,cached_direct_labels,load_experiment_features,
    rolling_direct_folds,run_direct_suite,aggregate_direct_folds,adaptive_direct_fold,select_validation_variants,
    combined_direct_report)


class DirectExperimentsTests(unittest.TestCase):
    def test_adaptive_validation_expands_only_using_past_and_preserves_test(self):
        t=pd.date_range('2024-01-01',periods=24*200,freq='h')
        ix=pd.MultiIndex.from_arrays([t,['K']*len(t)],names=['time','unit'])
        meta=pd.DataFrame({'eligible':True,'label':0.,'label_group':['bg::'+str(v.to_period('W')) for v in t],
                           'label_known_at':t+pd.Timedelta('24h')},index=ix)
        for day in (30,70,110,150,180):
            window=(t>=t[day*24]-pd.Timedelta('24h'))&(t<t[day*24])
            meta.loc[window,'label']=1.;meta.loc[window,'label_group']='event'+str(day)
            meta.loc[window,'label_known_at']=t[day*24]+pd.Timedelta('1h')
        fold={'fold':'fold_01','train_end':str(t[130*24]),'validation_end':str(t[160*24]),'test_end':str(t[-1])}
        plan=adaptive_direct_fold(meta,direct_options({'horizon_hours':24}),fold,{'validation_days':30})
        self.assertEqual(plan['split_status'],'validation_expanded')
        self.assertEqual(pd.Timestamp(plan['train_end']),t[100*24])
        self.assertEqual(plan['test_end'],fold['test_end'])
        self.assertGreaterEqual(plan['validation_events'],2);self.assertGreaterEqual(plan['train_events'],2)
        changed=meta.copy();changed.loc[t>=t[160*24],['label','label_group']]=[1.,'changed_test_event']
        self.assertEqual(plan,adaptive_direct_fold(changed,direct_options({'horizon_hours':24}),fold,{'validation_days':30}))

    def test_unavailable_threshold_is_not_a_fake_point_five_optimum(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            threshold,curve,status=direct_threshold(pd.Series([0,0,0]),np.array([.001,.002,.004]),direct_options())
        self.assertTrue(np.isinf(threshold));self.assertIn('single_class',status)
        rules,_=direct_thresholds({'threshold_table':curve,'options':{'macro_thresholds':True}})
        self.assertEqual(rules['default_05']['threshold'],.5)
        self.assertFalse(rules['best_f2']['available']);self.assertTrue(np.isinf(rules['best_f2']['threshold']))

    def test_variant_selection_never_uses_test_quality(self):
        rows=[dict(method='unit',name='K',horizon_hours=168,fold='fold_01',status='complete',
            validation_index_hash='same',validation_end='2025-01-01',variant=v,
            validation_average_precision=ap,test_average_precision=test,folder=v)
            for v,ap,test in [('full',.3,.99),('numeric',.4,.01),('numeric_states',.4,.9)]]
        chosen=select_validation_variants(pd.DataFrame(rows))
        self.assertEqual(chosen.selected_variant.iloc[0],'numeric')
        for r in rows:r['test_average_precision']=100-r['test_average_precision']
        self.assertEqual(select_validation_variants(pd.DataFrame(rows)).selected_variant.iloc[0],'numeric')

    def test_overall_report_sums_errors_and_keeps_alarm_state_between_versions(self):
        import json,matplotlib
        matplotlib.use('Agg')
        records=[];parts=[];options=direct_options({'horizon_hours':24,'alarm_selection':True})
        t=pd.date_range('2024-01-01',periods=24,freq='h')
        with tempfile.TemporaryDirectory() as temp:
            for i in range(2):
                a,b=i*12,(i+1)*12;folder=Path(temp)/f'fold_{i}';folder.mkdir()
                ix=pd.MultiIndex.from_arrays([t[a:b],['K']*12],names=['time','unit'])
                y=(np.arange(a,b)>=8)&(np.arange(a,b)<16)
                p=pd.DataFrame({'label':y.astype(float),'score':np.where(np.arange(a,b)>=8,.8,.1),
                    'evaluation_eligible':True,'label_group':np.where(y,'event','bg'),
                    'label_known_at':pd.Timestamp('2024-01-03'),'model_available_at':t[a]},index=ix)
                p.to_parquet(folder/'test_predictions.parquet');parts.append(p)
                threshold=.5 if i==0 else .6
                (folder/'settings.json').write_text(json.dumps({'options':options,'train_prevalence':.1,
                    'thresholds':{'default_05':{'threshold':.5,'status':'fixed','title':'0.5'},
                        'alarm_budget_recall':{'threshold':threshold,'status':'selected','title':'budget','category':'alarm_policy'}},'alarm_step':'1h'}))
                pd.DataFrame([dict(partition=k,rows=100,positive=20,negative=80,positive_events=2) for k in ('train','validation','test')]).to_csv(folder/'split_summary.csv',index=False)
                pd.DataFrame({'feature':['zero','useful'],'importance':[0.,5.]}).to_csv(folder/'feature_importance.csv',index=False)
                records.append(dict(method='unit',name='K',variant='numeric',horizon_hours=24,folder=str(folder),fold=f'fold_{i}',
                    validation_end=str(t[a]),test_end=str(t[b-1]+pd.Timedelta('1h'))))
            html,summary=combined_direct_report(pd.DataFrame(records),{},Path(temp)/'report')
            default=summary.loc[summary.decision.eq('default_05')].iloc[0]
            self.assertEqual(default.rows,24);self.assertEqual(default.true_positive,8)
            self.assertEqual(default.false_positive,8);self.assertEqual(default.false_negative,0)
            policy=summary.loc[summary.decision.eq('alarm_budget_recall')].iloc[0]
            self.assertEqual(policy.rolling_duplicate_notifications,0)
            self.assertIn('data:image/png;base64,',html);self.assertIn('f1-score',html)
            self.assertNotIn('<td>zero</td>',html)

    def test_score_plot_breaks_unobserved_intervals(self):
        import matplotlib.pyplot as plt
        from direct_risk import direct_plot_segments
        fig,ax=plt.subplots();t=pd.to_datetime(['2024-01-01 00:00','2024-01-03 00:00'])
        line=direct_plot_segments(ax,t,[.1,.8])[0]
        self.assertTrue(np.isnan(line.get_ydata()[1:-1]).all());plt.close(fig)

    def test_cache_has_eligible_rows_and_reuses_features_for_horizons(self):
        c,cfg=context(600,failures=[400]);options=direct_options()
        with tempfile.TemporaryDirectory() as folder:
            cache,table=prepare_direct_cache({'К-1':c},cfg,options,folder,progress=lambda s:None)
            x=pd.read_parquet(Path(cache['К-1']['folder'])/'features.parquet')
            self.assertLess(len(x),len(c['frame']))
            self.assertTrue(all(dtype==np.dtype('float32') for dtype in x.select_dtypes('number').dtypes))
            with patch('direct_experiments.direct_features',side_effect=AssertionError('must use cache')):
                restored,_=prepare_direct_cache({'К-1':c},cfg,options,folder,progress=lambda s:None)
                for h in (24,168):
                    labels,_=cached_direct_labels(c,restored['К-1'],cfg,direct_options({**options,'horizon_hours':h}))
                    self.assertEqual(len(x),len(labels))
            self.assertTrue((Path(cache['К-1']['folder'])/'tail.parquet').exists())

    def test_group_uses_intersection_not_fleet_union_and_respects_memory_limit(self):
        c,cfg=context();other=deepcopy(c);other['unit']='К-2'
        for key in ('frame','cleaned'):other[key]=other[key].rename(columns={'temperature':'different_temperature'})
        other['roles']={'load':'load','different_temperature':'target','state':'state'}
        other['declared_units']={'load':'t/h','different_temperature':'C','state':''}
        with tempfile.TemporaryDirectory() as folder:
            cache,_=prepare_direct_cache({'К-1':c,'К-2':other},cfg,direct_options(),folder,progress=lambda s:None)
            x,schema,_=load_experiment_features(cache,['К-1','К-2'],'full')
            self.assertFalse(any('temperature' in name for name in x))
            self.assertTrue(x.index.is_unique)
            with self.assertRaises(ValueError):load_experiment_features(cache,['К-1','К-2'],'full',max_mb=.0001)

    def test_macro_f_matches_sklearn_for_both_classes(self):
        from sklearn.metrics import fbeta_score
        y=pd.Series([1,0,0,1,0,1,0,0]);score=np.array([.9,.8,.7,.6,.5,.4,.3,.2])
        _,table,_=direct_threshold(y,score,direct_options())
        rules,table=direct_thresholds({'threshold_table':table,'options':{'macro_thresholds':True}})
        self.assertIn('best_macro_f1',rules)
        for row in table.itertuples():
            for beta,column in ((.5,'macro_f05'),(1,'macro_f1'),(2,'macro_f2')):
                self.assertAlmostEqual(getattr(row,column),fbeta_score(y,score>=row.threshold,beta=beta,average='macro',labels=[0,1],zero_division=0))

    def test_alarm_budget_requires_new_confirmed_warning_and_has_exact_ties(self):
        c,cfg=context(600,failures=[400]);options=direct_options({'alarm_selection':True,'alarm_candidates':5})
        _,meta,_,_=direct_dataset({'К-1':c},cfg,options)
        p=meta.loc[meta.eligible].copy();p['score']=np.where(p.label.eq(1),.9,.1)
        date=c['frame'].index[-1]
        rules,table=direct_alarm_thresholds(p,{'К-1':c},options,cfg['step'],date,c['frame'].index[0],date)
        self.assertEqual(rules['alarm_budget_recall']['threshold'],.9)
        metric,stream,events=direct_alarm_evaluate(p,{'К-1':c},options,.9,cfg['step'],date)
        self.assertEqual(metric['detected_events'],1);self.assertEqual(metric['false_notifications'],0)
        self.assertEqual(events.lead_hours.iloc[0],166.)
        always,_,_=direct_alarm_evaluate(p,{'К-1':c},options,.1,cfg['step'],date)
        self.assertEqual(always['detected_events'],0) # A background alarm is not a fresh failure warning.

    def test_alarm_state_does_not_restart_when_fold_threshold_changes(self):
        t=pd.date_range('2024-01-01',periods=8,freq='h')
        ix=pd.MultiIndex.from_arrays([t,['K']*8],names=['time','unit'])
        p=pd.DataFrame({'score':[.9]*4+[.6]*4,'label':0.,'label_group':'background',
            'label_known_at':pd.Timestamp('2024-01-02')},index=ix)
        thresholds=pd.Series([.8]*4+[.5]*4,index=ix)
        metric,stream,_=direct_alarm_evaluate(p,{},direct_options(),thresholds)
        self.assertEqual(metric['false_notifications'],1)

    def test_native_suite_is_sequential_and_keeps_negative_test_periods(self):
        c,cfg=context(24*320,failures=[24*d for d in (50,100,150,200,250,300)])
        other=deepcopy(c);other['unit']='К-2';cfg['groups']={'test':['К-1','К-2']}
        contexts={'К-1':c,'К-2':other}
        options=direct_options({'iterations':3,'early_stopping_rounds':2,'depth':2,'threads':1,
            'macro_thresholds':True,'alarm_selection':True,'alarm_candidates':5})
        rolling={'folds':2,'validation_days':45,'test_days':30,'end':None}
        folds=rolling_direct_folds(contexts,cfg,rolling)
        self.assertEqual(folds[0]['test_end'],folds[1]['validation_end'])
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            cache,_=prepare_direct_cache(contexts,cfg,options,Path(folder)/'cache',progress=lambda s:None)
            index,metrics=run_direct_suite(contexts,cfg,options,cache,['unit','group'],['numeric','full'],[24,168],rolling,folder,progress=lambda s:None)
            self.assertEqual(len(index),24,index.to_string())
            self.assertTrue(index.status.eq('complete').all(),index[['status','reason']].to_string())
            self.assertNotIn('model',index.columns) # Only paths/metadata retained, not fitted objects.
            self.assertTrue(metrics.loc[metrics.fold.eq('fold_01'),'positive'].eq(0).all())
            self.assertIn('alarm_budget_recall',set(metrics.decision))
            self.assertIn('best_macro_f1',set(metrics.decision))
            aggregate=aggregate_direct_folds(metrics)
            self.assertTrue(aggregate.folds.eq(2).all())


if __name__=='__main__':unittest.main()
