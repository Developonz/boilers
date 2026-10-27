"""Failure-oriented contracts for history, stops, uncertainty and deployment."""
from pathlib import Path
from contextlib import redirect_stdout
import io,json,sys,tempfile,unittest
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from monitoring_v3.config import load
from monitoring_v3.preparation import regimes,clean
from monitoring_v3.alarm_scores import score_views,alarm_stream,view_policy
from monitoring_v3.event_registry import event_type_at
from monitoring_v3.data import prepare_unit,registry,load_sources
from monitoring_v3.partitioning import partition_history
from monitoring_v3.pipeline import update_at,experiments,run,score_month
from monitoring_v3.interpretation import risk_labels,train_head,episode_features
from monitoring_v3.classification import probabilities
from monitoring_v3.runtime import Model
from monitoring_v3.regime_evidence import fit_evidence,evidence_features
from monitoring_v3.input_grid import normalize_grid
from monitoring_v3.importance import regressor_importance,write_regressor_importance
from test_runtime_performance import fixture


class QualityLifecycleTests(unittest.TestCase):
    def test_feature_importance_is_per_target_cached_and_does_not_change_predictions(self):
        from test_target_training import sample
        from monitoring_v3.models import NormalModel
        from unittest.mock import patch
        from catboost import CatBoostRegressor
        f,y,cfg=sample();model=NormalModel('CatBoost',cfg).fit_initial(f,y)
        before=model.expected(f,y)[0]
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'first';second=Path(tmp)/'second'
            write_regressor_importance(model,first,'v1')
            table=pd.read_csv(first/'feature_importance.csv')
            self.assertEqual(set(table.target),set(y));self.assertTrue(table.status.eq('available').all())
            np.testing.assert_allclose(table.groupby('target').importance_percent.sum(),100.)
            self.assertEqual(set(table.feature),set(f))
            cached=(first/'feature_importance.csv').read_bytes()
            with patch.object(CatBoostRegressor,'get_feature_importance',side_effect=AssertionError('cache ignored')):
                write_regressor_importance(model,first,'v1')
            self.assertEqual((first/'feature_importance.csv').read_bytes(),cached)
            write_regressor_importance(model,second,'v2',first)
            self.assertTrue(pd.read_csv(second/'feature_importance_delta.csv').delta_percentage_points.eq(0).all())
            with patch.object(CatBoostRegressor,'get_feature_importance',side_effect=RuntimeError('explanation unavailable')):
                self.assertTrue(regressor_importance(model).status.eq('importance_failed').all())
        pd.testing.assert_frame_equal(before,model.expected(f,y)[0])

    def test_group_holdout_preserves_positive_calibration_windows_by_purging_neighbour_train(self):
        cfg=load(overrides={'cb_initial_trees':4,'cb_update_trees':2,'cb_depth':2,'threads':1})
        ix=pd.date_range('2020-01-01',periods=1700,freq='6h')
        y=pd.Series(0.,index=ix);g=pd.Series([f'bg::{t.to_period("W")}' for t in ix],index=ix)
        for i,pos in enumerate((200,600,1000,1400)):
            y.iloc[pos:pos+4]=1.;g.iloc[pos:pos+4]=f'event::{i}'
        f=pd.DataFrame({'unit':'K','evidence':np.sin(np.arange(len(ix))/10)},index=ix)
        with tempfile.TemporaryDirectory() as tmp:
            report=train_head(f,y,g,cfg,tmp,horizon_hours=24,task='risk')
            self.assertEqual(report['status'],'trained')
            self.assertGreaterEqual(report['partitions']['deployment_calibration']['groups']['1.0'],2)
            head=joblib.load(Path(tmp)/'interpreter.joblib')
            tr=ix[[str(t) in head['row_fingerprints'] for t in ix]]
            cal=ix[g.isin(head['calibration_group_ids'])]
            # Only effective calibration rows count; reserved edge rows may be excluded.
            from monitoring_v3.classification import _calibration_gap
            effective=_calibration_gap(f,ix.isin(tr),ix.isin(cal),24)
            self.assertTrue(y.loc[effective].eq(1).any())
            self.assertGreater(report['training_overlap_excluded'],0)
    def test_smoothing_never_bridges_a_missing_hour_in_sparse_calibration(self):
        cfg=load();ix=pd.DatetimeIndex(['2024-01-01 00:00','2024-01-01 01:00','2024-01-01 03:00'])
        errors=pd.DataFrame({'sensor':[10.,10.,0.]},index=ix)
        views,_=score_views(errors,pd.Series({'sensor':1.}),cfg)
        self.assertEqual(views['mean::instant'].smoothed.iloc[-1],0.)

    def test_current_pairs_require_the_same_equipment_line(self):
        cfg=load(overrides={'first_test':'2024-02-01','min_sensor_rows':2})
        ix=pd.date_range('2024-01-01',periods=10,freq='h')
        names=['ток двигателя мв-#а','ток двигателя мв-#б','ток двигателя псу-#а','ток двигателя псу-#б']
        frame=pd.DataFrame(40.,index=ix,columns=names);roles=dict.fromkeys(names,'condition')
        spec=fit_evidence(frame,roles,'K',cfg)
        self.assertEqual(len(spec['pairs']),2)
        frame.iloc[-1]=[40,0,0,40]
        e=evidence_features(frame,spec)
        self.assertEqual(e.coal_evidence.iloc[-1],0)
        self.assertEqual(e.mill_without_feeder.iloc[-1],1)
        self.assertEqual(e.feeder_without_mill.iloc[-1],1)

    def test_duplicate_numeric_states_are_never_averaged_and_missing_last_state_is_preserved(self):
        raw=pd.DataFrame({'TimeValue':['2024-01-01','2024-01-01','2024-01-01 01:00'],
                          'load':['10,0','20,0','30,0'],'state':[0.,1.,np.nan]})
        raw['TimeValue']=pd.to_datetime(raw.TimeValue,format='mixed')
        frame,audit=normalize_grid(raw,numeric_columns=['load'],state_columns=['state'])
        self.assertEqual(frame.load.iloc[0],15.);self.assertEqual(frame.state.iloc[0],1.)
        self.assertTrue(pd.isna(frame.state.iloc[-1]));self.assertEqual(audit['duplicate_timestamps'],1)

    def test_risk_keeps_updating_after_last_emergency_and_holdout_never_trains(self):
        cfg=load(overrides={'cb_initial_trees':4,'cb_update_trees':2,'cb_depth':2,'threads':1})
        ix=pd.date_range('2020-01-01',periods=1500,freq='6h')
        y=pd.Series(0.,index=ix);g=pd.Series([f'bg::{t.to_period("W")}' for t in ix],index=ix)
        for i,pos in enumerate((120,400,650)):
            y.iloc[pos:pos+4]=1.;g.iloc[pos:pos+4]=f'event::{i}'
        x=pd.DataFrame({'unit':'K','value':np.sin(np.arange(len(ix))/11)},index=ix)
        with tempfile.TemporaryDirectory() as tmp:
            train_head(x.iloc[:1100],y.iloc[:1100],g.iloc[:1100],cfg,tmp,horizon_hours=24,task='risk')
            old=joblib.load(Path(tmp)/'interpreter.joblib')
            result=train_head(x,y,g,cfg,tmp,old,horizon_hours=24,task='risk')
            new=joblib.load(Path(tmp)/'interpreter.joblib')
            self.assertIn(result['status'],('trained','trained_uncalibrated'))
            self.assertGreater(pd.Timestamp(new['trained_through']),pd.Timestamp(old['trained_through']))
            self.assertEqual(new['model'].tree_count_,old['model'].tree_count_+cfg['cb_update_trees'])
            self.assertFalse(set(new['trained_group_ids'])&set(new['calibration_group_ids']))
            self.assertTrue(set(old['calibration_group_ids'])<=set(new['calibration_group_ids']))
            self.assertTrue((Path(tmp)/'feature_importance.csv').exists())
            self.assertTrue((Path(tmp)/'feature_importance_delta.csv').exists())
            limited={**cfg,'cb_max_trees':new['model'].tree_count_}
            extra=pd.date_range(ix[-1]+pd.Timedelta('6h'),periods=100,freq='6h')
            xx=pd.concat([x,pd.DataFrame({'unit':'K','value':0.},index=extra)])
            yy=pd.concat([y,pd.Series(0.,index=extra)])
            gg=pd.concat([g,pd.Series([f'bg::{t.to_period("W")}' for t in extra],index=extra)])
            result=train_head(xx,yy,gg,limited,tmp,new,horizon_hours=24,task='risk')
            capped=joblib.load(Path(tmp)/'interpreter.joblib')
            self.assertEqual(result['fit_status'],'tree_capacity_reached_previous_retained')
            self.assertEqual(capped['trained_through'],new['trained_through'])
            self.assertEqual(capped['ingested'],new['ingested'])

    def test_failed_new_head_cannot_silently_leave_an_old_head_in_export(self):
        from interpretation_training import attach_heads
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model').mkdir();(root/'model/risk_24h.joblib').write_bytes(b'previous')
            attach_heads(root,{'risk_24h':{'status':'insufficient_training_events'}},'2024-01-01')
            self.assertFalse((root/'model/risk_24h.joblib').exists())
            self.assertEqual(next((root/'interpretation_export_history').glob('*')).read_bytes(),b'previous')

    def test_initial_history_entirely_in_scheduled_calibration_weeks_is_rebalanced(self):
        cfg=load(overrides={'validate_candidate':False})
        times=pd.date_range('2020-01-01',periods=24000,freq='h')
        block=times.as_unit('ns').asi8//pd.Timedelta('168h').value
        times=times[(block%3)==2][:1200]
        ix=pd.MultiIndex.from_product([times,['K']],names=['time','unit'])
        f=pd.DataFrame({'load':100.},index=ix)
        train,_,cal,_,_,_=partition_history(f,cfg)
        self.assertGreaterEqual(train.sum(),500);self.assertGreaterEqual(cal.sum(),168)

    def test_rare_change_is_a_separate_event_not_a_month_long_average_alarm(self):
        cfg=load();ix=pd.date_range('2024-01-01',periods=80,freq='h')
        errors=pd.DataFrame({'continuous':0.,'rare':np.r_[np.zeros(40),np.ones(40)*10000]},index=ix)
        changes=pd.DataFrame({'rare':np.r_[np.zeros(40),1.,np.zeros(39)]},index=ix)
        views,_=score_views(errors,pd.Series(1.,index=errors.columns),cfg,event_sensors=['rare'],event_changes=changes)
        self.assertTrue(views['mean::instant'].score.eq(0).all())
        stream,_=alarm_stream(views['rare_changes::event'].score/.5,view_policy(cfg,'rare_changes::event'))
        self.assertEqual(stream.notification.sum(),1);self.assertTrue(stream.notification.iloc[40])

    def test_repair_then_reserve_is_not_a_conflict_and_future_documents_do_not_leak(self):
        cfg=load(overrides={'startup_hours':0});ix=pd.date_range('2024-01-01',periods=200,freq='h')
        steam=pd.Series(100.,index=ix);steam.iloc[80:150]=0
        state=pd.Series('Работа',index=ix);state.iloc[80:100]='Аварийный ремонт';state.iloc[100:150]='Резерв'
        events=registry(pd.DataFrame({'Состояние':state}),regimes(steam,25,cfg),cfg,'K')
        self.assertEqual(len(events),1);self.assertEqual(events.iloc[0].type,'Авария')
        row=events.iloc[0]
        self.assertEqual(event_type_at(row,ix[90])[0],'Авария')
        self.assertEqual(event_type_at(row,ix[79])[0],'Неизвестно')

    def test_disabled_validation_never_reserves_or_requires_validation(self):
        cfg=load(overrides={'validate_candidate':False,'min_train_rows':500})
        ix=pd.MultiIndex.from_product([pd.date_range('2023-01-01',periods=3000,freq='h'),['К-5']],names=['time','unit'])
        x=pd.DataFrame({'load':100.},index=ix)
        tr,val,cal,rc,rv,audit=partition_history(x,cfg)
        self.assertEqual(val.sum(),0);self.assertFalse(rv)
        self.assertGreaterEqual(tr.sum(),500);self.assertGreaterEqual(cal.sum(),168)
        self.assertFalse(set(ix[tr])&rc)

    def test_calibration_is_never_admitted_to_later_training(self):
        cfg=load(overrides={'validate_candidate':False})
        ix=pd.MultiIndex.from_product([pd.date_range('2023-01-01',periods=5000,freq='h'),['К-5']],names=['time','unit'])
        f=pd.DataFrame({'load':100.},index=ix)
        tr,_,_,cal,val,_=partition_history(f.iloc[:3000],cfg)
        old=dict(ingested=set(f.iloc[:3000].index[tr]),reserved_calibration=cal,reserved_validation=val)
        train,_,_,newcal,_,_=partition_history(f,cfg,old)
        self.assertFalse(set(f.index[train])&cal)
        self.assertFalse(set(old['ingested'])&newcal)

    def test_old_year_can_start_after_recent_month_of_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['validate_candidate']=False
            cfg['history_policy']['bootstrap']=False
            contexts,_=load_sources(cfg);c=contexts['К-1'];end=c['frame'].index[-1]+pd.Timedelta('60D')
            state,audit=update_at(contexts,next(experiments(cfg)),end,cfg)
            self.assertIsNotNone(state);self.assertEqual(audit['status'],'initial')
            self.assertGreater(audit['train_rows'],cfg['min_train_rows'])
            self.assertTrue(np.isfinite(state['thresholds']['К-1']['mean::instant']['1.0']['threshold']))

    def test_currents_do_not_cancel_oil_operation_or_prove_operation_alone(self):
        cfg=load(overrides={'startup_hours':0,'run_thresholds':{'K':25},'min_sensor_rows':2})
        ix=pd.date_range('2024-01-01',periods=12,freq='h')
        f=pd.DataFrame({'load':100.,'ток двигателя мв-а':0.,'ток двигателя псу-а':0.,
                        'расход мазута':5.,'target':20.,'Состояние':'Работа'},index=ix)
        roles={'load':'load','ток двигателя мв-а':'condition','ток двигателя псу-а':'condition',
               'расход мазута':'condition','target':'target'}
        c=prepare_unit(f,roles,'K',cfg,{})
        self.assertTrue(c['eligible'].all());self.assertTrue(c['regime']['evidence::fuel_mode'].eq('oil').all())
        f['load']=np.nan;f['ток двигателя мв-а']=40.;f['ток двигателя псу-а']=15.
        other=prepare_unit(f,roles,'K',cfg,{})
        self.assertFalse(other['eligible'].any());self.assertTrue(other['regime']['mode'].eq('неизвестно').all())

    def test_missing_load_is_not_a_physical_stop_and_unknown_stop_is_registered(self):
        cfg=load(overrides={'startup_hours':0,'load_hold_hours':0})
        ix=pd.date_range('2024-01-01',periods=12,freq='h')
        x=pd.Series([100,100,np.nan,np.nan,100,100,0,0,0,100,100,100],index=ix)
        m=regimes(x,25,cfg);events=registry(pd.DataFrame({'Состояние':'Работа'},index=ix),m,cfg,'K')
        self.assertEqual(len(events),1);self.assertEqual(events.iloc[0].stop,ix[6])
        self.assertEqual(events.iloc[0].type,'Неизвестно')
        for n in range(1,len(ix)+1):pd.testing.assert_frame_equal(regimes(x.iloc[:n],25,cfg),m.iloc[:n])
        default=load(overrides={'startup_hours':0})
        one_low=pd.Series([100,100,0,np.nan,np.nan,100],index=ix[:6])
        self.assertFalse(regimes(one_low,25,default).stop_confirmed.any())

    def test_unknown_stop_and_unobserved_future_do_not_label_negative(self):
        ix=pd.date_range('2024-01-01',periods=100,freq='h')
        events=pd.DataFrame([dict(stop=ix[40],type='Неизвестно',label_available_at=ix[41],event_id='stop')])
        observed=pd.Series(True,index=ix);observed.iloc[65]=False
        y,g,a=risk_labels(ix,events,24,ix[-1],ix[-1],observed=observed,return_audit=True)
        self.assertTrue(y.iloc[20:40].isna().all());self.assertTrue(pd.isna(y.iloc[50]))
        self.assertEqual(y.iloc[5],0);self.assertGreater(a['unobserved_future'],0)

    def test_already_observed_failure_does_not_wait_for_the_rest_of_the_horizon(self):
        ix=pd.date_range('2024-01-01',periods=13,freq='h')
        event=pd.DataFrame([dict(stop=ix[10],type='Авария',label_available_at=ix[11],
                                stop_detected_at=ix[11],event_id='event')])
        y,_=risk_labels(ix[:1],event,24,ix[-1],ix[-1],observed=pd.Series(True,index=ix))
        self.assertEqual(y.iloc[0],1.)
        y,_=risk_labels(ix[:1],event,24,ix[9],ix[9],observed=pd.Series(True,index=ix[:10]))
        self.assertTrue(pd.isna(y.iloc[0]))

    def test_three_events_train_model_even_when_test_has_no_failures(self):
        cfg=load(overrides={'cb_initial_trees':4,'cb_update_trees':2,'cb_depth':2,'threads':1})
        ix=pd.date_range('2020-01-01',periods=1200,freq='6h')
        y=pd.Series(0.,index=ix);g=pd.Series([f'bg::{t.to_period("W")}' for t in ix],index=ix)
        for i,pos in enumerate((120,400,720)):
            y.iloc[pos:pos+4]=1.;g.iloc[pos:pos+4]=f'event::{i}'
        rng=np.random.default_rng(73)
        x=pd.DataFrame({'unit':'K','evidence':y+rng.normal(0,.1,len(y))},index=ix)
        with tempfile.TemporaryDirectory() as tmp:
            result=train_head(x,y,g,cfg,tmp,horizon_hours=24,task='risk')
            self.assertEqual(result['status'],'trained_uncalibrated')
            self.assertEqual(result['test_status'],'single_class_test')
            head=joblib.load(Path(tmp)/'interpreter.joblib')
            p,_=probabilities(head,x.iloc[-5:],require_calibration=True)
            self.assertTrue(np.isnan(p).all());self.assertTrue(result['forward_checks'])

    def test_full_runtime_matches_month_and_bounded_prediction_preserves_ages(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['validate_candidate']=False;run(cfg)
            folder=next(Path(cfg['output']).glob('experiments/*/К-1/model_history/2024-03'))
            model=Model(folder);context=load_sources(cfg)[0]['К-1']
            full=model.score(context['frame'])
            start=pd.Timestamp('2024-03-20');bounded=model.score(context['frame'],period_start=start)
            pd.testing.assert_frame_equal(full.hourly.loc[start:],bounded.hourly.loc[start:])
            stored=pd.read_parquet(folder.parent.parent/'mean__instant__1.0.parquet')
            common=stored.index.intersection(full.hourly.index)
            common=common[(common>=pd.Timestamp('2024-03-01'))&(common<pd.Timestamp('2024-04-01'))]
            np.testing.assert_allclose(stored.loc[common,'ratio'],full.hourly.loc[common,('mean::instant','отношение')],equal_nan=True)
            periods=pd.read_csv(Path(cfg['output'])/'comparison_periods.csv')
            self.assertEqual(set(periods.policy),{'adaptive','frozen_initial'})
            self.assertTrue((folder.parent.parent/'frozen/mean__instant__1.0.parquet').exists())
            self.assertTrue((folder/'feature_importance.csv').exists())
            from notebook_reports import show_detail
            with self.assertRaisesRegex(ValueError,'Нет сохранённой важности'):
                show_detail(cfg,{'importance_month':'2099-01'})
            from sklearn.dummy import DummyClassifier
            from monitoring_v3.preparation import FrozenEncoder
            example=pd.DataFrame({'unit':['К-1','К-1']})
            head=dict(model=DummyClassifier().fit(example,['mode','other']),encoder=FrozenEncoder.fit(example),
                calibrator=None,threshold=.7,available_at=str(start))
            joblib.dump(head,folder/'interpreter.joblib')
            gated=model.score(context['frame']).interpretation
            self.assertIn('probability_status',gated);self.assertTrue(gated.loc[:start,'label'].eq('unknown').all())

    def test_group_classifier_collects_independent_events_from_both_boilers(self):
        from interpretation_training import train_risk_group
        cfg=load(overrides={'cb_initial_trees':4,'cb_update_trees':2,'cb_depth':2,'threads':1})
        raw_ix=pd.date_range('2020-01-01',periods=260*24,freq='h');ix=raw_ix[::6]
        rng=np.random.default_rng(63)
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output']=tmp;items=[];contexts={}
            for i,unit in enumerate(('K1','K2')):
                events=pd.DataFrame([dict(stop=raw_ix[(day+i*10)*24],start=raw_ix[(day+i*10)*24],type='Авария',
                    label_available_at=raw_ix[(day+i*10)*24],event_id=unit+str(day)) for day in (30,90,150,210)])
                frame=pd.DataFrame({'load':100.},index=raw_ix)
                context=dict(frame=frame,load='load',unit=unit,eligible=pd.Series(True,index=raw_ix),events=events)
                labels,_=risk_labels(ix,events,24,raw_ix[-1],raw_ix[-1])
                features=pd.DataFrame(dict(unit=unit,mean_abs=labels.fillna(0)+rng.normal(0,.1,len(ix)),
                    features_valid=True,episode_active=1.,episode_age_hours=0.,episode_sample_due=True,
                    episode_group=[str(t.to_period('W')) for t in ix]),index=ix)
                directory=Path(tmp)/unit;directory.mkdir()
                features.to_parquet(directory/'interpretation_features.parquet')
                items.append(dict(unit=unit,directory=directory));contexts[unit]=context
            result=train_risk_group(cfg,items,contexts,raw_ix[-1])
            self.assertIn(result['risk_24h']['status'],('trained','trained_uncalibrated'))
            self.assertEqual(result['risk_24h']['units'],['K1','K2'])
            self.assertEqual(len(result['risk_24h']['label_audit']),2)
            package=joblib.load(result['risk_24h']['shared_head'])
            self.assertEqual(package['training_units'],['K1','K2'])


if __name__=='__main__':unittest.main()
