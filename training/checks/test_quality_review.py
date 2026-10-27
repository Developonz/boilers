"""Independent adversarial cases found during the second end-to-end audit."""
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import io,json,sys,tempfile,unittest
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from monitoring_v3.config import load
from monitoring_v3.partitioning import partition_history
from monitoring_v3.alarm_scores import score_views
from monitoring_v3.preparation import regimes,FrozenEncoder
from monitoring_v3.input_grid import normalize_grid
from monitoring_v3.regime_evidence import fit_evidence,evidence_features
from monitoring_v3.data import load_sources,prepare_unit
from monitoring_v3.pipeline import run,transfer_preparation
from monitoring_v3.classification import train_classifier,CONTRACT
from monitoring_v3.runtime import Model
from test_runtime_performance import fixture


class QualityReviewTests(unittest.TestCase):
    def small_classifier(self):
        return load(overrides={'cb_initial_trees':4,'cb_update_trees':2,'cb_depth':2,'threads':1})

    def test_pooled_timestamp_cannot_split_when_row_limits_are_odd(self):
        cfg=load(overrides=dict(min_train_rows=11,calibration_rows=5,min_sensor_rows=1,
            validate_rows=5,calibration_limit=7,validate_candidate=False))
        times=pd.date_range('2024-01-01',periods=70,freq='h')
        ix=pd.MultiIndex.from_product([times,['A','B','C']],names=['time','unit'])
        frame=pd.DataFrame({'load':100.},index=ix)
        train,val,cal,rc,rv,_=partition_history(frame,cfg)
        for mask in (train,val,cal):
            self.assertTrue(pd.Series(mask,index=ix).groupby(level='time').nunique().le(1).all())
        self.assertGreaterEqual(cal.sum(),5);self.assertGreaterEqual(train.sum(),11)
        previous=dict(ingested=set(ix[train]),reserved_calibration=rc,reserved_validation=rv)
        extra=pd.MultiIndex.from_product([pd.date_range(times[-1]+pd.Timedelta('1h'),periods=70,freq='h'),['A','B','C']],names=ix.names)
        allf=pd.concat([frame,pd.DataFrame({'load':110.},index=extra)])
        train,val,cal,_,_,_=partition_history(allf,{**cfg,'validate_candidate':True},previous)
        for mask in (train,val,cal):self.assertTrue(pd.Series(mask,index=allf.index).groupby(level='time').nunique().le(1).all())

    def test_group_coverage_and_attribution_only_include_its_own_sensors(self):
        ix=pd.date_range('2024-01-01',periods=3,freq='h')
        errors=pd.DataFrame({'bearing':2.,'other':20.,'missing1':np.nan,'missing2':np.nan},index=ix)
        views,matrix=score_views(errors,pd.Series(1.,index=errors.columns),load(),{'bearing':'bearings','other':'other'})
        self.assertTrue(views['mean::instant'].score.isna().all())
        self.assertTrue(views['group:bearings::instant'].score.eq(2.).all())
        self.assertTrue(views['group:bearings::instant'].coverage.eq(1.).all())
        self.assertEqual(list(matrix['group:bearings::instant']),['bearing'])

    def test_valid_mixed_dates_do_not_disappear(self):
        raw=pd.DataFrame({'TimeValue':['2024-01-01','2024-01-01 01:00:00','2024-01-01T02:00:00'], 'load':[1,2,3]})
        grid,audit=normalize_grid(raw,numeric_columns=['load'])
        self.assertEqual(len(grid),3);self.assertEqual(audit['invalid_timestamps'],0)

    def test_single_positive_spike_does_not_establish_a_motor_operating_threshold(self):
        cfg=load(overrides={'first_test':'2024-03-01','min_sensor_rows':20})
        ix=pd.date_range('2024-01-01',periods=100,freq='h');name='ток двигателя мв-#а'
        raw=pd.DataFrame({name:np.r_[40.,np.zeros(99)]},index=ix)
        spec=fit_evidence(raw,{name:'condition'},'K',cfg)
        self.assertIsNone(spec['channels'][name]['threshold'])
        cfg['regime_policy']['current_thresholds']={'typo':10.}
        with self.assertRaisesRegex(ValueError,'Unknown current'):fit_evidence(raw,{name:'condition'},'K',cfg)

    def test_missing_donor_currents_reduce_coverage(self):
        ix=pd.date_range('2024-01-01',periods=3,freq='h')
        spec=dict(channels={'a':dict(group='mill',threshold=5.,typical=40.),'b':dict(group='mill',threshold=5.,typical=40.)},oil_channels=[])
        evidence=evidence_features(pd.DataFrame({'a':40.},index=ix),spec)
        self.assertTrue(evidence.mill_coverage.eq(.5).all())

    def test_unknown_load_does_not_preserve_a_confident_run_age(self):
        cfg=load(overrides={'startup_hours':2,'load_hold_hours':0})
        ix=pd.date_range('2024-01-01',periods=12,freq='h')
        x=pd.Series([100]*4+[np.nan]*2+[100]*6,index=ix)
        r=regimes(x,25,cfg)
        self.assertFalse(r.stop_confirmed.any())
        self.assertTrue(r['mode'].iloc[6:8].eq('неизвестно').all())
        self.assertEqual(r['mode'].iloc[8],'работа')
        self.assertTrue(r.since_start_hours.iloc[6:].isna().all())
        self.assertFalse(r.run_age_known.iloc[6:].any())

    def test_late_reviewed_labels_cannot_train_a_past_evaluation_model(self):
        cfg=self.small_classifier();ix=pd.date_range('2020-01-01',periods=1200,freq='6h')
        y=pd.Series(0.,index=ix);g=pd.Series([f'week::{t.to_period("W")}' for t in ix],index=ix)
        for i,pos in enumerate((120,400,720)):y.iloc[pos:pos+4]=1.;g.iloc[pos:pos+4]=f'failure::{i}'
        x=pd.DataFrame({'unit':'K','signal':np.sin(np.arange(len(ix)))},index=ix)
        available=pd.Series(ix+pd.Timedelta('24h'),index=ix);available.loc[y.eq(1)]=ix[-1]
        with tempfile.TemporaryDirectory() as tmp:
            report=train_classifier(x,y,g,cfg,tmp,horizon_hours=24,task='risk',label_available_at=available)
            self.assertIn(report['status'],('trained','trained_uncalibrated'))
            self.assertEqual(report['test_status'],'training_requires_two_classes')
            self.assertGreater(report['test']['late_labels_excluded'],0)
            self.assertTrue(all(f['status']=='insufficient_labels_known_at_boundary' for f in report['forward_checks']))

    def test_group_revision_starts_a_new_head_instead_of_reusing_old_group_assignments(self):
        cfg=self.small_classifier();ix=pd.date_range('2020-01-01',periods=500,freq='12h')
        x=pd.DataFrame({'signal':np.sin(np.arange(len(ix))),'unit':'K'},index=ix)
        y=pd.Series(np.where(np.arange(len(ix))%2,'cause_a','cause_b'),index=ix)
        g=pd.Series([str(t.to_period('W')) for t in ix],index=ix)
        with tempfile.TemporaryDirectory() as tmp:
            train_classifier(x,y,g,cfg,tmp);old=joblib.load(Path(tmp)/'interpreter.joblib')
            result=train_classifier(x,y,g+'::revised',cfg,tmp,old)
            new=joblib.load(Path(tmp)/'interpreter.joblib')
            self.assertEqual(result['migration'],'historical_features_labels_or_groups_revised')
            self.assertEqual(new['model'].tree_count_,cfg['cb_initial_trees'])

    def test_unusable_auxiliary_evaluation_does_not_discard_a_working_classifier(self):
        cfg=self.small_classifier();ix=pd.date_range('2020-01-01',periods=600,freq='12h')
        x=pd.DataFrame({'signal':np.r_[np.zeros(450),np.arange(150)],'unit':'K'},index=ix)
        y=pd.Series(np.where(np.arange(len(ix))%2,'a','b'),index=ix)
        g=pd.Series([str(t.to_period('W')) for t in ix],index=ix)
        with tempfile.TemporaryDirectory() as tmp:
            report=train_classifier(x,y,g,cfg,tmp)
            self.assertIn(report['status'],('trained','trained_uncalibrated'))
            self.assertEqual(report['test_status'],'no_variable_inputs')
            self.assertTrue((Path(tmp)/'interpreter.joblib').exists())

    def test_missing_first_model_month_is_counted_as_unscored(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg.update(min_train_rows=1250,validate_candidate=False)
            cfg['history_policy']['bootstrap']=False
            result=run(cfg)
            directory=next(Path(cfg['output']).glob('experiments/*/К-1'))
            data=pd.read_parquet(directory/'mean__instant__1.0.parquet')
            self.assertEqual(data.index[0],pd.Timestamp('2024-03-01'))
            self.assertTrue(data.loc['2024-03','score'].isna().all())
            self.assertTrue(result['summary'].unscored_hours.ge(31*24).all())
            self.assertTrue(result['summary'].working_unscored_hours.ge(31*24).all())

    def test_transfer_preparation_uses_donor_but_keeps_recipient_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(tmp);contexts,_=load_sources(cfg);donor=contexts['К-1']
            recipient={**donor,'unit':'К-2','threshold':90.}
            selected=transfer_preparation({'К-1':donor,'К-2':recipient},dict(scenario='transfer',sources=['К-1'],recipients=['К-2']),cfg)
            self.assertEqual(selected['К-2']['threshold'],donor['threshold'])
            self.assertEqual(selected['К-2']['threshold_source'],'donor:К-1')
            self.assertIs(selected['К-2']['events'],recipient['events'])
            self.assertEqual(recipient['threshold'],90.)

    def test_runtime_does_not_require_manual_events_from_the_future(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['validate_candidate']=False;run(cfg)
            path=next(Path(cfg['output']).glob('experiments/*/К-1/model'))
            model=Model(path);context=load_sources(cfg)[0]['К-1']
            model.config['event_overrides']=[dict(unit='К-1',stop='2099-01-01',type='Авария',label_available_at='2099-01-02')]
            result=model.score(context['frame'])
            self.assertGreater(len(result.hourly),0)

    def test_head_save_survives_interrupted_auxiliary_evaluation(self):
        cfg=self.small_classifier();ix=pd.date_range('2020-01-01',periods=600,freq='12h')
        x=pd.DataFrame({'signal':np.sin(np.arange(len(ix))),'unit':'K'},index=ix)
        y=pd.Series(np.where(np.arange(len(ix))%2,'a','b'),index=ix)
        g=pd.Series([str(t.to_period('W')) for t in ix],index=ix)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('monitoring_v3.classification._temporal_check',side_effect=KeyboardInterrupt('auxiliary check interrupted')):
                with self.assertRaises(KeyboardInterrupt):train_classifier(x,y,g,cfg,tmp)
            old=joblib.load(Path(tmp)/'interpreter.joblib')
            train_classifier(x,y,g,cfg,tmp,old)
            new=joblib.load(Path(tmp)/'interpreter.joblib')
            self.assertEqual(new['model'].tree_count_,old['model'].tree_count_)

    def test_custom_horizon_and_sampling_policy_attach_atomically(self):
        from sklearn.dummy import DummyClassifier
        from interpretation_training import attach_heads
        import hashlib
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['validate_candidate']=False;run(cfg)
            folder=next(Path(cfg['output']).glob('experiments/*/К-1/model'));directory=folder.parent
            sample=pd.DataFrame({'unit':['К-1','К-1'],'mean_abs':[0.,1.]})
            package=dict(contract=CONTRACT,model=DummyClassifier().fit(sample,['0.0','1.0']),encoder=FrozenEncoder.fit(sample),
                calibrator=None,threshold=.7,task='risk',horizon_hours=48,risk_policy={'samples':'continuous'},
                report={'probability_status':'uncalibrated_score_only'})
            source=Path(tmp)/'head.joblib';joblib.dump(package,source)
            statuses={'risk_48h':{'status':'trained_uncalibrated','shared_head':str(source)}}
            attach_heads(directory,statuses,'2024-04-01')
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            for name,digest in manifest.items():self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest)
            model=Model(folder);frame=load_sources(cfg)[0]['К-1']['frame'];result=model.score(frame)
            self.assertIn('risk_48h_score',result.interpretation)
            inactive=result.hourly['mean::instant']['тревога'].eq(False)&(result.interpretation.index>=pd.Timestamp('2024-04-01'))
            self.assertTrue(result.interpretation.loc[inactive,'risk_48h_score'].notna().any())
            before={p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
            with patch('joblib.dump',side_effect=RuntimeError('disk write failed')):
                with self.assertRaises(RuntimeError):attach_heads(directory,statuses,'2024-04-02')
            after={p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
            self.assertEqual(before,after)

    def test_constant_new_targets_can_still_correct_an_existing_regressor(self):
        from monitoring_v3.models import NormalModel
        from test_target_training import sample
        f,y,cfg=sample();model=NormalModel('CatBoost',cfg).fit_initial(f,y)
        old=model.predictors['temperature'][1];before=np.asarray(old.predict(model.encoder.native(f)))
        target=pd.DataFrame(100.,index=f.index,columns=y.columns)
        model._fit_trees(f,target,initial=False)
        self.assertGreater(model.predictors['temperature'][1].tree_count_,old.tree_count_)
        np.testing.assert_array_equal(before,old.predict(model.encoder.native(f)))
        self.assertLess((model.expected(f,target)[0].temperature-100).abs().median(),np.median(abs(before[:,1]-100)))

    def test_instrument_floor_defaults_are_applied_and_invalid_rules_are_rejected(self):
        from monitoring_v3.models import NormalModel
        from monitoring_v3.data import read_rules
        from test_target_training import sample
        f,y,cfg=sample()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rules.json';cfg['sensor_rules']=str(path)
            path.write_text(json.dumps({'defaults':{'resolution':3.},'sensors':{'pressure':{'scale_floor':5.}}}),encoding='utf-8')
            model=NormalModel('CatBoost',cfg).fit_initial(f,y)
            self.assertGreaterEqual(model.floor[0],3.);self.assertGreaterEqual(model.floor[1],5.)
            for bad in ({'defaults':{'mask_frozen':'False'}},{'defaults':{'min':0},'sensors':{'pressure':{'max':-1}}}):
                path.write_text(json.dumps(bad),encoding='utf-8')
                with self.assertRaises(ValueError):read_rules(path)

    def test_survival_is_censored_at_a_measurement_gap(self):
        from monitoring_v3.experiments import survival_landmarks
        ix=pd.date_range('2020-01-01',periods=100,freq='h')
        f=pd.DataFrame({'x':1.},index=ix[:1]);observed=pd.Series(True,index=ix);observed.iloc[20]=False
        events=pd.DataFrame([dict(stop=ix[50],type='Авария',label_available_at=ix[51],event_id='event')])
        label=survival_landmarks(f,events,ix[-1],ix[-1],observed)
        self.assertFalse(label.event.iloc[0]);self.assertEqual(label.duration_hours.iloc[0],20.)

    def test_retracted_normal_is_removed_by_a_visible_rebuild(self):
        from monitoring_v3.pipeline import update_at,experiments,score_context
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg['validate_candidate']=False
            contexts,_=load_sources(cfg);spec=next(experiments(cfg))
            state,_=update_at(contexts,spec,pd.Timestamp('2024-03-01'),cfg)
            removed=sorted(state['ingested'])[:20]
            contexts['К-1']['frame'].loc[[key[0] for key in removed],'Состояние']='Плановый ремонт'
            rebuilt,audit=update_at(contexts,spec,pd.Timestamp('2024-04-01'),cfg,state)
            self.assertEqual(audit['status'],'rebuilt_after_normal_retraction')
            self.assertEqual(audit['retracted_normal_rows'],20)
            self.assertFalse(set(removed)&rebuilt['ingested'])
            contexts['К-1']['frame']['Состояние']='Плановый ремонт'
            unsafe,audit=update_at(contexts,spec,pd.Timestamp('2024-04-01'),cfg,rebuilt)
            self.assertTrue(unsafe['unsafe_normal']);self.assertEqual(audit['status'],'unsafe_retracted_normal_history')
            views,_,_,_=score_context(unsafe,contexts['К-1'],spec,cfg)
            self.assertTrue(views['mean::instant'].score.isna().all())

    def test_review_history_respects_its_availability_dates(self):
        from monitoring_v3.data import registry
        from monitoring_v3.event_registry import event_type_at
        ix=pd.date_range('2024-01-01',periods=200,freq='h');load_value=pd.Series(100.,index=ix);load_value.iloc[30:80]=0.
        cfg=load(overrides={'startup_hours':0,'event_overrides':[
            dict(unit='K',stop=str(ix[30]),type='Авария',label_available_at=str(ix[100])),
            dict(unit='K',stop=str(ix[30]),type='Резерв',label_available_at=str(ix[50]))]})
        events=registry(pd.DataFrame({'Состояние':'Работа'},index=ix),regimes(load_value,25,cfg),cfg,'K')
        event=events.iloc[0]
        self.assertEqual(event_type_at(event,ix[40])[0],'Неизвестно')
        self.assertEqual(event_type_at(event,ix[60])[0],'Резерв')
        self.assertEqual(event_type_at(event,ix[120])[0],'Авария')

    def test_invalid_numeric_configuration_is_rejected_before_training(self):
        for bad in ({'startup_hours':float('nan')},{'alarm_budgets':[float('inf')]},
            {'run_thresholds':{'K':True}},{'regime_policy':{'unknown_recovery_hours':float('nan')}},
            {'risk_policy':{'sample_hours':.5},'step':'1h'}):
            with self.assertRaises(ValueError):load(overrides=bad)

    def test_short_repair_document_cannot_shorten_the_observed_physical_stop(self):
        from monitoring_v3.data import registry
        ix=pd.date_range('2024-01-01',periods=120,freq='h');cfg=load(overrides={'startup_hours':0})
        steam=pd.Series(100.,index=ix);steam.iloc[30:80]=0
        state=pd.Series('Работа',index=ix);state.iloc[30:40]='Аварийный ремонт'
        events=registry(pd.DataFrame({'Состояние':state}),regimes(steam,25,cfg),cfg,'K')
        self.assertEqual(events.iloc[0].end,ix[80])

    def test_group_training_uses_the_latest_available_unit_without_truncating_others(self):
        from notebook_reports import run_interpretation
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=self.small_classifier();cfg['output']=tmp;parent=Path(tmp)/'experiment';parent.mkdir()
            (parent/'experiment.json').write_text(json.dumps(dict(group='G',scenario='pooled',sources=['A','B'])),encoding='utf-8')
            items=[dict(experiment='experiment',unit=u,directory=parent/u,method='CatBoost',variant='conditions',scenario='pooled') for u in ('A','B')]
            contexts={u:dict(frame=pd.DataFrame(index=pd.date_range('2024-01-01',periods=length,freq='h'))) for u,length in [('A',100),('B',200)]}
            with patch('notebook_reports.experiment_directories',return_value=items),patch('notebook_reports.show_table'),patch(
                'interpretation_training.train_risk_group',return_value={'risk_24h':{'status':'no_observed_labelled_examples'}}) as group,patch(
                'interpretation_training.train',return_value={'cause':{'status':'requires_reviewed_episode_labels'}}):
                run_interpretation(cfg,contexts,attach=False)
            self.assertEqual(group.call_args.args[3],contexts['B']['frame'].index[-1])

    def test_moving_results_does_not_change_group_head_identity(self):
        from interpretation_training import train_risk_group
        import shutil
        cfg=self.small_classifier();cfg['risk_horizons']=[24]
        raw_ix=pd.date_range('2020-01-01',periods=200*24,freq='h');ix=raw_ix[::6]
        events=pd.DataFrame([dict(stop=raw_ix[day*24],type='Авария',label_available_at=raw_ix[day*24],event_id=str(day)) for day in (30,70,110,150)])
        contexts={'K':dict(unit='K',load='load',frame=pd.DataFrame({'load':100.},index=raw_ix),events=events,eligible=pd.Series(True,index=raw_ix))}
        features=pd.DataFrame(dict(unit='K',signal=np.sin(np.arange(len(ix))),features_valid=True,episode_active=1.,
            episode_age_hours=0.,episode_sample_due=True,episode_group=[str(t.to_period('W')) for t in ix]),index=ix)
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'original';folder=original/'experiment'/'K';folder.mkdir(parents=True)
            features.to_parquet(folder/'interpretation_features.parquet');cfg['output']=str(original)
            first=train_risk_group(cfg,[dict(experiment='E',unit='K',directory=folder)],contexts,raw_ix[-1])
            moved=Path(tmp)/'moved';shutil.copytree(original,moved);cfg['output']=str(moved)
            second=train_risk_group(cfg,[dict(experiment='E',unit='K',directory=moved/'experiment'/'K')],contexts,raw_ix[-1])
            self.assertEqual(second['risk_24h']['fit_status'],'no_new_labels')
            self.assertEqual(Path(first['risk_24h']['shared_head']).parent.parent.name,Path(second['risk_24h']['shared_head']).parent.parent.name)

    def test_relocating_unchanged_data_does_not_refit_checkpoint_models(self):
        import shutil
        from monitoring_v3.models import NormalModel
        with tempfile.TemporaryDirectory() as destination,redirect_stdout(io.StringIO()):
            with tempfile.TemporaryDirectory() as original:
                cfg=fixture(original);run(cfg);shutil.copytree(original,destination,dirs_exist_ok=True)
            root=Path(destination)
            cfg.update(data_root=str(root/'source'),sensor_file=str(root/'roles.xlsx'),sensor_rules=str(root/'rules.json'),output=str(root/'out'))
            with patch.object(NormalModel,'fit_initial',side_effect=AssertionError('unnecessary initial fit')),patch.object(
                NormalModel,'candidate',side_effect=AssertionError('unnecessary update')):
                result=run(cfg)
            self.assertTrue(result['status'].status.eq('complete').all())

    def test_repair_metadata_cannot_be_a_predictor(self):
        ix=pd.date_range('2020-01-01',periods=100,freq='h')
        frame=pd.DataFrame({'load':100.,'target':5.,'Состояние':'Работа'},index=ix)
        with self.assertRaisesRegex(ValueError,'Repair metadata'):
            prepare_unit(frame,{'load':'load','target':'target','Состояние':'state'},'K',load(),{},threshold=25.)

    def test_sparse_positive_classes_are_never_lost_from_replay(self):
        from monitoring_v3.classification import _replay_indices
        y=pd.Series(0.,index=pd.date_range('2020-01-01',periods=10000,freq='h'))
        g=pd.Series([f'negative::{i//100}' for i in range(len(y))],index=y.index)
        y.iloc[123]=1.;y.iloc[397]=1.;g.iloc[123]='event1';g.iloc[397]='event2'
        indices=_replay_indices(y,g,100,load()['risk_policy'])
        self.assertEqual(set(y.iloc[indices]),{0.,1.})
        self.assertEqual(set(g.iloc[indices].loc[y.iloc[indices].eq(1.)]),{'event1','event2'})

    def test_warm_and_reloaded_quantile_models_continue_identically(self):
        from monitoring_v3.models import NormalModel,snapshot_model
        from test_target_training import sample
        x,y,cfg=sample();model=NormalModel('CatBoost',cfg).fit_initial(x,y)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.joblib';joblib.dump(model,path)
            cold=joblib.load(path);warm=snapshot_model(model)
            changed=pd.DataFrame(100.,index=x.index,columns=y.columns)
            warm._fit_trees(x,changed,initial=False);cold._fit_trees(x,changed,initial=False)
            pd.testing.assert_frame_equal(warm.expected(x,changed)[0],cold.expected(x,changed)[0])

    def test_file_warm_starts_do_not_freeze_with_multiple_categorical_models(self):
        import subprocess
        source='\n'.join([
            'import sys,numpy as np,pandas as pd',
            'sys.path.insert(0,'+repr(str(ROOT/'lib'))+')',
            'from monitoring_v3.continuation import fit_continued',
            'from catboost import CatBoostRegressor',
            'rng=np.random.default_rng(17)',
            'x=pd.DataFrame({"load":rng.normal(size=60),"state":["s"+str(i%5) for i in range(60)]})',
            'models=[None,None]',
            'for month in range(60):',
            ' for unit in range(2):',
            '  m=CatBoostRegressor(iterations=2,depth=2,loss_function="MultiQuantile:alpha=0.25,0.5,0.75",learning_rate=.1,has_time=True,thread_count=2,verbose=False,allow_writing_files=False,boost_from_average=False,allow_const_label=True)',
            '  fit_continued(m,x,x.load+unit+month*.1,models[unit],cat_features=["state"])',
            '  models[unit]=m',
            'assert all(m.tree_count_==120 for m in models)',
            'print("FILE_CONTINUATION_OK")'])
        result=subprocess.run([sys.executable,'-X','utf8','-c',source],capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('FILE_CONTINUATION_OK',result.stdout)

    def test_correction_residuals_remain_relative_to_the_same_analog_reference(self):
        from monitoring_v3.models import AnalogCorrection
        from test_incremental import sample,settings
        x,y=sample();cfg=settings();cfg['memory_rows']=1000
        model=AnalogCorrection(cfg).fit_initial(x.iloc[:250],y.iloc[:250])
        base_before=model.base.expected(x,y)[0]
        memory_before=model.base.memory_y.copy()
        new_y=y.iloc[250:290]+10.
        model.update(x.iloc[250:290],new_y)
        pd.testing.assert_frame_equal(base_before,model.base.expected(x,y)[0])
        pd.testing.assert_frame_equal(memory_before,model.base.memory_y)
        pd.testing.assert_frame_equal(model.correction.memory_y.loc[new_y.index],new_y-base_before.loc[new_y.index])
        from monitoring_v3.pipeline import calibration_errors
        calibration_errors(model,x.iloc[300:],y.iloc[300:]+100.)
        self.assertTrue(model.expected(x.iloc[300:],y.iloc[300:])[1].ge(model.floor).all().all())

    def test_shared_target_model_preserves_instrument_and_calibrated_floors(self):
        from monitoring_v3.models import MultiTargetModel
        from monitoring_v3.pipeline import calibration_errors
        from test_target_training import sample
        x,y,cfg=sample()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rules.json';path.write_text(json.dumps({'defaults':{'resolution':5.}}),encoding='utf-8');cfg['sensor_rules']=str(path)
            model=MultiTargetModel('CatBoostMulti',cfg).fit_initial(x,y)
            self.assertTrue(model.expected(x,y)[1].ge(5.).all().all())
            errors=calibration_errors(model,x,y+100.)
            self.assertTrue(model.expected(x,y)[1].ge(model.floor).all().all())
            self.assertTrue(all(np.isfinite(d['constant_baseline_error']) for d in errors.attrs['calibration_sensor_diagnostics']))


if __name__=='__main__':unittest.main()
