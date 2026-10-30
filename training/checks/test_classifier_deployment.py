"""Native classifier export -> independent server -> matching decisions."""
from contextlib import redirect_stdout
from copy import deepcopy
import io,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BACKEND=next(p for p in (ROOT.parent/'backend',ROOT.parent/'dashboard/backend',ROOT.parent.parent/'backend')
             if (p/'classifier_service.py').is_file())
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'checks'),str(BACKEND)]
from test_direct_risk import context
from direct_risk import direct_options,direct_dataset,direct_fit,direct_features,direct_native
from classifier_export import export_classifier,export_direct_suite
from classifier_service import ClassifierService,_load
from catboost import Pool


class ClassifierDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name)
        cls.c,cls.cfg=context(3600,failures=[700,1700,2700])
        cls.cfg['sensor_rules']=str(cls.base/'rules.json');Path(cls.cfg['sensor_rules']).write_text('{}')
        cls.c['aliases']={'КА-1 Температура':'temperature'}
        cls.options=direct_options(dict(horizon_hours=48,iterations=12,depth=2,threads=1,train_step_hours=6,
            train_end=str(cls.c['frame'].index[1300]),validation_end=str(cls.c['frame'].index[2400]),early_stopping_rounds=5))
        with redirect_stdout(io.StringIO()):
            cls.x,meta,schema,_=direct_dataset({'К-1':cls.c},cls.cfg,cls.options)
            cls.package,*_=direct_fit(cls.x,meta,schema,cls.options,cls.base/'fit',verbose=False,contexts={'К-1':cls.c},cfg=cls.cfg)
        cls.package['experiment']=dict(method='unit',units=['К-1'],name='К-1',variant='full',horizon_hours=48)
        cls.export=export_classifier(cls.package,{'К-1':cls.c},cls.cfg,cls.base/'export')
        raw=cls.c['frame'].rename(columns={'temperature':'КА-1 Температура'}).reset_index(names='TimeValue')
        cls.data=cls.base/'hourly.parquet';raw.to_parquet(cls.data,index=False)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_portable_features_and_scores_match_training_without_future_data(self):
        runtime=_load(self.export);at=self.c['frame'].index[3200]
        raw=pd.read_parquet(self.data)
        features,meta,missing=runtime.features(raw,'К-1',at)
        self.assertFalse(missing)
        expected=self.x.xs('К-1',level='unit').loc[[at]]
        expected.index=expected.index.as_unit('ns')
        pd.testing.assert_frame_equal(features[self.package['columns']],expected[self.package['columns']],check_names=False,check_freq=False,atol=2e-5,rtol=2e-5)
        score=runtime.predict(raw,'К-1',at)[0].loc[at,'score']
        reference=self.package['model'].predict_proba(Pool(direct_native(expected,self.package['columns'],self.package['categorical']),cat_features=self.package['categorical']),thread_count=1)[0,1]
        self.assertAlmostEqual(score,reference,places=10)
        changed=raw.copy();changed.loc[pd.to_datetime(changed.TimeValue)>at,'КА-1 Температура']=100000
        changed['Состояние']='Аварийный ремонт';changed['_Тип']='ИСП'
        self.assertAlmostEqual(runtime.predict(changed,'К-1',at)[0].loc[at,'score'],score,places=10)

    def test_bounded_features_preserve_long_state_duration(self):
        c=deepcopy(self.c);c['frame']['state']='on'
        full,_,_=direct_features(c,self.cfg,self.options)
        at=c['frame'].index[3200]
        tail,_,_=direct_features(c,self.cfg,self.options,start=at)
        self.assertEqual(len(tail),400)
        pd.testing.assert_frame_equal(tail.iloc[:1],full.loc[[at]],check_names=False,check_freq=False,atol=2e-5,rtol=2e-5)
        self.assertGreater(float(tail.iloc[0]['cat::state::held_hours']),24)

    def test_server_configured_threshold_and_historical_guard(self):
        service=ClassifierService(self.base,self.data,dict(model_dir=str(self.export),unit='К-1',threshold=0.0))
        result=service.evaluate(self.c['frame'].index[3200])
        self.assertEqual(result['status'],'ok',result)
        self.assertTrue(result['risk']);self.assertEqual(result['threshold'],0.)
        self.assertTrue(result['confirmed'])
        self.assertIsNone(service._data)
        self.assertIs(service.evaluate(self.c['frame'].index[3200]),result)
        self.assertEqual(service.evaluate(self.c['frame'].index[1000])['status'],'not_yet_available')

    def test_last_usable_pair_and_fallback_05(self):
        root=self.base/'versions';root.mkdir(exist_ok=True)
        old=deepcopy(self.package);old['threshold']=.3;old['thresholds']={}
        new=deepcopy(self.package);new['available_at']=str(pd.Timestamp(old['available_at'])+pd.Timedelta('24h'))
        new['threshold']=np.inf;new['thresholds']={}
        export_classifier(old,{'К-1':self.c},self.cfg,root/'old')
        export_classifier(new,{'К-1':self.c},self.cfg,root/'new')
        (root/'index.json').write_text(json.dumps([{'folder':'old'},{'folder':'new'}]))
        service=ClassifierService(self.base,self.data,dict(model_dir=str(root),unit='К-1'))
        result=service.evaluate(self.c['frame'].index[3200])
        self.assertEqual(result['status'],'ok',result);self.assertEqual(result['modelVersion'],'old');self.assertEqual(result['threshold'],.3)
        fallback=self.base/'fallback';export_classifier(new,{'К-1':self.c},self.cfg,fallback)
        result=ClassifierService(self.base,self.data,dict(model_dir=str(fallback),unit='К-1')).evaluate(self.c['frame'].index[3200])
        self.assertEqual(result['threshold'],.5,result);self.assertEqual(result['thresholdSource'],'fallback_05')

    def test_stopped_boiler_is_unknown_risk_not_negative(self):
        result=ClassifierService(self.base,self.data,dict(model_dir=str(self.export),unit='К-1')).evaluate(self.c['frame'].index[2710])
        self.assertIsNone(result['risk'],result);self.assertEqual(result['label'],'Нет оценки')

    def test_damaged_export_rejected_without_pickle(self):
        broken=self.base/'damaged';export_classifier(self.package,{'К-1':self.c},self.cfg,broken)
        path=broken/'classifier.cbm';path.write_bytes(path.read_bytes()+b'bad')
        with self.assertRaisesRegex(ValueError,'Изменён файл'):_load(broken)

    def test_group_model_uses_exact_unit_and_can_be_shared(self):
        second=deepcopy(self.c);second['unit']='К-2';second['aliases']={'КА-2 Температура':'temperature'}
        second['frame']['temperature']+=1.;second['cleaned']['temperature']+=1.
        contexts={'К-1':self.c,'К-2':second}
        with redirect_stdout(io.StringIO()):
            x,meta,schema,_=direct_dataset(contexts,self.cfg,self.options)
            package,*_=direct_fit(x,meta,schema,self.options,self.base/'group_fit',verbose=False,contexts=contexts,cfg=self.cfg)
        package['experiment']=dict(method='group',units=list(contexts),name='test')
        target=export_classifier(package,contexts,self.cfg,self.base/'group_export')
        self.assertIn('unit',package['columns'])
        runtime=_load(target)
        at=second['frame'].index[3200]
        for unit,c in contexts.items():
            raw=c['frame'].reset_index(names='TimeValue')
            actual=runtime.predict(raw,unit,at)[0].loc[at,'score']
            expected=x.xs(unit,level='unit').loc[[at]]
            score=package['model'].predict_proba(Pool(direct_native(expected,package['columns'],package['categorical']),cat_features=package['categorical']),thread_count=1)[0,1]
            self.assertAlmostEqual(actual,score,places=10)
        self.assertIs(_load(target),runtime)

    def test_export_uses_frozen_rules_not_current_settings_file(self):
        path=Path(self.cfg['sensor_rules']);old=path.read_text()
        try:
            path.write_text('{"defaults":{"mask_spikes":true}}')
            folder=export_classifier(self.package,{'К-1':self.c},self.cfg,self.base/'frozen_export')
            config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['sensor_rules'],{})
        finally:path.write_text(old)

    def test_suite_export_without_reloading_telemetry(self):
        import joblib
        root=self.base/'suite';root.mkdir(exist_ok=True)
        source=root/'trained';source.mkdir(exist_ok=True)
        joblib.dump(self.package,source/'classifier.joblib')
        row=dict(method='unit',name='К-1',horizon_hours=48,variant='full',folder=str(source),validation_end=self.package['available_at'])
        pd.DataFrame([row]).to_csv(root/'validation_choices.csv',index=False)
        output=export_direct_suite(root,method='unit',horizons=[48])
        self.assertEqual(len(output),1)
        folder=Path(output.iloc[0].model_dir)
        self.assertTrue((folder/'index.json').is_file())
        risk=ClassifierService(self.base,self.data,dict(model_dir=str(folder),unit='К-1')).evaluate(self.c['frame'].index[3200])
        self.assertEqual(risk['status'],'ok',risk)


if __name__=='__main__':unittest.main()
