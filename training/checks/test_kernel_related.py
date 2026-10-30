"""Classical all-sensor AAKR/MSET and portable incremental models."""
from contextlib import redirect_stdout
from pathlib import Path
import io,json,sys,tempfile,unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'checks')]
from test_target_training import sample
from test_runtime_performance import fixture
from monitoring_v3.models import RelatedKernelModel,snapshot_model
from monitoring_v3.pipeline import experiments,run
from monitoring_v3.config import load
from monitoring_v3.runtime import Model


class KernelRelatedTests(unittest.TestCase):
    def test_experiment_grid_includes_both_kernel_variants(self):
        cfg=load(overrides={'models':['AAKR','MSET'],'feature_sets':['conditions','related','forecast']})
        specs=list(experiments(cfg))
        self.assertEqual({(s['method'],s['variant']) for s in specs},
                         {('AAKR','conditions'),('AAKR','related'),('MSET','conditions'),('MSET','related')})

    def test_classic_inputs_include_self_and_survive_incremental_update(self):
        x,y,cfg=sample();cfg['memory_rows']=50
        y['third']=y.temperature*.2+y.pressure
        f=pd.concat([x,y.add_prefix('sensor::')],axis=1)
        for method in ('AAKR','MSET'):
            with self.subTest(method=method):
                model=RelatedKernelModel(method,cfg).fit_initial(f,y)
                self.assertEqual(model.kernel.mode,'all')
                self.assertIsNone(model.kernel.lin)
                self.assertFalse(model.kernel.robust)
                if method=='MSET':self.assertEqual(model.kernel.lam,0.)
                self.assertFalse(any(c.startswith('sensor::') for c in model.encoder.columns))
                base=model.expected(f.iloc[:10],y.iloc[:10])[0]
                modified=y.iloc[:10].copy();modified['temperature']+=1000
                changed=f.iloc[:10].copy();changed['sensor::temperature']+=1000
                predicted=model.expected(changed,modified)[0]
                self.assertTrue(np.isfinite(predicted.temperature).all())
                self.assertGreater(float(abs(predicted.temperature-base.temperature).max()),1e-4)
                candidate=snapshot_model(model)
                ix=pd.date_range('2025-01-01',periods=20,freq='h')
                candidate.update(f.iloc[:20].set_axis(ix),y.iloc[:20].set_axis(ix))
                self.assertEqual(candidate.updates,1)
                pd.testing.assert_frame_equal(base,model.expected(f.iloc[:10],y.iloc[:10])[0])

    def test_incomplete_vectors_are_not_imputed_for_training_or_inference(self):
        x,y,cfg=sample();f=pd.concat([x,y.add_prefix('sensor::')],axis=1)
        y.loc[y.index[:10],'pressure']=np.nan
        for method in ('AAKR','MSET'):
            model=RelatedKernelModel(method,cfg).fit_initial(f,y)
            self.assertEqual(model.vector_diagnostics['complete_rows'],140)
            self.assertTrue(np.isfinite(model.kernel.Mx).all())
            self.assertEqual(model.counts.tolist(),[140,140])
            p,_=model.expected(f.iloc[:12],y.iloc[:12])
            self.assertTrue(p.iloc[:10].isna().all().all())
            self.assertTrue(np.isfinite(p.iloc[10:]).all().all())

    def test_classical_kernels_match_direct_matrix_formulas(self):
        x,y,cfg=sample();f=pd.concat([x,y.add_prefix('sensor::')],axis=1)
        cfg['memory_rows']=30
        for method in ('AAKR','MSET'):
            with self.subTest(method=method):
                model=RelatedKernelModel(method,cfg).fit_initial(f,y)
                predicted=model.expected(f.iloc[:10],y.iloc[:10])[0]
                k=model.kernel
                conditions=model._numeric_features(model._conditions(f.iloc[:10]))
                query=np.hstack([k._cond(conditions,np.empty((10,0))).astype(np.float32),
                                 ((y.iloc[:10]-k.xm)/k.xs).to_numpy().astype(np.float32)]).astype(float)
                memory=np.hstack([k.Mc,k.Mx]).astype(float)
                distance=((query[:,None]-memory[None])**2).mean(-1)
                rhs=np.exp(-(distance-distance.min(1,keepdims=True))/(k.h*k.h))
                if method=='MSET':
                    gram=np.exp(-((memory[:,None]-memory[None])**2).mean(-1)/(k.h*k.h))
                    weights=np.exp(-distance/(k.h*k.h))@np.linalg.pinv(gram,rcond=1e-10,hermitian=True)
                    expected=(weights@k.Mx)*k.xs+k.xm
                else:
                    weights=rhs
                    expected=(weights@k.Mx)/weights.sum(1,keepdims=True)*k.xs+k.xm
                np.testing.assert_allclose(predicted,expected,rtol=1e-5,atol=1e-4)

    def test_singular_dictionary_uses_pseudoinverse_without_ridge(self):
        x,y,cfg=sample()
        x=pd.concat([x.iloc[:20],x.iloc[:20]]).set_axis(pd.date_range('2024-01-01',periods=40,freq='h'))
        y=pd.concat([y.iloc[:20],y.iloc[:20]]).set_axis(x.index)
        model=RelatedKernelModel('MSET',cfg).fit_initial(x,y)
        p,_=model.expected(x,y)
        self.assertTrue(np.isfinite(p).all().all())
        self.assertEqual(model.kernel.lam,0.)
        self.assertLess(model.kernel.dictionary_rank,len(model.kernel.Mx))

    def test_kernel_related_models_export_load_and_finish_all_months(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg.update(models=['AAKR','MSET'],feature_sets=['related'],validate_candidate=False)
            result=run(cfg,train_classifiers=False)
            self.assertEqual(len(result['status']),2)
            self.assertTrue(result['status'].status.eq('complete').all(),result['status'].to_string())
            for directory in (Path(cfg['output'])/'experiments').glob('*/К-1/model'):
                model=Model(directory)
                metadata=json.loads((directory/'sensor_inputs.json').read_text(encoding='utf-8'))
                self.assertEqual(metadata['mode'],'all')
                self.assertFalse(metadata['exclude_self'])
                self.assertFalse(metadata['linear_baseline'])
                self.assertEqual(metadata['full_memory'],model.spec['method']=='MSET')
                if model.spec['method']=='MSET':self.assertEqual(metadata['ridge'],0.)
                self.assertGreater(model.state['model'].updates,0)
                self.assertTrue((directory/'related_inputs.csv').is_file())
                raw=pd.read_parquet(Path(tmp)/'source/К-1/hourly.parquet').set_index('TimeValue').iloc[-96:]
                scored=model.score(raw)
                self.assertTrue(scored.hourly[('mean::instant','ошибка')].notna().any())

    def test_pipeline_explains_missing_complete_vectors_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            cfg=fixture(tmp);cfg.update(models=['MSET'],feature_sets=['related'],minimum_coverage=.5)
            path=Path(tmp)/'source/К-1/hourly.parquet'
            raw=pd.read_parquet(path);raw['pressure']=np.nan;raw.to_parquet(path,index=False)
            result=run(cfg,train_classifiers=False)
            self.assertTrue(result['status'].status.eq('insufficient_history').all(),result['status'].to_string())
            experiment=next((Path(cfg['output'])/'experiments').glob('*'))
            updates=json.loads((experiment/'updates.json').read_text(encoding='utf-8'))
            self.assertTrue(all(row['status']=='insufficient_complete_training_vectors' for row in updates))
            blockers=pd.read_csv(experiment/'complete_vector_missingness.csv')
            self.assertIn('pressure',set(blockers.sensor))


if __name__=='__main__':unittest.main()
