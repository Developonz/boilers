"""01 uses shared stop/eligibility/admission rules and renders without fitting."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'checks')]
from test_runtime_performance import fixture
from monitoring_v3.data import load_sources,healthy
from monitoring_v3.partitioning import partition_history
from monitoring_v3.pipeline import dataset
from direct_risk import direct_features
import preparation_notebook as E
import detector_notebook as DR


class PreparationNotebookTests(unittest.TestCase):
    def test_full_notebook_shared_stops_and_admission_no_fit(self):
        import matplotlib
        matplotlib.use('Agg')
        old_cwd=Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp,warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cfg=fixture(tmp)
                source=Path(tmp)/'source/К-1/hourly.parquet'
                raw=pd.read_parquet(source)
                raw.loc[1800:1823,'load']=0.0  # No document; state remains "Работа".
                raw.loc[2150:2173,'load']=0.0
                raw.loc[2150:2173,'Состояние']='Аварийный ремонт'
                raw.loc[1900,'load']=0.0;raw.loc[1901,'load']=np.nan
                raw.loc[1700:1795,'pressure']=13.5
                raw.loc[1796:1799,'pressure']=14.0
                raw.to_parquet(source,index=False)
                (Path(tmp)/'rules.json').write_text(json.dumps({'sensors':{'pressure':{'kind':'continuous','mask_frozen':True}}}))
                n=json.loads((ROOT/'01_EDA.ipynb').read_text(encoding='utf-8'))
                ns={};html=[]
                with patch.object(E,'show_html',side_effect=html.append),patch.object(DR,'show_html',side_effect=html.append),\
                     patch('monitoring_v3.pipeline.run',side_effect=AssertionError('regression fitting forbidden')),\
                     patch('direct_risk.direct_fit',side_effect=AssertionError('classifier fitting forbidden')),redirect_stdout(io.StringIO()):
                    for i,c in enumerate(n['cells']):
                        if c['cell_type']!='code':continue
                        code=''.join(c['source']);compile(code,f'cell-{i}','exec');exec(code,ns)
                        if 'SOURCE_NOTEBOOK =' in code:
                            ns.update(SOURCE_NOTEBOOK=None,DATA_SETTINGS=cfg,EDA_OUTPUT=str(Path(tmp)/'eda'))
                            ns['EDA'].update(max_stop_plots=1,max_event_plots=1,max_analysis_rows=1000,max_heatmap_sensors=2)
                            ns['CLASSIFIER_PREVIEW'].update(enabled=True,horizons=[24])
                c=ns['CONTEXTS']['К-1'];stops=ns['STOPS'];ix=c['frame'].index
                self.assertEqual(int(stops.observed_stop.sum()),3)
                unknown=stops.loc[stops.no_ticket].iloc[0]
                self.assertEqual(unknown.physical_stop,ix[1800])
                self.assertEqual(unknown.detected_at,ix[1801])
                self.assertEqual(unknown.restart,ix[1824])
                self.assertEqual(unknown.type,'Неизвестно')
                self.assertEqual(unknown.physical_duration_hours,24.)
                self.assertNotIn(ix[1900],stops.physical_stop.tolist())
                _,meta,_=direct_features(c,cfg,ns['CLASSIFIER_PREVIEW']['options'])
                preview=ns['CLASSIFIER_LABELS'].iloc[0]
                self.assertEqual(preview.candidate_rows,int(meta.eligible.sum()))
                self.assertGreater(preview.positive_rows,0)
                self.assertGreater(preview.unknown_rows,0)
                x,y=dataset(c,'conditions',0,cfg)
                for row in ns['AVAILABILITY'].itertuples():
                    mask=healthy(c,row.cutoff,cfg)&y.notna().mean(axis=1).ge(cfg['minimum_coverage'])
                    self.assertEqual(row.admitted_rows,int(mask.sum()))
                    sample=x.loc[mask].copy()
                    sample.index=pd.MultiIndex.from_arrays([sample.index,np.repeat('К-1',len(sample))],names=['time','unit'])
                    *_,audit=partition_history(sample,cfg)
                    self.assertEqual(row.train_rows,audit['train'])
                    self.assertEqual(row.calibration_rows,audit['calibration'])
                rare=pd.read_csv(Path(tmp)/'eda/К-1/before_stops/plateau_changes_before_stops.csv')
                self.assertTrue(rare.sensor.eq('pressure').any())
                self.assertIn('data:image/png;base64,',''.join(html))
                self.assertTrue((Path(tmp)/'eda/summary.csv').exists())
                self.assertFalse(list((Path(tmp)/'eda').rglob('*.joblib')))
        finally:os.chdir(old_cwd)

    def test_saved_notebook_literals_are_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.ipynb'
            p.write_text(json.dumps({'cells':[{'cell_type':'code','source':[
                'raise RuntimeError("MUST NOT EXECUTE")\nSETTINGS={"groups":{"x":["К-1"]}}\nSENSOR_RULES={"sensors":{}}']}]}))
            self.assertEqual(E.saved_settings(p)['SETTINGS']['groups'],{'x':['К-1']})


if __name__=='__main__':unittest.main()
