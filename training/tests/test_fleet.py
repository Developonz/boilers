"""Regression and end-to-end checks; generated boilers are not production evidence."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
import numpy as np
import pandas as pd
import openpyxl
import legacy_fleet as F
import legacy_metrics as M
import k2_lib as K


def metric_context():
    ix = pd.date_range('2025-01-01', periods=24 * 40, freq='h')
    stop = ix[-1]
    events = pd.DataFrame([dict(start=stop, stop=stop, end=stop + pd.Timedelta('1D'), type=K.EV_FAIL)])
    return dict(t_ix=ix, STEP=pd.Timedelta('1h'), PER_H=1, FIRST_TEST=str(ix[0]),
                running=pd.Series(True, index=ix), scorable=pd.Series(True, index=ix),
                mode=pd.Series(K.M_RUN, index=ix), state=pd.Series(K.ST_WORK, index=ix),
                fails=events, events=events, hs=lambda h: int(h), SMOOTH_H=1, PERSIST_H=3,
                MERGE_GAP_H=12, MIN_LEAD_H=3, PRE_WINDOW_D=7, NEUTRAL_D=7, ALARM_MATCH='overlap',
                THRESHOLD_MULTIPLIERS=[.5, 1., 2.], THRESHOLD_VALIDATION_END=None,
                PROXIMITY_HALF_LIVES_D=[1, 3, 7], WINDOWS_D=[1, 7, 14])


def generated_data(root, units=('К-1', 'К-2', 'К-3'), periods=1800):
    rng = np.random.default_rng(123)
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    ix = pd.date_range('2024-01-01', periods=periods, freq='h')
    for number, unit in enumerate(units, 1):
        load = 100 + 10 * np.sin(np.arange(len(ix)) / 35) + rng.normal(0, 1, len(ix))
        state = np.full(len(ix), K.ST_WORK, dtype=object)
        state[1650:1680] = K.ST_FAIL; load[1650:1680] = 0
        prefix = f'KA-{number}'
        imbalance = f'{prefix} Небаланс пар/вода средний 10 мин (Fpv-Fpp-Frnp-N' + ('c' if number == 1 else 'с') + 'т)'
        columns = [(f'{prefix} Расход перегретого пара', 'нагрузка', 'расход пара'),
                   (f'{prefix} Температура', 'проверять', 'температура'),
                   (f'{prefix} Давление', 'проверять', 'давление'),
                   (imbalance, 'проверять', 'небаланс пар/вода'),
                   (f'{prefix} Механизм', 'состояние', 'механизм')]
        frame = pd.DataFrame({K.TIME_COL: ix, K.STATE_COL: state,
                              columns[0][0]: load, columns[1][0]: 200 + .3 * load + rng.normal(0, .2, len(ix)),
                              columns[2][0]: 12 + .02 * load + rng.normal(0, .02, len(ix)),
                              columns[3][0]: 2 + rng.normal(0, .15, len(ix)),
                              columns[4][0]: np.where(np.arange(len(ix)) % 100 < 70, 'Работа', 'Откл')})
        frame.loc[1600:1649, columns[1][0]] += 5
        folder = root / unit; folder.mkdir(parents=True)
        frame.to_parquet(folder / 'hourly.parquet', index=False)
        sheet = wb.create_sheet(f'КА-{number}')
        sheet.append(['col', 'роль', 'общий параметр', 'единица'])
        for name, role, key in columns:
            sheet.append([name, role, key, None])
    wb.save(root / 'sensor_roles.xlsx')


class MetricsTests(unittest.TestCase):
    def test_ongoing_and_no_coverage_gate(self):
        ns = metric_context(); ix = ns['t_ix']
        score = pd.Series(np.nan, index=ix)
        score.iloc[-9*24:-6*24] = 2.
        res = dict(score=score, thr=pd.Series(1., index=ix))
        m, ev, _ = M.evaluate(res, ns)
        self.assertEqual(m['recall'], 1.)
        self.assertEqual(m['precision'], 1.)
        self.assertTrue(ev.iloc[0]['продолжающаяся'])
        self.assertLess(ev.iloc[0]['рассчитано часов'], 0.5 * 165)

    def test_always_on_is_visible_in_occupancy(self):
        ns = metric_context(); ix = ns['t_ix']
        res = dict(score=pd.Series(2., index=ix), thr=pd.Series(1., index=ix))
        m, _, _ = M.evaluate(res, ns)
        self.assertEqual(m['recall'], 1.)
        self.assertEqual(m['% фона в тревоге'], 1.)

    def test_nan_does_not_bridge_episodes(self):
        ns = metric_context(); ix = ns['t_ix']
        score = pd.Series(0., index=ix); score.iloc[10:15] = 2; score.iloc[15:17] = np.nan; score.iloc[17:22] = 2
        _, ep, _, _ = M.alarm_series(dict(score=score, thr=pd.Series(1., index=ix)), ns)
        self.assertEqual(len(ep), 2)
        from k2_deploy import alarm_episodes
        self.assertEqual(len(alarm_episodes(score, 3, 12, pd.Timedelta('1h'))), 2)

    def test_soft_ap_ties_and_binary_equivalence(self):
        r = np.array([1., .5, 0., .2])
        self.assertAlmostEqual(M.soft_average_precision(r, np.ones(4)), r.mean())
        binary = np.array([0, 1, 1, 0, 1]); score = np.array([.1, .7, .7, .2, .4])
        from sklearn.metrics import average_precision_score
        self.assertAlmostEqual(M.soft_average_precision(binary, score), average_precision_score(binary, score))

    def test_threshold_split_and_ap_invariance(self):
        ns = metric_context(); ix = ns['t_ix']
        ns['THRESHOLD_VALIDATION_END'] = str(ix[24*25])
        res = dict(score=pd.Series(np.linspace(0, 3, len(ix)), index=ix), thr=pd.Series(1., index=ix))
        table = M.threshold_table(res, ns)
        self.assertEqual(len(table), 6)
        self.assertTrue((table[table['период'].eq('прошлый проверочный')]['всего остановов'] == 0).all())


class FleetTests(unittest.TestCase):
    def test_plan_multiple_groups_and_singleton(self):
        cfg = F.defaults(); cfg['GROUPS'] = {'A': ['К-1', 'К-2', 'К-3', 'К-4'], 'B': ['К-5','К-6','К-7'], 'C': ['К-8']}
        p = F.plan(cfg)
        self.assertEqual(p.status.eq('запланировано').sum(), 40)
        self.assertEqual((~p.status.eq('запланировано')).sum(), 2)
        self.assertEqual(len(p[(p.group == 'C') & p.status.eq('запланировано')]), 1)

    def test_end_to_end_and_cache(self):
        with tempfile.TemporaryDirectory(prefix='legacy_fleet_test_') as tmp:
            root = Path(tmp); generated_data(root)
            cfg = F.defaults()
            cfg.update(GROUPS={'pair': ['К-1', 'К-2'], 'single': ['К-3']}, DATA_ROOT=str(root),
                       SENSOR_LIST_FILE=str(root / 'sensor_roles.xlsx'),
                       FIRST_TEST='2024-03-01', TRAIN_HOURS=700, USE_NEURAL=False,
                       EXPORT=True, LEGACY_OUTPUT_DIR=str(root/'results'), LEGACY_MODELS_DIR=str(root/'models'),
                       CONTINUE_ON_ERROR=False, SHOW_PROGRESS_RESULTS=False)
            F.preflight(cfg)
            result = F.run(cfg)
            self.assertFalse(result['errors'])
            self.assertEqual(len(result['records']), 9)
            self.assertEqual(len(result['tables']['main']), 81)
            self.assertFalse(result['tables']['main']['PR AUC'].isna().all())
            self.assertTrue(result['tables']['main'].groupby(['group','scenario','target','donors'])['всего остановов'].min().eq(1).all())
            pair = F.paired_comparison(result)
            self.assertGreater(len(pair), 0)
            fits = pd.concat([r['fits'] for r in result['records']], ignore_index=True)
            self.assertIn('cache', fits.status.unique())
            self.assertTrue(list((root/'models').rglob('final/model.py')))
            self.assertTrue(list((root/'models').rglob('2024-03/model.py')))
            # A second run reads scores and already exported models; no fitting is needed.
            with patch.object(F, '_prepare', side_effect=AssertionError('checkpoint must avoid training')):
                again = F.run(cfg)
            pd.testing.assert_frame_equal(result['tables']['main'], again['tables']['main'])
            self.assertTrue(all(s['status'] == 'из сохранённого отчёта' for s in again['status']))
            # Standalone runner must retain HTML tables and images in the executed notebook.
            spec = importlib.util.spec_from_file_location('night_runner', ROOT/'run_overnight.py')
            runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
            cells = [dict(cell_type='code', source=["from IPython.display import display, HTML\ndisplay(HTML('<b>table</b><img src=\"data:image/png;base64,AA==\">'))"], metadata={}, outputs=[], execution_count=None)]
            nbpath = root/'display.ipynb'; nbpath.write_text(json.dumps(dict(cells=cells, metadata={}, nbformat=4, nbformat_minor=5)), encoding='utf-8')
            executed = runner.execute(nbpath, root/'display_out.ipynb')
            self.assertTrue(any(o.get('data', {}).get('text/html') for o in executed['cells'][0]['outputs']))

    def test_donor_without_training_hours_and_residual_transfer(self):
        with tempfile.TemporaryDirectory(prefix='legacy_residual_test_') as tmp:
            root = Path(tmp); generated_data(root, units=('К-1', 'К-2'))
            cfg = F.defaults()
            cfg.update(GROUPS={'pair': ['К-1', 'К-2']}, DATA_ROOT=str(root),
                       SENSOR_LIST_FILE=str(root/'sensor_roles.xlsx'), FIRST_TEST='2024-03-01',
                       TRAIN_HOURS=700, USE_NEURAL=False, MAIN_METHODS=[],
                       OTHER_METHODS=['Остатки analogs', 'Остатки AAKR', 'Остатки MSET'],
                       RESIDUAL_METHODS=['PCA', 'Махаланобис'], SCENARIOS=['pooled','transfer'],
                       EXPORT=True, LEGACY_OUTPUT_DIR=str(root/'results'), LEGACY_MODELS_DIR=str(root/'models'),
                       CONTINUE_ON_ERROR=False, SHOW_PROGRESS_RESULTS=False)
            result = F.run(cfg)
            self.assertFalse(result['errors'])
            self.assertEqual(len(result['records']), 4)
            self.assertTrue(all(any('Остатки MSET' in v for v in r['views']) for r in result['records']))
            # Existing loader should work for standalone two-stage exports too.
            from saved_model_runner import _runtime
            folder = next((root/'models').rglob('resid_aakr_pca/final/model.py')).parent
            with _runtime(folder, 'legacy') as (model, _):
                self.assertIsNotNone(model)
            # Check donor omission independently: no need to throw away unit 1's useful past.
            p = root/'К-2'/'hourly.parquet'; raw = pd.read_parquet(p)
            raw.loc[:1439, 'KA-2 Расход перегретого пара'] = 0
            raw.to_parquet(p, index=False)
            cfg.update(MAIN_METHODS=['analogs'], OTHER_METHODS=[], RESIDUAL_METHODS=[], EXPORT=False)
            job = F.plan(cfg).iloc[0]
            from types import SimpleNamespace
            ns = F._prepare(cfg, SimpleNamespace(**job.to_dict()))
            tm = ns['train_one'](lambda: ns['K'].AnalogCondDet(ns['COND_COLS']), pd.Timestamp('2024-03-01'), F._views('analogs'))
            self.assertIsNotNone(tm)
            self.assertEqual(tm['training_rows_by_unit']['К-2'], 0)

    def test_fallback_never_uses_future_model(self):
        import legacy_group as G
        engine = G.GroupTraining.__new__(G.GroupTraining)
        engine.ns = {'REUSE_PREVIOUS_MODEL': True}
        engine._successful = {}; engine.fit_audit = []; engine.residual_kind = None
        engine._for_target = lambda raw: raw
        class Detector:
            pass
        first = pd.Timestamp('2025-03-01')
        engine._train_raw = lambda make, cutoff, views, rows: dict(trained_cutoff=str(cutoff)) if cutoff == first else None
        self.assertIsNotNone(engine.train_one(Detector, first, []))
        reused = engine.train_one(Detector, pd.Timestamp('2025-04-01'), [])
        self.assertEqual(reused['trained_cutoff'], str(first))
        self.assertEqual(engine.fit_audit[-1]['status'], 'previous_model')
        self.assertIsNone(engine.train_one(Detector, pd.Timestamp('2025-02-01'), []))

    def test_original_training_and_new_adapter_agree(self):
        with tempfile.TemporaryDirectory(prefix='legacy_parity_') as tmp:
            root = Path(tmp); generated_data(root, units=('К-1',))
            cfg = F.defaults()
            cfg.update(GROUPS={'one': ['К-1']}, DATA_ROOT=str(root), SENSOR_LIST_FILE=str(root/'sensor_roles.xlsx'),
                       FIRST_TEST='2024-03-01', TRAIN_HOURS=700, USE_NEURAL=False, EXPORT=False,
                       LEGACY_OUTPUT_DIR=str(root/'results'), LEGACY_MODELS_DIR=str(root/'models'), SCENARIOS=['local'])
            from types import SimpleNamespace
            job = SimpleNamespace(**F.plan(cfg).iloc[0].to_dict())
            ns = F._prepare(cfg, job)
            engine = ns['GROUP_ENGINE']; cutoff = pd.Timestamp('2024-03-01')
            for name, make in [('analogs', lambda: ns['K'].AnalogCondDet(ns['COND_COLS'])),
                               ('AAKR', lambda: ns['KernelDet'](ns['COND_COLS'], 'aakr')),
                               ('MSET', lambda: ns['KernelDet'](ns['COND_COLS'], 'mset'))]:
                views = F._views(name)
                old = engine.original_train(make, cutoff, views)
                adapted = engine.train_one(make, cutoff, views)
                for view in views:
                    np.testing.assert_allclose(old['cal'][view['name']][0], adapted['cal'][view['name']][0], rtol=1e-7)
                    self.assertAlmostEqual(old['cal'][view['name']][1], adapted['cal'][view['name']][1], places=7)
                te = ns['scorable'].to_numpy() & (ns['t_ix'] >= cutoff)
                before, before_ix = ns['model_errors'](old['det'], old['prep'], te)
                after, after_ix = ns['model_errors'](adapted['det'], adapted['prep'], te)
                pd.testing.assert_index_equal(before_ix, after_ix)
                np.testing.assert_allclose(before, after, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
