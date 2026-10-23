"""Checks of chronological updates, cache migration and export without hidden fitting."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from test_fleet import generated_data
import legacy_fleet as F
import legacy_group as G
import numpy as np
import pandas as pd


def prepare(root, mode='fast', units=('К-1',), scenario='local'):
    cfg = F.defaults()
    cfg.update(GROUPS={'test': list(units)}, DATA_ROOT=str(root), SENSOR_LIST_FILE=str(root/'sensor_roles.xlsx'),
               FIRST_TEST='2024-03-01', TRAIN_HOURS=1000, EXPORT=False, USE_NEURAL=False,
               SCENARIOS=[scenario], LOCAL_SENSOR_SETS=['common'], TRAIN_UPDATE=mode,
               IMPORT_LEGACY_CACHE=False, SHOW_PROGRESS_RESULTS=False,
               LEGACY_OUTPUT_DIR=str(root/'results'), LEGACY_MODELS_DIR=str(root/'models'))
    job = SimpleNamespace(**F.plan(cfg).query("status == 'запланировано'").iloc[0].to_dict())
    return cfg, F._prepare(cfg, job)


class FastTests(unittest.TestCase):
    def test_updated_exports_predict_like_in_memory_models(self):
        from saved_model_runner import _runtime
        with tempfile.TemporaryDirectory(prefix='fast_export_') as tmp:
            root = Path(tmp); generated_data(root, units=('К-1',), periods=3300)
            path = root/'К-1/hourly.parquet'
            frame = pd.read_parquet(path)
            frame.loc[3200:, 'KA-1 Температура'] = frame.loc[3199, 'KA-1 Температура']
            frame.loc[3250, 'KA-1 Давление'] += 1000
            frame.to_parquet(path, index=False)
            cfg, _ = prepare(root)
            cfg['EXPORT'] = True
            job = SimpleNamespace(**F.plan(cfg).iloc[0].to_dict())
            ns = F._prepare(cfg, job)
            F._run_methods(ns)
            engine = ns['GROUP_ENGINE']
            for name, folder, make in [('analogs', 'analogs', lambda: ns['K'].AnalogCondDet(ns['COND_COLS'])),
                                       ('AAKR', 'aakr', lambda: ns['KernelDet'](ns['COND_COLS'], 'aakr')),
                                       ('MSET', 'mset', lambda: ns['KernelDet'](ns['COND_COLS'], 'mset'))]:
                path = root/'models/test'/job.key/'К-1'/folder/'final'
                saved_cfg = json.loads((path/'config.json').read_text(encoding='utf-8'))
                self.assertEqual(saved_cfg['training_update'], 'fast')
                self.assertEqual(saved_cfg['calibration_strategy'], 'fast_prequential')
                tm = engine._train_raw(make, ns['M_NEXT'], F._views(name), 'часть')
                rows = ns['scorable'].to_numpy() & ns['enough'].to_numpy()
                rows[:-100] = False
                expected, ix = ns['model_errors'](tm['det'], tm['prep'], rows)
                with _runtime(path, 'legacy') as (model, _):
                    actual, pos, _, _ = model.errors(ns['df'], rows, ns['scorable'].to_numpy())
                    pd.testing.assert_index_equal(ns['t_ix'][pos], ix)
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
                    for v in F._views(name):
                        np.testing.assert_allclose(model.q[v['name']], tm['cal'][v['name']][0])

    def test_prequential_errors_and_single_fit_for_all_main_methods(self):
        with tempfile.TemporaryDirectory(prefix='fast_updates_') as tmp:
            root = Path(tmp); generated_data(root, periods=3300)
            cfg, ns = prepare(root, units=('К-1', 'К-2'), scenario='pooled')
            engine = ns['GROUP_ENGINE']
            for name, make in [('analogs', lambda: ns['K'].AnalogCondDet(ns['COND_COLS'])),
                               ('AAKR', lambda: ns['KernelDet'](ns['COND_COLS'], 'aakr')),
                               ('MSET', lambda: ns['KernelDet'](ns['COND_COLS'], 'mset'))]:
                views = F._views(name)
                first, second = pd.Timestamp('2024-03-01'), pd.Timestamp('2024-04-01')
                raw = engine._train_raw(make, first, views, 'часть')
                self.assertEqual(raw['fit_calls'], 4)
                pool = engine._pool_namespace(False)
                mask = pool['masks'](second)
                new = mask & (engine.actual_index >= first) & (engine.actual_index < second)
                expected = []
                for left, right in engine.bounds.values():
                    rows = new.copy(); rows[:left] = False; rows[right:] = False
                    expected.append(pool['model_errors'](raw['det'], raw['prep'], rows))
                # A fast update must not call the former four-fit OOF procedure.
                with patch.object(engine, 'original_train', side_effect=AssertionError('unexpected OOF refit')):
                    updated = engine._train_raw(make, second, views, 'часть')
                self.assertEqual(updated['fit_calls'], 1)
                self.assertEqual(updated['update_mode'], 'fast_prequential')
                self.assertEqual(updated['update_error_rows'], int(new.sum()))
                for (error, index), (actual_error, actual_index) in zip(expected, updated['parts'][-len(expected):]):
                    pd.testing.assert_index_equal(index, actual_index)
                    np.testing.assert_allclose(error, actual_error)
                # All retained calibration errors refer to the current eligible past.
                for error, index in updated['parts']:
                    positions = engine.pool_index.get_indexer(index)
                    self.assertTrue(mask[positions].all())
                    self.assertTrue((engine.actual_index[positions] < second).all())
                # Final fitted detector is identical to fitting the legacy detector on this window.
                old = G._clone(engine.original_train, pool)(make, second, views, 'часть')
                a, ix = pool['model_errors'](updated['det'], updated['prep'], mask, max_rows=20)
                b, _ = pool['model_errors'](old['det'], old['prep'], mask, max_rows=20)
                np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)
                with patch.object(engine, 'original_train', side_effect=AssertionError('export cannot fit')):
                    missing = engine.model_for_export(make, pd.Timestamp('2024-05-01'), views)
                self.assertEqual(pd.Timestamp(missing['trained_cutoff']), second)
                # No new selected hours => no fit; a future model must never be exported for April.
                engine._train_raw(make, pd.Timestamp('2024-08-01'), views, 'часть')
                unchanged = engine._train_raw(make, pd.Timestamp('2024-08-02'), views, 'часть')
                self.assertEqual(unchanged['fit_calls'], 0)
                with patch.object(engine, 'original_train', side_effect=AssertionError('export cannot fit')):
                    exported = engine.model_for_export(make, pd.Timestamp('2024-04-15'), views)
                self.assertEqual(pd.Timestamp(exported['trained_cutoff']), second)
                self.assertIsNone(engine.model_for_export(make, pd.Timestamp('2024-02-01'), views))
                # The missing May predecessor was updated before June, irrespective of target gaps.
                may = engine._cached_raw(make(), pd.Timestamp('2024-05-01'), views, 'часть')
                self.assertIsNotNone(may)
            # Refit is a separate experiment/cache namespace.
            refit = F.plan(dict(cfg, TRAIN_UPDATE='refit'))
            self.assertTrue(refit.key.str.endswith('__refit').all())
            self.assertNotEqual(engine.signature, prepare(root, mode='refit', units=('К-1','К-2'), scenario='pooled')[1]['GROUP_ENGINE'].signature)

    def test_load_role_alignment_and_units(self):
        rows = {'К-1': [dict(role='load', key='расход острого пара за котлом', original='КА-1 Расход острого пара за котлом', units='т/ч')],
                'К-3': [dict(role='load', key='расход острого пара на 4 ст.', original='KA-3 Расход острого пара на 4 ст.', units='т/ч')]}
        aligned, changed = G.align_load_roles(rows)
        self.assertTrue(changed)
        self.assertEqual({v[0]['key'] for v in aligned.values()}, {'нагрузка котла'})
        self.assertIn('за котлом', aligned['К-1'][0]['original'])
        from saved_model_runner import _map
        # Recipient of an exported pooled model may have a differently named steam-flow point.
        raw = pd.DataFrame({'KA-3 Расход острого пара на 4 ст.': [100., 101.]})
        recipient = [dict(role='load', key='расход острого пара на 4 ст.',
                          original='KA-3 Расход острого пара на 4 ст.', units='т/ч')]
        mapped, audit = _map(raw, {'нагрузка котла': 'load'}, 'К-3', recipient, G.canonical_name, {})
        np.testing.assert_equal(mapped['нагрузка котла'].to_numpy(), [100., 101.])
        self.assertEqual(audit.iloc[0]['status'], 'matched')
        rows['К-3'][0].update(key='другой расход пара', units='кг/с')
        with self.assertRaises(ValueError):
            G.align_load_roles(rows)

    def test_old_noncausal_cache_is_not_imported(self):
        oldroot = Path(__file__).resolve().parents[1]/'_archive/fleet_verification_2026-10-19'
        if not oldroot.exists():
            self.skipTest('development-only frozen release-2.0 fixture is absent')
        with tempfile.TemporaryDirectory(prefix='old_cache_reject_') as tmp:
            import shutil
            cfg = F.defaults(); cfg.update(json.loads((oldroot/'smoke_settings.json').read_text(encoding='utf-8')))
            cfg.update(TRAIN_UPDATE='fast', IMPORT_LEGACY_CACHE=True, SHOW_PROGRESS_RESULTS=False,
                       LEGACY_OUTPUT_DIR=tmp)
            previous = Path(tmp)/'pair/local_common__К-1/К-1/hourly'
            previous.mkdir(parents=True)
            shutil.copy2(oldroot/'results/pair/local_common__К-1/К-1/hourly/analogs.joblib', previous/'analogs.joblib')
            job = SimpleNamespace(**F.plan(cfg).iloc[0].to_dict())
            ns = F._prepare(cfg, job)
            self.assertIsNone(ns['GROUP_ENGINE'].compat_signature)
            called = []
            scores = ns['cached']('analogs', lambda: called.append(True) or {'fresh': True},
                                  ns['sig_of'](ns['K'].AnalogCondDet(ns['COND_COLS'])))
            self.assertEqual(called, [True])
            self.assertEqual(scores, {'fresh': True})


if __name__ == '__main__':
    unittest.main()
