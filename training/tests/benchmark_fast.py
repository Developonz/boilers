"""Optional synthetic timing: python tests/benchmark_fast.py --output timing.json.

225 targets, 6500 selected training hours; measures calibration/model update only,
not the full fleet run or forecasting the evaluation month. No telemetry required.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from test_fast import prepare
from test_fleet import generated_data
import legacy_fleet as F
import numpy as np
import pandas as pd
import openpyxl


def benchmark(output):
    rows = []
    with tempfile.TemporaryDirectory(prefix='legacy_timing_') as tmp:
        root = Path(tmp); generated_data(root, units=('К-1',), periods=10000)
        path = root/'К-1/hourly.parquet'; frame = pd.read_parquet(path)
        rng = np.random.default_rng(42)
        load = frame['KA-1 Расход перегретого пара'].to_numpy()
        extra = {f'KA-1 Датчик {j}': 10 + .1*load + rng.normal(0, 1, len(frame)) for j in range(222)}
        pd.concat([frame, pd.DataFrame(extra)], axis=1).to_parquet(path, index=False)
        wb = openpyxl.load_workbook(root/'sensor_roles.xlsx')
        for col in extra:
            wb.active.append([col, 'проверять', col, None])
        wb.save(root/'sensor_roles.xlsx'); wb.close()
        cfg, _ = prepare(root)
        cfg.update(TRAIN_HOURS=6500, FIRST_TEST='2024-11-01')
        ns = F._prepare(cfg, SimpleNamespace(**F.plan(cfg).iloc[0].to_dict()))
        engine = ns['GROUP_ENGINE']
        for name in ('AAKR', 'MSET'):
            make = lambda: ns['KernelDet'](ns['COND_COLS'], name.lower())
            views = F._views(name)
            first, second = pd.Timestamp('2024-11-01'), pd.Timestamp('2024-12-01')
            engine._train_raw(make, first, views, 'часть')
            start = time.perf_counter()
            fast = engine._train_raw(make, second, views, 'часть')
            fast_s = time.perf_counter() - start
            pool = engine._pool_namespace(False)
            start = time.perf_counter()
            old = engine.original_train
            import legacy_group as G
            refit = G._clone(old, pool)(make, second, views, 'часть')
            refit_s = time.perf_counter() - start
            row = dict(method=name, sensors=len(engine.targets), training_hours=6500,
                       new_calibration_rows=fast['update_error_rows'], fast_seconds=fast_s,
                       refit_seconds=refit_s, speedup=refit_s/fast_s,
                       threshold_fast={v: float(cal[1]) for v, cal in fast['cal'].items()},
                       threshold_refit={v: float(cal[1]) for v, cal in refit['cal'].items()})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            Path(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='timing.json')
    benchmark(parser.parse_args().output)
