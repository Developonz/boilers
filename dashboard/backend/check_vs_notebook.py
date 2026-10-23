"""
Сверка дашборда с графиком ноутбука 02 перед аварийными ремонтами.

Для последних суток перед каждой аварией (по столбцу «Состояние» в hourly.parquet) сравнивает час за часом:
  ноутбук — отношение «сглаженная ошибка / порог» из результатов ноутбука (anomaly_results/hourly/<метод>.joblib),
  дашборд — то же отношение, посчитанное дашбордом, и индекс здоровья; какой моделью посчитан час
  (итоговая или модель месяца) и каких датчиков модели нет в данных.

Запуск из папки dashboard/backend (настройки — из backend/.env, как у самого дашборда):
    python check_vs_notebook.py
    python check_vs_notebook.py "C:/путь/к/anomaly_results/hourly/Аналоги+.joblib" 72
Второй аргумент — сколько часов перед остановом смотреть (по умолчанию 48).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import main
from k2_model import M_STOP

SMOOTH_H = 6                       # как SMOOTH_H в ноутбуке
FAIL_LABEL = 'Аварийный ремонт'


def notebook_ratio(path: Path, view: str) -> pd.Series:
    if path.suffix == '.parquet':
        data = pd.read_parquet(path)
        if 'ratio' not in data:
            raise SystemExit('В parquet нет ratio; выберите scores/<эксперимент>/<метод>/<котёл>__<вид>.parquet')
        return data['ratio'].rename('ноутбук')
    res = joblib.load(path)['views']
    if view not in res:
        raise SystemExit(f'В {path.name} нет вида «{view}». Есть: {", ".join(res)}')
    s = res[view]['score'].dropna()
    seg = (s.index.to_series().diff() > pd.Timedelta('3h')).cumsum()
    sm = s.groupby(seg).transform(lambda v: v.rolling(SMOOTH_H, min_periods=1).mean())
    return (sm / res[view]['thr'].reindex(sm.index)).rename('ноутбук')


def failures(svc) -> list:
    """Аварийные ремонты: (начало заявки, последний час работы котла перед ней)."""
    path = svc.hourly_path
    lab = pd.read_parquet(path, columns=['TimeValue', 'Состояние'])
    lab.index = pd.DatetimeIndex(pd.to_datetime(lab.pop('TimeValue'))).round('1h')
    lab = lab[~lab.index.duplicated(keep='last')].sort_index()['Состояние'].astype(object)
    is_fail = lab.map(lambda v: isinstance(v, str) and v.strip() == FAIL_LABEL)
    starts = lab.index[is_fail & ~is_fail.shift(fill_value=False)]
    df, _, _ = svc.data()
    mode = svc._mode(df)
    out = []
    for st in starts:
        run = mode[:st]
        run = run[run.ne(M_STOP)]
        if len(run):
            out.append((st, run.index[-1]))
    return out


def main_check(cache: Path, hours: int, tpp_id=None, ka_id=None):
    svc = main.FLEET.select(tpp_id, ka_id).service
    lm = svc.model()
    view = lm.view['name']
    months = svc.months()
    print(f'Модель: {lm.folder}  (обучена до {lm.trained_until}), вид «{view}»')
    print(f'Моделей месяцев: {len(months)}' + (f' ({min(months)} … {max(months)})' if months else
                                                ' — НЕТ: исторические моменты будут недоступны'))
    df, numeric, text = svc.data()
    need_txt = [c for c, _ in lm.bools]
    print(f'Состояний механизмов у модели: {len(need_txt)}, из них есть в данных: {sum(c in text for c in need_txt)}')
    nb = notebook_ratio(cache, view)
    print(f'Результаты ноутбука: {cache} ({nb.index.min():%Y-%m-%d} … {nb.index.max():%Y-%m-%d})\n')

    for start, stop in failures(svc):
        hrs = pd.date_range(stop - pd.Timedelta(hours=hours - 1), stop, freq='h')
        hrs = [h for h in hrs if nb.index.min() <= h]
        if not hrs:
            print(f'Авария {start:%Y-%m-%d}: до начала проверки в ноутбуке — пропуск\n')
            continue
        rows = []
        for h in hrs:
            ev = svc.evaluate(h)
            rows.append({'час': h, 'ноутбук': nb.get(h, np.nan), 'дашборд': ev.ratio_now, 'индекс': ev.health_now,
                         'режим': ev.mode_now, 'модель': ev.lm.folder.name, 'нет датчиков': len(ev.missing)})
        t = pd.DataFrame(rows).set_index('час')
        t['дашборд'] = t['дашборд'].astype(float)
        diff = (t['ноутбук'] - t['дашборд']).abs()
        print(f'=== Авария: заявка {start:%Y-%m-%d %H:%M}, котёл остановлен {stop:%Y-%m-%d %H:%M}; последние {hours} ч работы')
        print(f'    ноутбук: макс. отношение {t["ноутбук"].max():.2f}, часов выше порога {int((t["ноутбук"] > 1).sum())} | '
              f'дашборд: макс. {t["дашборд"].max():.2f}, выше порога {int((t["дашборд"] > 1).sum())}, '
              f'мин. индекс {t["индекс"].min():.0f} | макс. расхождение {diff.max():.3g}')
        print(t.iloc[::max(1, len(t) // 24)].to_string(float_format=lambda x: f'{x:.2f}'))
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Сверка выбранного котла с результатами ноутбука')
    parser.add_argument('cache', type=Path)
    parser.add_argument('hours', type=int, nargs='?', default=168)
    parser.add_argument('--tpp', default=None)
    parser.add_argument('--ka', default=None)
    args = parser.parse_args()
    main_check(args.cache, args.hours, args.tpp, args.ka)
