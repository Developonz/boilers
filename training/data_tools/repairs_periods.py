#!/usr/bin/env python3
"""
Периоды ремонтов по всем котлам из файла заявок — чтобы посмотреть глазами.

Правила те же, что в opc_to_wide.py (функции берутся оттуда, файл должен лежать рядом), поэтому состояние
у заявки здесь такое же, как в метках часовой таблицы:
  ХР                                        -> Резерв
  АР или категория АВ / НО                  -> Аварийный ремонт
  СР, ТР, КР с категорией «Плановый»        -> Плановый ремонт
  СР, ТР, КР с другой категорией            -> Аварийный ремонт
  ВПр, ЗРР, ИСП                             -> Резерв
  Отменённые / отклонённые заявки и заявки без начала в разметку не идут (столбец «учтена» = нет).

Результат (CSV с разделителем «;», открывается в Excel):
  1. ремонты_заявки.csv    — по строке на заявку: котёл, номер, тип, категория, статус, начало, окончание,
                              длительность, состояние, учтена ли и почему нет, содержание работ;
  2. ремонты_периоды.csv   — итоговая разметка по каждому котлу: непрерывные периоды состояния (при пересечении
                              заявок побеждает Аварийный > Плановый > Резерв > Работа) с номером заявки-«победителя»;
                              «Работа» — время внутри периода файла заявок без заявок;
  3. сводка в консоли      — по каждому котлу: сколько заявок и часов каждого состояния.

Запуск: python repairs_periods.py   (или python repairs_periods.py путь/к/файлу_заявок.csv)
Требуется: pandas, numpy, pyarrow (как для opc_to_wide.py).
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

import opc_to_wide as W

# ============================ ПАРАМЕТРЫ ============================
REPAIRS_FILE = 'Ремонты_НИТЭЦ.csv'      # файл заявок (как в opc_to_wide.py)
UNITS = None                            # None — все котлы и агрегаты из файла; или список, например ['К-2', 'К-3']
ONLY_BOILERS = True                     # True — только котлы («К-…»), без турбин и прочего
OUT_DIR = 'ремонты'                     # куда сохранить CSV
# ===================================================================


def read_tickets(path):
    """Все заявки всех агрегатов: последняя версия каждой заявки (файл — снимки по дням) и период, который покрывает файл."""
    enc = W.detect_encoding(path)
    df = pd.read_csv(path, sep=W.detect_sep(path, enc), encoding=enc, dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('"').lstrip('﻿') for c in df.columns]
    miss = [c for c in ('Дата', 'Оборудование') if c not in df.columns]
    if miss:
        sys.exit(f'В файле нет столбцов {miss}; есть: {list(df.columns)}')
    df['_d'] = W.snapshot_dates(df['Дата'])
    bad_d = df['_d'].isna()
    print(f'Строк в файле: {len(df):,}; дата не распознана: {int(bad_d.sum()):,}'
          + (' — примеры: ' + ', '.join(repr(v) for v in df.loc[bad_d, 'Дата'].unique()[:5]) if bad_d.any() else ''))
    df = df.dropna(subset=['_d']).sort_values('_d', kind='stable')
    tickets, seen = {}, Counter()
    diag = {}                                              # год снимка -> что в нём нашлось
    for d, raw in zip(df['_d'], df['Оборудование']):
        g = diag.setdefault(d.year, dict(rows=0, bad_json=0, units=Counter(), reps=0, keys=Counter(), eq_keys=Counter()))
        g['rows'] += 1
        items = W.parse_json(raw)
        if items is None:
            g['bad_json'] += 1
        for it in (items if isinstance(items, list) else [items] if isinstance(items, dict) else []):
            if not isinstance(it, dict):
                continue
            g['eq_keys'].update(it.keys())
            u = W.unit_key(it.get('Оборудование'))
            seen[u] += 1
            g['units'][u] += 1
            reps = it.get('Ремонты')
            for z in (W.parse_json(reps) if isinstance(reps, str) else reps) or []:
                if not isinstance(z, dict):
                    continue
                g['reps'] += 1
                g['keys'].update(z.keys())
                key = (u, z.get('Номер заявки') or json.dumps(z, ensure_ascii=False, sort_keys=True))
                cur = tickets.get(key)
                if cur is None:
                    tickets[key] = dict(z=z, first=d, last=d, n=1)
                else:
                    cur['n'] += 1
                    if d >= cur['last']:
                        cur.update(z=z, last=d)
    cov = (df['_d'].min().normalize(), df['_d'].max().normalize() + pd.Timedelta('1D'))
    print('\nПо годам снимков (строка файла = снимок заявок на дату):')
    all_keys = set().union(*(g['keys'] for g in diag.values())) if diag else set()
    for y in sorted(diag):
        g = diag[y]
        boilers = sorted(u for u in g['units'] if str(u).startswith('К-'))
        miss = sorted(all_keys - set(g['keys'])) if g['reps'] else []
        print(f'  {y}: снимков {g["rows"]:,}, JSON не разобран {g["bad_json"]:,}, записей заявок {g["reps"]:,}, '
              f'котлы: {", ".join(boilers) or "нет"}'
              + (f'; агрегаты (первые 8): {", ".join(str(u) for u, _ in g["units"].most_common(8))}' if not boilers else '')
              + (f'; в заявках нет полей: {", ".join(miss)}' if miss else ''))
    return tickets, seen, cov, len(df)


def ticket_table(tickets, cov):
    """Заявки -> таблица с типом, категорией, состоянием и тем, идёт ли заявка в разметку (как в opc_to_wide.load_repairs)."""
    rows = []
    for (u, _), t in tickets.items():
        z = t['z']
        typ, cat = W.repair_type(z.get('Тип ремонта')), W.repair_category(z.get('Категория ремонта'))
        rows.append({'котёл': u, 'заявка': str(z.get('Номер заявки') or ''), 'тип': typ,
                     'тип полностью': W.TYPES.get(typ, ''), 'категория': cat,
                     'статус': str(z.get('Статус заявки') or '').strip(),
                     'начало': W.ticket_time(z, 'Начало', 'Итоговое начало'),
                     'окончание': W.ticket_time(z, 'Окончание', 'Итоговое окончание'),
                     'состояние': W.state_of(typ, cat),
                     'содержание работ': str(z.get('Содержание работ') or '').strip(),
                     'создана': z.get('Дата создания'), 'снимков': t['n'],
                     'первый снимок': t['first'].date(), 'последний снимок': t['last'].date()})
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    R['начало'] = pd.to_datetime(R['начало'])
    R['окончание'] = pd.to_datetime(R['окончание'])
    R['учтена'], R['примечание'] = True, ''
    R.loc[R.начало.isna(), ['учтена', 'примечание']] = [False, 'нет времени начала']
    bad = R.статус.str.contains(W.EXCLUDE_STATUS, case=False, regex=True, na=False)
    R.loc[bad, ['учтена', 'примечание']] = [False, 'отменена / отклонена']
    R['конец в разметке'] = R.окончание.fillna(cov[1])
    R.loc[R.окончание.isna() & R.учтена, 'примечание'] = 'открыта: до конца файла заявок'
    back = R.учтена & (R['конец в разметке'] <= R.начало)
    R.loc[back, ['учтена', 'примечание']] = [False, 'окончание не позже начала']
    R['часов'] = ((R['конец в разметке'] - R.начало) / pd.Timedelta('1h')).round(1)
    R.loc[~R.учтена, 'часов'] = None
    return R.sort_values(['котёл', 'начало']).reset_index(drop=True)


def state_periods(R, cov):
    """Итоговая разметка котла: непрерывные периоды состояния; при пересечении — более важное состояние."""
    out = []
    prio = {s: i for i, s in enumerate(W.STATES)}
    for u, g in R[R.учтена].groupby('котёл'):
        g = g.assign(_p=g.состояние.map(prio))
        bounds = sorted(set([cov[0], cov[1]] + list(g.начало) + list(g['конец в разметке'])))
        bounds = [b for b in bounds if cov[0] <= b <= cov[1]]
        segs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            on = g[(g.начало <= a) & (g['конец в разметке'] >= b)]
            if len(on):
                w = on.sort_values(['_p', 'начало']).iloc[-1]
                st, tk, typ, cat = w.состояние, w.заявка, w.тип, w.категория
            else:
                st, tk, typ, cat = W.ST_WORK, '', '', ''
            if segs and segs[-1][2:] == [st, tk, typ, cat]:
                segs[-1][1] = b
            else:
                segs.append([a, b, st, tk, typ, cat])
        for a, b, st, tk, typ, cat in segs:
            out.append({'котёл': u, 'с': a, 'по': b, 'часов': round((b - a) / pd.Timedelta('1h'), 1),
                        'состояние': st, 'заявка': tk, 'тип': typ, 'категория': cat})
    return pd.DataFrame(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else REPAIRS_FILE
    tickets, seen, cov, n_snap = read_tickets(path)
    print(f'Файл: {path}; снимков {n_snap:,}, период {cov[0]:%Y-%m-%d} — {cov[1] - pd.Timedelta("1D"):%Y-%m-%d}; '
          f'заявок всего {len(tickets)}')
    R = ticket_table(tickets, cov)
    if R.empty:
        sys.exit('Заявок не найдено.')
    print('\nПервое и последнее событие по заявкам (начало заявки) для каждого агрегата в файле:')
    for u, g in R[R.начало.notna()].sort_values('начало').groupby('котёл'):
        a, b = g.iloc[0], g.iloc[-1]
        note = '' if a.учтена else f' [не учтена: {a.примечание}]'
        print(f'  {u:<8} первое: {a.начало:%d.%m.%Y %H:%M}  №{a.заявка} {a.тип} {a.категория or ""} -> {a.состояние}{note};'
              f'  последнее: {b.начало:%d.%m.%Y %H:%M}  №{b.заявка} {b.тип};  заявок {len(g)}')
    no_start = R[R.начало.isna()].котёл.value_counts()
    if len(no_start):
        print('  заявки без времени начала: ' + ', '.join(f'{u} — {n}' for u, n in no_start.items()))

    units = sorted(R.котёл.dropna().unique())
    if UNITS:
        want = {W.unit_key(u) for u in UNITS}
        units = [u for u in units if u in want]
    elif ONLY_BOILERS:
        units = [u for u in units if str(u).startswith('К-')]
    R = R[R.котёл.isin(units)].reset_index(drop=True)
    P = state_periods(R, cov)

    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ['котёл', 'заявка', 'тип', 'тип полностью', 'категория', 'состояние', 'статус', 'начало', 'окончание', 'часов',
            'учтена', 'примечание', 'содержание работ', 'создана', 'снимков', 'первый снимок', 'последний снимок']
    f1, f2 = os.path.join(OUT_DIR, 'ремонты_заявки.csv'), os.path.join(OUT_DIR, 'ремонты_периоды.csv')
    R[cols].assign(учтена=R.учтена.map({True: 'да', False: 'нет'})).to_csv(f1, sep=';', index=False, encoding='utf-8-sig')
    P.to_csv(f2, sep=';', index=False, encoding='utf-8-sig')

    print(f'\nКотлов: {len(units)} ({", ".join(units)})')
    for u in units:
        r, p = R[R.котёл == u], P[P.котёл == u]
        by_year = r.начало.dt.year.value_counts().sort_index()
        print(f'\n=== {u}: заявок {len(r)}, учтено {int(r.учтена.sum())}; по году начала: '
              + ', '.join(f'{int(y)} — {n}' for y, n in by_year.items())
              + (f', без начала — {int(r.начало.isna().sum())}' if r.начало.isna().any() else ''))
        print('  заявки по типу и состоянию:')
        t = r[r.учтена].groupby(['состояние', 'тип', 'категория']).size()
        for (st, typ, cat), n in t.items():
            print(f'    {st:<17} {typ:<4} {cat or "—":<12} {n}')
        skipped = r[~r.учтена].примечание.value_counts()
        if len(skipped):
            print('  не учтены: ' + ', '.join(f'{k} — {v}' for k, v in skipped.items()))
        h = p.groupby('состояние').часов.sum()
        print('  часов по итоговой разметке: ' + ', '.join(f'{s} {h.get(s, 0):,.0f}' for s in W.STATES[1:] if s in h))
        fails = p[p.состояние.eq(W.ST_FAIL)]
        if len(fails):
            print('  периоды аварийного ремонта:')
            for x in fails.itertuples():
                print(f'    {x.с:%d.%m.%Y %H:%M} — {x.по:%d.%m.%Y %H:%M} ({x.часов:,.0f} ч)  {x.тип} {x.категория} №{x.заявка}')
    print(f'\nСохранено: {f1} (заявки), {f2} (периоды состояний)')


if __name__ == '__main__':
    main()
