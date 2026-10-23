#!/usr/bin/env python3
"""
Все заявки на ремонт одного агрегата из файла заявок -> одна таблица CSV со всеми полями как есть.

В исходном файле каждая строка — снимок за день; в столбце «Оборудование» JSON со списком агрегатов, у каждого —
свои поля и список «Ремонты». Одна заявка повторяется во многих снимках: в результат идёт её последняя версия.

В CSV по строке на заявку:
  * все поля заявки из JSON (вложенные — через точку: «поле.подполе»);
  * поля агрегата из того же снимка (с приставкой «агрегат.»);
  * столбцы самого снимка (с приставкой «снимок.»);
  * сколько снимков с заявкой, первый и последний снимок.

Запуск: python parse_repairs.py
Требуется: pip install pandas
"""
import ast
import json
import re
import sys

import pandas as pd

# ============================ ПАРАМЕТРЫ ============================
SRC = 'Ремонты_НИТЭЦ.csv'          # файл заявок
UNIT = 'К-2'                       # агрегат: «К-2», «КА-2», «K2» — одно и то же
OUT = None                         # None — «заявки_<агрегат>.csv»
# ===================================================================

try:                                                       # консоль Windows: печать кириллицы без ошибки charmap
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ALIASES = {'КА': 'К', 'ТА': 'Т', 'ГА': 'Г'}               # «КА-2» и «К-2» — один агрегат
LAT2CYR = str.maketrans('ABCEHKMOPTX', 'АВСЕНКМОРТХ')


def unit_key(name):
    """Нормализованное имя агрегата: «КА-2», «K 2», «К2» -> «К-2»."""
    if name is None:
        return None
    s = re.sub(r'\s+', ' ', str(name).replace('\xa0', ' ')).strip().upper().translate(LAT2CYR)
    m = re.match(r'^([А-ЯЁ]{1,3})\s*[-/ ]?\s*(\d+)(?=\D|$)', s)
    return f'{ALIASES.get(m.group(1), m.group(1))}-{int(m.group(2))}' if m else s


def read_csv_any(path):
    """CSV с автоопределением кодировки (utf-8 / cp1251) и разделителя."""
    raw = open(path, 'rb').read(1 << 20)
    enc = 'cp1251'
    for e in ('utf-8-sig', 'cp1251'):
        try:
            raw[:len(raw) - 4].decode(e)
            enc = e
            break
        except UnicodeDecodeError:
            pass
    head = raw.decode(enc, errors='ignore').splitlines()[0]
    sep = max([';', ',', '\t', '|'], key=head.count)
    df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('"').lstrip('﻿') for c in df.columns]
    return df, enc, sep


def parse_json(v):
    """JSON из ячейки: обычный, с одинарными кавычками или «дважды закодированный» (строка внутри строки)."""
    for _ in range(3):
        if isinstance(v, (list, dict)) or v is None:
            return v
        if not isinstance(v, str) or not v.strip():
            return None
        try:
            v = json.loads(v)
        except Exception:
            try:
                v = ast.literal_eval(v)
            except Exception:
                return None
    return v if isinstance(v, (list, dict)) else None


def flatten(d, prefix=''):
    """Вложенный словарь -> плоский: {'a': {'b': 1}} -> {'a.b': 1}. Списки — одной строкой JSON."""
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, str) and v.strip()[:1] in '[{':
            v = parse_json(v) or v                     # JSON внутри строки
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def start_of(z):
    """Время начала заявки — только для сортировки строк («24:00 03.08.25» = конец суток)."""
    s = z.get('Начало')
    m = re.match(r'\s*(\d{1,2}):(\d{2})\s+(\d{1,2})\.(\d{1,2})\.(\d{2,4})\s*$', s) if isinstance(s, str) else None
    if m:
        h, mi, d, mo, y = map(int, m.groups())
        try:
            return pd.Timestamp(year=y + (2000 if y < 100 else 0), month=mo, day=d) + pd.Timedelta(hours=h, minutes=mi)
        except ValueError:
            pass
    return pd.to_datetime(z.get('Итоговое начало'), errors='coerce')


def main():
    df, enc, sep = read_csv_any(SRC)
    need = [c for c in ('Дата', 'Оборудование') if c not in df.columns]
    if need:
        sys.exit(f'В файле нет столбцов {need}; есть: {list(df.columns)}')
    snap = pd.to_datetime(df['Дата'], errors='coerce', dayfirst=False)
    if snap.isna().mean() > 0.5:
        snap = pd.to_datetime(df['Дата'], errors='coerce', dayfirst=True)
    df['_снимок'] = snap
    df = df.dropna(subset=['_снимок']).sort_values('_снимок')
    target = unit_key(UNIT)
    print(f'Файл: {SRC} (кодировка {enc}, разделитель {sep!r}); снимков: {len(df):,}, '
          f'{df["_снимок"].min():%Y-%m-%d} — {df["_снимок"].max():%Y-%m-%d}')

    tickets, seen_units, bad_json = {}, {}, 0
    row_cols = [c for c in df.columns if c not in ('Оборудование', '_снимок')]
    for row in df.to_dict('records'):
        items = parse_json(row['Оборудование'])
        if items is None:
            bad_json += 1
            continue
        for it in (items if isinstance(items, list) else [items]):
            if not isinstance(it, dict):
                continue
            key_u = unit_key(it.get('Оборудование'))
            seen_units[key_u] = seen_units.get(key_u, 0) + 1
            if key_u != target:
                continue
            reps = parse_json(it.get('Ремонты')) if isinstance(it.get('Ремонты'), str) else it.get('Ремонты')
            for z in reps or []:
                if not isinstance(z, dict):
                    continue
                num = z.get('Номер заявки') or json.dumps(z, ensure_ascii=False, sort_keys=True)
                cur = tickets.get(num)
                if cur is None:
                    tickets[num] = dict(z=z, it=it, row=row, first=row['_снимок'], last=row['_снимок'], n=1)
                else:
                    cur['n'] += 1
                    if row['_снимок'] >= cur['last']:              # последняя версия заявки
                        cur.update(z=z, it=it, row=row, last=row['_снимок'])
    if bad_json:
        print(f'Не разобран JSON в {bad_json} строках')
    if not tickets:
        found = ', '.join(f'{k} ({v})' for k, v in sorted(seen_units.items(), key=lambda kv: -kv[1])[:30])
        sys.exit(f'Заявок по агрегату «{UNIT}» (= {target}) не найдено. Агрегаты в файле: {found}')

    rows = []
    for t in sorted(tickets.values(), key=lambda t: (pd.isna(start_of(t['z'])), start_of(t['z']) if pd.notna(start_of(t['z'])) else 0)):
        rec = flatten(t['z'])                                                                   # поля заявки
        rec.update(flatten({k: v for k, v in t['it'].items() if k != 'Ремонты'}, 'агрегат.'))   # поля агрегата
        rec.update({f'снимок.{c}': t['row'].get(c) for c in row_cols})                          # столбцы снимка
        rec.update({'снимков с заявкой': t['n'], 'первый снимок': t['first'].date(), 'последний снимок': t['last'].date()})
        rows.append(rec)
    out = pd.DataFrame(rows)
    path = OUT or f'заявки_{target}.csv'
    out.to_csv(path, sep=';', index=False, encoding='utf-8-sig')
    print(f'Агрегат {target}: заявок {len(out)}, столбцов {out.shape[1]} -> {path}')
    print('Столбцы: ' + ', '.join(out.columns))


if __name__ == '__main__':
    main()
