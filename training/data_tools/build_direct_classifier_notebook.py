"""Portable notebook: sequential cached experiments, multiple horizons and folds."""
from pathlib import Path
from textwrap import dedent
import json
root=Path(__file__).resolve().parents[1];cells=[]
def md(s,id):cells.append(dict(cell_type='markdown',id=id,metadata={},source=dedent(s).strip()+'\n'))
def code(s,id,hidden=False):cells.append(dict(cell_type='code',id=id,metadata={'jupyter':{'source_hidden':True}} if hidden else {},execution_count=None,outputs=[],source=dedent(s).strip()+'\n'))
md('''
# Классификатор по датчикам: сравнение способов и проверка по времени

Выберите списки способов, признаков и горизонтов в пункте 1, затем выполните ячейки
сверху вниз. Регрессоры не нужны. Балансировка по умолчанию отключена.

**Память:** признаки готовятся отдельно по котлам и сохраняются в кеш на диске.
Нет общей матрицы для всех типов котлов и всех горизонтов. В обучении одновременно
находится одна матрица одного эксперимента и одна модель. Меняется горизонт —
меняются метки, а не создаётся ещё одна копия признаков. После эксперимента память освобождается.

**Проверка:** следующие друг за другом тестовые периоды с фиксированными общими датами.
Validation расширяется назад до достаточного числа прошлых аварий; train остаётся раньше неё.
Для всех наборов признаков одного сравнения даты одинаковы. Test без аварий не исключается.
Если прошлых событий всё равно мало, видны состав выборок и явная причина.
Неудачный подбор F-порога не подменяется на 0.5; 0.5 остаётся фиксированным baseline.

**Основной вывод — модель в целом:** classification report и confusion matrix
по всей последовательной тестовой истории, общий график и непрерывный расчёт уведомлений.
Отдельные версии находятся в закрытой диагностике. AUTO выбирает набор признаков
по прошлой validation и проверяется на будущем test.

Четыре основных порога сохранены. В скрытых подпунктах — macro-F для обоих классов
и подборы по устойчивым эпизодам с ограничениями ложных тревог. Все оптимумы
выбираются только на validation. Итоговые test-метрики не используются для подбора.

Функции обучения включены в этот файл. Для экспорта на дашборд обновите также `training/lib`
из архива: там находятся `classifier_export.py` и переносимая реализация инференса.
Можно положить рядом прежний `05_Классификатор.ipynb`: из него автоматически
читается DATA_SETTINGS с вашими путями. Код и результаты из другого ноутбука не выполняются.
''','direct-intro')
code('''
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from copy import deepcopy
import ast, json, os, sys, gc
import numpy as np
import pandas as pd
from IPython.display import display, HTML

TRAINING_DIR = next((p.resolve() for p in (Path.cwd(), Path.cwd()/"training")
                     if (p/"lib/monitoring_v3").is_dir()), None)
if TRAINING_DIR is None:
    raise FileNotFoundError("Откройте ноутбук из training или корня проекта.")
os.chdir(TRAINING_DIR)
for p in (TRAINING_DIR, TRAINING_DIR/"lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from notebook_reports import notebook_config
from monitoring_v3.data import load_sources
print("Папка проекта:", TRAINING_DIR)
''','direct-bootstrap')
md('''
Следующие две свёрнутые ячейки содержат функции. Их нужно выполнить, редактировать не требуется.
''','direct-helper-description')
code((root/'lib/direct_risk.py').read_text(encoding='utf-8'),'direct-core',True)
code((root/'lib/direct_experiments.py').read_text(encoding='utf-8'),'direct-suite',True)
md('''
## 1. Все настройки

- `TRAINING_METHODS`: `unit` — отдельный котёл, `group` — общая модель котлов одного
  типа из CFG.groups. Для группы используется пересечение общих признаков с одинаковыми
  физическими единицами. Однокотловая группа не повторяет unit, если выбран и unit.
- `FEATURE_VARIANTS`: `numeric` — числовые каналы и их динамика; `numeric_states` —
  плюс текущие состояния и числа переключений, **без длительности состояний и возраста
  пуска**; `full` — все согласованные признаки, включая эти длительности. Идентификатор
  котла сохраняется для групповых моделей во всех вариантах.
- `HORIZONS`: список, например `[24, 72, 168]`. По умолчанию `[168]`, чтобы первый запуск
  не умножал число обучений втрое. Матрицы не копируются по горизонтам, но время расчёта увеличивается.
- `ROLLING`: число непересекающихся следующих test-периодов, длительности validation/test
  в сутках и необязательный общий конец. Validation начинается с 90 суток и при необходимости
  расширяется назад шагом validation_step_days, максимум до max_validation_days.
  По умолчанию нужны 2 независимые аварии в train и 2 в validation, а также 2 недельные
  группы фона в validation. Эти минимумы настраиваются явно. Выбор дат не использует test.
  Между train/validation/test
  учитываются 24-часовые окна и доступность исходов; одна авария не переходит в обе части.
- `REPORT_FOLDS="none"`: основной общий отчёт без отдельных картинок фолдов.
  `"latest"`, `"all"` или список `['fold_01', ...]` добавляют закрытую диагностику версий.
  Все тестовые часы входят в общий результат независимо от этой настройки.
- `MAX_FEATURE_MATRIX_MB`: ограничение оценки активной матрицы, не всей памяти Python;
  `None` снимает ограничение. `OPTIONS.threads` ограничивает CPU CatBoost.
- `alarm_selection=True` добавляет два подбора: максимум обнаружения аварий и максимум
  событийного F1 при бюджете. `alarm_persist_hours=3` подтверждает превышение после 3 часов;
  `alarm_merge_hours=12` подавляет близкие повторные уведомления. Ограничения по умолчанию:
  не больше 2 ложных уведомлений / 720 фоновых часов и не больше 5% фона в тревоге.
  При отсутствии полезного решения может быть выбран вариант без тревог; это видно в отчёте.
- `macro_thresholds=True`: дополнительные F1/F2/F0.5 как среднее F двух классов с равным
  весом. Обычный F положительного класса тоже штрафует FP, macro дополнительно учитывает класс 0.

Числовой канал: текущее значение, лаги 12/24 ч, min/max/mean/std за 24 ч, тренды 6/24 ч,
изменения 1/6 ч. Категория: значение, наблюдаемая длительность и переключения 6/24 ч.
Временные признаки используют только прошлое и не переносятся через разрывы рабочего участка.
''','direct-settings-description')
code('''
SOURCE_NOTEBOOK = "02_Обучение.ipynb"
SOURCE_DATA_NOTEBOOK = "05_Классификатор.ipynb"
DATA_SETTINGS = {
    # "groups": {"БКЗ-500": ["К-5", "К-6", "К-7"]},
    # "data_paths": {"К-5": "D:/private/К-5/hourly.parquet"},
    # "run_thresholds": {"К-5": 140.0},
}
OUTPUT = "runs/direct_classifier_comparison"
TRAINING_METHODS = ["unit", "group"]
FEATURE_VARIANTS = ["numeric", "numeric_states", "full"]
HORIZONS = [168]  # Например [24, 72, 168]
ROLLING = {
    "folds": 3, "validation_days": 90, "test_days": 90, "end": None,
    "adaptive_validation": True, "max_validation_days": 730, "validation_step_days": 30,
    "min_train_events": 2, "min_validation_events": 2, "min_validation_negative_groups": 2,
}
REPORT_FOLDS = "none"  # общий отчёт; latest/all — дополнительная диагностика
MAX_FEATURE_MATRIX_MB = 2048
PREVIEW_UNIT = None  # None — первый котёл, например "К-5"

OPTIONS = direct_options({
    "window_hours": 24, "lags_hours": [12, 24], "trends_hours": [6, 24],
    "delta_hours": [1, 6], "switch_hours": [6, 24],
    "include_since_start": True, "include_startup": True,
    "minimum_numeric_coverage": 0.7, "min_lead_hours": 0,
    "train_step_hours": 6, "iterations": 400, "depth": 4,
    "learning_rate": 0.05, "early_stopping_rounds": 50, "threads": 2,
    "seed": 42, "balance_train": False, "event_weights": False,
    "threshold_metric": "f2", "minimum_precision": None,
    "macro_thresholds": True, "alarm_selection": True,
    "alarm_persist_hours": 3, "alarm_merge_hours": 12,
    "alarm_budget": 2.0, "alarm_background_fraction": 0.05,
    "alarm_candidates": 61, "restore": True,
})

def previous_settings(path):
    result = {}
    if not path.exists():
        return result
    book = json.loads(path.read_text(encoding="utf-8-sig"))
    for cell in book.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in ("SETTINGS", "DATA_SETTINGS", "SENSOR_RULES"):
                    try:
                        result[target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        print(target.id + " содержит выражения; задайте нужные пути явно в DATA_SETTINGS.")
    return result

PREVIOUS = previous_settings(TRAINING_DIR/SOURCE_NOTEBOOK)
OLD_DATA = previous_settings(TRAINING_DIR/SOURCE_DATA_NOTEBOOK)
CFG = notebook_config({
    **PREVIOUS.get("SETTINGS", {}), **OLD_DATA.get("DATA_SETTINGS", {}),
    **DATA_SETTINGS, "output": OUTPUT,
    "models": ["CatBoost"], "feature_sets": ["conditions"], "scenarios": ["local"],
}, PREVIOUS.get("SENSOR_RULES", {}))
OUT = Path(CFG["output"]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
display(HTML(direct_table_html("Настройки", pd.DataFrame({
    "параметр": list(OPTIONS), "значение": [str(v) for v in OPTIONS.values()]
}))))
print("Группы:", CFG["groups"])
print("Пути из прежнего 05:", bool(OLD_DATA.get("DATA_SETTINGS")))
print("Результаты:", OUT)
''','direct-settings')
md('''
## 2. Источники и режимы

Здесь читаются исходные данные и существующая карта каналов. Разметка ремонта
не подаётся во входы. Неизвестные остановы и неполное будущее не становятся нормой.
Начальный опорный период автоматической подготовки не используется как будто пороги
режимов уже были известны. При ручных порогах он может не требоваться.
''','direct-sources-description')
code('''
CONTEXTS, ISSUES = load_sources(CFG, OUT/"audit", strict=False)
if ISSUES:
    display(HTML(direct_table_html("Проблемы источников", pd.DataFrame(ISSUES), opened=True)))
    raise ValueError("Исправьте DATA_SETTINGS и повторите подготовку.")
if not CONTEXTS:
    raise ValueError("Нет доступных котлов.")
trim_direct_contexts(CONTEXTS)
print("Освобождены вспомогательные матрицы регрессоров; исходные данные и режимы сохранены.")
SOURCE_SUMMARY = pd.DataFrame([
    {"unit": u, "rows": len(c["frame"]), "first": c["frame"].index.min(),
     "last": c["frame"].index.max(), "load": c["load"], "run_threshold": c["threshold"]}
    for u, c in CONTEXTS.items()
])
display(HTML(direct_table_html("Источники", SOURCE_SUMMARY, opened=True)))
for u, c in CONTEXTS.items():
    display(HTML(direct_table_html(u + " · реестр остановов", c["events"])))
''','direct-sources')
md('''
## 3. Кеш признаков по котлам

Одновременно строится только один котёл. В файлах признаков хранятся пригодные часы
и float32; строки стоящего котла с тысячами пустых полей не размножаются.
Метки горизонтов считаются отдельно при обучении и кешируются без повторного построения признаков.
Большие таблицы не объединяются по всем котлам; ниже показана сводка и подробности одного котла.
''','direct-cache-description')
code('''
CACHE, CACHE_SUMMARY = prepare_direct_cache(CONTEXTS, CFG, OPTIONS, OUT/"feature_cache")
CACHE_SUMMARY.to_csv(OUT/"feature_cache_summary.csv", index=False, encoding="utf-8-sig")
display(HTML(direct_table_html("Размеры отдельных матриц (не хранятся вместе в памяти)", CACHE_SUMMARY, opened=True)))
unit = PREVIEW_UNIT or next(iter(CACHE))
folder = Path(CACHE[unit]["folder"])
profile = pd.read_csv(folder/"profile.csv")
display(HTML(direct_table_html(unit + " · признаки, описания, типы и пропуски", profile)))
preview = pd.read_parquet(folder/"preview.parquet")
display(HTML(direct_table_html(unit + " · пример пяти моментов", preview.T.reset_index(names="feature"))))
del profile, preview
gc.collect()
''','direct-cache')
md('''
## 4. План запусков и общие временные границы

Число попыток равно числу семейств × наборов признаков × горизонтов × периодов.
Времени потребуется больше, чем одной модели; память не умножается на число запусков.
Завершённые совместимые модели восстанавливаются. При прерывании готовые результаты
остаются на диске, пункт 7 можно выполнять без возврата всей функции обучения.
''','direct-plan-description')
code('''
FAMILIES = experiment_families(CONTEXTS, CFG, TRAINING_METHODS)
FOLDS = rolling_direct_folds(CONTEXTS, CFG, ROLLING)
PLAN = pd.DataFrame([{
    "method": f["method"], "name": f["name"], "units": ", ".join(f["units"]),
    "variants": len(FEATURE_VARIANTS), "horizons": len(HORIZONS), "folds": len(FOLDS),
} for f in FAMILIES])
print("Всего попыток:", len(FAMILIES)*len(FEATURE_VARIANTS)*len(HORIZONS)*len(FOLDS))
print("Однокотловые group не дублируют unit.")
display(HTML(direct_table_html("План", PLAN, opened=True)))
display(HTML(direct_table_html("Диагностика: даты тестов и начальная validation", pd.DataFrame(FOLDS))))
''','direct-plan')
md('''
## 5. Последовательное обучение и проверка

Во всех частях естественный баланс; train прорежен одинаковым шагом времени.
Ни один тест не используется в подборе порога. В отдельном периоде с одним классом
AP/ROC AUC не определены, но ошибки классификации и ложные тревоги учитываются.
Если у короткой истории нет наблюдений в позднем периоде, это явный статус;
истории остальных котлов не обрезаются по самому короткому.

Приоритетные четыре порога — по баллам. Дополнительные macro-варианты тоже по баллам.
Дополнительные бюджетные варианты оцениваются **после подтверждения и подавления повторов**.
Тревога, начавшаяся задолго до предаварийного окна, не получает автоматический зачёт
обнаружения только за то, что продолжала гореть. Неизвестные исходы учитываются отдельно.
''','direct-train-description')
code('''
RUN_INDEX, FOLD_METRICS = run_direct_suite(
    CONTEXTS, CFG, OPTIONS, CACHE,
    TRAINING_METHODS, FEATURE_VARIANTS, HORIZONS, ROLLING,
    OUT, max_mb=MAX_FEATURE_MATRIX_MB,
)
print("Завершено моделей:", int(RUN_INDEX.status.eq("complete").sum()))
print("Все статусы и причины:")
display(HTML(direct_table_html("Журнал запусков", RUN_INDEX)))
''','direct-train')
md('''
## 6. Сохранённый инференс без повторной матрицы

Проверка использует сохранённый последний пригодный момент котла и последнюю модель
каждого эксперимента. Модель не применяется к датам раньше её доступности.
Новые реальные данные требуют повторной подготовки исходной истории тем же контрактом.
''','direct-inference-description')
code('''
import joblib
index_path = OUT/"experiment_index.csv"
CURRENT_INDEX = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
completed = CURRENT_INDEX.loc[CURRENT_INDEX.status.eq("complete")] if "status" in CURRENT_INDEX else pd.DataFrame()
latest = (completed.sort_values("test_end").groupby("experiment", sort=False).tail(1)
          if not completed.empty else pd.DataFrame())
choices_path = OUT/"validation_choices.csv"
if choices_path.exists():
    choices = pd.read_csv(choices_path)
    if not choices.empty:
        latest = choices.sort_values("validation_end").groupby(["method", "name", "horizon_hours"], sort=False).tail(1)
PREDICTIONS_NOW = []
for item in latest.itertuples():
    package = joblib.load(Path(item.folder)/"classifier.joblib")
    for unit in package["experiment"]["units"]:
        last = pd.read_parquet(Path(CACHE[unit]["folder"])/"tail.parquet")
        if last.empty or last.index[-1] < pd.Timestamp(package["available_at"]):
            continue
        result = direct_predict(package, last)
        result["unit"] = unit
        result["experiment"] = item.experiment
        PREDICTIONS_NOW.append(result.reset_index())
    del package
    gc.collect()
LATEST_PREDICTIONS = pd.concat(PREDICTIONS_NOW, ignore_index=True) if PREDICTIONS_NOW else pd.DataFrame()
display(HTML(direct_table_html("Последние доступные оценки по экспериментам", LATEST_PREDICTIONS)))
''','direct-inference')
md('''
## 7. Все отчёты и сравнение

Можно запускать отдельно после прерывания: читается индекс уже сохранённых моделей,
обучение не запускается. Под стрелками модели/котла показано качество за всю тестовую
историю, а не отдельные отчёты по фолдам. Большие таблицы и ненулевая важность свёрнуты.
PNG встроены и не зависят от matplotlib backend.

В сводке Precision/Recall/F считаются по суммарным ошибкам непересекающихся test-периодов.
ROC AUC и AP показаны как среднее/min/max **отдельных** периодов: баллы разных
моделей с разными шкалами не склеиваются в один якобы общий AUC. Число доступных
версий видно в `versions`, число версий с определимым AUC/AP — в `rankable_versions`.
Средний случайный AP сравнивается на тех же определимых периодах. Для каждого горизонта
своя целевая метка и собственная доля положительных, поэтому сравнивайте также lift к baseline.

Среди baseline есть случайное угадывание, постоянный прогноз по train и **«всегда тревога»**.
Событийные показатели по отдельным периодам могут иметь неполное предаварийное окно
на границе: смотрите покрытие в отчёте. Для бюджетных вариантов дополнительно
считается единая последовательная история: состояние подтверждения/подавления
повторов не сбрасывается на границе смены модели, а порог меняется на выбранный
по прошлому validation. Эти поля имеют префикс `rolling_`.
Это диагностическая проверка; окончательная
оценка после разработки требует ещё не использованного периода.
''','direct-results-description')
code('''
index_path = OUT/"experiment_index.csv"
if not index_path.exists():
    raise FileNotFoundError("Пока нет сохранённых запусков: выполните пункт 5.")
RUN_INDEX = pd.read_csv(index_path)
ROLLING_SUMMARY, ALL_FOLD_METRICS = render_direct_suite(
    RUN_INDEX, CONTEXTS, OUT, report_folds=REPORT_FOLDS,
)
''','direct-results-display')
md('''
## 8. Экспорт классификатора для дашборда

Экспорт не обучает модели: он упаковывает уже сохранённые версии, их параметры подготовки и пороги.
Копируйте на сервер целиком каталог семьи/горизонта, включая index.json и все версии.
AUTO использует выбор набора признаков по прошлой validation, а не по test.
Unit/group выбирается явно; для общей модели укажите в boilers.json имя котла из экспорта.

Новая версия формирования признаков не использует _Тип из заявок для расчёта окон.
Если веса обучались старой версией, сначала выполните обновлённые разделы 2–5:
пересчёт этого классификатора необходим, прежние результаты останутся в старом кеше.
Подготовка и пороги после экспорта фиксированы; сервер не запускает обучение.
''','direct-export-description')
code('''
from classifier_export import export_direct_suite

EXPORT_METHODS = list(TRAINING_METHODS)  # ["unit"], ["group"] либо оба
EXPORT_VARIANT = "AUTO"      # либо "numeric", "numeric_states", "full"
EXPORT_HORIZONS = [168]      # должны быть обучены выше
EXPORT_DIR = OUT / "dashboard_export"

_exports, _export_errors = [], []
for _method in EXPORT_METHODS:
    try:
        _exports.append(export_direct_suite(OUT, output=EXPORT_DIR,
            method=_method, variant=EXPORT_VARIANT, horizons=EXPORT_HORIZONS))
    except ValueError as _error:
        _export_errors.append({"method": _method, "status": "not_exported", "reason": str(_error)})
if _export_errors:
    display(pd.DataFrame(_export_errors))
EXPORTS = pd.concat(_exports, ignore_index=True) if _exports else pd.DataFrame()
display(EXPORTS)
if EXPORTS.empty:
    raise ValueError("Нет готового экспорта. Смотрите причины выше; при старом контракте повторите обучение в обновлённом 05.")
print("На сервер копируется каталог из model_dir целиком (с версиями и index.json).")
''','direct-export')
book=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (ipykernel)',language='python',name='python3'),language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
target=root/'05_Классификатор_без_регрессора.ipynb'
target.write_text(json.dumps(book,ensure_ascii=False,indent=1),encoding='utf-8')
print(target)
