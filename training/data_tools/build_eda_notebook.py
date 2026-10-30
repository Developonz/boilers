"""Build read-only 01 using the actual monitoring_v3 preparation rules."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def cell(kind,source):
    source=source.strip()+'\n'
    value=dict(cell_type=kind,id=hashlib.sha256((kind+source).encode()).hexdigest()[:12],metadata={},source=source.splitlines(keepends=True))
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value


def build():
    cells=[]
    def md(s):cells.append(cell('markdown',s))
    def code(s):cells.append(cell('code',s))
    md('''# 01 · Анализ данных перед обучением

Цель — проверить исходную историю, режимы, остановы, качество измерений и объём пригодных данных **до обучения**.
Модели и классификаторы здесь не обучаются. Используется та же подготовка `monitoring_v3`, что в 02 и 05:
порог нагрузки, токи приводов, подтверждение остановов, очистка и общий реестр событий.

Выполняйте ячейки сверху вниз. Таблицы pandas и графики matplotlib выводятся прямо в ноутбуке;
блоки котлов и большие таблицы открываются стрелками. Картинки встроены как PNG и дополнительно сохранены.
Сохраните ноутбук после выполнения, чтобы выводы остались в файле.

Пути и группы по умолчанию читаются из **сохранённых** SETTINGS в 02; код 02 не исполняется.
Если значения там содержат вычисления, задайте SOURCE_NOTEBOOK=None и параметры непосредственно в DATA_SETTINGS.
Старая версия 01 лежит в `archive/2026-10-28/01_EDA.ipynb`.
''')
    code('''import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import os, sys
import pandas as pd

_candidates = [Path.cwd(), Path.cwd()/"training"]
TRAINING_DIR = next((p.resolve() for p in _candidates if (p/"lib/monitoring_v3").is_dir()), None)
if TRAINING_DIR is None:
    raise FileNotFoundError("Откройте ноутбук из корня проекта или training, рядом должна быть обновлённая папка lib.")
os.chdir(TRAINING_DIR)
for _path in (TRAINING_DIR, TRAINING_DIR/"lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from notebook_reports import notebook_config
from monitoring_v3.config import load
from monitoring_v3.data import load_sources
import detector_notebook as DR
import preparation_notebook as E
print("Python:", sys.executable)
print("Рабочая папка:", TRAINING_DIR)
''')
    md('''## 1. Настройки

`SOURCE_NOTEBOOK` — откуда взять SETTINGS и SENSOR_RULES. `None` — использовать incremental_settings.json.
`DATA_SETTINGS` переопределяет пути, группы, run_thresholds и остальные параметры подготовки.
`UNITS=None` — все котлы выбранных групп; список позволяет оставить нужные котлы.
`SENSOR_RULES_OVERRIDE` дополняет правила из 02 по отдельным датчикам.
`EDA_OUTPUT` — отдельная папка отчётов, веса и результаты обучения туда не переносятся.

Параметры EDA:

* `max_stop_plots`: сколько отдельных остановов показать; сначала без заявок, затем самые долгие. Все остановы остаются в таблице.
* `stop_before_hours`, `stop_after_hours`: границы графика вокруг останова/пуска.
* `pre_stop_hours`: окно разбора измерений перед физическим остановом.
* `max_event_plots`: ограничение числа картинок предыстории; таблицы содержат все события.
* `max_analysis_rows`: предел строк для корреляций; выбираются равномерно по истории, без изменения исходных данных.
* `max_heatmap_sensors`: число каналов на корреляционной карте; таблица сильных пар учитывает все проверяемые каналы.
* `correlation_threshold`: модуль Spearman для таблицы сильных связей.
* `balance_sensors`: необязательные проверенные пары потоков; единицы должны совпадать.
* `balance_threshold_percent`, `balance_persist_hours`: граница и длительность эпизода расхождения вода–пар.

Необязательный CLASSIFIER_PREVIEW считает исходы как в 05 без большой матрицы признаков.
Подсчёт всех горизонтов на длинной истории может занимать время; по умолчанию выключен.
Если меняли правила допуска/пуска/горизонт в 05, задайте те же значения здесь.
''')
    code('''SOURCE_NOTEBOOK = "02_Обучение.ipynb"
DATA_SETTINGS = {
    # "groups": {"БКЗ-500": ["К-5", "К-6", "К-7"]},
    # "data_paths": {"К-5": "D:/private/К-5/hourly.parquet"},
    # "run_thresholds": {"К-5": 140},
}
UNITS = None  # например ["К-5"]
SENSOR_RULES_OVERRIDE = {"defaults": {}, "sensors": {}}
EDA_OUTPUT = "runs/eda_v3"
EDA = {
    "max_stop_plots": 6,
    "stop_before_hours": 48, "stop_after_hours": 24,
    "pre_stop_hours": 168, "max_event_plots": 6,
    "max_analysis_rows": 12000, "max_heatmap_sensors": 30,
    "correlation_threshold": 0.95,
    "balance_sensors": {},  # {"К-5": {"water": "точное имя", "steam": "точное имя"}}
    "balance_threshold_percent": 3.0, "balance_persist_hours": 6,
}
CLASSIFIER_PREVIEW = {
    "enabled": False, "horizons": [24, 168],
    "options": {"include_startup": True, "minimum_numeric_coverage": 0.7, "min_lead_hours": 0},
}
''')
    code('''_saved = E.saved_settings(SOURCE_NOTEBOOK) if SOURCE_NOTEBOOK else {}
_settings = {**_saved.get("SETTINGS", {}), **DATA_SETTINGS}
_rules = _saved.get("SENSOR_RULES", {})
_rules = {
    "defaults": {**_rules.get("defaults", {}), **SENSOR_RULES_OVERRIDE.get("defaults", {})},
    "sensors": {name: {**_rules.get("sensors", {}).get(name, {}),
                       **SENSOR_RULES_OVERRIDE.get("sensors", {}).get(name, {})}
                for name in set(_rules.get("sensors", {})) | set(SENSOR_RULES_OVERRIDE.get("sensors", {}))},
}
CFG = notebook_config(_settings, _rules)
if UNITS is not None:
    _groups = {g: [u for u in units if u in UNITS] for g, units in CFG["groups"].items()}
    _groups = {g: units for g, units in _groups.items() if units}
    if set(UNITS)-{u for units in _groups.values() for u in units}:
        raise ValueError("В UNITS есть котлы, отсутствующие в группах SETTINGS.")
    CFG = load(overrides={**CFG, "groups": _groups})
if EDA["max_analysis_rows"] < 20 or not 0 < EDA["correlation_threshold"] <= 1:
    raise ValueError("Проверьте max_analysis_rows и correlation_threshold.")
if EDA["balance_threshold_percent"] <= 0 or EDA["balance_persist_hours"] <= 0:
    raise ValueError("Порог и длительность баланса должны быть положительными.")
EDA_ROOT = Path(EDA_OUTPUT)
if EDA_ROOT.resolve() == Path(CFG["output"]).resolve():
    raise ValueError("Задайте для EDA_OUTPUT отдельную папку, отличную от output обучения.")
EDA_ROOT.mkdir(parents=True, exist_ok=True)
DR.show_html(DR.table("Фактические параметры подготовки", pd.DataFrame({
    "parameter": list(CFG), "value": [str(v) for v in CFG.values()]})))
''')
    md('''## 2. Источники, измерения и режимы

Сначала смотрим реальную нагрузку и выбранный порог, пропуски, назначения каналов, число входов и целей,
примеры исходных и подготовленных данных. Отметка «Работа» из заявок сама по себе не доказывает работу котла.
Режим определяется по наблюдаемой нагрузке и согласованности с токами оборудования.
Если нагрузка низкая, но угольная система явно работает, это противоречие, а не уверенный останов.

Все пути, роли, единицы и правила берутся из того же источника, что при обучении.
При проблеме одного источника исправьте SETTINGS; остальные доступные котлы можно изучать отдельно.
''')
    code('''CONTEXTS, ISSUES = load_sources(CFG, EDA_ROOT/"audit", strict=False)
DATA_SUMMARY = DR.preparation_report(CONTEXTS, ISSUES, {**CFG, "output": str(EDA_ROOT)})
if not CONTEXTS:
    raise ValueError("Нет доступных источников: проверьте DATA_SETTINGS и таблицу проблем выше.")
''')
    md('''## 3. Физические остановы и заявки — включая остановы без заявок

Это **тот же реестр событий, что в 05**, без второй независимой эвристики.
Сначала фиксируется наблюдаемый останов: низкая нагрузка подтверждается несколькими реальными измерениями
согласно regime_policy.stop_hours. Пропуск нагрузки или противоречащие ей токи не подтверждают останов.
Затем к останову привязываются отметки ремонтов из часовой истории.

Показываем отдельно: физический останов, момент подтверждения, возврат работы, первую отметку ремонта,
остановы без заявок, заявки без наблюдаемого останова и конфликты типов.
Привязка отметок выполняется существующим общим реестром в окрестности 72 часов — это допущение требует проверки,
если рядом несколько остановов. Начало файла уже при выключенном котле не объявляется новым наблюдаемым остановом.

«Без заявки» здесь означает отсутствие привязанной отметки в данных. Причину такого останова автоматически не придумываем.
В классификаторе он остаётся неизвестным исходом, пока нет подтверждённой разметки.
''')
    code('''STOPS = E.stops_report(CONTEXTS, CFG, EDA_ROOT, EDA)
''')
    md('''## 4. Сколько истории действительно остаётся для обучения

Показываем для каждого месячного среза последовательный отсев строк: режим, входы, отметка работы,
выдержка истории, события и покрытие проверяемых датчиков. Каждая исключённая строка относится к первой
сработавшей причине, поэтому суммы не задваиваются.

Проверка train/calibration использует реальный partition_history из 02 для **первоначального local-обучения**.
Ни одна модель здесь не строится. Это не прогноз того, примет ли 02 последующее обновление;
для этого нужны фактические обученные состояния и проверки кандидата. Итоги реального запуска находятся в 02.
''')
    code('''AVAILABILITY = E.availability_report(CONTEXTS, CFG, EDA_ROOT)
''')
    md('''## 5. Качество датчиков и долгие постоянные значения

Считаем пропуски до и после очистки, уникальные значения, долю самого частого значения,
наблюдаемую длительность плато, признаки скачков/залипания и реально замаскированные строки.
Статистика считается на допустимой работе. Длинное плато само по себе не исключает датчик:
редко обновляемый канал может стать важным именно при изменении. Проверьте отдельно его назначение и SENSOR_RULES.
''')
    code('''SENSOR_QUALITY = E.quality_report(CONTEXTS, CFG, EDA_ROOT)
''')
    md('''## 6. Связи датчиков, нагрузка и состояния оборудования

Используем Spearman, таблицу сильных пар и сравнение типичной нагрузки при разных состояниях оборудования.
Признаки не выбираются и каналы не удаляются автоматически. Связь может объясняться общей нагрузкой,
режимом, одинаковой физической точкой или ошибкой сопоставления — это разные причины.
Тяжёлые корреляции ограничены max_analysis_rows; факт ограничения подписан в выводе.
''')
    code('''LOAD_RELATIONSHIPS = E.relationships_report(CONTEXTS, CFG, EDA_ROOT, EDA)
''')
    md('''## 7. Изменение нормы, распределения и память

Смотрим месячные медианы и грубую поправку на нагрузку по опорному периоду подготовки.
Это описательная база EDA, а не регрессор из 02. Она помогает заметить смену уровня, но не отделяет
все виды топлива и состояния оборудования. У нормированного сдвига постоянного канала нет определённого масштаба.
Отдельно считаем наблюдаемые хвосты и автокорреляции 1/24/168 часов без соединения разных рабочих участков.
''')
    code('''DISTRIBUTION_DIAGNOSTICS = E.drift_report(CONTEXTS, CFG, EDA_ROOT, EDA)
''')
    md('''## 8. Что происходило перед физическими остановами

Сравниваем последние pre_stop_hours часов перед остановом с описательной базой нагрузки;
показываем ведущие отклоняющиеся каналы и изменения после плато не короче 24 часов.
Результаты отдельно привязаны к авариям, плановым и неизвестным остановам. Без заявки не означает авария.
Это разбор истории с уже известными событиями, а не независимая оценка прогностической модели.
''')
    code('''PRE_STOP_CHANNELS = E.before_stops_report(CONTEXTS, CFG, EDA_ROOT, EDA)
''')
    md('''## 9. Необязательно: сколько исходов доступно классификатору 05

При включении CLASSIFIER_PREVIEW подсчитываем предаварийные, фоновые и неизвестные часы по каждому горизонту,
а также число независимых аварий. Огромные окна признаков не создаются; обучение не запускается.
Это подсчёт до временного разделения, поэтому достаточность train/validation/test определяется уже в 05.
Неизвестный останов и неполное будущее не превращаются в отрицательный класс.
''')
    code('''if CLASSIFIER_PREVIEW["enabled"]:
    CLASSIFIER_LABELS = E.readiness_report(CONTEXTS, CFG, EDA_ROOT,
        CLASSIFIER_PREVIEW["horizons"], CLASSIFIER_PREVIEW["options"])
else:
    CLASSIFIER_LABELS = pd.DataFrame()
    print("Подсчёт исходов выключен. Для него задайте CLASSIFIER_PREVIEW['enabled'] = True.")
''')
    md('''## 10. Необязательно: баланс вода–пар

Если заданы проверенные пары потоков, смотрим (вода − пар) / пар, длительные расхождения и эпизоды.
При несовпадающих или неизвестных единицах вычитание не выполняется.
Продувка, отборы, накопление среды и погрешности тоже влияют на баланс: эпизод не равен диагнозу утечки.
''')
    code('''BALANCE_EPISODES = E.balance_report(CONTEXTS, CFG, EDA_ROOT, EDA)
''')
    md('''## 11. Итог и дальнейший маршрут

Сначала проверьте правильность порога нагрузки, токов, сопоставления физических точек и единиц.
Затем — остановы без заявок/заявки без остановов, качество и достаточность истории.
Исправления вводятся явно в общие SETTINGS/SENSOR_RULES; EDA ничего туда автоматически не записывает.

* **01** — анализ и проверка подготовки.
* **02** — обучение/обновление регрессоров, калибровка и все результаты, включая просмотр с диска.
* **03** — необязательная проверка одного экспортированного комплекта модели на новых данных перед дашбордом.
* **05** — отдельное обучение и оценка классификатора по окнам исходных датчиков.
* Прежний **04** перенесён в archive/2026-10-28: для текущего маршрута он не требуется.

Все CSV, PNG и HTML дополнительно сохраняются в EDA_OUTPUT. Основные результаты видны в этом ноутбуке.
Старый 01 сохранён в архиве, включая прежний подробный расчёт по аналогам.
''')
    code('''FINAL_SUMMARY = DATA_SUMMARY.copy()
if len(STOPS):
    _stop_summary = STOPS.groupby("unit").agg(
        physical_stops=("observed_stop", "sum"),
        stops_without_tickets=("no_ticket", "sum"),
        tickets_without_stop=("missing_physical_stop", "sum"))
    FINAL_SUMMARY = FINAL_SUMMARY.merge(_stop_summary, on="unit", how="left")
if len(AVAILABILITY):
    _ready = AVAILABILITY.loc[AVAILABILITY.status.eq("ready_for_initial_fit")].groupby("unit").cutoff.min()
    FINAL_SUMMARY["first_ready_local_initial"] = FINAL_SUMMARY.unit.map(_ready)
E.save_table(FINAL_SUMMARY, EDA_ROOT/"summary.csv")
DR.show_html(DR.table("Итог по котлам", FINAL_SUMMARY, True))
print("Отчёты:", EDA_ROOT.resolve())
''')
    target=ROOT/'01_EDA.ipynb'
    notebook=json.loads(target.read_text(encoding='utf-8'))
    notebook['cells']=cells
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return target


if __name__=='__main__':print(build())
