"""Build the self-contained presentation code in notebook 02 (no training here)."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'02_Обучение.ipynb'


def cell(kind,source,hidden=False):
    value={'cell_type':kind,'id':hashlib.sha256((kind+source).encode()).hexdigest()[:12],
           'metadata':{},'source':source.strip().splitlines(keepends=True)}
    if kind=='code':
        value.update(execution_count=None,outputs=[])
        if hidden:value['metadata']['jupyter']={'source_hidden':True}
    return value


def build():
    # Keep the user's explicitly entered source/training configuration on rebuild.
    settings={'models':['CatBoost'],'feature_sets':['conditions'],'scenarios':['local']}
    rules={'defaults':{},'sensors':{}}
    old=json.loads(TARGET.read_text(encoding='utf-8'))
    for c in old['cells']:
        if c['cell_type']!='code':continue
        try:tree=ast.parse(''.join(c['source']))
        except SyntaxError:continue
        for node in tree.body:
            if not isinstance(node,ast.Assign):continue
            for name in node.targets:
                if not isinstance(name,ast.Name) or name.id not in ('SETTINGS','SENSOR_RULES'):continue
                try:value=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                if name.id=='SETTINGS':settings=value
                else:rules=value
    cells=[]
    def md(s):cells.append(cell('markdown',s))
    def code(s,hidden=False):cells.append(cell('code',s,hidden))
    md('''# 02 · Обучение регрессоров и оценка тревог

Этот ноутбук готовит данные, обучает и постепенно обновляет детекторы, калибрует их пороги,
а затем показывает результаты **здесь, в ноутбуке**. Классификаторы в нём не обучаются.
Для классификатора по исходным датчикам используется `05_Классификатор_без_регрессора.ipynb`.

Выполняйте разделы сверху вниз. Чтобы посмотреть уже сохранённый расчёт, достаточно разделов **1, 2 и 4–6**:
ячейку обучения в разделе 3 можно пропустить. Переменные `RESULT` и `LIVE` для просмотра не нужны.
Блоки котлов и большие таблицы открываются стрелками. Изображения встроены как PNG, поэтому видны и после сохранения ноутбука.

Предыдущий 02 сохранён отдельно в `02_Обучение_до_переработки_2026-10-28.ipynb`.
''')
    code('''import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import os, sys, json, inspect
import pandas as pd
from IPython.display import display, HTML

_candidates = [Path.cwd(), Path.cwd()/"training"]
TRAINING_DIR = next((p.resolve() for p in _candidates if (p/"lib/monitoring_v3").is_dir()), None)
if TRAINING_DIR is None:
    raise FileNotFoundError("Откройте ноутбук из папки проекта или training.")
os.chdir(TRAINING_DIR)
for _path in (TRAINING_DIR, TRAINING_DIR/"lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from notebook_reports import notebook_config
from monitoring_v3.data import load_sources
import monitoring_v3.pipeline as LF
print("Python:", sys.executable)
print("Папка проекта:", TRAINING_DIR)
''')
    md('''## 1. Настройки

`SETTINGS` переопределяет `incremental_settings.json`; `SENSOR_RULES` — правила датчиков.
Здесь сохраняются прежние пути и группы, если вы их уже указывали в 02.

Основные настройки обучения:

* `models`: список моделей, например `["CatBoost"]`.
* `feature_sets`: `["conditions"]`, `["related"]`, `["forecast"]` или несколько вариантов.
  Для CatBoost conditions использует нагрузку и условия; related дополнительно использует другие датчики;
  forecast — только прошлые измерения с горизонтом из `horizons`. Related автоматически выбирает до 6 связанных датчиков
  по корреляции Спирмена только на обучающей части. Список фиксируется в модели; `related_inputs` задаёт ручные исключения.
  Если подходящих связей нет, конкретный датчик использует только условия; это явно отмечается в таблице связей.
  Для AAKR и MSET related — классический автоассоциативный режим `all`: полный вектор проверяемых датчиков,
  включая сам восстанавливаемый датчик, вместе с условиями и состояниями. Веса общие для всех выходов.
  AAKR использует ядерное среднее, MSET — систему сходства по всей сохранённой памяти нормы.
  MSET вычисляет псевдообратную матрицу, без добавления λI и без нормирования весов на их сумму.
  Классические варианты используют только полные исходные векторы: пропуски не заполняются,
  вычисляемые признаки режима не добавляются. При пропуске в текущем векторе оценка часа недоступна.
  Отчёт показывает число полных нормальных векторов и каналы, из-за которых строки исключаются.
  Дополнительная линейная коррекция и второй устойчивый проход отключены для этих классических вариантов.
  Поиск 6 корреляционных соседей относится только к CatBoost. Вариант conditions для AAKR/MSET сохраняет
  прежнюю реализацию с автоматически выбранными соседями; это не чистая регрессия только по условиям.
* `related_policy`: `{"mode": "auto", "max_inputs": 6, "min_abs_correlation": 0.2, "sample_rows": 5000}`.
  `mode="manual"` требует списки `related_inputs` для всех целей; пустой список означает обучение только по условиям.
* `scenarios`: `["local"]` — отдельно по котлам, `["pooled"]` — общая модель группы; возможно несколько вариантов.
* `first_test`: начало основного отчётного периода. Месячные модели до него могут строиться для исторической проверки.
* `run_thresholds`: пороги **нагрузки**, не пороги тревог. Например `{"К-5": 140}`.
* `validate_candidate`: проверять новую модель перед принятием; `False` ускоряет обновление с отказом от этой проверки.
* `output`: каталог расчёта. Для продолжения используйте прежний каталог; изменённые данные/параметры могут вызвать пересчёт.

Настройки просмотра `REPORT` не меняют обучение:

* `views`: схемы для подробных отчётов и графиков; `None` — все сохранённые схемы.
* `budgets`: бюджеты тревог из `alarm_budgets`; `None` — основной `alarm_budget`.
* `units`, `experiments`: списки для фильтрации; `None` — все выбранные в SETTINGS.
* `start`, `end`: даты просмотра; `None` — от first_test до конца рассчитанной истории. Метрики пересчитываются для выбранного периода.
* `sensors`: список датчиков для графика «факт / прогноз»; `None` — первые три.

В итоговую таблицу попадают все завершённые схемы и бюджеты, даже если подробные картинки ограничены `views`.
''')
    import pprint
    code('SETTINGS = '+pprint.pformat(settings,width=100,sort_dicts=False)+'''
# При необходимости добавьте сюда:
# "data_paths": {"К-5": "D:/private/К-5/hourly.parquet"},
# "groups": {"БКЗ-500": ["К-5", "К-6", "К-7"]},
# "feature_sets": ["conditions", "related"],
# "first_test": "2025-03-01", "output": "runs/my_experiment",
# "run_thresholds": {"К-5": 140}, "validate_candidate": False,

SENSOR_RULES = '''+pprint.pformat(rules,width=100,sort_dicts=False)+'''

REPORT = {
    "views": ["mean::instant", "top5::instant", "max::instant", "fraction::instant", "mean::slow48"],
    "budgets": None,
    "units": None, "experiments": None,
    "start": None, "end": None,
    "sensors": None,
}
CFG = notebook_config(SETTINGS, SENSOR_RULES)
''')
    md('''### Служебный код вывода

Следующая ячейка содержит функции отчётов. Просто выполните её; изменять код не требуется.
В ней нет обучения моделей или классификаторов.
''')
    source=(ROOT/'lib/detector_notebook.py').read_text(encoding='utf-8')
    code('import types\n_DETECTOR_REPORT = types.ModuleType("_detector_report_02")\nexec('+repr(source)+', _DETECTOR_REPORT.__dict__)\nDR = _DETECTOR_REPORT\nDR.show_html(DR.table("Действующие настройки", pd.DataFrame({"parameter": list(CFG), "value": [str(v) for v in CFG.values()]})))',True)
    md('''## 2. Данные до обучения

Показываем размер и период истории, число каналов и признаков, пропуски, типы, примеры X и y,
режимы работы, фактический порог нагрузки и события. Большие таблицы скрыты по умолчанию.
Отметки ремонтов используются для отбора и оценки, **не подаются регрессору как признаки**.
''')
    code('''CFG = notebook_config(SETTINGS, SENSOR_RULES)
CONTEXTS, ISSUES = load_sources(CFG, Path(CFG["output"])/"audit", strict=False)
DATA_SUMMARY = DR.preparation_report(CONTEXTS, ISSUES, CFG)
''')
    md('''## 3. Обучение и последовательная проверка

Для каждого месячного среза используются только сведения, доступные к его началу.
Допущенная прошлая норма разделяется на обучение, проверку кандидата (если включена) и калибровку.
Калибровка выбирает порог по ограничениям на частоту фоновых уведомлений и долю фона в тревоге;
это **не** поиск лучшего F1 на тестовых авариях. Следующий месяц оценивается уже зафиксированной моделью и порогом.
Затем история расширяется и предыдущая модель обновляется. При непригодном обновлении сохраняется предыдущая модель.

Существующий обновляемый индикатор показывает текущую операцию и датчик. После завершения запуска автоматически
выводятся отчёты раздела 4. Сохранённые модели доступны для дашборда после каждого месячного среза.
Ошибка отдельного эксперимента сохраняется вместе с трейсбэком и не отменяет обучение остальных моделей.
Статусы записываются в status.csv по ходу расчёта; неуспешные эксперименты явно показаны в отчёте.
При остановке ячейки можно выполнить раздел 4 отдельно. Неполный расчёт помечается явно;
частичные оценки читаются из месячного кеша, если его подпись совпадает с checkpoint и исходными данными.
''')
    code('''if ISSUES:
    raise ValueError("Исправьте проблемы источников в SETTINGS и повторите раздел 2.")

TRAINING_RESULT = None
try:
    if "train_classifiers" in inspect.signature(LF.run).parameters:
        TRAINING_RESULT = LF.run(CFG, train_classifiers=False)
    else:
        # Совместимость со старой библиотекой: обучение риска также отключено.
        print("Используется старая библиотека. Для ускорения расчёта без признаков риска обновите training/lib.")
        TRAINING_RESULT = LF.run(CFG, interpretation={"run": False, "attach": False, "as_of": None})
except KeyboardInterrupt:
    print("Расчёт остановлен. Готовые модели сохранены. Для просмотра выполните раздел 4.")
else:
    METRICS = DR.render_detector_results(CFG, CONTEXTS, REPORT)
''')
    md('''## 4. Все результаты в ноутбуке — повторный просмотр без обучения

Эта ячейка работает независимо от TRAINING_RESULT и LIVE. Она читает сохранённые результаты **всех** выбранных
котлов и вариантов модели. При первом запуске отчёты уже появились после обучения; повторять ячейку нужно для
другого периода/набора схем либо после остановки обучения. Чтобы посмотреть другой output, измените SETTINGS,
повторите разделы 1–2 и выполните эту ячейку.

В каждом сворачиваемом блоке: месячные решения и отбор часов; регрессионные метрики по датчикам;
важность ненулевых признаков; classification report и confusion matrix, ROC/PR;
графики отклонений, порогов, режимов и тревог; таблицы событий; факт и прогноз датчиков.

**Как читать оценки:**

* MAE и RMSE измеряют ошибку прогноза конкретного датчика. R² может быть отрицательным.
* Event Recall — доля аварий, перед которыми пришло уведомление в разрешённом окне.
* Notification Precision — доля уведомлений с известным исходом, которые попали перед аварией.
* Почасовой classification report оценивает **наличие подтверждённой тревоги в каждом часе**;
  положительная метка означает окно перед размеченной аварией, а не любой вид физической аномалии.
  Его Precision/Recall отличаются от событийных. Нейтральные интервалы и неизвестные остановы исключаются по правилам детектора.
* ROC AUC и Average Precision оценивают ранжирование часов по баллу отклонения. Случайный AP равен доле положительных часов,
  случайный ROC AUC — 0.5. При одном классе обе оценки не приводятся.
* Фоновые уведомления/720 ч и доля фона в тревоге показывают эксплуатационную цену обнаружения.
* Недоступные для диагностики часы не считаем отрицательным классом. Смотрите availability_operating и число ненаблюдаемых аварий.

Таблицы CSV, PNG и HTML дополнительно сохраняются в output/notebook_report и в notebook_report каждого эксперимента.
Старый расчёт может не содержать MAE/RMSE/R² или самих прогнозов — тогда это явно указано; выдуманные значения не подставляются.
''')
    code('''METRICS = DR.render_detector_results(CFG, CONTEXTS, REPORT)
''')
    md('''## 5. Проверка конкретных участков: с авариями и без

Меняйте только период и котёл: повторного обучения не будет. Для выбора схем используйте названия из итоговой таблицы.
Графики показывают сохранённые прогнозы **модели своего месяца**, а не прогноз последней модели на старой истории.
Выберите период перед аварией из таблицы событий и сопоставимый рабочий участок без аварии — при близкой нагрузке и составе оборудования.
По обоим участкам видны ошибки, тревоги, пропуски диагностики и факт/прогноз выбранных датчиков.

По умолчанию фильтры ниже совпадают с REPORT, поэтому ячейка готова к выполнению без заполнения примеров дат.
''')
    code('''SEGMENT = {
    **REPORT,
    # "units": ["К-5"],
    # "start": "2025-05-01", "end": "2025-05-14 23:00",
    # "sensors": ["точное название датчика из таблицы"],
    "views": ["mean::instant"],
}
SEGMENT_METRICS = DR.render_detector_results(CFG, CONTEXTS, SEGMENT)
''')
    md('''## 6. Сохранённые модели и вопрос о классификаторе

`models.csv` — список экспортов; model — текущая модель, model_history — месячные версии, checkpoints — состояния для продолжения.
Не удаляйте их, если планируете продолжать обучение или использовать модель на дашборде.

Обученный классификатор из 05 можно смотреть на знакомых участках, но это только разбор поведения.
Для оценки качества используйте его **test_predictions.parquet** для отложенных тестовых периодов:
они не участвовали в обучении и подборе порога соответствующей модели. Порог выбирается на validation, затем фиксируется для test.
В 05 оценки на размеченных предаварийных и фоновых часах уже показаны вместе с графиками.

Если классификатор обучен на всей интересующей истории, честного теста на этой же истории нет:
нужны новые более поздние наблюдения либо повторное обучение с исторической границей и отложенным будущим периодом.
Просмотр модели на её обучении не доказывает способность предупреждать будущие аварии.
Классификатор из 05 использует исходные датчики и свои окна; в 02 его оценки не подмешиваются в тревоги регрессоров.
''')
    code('''MODELS = DR.read_csv(Path(CFG["output"])/"models.csv")
DR.show_html(DR.table("Экспорты моделей для дашборда", MODELS))
print("Общий отчёт:", (Path(CFG["output"])/"notebook_report").resolve())
''')
    md('''## 7. Только просмотр готового классификатора из 05

Обучение не запускается, модель даже не загружается: используются сохранённые тестовые баллы.
Если 05 ещё не обучен или результаты находятся в другом каталоге, появится понятное сообщение.
Задайте `root` равным OUTPUT из 05 либо `file` — конкретный test_predictions.parquet.

`folds="latest"` показывает последний тест каждого варианта, `"all"` — все.
`threshold_rule="best_f2"` использует сохранённый порог, выбранный в 05 на validation.
Если такого порога нет, применяется 0.5 с явной подписью. `units`, `start`, `end` ограничивают просмотр.
На графике видны баллы и предаварийные интервалы; распределения баллов по двум классам помогают увидеть,
различает ли модель предаварийные и фоновые часы. Classification report и confusion matrix считаются
только по часам с известным исходом, для которых модель уже была доступна.
Эти оценки относятся к выбранному периоду/модели и не усредняются с оценками регрессоров.
''')
    code('''CLASSIFIER_VIEW = {
    "root": "runs/direct_classifier_comparison",  # OUTPUT из 05
    "file": None,  # можно указать полный путь к test_predictions.parquet
    "folds": "latest",  # "all" — все тестовые периоды
    "threshold_rule": "best_f2",
    "units": None, "start": None, "end": None,
}
CLASSIFIER_TEST_METRICS = DR.saved_classifier_report(CLASSIFIER_VIEW, CFG["output"])
''')
    old['cells']=cells
    TARGET.write_text(json.dumps(old,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return TARGET


if __name__=='__main__':print(build())
