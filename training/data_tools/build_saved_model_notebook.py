"""Optional 03: exercise an export without fitting or calibrating it."""
from pathlib import Path
import json
from build_eda_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def md(s):cells.append(cell('markdown',s))
    def code(s):cells.append(cell('code',s))
    md('''# 03 · Необязательная проверка экспорта для дашборда

**Для обычного обучения этот ноутбук не нужен.** Историческое качество и все графики смотрите в 02.

03 нужен, когда хотите проверить, что **конкретная сохранённая папка модели** загружается и работает:
перед подключением к дашборду, на новом периоде данных или на другом котле с совместимыми каналами.
Здесь используется один зафиксированный экспорт; дообучения и пересчёта порогов нет.
Это отличается от 02, где история проверяется последовательностью месячных моделей.

Используйте отдельное ядро для этого ноутбука. WARMUP_HOURS=None сохраняет всю доступную предысторию подготовки.
По умолчанию данные до даты доступности модели не оцениваются. Проверка последней моделью её собственного прошлого
не заменяет честный исторический тест из 02.
''')
    code('''import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import os, sys, json
_candidates = [Path.cwd(), Path.cwd()/"training"]
TRAINING_DIR = next((p.resolve() for p in _candidates if (p/"lib/monitoring_v3").is_dir()), None)
if TRAINING_DIR is None:
    raise FileNotFoundError("Откройте ноутбук из корня проекта или training.")
os.chdir(TRAINING_DIR)
for _path in (TRAINING_DIR, TRAINING_DIR/"lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import saved_model_runner as SR
import detector_notebook as DR
from preparation_notebook import saved_settings
''')
    md('''## 1. Выберите экспорт и данные

MODELS_ROOT=None берёт output из сохранённых настроек 02, затем из incremental_settings.json.
MODEL_DIR — точный folder из таблицы ниже (config.json, model.py и веса в одной папке).
DATA_PATH — hourly.parquet выбранного котла. TARGET_UNIT — его имя в таблице ролей.
PERIOD_START/PERIOD_END ограничивают период проверки; None — доступный период после даты модели.
SENSOR_FILE=None — таблица ролей из настроек; REQUIRE_UNSEEN_UNIT=True требует, чтобы котёл не участвовал в обучении.
MODEL_VIEW=None показывает mean::instant либо первую доступную схему; можно указать точное название.
''')
    code('''MODELS_ROOT = None
MODEL_DIR = ""
DATA_PATH = ""
TARGET_UNIT = "К-5"
SENSOR_FILE = None
PERIOD_START = None
PERIOD_END = None
WARMUP_HOURS = None
REQUIRE_UNSEEN_UNIT = False
MODEL_VIEW = None
CHECK_OUTPUT = "saved_model_checks"
''')
    code('''_values = json.loads(Path("incremental_settings.json").read_text(encoding="utf-8")) if Path("incremental_settings.json").exists() else {}
_values.update(saved_settings("02_Обучение.ipynb").get("SETTINGS", {}))
_models_root = MODELS_ROOT if MODELS_ROOT is not None else _values.get("output", "runs/incremental")
_sensor_file = SENSOR_FILE if SENSOR_FILE is not None else _values.get("sensor_file", "sensor_roles.xlsx")
AVAILABLE_MODELS = SR.discover_models(_models_root)
DR.show_html(DR.table("Экспорты: выберите одну папку для MODEL_DIR", AVAILABLE_MODELS))
''')
    md('''## 2. Запуск сохранённой модели

При пустых MODEL_DIR/DATA_PATH расчёт пропускается. После задания путей:
проверяются каналы и единицы, применяется сохранённая подготовка, веса и пороги;
затем выводятся сводка работы, тревоги и график прямо здесь.
Результаты дополнительно сохраняются в CHECK_OUTPUT. Файлы экспорта проверяются на неизменность.

Число тревог на новых данных — эксплуатационная статистика. Без разметки исходов это не Precision/Recall.
Для сравнения качества по авариям используйте последовательные оценки 02.
''')
    code('''CHECK = None
if MODEL_DIR and DATA_PATH:
    _metadata = json.loads((Path(MODEL_DIR)/"config.json").read_text(encoding="utf-8"))
    _flavor = "incremental" if _metadata.get("core") == "incremental" else "legacy" if "core" in _metadata else "actual"
    CHECK = SR.run(MODEL_DIR, DATA_PATH, TARGET_UNIT, flavor=_flavor, sensor_file=_sensor_file,
        period_start=PERIOD_START, period_end=PERIOD_END, warmup_hours=WARMUP_HOURS,
        require_unseen_unit=REQUIRE_UNSEEN_UNIT, output_dir=CHECK_OUTPUT)
    DR.show_html(DR.table("Работа выбранного экспорта", CHECK["summary"], True))
    DR.show_html(DR.table("Все тревоги", CHECK["alarms"]))
    DR.show_html(DR.table("Сопоставление и доступность каналов", CHECK["audit"]))
    _view = MODEL_VIEW or ("mean::instant" if "mean::instant" in CHECK["scores"] else next(iter(CHECK["scores"])))
    DR.show_html(DR.figure_html(SR.plot(CHECK, view=_view), CHECK["root"]/"notebook_overview.png"))
    print("Результаты:", CHECK["root"])
else:
    print("03 необязателен. Для проверки экспорта заполните MODEL_DIR и DATA_PATH; для обычного просмотра используйте 02.")
''')
    target=ROOT/'03_Проверка_сохранённой_модели.ipynb'
    n=json.loads(target.read_text(encoding='utf-8'));n['cells']=cells
    target.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return target


if __name__=='__main__':print(build())
