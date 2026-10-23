"""Independent data/model services for the configured boilers."""
from dataclasses import dataclass
import json
from pathlib import Path

from k2_model import K2Service, DataUnavailable


@dataclass
class Boiler:
    tpp: dict
    unit: dict
    service: K2Service

    @property
    def key(self):
        return self.tpp['id'], self.unit['id']


class Fleet:
    def __init__(self, boilers):
        self.boilers = list(boilers)
        if not self.boilers:
            raise ValueError('В конфигурации нет включённых котлов.')
        keys = [b.key for b in self.boilers]
        if len(keys) != len(set(keys)):
            raise ValueError('Пара ТЭЦ / котёл повторяется в конфигурации.')

    def select(self, tpp_id=None, ka_id=None):
        found = [b for b in self.boilers if (not tpp_id or b.tpp['id'] == tpp_id)
                 and (not ka_id or b.unit['id'] == ka_id)]
        if not found:
            raise DataUnavailable('Для выбранной ТЭЦ и котла нет настройки данных и модели.')
        if ka_id and not tpp_id and len(found) > 1:
            raise DataUnavailable('Этот номер котла есть на нескольких ТЭЦ: укажите ТЭЦ.')
        return found[0]

    def stations(self):
        result = {}
        for b in self.boilers:
            station = result.setdefault(b.tpp['id'], dict(b.tpp, ka_count=0))
            station['ka_count'] += 1
        return list(result.values())


def load_fleet(base_dir, configured, fallback, service_defaults):
    """Paths in JSON are absolute or relative to the dashboard folder, like .env."""
    path = Path(configured) if configured else base_dir / 'data' / 'boilers.json'
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        if configured:
            raise ValueError(f'Не найден FLEET_CONFIG: {path}')
        rows, defaults = [fallback], {}
    else:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        rows, defaults = data.get('boilers'), data.get('defaults', {})
        if not isinstance(rows, list) or not isinstance(defaults, dict):
            raise ValueError(f'{path}: нужны список boilers и объект defaults.')
    boilers = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Каждый котёл должен быть объектом JSON.')
        if row.get('enabled', True) is False:
            continue
        cfg = {**service_defaults, **defaults, **row}
        for name in ('tpp_id', 'ka_id', 'name', 'data_path', 'model_dir'):
            if not isinstance(cfg.get(name), str) or not cfg[name].strip():
                raise ValueError(f'Для котла обязательно поле {name}.')
        history = cfg.get('history_days')
        if history is not None and (not isinstance(history, int) or isinstance(history, bool) or history < 4):
            raise ValueError('history_days: null (вся история) или целое число не меньше 4.')
        window = cfg['window_hours']
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError('window_hours должен быть положительным целым числом.')
        service = K2Service(base_dir, cfg['data_path'], cfg['model_dir'], cfg['view'], history,
                            window, cfg['health_good'], cfg['health_zero'], cfg['levels'],
                            months_dir=cfg.get('history_dir', ''), model_unit=cfg.get('model_unit', ''))
        boilers.append(Boiler(
            dict(id=cfg['tpp_id'], name=cfg.get('tpp_name', cfg['tpp_id'])),
            dict(id=cfg['ka_id'], name=cfg['name'], description=cfg.get('description', '')),
            service))
    names = {}
    for b in boilers:
        if b.tpp['id'] in names and names[b.tpp['id']] != b.tpp['name']:
            raise ValueError('У одной ТЭЦ указаны разные названия.')
        names[b.tpp['id']] = b.tpp['name']
    return Fleet(boilers)
