# -*- coding: utf-8 -*-
"""
Чтение телеметрии из MS SQL (pymssql) — код прежней версии дашборда, перенесён из main.py без изменений.

СЕЙЧАС НЕ ИСПОЛЬЗУЕТСЯ: дашборд работает от часовой таблицы hourly.parquet и модели (main.py, k2_model.py).
Для перехода на БД модели нужна та же часовая таблица, что при обучении: из записей ValueOPC (значение пишется
при изменении) — среднее, взвешенное по времени, за каждый час с переносом последнего значения через границу часа
(ячейка пустая, если значение известно меньше половины часа); для текстовых датчиков — состояние, державшееся
в часе дольше всего; имена столбцов — как в opc_to_wide.py (дубли имён — «имя [код]»). История — не меньше
MODEL_HISTORY_DAYS суток до момента оценки. Тогда K2Service.data() можно будет кормить из этой таблицы.

Перед импортом загрузите backend/.env (main.load_local_env), иначе параметры подключения будут пустыми.
Зависимость: pymssql.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException

TIME_COLUMN = "TimeValue"

# Источник телеметрии — MS SQL. ТЭЦ и КА также могут читаться из справочников БД.
# Настройки и запросы передаются через переменные окружения, чтобы логины и
# пароли не попадали в репозиторий. Для OPC-схемы поддерживаются исходные поля
# TimeValue, Kod_ItemOPC, Name, CurrValue, CurrStatus. Backend сам приводит их
# к универсальному контракту TimeValue / ParameterName / Value.
DATA_SOURCE = os.getenv("DATA_SOURCE", "mssql").strip().lower()
MSSQL_HOST = os.getenv("MSSQL_HOST", "").strip()
MSSQL_PORT = int(os.getenv("MSSQL_PORT", "1433"))
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "monitoring").strip()
MSSQL_USER = os.getenv("MSSQL_USER", "").strip()
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "")
MSSQL_CHARSET = os.getenv("MSSQL_CHARSET", "UTF-8").strip()
MSSQL_LOGIN_TIMEOUT = int(os.getenv("MSSQL_LOGIN_TIMEOUT", "10"))
MSSQL_QUERY_TIMEOUT = int(os.getenv("MSSQL_QUERY_TIMEOUT", "30"))
MSSQL_CATALOG_SOURCE = os.getenv("MSSQL_CATALOG_SOURCE", "database").strip().lower()
MSSQL_MEASUREMENTS_MODE = os.getenv("MSSQL_MEASUREMENTS_MODE", "equipment_fk").strip().lower()
MSSQL_HISTORY_START = os.getenv("MSSQL_HISTORY_START", "").strip()
MSSQL_HISTORY_HOURS = max(1, int(os.getenv("MSSQL_HISTORY_HOURS", "24")))
MSSQL_GOOD_STATUSES = {
    item.strip().casefold()
    for item in os.getenv("MSSQL_GOOD_STATUSES", "").split(",")
    if item.strip()
}
MSSQL_TPPS_QUERY = os.getenv(
    "MSSQL_TPPS_QUERY",
    """SELECT CAST(t.TppId AS varchar(20)) AS id,
              t.Name AS name,
              COUNT(b.BoilerUnitId) AS ka_count
       FROM monitoring.dbo.Tpp AS t
       LEFT JOIN monitoring.dbo.BoilerUnit AS b
           ON b.TppId = t.TppId AND b.IsActive = 1
       WHERE t.IsActive = 1
       GROUP BY t.TppId, t.Name, t.SortOrder
       ORDER BY t.SortOrder, t.Name""",
)
MSSQL_KAS_QUERY = os.getenv(
    "MSSQL_KAS_QUERY",
    """SELECT CAST(b.BoilerUnitId AS varchar(20)) AS id,
              b.Name AS name,
              b.Description AS description,
              b.OpcNamePrefix AS opc_name_prefix
       FROM monitoring.dbo.BoilerUnit AS b
       WHERE b.TppId = %s AND b.IsActive = 1
       ORDER BY b.SortOrder, b.Name""",
)
MSSQL_MEASUREMENTS_QUERY = os.getenv(
    "MSSQL_MEASUREMENTS_QUERY",
    """SELECT v.TimeValue, v.Kod_ItemOPC, i.Name, v.CurrValue, v.CurrStatus
       FROM monitoring.dbo.ValueOPC AS v
       INNER JOIN monitoring.dbo.ItemOPC AS i
           ON v.Kod_ItemOPC = i.Kod_ItemOPC
       WHERE i.BoilerUnitId = %s AND v.TimeValue > %s
       ORDER BY v.TimeValue, v.Kod_ItemOPC""",
)


def using_mssql() -> bool:
    return DATA_SOURCE == "mssql"


def using_database_catalog() -> bool:
    """Нужно ли брать справочники ТЭЦ/КА из БД, а не из локального config."""
    return using_mssql() and MSSQL_CATALOG_SOURCE == "database"


def query_mssql(query: str, params: Tuple[Any, ...] = ()) -> pd.DataFrame:
    """Выполняет параметризованный запрос через pymssql и возвращает DataFrame."""
    missing = [
        name
        for name, value in (
            ("MSSQL_HOST", MSSQL_HOST),
            ("MSSQL_DATABASE", MSSQL_DATABASE),
            ("MSSQL_USER", MSSQL_USER),
            ("MSSQL_PASSWORD", MSSQL_PASSWORD),
        )
        if not value
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Не заданы параметры подключения к MS SQL: {', '.join(missing)}",
        )
    try:
        import pymssql
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Не установлен pymssql. Выполните pip install -r backend/requirements.txt",
        ) from exc
    try:
        with pymssql.connect(
            server=MSSQL_HOST,
            port=MSSQL_PORT,
            user=MSSQL_USER,
            password=MSSQL_PASSWORD,
            database=MSSQL_DATABASE,
            charset=MSSQL_CHARSET,
            login_timeout=MSSQL_LOGIN_TIMEOUT,
            timeout=MSSQL_QUERY_TIMEOUT,
        ) as connection:
            return pd.read_sql_query(query, connection, params=params)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Ошибка соединения с MS SQL: {exc}") from exc


def database_tpp_list() -> List[Dict[str, Any]]:
    rows = query_mssql(MSSQL_TPPS_QUERY)
    required = {"id", "name"}
    if not required.issubset(rows.columns):
        raise HTTPException(status_code=500, detail="MSSQL_TPPS_QUERY должен вернуть поля id и name")
    result = []
    for record in rows.to_dict(orient="records"):
        tpp_id = str(record["id"])
        result.append({
            "id": tpp_id,
            "name": str(record["name"]),
            "ka_count": int(record.get("ka_count", 0) or 0),
        })
    return result


def database_ka_list(tpp_id: str) -> List[Dict[str, Any]]:
    rows = query_mssql(MSSQL_KAS_QUERY, (tpp_id,))
    required = {"id", "name"}
    if not required.issubset(rows.columns):
        raise HTTPException(status_code=500, detail="MSSQL_KAS_QUERY должен вернуть поля id и name")
    return [
        {
            "id": str(record["id"]),
            "name": str(record["name"]),
            "description": str(record.get("description", "") or ""),
            "opc_name_prefix": str(record.get("opc_name_prefix", record["name"]) or record["name"]),
            "csv_file": "",
        }
        for record in rows.to_dict(orient="records")
    ]


def _find_mssql_column(rows: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Находит поле результата SQL без зависимости от регистра и псевдонима."""
    lookup = {str(column).strip().casefold(): str(column) for column in rows.columns}
    for candidate in candidates:
        column = lookup.get(candidate.casefold())
        if column:
            return column
    return None


def _normalize_opc_status(value: Any) -> str:
    """Приводит числовой или строковый CurrStatus к виду для сравнения."""
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    try:
        number = float(text.replace(",", "."))
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def normalize_mssql_measurements(rows: pd.DataFrame, ka_name: str) -> pd.DataFrame:
    """Преобразует OPC/универсальный long-format в широкий DataFrame аналитики.

    Поддерживаемые исходные поля:
      * OPC: TimeValue, Kod_ItemOPC, Name, CurrValue, CurrStatus;
      * универсальные псевдонимы: TimeValue, ParameterName, Value, Status.
    """
    time_column = _find_mssql_column(rows, "TimeValue")
    name_column = _find_mssql_column(rows, "ParameterName", "Name")
    value_column = _find_mssql_column(rows, "Value", "CurrValue")
    status_column = _find_mssql_column(rows, "Status", "CurrStatus")
    missing = [
        label
        for label, column in (
            ("TimeValue", time_column),
            ("Name/ParameterName", name_column),
            ("CurrValue/Value", value_column),
        )
        if column is None
    ]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "MSSQL_MEASUREMENTS_QUERY не вернул обязательные поля: "
                + ", ".join(missing)
            ),
        )
    if rows.empty:
        return pd.DataFrame(columns=[TIME_COLUMN])

    selected_columns = [time_column, name_column, value_column]
    if status_column:
        selected_columns.append(status_column)
    source = rows.loc[:, selected_columns].copy()
    rename_map = {
        time_column: TIME_COLUMN,
        name_column: "ParameterName",
        value_column: "Value",
    }
    if status_column:
        rename_map[status_column] = "Status"
    source = source.rename(columns=rename_map)
    source[TIME_COLUMN] = pd.to_datetime(source[TIME_COLUMN], errors="coerce")
    source = source.dropna(subset=[TIME_COLUMN, "ParameterName"])
    source["ParameterName"] = source["ParameterName"].astype(str).str.strip()
    source = source[source["ParameterName"] != ""]

    rejected_by_status = 0
    if "Status" in source.columns and MSSQL_GOOD_STATUSES:
        normalized_good_statuses = {
            _normalize_opc_status(value) for value in MSSQL_GOOD_STATUSES
        }
        quality_mask = source["Status"].map(_normalize_opc_status).isin(normalized_good_statuses)
        rejected_by_status = int((~quality_mask).sum())
        source = source[quality_mask]

    # Числа иногда приходят из OPC строками с десятичной запятой. Состояния
    # оборудования при этом остаются строками и не теряются.
    def normalize_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        try:
            return float(stripped.replace(" ", "").replace(",", "."))
        except ValueError:
            return stripped

    source["Value"] = source["Value"].map(normalize_value)
    source = source.sort_values(TIME_COLUMN)
    source = source.groupby([TIME_COLUMN, "ParameterName"], as_index=False)["Value"].last()
    wide = source.pivot(index=TIME_COLUMN, columns="ParameterName", values="Value").reset_index()
    wide.columns.name = None

    prefix_pattern = re.compile(rf"^{re.escape(ka_name)}(?:\s|$)", re.IGNORECASE)
    wide = wide.rename(
        columns={
            column: str(column) if prefix_pattern.search(str(column)) else f"{ka_name} {column}"
            for column in wide.columns
            if column != TIME_COLUMN
        }
    )
    wide = wide.sort_values(TIME_COLUMN).reset_index(drop=True)
    wide.attrs["mssqlQuality"] = {
        "rowsReceived": int(len(rows)),
        "rowsAccepted": int(len(source)),
        "rowsRejectedByStatus": rejected_by_status,
        "statusFilterEnabled": bool(MSSQL_GOOD_STATUSES),
    }
    return wide


def load_mssql_data(
    tpp_id: str,
    ka_id: str,
    ka_name: str,
    opc_name_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Загружает измерения выбранного КА из MS SQL и готовит их для React."""
    if MSSQL_HISTORY_START:
        try:
            history_start: Any = datetime.fromisoformat(MSSQL_HISTORY_START)
        except ValueError:
            history_start = MSSQL_HISTORY_START
    else:
        history_start = datetime.now() - timedelta(hours=MSSQL_HISTORY_HOURS)

    if MSSQL_MEASUREMENTS_MODE == "equipment_fk":
        # ka_id — первичный ключ dbo.BoilerUnit, связанный с ItemOPC.BoilerUnitId.
        params: Tuple[Any, ...] = (ka_id, history_start)
    elif MSSQL_MEASUREMENTS_MODE == "opc":
        prefix = (opc_name_prefix or ka_name).strip()
        params: Tuple[Any, ...] = (f"{prefix}%", history_start)
    else:
        raise HTTPException(
            status_code=500,
            detail=(
                "Неизвестный MSSQL_MEASUREMENTS_MODE. "
                "Поддерживаются equipment_fk и opc."
            ),
        )
    rows = query_mssql(MSSQL_MEASUREMENTS_QUERY, params)
    return normalize_mssql_measurements(rows, ka_name)
