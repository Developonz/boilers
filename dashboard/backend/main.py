# -*- coding: utf-8 -*-
"""
Предиктивный дашборд котлоагрегата — Backend API (FastAPI).

Данные — часовая таблица hourly.parquet (от opc_to_wide.py), оценка — модель из папки models/ (экспорт k2_deploy,
по умолчанию «Аналоги+», вид «несколько датчиков»). Норма, отклонения, уровни параметров, тревоги и индекс
здоровья берутся только из модели; расчёт — на выбранный момент (параметр at, по умолчанию конец файла),
на экран выводится окно последних WINDOW_HOURS часов до него.

Как это считается:
  * норма параметра — коридор модели для текущего режима: ожидание ± обычная ошибка датчика × разброс аналогов;
  * отклонение параметра — ошибка / обычная ошибка датчика (1 — граница нормы), среднее за время сглаживания модели;
    уровень: Г < 1, В 1–2, Б 2–3, А ≥ 3 (LEVEL_V / LEVEL_B / LEVEL_A);
  * индекс здоровья — из отношения «сглаженная ошибка модели / её порог»: ≤ 0.5 -> 100, порог -> 50, ≥ 3 -> 0;
  * техсостояние — тревога модели (выше порога PERSIST ч подряд) / превышение порога / норма; пока котёл стоит
    или идёт пуск, модель не оценивает.
Модели: после даты обучения итоговой модели момент считает она; раньше — модель месяца этого момента (обучена только
на прошлом, как онлайн-оценка в ноутбуке 02), если папки месяцев выгружены («<папка модели>_history/ГГГГ-ММ»).
Иначе итоговая модель «помнит» часы, на которых училась, и показывала бы их как норму.
На экран выводятся только датчики, которые модель получает на вход; тексты — без терминов модели.

Запуск:  python main.py    (по умолчанию http://localhost:8000)
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from k2_model import M_PRESTOP, M_RUN, M_START, M_STOP, DataUnavailable, Evaluation, resolve_hourly_path
from fleet_config import load_fleet

# ============================================================================
# Конфигурация
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DIST_DIR = BASE_DIR / "predictive_maintenance_dashboard" / "dist"
TPP_KA_CONFIG_FILE = DATA_DIR / "tpp_ka_config.json"
TIME_COLUMN = "TimeValue"


def load_local_env() -> None:
    """Загружает backend/.env без внешней зависимости; системные env приоритетнее."""
    env_file = BASE_DIR / "backend" / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

# Load SSO settings after backend/.env; a missing auth module must not disable protection.
from auth import auth_settings, get_keycloak_client, require_authentication, get_current_user_id

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DATA_SOURCE = os.getenv("DATA_SOURCE", "hourly").strip().lower()
HOURLY_PATH = os.getenv("HOURLY_PATH", "").strip()
MODEL_DIR = os.getenv("MODEL_DIR", "models").strip() or "models"
MODEL_VIEW = os.getenv("MODEL_VIEW", "Аналоги+: несколько датчиков").strip()
_history = os.getenv("MODEL_HISTORY_DAYS", "11").strip()
MODEL_HISTORY_DAYS = None if _history.lower() in ("all", "none", "null") else int(_history)
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "24"))
HEALTH_RATIO_GOOD = float(os.getenv("HEALTH_RATIO_GOOD", "0.5"))
HEALTH_RATIO_ZERO = float(os.getenv("HEALTH_RATIO_ZERO", "3.0"))
LEVELS = (float(os.getenv("LEVEL_V", "1")), float(os.getenv("LEVEL_B", "2")), float(os.getenv("LEVEL_A", "3")))
MODEL_MONTHS_DIR = os.getenv("MODEL_MONTHS_DIR", "").strip()   # пусто — «<папка модели>_history» рядом с ней
UNIT_NAME = os.getenv("UNIT_NAME", "КА-2").strip()
UNIT_TPP_ID = os.getenv("UNIT_TPP_ID", "").strip()

SERVICE_DEFAULTS = dict(view=MODEL_VIEW, history_days=MODEL_HISTORY_DAYS, window_hours=WINDOW_HOURS,
                        health_good=HEALTH_RATIO_GOOD, health_zero=HEALTH_RATIO_ZERO, levels=LEVELS)

EQUIPMENT_RE = re.compile(r"^([КK][АA][-\s]?\d+)", re.IGNORECASE)      # «КА-8» и латиницей «KA-8»

TREND_ARROW = {"increasing": "▲ растёт", "decreasing": "▼ падает", "stable": "→ стабильно"}
TREND_LABEL = {"increasing": "рост", "decreasing": "снижение", "stable": "стабильно"}

RISK_ORDER = ("A", "B", "V", "G", "N")
RISK_TITLES = {"A": "критический", "B": "высокий", "V": "значимый", "G": "норма", "N": "не оценивается"}
MODE_LABELS = {M_RUN: "Работа", M_PRESTOP: "Перед остановом", M_START: "Пуск", M_STOP: "Стоит"}

CATEGORY_LABELS = {
    "Pressure": "Давление",
    "Temperature": "Температура",
    "Flow": "Расход",
    "Gases": "Газы",
    "State": "Состояние",
    "Other": "Прочее",
}
CATEGORY_UNITS = {"Pressure": "кгс/см²", "Temperature": "°C", "Flow": "т/ч", "Gases": "%"}

# «Что проверить» при отклонении — подсказка по виду параметра (сам факт отклонения даёт модель)
RECOMMENDATIONS = {
    "Pressure": "регулятор давления, предохранительные клапаны и импульсные линии датчика.",
    "Temperature": "горелочные устройства, тепловосприятие поверхностей нагрева и датчик температуры.",
    "Flow": "расходомер, баланс пара/воды и работу регулирующей арматуры.",
    "Gases": "топочный режим, присосы и газоанализатор.",
    "State": "состояние механизма и цепи сигнализации.",
    "Other": "датчик и технологический режим.",
}

# Ключевые параметры для KPI-карточек: (подпись, ключевые слова в имени)
KEY_SPECS: List[Tuple[str, List[str]]] = [
    ("Давление острого пара", ["давление острого пара"]),
    ("Давление в барабане", ["давление в барабане"]),
    ("Давление питательной воды", ["давление питательной воды"]),
    ("Расход острого пара", ["расход острого пара за котлом"]),
    ("Расход питательной воды", ["расход питательной воды"]),
    ("Температура свежего пара", ["температура свежего пара"]),
    ("Температура питательной воды", ["температура питательной воды"]),
    ("Температура мазута", ["температура мазута"]),
    ("O₂ слева", ["содержание о2", "слева"]),
    ("O₂ справа", ["содержание о2", "справа"]),
]

app = FastAPI(
    title="Предиктивный дашборд API",
    version="4.0.0",
    description="Мониторинг котлоагрегата: часовые данные и модель аномалий",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Keycloak SSO: routes and frontend contract from the supplied main.py
# ============================================================================
@app.get("/auth/login")
async def login(next: Optional[str] = Query(None)):
    """Редирект на Keycloak для SSO авторизации."""
    redirect_uri = f"{auth_settings.API_URL}/auth/callback"
    
    params = {
        "client_id": auth_settings.KEYCLOAK_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
    }
    
    if next:
        params["state"] = next
    
    from urllib.parse import urlencode
    keycloak_login_url = (
        f"{auth_settings.KEYCLOAK_SERVER_URL}/realms/{auth_settings.KEYCLOAK_REALM}"
        f"/protocol/openid-connect/auth?{urlencode(params)}"
    )
    
    return RedirectResponse(url=keycloak_login_url)


@app.get("/auth/callback")
async def auth_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
):
    """
    OAuth2 callback endpoint.
    Обменивает code на токены Keycloak и возвращает HTML страницу,
    которая сохраняет токены и редиректит на приложение.
    """
    from fastapi.responses import HTMLResponse
    
    if error:
        error_msg = error_description or error
        redirect_url = f"/#/auth/error?error={error_msg}"
        return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head><title>Auth Error</title></head>
<body>
<script>
function deleteCookie(name) {{
    document.cookie = name + '=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
}}
deleteCookie('access_token');
deleteCookie('refresh_token');
window.location.href = '{redirect_url}';
</script>
<p>Authentication error: {error_msg}</p>
<p>Redirecting...</p>
</body>
</html>
        """)
    
    if not code:
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>Auth Error</title></head>
<body>
<script>
window.location.href = '/#/auth/error?error=missing_code';
</script>
<p>Missing authorization code</p>
</body>
</html>
        """, status_code=400)
    
    try:
        keycloak = get_keycloak_client()
        redirect_uri = f"{auth_settings.API_URL}/auth/callback"
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[CALLBACK] Exchanging code for token...")
        
        tokens = await keycloak.exchange_code_for_token(code, redirect_uri)
        
        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        
        logger.info("[CALLBACK] Tokens received")
        
        # Determine redirect URL (state param or root)
        redirect_target = state if state else "/"
        
        return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head><title>Authentication Successful</title></head>
<body>
<script>
(function() {{
    // Токены передаются сервером напрямую — НЕ из URL-параметров
    var accessToken = '{access_token}';
    var refreshToken = '{refresh_token}';
    
    // Функции для работы с cookie
    function setCookie(name, value, maxAgeSeconds) {{
        var expires = new Date(Date.now() + maxAgeSeconds * 1000).toUTCString();
        document.cookie = name + '=' + encodeURIComponent(value) + '; path=/; expires=' + expires + '; SameSite=Lax';
    }}
    
    var COOKIE_MAX_AGE = 7 * 24 * 60 * 60; // 7 дней
    
    if (accessToken) {{
        setCookie('access_token', accessToken, COOKIE_MAX_AGE);
    }}
    if (refreshToken) {{
        setCookie('refresh_token', refreshToken, COOKIE_MAX_AGE);
    }}
    
    // Clear URL parameters
    var redirectTo = '{redirect_target}';
    window.history.replaceState(null, '', window.location.pathname);
    
    // Redirect to app
    window.location.href = redirectTo;
}})();
</script>
<p>Authentication successful. Redirecting...</p>
</body>
</html>
        """)
        
    except Exception as e:
        import logging
        logging.error(f"Auth callback error: {e}")
        return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head><title>Auth Error</title></head>
<body>
<script>
function deleteCookie(name) {{
    document.cookie = name + '=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
}}
deleteCookie('access_token');
deleteCookie('refresh_token');
window.location.href = '/#/auth/error?error=token_exchange_failed';
</script>
<p>Authentication failed: {e}</p>
<p>Redirecting...</p>
</body>
</html>
        """)


@app.get("/auth/error")
async def auth_error(error: Optional[str] = Query(None)):
    """Показывает ошибку авторизации."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=401,
        content={"error": error or "Unknown error", "authenticated": False}
    )


@app.post("/auth/refresh")
async def refresh_token(request: Request):
    """Обновление access token через refresh token."""
    try:
        body = await request.json()
        refresh_token = body.get("refresh_token")
        
        if not refresh_token:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Missing refresh_token")
        
        keycloak = get_keycloak_client()
        tokens = await keycloak.refresh_token(refresh_token)
        
        return {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token", refresh_token),
            "expires_in": tokens.get("expires_in"),
            "token_type": tokens.get("token_type", "Bearer"),
        }
    except Exception as e:
        import logging
        logging.error(f"Token refresh error: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Failed to refresh token")


@app.post("/auth/logout")
async def logout(request: Request):
    """Выход из системы - отзыв токена в Keycloak."""
    try:
        body = await request.json()
        refresh_token = body.get("refresh_token")
        
        if refresh_token:
            keycloak = get_keycloak_client()
            await keycloak.revoke_token(refresh_token, "refresh_token")
        
        return {"message": "Logged out successfully"}
    except Exception as e:
        import logging
        logging.error(f"Logout error: {e}")
        return {"message": "Logged out"}


@app.get("/auth/userinfo")
async def get_user_info(
    user_info: dict = Depends(require_authentication),
):
    """Получение информации о текущем пользователе."""
    return user_info


@app.get("/auth/debug")
async def auth_debug(
    user_id: Optional[str] = Depends(get_current_user_id),
    request: Request = None,
):
    """
    Диагностический эндпоинт — показывает, пришёл ли токен.
    Используйте в консоли браузера:
      fetch('/auth/debug').then(r=>r.json()).then(console.log)
    """
    import logging
    logger2 = logging.getLogger(__name__)
    
    # Читаем Authorization напрямую из заголовков запроса
    auth_header = None
    if request:
        auth_header = request.headers.get("authorization", None)
    
    logger2.warning("[AUTH DEBUG] authenticated=%s, auth_header_received=%s", user_id is not None, bool(auth_header))
    
    return {
        "user_id": user_id,
        "authenticated": user_id is not None,
        "auth_header_received": auth_header is not None,
        "auth_header_preview": None,
        "note": (
            "OK — токен отправляется" if user_id else
            "ОШИБКА: токен НЕ отправляется с запросом (auth_header=None)"
        ),
    }


@app.get("/auth/me")
async def get_current_user(
    user_id: Optional[str] = Depends(get_current_user_id),
):
    """Получение ID текущего пользователя (если авторизован)."""
    if user_id:
        return {"user_id": user_id, "authenticated": True}
    return {"authenticated": False}


@app.get("/auth/status")
async def auth_status(user_id: Optional[str] = Depends(get_current_user_id)):
    """Проверка статуса авторизации."""
    import logging
    logging.warning(f"[AUTH] /auth/status called — user_id: {user_id}")
    return {"authenticated": user_id is not None}



# ============================================================================
# Мелкие помощники
# ============================================================================
def fnum(value: Any, digits: int = 4) -> Optional[float]:
    """Число для JSON: None вместо NaN/inf и нечисел."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def fmt(value: Any, unit: str = "") -> str:
    """Число для текста причин и рекомендаций."""
    v = fnum(value, 6)
    if v is None:
        return "—"
    digits = 1 if abs(v) >= 100 else 2 if abs(v) >= 1 else 3
    return f"{v:.{digits}f} {unit}".strip()


def iso(value: Any) -> Optional[str]:
    return pd.Timestamp(value).isoformat() if value is not None and not pd.isna(value) else None


def parse_at(value: Optional[str]) -> Optional[pd.Timestamp]:
    """Момент из запроса (ISO, например 2026-08-20T13:00); пусто — конец файла. Округляется вниз до часа."""
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Не удалось разобрать дату «{value}»") from exc
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.floor("h")


def service_call(func, *args, **kwargs):
    """Ошибки данных и модели -> понятный ответ 503 с текстом причины."""
    if DATA_SOURCE != "hourly":
        raise HTTPException(
            status_code=503,
            detail=(f"Источник «{DATA_SOURCE}» пока не подключён к модели: модели нужна часовая таблица. "
                    "Установите DATA_SOURCE=hourly (см. mssql_source.py — что нужно для БД)."),
        )
    try:
        return func(*args, **kwargs)
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def detect_equipment(columns: List[str]) -> str:
    for col in columns:
        match = EQUIPMENT_RE.match(str(col))
        if match:
            return match.group(1).replace(" ", "-")
    return UNIT_NAME


def display_name(col: str, prefix: str = "") -> str:
    """Убирает префикс оборудования из имени колонки."""
    name = col
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    name = name.strip().lstrip("-—: ").strip()
    return name or col


def is_state_name(col: str) -> bool:
    low = col.lower()
    return "состо" in low or "state" in low or "режим" in low


def categorize_column(col: str) -> str:
    low = col.lower()
    if is_state_name(col):
        return "State"
    if "давлен" in low or "press" in low:
        return "Pressure"
    if "расход" in low or "flow" in low:
        return "Flow"
    if "содержан" in low or re.search(r"о2|o2", low):
        return "Gases"
    if "темпера" in low or re.search(r"(^|\s)т\.?\s", low) or low.startswith("т "):
        return "Temperature"
    return "Other"


def guess_unit(col: str) -> str:
    return CATEGORY_UNITS.get(categorize_column(col), "")


def icon_for(category: str) -> str:
    return {"Pressure": "pressure", "Temperature": "temperature", "Flow": "flow", "Gases": "gas",
            "State": "state"}.get(category, "other")


def calculate_trend(values: List[float]) -> str:
    """Направление тренда по знаку наклона линейной регрессии (справочно, на оценку не влияет)."""
    n = len(values)
    if n < 2:
        return "stable"
    sum_x = sum(range(n))
    sum_y = sum(values)
    sum_xy = sum(i * values[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))
    denom = n * sum_x2 - sum_x * sum_x
    mean_value = sum_y / n
    if denom == 0 or not mean_value:
        return "stable"
    relative_slope = ((n * sum_xy - sum_x * sum_y) / denom) / abs(mean_value) * 100
    if relative_slope > 0.5:
        return "increasing"
    if relative_slope < -0.5:
        return "decreasing"
    return "stable"


def calculate_statistics(series: pd.Series) -> Dict[str, Any]:
    """Описательная статистика окна показа (справочно)."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return {"current": None, "first": None, "min": None, "max": None, "mean": None, "median": None, "std": None,
                "range": None, "count": 0, "trend": "stable", "changeAbs": None, "changePct": None, "volatility": None}
    values = [float(v) for v in clean.tolist()]
    mean_value = sum(values) / len(values)
    std_value = float(clean.std(ddof=0)) if len(values) > 1 else 0.0
    first_value, last_value = values[0], values[-1]
    change_abs = last_value - first_value
    if abs(first_value) > 1e-9:
        change_pct = change_abs / abs(first_value) * 100
    else:
        change_pct = 0.0 if abs(change_abs) < 1e-9 else 100.0
    return {
        "current": fnum(last_value),
        "first": fnum(first_value),
        "min": fnum(min(values)),
        "max": fnum(max(values)),
        "mean": fnum(mean_value),
        "median": fnum(float(clean.median())),
        "std": fnum(std_value),
        "range": fnum(max(values) - min(values)),
        "count": len(values),
        "trend": calculate_trend(values),
        "changeAbs": fnum(change_abs),
        "changePct": fnum(change_pct, 2),
        "volatility": fnum(std_value / abs(mean_value) * 100, 2) if abs(mean_value) > 1e-9 else 0.0,
    }


# ============================================================================
# Каталог и независимые службы данных/моделей котлов
# ============================================================================
def legacy_catalog() -> Dict[str, Dict[str, str]]:
    """Станция и агрегат из data/tpp_ka_config.json: агрегат UNIT_NAME, станция UNIT_TPP_ID
    (по умолчанию — первая, где такой агрегат есть)."""
    tpp = {"id": "tpp-1", "name": "ТЭЦ"}
    unit = {"id": UNIT_NAME.lower(), "name": UNIT_NAME, "description": f"Котлоагрегат {UNIT_NAME}"}
    if TPP_KA_CONFIG_FILE.exists():
        config = json.loads(TPP_KA_CONFIG_FILE.read_text(encoding="utf-8"))
        for station in config.get("tpps", []):
            if UNIT_TPP_ID and station.get("id") != UNIT_TPP_ID:
                continue
            ka = next((k for k in station.get("kas", []) if k.get("name") == UNIT_NAME), None)
            if ka or UNIT_TPP_ID:
                tpp = {"id": str(station["id"]), "name": str(station.get("name", station["id"]))}
                if ka:
                    unit = {"id": str(ka["id"]), "name": str(ka["name"]),
                            "description": str(ka.get("description", "") or "")}
                break
    return {"tpp": tpp, "unit": unit}


_single = legacy_catalog()
FLEET = load_fleet(BASE_DIR, os.getenv("FLEET_CONFIG", "").strip(),
    dict(tpp_id=_single['tpp']['id'], tpp_name=_single['tpp']['name'],
         ka_id=_single['unit']['id'], name=_single['unit']['name'], description=_single['unit']['description'],
         data_path=str(resolve_hourly_path(BASE_DIR, HOURLY_PATH)), model_dir=MODEL_DIR,
         history_dir=MODEL_MONTHS_DIR, model_unit=os.getenv('MODEL_UNIT', '').strip()), SERVICE_DEFAULTS)


def get_tpp_list() -> List[Dict[str, Any]]:
    return FLEET.stations()


def get_ka_list(tpp_id: str) -> List[Dict[str, Any]]:
    return [dict(b.unit) for b in FLEET.boilers if b.tpp['id'] == tpp_id]


def check_selection(tpp_id: Optional[str], ka_id: Optional[str]):
    try:
        return FLEET.select(tpp_id, ka_id)
    except DataUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ============================================================================
# Параметры
# ============================================================================
def position_text(se, unit: str, smooth: int) -> str:
    """Где значение относительно нормы режима (для рекомендации и текста предупреждения)."""
    fact = se.fact
    if fact is None or se.low is None or se.high is None:
        return f"Отклонение ×{se.ratio:.1f} от обычного."
    if fact > se.high:
        return f"Выше верхней границы нормы режима ({fmt(se.high, unit)}) на {fmt(fact - se.high, unit)}."
    if fact < se.low:
        return f"Ниже нижней границы нормы режима ({fmt(se.low, unit)}) на {fmt(se.low - fact, unit)}."
    return f"Сейчас в пределах нормы режима, но за последние {smooth} ч отклонение в среднем ×{se.ratio:.1f} от обычного."


def smooth_hours(ev: Evaluation) -> int:
    return int(ev.lm.cfg.get("smooth_steps", 6))


def not_evaluated_advice(ev: Evaluation, se) -> str:
    """Что делать, если для параметра в этот час нет оценки."""
    if ev.mode_now in (M_STOP, M_START):
        return "Оценка появится после выхода котла в рабочий режим."
    if se.reason.startswith("нетипичный режим"):
        return "Наблюдать за параметром: для такого режима нормы нет."
    if se.reason.startswith("одиночный скачок"):
        return "Наблюдать за параметром."
    if se.reason.startswith("мало данных по датчику"):
        return "Норма появится, когда по датчику наберётся достаточно данных."
    return "Проверить поступление данных по датчику."


def model_block(se) -> Dict[str, Any]:
    return {
        "role": se.role,
        "roleLabel": "режимный параметр" if se.role == "condition" else "",
        "evaluated": se.evaluated,
        "notEvaluatedReason": se.reason or None,
        "expected": fnum(se.expected),
        "spread": fnum(se.spread),
        "low": fnum(se.low),
        "high": fnum(se.high),
        "fact": fnum(se.fact),
        "deviation": fnum(se.fact - se.expected) if se.fact is not None and se.expected is not None else None,
        "errorRatio": fnum(se.ratio, 3),
        "errorRatioNow": fnum(se.ratio_now, 3),
        "contributionPct": fnum(se.share, 1),
        "mainInAlarm": se.main_in_alarm,
    }


def numeric_parameter(ev: Evaluation, col: str, prefix: str) -> Dict[str, Any]:
    category = categorize_column(col)
    unit = guess_unit(col)
    stats = calculate_statistics(ev.window[col])
    se = ev.sensors[col]
    frozen_h = ev.frozen_hours.get(col)
    smooth = smooth_hours(ev)

    reasons: List[str] = []
    if se.evaluated:
        if se.expected is not None:
            line = f"сейчас {fmt(se.fact, unit)} при ожидаемых {fmt(se.expected, unit)} для текущего режима"
            if se.low is not None:
                line += f"; норма режима {fmt(se.low)}…{fmt(se.high, unit)}"
            reasons.append(line)
        reasons.append(f"отклонение ×{se.ratio:.2f} от обычного в среднем за {smooth} ч (больше 1 — вне нормы)")
        if se.share is not None and se.share >= 1:
            reasons.append(f"вклад в отклонение котла {se.share:.0f}%")
        if se.main_in_alarm:
            reasons.append("один из главных параметров текущей тревоги")
        outside = se.fact is not None and se.low is not None and not (se.low <= se.fact <= se.high)
        if se.level in ("A", "B", "V"):
            recommendation = f"{position_text(se, unit, smooth)} Что проверить: {RECOMMENDATIONS[category]}"
        elif outside:
            recommendation = (f"{position_text(se, unit, smooth)} В среднем за {smooth} ч отклонение ×{se.ratio:.2f} "
                              "от обычного — пока в пределах нормы. Наблюдать за параметром.")
        else:
            recommendation = "В пределах нормы для текущего режима — продолжать наблюдение."
    elif se.role == "condition":
        recommendation = "Режимный параметр: по нему определяется режим работы котла, нормы для него нет."
    else:
        if se.reason:
            reasons.append(se.reason)
        recommendation = not_evaluated_advice(ev, se)
    if frozen_h:
        reasons.append(f"значение не меняется {frozen_h} ч подряд — возможен отказ датчика")
        recommendation = "Проверить датчик и канал передачи: показания не меняются."

    norm = None
    if se.low is not None and se.high is not None:
        norm = {"low": fnum(se.low), "high": fnum(se.high), "label": f"{fmt(se.low)}…{fmt(se.high, unit)}"}
    return {
        "column": col,
        "name": display_name(col, prefix),
        "unit": unit,
        "category": category,
        "categoryLabel": CATEGORY_LABELS.get(category, CATEGORY_LABELS["Other"]),
        "icon": icon_for(category),
        "type": "numeric",
        **{k: v for k, v in stats.items() if k != "trend"},
        "trend": TREND_ARROW[stats["trend"]],
        "trendDirection": stats["trend"],
        "trendLabel": TREND_LABEL[stats["trend"]],
        "frozen": frozen_h is not None,
        "frozenHours": frozen_h,
        "riskLevel": se.level,
        "riskScore": fnum(se.ratio, 3) if se.evaluated else None,
        "riskReasons": reasons,
        "norm": norm,
        "recommendation": recommendation,
        "model": model_block(se),
        "sparkline": [fnum(v) for v in pd.to_numeric(ev.window[col], errors="coerce").dropna().tolist()],
    }


def state_parameter(ev: Evaluation, col: str, prefix: str, usual: Dict[str, str]) -> Dict[str, Any]:
    """Текстовое состояние механизма; модель видит его как часть режима («не в обычном состоянии»)."""
    values = ev.window[col]
    current = next((str(v) for v in reversed(values.tolist()) if v is not None and not pd.isna(v)), None)
    distribution: Dict[str, int] = {}
    for value in values.tolist():
        key = str(value) if value is not None and not pd.isna(value) else "нет данных"
        distribution[key] = distribution.get(key, 0) + 1
    usual_state = usual.get(col)
    unusual = bool(usual_state and current is not None and current != usual_state)
    reasons = [f"обычное состояние — «{usual_state}»"] if usual_state else []
    if unusual:
        reasons.append(f"сейчас не в обычном состоянии: «{current}»")
    return {
        "column": col,
        "name": display_name(col, prefix),
        "unit": "",
        "category": "State",
        "categoryLabel": CATEGORY_LABELS["State"],
        "icon": "state",
        "type": "categorical",
        "current": current,
        "currentLabel": current,
        "usualState": usual_state,
        "unusual": unusual,
        "min": None, "max": None, "mean": None, "median": None, "std": None, "range": None,
        "count": int(values.notna().sum()),
        "trend": TREND_ARROW["stable"], "trendDirection": "stable", "trendLabel": TREND_LABEL["stable"],
        "changeAbs": None, "changePct": None, "volatility": None,
        "frozen": False, "frozenHours": None,
        "riskLevel": "N", "riskScore": None, "riskReasons": reasons,
        "norm": None,
        "recommendation": f"Проверить {RECOMMENDATIONS['State']}" if unusual else "Продолжать наблюдение.",
        "model": None,
        "distribution": distribution,
    }


def find_column(columns: List[str], prefix: str, keywords: List[str]) -> Optional[str]:
    for col in columns:
        name = display_name(col, prefix).lower()
        if all(keyword in name for keyword in keywords):
            return col
    return None


def build_kpis(ev: Evaluation, prefix: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    kpis: List[Dict[str, Any]] = []
    used: set = set()
    for label, keywords in KEY_SPECS:
        col = find_column(ev.numeric, prefix, keywords)
        if not col or col in used:
            continue
        used.add(col)
        param = parameters.get(display_name(col, prefix))
        if not param:
            continue
        kpis.append({
            "id": f"kpi-{len(kpis)}",
            "name": label,
            "parameterName": param["name"],
            "column": col,
            "value": param["current"],
            "unit": param["unit"],
            "trend": param["trend"],
            "trendDirection": param["trendDirection"],
            "changePct": param["changePct"],
            "riskLevel": param["riskLevel"],
            "riskScore": param["riskScore"],
            "min": param["min"],
            "max": param["max"],
            "norm": param["norm"],
            "expected": (param.get("model") or {}).get("expected"),
            "role": (param.get("model") or {}).get("role"),
            "sparkline": param["sparkline"],
        })
    return kpis


# ============================================================================
# Состояние агрегата, предупреждения, сводка
# ============================================================================
def technical_state(ev: Evaluation) -> Dict[str, Any]:
    """Режим и техсостояние агрегата на момент at — только из модели."""
    mode_label = MODE_LABELS.get(ev.mode_now, ev.mode_now)
    operating_mode = "stopped" if ev.mode_now == M_STOP else "unknown" if ev.mode_now == 'неизвестно' else "running"
    ratio = ev.ratio_now
    persist = ev.lm.cfg.get("persist_steps", 3)
    if ev.not_scored_reason:
        status, level = "unknown", "N"
        label = {M_STOP: "Котёл стоит", M_START: "Пуск котла"}.get(ev.mode_now, "Нет данных")
        trend = ev.not_scored_reason[:1].upper() + ev.not_scored_reason[1:]
    elif ev.alarm_now:
        status, label, level = "critical", "Устойчивое отклонение", "A"
        trend = "Устойчивое отклонение от нормы режима"
    elif ratio is not None and ratio > 1:
        status, label, level = "warning", "Отклонение", "B"
        trend = f"Отклонение выше допустимого — тревога, если продержится {persist} ч"
    else:
        status, label, level = "normal", "Норма", "G"
        trend = "Параметры в пределах нормы режима"
    return {
        "operatingMode": operating_mode,
        "operatingModeLabel": mode_label,
        "technicalStatus": status,
        "technicalStatusLabel": label,
        "riskLevel": level,
        "healthScore": ev.health_now,
        "healthMeaning": "Соответствие ожидаемому режиму; это не вероятность аварии и не остаточный ресурс",
        "healthTrend": trend,
        "modelRatio": fnum(ratio, 3),
        "modelAlarm": ev.alarm_now,
        "interpretation": ev.interpretation,
        "statusCritical": status == "critical",
    }


def health_series(ev: Evaluation) -> List[Dict[str, Any]]:
    part = ev.health.loc[ev.window.index]
    return [{"time": t.isoformat(), "value": fnum(v, 0)} for t, v in part.items()]


def episode_sensors(ev: Evaluation, start: pd.Timestamp, end: pd.Timestamp, prefix: str) -> List[Dict[str, Any]]:
    """Три главных датчика эпизода: 90-й процентиль «ошибка / обычная» за эпизод (как в model.py)."""
    if ev.errors is None:
        return []
    part = ev.errors.loc[(ev.errors.index >= start) & (ev.errors.index < end)]
    if not len(part):
        return []
    top = part.quantile(0.9).dropna().nlargest(3)
    return [{"name": display_name(c, prefix), "column": c, "ratio": fnum(v, 2)} for c, v in top.items() if v > 0]


def build_alerts(ev: Evaluation, prefix: str, parameters: Dict[str, Any], kpi_names: set) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    smooth = smooth_hours(ev)
    win0 = ev.window.index[0]
    step = pd.Timedelta("1h")
    for ep in ev.episodes:
        if ep["end"] <= win0 and not ep["active"]:
            continue
        active = ep["active"]
        end = min(ep["end"], ev.at + step)
        hours = int(round((end - ep["start"]) / step))
        sensors = episode_sensors(ev, ep["start"], end, prefix)
        span = ev.health.loc[(ev.health.index >= ep["start"]) & (ev.health.index < end)]
        health_min = fnum(span.min(), 0) if span.notna().any() else None
        when = (f"с {ep['start']:%d.%m %H:%M}, идёт {hours} ч" if active
                else f"{ep['start']:%d.%m %H:%M} — {ep['end']:%d.%m %H:%M} ({hours} ч), завершилась")
        alerts.append({
            "kind": "model",
            "id": f"model-{ep['start']:%Y%m%d%H}",
            "parameter": "Тревога" if active else "Тревога (завершилась)",
            "column": None,
            "value": ev.health_now if active else health_min,
            "valueLabel": "Индекс здоровья" if active else "Минимальный индекс",
            "unit": "",
            "riskLevel": "A" if active else "V",
            "riskScore": ep["peak"],
            "changePct": None,
            "category": "Boiler",
            "categoryLabel": "Котёл",
            "isKpi": False,
            "active": active,
            "start": iso(ep["start"]),
            "end": iso(ep["end"]),
            "hours": hours,
            "peak": fnum(ep["peak"], 2),
            "sensors": sensors,
            "message": f"Устойчивое отклонение от нормы режима {when}.",
            "reasons": [f"{s['name']}: отклонение ×{s['ratio']:.1f} от обычного" for s in sensors],
            "recommendation": ("Осмотреть узлы главных параметров тревоги и проверить сами датчики."
                               if active else "Тревога завершилась — проверить, что стало её причиной."),
        })
    if not ev.alarm_now and ev.ratio_now is not None and ev.ratio_now > 1:
        persist = ev.lm.cfg.get("persist_steps", 3)
        alerts.append({
            "kind": "model",
            "id": "model-over",
            "parameter": "Отклонение выше допустимого",
            "column": None,
            "value": ev.health_now,
            "valueLabel": "Индекс здоровья",
            "unit": "",
            "riskLevel": "B",
            "riskScore": ev.ratio_now,
            "changePct": None,
            "category": "Boiler",
            "categoryLabel": "Котёл",
            "isKpi": False,
            "active": True,
            "sensors": episode_sensors(ev, ev.at, ev.at + step, prefix),
            "message": f"Индекс здоровья ниже 50: тревога, если отклонение продержится {persist} ч подряд.",
            "reasons": [],
            "recommendation": "Наблюдать за главными параметрами отклонения.",
        })
    for name, param in parameters.items():
        model = param.get("model") or {}
        if param.get("type") != "numeric" or not model.get("evaluated") or param["riskLevel"] not in ("A", "B", "V"):
            continue
        se = ev.sensors[param["column"]]
        alerts.append({
            "kind": "sensor",
            "id": param["column"][:60],
            "parameter": name,
            "column": param["column"],
            "value": param["current"],
            "unit": param["unit"],
            "riskLevel": param["riskLevel"],
            "riskScore": param["riskScore"],
            "changePct": param["changePct"],
            "deviation": model.get("deviation"),
            "expected": model.get("expected"),
            "norm": param["norm"],
            "category": param["category"],
            "categoryLabel": param["categoryLabel"],
            "isKpi": name in kpi_names,
            "message": position_text(se, param["unit"], smooth),
            "reasons": param["riskReasons"],
            "recommendation": f"Проверить {RECOMMENDATIONS[param['category']]}",
        })
    weight = {"A": 0, "B": 1, "V": 2}
    alerts.sort(key=lambda a: (weight.get(a["riskLevel"], 3), a["kind"] != "model", -(a["riskScore"] or 0), a["parameter"]))
    return alerts


def build_summary(ev: Evaluation, prefix: str, parameters: Dict[str, Any], states: List[Dict[str, Any]],
                  alerts: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    numeric = {n: p for n, p in parameters.items() if p.get("type") == "numeric"}
    by_risk = {level: 0 for level in RISK_ORDER}
    for param in numeric.values():
        by_risk[param["riskLevel"]] = by_risk.get(param["riskLevel"], 0) + 1

    by_category: List[Dict[str, Any]] = []
    for category, label in CATEGORY_LABELS.items():
        items = [p for p in numeric.values() if p["category"] == category]
        if not items:
            continue
        scored = [p for p in items if p["riskLevel"] != "N"]
        worst = min(scored, key=lambda p: RISK_ORDER.index(p["riskLevel"]))["riskLevel"] if scored else "N"
        by_category.append({
            "category": category,
            "label": label,
            "total": len(items),
            "evaluated": len(scored),
            "alerts": sum(1 for p in items if p["riskLevel"] in ("A", "B", "V")),
            "worstRisk": worst,
            "risky": sum(1 for p in items if p["riskLevel"] in ("A", "B")),
        })

    trend_counts = {"increasing": 0, "decreasing": 0, "stable": 0}
    for param in numeric.values():
        trend_counts[param["trendDirection"]] += 1

    evaluated = [p for p in numeric.values() if (p.get("model") or {}).get("evaluated")]
    worst = max(evaluated, key=lambda p: p["riskScore"] or 0) if evaluated else None
    window_cols = [c for c in ev.window.columns]
    return {
        **state,
        "status": state["operatingModeLabel"],
        "equipment": prefix,
        "byRisk": by_risk,
        "byCategory": by_category,
        "trendCounts": trend_counts,
        "totalParameters": len(numeric),
        "checkedParameters": sum(1 for p in numeric.values() if (p.get("model") or {}).get("role") == "checked"),
        "evaluatedParameters": len(evaluated),
        "states": states,
        "frozen": [
            {"name": p["name"], "value": p["current"], "unit": p["unit"], "categoryLabel": p["categoryLabel"],
             "hours": p["frozenHours"], "inModel": (p.get("model") or {}).get("role") == "checked"}
            for p in numeric.values() if p.get("frozen")
        ],
        "unavailable": [
            {"name": display_name(c, prefix), "column": c} for c in window_cols if ev.window[c].isna().all()
        ],
        "missing": [{"name": display_name(c, prefix), "column": c} for c in ev.missing],
        "worst": ({"name": worst["name"], "riskLevel": worst["riskLevel"], "riskScore": worst["riskScore"],
                   "current": worst["current"], "unit": worst["unit"]} if worst else None),
        "alertsTotal": len(alerts),
        "healthSeries": health_series(ev),
    }


# ============================================================================
# Набор данных на момент (кэш)
# ============================================================================
_bundle_cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_bundle_lock = threading.Lock()


def get_bundle(at: Optional[pd.Timestamp], boiler=None) -> Dict[str, Any]:
    boiler = boiler or FLEET.select()
    ev = boiler.service.evaluate(at)
    key = (*boiler.key, ev.at)
    with _bundle_lock:
        if key in _bundle_cache and _bundle_cache[key]["ev"] is ev:
            _bundle_cache.move_to_end(key)
            return _bundle_cache[key]
        bundle = build_bundle(ev, boiler)
        if boiler.service.history_h is None:
            for previous in list(_bundle_cache):
                if previous[:2] == boiler.key:
                    del _bundle_cache[previous]
        _bundle_cache[key] = bundle
        while len(_bundle_cache) > max(8, len(FLEET.boilers) * 2):
            _bundle_cache.popitem(last=False)
        return bundle


def build_bundle(ev: Evaluation, boiler) -> Dict[str, Any]:
    cat = dict(tpp=boiler.tpp, unit=boiler.unit)
    unit_name = boiler.unit['name']
    prefix = unit_name if any(str(c).startswith(unit_name) for c in ev.numeric) else detect_equipment(ev.numeric)
    usual = {**dict(boiler.service.model().bools), **dict(ev.lm.bools)}
    parameters: Dict[str, Any] = {}
    states: List[Dict[str, Any]] = []
    for col in ev.text:
        param = state_parameter(ev, col, prefix, usual)
        parameters[param["name"]] = param
        states.append(param)
    for col in ev.numeric:
        param = numeric_parameter(ev, col, prefix)
        parameters[param["name"]] = param
    kpis = build_kpis(ev, prefix, parameters)
    alerts = build_alerts(ev, prefix, parameters, {k["parameterName"] for k in kpis})
    state = technical_state(ev)
    summary = build_summary(ev, prefix, parameters, states, alerts, state)
    return {
        "ev": ev,
        "service": boiler.service,
        "prefix": prefix,
        "parameters": parameters,
        "states": states,
        "kpis": kpis,
        "alerts": alerts,
        "summary": summary,
        "state": state,
        "tpp": cat["tpp"],
        "unit": cat["unit"],
    }


def model_brief(service, lm=None) -> Dict[str, Any]:
    info = service.model_info(lm)
    return {key: info[key] for key in ("folder", "title", "view", "threshold", "trainedUntil", "quality",
                                       "smoothHours", "persistHours", "historyHours", "windowHours", "levelThresholds")}


def data_source_label() -> str:
    return "архив данных"


def overview_of(bundle: Dict[str, Any]) -> Dict[str, Any]:
    ev, state = bundle["ev"], bundle["state"]
    return {
        "equipmentName": bundle["unit"]["name"],
        "equipmentType": "Котлоагрегат",
        **state,
        "status": state["operatingModeLabel"],
        "asOf": ev.at.isoformat(),
        "lastUpdate": ev.at.isoformat(),
        "firstUpdate": ev.window.index[0].isoformat(),
        "observationHours": len(ev.window),
        "dataPoints": int(len(ev.window)),
        "columns": int(len(ev.window.columns)),
        "dataRange": bundle["service"].data_range(),
        "dataSource": "hourly",
        "dataSourceLabel": data_source_label(),
        "model": model_brief(bundle["service"], ev.lm),
        "tppId": bundle["tpp"]["id"],
        "tppName": bundle["tpp"]["name"],
        "kaId": bundle["unit"]["id"],
        "kaDescription": bundle["unit"]["description"],
    }


def find_parameter(bundle: Dict[str, Any], key: str) -> Tuple[str, Dict[str, Any]]:
    target = key.strip().lower()
    for name, param in bundle["parameters"].items():
        if target in (name.lower(), str(param["column"]).lower()):
            return name, param
    for name, param in bundle["parameters"].items():
        if target in name.lower() or target in str(param["column"]).lower():
            return name, param
    raise HTTPException(status_code=404, detail=f"Параметр '{key}' не найден")


def parameter_timeline(ev: Evaluation, col: str) -> List[Dict[str, Any]]:
    """Ряд окна показа: значение, ожидание и коридор нормы режима, отклонение (×) по часам."""
    lm = ev.lm
    q = fnum(lm.q.get(col), 9)
    exp = ev.expected
    has_exp = exp is not None and (col, "ожидание") in exp.columns
    has_err = ev.errors_smooth is not None and col in ev.errors_smooth.columns
    points: List[Dict[str, Any]] = []
    for t in ev.window.index:
        point: Dict[str, Any] = {"time": t.isoformat(), "value": fnum(ev.window.at[t, col])}
        if has_exp and t in exp.index:
            e, s = fnum(exp.at[t, (col, "ожидание")]), fnum(exp.at[t, (col, "разброс")])
            if e is not None:
                point["expected"] = e
                if s is not None and q is not None:
                    point["low"], point["high"] = fnum(e - q * s), fnum(e + q * s)
        if has_err and t in ev.errors_smooth.index:
            point["error"] = fnum(ev.errors_smooth.at[t, col], 3)
        points.append(point)
    return points


# ============================================================================
# Сводка всех настроенных станций и котлов
# ============================================================================
def unit_card(bundle: Dict[str, Any]) -> Dict[str, Any]:
    ev, state = bundle["ev"], bundle["state"]
    parameters = bundle["parameters"]

    def find(*variants: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        for fragments in variants:
            for param in parameters.values():
                name = str(param["name"]).casefold()
                if param.get("type") == "numeric" and all(f in name for f in fragments):
                    return param
        return None

    power = find(("электр", "мощност"), ("активн", "мощност"))
    load = find(("нагрузк",))
    efficiency = find(("кпд", "брутто"))
    steam = find(("расход", "острого пара"), ("паропроизвод",))
    sensors = []
    for kpi in bundle["kpis"][:4]:
        param = parameters[kpi["parameterName"]]
        level = param["riskLevel"]
        sensors.append({
            "id": param["column"],
            "name": param["name"],
            "shortName": kpi["name"],
            "value": param["current"],
            "unit": param["unit"],
            "status": "critical" if level == "A" else "warning" if level in ("B", "V") else "normal",
            "riskLevel": level,
            "norm": param["norm"],
            "role": (param.get("model") or {}).get("role"),
            "sparkline": param["sparkline"],
        })
    series = health_series(ev)
    return {
        "id": bundle["unit"]["id"],
        "name": bundle["unit"]["name"],
        "description": bundle["unit"]["description"],
        "dataAvailable": True,
        "dataError": None,
        **state,
        "status": state["technicalStatus"],
        "statusLabel": state["technicalStatusLabel"],
        "online": state["operatingMode"] == "running",
        "activeAlerts": len(bundle["alerts"]),
        "capacityMw": None,
        "outputMw": power["current"] if power else None,
        "steamOutputTph": steam["current"] if steam else None,
        "loadPct": load["current"] if load and load["unit"] == "%" else None,
        "efficiencyPct": efficiency["current"] if efficiency else None,
        "boilerGrossEfficiencyPct": efficiency["current"] if efficiency else None,
        "sparkline": [p["value"] for p in series],
        "series": series,
        "sensors": sensors,
        "lastUpdate": ev.at.isoformat(),
    }


def empty_unit(unit: Dict[str, str], error: str) -> Dict[str, Any]:
    return {
        "id": unit["id"], "name": unit["name"], "description": unit.get("description", ""),
        "dataAvailable": False, "dataError": error,
        "healthScore": None, "healthTrend": None, "modelRatio": None, "modelAlarm": False,
        "status": "unknown", "statusLabel": "Нет данных", "technicalStatus": "unknown", "technicalStatusLabel": "Нет данных",
        "operatingMode": "unknown", "operatingModeLabel": "Нет данных", "riskLevel": "N", "statusCritical": False,
        "online": False, "activeAlerts": 0, "capacityMw": None, "outputMw": None, "steamOutputTph": None,
        "loadPct": None, "efficiencyPct": None, "boilerGrossEfficiencyPct": None,
        "sparkline": [], "series": [], "sensors": [], "lastUpdate": None,
    }


def _average(values: List[Optional[float]], digits: int = 1) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return round(sum(valid) / len(valid), digits) if valid else None


def _sum_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return round(sum(valid), 1) if valid else None


def aggregate_trend(units):
    points = {}
    for unit in units:
        for point in unit.get('series', []):
            if point.get('value') is not None:
                points.setdefault(point['time'], []).append(point['value'])
    return [dict(time=time, value=_average(values)) for time, values in sorted(points.items())]


def station_overview(tpp, units):
    available = [u for u in units if u["dataAvailable"]]
    running = [u for u in available if u["operatingMode"] == "running"]
    critical = sum(u["technicalStatus"] == "critical" for u in available)
    warning = sum(u["technicalStatus"] == "warning" for u in available)
    if critical:
        station_status, station_label = "critical", "Есть тревоги"
    elif warning:
        station_status, station_label = "warning", "Есть отклонения"
    elif any(u["technicalStatus"] == "normal" for u in available):
        station_status, station_label = "normal", "Норма"
    else:
        station_status, station_label = "unknown", "Нет оценки" if available else "Нет данных"
    health = _average([u["healthScore"] for u in available], 0)
    trend = aggregate_trend(units)
    station = {
        "id": tpp["id"],
        "name": tpp["name"],
        "healthScore": health,
        "status": station_status,
        "statusLabel": station_label,
        "unitCount": len(units),
        "dataAvailableCount": len(available),
        "onlineCount": len(running),
        "runningCount": len(running),
        "normalCount": sum(u["technicalStatus"] == "normal" for u in available),
        "attentionCount": warning,
        "criticalCount": critical,
        "reserveCount": sum(u["operatingMode"] == "stopped" for u in available),
        "activeAlerts": sum(u["activeAlerts"] for u in available),
        "capacityMw": None,
        "generationMw": _sum_or_none([u["outputMw"] for u in available]),
        "electricPowerMw": _sum_or_none([u["outputMw"] for u in available]),
        "steamOutputTph": _sum_or_none([u["steamOutputTph"] for u in available]),
        "loadPct": _average([u["loadPct"] for u in available]),
        "efficiencyPct": _average([u["efficiencyPct"] for u in available]),
        "boilerGrossEfficiencyPct": _average([u["boilerGrossEfficiencyPct"] for u in available]),
        "trend": [p["value"] for p in trend],
        "healthTrend": trend,
        "lastUpdate": max((u["lastUpdate"] for u in available if u["lastUpdate"]), default=None),
        "units": units,
    }
    return station


def build_fleet_overview(at: Optional[pd.Timestamp]) -> Dict[str, Any]:
    alerts, notices, units_by_station, ranges = [], [], {}, []
    for boiler in FLEET.boilers:
        try:
            bundle = service_call(get_bundle, at, boiler)
            unit = unit_card(bundle)
            for alert in bundle['alerts']:
                alerts.append(dict(alert,
                    id=f"{boiler.tpp['id']}-{unit['id']}-{alert['id']}",
                    stationId=boiler.tpp['id'], stationName=boiler.tpp['name'],
                    unitId=unit['id'], unitName=unit['name'],
                    severity='critical' if alert['riskLevel'] == 'A' else 'warning',
                    title=alert['parameter'], time=unit['lastUpdate']))
        except HTTPException as exc:
            unit = empty_unit(boiler.unit, str(exc.detail))
            notices.append(f"{boiler.tpp['name']} / {boiler.unit['name']}: {exc.detail}")
        units_by_station.setdefault(boiler.tpp['id'], []).append(unit)
        try:
            ranges.append(boiler.service.data_range())
        except DataUnavailable:
            pass
    stations = [station_overview(tpp, units_by_station[tpp['id']]) for tpp in FLEET.stations()]
    units = [u for station in stations for u in station['units']]
    available = [u for u in units if u['dataAvailable']]
    running = [u for u in available if u['operatingMode'] == 'running']
    critical = sum(u['technicalStatus'] == 'critical' for u in available)
    warning = sum(u['technicalStatus'] == 'warning' for u in available)
    normal = sum(u['technicalStatus'] == 'normal' for u in available)
    reserve = sum(u['operatingMode'] == 'stopped' for u in available)
    trend = aggregate_trend(units)
    as_of = max((u['lastUpdate'] for u in available if u['lastUpdate']), default=None)
    data_range = dict(start=min((r['start'] for r in ranges if r.get('start')), default=None),
                      end=max((r['end'] for r in ranges if r.get('end')), default=None))
    return {
        'generatedAt': as_of, 'asOf': as_of, 'dataRange': data_range,
        'dataSource': 'hourly', 'dataSourceLabel': data_source_label(), 'isSynthetic': False,
        'dataStatus': 'ok' if available else 'no_data', 'notice': '\n'.join(notices) or None,
        'summary': {
            'stationsTotal': len(stations), 'unitsTotal': len(units), 'unitsWithData': len(available),
            'unitsOnline': len(running), 'unitsRunning': len(running), 'unitsNormal': normal,
            'unitsAttention': warning, 'unitsCritical': critical, 'unitsWithDeviations': warning + critical,
            'unitsReserve': reserve, 'averageHealth': _average([u['healthScore'] for u in available], 0),
            'activeAlerts': len(alerts), 'totalGenerationMw': _sum_or_none([u['outputMw'] for u in available]),
            'totalElectricPowerMw': _sum_or_none([u['outputMw'] for u in available]),
            'efficiencyPct': _average([u['efficiencyPct'] for u in available]),
            'averageBoilerGrossEfficiencyPct': _average([u['boilerGrossEfficiencyPct'] for u in available]),
            'availabilityPct': round(len(running) / len(available) * 100, 1) if available else None,
            'runningPct': round(len(running) / len(available) * 100, 1) if available else None,
        },
        'healthSparkline': [p['value'] for p in trend], 'healthTrend': trend,
        'generationSparkline': [], 'generationTrend': [], 'stations': stations, 'alerts': alerts,
    }


# ============================================================================
# REST API
# ============================================================================
@app.get("/api/health")
def health_check(tpp_id: Optional[str] = Query(None), ka_id: Optional[str] = Query(None),
                 deep: bool = Query(False, description="Проверить данные и выполнить расчёт модели")):
    """Docker checks process liveness; explicit deep checks also evaluate a model."""
    if not deep and not (tpp_id or ka_id):
        return {"status": "healthy", "timestamp": datetime.now().isoformat(),
                "source": "hourly", "boilers": len(FLEET.boilers)}
    try:
        boiler = check_selection(tpp_id, ka_id)
        bundle = service_call(get_bundle, None, boiler)
        ev = bundle["ev"]
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "source": "hourly",
            "file": str(boiler.service.hourly_path),
            "asOf": ev.at.isoformat(),
            "rows": int(len(ev.hist)),
            "columns": int(ev.hist.shape[1]),
            "equipment": bundle["prefix"],
            "model": model_brief(bundle["service"], ev.lm),
            "missingSensors": len(ev.missing),
        }
    except HTTPException as exc:
        return {"status": "degraded", "error": str(exc.detail), "timestamp": datetime.now().isoformat()}


@app.get("/api/config", dependencies=[Depends(require_authentication)])
def get_config(tpp_id: Optional[str] = Query(None), ka_id: Optional[str] = Query(None)):
    """Конфигурация выбранного котла."""
    boiler = check_selection(tpp_id, ka_id)
    service = boiler.service
    try:
        data_range = service.data_range()
    except DataUnavailable:
        data_range = None
    try:
        model = service.model_info()
    except DataUnavailable as exc:
        model = {"error": str(exc)}
    return {
        "equipment": boiler.unit["name"],
        "equipmentType": "Котлоагрегат",
        "categoryLabels": CATEGORY_LABELS,
        "riskTitles": RISK_TITLES,
        "tpps": get_tpp_list(),
        "dataSource": "hourly",
        "dataFile": str(service.hourly_path),
        "dataRange": data_range,
        "windowHours": service.window_h,
        "model": model,
    }


@app.get("/api/model", dependencies=[Depends(require_authentication)])
def get_model(tpp_id: Optional[str] = Query(None), ka_id: Optional[str] = Query(None)):
    """Модель: метод, вид, порог, условия, состояния механизмов, роли датчиков."""
    service = check_selection(tpp_id, ka_id).service
    info = service_call(service.model_info)
    lm = service.model()
    info["sensors"] = [{"column": c, "role": r, "usualError": fnum(lm.q.get(c))} for c, r in lm.roles.items()]
    return info


@app.get("/api/tpps", dependencies=[Depends(require_authentication)])
def get_tpps():
    return {"tpps": get_tpp_list()}


@app.get("/api/fleet-overview", dependencies=[Depends(require_authentication)])
def get_fleet_overview(at: Optional[str] = Query(None, description="Момент ISO; пусто — конец файла")):
    """Первый экран: все настроенные станции и котлы на выбранный момент."""
    return build_fleet_overview(parse_at(at))


@app.get("/api/tpps/{tpp_id}/kas", dependencies=[Depends(require_authentication)])
def get_tpp_kas(tpp_id: str):
    if tpp_id not in {b.tpp["id"] for b in FLEET.boilers}:
        raise HTTPException(status_code=404, detail="ТЭЦ не найдена")
    return {"tppId": tpp_id, "kas": get_ka_list(tpp_id)}


@app.get("/api/dashboard", dependencies=[Depends(require_authentication)])
def get_dashboard_data(
    tpp_id: Optional[str] = Query(None),
    ka_id: Optional[str] = Query(None),
    at: Optional[str] = Query(None, description="Момент ISO; пусто — конец файла"),
):
    """Полный набор данных для всех блоков дашборда на момент at."""
    boiler = check_selection(tpp_id, ka_id)
    bundle = service_call(get_bundle, parse_at(at), boiler)
    return {
        "overview": overview_of(bundle),
        "kpis": bundle["kpis"],
        "alerts": bundle["alerts"],
        "parameters": bundle["parameters"],
        "summary": bundle["summary"],
        "unit": unit_card(bundle),
    }


@app.get("/api/parameters", dependencies=[Depends(require_authentication)])
def get_parameters(tpp_id: Optional[str] = Query(None), ka_id: Optional[str] = Query(None),
                   at: Optional[str] = Query(None)):
    boiler = check_selection(tpp_id, ka_id)
    bundle = service_call(get_bundle, parse_at(at), boiler)
    return {
        "parameters": bundle["parameters"],
        "states": bundle["states"],
        "equipment": bundle["prefix"],
        "lastUpdate": bundle["ev"].at.isoformat(),
        "totalRecords": int(len(bundle["ev"].window)),
    }


@app.get("/api/parameter/{param_name:path}", dependencies=[Depends(require_authentication)])
def get_parameter(param_name: str, tpp_id: Optional[str] = Query(None), ka_id: Optional[str] = Query(None),
                  at: Optional[str] = Query(None)):
    """Один параметр: оценка модели и ряд окна показа (значение, ожидание и коридор нормы режима)."""
    boiler = check_selection(tpp_id, ka_id)
    bundle = service_call(get_bundle, parse_at(at), boiler)
    name, param = find_parameter(bundle, param_name)
    result = dict(param)
    if result.get("type") == "numeric":
        result["timeline"] = parameter_timeline(bundle["ev"], param["column"])
    else:
        ev = bundle["ev"]
        result["timeline"] = [{"time": t.isoformat(), "value": v} for t, v in ev.window[param["column"]].items()]
    result["windowHours"] = len(bundle["ev"].window)
    result["asOf"] = bundle["ev"].at.isoformat()
    return result


def window_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.reset_index().to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                record[key] = None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
    return records


@app.get("/api/columns", dependencies=[Depends(require_authentication)])
def get_columns(tpp_id: Optional[str] = Query(None), ka_id: Optional[str] = Query(None), at: Optional[str] = Query(None)):
    boiler = check_selection(tpp_id, ka_id)
    bundle = service_call(get_bundle, parse_at(at), boiler)
    ev, prefix = bundle["ev"], bundle["prefix"]
    columns_info = [{"name": TIME_COLUMN, "type": "datetime", "displayName": TIME_COLUMN, "category": None,
                     "unit": "", "hasData": True, "modelRole": None}]
    for col in ev.window.columns:
        se = ev.sensors.get(col)
        columns_info.append({
            "name": col,
            "type": "numeric" if col in set(ev.numeric) else "state",
            "displayName": display_name(col, prefix),
            "category": categorize_column(col),
            "unit": guess_unit(col),
            "hasData": bool(ev.window[col].notna().any()),
            "modelRole": se.role if se else None,
        })
    return {"columns": columns_info, "equipment": prefix, "total": len(columns_info)}


@app.get("/api/data", dependencies=[Depends(require_authentication)])
def get_data(
    tpp_id: Optional[str] = Query(None),
    ka_id: Optional[str] = Query(None),
    at: Optional[str] = Query(None),
    columns: Optional[str] = None,
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """Часовые данные окна показа с постраничной выдачей."""
    boiler = check_selection(tpp_id, ka_id)
    df = service_call(get_bundle, parse_at(at), boiler)["ev"].window
    if columns:
        selected = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if selected:
            df = df[selected]
    return {"data": window_records(df.iloc[offset:offset + limit]), "total": len(df), "limit": limit, "offset": offset}


@app.get("/api/export", dependencies=[Depends(require_authentication)])
def export_data(
    tpp_id: Optional[str] = Query(None),
    ka_id: Optional[str] = Query(None),
    at: Optional[str] = Query(None),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    """Экспорт часовых данных окна показа."""
    boiler = check_selection(tpp_id, ka_id)
    df = service_call(get_bundle, parse_at(at), boiler)["ev"].window
    if format == "csv":
        return PlainTextResponse(
            content=df.reset_index().to_csv(index=False),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="ka_export.csv"'},
        )
    records = window_records(df)
    return {"data": records, "count": len(records)}


# ============================================================================
# Раздача собранного SPA (если выполнена сборка фронтенда)
# ============================================================================
if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Отдаёт собранный SPA либо справочную информацию об API."""
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "name": "Предиктивный дашборд API",
        "version": app.version,
        "status": "running",
        "docs": "/docs",
        "hint": "Сборка фронтенда не найдена. Выполните: cd predictive_maintenance_dashboard && npm run build",
    }


@app.get("/{path:path}", include_in_schema=False)
async def serve_spa(path: str):
    """SPA fallback — отдаёт index.html для всех фронтенд-роутов (/login, /dashboard и т.д.)."""
    # Исключаем API-роуты и статику — они обрабатываются раньше
    if path.startswith("api/") or path.startswith("docs") or path.startswith("auth/"):
        # API и docs — вернуть 404 вместо index.html
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("Not Found", status_code=404)
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return PlainTextResponse("Not Found", status_code=404)



# ============================================================================
# Точка входа
# ============================================================================
if __name__ == "__main__":
    line = "=" * 62
    print(line)
    print("Предиктивный дашборд — Backend API")
    print(line)
    for boiler in FLEET.boilers:
        service = boiler.service
        print(f"{boiler.tpp['name']} / {boiler.unit['name']}")
        print(f"  Данные: {service.hourly_path}")
        print(f"  Модель: {service.model_root}; вид: {service.view_name}")
        print(f"  История: {service.history_h or 'вся доступная'} ч; окно: {service.window_h} ч")
        try:
            months = sorted(service.months())
            print(f"  Модели месяцев: {len(months)} (без них прошлые месяцы не оцениваются)")
        except DataUnavailable as exc:
            print(f"  Модель недоступна: {exc}")
    print(f"SPA (dist):  {'найден' if (DIST_DIR / 'index.html').exists() else 'не собран'}")
    print(f"Дашборд:    http://localhost:{PORT}")
    print(f"API-доки:   http://localhost:{PORT}/docs")
    print(line)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
