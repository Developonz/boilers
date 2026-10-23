@echo off
chcp 65001 >nul
setlocal
title Предиктивный дашборд - Котлоагрегат
cd /d "%~dp0"

echo ============================================================
echo  Предиктивный дашборд — запуск
echo  Python FastAPI + React
echo ============================================================
echo.

rem --- Поиск Python -----------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ОШИБКА] Python не найден в PATH.
  echo Установите Python 3.9+ с https://www.python.org/downloads/
  pause
  exit /b 1
)
echo [1/4] Python найден:
%PY% --version

rem --- Проверка данных --------------------------------------------------
echo.
dir /s /b "data\hourly.parquet" >nul 2>nul
if errorlevel 1 (
  echo [2/4] [ПРЕДУПРЕЖДЕНИЕ] Не найден data\hourly.parquet.
  echo       Положите его в папку data или задайте HOURLY_PATH в backend\.env.
) else (
  echo [2/4] Данные найдены: hourly.parquet
)
dir /s /b "models\config.json" >nul 2>nul
if errorlevel 1 (
  echo       [ПРЕДУПРЕЖДЕНИЕ] В папке models нет модели ^(папки с config.json и model.py^).
) else (
  echo       Модель найдена в папке models
)

rem --- Проверка зависимостей --------------------------------------------
echo.
%PY% -c "import fastapi, uvicorn, pandas, pyarrow, sklearn" >nul 2>nul
if errorlevel 1 (
  echo [3/4] Установка зависимостей backend...
  %PY% -m pip install -r backend\requirements.txt
  if errorlevel 1 (
    echo [ОШИБКА] Не удалось установить зависимости.
    pause
    exit /b 1
  )
) else (
  echo [3/4] Зависимости backend установлены
)

rem --- Режим проверки ---------------------------------------------------
if /i "%~1"=="--check" (
  echo.
  echo Проверка окружения завершена успешно.
  exit /b 0
)

rem --- Запуск -----------------------------------------------------------
echo.
echo [4/4] Запуск сервера...
echo.
echo   Дашборд:  http://localhost:8000
echo   API-доки: http://localhost:8000/docs
echo   Остановка: Ctrl+C
echo.
echo ============================================================
echo.

if /i "%~1"=="--no-browser" goto run
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8000'"

:run
cd backend
%PY% main.py
