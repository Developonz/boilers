#!/usr/bin/env bash
# ============================================================================
#  Предиктивный дашборд — запуск (Linux / macOS)
#  Использование:
#     ./start.sh              запуск backend (дашборд открывается на :8000)
#     ./start.sh --dev        backend + Vite dev-сервер для разработки
#     ./start.sh --check      только проверка окружения
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"

echo "============================================================"
echo " Предиктивный дашборд — запуск"
echo " Python FastAPI + React"
echo "============================================================"
echo

# --- Поиск Python -----------------------------------------------------------
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "ОШИБКА: Python не найден. Установите Python 3.9+"
  exit 1
fi
echo "[1/4] Python: $PY ($($PY --version 2>&1))"

# --- Данные -----------------------------------------------------------------
HOURLY_FILE="$(find data -name hourly.parquet 2>/dev/null | head -1)"
if [ -n "$HOURLY_FILE" ]; then
  echo "[2/4] Данные: $HOURLY_FILE"
else
  echo "[2/4] ПРЕДУПРЕЖДЕНИЕ: не найден data/hourly.parquet (или задайте HOURLY_PATH в backend/.env)"
fi
MODEL_CFG="$(find models -maxdepth 2 -name config.json 2>/dev/null | head -1)"
if [ -n "$MODEL_CFG" ]; then
  echo "      Модель: $(dirname "$MODEL_CFG")"
else
  echo "      ПРЕДУПРЕЖДЕНИЕ: в папке models нет модели (папки с config.json и model.py)"
fi

# --- Зависимости ------------------------------------------------------------
if $PY -c "import fastapi, uvicorn, pandas, pyarrow, sklearn" >/dev/null 2>&1; then
  echo "[3/4] Зависимости backend установлены"
else
  echo "[3/4] Установка зависимостей backend..."
  $PY -m pip install -q -r backend/requirements.txt
fi

if [ "${1:-}" = "--check" ]; then
  echo
  echo "Проверка окружения завершена успешно."
  exit 0
fi

# --- Режим разработки -------------------------------------------------------
if [ "${1:-}" = "--dev" ]; then
  FRONTEND_DIR="predictive_maintenance_dashboard"
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "Установка зависимостей фронтенда (npm install)..."
    (cd "$FRONTEND_DIR" && npm install)
  fi
  echo "Запуск Vite dev-сервера: http://localhost:5173"
  (cd "$FRONTEND_DIR" && npm run dev) &
  DEV_PID=$!
  trap 'kill $DEV_PID 2>/dev/null || true' EXIT
fi

# --- Запуск backend ---------------------------------------------------------
echo
echo "[4/4] Запуск backend..."
echo
echo "  Дашборд:   http://localhost:${PORT}"
echo "  API-доки:  http://localhost:${PORT}/docs"
echo "  Остановка: Ctrl+C"
echo

export PORT
cd backend
exec $PY main.py
