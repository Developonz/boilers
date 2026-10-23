FROM node:20-alpine AS frontend-build

WORKDIR /build/frontend
COPY predictive_maintenance_dashboard/package.json predictive_maintenance_dashboard/package-lock.json ./
RUN npm ci
COPY predictive_maintenance_dashboard/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r backend/requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/
COPY models/ ./models/
COPY --from=frontend-build /build/frontend/dist/ ./predictive_maintenance_dashboard/dist/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=6s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)"]

WORKDIR /app/backend
CMD ["python", "main.py"]
