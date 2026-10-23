/**
 * Предиктивный дашборд — главный компонент React.
 *
 * Все блоки работают на данных Python/FastAPI бэкенда; норма, отклонения, уровни и индекс здоровья
 * считает модель аномалий на выбранный момент — фильтр «Момент» в шапке, по умолчанию конец файла.
 * Тексты интерфейса — для эксплуатации, без терминов модели:
 *   Детализация — индекс здоровья, коэффициенты отклонения параметров, предупреждения,
 *                состояния механизмов, диагностика каналов;
 *   Параметры  — поиск, фильтры, сортировка, переход к деталям;
 *   Алерты     — тревоги по котлу и параметры вне нормы режима;
 *   Модальное окно — оценка параметра, график с нормой режима, рекомендация.
 */
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import './styles.css';
import { getAccessToken } from './lib/auth';

const INTERPRETATION_LABELS = {
  mode_change: 'Смена режима',
  transition: 'Переходный процесс',
  measurement_fault: 'Возможная проблема измерения',
  process_change: 'Изменение состояния узла',
  pre_failure: 'Признаки предаварийного состояния',
};

// ============================================================================
// Конфигурация
// ============================================================================

/** Fetch wrapper — автоматически подставляет Bearer-токен из cookie.
 *  Все API-вызовы дашборда проходят через эту функцию. */
async function authFetch(url, options = {}) {
  const token = getAccessToken();
  const headers = {
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  return fetch(url, { ...options, headers });
}

// ============================================================================
// Конфигурация
// ============================================================================
const DASHBOARD_CACHE_TTL_MS = 60 * 60 * 1000;
// Ссылка на другой дашборд в шапке (пока фиктивная): VITE_OTHER_DASHBOARD_URL / VITE_OTHER_DASHBOARD_LABEL при сборке
const ENV = (import.meta && import.meta.env) || {};
const OTHER_DASHBOARD = {
  url: ENV.VITE_OTHER_DASHBOARD_URL || '#',
  label: ENV.VITE_OTHER_DASHBOARD_LABEL || 'Другой дашборд',
};
// Границы коэффициента отклонения (если бэкенд не передал свои)
const DEFAULT_LEVEL_THRESHOLDS = { V: 1, B: 2, A: 3 };
const DASHBOARD_PREFETCH_WORKERS = 3;

function dashboardCacheKey(tppId, kaId, asOf) {
  return `${tppId}:${kaId}:${asOf || 'end'}`;
}

/** ISO-время бэкенда (без часового пояса) -> значение поля datetime-local. */
function toInputValue(iso) {
  return iso ? String(iso).slice(0, 16) : '';
}

/** Базовый адрес API: всегда тот же origin, что и страница (SPA отдаёт тот же backend).
 *  Важно: не подставлять http://<host>:8000 — со страницы, открытой через HTTPS-прокси
 *  (443), браузер заблокирует такие запросы как mixed content.
 *  Если API живёт на отдельном адресе — задайте VITE_API_BASE при сборке. */
function resolveApiBase() {
  if (typeof window !== 'undefined' && window.location) {
    const { protocol } = window.location;
    if (protocol === 'http:' || protocol === 'https:') return '';
  }
  return 'http://localhost:8000';
}

const API_BASE =
  (import.meta && import.meta.env && import.meta.env.VITE_API_BASE) || resolveApiBase();

const RISK_INFO = {
  A: { label: 'А', name: 'критический', className: 'risk-a', barClass: 'bar-a' },
  B: { label: 'Б', name: 'высокий', className: 'risk-b', barClass: 'bar-b' },
  V: { label: 'В', name: 'значимый', className: 'risk-v', barClass: 'bar-v' },
  G: { label: 'Г', name: 'норма', className: 'risk-g', barClass: 'bar-g' },
  N: { label: '—', name: 'нет оценки', className: 'risk-n', barClass: 'bar-n' },
};

const RISK_LEVELS = ['A', 'B', 'V', 'G', 'N'];
const ALERT_LEVELS = ['A', 'B', 'V'];

const SORT_OPTIONS = [
  { value: 'risk', label: 'По отклонению' },
  { value: 'change', label: 'По изменению' },
  { value: 'name', label: 'По названию' },
  { value: 'current', label: 'По значению' },
  { value: 'category', label: 'По категории' },
];

// ============================================================================
// Хелперы форматирования (устойчивы к null / NaN / строкам)
// ============================================================================
function isNum(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
}

function formatValue(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '--';
  if (!isNum(value)) return String(value);
  const num = Number(value);
  if (Math.abs(num) >= 10000) {
    return num.toLocaleString('ru-RU', { maximumFractionDigits: digits });
  }
  return num.toFixed(digits);
}

function formatSigned(value, digits = 2) {
  if (!isNum(value)) return '--';
  const num = Number(value);
  return `${num > 0 ? '+' : ''}${num.toFixed(digits)}`;
}

function formatPercent(value, digits = 1) {
  if (!isNum(value)) return '--';
  return `${formatSigned(value, digits)} %`;
}

function formatDateTime(iso) {
  if (!iso) return '--';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatTime(iso) {
  if (!iso) return '--';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
}

function trendClass(direction) {
  if (direction === 'increasing') return 'trend-up';
  if (direction === 'decreasing') return 'trend-down';
  return 'trend-flat';
}

function riskOf(level) {
  return RISK_INFO[level] || RISK_INFO.N;
}

function formatRatio(value, digits = 2) {
  return isNum(value) ? `×${Number(value).toFixed(digits)}` : '—';
}

/** Норма режима или пометка «режимный параметр» / «—». */
function normLabel(parameter) {
  if (parameter?.norm?.label) return parameter.norm.label;
  return parameter?.model?.roleLabel || '—';
}

async function apiError(response) {
  try {
    const payload = await response.json();
    if (payload?.detail) return String(payload.detail);
    if (payload?.error) return String(payload.error);
  } catch {
    // Ответ мог быть не JSON.
  }
  return `Сервер вернул код ${response.status}`;
}

/** Короткая строка оценки для карточек параметров. */
function ModelNote({ parameter }) {
  const model = parameter?.model;
  if (!model) return null;
  if (!model.evaluated) {
    return <p className="sensor-model-note is-muted">{model.notEvaluatedReason || model.roleLabel}</p>;
  }
  return (
    <p className="sensor-model-note">
      Ожидаемое {formatValue(model.expected, 2)} {parameter.unit} · отклонение {formatRatio(model.errorRatio)}
    </p>
  );
}

// ============================================================================
// Базовые элементы интерфейса
// ============================================================================

/** Badge — индикатор уровня риска. */
function Badge({ level }) {
  const risk = riskOf(level);
  return (
    <span className={`badge ${risk.className}`} title={`Уровень «${risk.label}» — ${risk.name}`}>
      {risk.label}
    </span>
  );
}

function LoadingSpinner({ text = 'Загрузка данных...' }) {
  return (
    <div className="loading-spinner">
      <div className="spinner" />
      <p>{text}</p>
    </div>
  );
}

function ErrorMessage({ message, onRetry }) {
  return (
    <div className="error-message">
      <span className="error-icon" role="img" aria-label="Ошибка">
        ⚠️
      </span>
      <p>{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry}>
          Повторить
        </button>
      )}
    </div>
  );
}

function EmptyState({ text, icon = '🛰️' }) {
  return (
    <div className="alerts-empty">
      <span role="img" aria-label="Нет данных">
        {icon}
      </span>
      <p>{text}</p>
    </div>
  );
}

/** ScoreRing — круговой индикатор индекса здоровья 0…100 (цвет: ≥ 75 зелёный, ≥ 50 жёлтый, ниже — красный). */
function ScoreRing({ score, size = 96 }) {
  const circumference = 251.2;
  const safeScore = isNum(score) ? Math.max(0, Math.min(100, Math.round(Number(score)))) : 0;
  const offset = circumference * (1 - safeScore / 100);
  const color = !isNum(score) ? '#7f91aa' : safeScore >= 75 ? '#30ae7f' : safeScore >= 50 ? '#f5b85b' : '#ef5555';
  return (
    <div className="health-ring" style={{ width: size, height: size, flexBasis: size, '--ring-size': `${size}px` }} role="img"
      aria-label={isNum(score) ? `Индекс здоровья ${safeScore} из 100` : 'Индекс здоровья: нет оценки'}>
      <svg viewBox="0 0 96 96" aria-hidden="true">
        <circle className="ring-track" cx="48" cy="48" r="40" />
        <circle
          className="ring-progress"
          cx="48"
          cy="48"
          r="40"
          style={{ stroke: color }}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="ring-label">
        <strong>{isNum(score) ? safeScore : '--'}</strong>
        <span>из 100</span>
      </div>
    </div>
  );
}

/** HealthRing — индекс здоровья агрегата (ниже 50 — тревога). */
function HealthRing({ score, trend, equipment }) {
  return (
    <div className="health-content">
      <ScoreRing score={score} />
      <div>
        <p className="metric-delta warning">{trend || 'Нет данных'}</p>
        <p className="metric-note">
          Индекс здоровья {equipment || ''}: 100 — как в нормальной работе, ниже 50 — тревога
        </p>
      </div>
    </div>
  );
}

/** Подпись диапазона коэффициента отклонения для буквы (А ≥ 3, Б 2–3, В 1–2, Г < 1). */
function levelRange(level, thresholds) {
  const t = { ...DEFAULT_LEVEL_THRESHOLDS, ...(thresholds || {}) };
  const f = (x) => String(x).replace('.', ',');
  if (level === 'A') return `≥ ${f(t.A)}`;
  if (level === 'B') return `${f(t.B)}–${f(t.A)}`;
  if (level === 'V') return `${f(t.V)}–${f(t.B)}`;
  if (level === 'G') return `< ${f(t.V)}`;
  return 'нет оценки';
}

/** AutoFitText — текст максимально крупный, но помещающийся в ширину карточки (подстраивается при изменении размера). */
function AutoFitText({ children, max = 30, min = 11, className = '' }) {
  const ref = useRef(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const fit = () => {
      let lo = min;
      let hi = max;
      el.style.fontSize = `${hi}px`;
      if (el.scrollWidth <= el.clientWidth) return;
      while (hi - lo > 0.5) {
        const mid = (lo + hi) / 2;
        el.style.fontSize = `${mid}px`;
        if (el.scrollWidth <= el.clientWidth) lo = mid; else hi = mid;
      }
      el.style.fontSize = `${lo}px`;
    };
    fit();
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(fit) : null;
    observer?.observe(el);
    return () => observer?.disconnect();
  }, [children, max, min]);
  return <strong ref={ref} className={`auto-fit ${className}`}>{children}</strong>;
}

/** RiskBars — распределение параметров по коэффициенту отклонения от нормы режима. */
function RiskBars({ byRisk, activeRisk, onSelect, thresholds }) {
  const total = RISK_LEVELS.reduce((sum, level) => sum + (byRisk?.[level] || 0), 0);
  return (
    <div className="risk-bars">
      {RISK_LEVELS.map((level) => {
        const info = riskOf(level);
        const count = byRisk?.[level] || 0;
        const percent = total ? Math.round((count / total) * 100) : 0;
        const isActive = activeRisk === level;
        return (
          <button
            key={level}
            type="button"
            className={`risk-bar ${isActive ? 'is-active' : ''}`}
            onClick={() => onSelect(isActive ? 'all' : level)}
            title={`Показать параметры с коэффициентом отклонения ${levelRange(level, thresholds)} (${info.label} — ${info.name})`}
          >
            <span className={info.barClass}>{info.label}</span>
            <span className="bar-track">
              <span className={`bar-fill ${info.barClass}`} style={{ width: `${percent}%` }} />
            </span>
            <span className="bar-meta">
              <strong>{count}</strong> <span className="muted">{info.name}{level !== 'N' ? ` · ${levelRange(level, thresholds)}` : ''}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** CategoryBlocks — сводка по категориям параметров (кликабельно). */
function CategoryBlocks({ categories, activeCategory, onSelect }) {
  if (!categories || categories.length === 0) {
    return <EmptyState text="Категории параметров не определены" icon="📚" />;
  }
  return (
    <div className="category-grid">
      {categories.map((category) => {
        const isActive = activeCategory === category.category;
        return (
          <button
            key={category.category}
            type="button"
            className={`category-card ${isActive ? 'is-active' : ''}`}
            onClick={() => onSelect(isActive ? 'all' : category.category)}
            title={`Показать параметры категории «${category.label}»`}
          >
            <div className="category-card-header">
              <span className="category-card-title">{category.label}</span>
              <Badge level={category.worstRisk} />
            </div>
            <p className="category-card-value">{category.total}</p>
            <p className="category-card-note">
              параметров
              {category.alerts > 0
                ? ` • ${category.alerts} с отклонениями`
                : category.evaluated ? ' • отклонений нет' : ''}
            </p>
          </button>
        );
      })}
    </div>
  );
}

/** TrendSummary — распределение направлений тренда. */
function TrendSummary({ trendCounts }) {
  const items = [
    { key: 'increasing', label: 'Растут', className: 'trend-up' },
    { key: 'decreasing', label: 'Падают', className: 'trend-down' },
    { key: 'stable', label: 'Стабильны', className: 'trend-flat' },
  ];
  return (
    <div className="trend-summary">
      {items.map((item) => (
        <div key={item.key} className="trend-summary-item">
          <strong className={item.className}>{trendCounts?.[item.key] ?? 0}</strong>
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
}

/** MiniChart — значения по часам и коридор нормы режима (ожидание ± обычная ошибка × разброс). */
function MiniChart({ timeline, unit = '' }) {
  const points = useMemo(() => (Array.isArray(timeline) ? timeline.filter(Boolean) : []), [timeline]);
  const values = points.filter((point) => isNum(point.value)).map((point) => Number(point.value));

  if (values.length === 0) {
    return <div className="mini-chart-empty">Нет данных для построения графика</div>;
  }

  const W = 720;
  const H = 240;
  const pad = { top: 20, right: 18, bottom: 30, left: 66 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  const bounds = [...values];
  points.forEach((point) => {
    ['low', 'high', 'expected'].forEach((key) => {
      if (isNum(point[key])) bounds.push(Number(point[key]));
    });
  });
  let lo = Math.min(...bounds);
  let hi = Math.max(...bounds);
  if (hi - lo < 1e-9) {
    lo -= 1;
    hi += 1;
  }
  const margin = (hi - lo) * 0.06;
  lo -= margin;
  hi += margin;
  const span = hi - lo;
  const x = (index) =>
    pad.left + (points.length === 1 ? innerW / 2 : (index / (points.length - 1)) * innerW);
  const y = (value) => pad.top + innerH - ((Number(value) - lo) / span) * innerH;

  // Отрезки подряд идущих часов, где поле есть (котёл стоял или оценки не было — разрыв)
  const runs = (has) => {
    const result = [];
    let current = [];
    points.forEach((point, index) => {
      if (has(point)) {
        current.push(index);
      } else if (current.length) {
        result.push(current);
        current = [];
      }
    });
    if (current.length) result.push(current);
    return result;
  };
  const halfStep = Math.max(3, innerW / Math.max(points.length - 1, 1) / 2);
  const corridor = runs((point) => isNum(point.low) && isNum(point.high)).map((idx) => {
    if (idx.length === 1) {
      const i = idx[0];
      return `M${x(i) - halfStep},${y(points[i].high)} L${x(i) + halfStep},${y(points[i].high)} `
        + `L${x(i) + halfStep},${y(points[i].low)} L${x(i) - halfStep},${y(points[i].low)} Z`;
    }
    const top = idx.map((i, k) => `${k ? 'L' : 'M'}${x(i).toFixed(2)},${y(points[i].high).toFixed(2)}`).join(' ');
    const bottom = [...idx].reverse().map((i) => `L${x(i).toFixed(2)},${y(points[i].low).toFixed(2)}`).join(' ');
    return `${top} ${bottom} Z`;
  });
  const linePath = (key) => runs((point) => isNum(point[key]))
    .map((idx) => idx.map((i, k) => `${k ? 'L' : 'M'}${x(i).toFixed(2)},${y(points[i][key]).toFixed(2)}`).join(' '))
    .join(' ');
  const valuePath = linePath('value');
  const expectedPath = linePath('expected');
  const isOutside = (point) => isNum(point.value) && isNum(point.low) && isNum(point.high)
    && (Number(point.value) < Number(point.low) || Number(point.value) > Number(point.high));
  const outsideCount = points.filter(isOutside).length;
  const gridValues = [hi, (hi + lo) / 2, lo];
  const digits = (value) => (Math.abs(value) >= 100 ? 0 : 2);

  return (
    <div className="chart-wrap">
      <svg className="mini-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="График параметра и нормы режима">
        {corridor.map((d, index) => <path key={`corridor-${index}`} className="chart-corridor" d={d} />)}

        {gridValues.map((value, index) => (
          <g key={`grid-${index}`}>
            <line className="chart-grid" x1={pad.left} y1={y(value)} x2={W - pad.right} y2={y(value)} />
            <text className="chart-axis-label" x={pad.left - 10} y={y(value) + 4} textAnchor="end">
              {value.toFixed(digits(value))}
            </text>
          </g>
        ))}

        {expectedPath && <path className="chart-expected" d={expectedPath} />}
        <path className="chart-line" d={valuePath} />

        {points.map((point, index) => (isNum(point.value) ? (
          <circle
            key={`point-${index}`}
            className={`chart-dot ${isOutside(point) ? 'is-out' : ''}`}
            cx={x(index)}
            cy={y(point.value)}
            r={points.length > 40 ? 2.4 : 4.2}
          >
            <title>
              {formatDateTime(point.time)} — {formatValue(point.value, 2)} {unit}
              {isNum(point.expected) ? `\nожидание ${formatValue(point.expected, 2)} ${unit}` : ''}
              {isNum(point.low) ? `\nнорма режима ${formatValue(point.low, 2)}…${formatValue(point.high, 2)}` : ''}
              {isNum(point.error) ? `\nотклонение ${formatRatio(point.error)} от обычного` : ''}
            </title>
          </circle>
        ) : null))}

        <text className="chart-axis-label" x={pad.left} y={H - 8}>
          {formatDateTime(points[0].time)}
        </text>
        <text className="chart-axis-label" x={W - pad.right} y={H - 8} textAnchor="end">
          {formatDateTime(points[points.length - 1].time)}
        </text>
      </svg>
      <p className="chart-caption">
        Часов: {points.length} • минимум {formatValue(Math.min(...values), 2)} {unit} • максимум{' '}
        {formatValue(Math.max(...values), 2)} {unit}
        {corridor.length
          ? ` • полоса — норма для текущего режима, пунктир — ожидаемое значение; вне нормы: ${outsideCount} ч`
          : ' • норма для этого параметра не задаётся'}
      </p>
    </div>
  );
}

/** Компактный график для сводных карточек и каталога датчиков. */
function Sparkline({ values = [], tone = 'blue', label = 'Динамика показателя' }) {
  const points = (Array.isArray(values) ? values : []).filter(isNum).map(Number);
  if (points.length < 2) return <div className="sparkline-placeholder" />;
  const width = 180;
  const height = 52;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const coords = points.map((value, index) => ({
    x: (index / (points.length - 1)) * width,
    y: 5 + (height - 10) * (1 - ((value - min) / span)),
  }));
  const line = coords.map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ');
  const area = `${line} L${width},${height} L0,${height} Z`;
  return (
    <svg className={`sparkline sparkline-${tone}`} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      <path className="sparkline-area" d={area} />
      <path className="sparkline-line" d={line} />
      <circle className="sparkline-last" cx={coords[coords.length - 1].x} cy={coords[coords.length - 1].y} r="3.2" />
    </svg>
  );
}

/** Цвет индекса здоровья: < 50 — тревога, < 75 — внимание. */
function healthTone(score) {
  if (!isNum(score)) return 'blue';
  if (Number(score) < 50) return 'red';
  if (Number(score) < 75) return 'amber';
  return 'green';
}

function statusTone(status, score) {
  if (status === 'critical') return 'red';
  if (status === 'warning') return 'amber';
  if (status === 'offline' || status === 'unknown') return 'blue';
  return healthTone(score);
}

/** Первая страница — единая сводка по всем станциям. */
function FleetOverview({ data, loading, error, onRetry, onOpenUnit }) {
  if (loading && !data) return <LoadingSpinner text="Собираем сводку по всем станциям…" />;
  if (error && !data) return <ErrorMessage message={error} onRetry={onRetry} />;
  if (!data) return <EmptyState text="Сводные данные пока недоступны" icon="🏭" />;

  const summary = data.summary || {};
  const monitoredStation = (data.stations || []).find((station) => station.status !== 'normal') || data.stations?.[0];
  const monitoredUnit = monitoredStation?.units?.find((unit) => unit.status === 'critical')
    || monitoredStation?.units?.find((unit) => unit.status === 'warning')
    || monitoredStation?.units?.[0];
  const metrics = [
    {
      label: 'Средний индекс здоровья',
      value: isNum(summary.averageHealth) ? formatValue(summary.averageHealth, 0) : '--',
      note: isNum(summary.averageHealth) ? '100 — как в нормальной работе, ниже 50 — тревога' : 'нет оценки',
      tone: healthTone(summary.averageHealth),
      ring: true,
    },
    {
      label: 'Агрегаты в работе',
      value: `${summary.unitsRunning ?? summary.unitsOnline ?? 0}/${summary.unitsTotal ?? 0}`,
      note: isNum(summary.runningPct ?? summary.availabilityPct)
        ? `${formatValue(summary.runningPct ?? summary.availabilityPct, 1)}% агрегатов в работе`
        : 'режим работы не получен',
      tone: 'green',
    },
    {
      label: 'Агрегаты с отклонениями',
      value: summary.unitsWithDeviations ?? ((summary.unitsAttention ?? 0) + (summary.unitsCritical ?? 0)),
      note: `${summary.unitsCritical ?? 0} с тревогой`,
      tone: summary.unitsCritical ? 'red' : summary.unitsAttention ? 'amber' : 'green',
    },
    {
      label: 'Суммарная электрическая мощность',
      value: isNum(summary.totalElectricPowerMw ?? summary.totalGenerationMw)
        ? `${formatValue(summary.totalElectricPowerMw ?? summary.totalGenerationMw, 1)} МВт`
        : '--',
      note: isNum(summary.averageBoilerGrossEfficiencyPct ?? summary.efficiencyPct)
        ? `Средний КПД котлоагрегатов брутто — ${formatValue(summary.averageBoilerGrossEfficiencyPct ?? summary.efficiencyPct, 1)}%`
        : 'канал КПД не получен',
      tone: 'blue',
    },
  ];

  return (
    <div className="fleet-view">
      {data.notice && <div className="fleet-notice">⚠ {data.notice}</div>}

      <section className="fleet-metrics" aria-label="Общие показатели">
        {metrics.map((metric) => (
          <article className={`fleet-metric tone-${metric.tone} ${metric.ring ? 'has-ring' : ''}`} key={metric.label}>
            <div>
              <span>{metric.label}</span>
              {metric.ring ? null : <strong>{metric.value}</strong>}
              <small>{metric.note}</small>
            </div>
            {metric.ring ? <ScoreRing score={summary.averageHealth} size={78} /> : null}
          </article>
        ))}
      </section>

      {monitoredUnit?.sensors?.length > 0 && (
        <section className="fleet-sensors" aria-label="Ключевые технологические параметры">
          <div className="fleet-sensors-heading">
            <div><span>Ключевые технологические параметры</span><strong>{monitoredUnit.name}</strong></div>
            <button type="button" onClick={() => onOpenUnit(monitoredStation.id, monitoredUnit.id)}>Открыть агрегат →</button>
          </div>
          <div className="fleet-sensor-grid">
            {monitoredUnit.sensors.slice(0, 4).map((sensor) => {
              const tone = levelTone(sensor.riskLevel);
              return (
                <article className={`fleet-sensor tone-${tone}`} key={sensor.id}>
                  <span>{sensor.shortName || sensor.name}</span>
                  <div><strong>{formatValue(sensor.value, 1)}</strong><small>{sensor.unit}</small></div>
                  <Sparkline values={sensor.sparkline} tone={tone} label={`Динамика ${sensor.name}`} />
                  <p className="fleet-sensor-norm">
                    {sensor.norm?.label ? `норма режима: ${sensor.norm.label}` : sensor.role === 'condition' ? 'режимный параметр' : '—'}
                  </p>
                </article>
              );
            })}
          </div>
        </section>
      )}

      <section className="dashboard-section fleet-stations-section">
        <div className="section-heading modern-heading">
          <div>
            <p className="eyebrow">Станции</p>
            <h2>Карта эксплуатационного состояния</h2>
          </div>
          <span className="live-pill"><i /> данные обновлены {formatTime(data.generatedAt)}</span>
        </div>
        <div className="station-grid">
          {(data.stations || []).map((station) => (
            <article className="station-card" key={station.id}>
              <div className="station-card-head">
                <div>
                  <span className="station-caption">Тепловая электростанция</span>
                  <h3>{station.name}</h3>
                </div>
                <div className="station-ring">
                  <small>Индекс здоровья</small>
                  <ScoreRing score={station.healthScore} size={78} />
                </div>
              </div>
              <Sparkline values={station.trend} tone={statusTone(station.status, station.healthScore)} label={`Динамика ${station.name}`} />
              <div className="station-stats">
                <span><strong>{formatValue(station.electricPowerMw ?? station.generationMw, 1)}</strong>{isNum(station.electricPowerMw ?? station.generationMw) ? ' МВт эл. мощности' : ' эл. мощность не получена'}</span>
                <span><strong>{formatValue(station.loadPct, 1)}</strong>{isNum(station.loadPct) ? '% нагрузка' : ' нагрузка не получена'}</span>
                <span><strong>{station.activeAlerts}</strong> активных отклонений</span>
              </div>
              <div className="unit-list">
                {(station.units || []).map((unit) => (
                  <button type="button" className="unit-row" key={unit.id} onClick={() => onOpenUnit(station.id, unit.id)}>
                    <span className={`unit-dot tone-${statusTone(unit.status, unit.healthScore)}`} />
                    <span className="unit-name"><strong>{unit.name}</strong><small>Режим: {unit.operatingModeLabel || (unit.online ? 'В работе' : 'В резерве')}</small></span>
                    <span className="unit-technical"><small>Техсостояние</small><strong>{unit.technicalStatusLabel || unit.statusLabel || unit.status}</strong></span>
                    <span className={`unit-health tone-${statusTone(unit.status, unit.healthScore)}`}><small>Индекс</small>{isNum(unit.healthScore) ? unit.healthScore : '--'}</span>
                    <span className="unit-arrow">→</span>
                  </button>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function levelTone(level) {
  if (level === 'A') return 'red';
  if (level === 'B' || level === 'V') return 'amber';
  if (level === 'G') return 'green';
  return 'blue';
}

function SensorCard({ parameter, onSelect }) {
  const isNumeric = parameter.type === 'numeric';
  const tone = levelTone(parameter.riskLevel);
  const span = isNum(parameter.max) && isNum(parameter.min) ? Number(parameter.max) - Number(parameter.min) : 0;
  const position = span > 0 && isNum(parameter.current)
    ? Math.max(3, Math.min(97, ((Number(parameter.current) - Number(parameter.min)) / span) * 100))
    : 50;
  return (
    <article className={`sensor-card tone-${tone}`} tabIndex={0} role="button" onClick={() => onSelect(parameter.name)} onKeyDown={(event) => event.key === 'Enter' && onSelect(parameter.name)}>
      <div className="sensor-card-head">
        <span className="sensor-category">{parameter.categoryLabel}</span>
        <Badge level={parameter.riskLevel} />
      </div>
      <h3>{parameter.name}</h3>
      <div className="sensor-reading">
        <strong>{isNumeric ? formatValue(parameter.current, 2) : (parameter.currentLabel || parameter.current || '--')}</strong>
        {isNumeric && parameter.unit ? <span>{parameter.unit}</span> : null}
      </div>
      {isNumeric ? (
        <>
          <Sparkline values={parameter.sparkline} tone={tone} label={`Динамика ${parameter.name}`} />
          <div className="sensor-range" aria-label="Положение в диапазоне">
            <span style={{ left: `${position}%` }} />
          </div>
        </>
      ) : (
        <div className={`sensor-state-visual ${parameter.unusual ? 'is-unusual' : ''}`}>
          <i /> {parameter.usualState ? `Обычное состояние: ${parameter.usualState}` : 'Состояние механизма'}
        </div>
      )}
      <div className="sensor-card-foot">
        <span>{isNumeric ? (parameter.norm ? `норма: ${parameter.norm.label}` : `${formatValue(parameter.min, 1)} — ${formatValue(parameter.max, 1)} ${parameter.unit}`) : 'Дискретный канал'}</span>
        <span className={trendClass(parameter.trendDirection)}>{isNumeric ? formatPercent(parameter.changePct) : parameter.trendLabel || 'стабильно'}</span>
      </div>
      {isNumeric && <ModelNote parameter={parameter} />}
    </article>
  );
}

/** KPICard — карточка ключевого показателя (клик открывает детали). */
function KPICard({ kpi, onClick }) {
  const isWarning = kpi.riskLevel === 'A' || kpi.riskLevel === 'B';
  return (
    <article
      className={`kpi-card ${isWarning ? 'warning-card' : ''}`}
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onClick();
        }
      }}
      title={`Открыть детали: ${kpi.parameterName}`}
    >
      <div className="kpi-header">
        <span className="kpi-name">{kpi.name}</span>
        <Badge level={kpi.riskLevel} />
      </div>
      <div className="kpi-value">
        <strong>{formatValue(kpi.value, 2)}</strong>
        {kpi.unit ? <span className="unit">{kpi.unit}</span> : null}
      </div>
      <div className="kpi-footer">
        <span className={`kpi-trend ${trendClass(kpi.trendDirection)}`}>
          {kpi.trend} ({formatPercent(kpi.changePct)})
        </span>
        <span className="kpi-norm" title="Норма для текущего режима">
          {kpi.norm ? `норма ${kpi.norm.label}` : kpi.role === 'condition' ? 'режимный параметр' : 'нет оценки'}
        </span>
      </div>
      {isNum(kpi.expected) && (
        <p className="sensor-model-note">Ожидаемое для режима: {formatValue(kpi.expected, 2)} {kpi.unit}</p>
      )}
    </article>
  );
}

/** AlertList — читаемый журнал предупреждений с фильтрами и пагинацией. */
function AlertList({ alerts, onSelect, filter, onFilterChange, compact = false }) {
  const list = alerts || [];
  const [localFilter, setLocalFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [signalType, setSignalType] = useState('all');
  const [sort, setSort] = useState('priority');
  const activeFilter = filter ?? localFilter;

  const updateFilter = (level) => {
    if (onFilterChange) onFilterChange(level);
    else setLocalFilter(level);
  };

  const categories = useMemo(() => {
    const values = new Map();
    list.forEach((alert) => {
      if (alert.category) values.set(alert.category, alert.categoryLabel || alert.category);
    });
    return Array.from(values, ([value, label]) => ({ value, label }))
      .sort((a, b) => a.label.localeCompare(b.label, 'ru'));
  }, [list]);

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const riskWeight = { A: 4, B: 3, V: 2, G: 1 };
    const result = list.filter((alert) => {
      if (activeFilter !== 'all' && alert.riskLevel !== activeFilter) return false;
      if (category !== 'all' && alert.category !== category) return false;
      if (signalType === 'model' && alert.kind !== 'model') return false;
      if (signalType === 'kpi' && !alert.isKpi) return false;
      if (signalType === 'sensor' && (alert.isKpi || alert.kind === 'model')) return false;
      if (normalizedQuery) {
        const searchable = [
          alert.parameter,
          alert.message,
          alert.categoryLabel,
          alert.recommendation,
          ...(alert.reasons || []),
          ...(alert.sensors || []).map((sensor) => sensor.name),
        ].join(' ').toLowerCase();
        if (!searchable.includes(normalizedQuery)) return false;
      }
      return true;
    });

    return [...result].sort((a, b) => {
      if (sort === 'change') return Math.abs(Number(b.changePct) || 0) - Math.abs(Number(a.changePct) || 0);
      if (sort === 'name') return String(a.parameter).localeCompare(String(b.parameter), 'ru');
      const levelDifference = (riskWeight[b.riskLevel] || 0) - (riskWeight[a.riskLevel] || 0);
      if (levelDifference) return levelDifference;
      const kindDifference = Number(b.kind === 'model') - Number(a.kind === 'model');
      return kindDifference || (Number(b.riskScore) || 0) - (Number(a.riskScore) || 0);
    });
  }, [list, activeFilter, category, signalType, query, sort]);

  const counts = ALERT_LEVELS.reduce((acc, level) => {
    acc[level] = list.filter((a) => a.riskLevel === level).length;
    return acc;
  }, {});

  const visible = filtered;

  const filtersActive = activeFilter !== 'all'
    || category !== 'all'
    || signalType !== 'all'
    || query.trim();

  const resetFilters = () => {
    updateFilter('all');
    setQuery('');
    setCategory('all');
    setSignalType('all');
    setSort('priority');
  };

  if (list.length === 0) {
    return <EmptyState text="Предупреждений нет — параметры в пределах нормы режима" icon="✅" />;
  }

  return (
    <div className={`alerts-panel ${compact ? 'alerts-panel-compact' : 'alerts-full'}`}>
      <div className="alerts-filter-panel">
        <div className="filter-row" aria-label="Фильтр по уровню предупреждения">
          <button
            type="button"
            className={`chip ${activeFilter === 'all' ? 'is-active' : ''}`}
            onClick={() => updateFilter('all')}
          >
            Все ({list.length})
          </button>
          {ALERT_LEVELS.map((level) => (
            <button
              key={level}
              type="button"
              className={`chip alert-filter-${level.toLowerCase()} ${activeFilter === level ? 'is-active' : ''}`}
              onClick={() => updateFilter(level)}
            >
              {riskOf(level).name} · {riskOf(level).label} ({counts[level]})
            </button>
          ))}
        </div>

        <div className="alerts-toolbar">
          <label className="search-field alert-search">
            <span className="sr-only">Поиск предупреждения</span>
            <input
              type="search"
              placeholder="Поиск по параметру, причине или рекомендации…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>

          <label className="select-field">
            <span>Категория</span>
            <select value={category} onChange={(event) => setCategory(event.target.value)}>
              <option value="all">Все категории</option>
              {categories.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
          </label>

          <label className="select-field">
            <span>Тип сигнала</span>
            <select value={signalType} onChange={(event) => setSignalType(event.target.value)}>
              <option value="all">Все сигналы</option>
              <option value="model">Тревоги по котлу</option>
              <option value="kpi">Только KPI</option>
              <option value="sensor">Остальные датчики</option>
            </select>
          </label>

          <label className="select-field">
            <span>Сортировка</span>
            <select value={sort} onChange={(event) => setSort(event.target.value)}>
              <option value="priority">Сначала критические</option>
              <option value="change">По величине изменения</option>
              <option value="name">По названию</option>
            </select>
          </label>

          {filtersActive && (
            <button type="button" className="action-button alert-reset" onClick={resetFilters}>
              Сбросить
            </button>
          )}
        </div>
      </div>

      <div className="alerts-result-line" aria-live="polite">
        <span>Найдено <strong>{filtered.length}</strong> из {list.length}</span>
      </div>

      {visible.length === 0 ? (
        <EmptyState text="По выбранным фильтрам предупреждений нет" icon="🔍" />
      ) : (
        <div className="alerts-list">
          {visible.map((alert) => (
            <article
              key={alert.id}
              className={`alert-item alert-${alert.riskLevel.toLowerCase()}`}
            >
              <div className="alert-severity" aria-hidden="true">
                <span>{alert.riskLevel === 'A' ? '!' : alert.riskLevel === 'B' ? '↑' : '•'}</span>
              </div>
              <div className="alert-body">
                <div className="alert-heading">
                  <div className="alert-title">
                    <span className="alert-category">{alert.categoryLabel || 'Телеметрия'}</span>
                    <h3>{alert.parameter}</h3>
                  </div>
                  <div className="alert-tags">
                    <Badge level={alert.riskLevel} />
                    {alert.isKpi ? <span className="tag">KPI</span> : null}
                  </div>
                </div>

                <p className="alert-message">{alert.message}</p>
                {alert.kind === 'model' && alert.sensors?.length > 0 && (
                  <div className="alert-sensors">
                    <span>Главные параметры:</span>
                    {alert.sensors.map((sensor) => (
                      <button type="button" key={sensor.column} onClick={() => onSelect(sensor.name)}>
                        {sensor.name} <strong>{formatRatio(sensor.ratio, 1)}</strong>
                      </button>
                    ))}
                  </div>
                )}
                {alert.recommendation && (
                  <p className="alert-recommendation">
                    <span>Рекомендуем</span>
                    {alert.recommendation}
                  </p>
                )}

                {alert.kind === 'model' ? (
                  <div className="alert-footer">
                    <div className="alert-reading">
                      <span>{alert.valueLabel || 'Индекс здоровья'}</span>
                      <strong>{isNum(alert.value) ? formatValue(alert.value, 0) : '--'} <small>из 100</small></strong>
                    </div>
                    <div className="alert-reading">
                      <span>Длительность</span>
                      <strong>{isNum(alert.hours) ? `${alert.hours} ч` : '—'}</strong>
                    </div>
                  </div>
                ) : (
                  <div className="alert-footer">
                    <div className="alert-reading">
                      <span>Текущее значение</span>
                      <strong>{formatValue(alert.value, 2)} <small>{alert.unit}</small></strong>
                    </div>
                    <div className="alert-reading">
                      <span>От ожидания для режима</span>
                      <strong className={Number(alert.deviation) < 0 ? 'trend-down' : 'trend-up'}>
                        {formatSigned(alert.deviation, 2)} <small>{alert.unit}</small>
                      </strong>
                    </div>
                    <div className="alert-reading">
                      <span>Норма режима</span>
                      <strong>{alert.norm?.label || '—'}</strong>
                    </div>
                    <button type="button" className="alert-open" onClick={() => onSelect(alert.parameter)}>
                      Открыть параметр <span aria-hidden="true">→</span>
                    </button>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      )}

    </div>
  );
}

/** StateCards — состояния механизмов (дымососы, вентиляторы и т.п.). */
function StateCards({ states, onSelect }) {
  if (!states || states.length === 0) {
    return <EmptyState text="Нет данных о состоянии механизмов" icon="⚙️" />;
  }
  return (
    <div className="state-grid">
      {states.map((state) => (
        <article
          key={state.column}
          className="state-card"
          role="button"
          tabIndex={0}
          onClick={() => onSelect(state.name)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              onSelect(state.name);
            }
          }}
          title="Открыть детали"
        >
          <div className="state-card-header">
            <span className="state-card-name">{state.name}</span>
            <span className={`state-code ${state.type === 'categorical' ? 'is-text' : ''} ${state.unusual ? 'is-unusual' : ''}`}>
              {state.currentLabel || '--'}
            </span>
          </div>
          <p className="state-card-note">
            {state.usualState ? `Обычное состояние: ${state.usualState}` : '—'}
          </p>
          <p className={`state-card-note ${state.unusual ? 'trend-up' : 'trend-flat'}`}>
            {state.unusual ? 'Не в обычном состоянии' : state.usualState ? 'В обычном состоянии' : '—'}
          </p>
        </article>
      ))}
    </div>
  );
}

/** DiagnosticsPanels — каналы без изменения значения и каналы без данных. */
function DiagnosticsPanels({ frozen, unavailable, missing, onSelect }) {
  const empty = [...(unavailable || []), ...(missing || [])];
  const hasFrozen = frozen && frozen.length > 0;
  if (!hasFrozen && empty.length === 0) {
    return <EmptyState text="Каналов без изменения значения и каналов без данных нет" icon="✅" />;
  }
  return (
    <div className="diagnostics-grid">
      {hasFrozen && (
        <div className="diagnostics-card">
          <div className="diagnostics-header">
            <span className="diagnostics-title">Каналы без изменения значения</span>
            <span className="diagnostics-count">{frozen.length}</span>
          </div>
          <p className="diagnostics-note">
            Значение не меняется ≥ 24 ч при работе котла у датчика, который обычно меняется каждый час, —
            возможен отказ датчика. Такие значения в оценку не берутся.
          </p>
          <ul className="diagnostics-list">
            {frozen.map((item) => (
              <li key={item.name}>
                <button type="button" onClick={() => onSelect(item.name)}>
                  {item.name}
                </button>
                <span>
                  {formatValue(item.value, 2)} {item.unit} · {item.hours} ч
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {empty.length > 0 && (
        <div className="diagnostics-card">
          <div className="diagnostics-header">
            <span className="diagnostics-title">Каналы без данных</span>
            <span className="diagnostics-count">{empty.length}</span>
          </div>
          <p className="diagnostics-note">
            Значений нет — канал не опрашивается или не передан в телеметрию.
          </p>
          <ul className="diagnostics-list muted-list">
            {empty.map((item) => (
              <li key={item.column}>
                <span className="static-name">{item.name}</span>
                <span>нет данных</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// Колонки таблицы параметров
const TABLE_COLUMNS = [
  { key: 'name', label: 'Параметр', sortable: true },
  { key: 'category', label: 'Категория', sortable: true },
  { key: 'current', label: 'Текущее', sortable: true, align: 'right' },
  { key: 'norm', label: 'Норма режима', sortable: false, align: 'right' },
  { key: 'risk', label: 'Коэффициент отклонения', sortable: true, align: 'right' },
  { key: 'change', label: 'Δ за окно', sortable: true, align: 'right' },
  { key: 'trend', label: 'Тренд', sortable: false },
  { key: 'level', label: 'Группа', sortable: false, align: 'center' },
];

/** ParameterTable — таблица параметров с сортировкой и переходом к деталям. */
function ParameterTable({ parameters, sortKey, sortDirection, onSort, onSelect }) {
  if (!parameters || parameters.length === 0) {
    return <EmptyState text="Параметры не найдены — измените условия фильтрации" icon="🔎" />;
  }
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {TABLE_COLUMNS.map((column) => (
              <th key={column.key} style={column.align ? { textAlign: column.align } : undefined}>
                {column.sortable ? (
                  <button type="button" className="th-sort" onClick={() => onSort(column.key)}>
                    {column.label}
                    <span className="sort-arrow">
                      {sortKey === column.key ? (sortDirection === 'asc' ? '▲' : '▼') : '↕'}
                    </span>
                  </button>
                ) : (
                  column.label
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {parameters.map((param) => (
            <tr
              key={param.column}
              tabIndex={0}
              className={param.riskLevel === 'A' ? 'critical-row clickable-row' : 'clickable-row'}
              onClick={() => onSelect(param.name)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') onSelect(param.name);
              }}
              title="Открыть детали параметра"
            >
              <td>
                <span className="parameter-cell">
                  {param.name}
                  <span>{param.type === 'state' ? 'Состояние механизма' : param.column}</span>
                </span>
              </td>
              <td>{param.categoryLabel}</td>
              <td className="numeric">
                {param.type === 'numeric' ? formatValue(param.current, 2) : (param.currentLabel || '--')}
                {param.unit ? <span className="unit">{param.unit}</span> : null}
              </td>
              <td className="numeric">{param.type === 'numeric' ? normLabel(param) : '—'}</td>
              <td className="numeric">{formatRatio(param.riskScore)}</td>
              <td className={`numeric ${trendClass(param.trendDirection)}`}>
                {formatPercent(param.changePct)}
              </td>
              <td className={`trend ${trendClass(param.trendDirection)}`}>{param.trend}</td>
              <td style={{ textAlign: 'center' }}>
                <Badge level={param.riskLevel} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** AsOfFilter — момент, на который показывается состояние (по умолчанию — конец файла данных). */
function AsOfFilter({ value, effective, range, onChange }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => { setDraft(value); }, [value]);
  // Поле даты меняет значение на каждом шаге ввода — пересчёт только после паузы
  useEffect(() => {
    if (draft === value) return undefined;
    const timer = window.setTimeout(() => onChange(draft), 700);
    return () => window.clearTimeout(timer);
  }, [draft, value, onChange]);

  return (
    <div className="asof-filter" title="Момент, на который показывается состояние котла; по умолчанию — конец файла данных">
      <label>
        <span>Момент</span>
        <input
          type="datetime-local"
          step="3600"
          value={draft || toInputValue(effective)}
          min={toInputValue(range?.minAsOf)}
          max={toInputValue(range?.end)}
          disabled={!range}
          onChange={(event) => setDraft(event.target.value)}
        />
      </label>
      <button type="button" onClick={() => { setDraft(''); onChange(''); }} disabled={!value && !draft}>
        Конец файла
      </button>
    </div>
  );
}

/** ParameterModal — оценка параметра, график с нормой режима, рекомендация. */
function ParameterModal({ parameterName, detail, loading, error, onClose, onRetry }) {
  useEffect(() => {
    const handleKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const isNumeric = detail?.type === 'numeric';
  const model = detail?.model || null;
  const distribution = detail?.distribution
    ? Object.entries(detail.distribution).map(([label, count]) => ({ label, count }))
    : [];

  return (
    <div className="modal-overlay" role="presentation" onClick={onClose}>
      <div
        className="modal-content wide"
        role="dialog"
        aria-modal="true"
        aria-label={parameterName}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2>{detail?.name || parameterName}</h2>
            {detail && (
              <p className="modal-subtitle">
                {detail.categoryLabel}
                {detail.unit ? ` • ${detail.unit}` : ''} •{' '}
                {detail.type === 'numeric'
                  ? 'числовой параметр'
                  : detail.type === 'state'
                    ? 'состояние механизма'
                    : 'текстовое состояние'}
              </p>
            )}
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Закрыть">
            ✕
          </button>
        </div>

        <div className="modal-body">
          {loading && <LoadingSpinner text="Загрузка параметра..." />}

          {!loading && error && <ErrorMessage message={error} onRetry={onRetry} />}

          {!loading && !error && detail && (
            <>
              <div className="modal-badges">
                <Badge level={detail.riskLevel} />
                <span className="modal-risk-name">Уровень: {riskOf(detail.riskLevel).name}</span>
                {isNum(detail.riskScore) && (
                  <span className="modal-risk-score">
                    отклонение {formatRatio(detail.riskScore)} от обычного
                  </span>
                )}
                {model?.roleLabel && <span className="tag">{model.roleLabel}</span>}
                {model?.mainInAlarm && <span className="tag warn">главный параметр тревоги</span>}
                {detail.frozen && <span className="tag warn">без изменения {detail.frozenHours} ч</span>}
                {detail.unusual && <span className="tag warn">не в обычном состоянии</span>}
              </div>

              {model?.evaluated && (
                <div className="anomaly-scale" aria-label="Шкала уровня отклонения">
                  <strong>
                    Уровень — во сколько раз отклонение от нормы режима больше обычного (в среднем за последние часы)
                  </strong>
                  <span>Г &lt; 1</span><span>В 1–2</span><span>Б 2–3</span><span>А ≥ 3</span>
                </div>
              )}

              {isNumeric ? (
                <div className="modal-stats">
                  <div className="stat-item">
                    <span className="stat-label">Текущее</span>
                    <span className="stat-value">{formatValue(detail.current, 2)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Ожидаемое для режима</span>
                    <span className="stat-value">{formatValue(model?.expected, 2)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Норма режима</span>
                    <span className="stat-value">{detail.norm?.label || '—'}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">От ожидания</span>
                    <span className={`stat-value ${Number(model?.deviation) < 0 ? 'trend-down' : 'trend-up'}`}>
                      {formatSigned(model?.deviation, 2)}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Вклад в отклонение котла</span>
                    <span className="stat-value">{isNum(model?.contributionPct) ? `${formatValue(model.contributionPct, 0)}%` : '—'}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Изменение за окно</span>
                    <span className={`stat-value ${trendClass(detail.trendDirection)}`}>
                      {formatPercent(detail.changePct)}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Минимум за окно</span>
                    <span className="stat-value">{formatValue(detail.min, 2)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Максимум за окно</span>
                    <span className="stat-value">{formatValue(detail.max, 2)}</span>
                  </div>
                </div>
              ) : (
                <div className="modal-stats">
                  <div className="stat-item">
                    <span className="stat-label">Текущее</span>
                    <span className="stat-value">{detail.currentLabel || '--'}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Обычное состояние</span>
                    <span className="stat-value">{detail.usualState || '—'}</span>
                  </div>
                </div>
              )}

              {detail.riskReasons && detail.riskReasons.length > 0 && (
                <div className="modal-block">
                  <h3>Оценка</h3>
                  <ul className="reason-list">
                    {detail.riskReasons.map((reason, index) => (
                      <li key={index}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="modal-block recommendation">
                <h3>Рекомендация</h3>
                <p>{detail.recommendation || 'Продолжить наблюдение за параметром.'}</p>
              </div>

              {isNumeric && (
                <div className="modal-chart">
                  <h3>Динамика за {detail.windowHours ?? '--'} ч: значение и норма режима</h3>
                  <MiniChart timeline={detail.timeline} unit={detail.unit} />
                </div>
              )}

              {!isNumeric && distribution.length > 0 && (
                <div className="modal-block">
                  <h3>Распределение значений за окно</h3>
                  <div className="distribution-list">
                    {distribution.map((item) => {
                      const total = distribution.reduce((sum, entry) => sum + entry.count, 0);
                      const percent = total ? Math.round((item.count / total) * 100) : 0;
                      return (
                        <div key={item.label} className="distribution-row">
                          <span className="distribution-label">{item.label}</span>
                          <span className="bar-track">
                            <span className="bar-fill bar-g" style={{ width: `${percent}%` }} />
                          </span>
                          <span className="distribution-value">
                            {item.count} ч • {percent}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="modal-meta">
                <span>
                  Канал источника: <code>{detail.column}</code>
                </span>
                <span>
                  Часов с данными: <strong>{detail.count ?? '--'}</strong>
                </span>
                <span>
                  Момент оценки: <strong>{formatDateTime(detail.asOf)}</strong>
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
export default function App() {
  const urlParams = useMemo(
    () => (typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : new URLSearchParams()),
    [],
  );

  const [dashboard, setDashboard] = useState(null);
  const [fleet, setFleet] = useState(null);
  const [fleetLoading, setFleetLoading] = useState(true);
  const [fleetError, setFleetError] = useState(null);
  const [tpps, setTpps] = useState([]);
  const [availableKas, setAvailableKas] = useState([]);
  const [selectedTpp, setSelectedTpp] = useState(() => urlParams.get('tpp') || '');
  const [selectedKa, setSelectedKa] = useState(() => urlParams.get('ka') || '');
  // Момент, на который показывается состояние ('' — конец файла)
  const [asOf, setAsOf] = useState(() => urlParams.get('at') || '');
  const [selectionLoading, setSelectionLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(() => {
    if (typeof window === 'undefined') return 'dark';
    const saved = window.localStorage.getItem('dashboard-theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });
  const [activeTab, setActiveTab] = useState(() => {
    const tab = urlParams.get('tab');
    if (['fleet', 'overview', 'parameters', 'alerts'].includes(tab)) return tab;
    return urlParams.get('tpp') && urlParams.get('ka') ? 'overview' : 'fleet';
  });

  const [search, setSearch] = useState(() => urlParams.get('search') || '');
  const [categoryFilter, setCategoryFilter] = useState(() => urlParams.get('category') || 'all');
  const [riskFilter, setRiskFilter] = useState(() => urlParams.get('risk') || 'all');
  const [sortKey, setSortKey] = useState('risk');
  const [sortDirection, setSortDirection] = useState('desc');
  const [parameterView, setParameterView] = useState('cards');

  const [alertFilter, setAlertFilter] = useState('all');

  const [selectedName, setSelectedName] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(null);

  const detailRequestRef = useRef(0);
  const dashboardRequestRef = useRef(0);
  const fleetRequestRef = useRef(0);
  const fleetAbortRef = useRef(null);
  const fleetRef = useRef(null);
  const dashboardCacheRef = useRef(new Map());
  const dashboardInflightRef = useRef(new Map());

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem('dashboard-theme', theme);
  }, [theme]);

  const fetchDashboardPayload = useCallback(async (tppId, kaId, { force = false } = {}) => {
    const key = dashboardCacheKey(tppId, kaId, asOf);
    const cached = dashboardCacheRef.current.get(key);
    const isFresh = cached && Date.now() - cached.cachedAt < DASHBOARD_CACHE_TTL_MS;
    if (!force && isFresh) return cached;

    const inflight = dashboardInflightRef.current.get(key);
    if (inflight) return inflight;

    const request = (async () => {
      const query = new URLSearchParams({ tpp_id: tppId, ka_id: kaId });
      if (asOf) query.set('at', asOf);
      const response = await authFetch(`${API_BASE}/api/dashboard?${query}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(await apiError(response));
      const payload = await response.json();
      const entry = { payload, cachedAt: Date.now() };
      dashboardCacheRef.current.set(key, entry);
      return entry;
    })();

    dashboardInflightRef.current.set(key, request);
    try {
      return await request;
    } finally {
      if (dashboardInflightRef.current.get(key) === request) {
        dashboardInflightRef.current.delete(key);
      }
    }
  }, [asOf]);

  const prepareDashboardSelection = useCallback((tppId, kaId) => {
    dashboardRequestRef.current += 1;
    const cached = dashboardCacheRef.current.get(dashboardCacheKey(tppId, kaId, asOf));
    if (cached) {
      setDashboard(cached.payload);
      setLoading(false);
    } else {
      setDashboard(null);
      setLoading(true);
    }
    setError(null);
    setSelectedName(null);
    setDetail(null);
  }, [asOf]);

  const loadFleet = useCallback(async () => {
    const requestId = ++fleetRequestRef.current;
    fleetAbortRef.current?.abort();
    const controller = new AbortController();
    fleetAbortRef.current = controller;
    setFleetLoading(true);
    try {
      const query = asOf ? `?${new URLSearchParams({ at: asOf })}` : '';
      const response = await authFetch(`${API_BASE}/api/fleet-overview${query}`, {
        cache: 'no-store', signal: controller.signal,
      });
      if (!response.ok) throw new Error(await apiError(response));
      const payload = await response.json();
      if (requestId !== fleetRequestRef.current) return;
      fleetRef.current = payload;
      setFleet(payload);
      setFleetError(null);
    } catch (err) {
      if (err?.name !== 'AbortError' && requestId === fleetRequestRef.current) {
        setFleetError(err?.message || 'Не удалось загрузить сводку по станциям');
      }
    } finally {
      if (requestId === fleetRequestRef.current) {
        fleetAbortRef.current = null;
        setFleetLoading(false);
      }
    }
  }, [asOf]);

  useEffect(() => {
    if (activeTab !== 'fleet') return undefined;
    loadFleet();
    return () => {
      fleetRequestRef.current += 1;
      fleetAbortRef.current?.abort();
      fleetAbortRef.current = null;
    };
  }, [activeTab, loadFleet]);

  useEffect(() => {
    if (!fleet?.isSynthetic || !Array.isArray(fleet.stations)) return undefined;
    let cancelled = false;
    const queue = fleet.stations.flatMap((station) =>
      (station.units || []).map((unit) => ({ tppId: station.id, kaId: unit.id })),
    );

    const warmDashboardCache = async () => {
      let cursor = 0;
      const worker = async () => {
        while (!cancelled && cursor < queue.length) {
          const item = queue[cursor];
          cursor += 1;
          try {
            await fetchDashboardPayload(item.tppId, item.kaId);
          } catch {
            // Фоновый прогрев не должен мешать обычной загрузке по выбору пользователя.
          }
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(DASHBOARD_PREFETCH_WORKERS, queue.length) }, worker),
      );
    };

    const startWarmup = () => { void warmDashboardCache(); };
    const idleHandle = window.requestIdleCallback
      ? window.requestIdleCallback(startWarmup, { timeout: 1200 })
      : window.setTimeout(startWarmup, 250);

    return () => {
      cancelled = true;
      if (window.cancelIdleCallback && typeof idleHandle === 'number') {
        window.cancelIdleCallback(idleHandle);
      } else {
        window.clearTimeout(idleHandle);
      }
    };
  }, [fleet, fetchDashboardPayload]);

  // --------------------------------------------------------------------------
  // Общий фильтр: ТЭЦ → доступные котлоагрегаты
  // --------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    const loadTpps = async () => {
      try {
        const response = await authFetch(`${API_BASE}/api/tpps`, { cache: 'no-store' });
        if (!response.ok) throw new Error(await apiError(response));
        const payload = await response.json();
        const list = Array.isArray(payload.tpps) ? payload.tpps : [];
        if (cancelled) return;
        setTpps(list);
        setSelectedTpp((current) =>
          list.some((tpp) => tpp.id === current) ? current : (list[0]?.id || ''),
        );
      } catch (err) {
        if (!cancelled) {
          setError(err?.message || 'Не удалось загрузить список ТЭЦ');
          setSelectionLoading(false);
        }
      }
    };
    loadTpps();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selectedTpp) return undefined;
    let cancelled = false;
    setSelectionLoading(true);
    const loadKas = async () => {
      try {
        const response = await authFetch(`${API_BASE}/api/tpps/${encodeURIComponent(selectedTpp)}/kas`, {
          cache: 'no-store',
        });
        if (!response.ok) throw new Error(await apiError(response));
        const payload = await response.json();
        const list = Array.isArray(payload.kas) ? payload.kas : [];
        if (cancelled) return;
        setAvailableKas(list);
        setSelectedKa((current) =>
          list.some((ka) => ka.id === current) ? current : (list[0]?.id || ''),
        );
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err?.message || 'Не удалось загрузить список котлоагрегатов');
      } finally {
        if (!cancelled) setSelectionLoading(false);
      }
    };
    loadKas();
    return () => { cancelled = true; };
  }, [selectedTpp]);

  // --------------------------------------------------------------------------
  // Загрузка данных дашборда
  // --------------------------------------------------------------------------
  const loadDashboard = useCallback(async ({ silent = false } = {}) => {
    if (!selectedTpp || !selectedKa) return;
    const key = dashboardCacheKey(selectedTpp, selectedKa, asOf);
    const cached = dashboardCacheRef.current.get(key);
    const isFresh = cached && Date.now() - cached.cachedAt < DASHBOARD_CACHE_TTL_MS;

    if (!silent && cached) {
      setDashboard(cached.payload);
      setLoading(false);
      setError(null);
      if (isFresh) return;
    }

    const requestId = dashboardRequestRef.current + 1;
    dashboardRequestRef.current = requestId;
    if (silent || cached) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    try {
      const entry = await fetchDashboardPayload(selectedTpp, selectedKa, {
        force: Boolean(silent || (cached && !isFresh)),
      });
      if (dashboardRequestRef.current === requestId) {
        setDashboard(entry.payload);
        setError(null);
      }
    } catch (err) {
      if (dashboardRequestRef.current === requestId) {
        setError(err && err.message ? err.message : 'Не удалось загрузить данные дашборда');
      }
    } finally {
      if (dashboardRequestRef.current === requestId) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [selectedTpp, selectedKa, asOf, fetchDashboardPayload]);

  useEffect(() => {
    if (activeTab !== 'fleet' && selectedTpp && selectedKa && !dashboard) loadDashboard();
  }, [activeTab, selectedTpp, selectedKa, dashboard, loadDashboard]);

  // --------------------------------------------------------------------------
  // Детали параметра
  // --------------------------------------------------------------------------
  const openParameter = useCallback(async (name) => {
    if (!name) return;
    setSelectedName(name);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
    const requestId = detailRequestRef.current + 1;
    detailRequestRef.current = requestId;
    try {
      const query = new URLSearchParams({ tpp_id: selectedTpp, ka_id: selectedKa });
      if (asOf) query.set('at', asOf);
      const response = await authFetch(`${API_BASE}/api/parameter/${encodeURIComponent(name)}?${query}`, {
        cache: 'no-store',
      });
      if (!response.ok) {
        throw new Error(
          response.status === 404
            ? `Параметр «${name}» не найден в данных`
            : `Сервер вернул код ${response.status}`,
        );
      }
      const payload = await response.json();
      if (detailRequestRef.current === requestId) {
        setDetail(payload);
      }
    } catch (err) {
      if (detailRequestRef.current === requestId) {
        setDetailError(err && err.message ? err.message : 'Ошибка загрузки параметра');
      }
    } finally {
      if (detailRequestRef.current === requestId) {
        setDetailLoading(false);
      }
    }
  }, [selectedTpp, selectedKa, asOf]);

  const closeModal = useCallback(() => {
    detailRequestRef.current += 1;
    setSelectedName(null);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(false);
  }, []);

  // Смена момента: адрес страницы, перерасчёт открытого агрегата (старые данные видны до ответа)
  const asOfRef = useRef(asOf);
  useEffect(() => {
    if (asOfRef.current === asOf) return;
    asOfRef.current = asOf;
    const params = new URLSearchParams(window.location.search);
    if (asOf) params.set('at', asOf);
    else params.delete('at');
    const search = params.toString();
    window.history.replaceState(null, '', `${window.location.pathname}${search ? `?${search}` : ''}`);
    closeModal();
    if (activeTab === 'fleet') setDashboard(null);
    else loadDashboard({ silent: true });
  }, [asOf, activeTab, loadDashboard, closeModal]);

  // Глубокая ссылка: /?param=<название параметра> сразу открывает карточку
  useEffect(() => {
    const initialParam = new URLSearchParams(window.location.search).get('param');
    if (initialParam) {
      openParameter(initialParam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);



  // --------------------------------------------------------------------------
  // Производные данные
  // --------------------------------------------------------------------------
  const overview = dashboard?.overview;
  const summary = dashboard?.summary;
  const dataRange = overview?.dataRange || fleet?.dataRange;
  const effectiveAsOf = (activeTab === 'fleet' ? fleet?.asOf : overview?.asOf) || dataRange?.end;
  const kpis = dashboard?.kpis || [];
  const alerts = dashboard?.alerts || [];
  const equipment = overview?.equipmentName || '—';

  const allParameters = useMemo(() => Object.values(dashboard?.parameters || {}), [dashboard]);

  const filteredParameters = useMemo(() => {
    const query = search.trim().toLowerCase();
    const riskWeight = { A: 4, B: 3, V: 2, G: 1 };

    const list = allParameters.filter((param) => {
      if (categoryFilter !== 'all' && param.category !== categoryFilter) return false;
      if (riskFilter !== 'all' && param.riskLevel !== riskFilter) return false;
      if (query) {
        const haystack = `${param.name} ${param.categoryLabel} ${param.column}`.toLowerCase();
        if (!haystack.includes(query)) return false;
      }
      return true;
    });

    const sorted = [...list].sort((a, b) => {
      switch (sortKey) {
        case 'name':
          return String(a.name).localeCompare(String(b.name), 'ru');
        case 'category':
          return String(a.categoryLabel || '').localeCompare(String(b.categoryLabel || ''), 'ru');
        case 'current':
          return (Number(a.current) || 0) - (Number(b.current) || 0);
        case 'change':
          return Math.abs(Number(a.changePct) || 0) - Math.abs(Number(b.changePct) || 0);
        case 'risk':
        default: {
          const diff = (Number(a.riskScore) || 0) - (Number(b.riskScore) || 0);
          if (diff !== 0) return diff;
          return (riskWeight[a.riskLevel] || 0) - (riskWeight[b.riskLevel] || 0);
        }
      }
    });

    if (sortDirection === 'desc') sorted.reverse();
    return sorted;
  }, [allParameters, search, categoryFilter, riskFilter, sortKey, sortDirection]);

  const handleSort = useCallback(
    (key) => {
      if (sortKey === key) {
        setSortDirection((direction) => (direction === 'asc' ? 'desc' : 'asc'));
      } else {
        setSortKey(key);
        setSortDirection(key === 'name' || key === 'category' ? 'asc' : 'desc');
      }
    },
    [sortKey],
  );

  const showRisk = useCallback((level) => {
    setRiskFilter(level);
    setActiveTab('parameters');
  }, []);

  const showCategory = useCallback((category) => {
    setCategoryFilter(category);
    setActiveTab('parameters');
  }, []);

  const resetFilters = useCallback(() => {
    setSearch('');
    setCategoryFilter('all');
    setRiskFilter('all');
  }, []);

  const filtersActive = search !== '' || categoryFilter !== 'all' || riskFilter !== 'all';

  // --------------------------------------------------------------------------
  // Состояния загрузки и ошибок
  // --------------------------------------------------------------------------
  const fatalError = activeTab !== 'fleet' && error && !dashboard;

  if (activeTab === 'fleet' && fleetLoading && !fleet) {
    return (
      <main className="app-shell">
        <LoadingSpinner text="Собираем сводку по всем станциям…" />
      </main>
    );
  }

  if (activeTab !== 'fleet' && !fatalError && (loading || selectionLoading) && !dashboard) {
    return (
      <main className="app-shell">
        <LoadingSpinner text="Загрузка данных дашборда..." />
      </main>
    );
  }

  if (fatalError) {
    return (
      <main className="app-shell">
        <ErrorMessage
          message={`Не удалось получить данные с ${API_BASE || 'API'}: ${error}`}
          onRetry={() => window.location.reload()}
        />
        <div className="error-help">
          <h3>Как запустить дашборд</h3>
          <ul>
            <li>
              Запустите backend: <code>cd backend; python main.py</code>
            </li>
            <li>
              Откройте <code>http://localhost:8000</code> — собранный дашборд раздаёт сам backend.
            </li>
            <li>
              Режим разработки фронтенда:{' '}
              <code>cd predictive_maintenance_dashboard; npm run dev</code>
            </li>
            <li>
              Проверка API: <code>http://localhost:8000/api/health</code>
            </li>
          </ul>
        </div>
      </main>
    );
  }

  const unitScopeActive = activeTab !== 'fleet';
  const levelThresholds = overview?.model?.levelThresholds;
  const unitTabs = [
    { id: 'overview', label: 'Сводка' },
    { id: 'parameters', label: `Параметры (${allParameters.length})` },
    { id: 'alerts', label: `Предупреждения (${alerts.length})` },
  ];

  return (
    <main className="app-shell">
      <header className="app-toolbar">
        <nav className="scope-tabs" aria-label="Область мониторинга">
          <button
            type="button"
            className={!unitScopeActive ? 'active' : ''}
            onClick={() => {
              dashboardRequestRef.current += 1;
              setActiveTab('fleet');
            }}
          >
            Все станции
          </button>
          <button
            type="button"
            className={unitScopeActive ? 'active' : ''}
            onClick={() => setActiveTab('overview')}
          >
            Детализация
          </button>
        </nav>

        <a
          className="other-dashboard-link"
          href={OTHER_DASHBOARD.url}
          target={OTHER_DASHBOARD.url === '#' ? undefined : '_blank'}
          rel="noreferrer"
          title={OTHER_DASHBOARD.url === '#' ? 'Ссылка появится позже' : OTHER_DASHBOARD.label}
          onClick={(event) => { if (OTHER_DASHBOARD.url === '#') event.preventDefault(); }}
        >
          <span aria-hidden="true">↗</span> {OTHER_DASHBOARD.label}
        </a>

        <div className="header-status toolbar-actions">
          <AsOfFilter value={asOf} effective={effectiveAsOf} range={dataRange} onChange={setAsOf} />
          <button
            type="button"
            className="theme-toggle"
            aria-label={theme === 'dark' ? 'Включить светлую тему' : 'Включить тёмную тему'}
            aria-pressed={theme === 'light'}
            onClick={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')}
          >
            <span>{theme === 'dark' ? '☀' : '☾'}</span>
            {theme === 'dark' ? 'Светлая' : 'Тёмная'}
          </button>
          {unitScopeActive && <span className="status-chip status-chip-mode">
            <i />
            Режим: {overview?.operatingModeLabel || overview?.status || 'нет данных'}
          </span>}
          <button
            type="button"
            className="action-button primary"
            onClick={() => !unitScopeActive ? loadFleet() : loadDashboard({ silent: true })}
            disabled={!unitScopeActive ? fleetLoading : refreshing}
          >
            ⟳ Обновить
          </button>
        </div>
      </header>

      {unitScopeActive && (
        <section className="equipment-context" aria-label="Выбор оборудования">
          <div className="equipment-context-title">
            <div>
              <h1>{overview?.tppName || tpps.find((item) => item.id === selectedTpp)?.name || 'Станция'}</h1>
            </div>
            <p>
              {allParameters.length} параметров · данные за {overview?.observationHours ?? '--'} ч до {formatDateTime(overview?.asOf)}
            </p>
          </div>
          <div className="dashboard-selector compact-selector">
            <label className="select-field">
              <span>Станция</span>
              <select
                value={selectedTpp}
                onChange={(event) => {
                  const tppId = event.target.value;
                  const station = fleetRef.current?.stations?.find((item) => item.id === tppId);
                  const nextKas = station?.units || [];
                  const kaId = nextKas[0]?.id || '';
                  setSelectedTpp(tppId);
                  if (nextKas.length) setAvailableKas(nextKas);
                  setSelectedKa(kaId);
                  if (kaId) prepareDashboardSelection(tppId, kaId);
                  setActiveTab('overview');
                }}
                disabled={selectionLoading || tpps.length === 0}
              >
                {tpps.map((tpp) => (
                  <option key={tpp.id} value={tpp.id}>
                    {tpp.name}{tpp.ka_count ? ` (${tpp.ka_count} КА)` : ''}
                  </option>
                ))}
              </select>
            </label>
            <label className="select-field">
              <span>Котлоагрегат</span>
              <select
                value={selectedKa}
                onChange={(event) => {
                  const kaId = event.target.value;
                  setSelectedKa(kaId);
                  prepareDashboardSelection(selectedTpp, kaId);
                  setActiveTab('overview');
                }}
                disabled={selectionLoading || availableKas.length === 0}
              >
                {availableKas.map((ka) => (
                  <option key={ka.id} value={ka.id}>{ka.name}</option>
                ))}
              </select>
            </label>
            {selectionLoading && <span className="selector-hint">Загрузка списка КА…</span>}
          </div>
          <div className="equipment-status-strip" aria-label="Режим и техническое состояние агрегата">
            <div className="is-big"><span>Режим работы</span><AutoFitText>{overview?.operatingModeLabel || overview?.status || '--'}</AutoFitText></div>
            <div className="is-big"><span>Техническое состояние</span><AutoFitText className={`tone-${statusTone(overview?.technicalStatus, summary?.healthScore)}`}>{overview?.technicalStatusLabel || summary?.technicalStatusLabel || '--'}</AutoFitText></div>
            <div className="is-big"><span>Индекс здоровья</span><AutoFitText className={`tone-${healthTone(summary?.healthScore)}`}>{isNum(summary?.healthScore) ? `${summary.healthScore} из 100` : 'нет оценки'}</AutoFitText></div>
            <div className="is-big"><span>Предупреждений</span><AutoFitText>{summary?.alertsTotal ?? alerts.length}</AutoFitText></div>
            <div><span>Наибольшее отклонение</span><strong>{summary?.worst ? `${summary.worst.name} · ${formatRatio(summary.worst.riskScore, 1)}` : '—'}</strong></div>
          </div>
          <nav className="unit-tabs" aria-label="Разделы выбранного котлоагрегата">
            {unitTabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={activeTab === tab.id ? 'active' : ''}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </section>
      )}

      {activeTab !== 'fleet' && summary?.statusCritical && (
        <div className="alert-banner">
          <span className="alert-banner-icon" role="img" aria-label="Аварийное состояние">
            ⛔
          </span>
          <div>
            <strong>Техническое состояние: «{overview?.technicalStatusLabel || summary?.technicalStatusLabel}»</strong>
            <p>
              Отклонение от нормы режима держится {overview?.model?.persistHours ?? 3} ч и дольше. Требуется осмотр узлов
              главных параметров тревоги (раздел «Предупреждения») и проверка самих датчиков.
            </p>
          </div>
        </div>
      )}

      {activeTab !== 'fleet' && summary?.modelRatio > 1 && summary?.interpretation && (
        <section className="alert-banner" aria-label="Интерпретация отклонения">
          <div>
            <strong>
              {summary.interpretation.label && summary.interpretation.label !== 'unknown'
                ? (INTERPRETATION_LABELS[summary.interpretation.label] || summary.interpretation.label)
                : 'Смысл отклонения пока не определён'}
            </strong>
            {summary.interpretation.label !== 'unknown' && Number.isFinite(summary.interpretation.confidence) && (
              <p>Уверенность оценки: {Math.round(summary.interpretation.confidence * 100)}%.</p>
            )}
            {[24, 168].map((hours) => Number.isFinite(summary.interpretation[`risk_${hours}h`]) && (
              <p key={hours}>
                Оценка риска аварийного останова в течение {hours === 24 ? 'суток' : 'недели'}:{' '}
                {Math.round(summary.interpretation[`risk_${hours}h`] * 100)}%.
              </p>
            ))}
          </div>
        </section>
      )}

      {error && (
        <div className="inline-error">
          ⚠️ {error} — показаны последние успешно полученные данные
        </div>
      )}

      {activeTab === 'fleet' && (
        <FleetOverview
          data={fleet}
          loading={fleetLoading}
          error={fleetError}
          onRetry={loadFleet}
          onOpenUnit={(tppId, kaId) => {
            setSelectedTpp(tppId);
            setSelectedKa(kaId);
            const station = fleetRef.current?.stations?.find((item) => item.id === tppId);
            if (station?.units?.length) setAvailableKas(station.units);
            prepareDashboardSelection(tppId, kaId);
            setActiveTab('overview');
          }}
        />
      )}


      {/* ------------------------- Детализация ------------------------- */}
      {activeTab === 'overview' && (
        <>
          <section className="dashboard-section" aria-labelledby="summary-title">
            <div className="section-heading">
              <span>01</span>
              <h2 id="summary-title">Сводка состояния</h2>
              <p>Индекс здоровья и коэффициенты отклонения параметров от нормы текущего режима</p>
            </div>
            <div className="kpi-grid">
              <article className="metric-card">
                <p className="metric-label">Индекс здоровья</p>
                <HealthRing
                  score={summary?.healthScore}
                  trend={summary?.healthTrend}
                  equipment={equipment}
                />
                <Sparkline
                  values={(summary?.healthSeries || []).map((point) => point.value)}
                  tone={healthTone(summary?.healthScore)}
                  label="Индекс здоровья за окно"
                />
              </article>

              <article className="metric-card">
                <p className="metric-label">Параметры по коэффициенту отклонения</p>
                <RiskBars byRisk={summary?.byRisk} activeRisk={riskFilter} onSelect={showRisk} thresholds={levelThresholds} />
              </article>

              <article className="metric-card failure-card">
                <p className="metric-label">Активных предупреждений</p>
                <strong>{summary?.alertsTotal ?? alerts.length}</strong>
                <p className="failure-text">тревог и параметров вне нормы режима</p>
                <p className="metric-delta warning">
                  Критических: {summary?.byRisk?.A ?? 0} • Высоких: {summary?.byRisk?.B ?? 0} • Значимых: {summary?.byRisk?.V ?? 0}
                </p>
              </article>
            </div>
          </section>

          <section className="dashboard-section" aria-labelledby="categories-title">
            <div className="section-heading">
              <span>02</span>
              <h2 id="categories-title">Категории и тренды</h2>
              <p>Нажатие на категорию открывает отфильтрованный список параметров</p>
            </div>
            <div className="overview-split">
              <CategoryBlocks
                categories={summary?.byCategory}
                activeCategory={categoryFilter}
                onSelect={showCategory}
              />
              <div className="side-panel">
                <h3>Тренд параметров за последние {overview?.observationHours ?? '--'} ч</h3>
                <TrendSummary trendCounts={summary?.trendCounts} />
                <h3>Данные</h3>
                <ul className="meta-list">
                  <li>
                    <span>Всего параметров</span>
                    <strong>{allParameters.length}</strong>
                  </li>
                  <li>
                    <span>Предупреждений</span>
                    <strong>{summary?.alertsTotal ?? alerts.length}</strong>
                  </li>
                  {summary?.worst && (
                    <li>
                      <span>Наибольшее отклонение</span>
                      <button
                        type="button"
                        className="link-button"
                        onClick={() => openParameter(summary.worst.name)}
                      >
                        {summary.worst.name}
                      </button>
                    </li>
                  )}
                </ul>
              </div>
            </div>
          </section>

          {kpis.length > 0 && (
            <section className="dashboard-section" aria-labelledby="kpi-title">
              <div className="section-heading">
                <span>03</span>
                <h2 id="kpi-title">Ключевые технологические параметры</h2>
                <p>Нажатие на карточку открывает детали параметра</p>
              </div>
              <div className="kpi-cards-grid">
                {kpis.map((kpi) => (
                  <KPICard
                    key={kpi.id}
                    kpi={kpi}
                    onClick={() => openParameter(kpi.parameterName)}
                  />
                ))}
              </div>
            </section>
          )}

          <section className="dashboard-section" aria-labelledby="states-title">
            <div className="section-heading">
              <span>04</span>
              <h2 id="states-title">Состояние механизмов</h2>
              <p>Положения задвижек и состояния механизмов; выделены те, что не в обычном состоянии</p>
            </div>
            <StateCards states={summary?.states} onSelect={openParameter} />
          </section>

          <section className="dashboard-section" aria-labelledby="diagnostics-title">
            <div className="section-heading">
              <span>05</span>
              <h2 id="diagnostics-title">Диагностика каналов</h2>
              <p>Каналы без изменения значения и каналы без данных</p>
            </div>
            <DiagnosticsPanels
              frozen={summary?.frozen}
              unavailable={summary?.unavailable}
              missing={summary?.missing}
              onSelect={openParameter}
            />
          </section>

          <section className="dashboard-section" aria-labelledby="alerts-title">
            <div className="section-heading">
              <span>06</span>
              <h2 id="alerts-title">Активные предупреждения</h2>
              <p>Поиск, фильтры и рекомендации по каждому отклонению</p>
            </div>
            <AlertList
              alerts={alerts}
              onSelect={openParameter}
              compact
            />
          </section>
        </>
      )}


      {/* -------------------------- Параметры -------------------------- */}
      {activeTab === 'parameters' && (
        <section className="dashboard-section" aria-labelledby="params-title">
          <div className="section-heading">
            <span>01</span>
            <h2 id="params-title">Все параметры {equipment}</h2>
            <p>Поиск, фильтры и сортировка • нажатие на строку открывает детали</p>
          </div>

          <div className="toolbar">
            <label className="search-field">
              <span className="sr-only">Поиск параметра</span>
              <input
                type="search"
                placeholder="Поиск по названию, категории или каналу…"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>

            <label className="select-field">
              <span>Категория</span>
              <select
                value={categoryFilter}
                onChange={(event) => setCategoryFilter(event.target.value)}
              >
                <option value="all">Все категории</option>
                {(summary?.byCategory || []).map((category) => (
                  <option key={category.category} value={category.category}>
                    {category.label} ({category.total})
                  </option>
                ))}
              </select>
            </label>

            <label className="select-field">
              <span>Коэффициент отклонения</span>
              <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
                <option value="all">Все</option>
                {RISK_LEVELS.map((level) => (
                  <option key={level} value={level}>
                    {riskOf(level).label} ({levelRange(level, levelThresholds)}) — {riskOf(level).name} ({summary?.byRisk?.[level] ?? 0})
                  </option>
                ))}
              </select>
            </label>

            <label className="select-field">
              <span>Сортировка</span>
              <select
                value={`${sortKey}:${sortDirection}`}
                onChange={(event) => {
                  const [key, direction] = event.target.value.split(':');
                  setSortKey(key);
                  setSortDirection(direction);
                }}
              >
                {SORT_OPTIONS.flatMap((option) => [
                  <option key={`${option.value}:desc`} value={`${option.value}:desc`}>
                    {option.label} ↓
                  </option>,
                  <option key={`${option.value}:asc`} value={`${option.value}:asc`}>
                    {option.label} ↑
                  </option>,
                ])}
              </select>
            </label>

            {filtersActive && (
              <button type="button" className="action-button" onClick={resetFilters}>
                ✕ Сбросить фильтры
              </button>
            )}

            <div className="view-switch" role="group" aria-label="Вид параметров">
              <button type="button" className={parameterView === 'cards' ? 'is-active' : ''} onClick={() => setParameterView('cards')} title="Карточки">▦</button>
              <button type="button" className={parameterView === 'table' ? 'is-active' : ''} onClick={() => setParameterView('table')} title="Таблица">☷</button>
            </div>
          </div>

          <p className="result-count">
            Найдено {filteredParameters.length} из {allParameters.length} параметров
            {riskFilter !== 'all' ? ` • коэффициент отклонения ${riskOf(riskFilter).label} (${levelRange(riskFilter, levelThresholds)})` : ''}
            {categoryFilter !== 'all'
              ? ` • категория: ${(summary?.byCategory || []).find((c) => c.category === categoryFilter)?.label || categoryFilter}`
              : ''}
          </p>

          {parameterView === 'cards' ? (
            filteredParameters.length ? (
              <div className="sensor-grid">
                {filteredParameters.map((parameter) => (
                  <SensorCard key={parameter.column} parameter={parameter} onSelect={openParameter} />
                ))}
              </div>
            ) : <EmptyState text="Параметры не найдены — измените условия фильтрации" icon="🔎" />
          ) : (
            <div className="table-card">
              <ParameterTable
                parameters={filteredParameters}
                sortKey={sortKey}
                sortDirection={sortDirection}
                onSort={handleSort}
                onSelect={openParameter}
              />
            </div>
          )}

        </section>
      )}

      {/* ---------------------------- Алерты ---------------------------- */}
      {activeTab === 'alerts' && (
        <section className="dashboard-section" aria-labelledby="alerts-tab-title">
          <div className="section-heading">
            <span>01</span>
            <h2 id="alerts-tab-title">Все предупреждения</h2>
            <p>Тревоги по котлу и параметры вне нормы текущего режима</p>
          </div>
          <AlertList
            alerts={alerts}
            onSelect={openParameter}
            filter={alertFilter}
            onFilterChange={setAlertFilter}
          />
        </section>
      )}

      {selectedName && (
        <ParameterModal
          parameterName={selectedName}
          detail={detail}
          loading={detailLoading}
          error={detailError}
          onClose={closeModal}
          onRetry={() => openParameter(selectedName)}
        />
      )}

      <footer className="dashboard-footer">
        {activeTab === 'fleet' ? (
          <>
            <p>Единый центр мониторинга • {fleet?.summary?.stationsTotal ?? '--'} станции • {fleet?.summary?.unitsTotal ?? '--'} котлоагрегатов</p>
            <p>Источник: {fleet?.dataSourceLabel || '--'} • момент оценки {formatDateTime(fleet?.asOf)}</p>
          </>
        ) : (
          <>
            <p>
              Источник: {overview?.dataSourceLabel || '--'}, {overview?.dataPoints ?? '--'} ч × {overview?.columns ?? '--'} каналов •
              окно наблюдения с {formatDateTime(overview?.firstUpdate)} по {formatDateTime(overview?.lastUpdate)}
            </p>
          </>
        )}
      </footer>
    </main>
  );
}

