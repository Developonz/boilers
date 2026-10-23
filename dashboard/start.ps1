# ============================================================================
#  Предиктивный дашборд — запуск на Windows (PowerShell)
#  Использование:
#     .\start.ps1                 запуск + открытие браузера
#     .\start.ps1 -NoBrowser      запуск без браузера
#     .\start.ps1 -Check          только проверка окружения
# ============================================================================
[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$Check,
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

Write-Host '============================================================'
Write-Host ' Предиктивный дашборд — запуск'
Write-Host ' Python FastAPI + React'
Write-Host '============================================================'
Write-Host ''

# --- Поиск Python -----------------------------------------------------------
$python = $null
foreach ($candidate in @('py', 'python', 'python3')) {
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($command) {
        $python = $command.Source
        if ($candidate -eq 'py') { $pythonArgs = @('-3') } else { $pythonArgs = @() }
        break
    }
}
if (-not $python) {
    Write-Host '[ОШИБКА] Python не найден в PATH. Установите Python 3.9+.' -ForegroundColor Red
    exit 1
}
Write-Host "[1/4] Python: $python"
& $python @pythonArgs --version

# --- Проверка данных --------------------------------------------------------
Write-Host ''
$hourly = Get-ChildItem -Path (Join-Path $PSScriptRoot 'data') -Filter 'hourly.parquet' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($hourly) {
    Write-Host "[2/4] Данные: $($hourly.FullName)"
} else {
    Write-Host '[2/4] [ПРЕДУПРЕЖДЕНИЕ] Не найден data\hourly.parquet (или задайте HOURLY_PATH в backend\.env).' -ForegroundColor Yellow
}
$model = Get-ChildItem -Path (Join-Path $PSScriptRoot 'models') -Filter 'config.json' -Recurse -Depth 1 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($model) {
    Write-Host "      Модель: $($model.DirectoryName)"
} else {
    Write-Host '      [ПРЕДУПРЕЖДЕНИЕ] В папке models нет модели (папки с config.json и model.py).' -ForegroundColor Yellow
}

# --- Зависимости ------------------------------------------------------------
Write-Host ''
& $python @pythonArgs -c 'import fastapi, uvicorn, pandas, pyarrow, sklearn' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host '[3/4] Установка зависимостей backend...'
    & $python @pythonArgs -m pip install -r 'backend\requirements.txt'
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[ОШИБКА] Не удалось установить зависимости.' -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host '[3/4] Зависимости backend установлены'
}

if ($Check) {
    Write-Host ''
    Write-Host 'Проверка окружения завершена успешно.' -ForegroundColor Green
    exit 0
}

# --- Запуск -----------------------------------------------------------------
Write-Host ''
Write-Host '[4/4] Запуск сервера...'
Write-Host ''
Write-Host "  Дашборд:   http://localhost:$Port"
Write-Host "  API-доки:  http://localhost:$Port/docs"
Write-Host '  Остановка: Ctrl+C'
Write-Host ''

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 4
        Start-Process $url
    } -ArgumentList "http://localhost:$Port" | Out-Null
}

$env:PORT = "$Port"
Set-Location -LiteralPath (Join-Path $PSScriptRoot 'backend')
& $python @pythonArgs 'main.py'

