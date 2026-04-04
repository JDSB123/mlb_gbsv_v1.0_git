<#
.SYNOPSIS
    One-command environment setup for MLB GBSV v1.0.
    Creates (or rebuilds) the .venv, installs all dependencies, and validates.

.DESCRIPTION
    This script ensures a reproducible Python 3.12 environment:
      1. Locates Python 3.12 on PATH or common install locations
      2. Creates .venv (or rebuilds with --force)
      3. Installs pinned dependencies from requirements-lock.txt
      4. Installs mlbv1 in editable mode with dev extras
      5. Runs import smoke test + pytest

.PARAMETER Force
    Delete and recreate .venv from scratch.

.PARAMETER SkipTests
    Skip the pytest validation step.

.EXAMPLE
    .\scripts\setup_env.ps1
    .\scripts\setup_env.ps1 -Force
    .\scripts\setup_env.ps1 -Force -SkipTests
#>
param(
    [switch]$Force,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$requiredMajor = 3
$requiredMinor = 12

# ── 1. Find Python 3.12 ─────────────────────────────────────────────────────
function Find-Python312 {
    $candidates = @(
        "python3.12",
        "python3",
        "python"
    )
    # Also check common Windows install paths
    $paths = @(
        "C:\Program Files\Python312\python.exe",
        "C:\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) { $candidates = @($p) + $candidates }
    }

    foreach ($candidate in $candidates) {
        try {
            $ver = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver -eq "$requiredMajor.$requiredMinor") {
                $exe = & $candidate -c "import sys; print(sys.executable)" 2>$null
                return $exe
            }
        } catch {}
    }
    return $null
}

$python = Find-Python312
if (-not $python) {
    Write-Host "ERROR: Python $requiredMajor.$requiredMinor not found." -ForegroundColor Red
    Write-Host "Install from https://www.python.org/downloads/ and ensure it's on PATH." -ForegroundColor Yellow
    exit 1
}
Write-Host "Found Python $requiredMajor.$requiredMinor at: $python" -ForegroundColor Green

# ── 2. Create / rebuild .venv ────────────────────────────────────────────────
$venvDir = Join-Path $repoRoot ".venv"
if ($Force -and (Test-Path $venvDir)) {
    Write-Host "Removing existing .venv..." -ForegroundColor Yellow
    deactivate 2>$null
    Remove-Item -Recurse -Force $venvDir
}

if (-not (Test-Path (Join-Path $venvDir "pyvenv.cfg"))) {
    Write-Host "Creating .venv..." -ForegroundColor Cyan
    & $python -m venv $venvDir
} else {
    # Verify existing venv uses the right Python
    $cfg = Get-Content (Join-Path $venvDir "pyvenv.cfg") | Where-Object { $_ -match "^version\s*=" }
    if ($cfg -notmatch "$requiredMajor\.$requiredMinor") {
        Write-Host "WARNING: .venv uses wrong Python version. Rebuilding..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvDir
        & $python -m venv $venvDir
    } else {
        Write-Host ".venv already exists with Python $requiredMajor.$requiredMinor" -ForegroundColor Green
    }
}

# ── 3. Activate ─────────────────────────────────────────────────────────────
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    $activateScript = Join-Path $venvDir "bin/activate.ps1"
}
& $activateScript

# ── 4. Upgrade pip ──────────────────────────────────────────────────────────
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# ── 5. Install from lock file + editable ────────────────────────────────────
$lockFile = Join-Path $repoRoot "requirements-lock.txt"
if (Test-Path $lockFile) {
    Write-Host "Installing pinned dependencies from requirements-lock.txt..." -ForegroundColor Cyan
    pip install -r $lockFile --quiet
} else {
    Write-Host "WARNING: requirements-lock.txt not found, installing from pyproject.toml only" -ForegroundColor Yellow
}

Write-Host "Installing mlbv1 in editable mode with dev extras..." -ForegroundColor Cyan
pip install -e ".[dev]" --quiet

# ── 6. Validate ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Validation ===" -ForegroundColor Cyan

$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$pyExe = python -c "import sys; print(sys.executable)"
Write-Host "Python: $pyVer" -ForegroundColor White
Write-Host "Executable: $pyExe" -ForegroundColor White

# Quick import test
$importResult = python -c "
import importlib, sys
mods = ['mlbv1','mlbv1.config','mlbv1.data.loader','mlbv1.models.ensemble',
        'mlbv1.pipeline.consensus','mlbv1.tracking.database','numpy','pandas',
        'sklearn','xgboost','lightgbm','flask','optuna']
fail = 0
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        print(f'FAIL: {m} - {e}', file=sys.stderr)
        fail += 1
print(f'{len(mods)-fail}/{len(mods)} modules OK')
if fail: sys.exit(1)
" 2>&1
Write-Host $importResult -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })

if (-not $SkipTests) {
    Write-Host ""
    Write-Host "Running tests..." -ForegroundColor Cyan
    python -m pytest tests/ --tb=short -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "TESTS FAILED" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Environment setup complete." -ForegroundColor Green
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
