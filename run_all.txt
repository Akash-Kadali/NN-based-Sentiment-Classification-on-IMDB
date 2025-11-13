@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ====== CONFIG ======
set "USE_VENV=1"          REM 0 to use system Python
set "BASE_EPOCHS=3"       REM quick sanity check
set "GRID_EPOCHS=5"       REM per-sweep epochs
REM =====================

REM Move to this script's folder (repo root)
pushd "%~dp0"

REM Ensure expected folders exist
for %%D in (data src results results\plots results\experiments logs) do (
  if not exist "%%D" mkdir "%%D"
)

REM Make src a package (for relative imports)
if not exist "src\__init__.py" type nul > "src\__init__.py"

REM Environment for clean logs/plots
set "MPLBACKEND=Agg"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "TOKENIZERS_PARALLELISM=false"

REM Pick a Python launcher
set "PYEXE="
where py >nul 2>&1 && set "PYEXE=py"
if not defined PYEXE (
  where python >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
  echo [ERROR] Python not found on PATH. Install Python 3.x and try again.
  goto :fail
)

REM Create/activate venv if requested
if "%USE_VENV%"=="1" (
  if not exist ".venv\Scripts\python.exe" (
    echo [+] Creating virtual environment .venv ...
    %PYEXE% -3 -m venv .venv || goto :fail
  )
  call ".venv\Scripts\activate.bat"
  set "PYEXE=python"
)

REM Install dependencies
echo [+] Installing dependencies...
%PYEXE% -m pip install --upgrade pip || goto :fail
if exist requirements.txt (
  %PYEXE% -m pip install -r requirements.txt || goto :fail
) else (
  echo [WARN] requirements.txt not found. Continuing...
)

REM Stable timestamp for log filename (PowerShell) with fallback
set "TS="
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "TS=%%I"
if not defined TS (
  set "TS=%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
  set "TS=%TS: =0%"
)
set "LOG=logs\run_%TS%.log"
echo ==== RUN START %DATE% %TIME% ==== > "%LOG%"

REM Quick environment report
call :log "[env] PY=%PYEXE%  VENV=%USE_VENV%  BASE_EPOCHS=%BASE_EPOCHS%  GRID_EPOCHS=%GRID_EPOCHS%"
call :log "[env] MPLBACKEND=%MPLBACKEND%  TOKENIZERS_PARALLELISM=%TOKENIZERS_PARALLELISM%"

REM ---- Baseline sanity check ----
call :log "[+] Baseline sanity check (LSTM/tanh/adam, L=50) ..."
%PYEXE% -m src.train --mode single --arch lstm --activation tanh --optimizer adam --seq-len 50 --epochs %BASE_EPOCHS% >> "%LOG%" 2>&1 || goto :fail

REM ---- Controlled studies (vary one factor at a time) ----
call :log "[+] Sweep: arch ..."
%PYEXE% -m src.train --mode grid --vary arch --epochs %GRID_EPOCHS% >> "%LOG%" 2>&1 || goto :fail

call :log "[+] Sweep: activation ..."
%PYEXE% -m src.train --mode grid --vary activation --epochs %GRID_EPOCHS% >> "%LOG%" 2>&1 || goto :fail

call :log "[+] Sweep: optimizer ..."
%PYEXE% -m src.train --mode grid --vary optimizer --epochs %GRID_EPOCHS% >> "%LOG%" 2>&1 || goto :fail

call :log "[+] Sweep: seq_len ..."
%PYEXE% -m src.train --mode grid --vary seq_len --epochs %GRID_EPOCHS% >> "%LOG%" 2>&1 || goto :fail

call :log "[+] Sweep: grad_clip ..."
%PYEXE% -m src.train --mode grid --vary grad_clip --epochs %GRID_EPOCHS% >> "%LOG%" 2>&1 || goto :fail

REM ---- Plots for report ----
call :log "[+] Generating plots..."
%PYEXE% -m src.evaluate --plot acc_f1_vs_seq_len --arch lstm --activation tanh --optimizer adam --grad-clip 0 >> "%LOG%" 2>&1 || goto :fail
%PYEXE% -m src.evaluate --plot best_worst_losses >> "%LOG%" 2>&1 || goto :fail

echo ==== RUN OK %DATE% %TIME% ==== >> "%LOG%"
echo [OK] Finished. Metrics: results\metrics.csv  Plots: results\plots\
goto :ok

:fail
echo ==== RUN FAILED %DATE% %TIME% ==== >> "%LOG%"
echo [FAIL] See "%LOG%" for the exact error line.
goto :end

:ok
REM Optional: deactivate venv on success
if "%USE_VENV%"=="1" (
  call ".venv\Scripts\deactivate.bat" 2>nul
)

:end
popd
endlocal
pause
goto :eof

REM ---------- helpers ----------
:log
set "LMSG=%~1"
echo %LMSG%
>> "%LOG%" echo %LMSG%
exit /b
