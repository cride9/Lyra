@echo off
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"
set "LYRA_PYTHON=%LOCALAPPDATA%\Lyra\venv\Scripts\python.exe"
if not exist "%LYRA_PYTHON%" set "LYRA_PYTHON=..\osu-ai-mapper\.venv\Scripts\python.exe"
if not exist "%LYRA_PYTHON%" (
  echo Run Install.ps1 first.
  pause
  exit /b 1
)
echo Lyra: http://127.0.0.1:7862
"%LYRA_PYTHON%" -u -m lyra ui --port 7862 --open-browser
if errorlevel 1 pause
