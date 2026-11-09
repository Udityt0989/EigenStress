@echo off
REM Singularity Finder launcher (Windows).
REM Everything (libraries + download cache) is installed INSIDE this folder:
REM   .venv\       private Python environment
REM   pip-cache\   downloaded package files
REM Nothing is installed into C:\Users\...\AppData. Delete this folder to remove everything.

cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0pip-cache"
set "PYTHONNOUSERSITE=1"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating a private Python environment in "%~dp0.venv"
    echo This takes a few minutes. Yellow "not on PATH" warnings are harmless.
    python -m venv ".venv" || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

echo Starting the app - it opens in your browser at http://localhost:8501
echo Keep this window open while using the app. Close it to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
exit /b

:fail
echo.
echo Setup failed. Send a screenshot of the lines above.
echo To retry from scratch, delete the ".venv" folder and run this file again.
pause
