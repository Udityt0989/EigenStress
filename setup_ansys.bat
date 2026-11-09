@echo off
REM EigenStress - one-time setup for the ANSYS scripts (Windows).
REM Creates a private Python environment (.venv) inside this folder and installs everything,
REM including PyMAPDL (ansys-mapdl-core). Needs Python 3.10-3.12 installed (python.org, tick "Add to PATH").

cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0pip-cache"
set "PYTHONNOUSERSITE=1"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment in "%~dp0.venv" ...
    python -m venv ".venv" || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt ansys-mapdl-core || goto :fail

echo.
echo Setup done. Next step (quick test, about 1 minute, starts ANSYS MAPDL in the background):
echo     .venv\Scripts\python.exe ansys_validate.py --parts lbracket
echo See README_PERSON_A.md for everything else.
pause
exit /b

:fail
echo.
echo Setup failed. Send a screenshot of the lines above.
pause
