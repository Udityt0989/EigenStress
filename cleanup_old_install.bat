@echo off
REM OPTIONAL: removes the copies of the app's libraries that the first launcher put into
REM C:\Users\...\AppData\Local\Python\...  (frees about 1 GB on C:).
REM Only run this if you don't use these libraries for other Python work.
echo This will uninstall the app's libraries from your main (C:) Python.
choice /M "Continue"
if errorlevel 2 exit /b
python -m pip uninstall -y streamlit plotly ezdxf gmsh scikit-fem scikit-learn scipy pandas matplotlib joblib pyarrow altair pydeck tornado watchdog
python -m pip cache purge
echo Done.
pause
