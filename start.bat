@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    py -m venv venv
    if errorlevel 1 goto failed
)
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Open http://127.0.0.1:8001/docs
venv\Scripts\python.exe -m uvicorn standalone:app --host 127.0.0.1 --port 8001
pause
exit /b
:failed
echo Setup failed. Check the error message above.
pause
