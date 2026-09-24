@echo off
cd /d "%~dp0"
echo Starting AI Intelligent Surveillance Platform...
echo.
echo IMPORTANT: Demo mode runs WITHOUT Uvicorn auto-reload so AI jobs are not lost.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
