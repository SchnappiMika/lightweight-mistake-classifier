@echo off
cd /d "%~dp0"
echo Starting error classification system...
python src/main.py
echo.
echo Program exited. Press any key to close...
pause >nul
