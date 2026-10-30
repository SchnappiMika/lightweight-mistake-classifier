@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 (
    py -3 "tools\launch.py" --install-ocr
    goto finish
)
where python >nul 2>&1
if not errorlevel 1 (
    python "tools\launch.py" --install-ocr
    goto finish
)
echo Python was not found. Install Python 3.12 or 3.13, then try again.
pause
exit /b 1
:finish
if errorlevel 1 (
    echo OCR setup failed. Check the startup log.
    pause
    exit /b 1
)
echo OCR setup finished. Restart the software to use image text recognition.
pause
endlocal
