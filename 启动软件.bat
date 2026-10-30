@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 (
    py -3 "tools\launch.py" --detached %*
    goto finish
)
where python >nul 2>&1
if not errorlevel 1 (
    python "tools\launch.py" --detached %*
    goto finish
)
echo Python was not found. Install Python 3.12 or 3.13, then run this file again.
pause
exit /b 1
:finish
if errorlevel 1 (
    echo Startup failed. Keep this message and check the startup log.
    pause
    exit /b 1
)
endlocal
