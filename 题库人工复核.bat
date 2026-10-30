@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
if not exist ".venv\Scripts\pythonw.exe" (
    echo 请先通过“启动软件.bat”准备本项目运行环境。
    pause
    exit /b 1
)
start "" /D "%~dp0" ".venv\Scripts\pythonw.exe" -m tools.review_acceptance %*
endlocal
