@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动错题自动分类管理系统...
python src/main.py
if errorlevel 1 (
    echo.
    echo 启动失败！请确认已安装 Python 和所需依赖。
    echo 按任意键退出...
    pause >nul
)
