from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = Path(os.environ.get("MC_RUNTIME_DIR", str(ROOT / ".venv")))
STARTUP_LOG = ROOT / "启动日志.txt"
CORE_IMPORTS = (
    "import jieba, numpy, scipy, sklearn, pandas, openpyxl; "
    "from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg; "
    "from app.qt_runtime import configure_qt_plugins; "
    "configure_qt_plugins(); "
    "app = QtWidgets.QApplication([])"
)


def environment_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def dependencies_ready(python: Path) -> bool:
    if not python.is_file():
        return False
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.Popen([str(python), "-c", CORE_IMPORTS], cwd=str(ROOT),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace", **options)
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, **options)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        stderr += "\n界面环境初始化超时，请检查运行环境目录和 Qt 安装。\n"
        with STARTUP_LOG.open("a", encoding="utf-8") as logfile:
            logfile.write(stdout + stderr)
        return False
    if process.returncode:
        with STARTUP_LOG.open("a", encoding="utf-8") as logfile:
            logfile.write(stdout + stderr)
    return process.returncode == 0


def install_dependencies(python: Path, requirements: str = "requirements.txt", repair: bool = False) -> int:
    print("正在准备运行环境，首次使用需要联网下载依赖，可能需要几分钟。", flush=True)
    print("安装进度记录在：{}".format(STARTUP_LOG), flush=True)
    with STARTUP_LOG.open("a", encoding="utf-8") as logfile:
        logfile.write("\n{} 安装依赖：{}\n".format(datetime.now().isoformat(timespec="seconds"), requirements))
        logfile.flush()
        pip_check = subprocess.run([str(python), "-m", "pip", "--version"],
                                   stdout=logfile, stderr=subprocess.STDOUT)
        if pip_check.returncode:
            bootstrap = subprocess.run([str(python), "-m", "ensurepip", "--upgrade"],
                                       stdout=logfile, stderr=subprocess.STDOUT)
            if bootstrap.returncode:
                return bootstrap.returncode
        arguments = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
        if repair:
            arguments.extend(["--force-reinstall", "--no-deps"])
        result = subprocess.run(
            arguments + ["-r", str(ROOT / requirements)], cwd=str(ROOT),
            stdout=logfile, stderr=subprocess.STDOUT)
    return result.returncode


def start_application(python: Path, detached: bool = False) -> int:
    command = [str(python), str(ROOT / "main.py")]
    if not detached or os.name != "nt":
        return subprocess.run(command, cwd=str(ROOT)).returncode
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               close_fds=True, creationflags=flags)
    return 0 if process.pid else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="检查环境并启动错题分类软件")
    parser.add_argument("--check", action="store_true", help="只检查环境，不启动软件")
    parser.add_argument("--install-only", action="store_true", help="只准备运行环境")
    parser.add_argument("--runtime-path", action="store_true", help="显示独立运行环境路径")
    parser.add_argument("--install-ocr", action="store_true", help="安装并验证图片文字识别组件")
    parser.add_argument("--detached", action="store_true", help="在 Windows 上脱离启动终端运行界面")
    args = parser.parse_args()
    if args.runtime_path:
        print(VENV_DIR)
        return 0
    if sys.version_info < (3, 9):
        print("需要 Python 3.9 或更高版本，建议使用 Python 3.12 或 3.13。")
        return 1
    python = environment_python()
    try:
        print("正在检查运行环境，请稍候…", flush=True)
        if not python.is_file():
            if args.check:
                print("尚未准备运行环境，请双击「启动软件.bat」。")
                return 1
            print("正在创建项目独立运行环境：{}".format(VENV_DIR), flush=True)
            venv.EnvBuilder(with_pip=True).create(str(VENV_DIR))
        ready = dependencies_ready(python)
        if args.check:
            print("核心运行环境正常。" if ready else "核心依赖不完整，请重新双击「启动软件.bat」。")
            return 0 if ready else 1
        if not ready:
            if install_dependencies(python) or not dependencies_ready(python):
                print("环境准备失败，请检查网络，并查看：{}".format(STARTUP_LOG))
                return 1
        if args.install_ocr:
            if install_dependencies(python, "requirements-ocr.txt"):
                print("图片识别组件安装失败，请查看：{}".format(STARTUP_LOG))
                return 1
            with STARTUP_LOG.open("a", encoding="utf-8") as logfile:
                model_check = subprocess.run(
                    [str(python), "-c", "from app.core.preprocess.image_ocr import model_paths; model_paths(verify_hash=True)"],
                    cwd=str(ROOT), stdout=logfile, stderr=subprocess.STDOUT)
            if model_check.returncode:
                print("本地模型校验失败，正在重新安装图片识别组件以修复模型。", flush=True)
                if install_dependencies(python, "requirements-ocr.txt", repair=True):
                    print("图片识别组件修复失败，请查看：{}".format(STARTUP_LOG))
                    return 1
            return subprocess.run([str(python), str(ROOT / "tools" / "check_image_ocr.py")],
                                  cwd=str(ROOT)).returncode
        if args.install_only:
            print("运行环境准备完成，可以双击「启动软件.bat」。")
            return 0
        print("正在启动软件；图片识别和听力转写还需要单独安装识别组件。", flush=True)
        result_code = start_application(python, detached=args.detached)
        if result_code:
            print("软件未正常退出，请查看工作区中的 logs/app.log。")
        return 0 if result_code == 0 else 1
    except (OSError, subprocess.SubprocessError) as exc:
        print("启动失败：{}".format(exc))
        print("请检查 Python 安装、目录写入权限，以及：{}".format(STARTUP_LOG))
        return 1


if __name__ == "__main__":
    sys.exit(main())
