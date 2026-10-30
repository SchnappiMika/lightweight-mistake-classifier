# -*- coding: utf-8 -*-
"""程序入口。

    python main.py            正常启动
    python main.py --reset    清空工作区后启动（恢复出厂）
    python main.py --version  查看版本

Windows / MacOS 打包后双击 exe / app 即可运行，无需任何环境配置。
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 让源码直接运行时也能 import app 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

# Windows 的无终端发布版没有标准流；第三方库仍可能向其写日志。
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

if "--media-session-worker" in sys.argv:
    from app.core.preprocess.media_server import main as media_session_main
    worker_index = sys.argv.index("--media-session-worker")
    sys.exit(media_session_main(sys.argv[worker_index + 1:]))

if "--media-recognition-worker" in sys.argv:
    from app.core.preprocess.media_recognition_worker import main as media_worker_main
    worker_index = sys.argv.index("--media-recognition-worker")
    sys.exit(media_worker_main(sys.argv[worker_index + 1:]))

if "--image-ocr-worker" in sys.argv:
    from app.core.preprocess.image_ocr_worker import main as image_worker_main
    worker_index = sys.argv.index("--image-ocr-worker")
    sys.exit(image_worker_main(sys.argv[worker_index + 1:]))

if len(sys.argv) > 1 and sys.argv[1] == "--release-check":
    from app.release_check import main as release_check_main
    sys.exit(release_check_main(sys.argv[2:]))

from app import config  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    """日志同时输出到控制台与工作区滚动日志文件（便于排查用户现场问题）。"""
    config.ensure_workspace()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    try:
        fh = RotatingFileHandler(str(config.LOG_DIR / "app.log"), maxBytes=2 * 1024 * 1024,
                                 backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    logging.getLogger("jieba").setLevel(logging.WARNING)


def excepthook(exc_type, exc_value, exc_tb) -> None:
    """全局异常兜底：弹窗告知而不是直接闪退（对应"无崩溃"目标）。"""
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logging.getLogger("main").critical("未捕获异常：\n%s", text)
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("程序遇到问题")
            box.setText("发生了一个未预期的错误。请查看错误详情；如需重启，请先确认当前数据是否已保存。")
            box.setInformativeText("日志位置：{}".format(config.LOG_DIR / "app.log"))
            box.setDetailedText(text)
            box.exec_()
    except Exception:                       # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=config.APP_NAME)
    ap.add_argument("--reset", action="store_true", help="清空工作区数据后启动")
    ap.add_argument("--verbose", "-v", action="store_true", help="输出调试日志")
    ap.add_argument("--version", action="store_true", help="显示版本号")
    args = ap.parse_args()

    if args.version:
        print("{} v{}".format(config.APP_NAME, config.APP_VERSION))
        return 0

    if args.reset and config.WORKSPACE_DIR.exists():
        shutil.rmtree(config.WORKSPACE_DIR, ignore_errors=True)

    setup_logging(args.verbose)
    log = logging.getLogger("main")
    log.info("%s v%s 启动 | 工作区 %s", config.APP_NAME, config.APP_VERSION,
             config.WORKSPACE_DIR)

    # 高分屏适配：必须在创建 QApplication 之前设置
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from app.qt_runtime import configure_qt_plugins

    configure_qt_plugins()

    # 缩放比例原值透传。Qt 5 默认策略是「四舍五入到整数倍」：
    # Windows 上最常见的 125% 会被舍成 100%、150% 会被舍成 200%。
    # 舍成 100% 时，本程序比系统里其它软件小一圈，中文小字尤其吃力。
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    try:                                    # Qt 5.14+ 才有这个接口
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except AttributeError:
        pass
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setApplicationVersion(config.APP_VERSION)
    app.setOrganizationName(config.ORG_NAME)

    sys.excepthook = excepthook

    from app.ui import icons, theme
    from app.ui.main_window import MainWindow

    win = MainWindow()
    theme.apply(app, theme.current())
    win.nav.refresh_icons(__import__("app.ui.main_window", fromlist=["NAV_ENTRIES"]).NAV_ENTRIES)
    app.setWindowIcon(icons.app_icon())
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
