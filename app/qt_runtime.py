from __future__ import annotations

import os
from pathlib import Path


def configure_qt_plugins() -> None:
    import PyQt5
    from PyQt5.QtCore import QCoreApplication

    plugins = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    if plugins.is_dir():
        os.environ.pop("QT_PLUGIN_PATH", None)
        platform_plugins = plugins / "platforms"
        if platform_plugins.is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_plugins)
        QCoreApplication.addLibraryPath(str(plugins))
