"""Locate the separate formula runtime without importing its heavy dependencies."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def formula_runtime() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[3]
    frozen = getattr(sys, "frozen", False)
    bundle = Path(sys.executable).parent / "formula_runtime"
    explicit = os.environ.get("MC_FORMULA_PYTHON")
    python = (Path(explicit) if explicit else
              bundle / "python" / "python.exe" if frozen else
              root / ".venv-formula" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    backend = (bundle / "formula_recognition_backend.py" if frozen else
               Path(__file__).with_name("formula_recognition_backend.py"))
    if not python.is_file():
        raise FileNotFoundError("公式识别独立环境尚未准备：{}。请按公式组件说明安装，或设置 MC_FORMULA_PYTHON。".format(python))
    if not backend.is_file():
        raise FileNotFoundError("公式识别组件缺少程序文件：{}".format(backend))
    return python, backend
