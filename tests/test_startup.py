from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_qt_starts_with_explicit_plugin_path():
    pytest.importorskip("PyQt5")
    code = """
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QWidget
from app.qt_runtime import configure_qt_plugins

QCoreApplication.setLibraryPaths([])
configure_qt_plugins()
application = QApplication([])
window = QWidget()
window.show()
application.processEvents()
window.close()
print('QT_STARTUP_OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8"),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "QT_STARTUP_OK" in result.stdout


@pytest.mark.parametrize("models_valid,repair_succeeds", [(True, True), (False, True), (False, False)])
def test_ocr_installer_repairs_bad_models_before_image_self_check(tmp_path, monkeypatch, models_valid, repair_succeeds):
    from tools import launch

    installs = []
    processes = []
    monkeypatch.setattr(sys, "argv", ["launch.py", "--install-ocr"])
    monkeypatch.setattr(launch, "STARTUP_LOG", tmp_path / "startup.log")
    monkeypatch.setattr(launch, "environment_python", lambda: Path(sys.executable))
    monkeypatch.setattr(launch, "dependencies_ready", lambda python: True)
    def install(python, requirements="requirements.txt", repair=False):
        installs.append((requirements, repair))
        return 0 if not repair or repair_succeeds else 1
    def run_process(command, **kwargs):
        processes.append(command)
        return subprocess.CompletedProcess(command, 0 if len(processes) > 1 or models_valid else 1)
    monkeypatch.setattr(launch, "install_dependencies", install)
    monkeypatch.setattr(launch.subprocess, "run", run_process)
    assert launch.main() == (0 if models_valid or repair_succeeds else 1)
    assert installs[0] == ("requirements-ocr.txt", False)
    if models_valid:
        assert len(installs) == 1
    else:
        assert installs[1] == ("requirements-ocr.txt", True)
    if models_valid or repair_succeeds:
        assert "check_image_ocr.py" in str(processes[-1][-1])
    else:
        assert len(processes) == 1
