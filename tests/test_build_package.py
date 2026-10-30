"""Check release architecture and helper behaviour without running PyInstaller."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from tools import build


def test_media_analysis_excludes_desktop_and_gui_excludes_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "ROOT", tmp_path)
    spec = build.build_spec(True)
    tree = ast.parse(spec.read_text(encoding="utf-8"))
    analyses = {node.targets[0].id: node.value for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "Analysis"}
    assert set(analyses) == {"gui", "media"}
    gui_excludes = next(kw.value for kw in analyses["gui"].keywords if kw.arg == "excludes")
    media_excludes = next(kw.value for kw in analyses["media"].keywords if kw.arg == "excludes")
    assert "rapidocr" in ast.literal_eval(gui_excludes.right)
    assert "onnxruntime" in ast.literal_eval(gui_excludes.right)
    assert "PyQt5" in ast.literal_eval(media_excludes.right)


def test_worker_error_result_does_not_import_desktop(tmp_path):
    output = tmp_path / "result.json"
    script = (
        "import sys; from tools.media_worker_entry import main; "
        "code=main(['image', sys.argv[1], sys.argv[2]]); "
        "assert not any(n.startswith('PyQt5') for n in sys.modules); "
        "raise SystemExit(code)"
    )
    result = subprocess.run([sys.executable, "-c", script,
                             str(tmp_path / "missing.png"), str(output)],
                            cwd=str(Path(__file__).resolve().parents[1]),
                            capture_output=True, timeout=30)
    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8"))["error"]
