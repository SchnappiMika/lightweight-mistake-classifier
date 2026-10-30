import hashlib

from tools.validate_release import binary_manifest, build_manifest


def test_source_manifest_excludes_environment_build_and_logs(tmp_path):
    for directory in ("app", ".venv", "dist", "build", ".startup-check", "app/__pycache__"):
        folder = tmp_path / directory
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sample.py").write_text("x", encoding="utf-8")
    (tmp_path / "启动日志.txt").write_text("private", encoding="utf-8")
    (tmp_path / "build.log").write_text("log", encoding="utf-8")
    assert build_manifest(tmp_path)["files"] == ["app/sample.py"]


def test_binary_manifest_detects_missing_worker_and_tracks_bytes(tmp_path):
    exe = tmp_path / "错题自动分类.exe"
    exe.write_bytes(b"gui")
    report = binary_manifest(tmp_path)
    assert "media_worker.exe" in report["missing"]
    assert report["verification"] == "static_only"
    assert report["files"][0]["sha256"] == hashlib.sha256(b"gui").hexdigest()
    assert report["total_bytes"] == 3
