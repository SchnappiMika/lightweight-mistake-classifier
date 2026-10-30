# -*- coding: utf-8 -*-
"""公开素材验收工具的离线回归测试。"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from tools.run_public_acceptance import write_report


def test_public_manifest_has_traceability_and_unreviewed_labels():
    manifest = Path("samples/public_acceptance/manifest.json")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(rows) >= 5
    for row in rows:
        assert row["source_url"].startswith("https://commons.wikimedia.org/")
        assert row["license"]
        assert row["review_status"] == "待人工复核"
        assert "gold_knowledge" in row and row["gold_knowledge"] == ""


def test_public_report_keeps_model_missing_result(tmp_path):
    results = [{
        "id": "audio", "subject": "大学英语", "qtype": "listening",
        "license": "CC BY-SA 4.0", "status": "model_not_ready",
        "duration_seconds": 2.0, "elapsed_ms": 4.0,
        "error": "本地模型目录为空", "transcript": "",
    }]
    write_report(tmp_path, results)
    report = (tmp_path / "验收记录.md").read_text(encoding="utf-8")
    assert "model_not_ready" in report
    assert "不能直接用于申报书准确率" in report
    assert "转写成功 0/1 条" in report
    assert "已使用本地 faster-whisper base 模型完成转写" not in report


def test_public_report_reflects_success_without_claiming_missing_model(tmp_path):
    write_report(tmp_path, [{"id": "audio", "qtype": "listening", "license": "CC BY-SA 4.0",
                             "status": "recognized", "transcript": "Hello", "error": "",
                             "elapsed_ms": 2000, "wall_elapsed_ms": 2500}])
    report = (tmp_path / "验收记录.md").read_text(encoding="utf-8")
    assert "转写成功 1/1 条" in report
    assert "2500.0ms" in report
    assert "模型目录未准备" not in report
    assert "不能当作每题全流程耗时达标证据" in report


def _audio_manifest(tmp_path, digest):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test audio")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "audio", "subject": "大学英语", "qtype": "listening",
                                    "source_url": "https://example.org/audio", "license": "CC0",
                                    "download_status": "downloaded", "local_path": str(source),
                                    "sha256": digest}]), encoding="utf-8")
    return manifest


def test_changed_source_is_rejected_before_recognition(tmp_path, monkeypatch):
    from tools.run_public_acceptance import run
    from app.core.preprocess import media_recognition

    def unexpected(*args):
        raise AssertionError("changed source must not be transcribed")

    monkeypatch.setattr(media_recognition, "recognize_isolated", unexpected)
    result = run(_audio_manifest(tmp_path, "wrong digest"))[0]
    assert result["status"] == "source_changed"
    assert "transcript" not in result


def test_transcription_failure_is_not_mislabeled_as_missing_model(tmp_path, monkeypatch):
    from tools import run_public_acceptance
    from app.core.preprocess import media_recognition

    monkeypatch.setattr(run_public_acceptance, "audio_info", lambda path: {"duration_seconds": 1})
    monkeypatch.setattr(media_recognition, "recognize_isolated", lambda *args: SimpleNamespace(
        text="", error="invalid audio stream", engine="faster-whisper", elapsed_ms=20))
    result = run_public_acceptance.run(_audio_manifest(
        tmp_path, hashlib.sha256(b"test audio").hexdigest()))[0]
    assert result["status"] == "failed"
    assert result["error"] == "invalid audio stream"
    assert result["wall_elapsed_ms"] >= 0


def test_formula_uses_latex_backend_instead_of_printed_text_ocr(tmp_path, monkeypatch):
    from tools import run_public_acceptance
    from app.core.preprocess import media_recognition, image_ocr
    source = tmp_path / "formula.png"
    source.write_bytes(b"formula image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{
        "id": "formula", "subject": "高等数学", "qtype": "formula",
        "source_url": "https://example.org/formula", "license": "CC0",
        "download_status": "downloaded", "local_path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }]), encoding="utf-8")
    def recognize(path, kind):
        assert kind == "formula"
        return SimpleNamespace(text=r"\frac{dy}{dx}", engine="pix2tex", elapsed_ms=123, error="")
    def forbidden():
        raise AssertionError("公式不得走印刷文字OCR")
    monkeypatch.setattr(media_recognition, "recognize_isolated", recognize)
    monkeypatch.setattr(image_ocr, "get_engine", forbidden)
    result = run_public_acceptance.run(manifest)[0]
    assert result["latex"] == r"\frac{dy}{dx}"
    assert result["engine"] == "pix2tex" and result["status"] == "recognized"
