import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.measure_performance import (_environment, _run_worker, main, media_measure,
                                       percentile, summarize, text_classification)


def test_percentile_and_summary_are_stable():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.95) == 4.0
    result = summarize(values)
    assert result["count"] == 4
    assert result["mean_ms"] == 2.5
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0


def test_empty_summary_is_serializable():
    assert summarize([]) == {
        "count": 0, "mean_ms": 0.0, "p95_ms": 0.0,
        "min_ms": 0.0, "max_ms": 0.0,
    }


def test_environment_contains_reproducibility_fields():
    result = _environment()
    assert result["platform"]
    assert result["python"]
    assert isinstance(result["cpu_count"], int)


def test_media_failure_counts_in_denominator(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps([{"id": "a"}, {"id": "b"}]), encoding="utf-8")
    monkeypatch.setattr("tools.measure_performance._run_worker",
                        lambda args, workspace, model_dir: {"error": "offline media worker failed"})
    result = media_measure(tmp_path, tmp_path / "workspace")
    assert result["failures"] == 2
    assert result["per_item"]["attempted"] == 2
    assert result["per_item"]["failures"] == 2
    assert result["items_without_wall_timing"] == 2
    assert "完整导入" in result["timing_scope"]


def test_missing_timing_is_not_counted_as_zero(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps([{"id": "a"}, {"id": "b"}]), encoding="utf-8")
    payload = {"results": [{"status": "recognized", "wall_elapsed_ms": 20.0},
                           {"status": "missing_source"}], "subprocess_wall_ms": 100.0}
    monkeypatch.setattr("tools.measure_performance._run_worker", lambda *args: payload)
    result = media_measure(tmp_path, tmp_path / "workspace")
    assert result["total"] == 2 and result["failures"] == 1
    assert result["per_item"]["mean_ms"] == 20.0
    assert result["per_item"]["attempted"] == 2
    assert result["batch_wall_ms_per_manifest_item"] == 50.0
    assert result["items_without_wall_timing"] == 1


def test_worker_environment_is_fresh_and_model_path_explicit(monkeypatch, tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    sentinel = original / "settings.json"
    sentinel.write_bytes(b"real-settings-sentinel")
    monkeypatch.setenv("MC_WORKSPACE", str(original))
    model_root = tmp_path / "existing-models"
    recorded = {}

    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        child_workspace = Path(environment["MC_WORKSPACE"])
        recorded.update(workspace=child_workspace, environment=environment)
        assert child_workspace.is_dir() and not list(child_workspace.iterdir())
        assert child_workspace != original
        assert environment["MC_MODEL_DIR"] == str(model_root.resolve())
        return SimpleNamespace(stdout='{"results": []}\n', stderr="", returncode=0)

    monkeypatch.setattr("tools.measure_performance.subprocess.run", fake_run)
    result = _run_worker(["--worker-media", "manifest.json"], tmp_path / "scratch", model_root)
    assert not recorded["workspace"].exists()
    assert result["workspace_deleted_after_run"]
    assert sentinel.read_bytes() == b"real-settings-sentinel"
    assert os.environ["MC_WORKSPACE"] == str(original)


def test_real_text_subprocess_does_not_use_original_workspace(tmp_path, monkeypatch):
    original = tmp_path / "user-workspace"
    original.mkdir()
    sentinels = {"settings.json": b"invalid settings", "data/schemes.json": b"invalid schemes",
                 "model/weight_rules.json": b"invalid weights", "model/label_profiles.json": b"invalid learning"}
    for name, content in sentinels.items():
        target = original / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    monkeypatch.setenv("MC_WORKSPACE", str(original))
    result = text_classification(2, tmp_path / "scratch")
    assert not result.get("error"), result
    assert result["count"] == 2 and result["failures"] == 0
    assert result["cold_first_call"]["status"] == "classified"
    assert result["module_import_ms"] >= 0
    assert result["initialization_ms"] >= 0
    assert result["workspace_deleted_after_run"]
    assert result["workspace_was_empty_at_start"]
    for name, content in sentinels.items():
        assert (original / name).read_bytes() == content
    assert result["builtin_scheme_sha256"]


def test_direct_worker_requires_isolation_marker(monkeypatch):
    monkeypatch.delenv("MC_PERF_ISOLATED_WORKER", raising=False)
    with pytest.raises(SystemExit):
        main(["--worker-text", "1"])


def test_historical_report_cannot_be_overwritten():
    from tools.measure_performance import ROOT
    with pytest.raises(SystemExit):
        main(["--output", str(ROOT / "samples/performance_report.json")])
