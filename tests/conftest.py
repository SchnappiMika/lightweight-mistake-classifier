# -*- coding: utf-8 -*-
"""pytest 公共夹具。

每个测试会话使用独立的临时工作区，保证测试之间互不干扰、可重复执行。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在导入 app.config 之前设置工作区
_TMP = Path(tempfile.mkdtemp(prefix="mc_test_"))
os.environ["MC_WORKSPACE"] = str(_TMP)

from app import config  # noqa: E402
from app.core.classify.engine import ClassificationEngine  # noqa: E402
from app.core.preprocess import pipeline  # noqa: E402
from app.core.storage import Repository  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _warm():
    pipeline.warmup()


@pytest.fixture(scope="function")
def repo(tmp_path, monkeypatch):
    """每个测试一个干净仓库。"""
    ws = tmp_path / "ws"
    for name, sub in (("WORKSPACE_DIR", ""), ("DATA_DIR", "data"), ("MEDIA_DIR", "media"),
                      ("EXPORT_DIR", "exports"), ("LOG_DIR", "logs"), ("MODEL_DIR", "model")):
        monkeypatch.setattr(config, name, ws / sub if sub else ws)
    monkeypatch.setattr(config, "SCHEME_FILE", ws / "data" / "schemes.json")
    monkeypatch.setattr(config, "QUESTION_FILE", ws / "data" / "questions.json")
    monkeypatch.setattr(config, "FEEDBACK_FILE", ws / "data" / "feedback_log.json")
    monkeypatch.setattr(config, "WEIGHT_FILE", ws / "model" / "weight_rules.json")
    monkeypatch.setattr(config, "PROFILE_FILE", ws / "model" / "label_profiles.json")
    monkeypatch.setattr(config, "SETTINGS_FILE", ws / "settings.json")
    return Repository()


@pytest.fixture
def math_engine(repo):
    return ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_MATH))
