from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from app import config
from app.core.dataio import backup, importer
from app.core.models import CategoryLabel, Dimension, FeedbackRecord, Question, Scheme
from app.core.storage import Repository, read_json, write_json


def switch_workspace(monkeypatch, folder):
    for name, suffix in (("WORKSPACE_DIR", ""), ("DATA_DIR", "data"), ("MEDIA_DIR", "media"),
                         ("MODEL_DIR", "model"), ("EXPORT_DIR", "exports"), ("LOG_DIR", "logs"),
                         ("QUESTION_FILE", "data/questions.json"), ("SCHEME_FILE", "data/schemes.json"),
                         ("FEEDBACK_FILE", "data/feedback_log.json"), ("SETTINGS_FILE", "settings.json"),
                         ("PROFILE_FILE", "model/label_profiles.json"), ("WEIGHT_FILE", "model/weight_rules.json")):
        monkeypatch.setattr(config, name, folder / suffix if suffix else folder)


def sample_data(repo, tmp_path):
    scheme = Scheme(name="自建数学", dims=[Dimension(labels=[CategoryLabel(name="积分")])])
    repo.add_scheme(scheme)
    media = tmp_path / "原始附件.png"
    media.write_bytes(b"original-media-content")
    question = Question(raw_text="计算定积分", media_path=repo.import_media(str(media)),
                        scheme_id=scheme.id, classified=True,
                        labels={"knowledge": scheme.dims[0].labels[0].id},
                        label_names={"knowledge": "积分"}, corrected={"knowledge": True})
    repo.add_questions([question])
    repo.add_feedback(FeedbackRecord(question_id=question.id, scheme_id=scheme.id, tokens=["积分"]))
    repo.settings = {"current_scheme_id": scheme.id, "theme": "dark", "guide_done": True}
    write_json(config.WEIGHT_FILE, {"rules": {"数学||text": {"text": 0.8}}, "hits": {"数学||text": 2}})
    write_json(config.PROFILE_FILE, {scheme.id: {scheme.dims[0].labels[0].id:
               {"label_id": scheme.dims[0].labels[0].id, "delta_tf": {"积分": 2.5}, "n_feedback": 1}}})
    (config.EXPORT_DIR / "历史导出.txt").write_text("复习记录", encoding="utf-8")
    return question, scheme


def rewrite_archive(source, destination, change, update_hashes=True):
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    change(files)
    if update_hashes:
        manifest = json.loads(files["manifest.json"])
        manifest["files"] = {name: {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                             for name, value in files.items() if name != "manifest.json"}
        files["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(destination, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)


def test_backup_restores_to_new_workspace_with_learning_and_attachments(repo, tmp_path, monkeypatch):
    question, scheme = sample_data(repo, tmp_path)
    weights, profiles = read_json(config.WEIGHT_FILE), read_json(config.PROFILE_FILE)
    path = backup.create_backup(repo, tmp_path / "完整备份.zip")
    with zipfile.ZipFile(path) as archive:
        records = json.loads(archive.read("data/questions.json"))
        assert records[0]["media_path"].startswith("media/")
        assert not any(name.startswith(("logs/", "backups/", ".venv/")) for name in archive.namelist())
    switch_workspace(monkeypatch, tmp_path / "新电脑" / "工作区")
    target = Repository()
    target.add_questions([Question(raw_text="恢复前的题")])
    safety = backup.restore_backup(target, path)
    restored = target.question_by_id(question.id)
    assert restored.corrected == question.corrected
    assert target.scheme_by_id(scheme.id).name == scheme.name
    assert len(target.feedback) == 1 and target.feedback[0].tokens == ["积分"]
    assert read_json(config.WEIGHT_FILE) == weights
    assert read_json(config.PROFILE_FILE) == profiles
    assert target.settings["theme"] == "dark"
    assert Path(restored.media_path).is_relative_to(config.MEDIA_DIR)
    assert Path(restored.media_path).read_bytes() == b"original-media-content"
    assert (config.EXPORT_DIR / "历史导出.txt").read_text(encoding="utf-8") == "复习记录"
    backup.restore_backup(target, safety)
    assert [item.raw_text for item in target.questions] == ["恢复前的题"]


def test_external_legacy_attachment_is_included(repo, tmp_path):
    attachment = tmp_path / "旧图片.png"
    attachment.write_bytes(b"legacy")
    repo.add_questions([Question(raw_text="旧数据", media_path=str(attachment))])
    archive = backup.create_backup(repo, tmp_path / "备份.zip")
    backup.restore_backup(repo, archive)
    assert Path(repo.questions[0].media_path).parent == config.MEDIA_DIR
    assert Path(repo.questions[0].media_path).read_bytes() == b"legacy"


@pytest.mark.parametrize("limit,value", [("MAX_SIZE", 1024), ("MAX_FILES", 3)])
def test_backup_limits_do_not_replace_an_existing_backup(repo, tmp_path, monkeypatch, limit, value):
    destination = tmp_path / "原有备份.zip"
    destination.write_bytes(b"existing-backup")
    monkeypatch.setattr(backup, limit, value)
    with pytest.raises(backup.BackupError):
        backup.create_backup(repo, destination)
    assert destination.read_bytes() == b"existing-backup"


@pytest.mark.parametrize("corruption", ["checksum", "path", "type", "missing", "version", "duplicate", "profile"])
def test_bad_backup_never_replaces_current_data(repo, tmp_path, corruption):
    repo.add_questions([Question(raw_text="要保留的题")])
    original = backup.create_backup(repo, tmp_path / "正常.zip")
    damaged = tmp_path / "损坏.zip"

    def change(files):
        if corruption == "checksum":
            files["data/questions.json"] += b" "
        elif corruption == "path":
            files["media/../../outside.txt"] = b"bad"
        elif corruption == "type":
            files["data/questions.json"] = b'[{"id":"q1", "raw_text":[], "labels":{}}]'
        elif corruption == "missing":
            del files["data/schemes.json"]
        elif corruption == "profile":
            files["model/label_profiles.json"] = b'{"scheme":{"label":{"delta_tf":[]}}}'
        elif corruption == "version":
            manifest = json.loads(files["manifest.json"])
            manifest["version"] = 999
            files["manifest.json"] = json.dumps(manifest).encode()
        elif corruption == "duplicate":
            files["DATA/questions.json"] = files["data/questions.json"]

    rewrite_archive(original, damaged, change, update_hashes=corruption != "checksum")
    before = config.QUESTION_FILE.read_bytes()
    with pytest.raises(backup.BackupError):
        backup.restore_backup(repo, damaged)
    assert config.QUESTION_FILE.read_bytes() == before
    assert repo.questions[0].raw_text == "要保留的题"
    assert not (config.WORKSPACE_DIR / "backups").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_restore_rolls_back_when_installing_new_files_fails(repo, tmp_path, monkeypatch):
    repo.add_questions([Question(raw_text="备份里的题")])
    path = backup.create_backup(repo, tmp_path / "备份.zip")
    repo.questions[0].raw_text = "当前正在用的题"
    repo.save_questions()
    replace = backup.os.replace
    failed = []

    def fail_install(source, destination):
        if Path(source).name == "settings.json" and Path(source).parent.name == "incoming" and not failed:
            failed.append(True)
            raise OSError("模拟恢复时磁盘写入失败")
        return replace(source, destination)

    (config.MEDIA_DIR / "保留.png").write_bytes(b"keep")
    monkeypatch.setattr(backup.os, "replace", fail_install)
    with pytest.raises(OSError, match="模拟"):
        backup.restore_backup(repo, path)
    assert repo.questions[0].raw_text == "当前正在用的题"
    assert (config.MEDIA_DIR / "保留.png").read_bytes() == b"keep"
    assert list((config.WORKSPACE_DIR / "backups").glob("*.zip"))


def test_missing_attachment_does_not_leave_a_false_success_backup(repo, tmp_path):
    repo.add_questions([Question(raw_text="有丢失附件", media_path=str(tmp_path / "丢失.png"))])
    destination = tmp_path / "不应产生.zip"
    with pytest.raises(backup.BackupError, match="附件不存在"):
        backup.create_backup(repo, destination)
    assert not destination.exists()


def test_restore_retains_recovery_files_if_rollback_also_fails(repo, tmp_path, monkeypatch):
    repo.add_questions([Question(raw_text="备份中的题")])
    archive = backup.create_backup(repo, tmp_path / "备份.zip")
    repo.questions[0].raw_text = "恢复前的题"
    repo.save_questions()
    replace = backup.os.replace

    def fail_install_and_rollback(source, destination):
        path = Path(source)
        if ((path.parent.name == "incoming" and path.name == "settings.json")
                or (path.parent.name == "previous" and path.name == "data")):
            raise OSError("模拟持续磁盘故障")
        return replace(source, destination)

    monkeypatch.setattr(backup.os, "replace", fail_install_and_rollback)
    with pytest.raises(backup.BackupError, match="恢复和回滚均失败"):
        backup.restore_backup(repo, archive)
    retained = list(config.WORKSPACE_DIR.glob(".restore-*/previous/data/questions.json"))
    assert len(retained) == 1
    assert read_json(retained[0])[0]["raw_text"] == "恢复前的题"
    assert list((config.WORKSPACE_DIR / "backups").glob("*.zip"))


def test_gbk_csv_relative_attachment_is_archived(repo, tmp_path):
    source = tmp_path / "题目来源"
    source.mkdir()
    (source / "配图.png").write_bytes(b"picture")
    table = source / "题目.csv"
    table.write_bytes("题干,附件\n计算斜面受力,配图.png\n".encode("gbk"))
    question = importer.import_excel(str(table))[0]
    assert question.raw_text == "计算斜面受力"
    assert Path(question.media_path).parent == config.MEDIA_DIR
    (source / "配图.png").unlink()
    assert Path(question.media_path).read_bytes() == b"picture"


def test_spreadsheet_keeps_good_rows_when_attachment_missing(repo, tmp_path):
    table = tmp_path / "题目.csv"
    table.write_text("题干,附件\n第一道,\n第二道,不存在.png\n第三道,\n", encoding="utf-8")
    errors = []
    items = importer.import_excel(str(table), errors=errors)
    assert [item.raw_text for item in items] == ["第一道", "第三道"]
    assert len(errors) == 1 and "第 3 行" in errors[0]


def test_repeated_import_deduplicates_by_attachment_content(repo, tmp_path):
    source = tmp_path / "a.png"
    source.write_bytes(b"same")
    renamed = tmp_path / "b.png"
    renamed.write_bytes(b"same")
    first = importer.import_single("一道题", media_path=str(source))
    second = importer.import_single("一道题", media_path=str(renamed))
    assert repo.add_imported_questions([first, second]) == (1, 1)
    assert repo.add_imported_questions([second]) == (0, 1)
    assert repo.add_imported_questions([second], skip_duplicates=False) == (1, 0)


def test_importing_same_attachment_reuses_existing_copy(repo, tmp_path):
    source = tmp_path / "a.png"
    source.write_bytes(b"same")
    first = repo.import_media(str(source))
    assert repo.import_media(str(source)) == first
    assert repo.import_media(first) == first
    assert len(list(config.MEDIA_DIR.iterdir())) == 1


def test_failed_import_save_does_not_leave_unsaved_duplicates(repo, monkeypatch):
    original = Question(raw_text="原有错题")
    repo.add_questions([original])
    previous = config.QUESTION_FILE.read_bytes()

    def fail_save():
        raise OSError("磁盘不可写")

    with monkeypatch.context() as changes:
        changes.setattr(repo, "save_questions", fail_save)
        with pytest.raises(OSError, match="磁盘不可写"):
            repo.add_imported_questions([Question(raw_text="导入的新题")])
    assert repo.questions == [original]
    assert config.QUESTION_FILE.read_bytes() == previous
    assert repo.add_imported_questions([Question(raw_text="导入的新题")]) == (1, 0)
