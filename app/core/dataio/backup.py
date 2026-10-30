from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Union, get_args, get_origin, get_type_hints

from app import config
from app.core.models import FeatureBundle, FeedbackRecord, Question, Scheme
from app.core.storage import write_json

ROOTS = ("data", "media", "model", "exports", "settings.json")
REQUIRED = ("data/questions.json", "data/schemes.json", "data/feedback_log.json", "settings.json")
MAX_SIZE = 4 * 1024 ** 3
MAX_JSON_SIZE = 64 * 1024 ** 2
MAX_FILES = 50000
# Reserved bucket written by ClassificationEngine.save_profiles().
FEEDBACK_STATISTICS_KEY = "__feedback_document_statistics__"


class BackupError(ValueError):
    pass


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _safe_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or path.as_posix() != name
            or any(part in (".", "..") or part.endswith((".", " "))
                   or any(char in part for char in '<>:"|?*')
                   or part.split(".")[0].upper() in {
                       "CON", "PRN", "AUX", "NUL",
                       *["COM{}".format(number) for number in range(1, 10)],
                       *["LPT{}".format(number) for number in range(1, 10)]}
                   for part in path.parts)):
        raise BackupError("备份中含有不安全的文件路径：{}".format(name))
    if path.parts[0] not in ROOTS or (path.parts[0] == "settings.json" and len(path.parts) != 1):
        raise BackupError("备份中含有未知目录：{}".format(name))
    return path


def _validate_value(value, hint):
    if hint is Any:
        return
    origin, arguments = get_origin(hint), get_args(hint)
    if origin is Union:
        for candidate in arguments:
            try:
                _validate_value(value, candidate)
                return
            except BackupError:
                pass
    elif origin is list and isinstance(value, list):
        for item in value:
            _validate_value(item, arguments[0])
        return
    elif origin is dict and isinstance(value, dict):
        for key, item in value.items():
            _validate_value(key, arguments[0])
            _validate_value(item, arguments[1])
        return
    elif is_dataclass(hint) and isinstance(value, dict):
        hints = get_type_hints(hint)
        for key, item in value.items():
            if key in hints:
                _validate_value(item, hints[key])
        return
    elif hint is float and type(value) in (float, int) and math.isfinite(value):
        return
    elif hint in (str, int, bool, type(None)) and type(value) is hint:
        return
    raise BackupError("备份数据的字段类型不正确")


def _validate_payload(folder):
    values = {}
    for name in REQUIRED + ("model/weight_rules.json", "model/label_profiles.json"):
        path = folder / name
        if not path.is_file():
            if name in REQUIRED:
                raise BackupError("备份缺少必要文件：{}".format(name))
            continue
        if path.stat().st_size > MAX_JSON_SIZE:
            raise BackupError("备份数据文件过大：{}".format(name))
        try:
            values[name] = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise BackupError("备份数据无法读取：{}".format(name)) from exc
    for name, cls in ((REQUIRED[0], Question), (REQUIRED[1], Scheme), (REQUIRED[2], FeedbackRecord)):
        records = values[name]
        if not isinstance(records, list):
            raise BackupError("备份记录格式错误：{}".format(name))
        identities = set()
        for record in records:
            _validate_value(record, cls)
            identity = record.get("id")
            if not identity or identity in identities:
                raise BackupError("备份中存在缺失或重复的记录编号")
            identities.add(identity)
            if cls is Question and record.get("feature"):
                _validate_value(record["feature"], FeatureBundle)
    _validate_value(values["settings.json"], Dict[str, Any])
    for key, hint in (("current_scheme_id", str), ("theme", str), ("guide_done", bool)):
        if key in values["settings.json"]:
            _validate_value(values["settings.json"][key], hint)
    for question in values[REQUIRED[0]]:
        media = question.get("media_path", "")
        if media:
            path = _safe_name(media)
            if path.parts[0] != "media" or not (folder / media).is_file():
                raise BackupError("备份中的附件缺失：{}".format(media))
    weights = values.get("model/weight_rules.json", {})
    _validate_value(weights, Dict[str, Any])
    _validate_value(weights.get("rules", {}), Dict[str, Dict[str, float]])
    _validate_value(weights.get("hits", {}), Dict[str, int])
    for key in set(weights.get("rules", {})) | set(weights.get("hits", {})):
        if len(key.split("||")) != 2:
            raise BackupError("备份中的学习权重分组格式错误")
    profiles = values.get("model/label_profiles.json", {})
    _validate_value(profiles, Dict[str, Any])
    for key, bucket in profiles.items():
        if key == FEEDBACK_STATISTICS_KEY:
            _validate_value(bucket, Dict[str, Dict[str, Any]])
            for statistics in bucket.values():
                n_docs = statistics.get("n_docs", 0)
                frequencies = statistics.get("df", {})
                _validate_value(n_docs, int)
                _validate_value(frequencies, Dict[str, int])
                if n_docs < 0 or any(count < 0 or count > n_docs
                                     for count in frequencies.values()):
                    raise BackupError("备份中的反馈文档统计超出有效范围")
            continue
        _validate_value(bucket, Dict[str, Dict[str, Any]])
        for profile in bucket.values():
            _validate_value(profile.get("delta_tf", {}), Dict[str, float])
            _validate_value(profile.get("n_feedback", 0), int)
    return values


def create_backup(repo, destination):
    workspace = config.WORKSPACE_DIR.resolve()
    destination = Path(destination).resolve()
    for root in ROOTS:
        target = workspace / root
        if destination == target or target in destination.parents:
            raise BackupError("请把备份保存在工作区 backups 目录或其他目录，不能放入被备份的数据目录")
    repo.save_all()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".backup-", dir=str(workspace)) as temporary:
        snapshot = Path(temporary)
        for root in ROOTS:
            source = workspace / root
            if not source.exists():
                continue
            paths = [source] if source.is_file() else source.rglob("*")
            for path in paths:
                if path.is_symlink() or path.resolve() != path.absolute():
                    raise BackupError("工作区包含链接文件，请先将其转换为普通文件：{}".format(path))
                if path.is_file():
                    relative = path.relative_to(workspace)
                    target = snapshot / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        questions = [question.to_dict() for question in repo.questions]
        archived = {}
        for question in questions:
            if not question["media_path"]:
                continue
            source = Path(question["media_path"])
            if not source.is_absolute():
                source = workspace / source
            source = source.resolve()
            if not source.is_file():
                raise BackupError("附件不存在，无法生成完整备份：{}".format(source))
            if source not in archived:
                try:
                    relative = source.relative_to(workspace / "media")
                    name = "media/" + relative.as_posix()
                except ValueError:
                    name = "media/{}_{}".format(uuid.uuid4().hex, source.name)
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                archived[source] = name
            question["media_path"] = archived[source]
        write_json(snapshot / REQUIRED[0], questions)
        _validate_payload(snapshot)
        entries = {}
        total_size = 0
        for path in sorted(snapshot.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(snapshot).as_posix()
            _safe_name(name)
            size = path.stat().st_size
            total_size += size
            if total_size > MAX_SIZE:
                raise BackupError("当前备份超过 4GB，请先整理历史导出或附件")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries[name] = {"size": size, "sha256": digest.hexdigest()}
        manifest = {"format": "mistake-classifier-backup", "version": 1,
                    "app_version": config.APP_VERSION,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "questions": len(questions), "files": entries}
        manifest_bytes = _json_bytes(manifest)
        if len(entries) + 1 > MAX_FILES or total_size + len(manifest_bytes) > MAX_SIZE:
            raise BackupError("备份超过大小或文件数量限制，请先整理历史导出及附件")
        descriptor, temporary_zip = tempfile.mkstemp(prefix=".backup-", suffix=".zip", dir=destination.parent)
        os.close(descriptor)
        try:
            with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", manifest_bytes)
                for name in entries:
                    archive.write(snapshot / name, name)
            os.replace(temporary_zip, destination)
        finally:
            if os.path.exists(temporary_zip):
                os.unlink(temporary_zip)
    return str(destination)


def _unpack(archive_path, staging):
    try:
        with zipfile.ZipFile(archive_path) as archive:
            information = archive.infolist()
            if len(information) > MAX_FILES or sum(item.file_size for item in information) > MAX_SIZE:
                raise BackupError("备份文件过大或文件数量过多")
            if archive.getinfo("manifest.json").file_size > MAX_JSON_SIZE:
                raise BackupError("备份清单过大")
            manifest = json.loads(archive.read("manifest.json"))
            if (not isinstance(manifest, dict) or manifest.get("format") != "mistake-classifier-backup"
                    or manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict)):
                raise BackupError("不是受支持的错题软件备份")
            entries = manifest["files"]
            names = [item.filename for item in information]
            if (len({name.casefold() for name in names}) != len(names)
                    or set(names) != set(entries) | {"manifest.json"}):
                raise BackupError("备份文件清单不完整或存在重复路径")
            for name, entry in entries.items():
                relative = _safe_name(name)
                item = archive.getinfo(name)
                if (not isinstance(entry, dict) or item.file_size != entry.get("size")
                        or stat.S_ISLNK(item.external_attr >> 16) or item.is_dir()):
                    raise BackupError("备份文件信息不正确：{}".format(name))
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(item) as source, target.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        output.write(chunk)
                if digest.hexdigest() != entry.get("sha256"):
                    raise BackupError("备份校验失败：{}".format(name))
        return _validate_payload(staging)
    except (zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise BackupError("备份文件损坏或无法读取：{}".format(exc)) from exc


def restore_backup(repo, archive_path):
    workspace = config.WORKSPACE_DIR.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".restore-", dir=str(workspace)))
    retain_recovery = False
    try:
        staging = temporary / "incoming"
        staging.mkdir()
        values = _unpack(archive_path, staging)
        for question in values[REQUIRED[0]]:
            if question.get("media_path"):
                question["media_path"] = str(workspace / question["media_path"])
        write_json(staging / REQUIRED[0], values[REQUIRED[0]])
        safety = workspace / "backups" / "恢复前_{}_{}.zip".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:8])
        create_backup(repo, safety)
        previous = temporary / "previous"
        previous.mkdir()
        moved, installed = [], []
        try:
            for root in ROOTS:
                destination = workspace / root
                if destination.exists():
                    os.replace(destination, previous / root)
                    moved.append(root)
                incoming = staging / root
                if incoming.exists():
                    os.replace(incoming, destination)
                    installed.append(root)
            repo.load_all()
        except BaseException:
            try:
                for root in reversed(installed):
                    os.replace(workspace / root, staging / root)
                for root in reversed(moved):
                    os.replace(previous / root, workspace / root)
                repo.load_all()
            except BaseException as rollback_error:
                retain_recovery = True
                raise BackupError("恢复和回滚均失败，请保留现场并从恢复前备份处理。"
                                  "备份：{}；原数据暂存：{}；错误：{}".format(
                                      safety, previous, rollback_error)) from rollback_error
            raise
    finally:
        if not retain_recovery:
            shutil.rmtree(temporary)
    return str(safety)
