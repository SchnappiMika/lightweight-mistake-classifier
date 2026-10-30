# -*- coding: utf-8 -*-
"""本地离线持久化层。

采用 JSON + 原子替换（tmp -> os.replace）的方式落盘：
* 无数据库依赖，符合"轻量化 / 免配置 / 打包即用"的目标；
* 原子替换减少单个 JSON 文件写到一半的风险，完整恢复使用独立备份；
* 文件人类可读，用户可直接编辑或备份，也便于开源后二次开发。
"""
from __future__ import annotations

import json
import filecmp
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import config
from app.core.models import FeedbackRecord, Question, Scheme

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 基础 JSON 读写
# --------------------------------------------------------------------------- #
def read_json(path: Path, default: Any = None) -> Any:
    try:
        if not Path(path).exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:      # 文件损坏时降级为默认值并备份现场
        log.warning("读取 %s 失败：%s", path, exc)
        try:
            shutil.copy2(path, str(path) + ".corrupt")
        except OSError:
            pass
        return default


def write_json(path: Path, data: Any) -> None:
    """原子写入，避免半截文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# 仓库
# --------------------------------------------------------------------------- #
class Repository:
    """统一的数据访问入口（分类体系 / 错题 / 反馈日志 / 设置）。"""

    def __init__(self) -> None:
        config.ensure_workspace()
        self.schemes: List[Scheme] = []
        self.questions: List[Question] = []
        self.feedback: List[FeedbackRecord] = []
        self.settings: Dict[str, Any] = {}
        self.load_all()

    # ---------------- 加载 ---------------- #
    def load_all(self) -> None:
        self.settings = read_json(config.SETTINGS_FILE, {}) or {}
        raw_schemes = read_json(config.SCHEME_FILE, None)
        if not raw_schemes:
            raw_schemes = self._seed_schemes()
            write_json(config.SCHEME_FILE, raw_schemes)
        self.schemes = [Scheme.from_dict(x) for x in raw_schemes]
        self.questions = [Question.from_dict(x) for x in (read_json(config.QUESTION_FILE, []) or [])]
        self.feedback = [FeedbackRecord.from_dict(x)
                         for x in (read_json(config.FEEDBACK_FILE, []) or [])]

    @staticmethod
    def _seed_schemes() -> List[Dict[str, Any]]:
        """首次启动时从 resources/schemes/ 载入内置分类体系种子。"""
        out: List[Dict[str, Any]] = []
        seed_dir = Path(config.SCHEME_SEED_DIR)
        if seed_dir.exists():
            for p in sorted(seed_dir.glob("*.json")):
                data = read_json(p, None)
                if isinstance(data, list):
                    out.extend(data)
                elif isinstance(data, dict):
                    out.append(data)
        return out

    # ---------------- 保存 ---------------- #
    def save_schemes(self) -> None:
        write_json(config.SCHEME_FILE, [s.to_dict() for s in self.schemes])

    def save_questions(self) -> None:
        write_json(config.QUESTION_FILE, [q.to_dict() for q in self.questions])

    def save_feedback(self) -> None:
        write_json(config.FEEDBACK_FILE, [f.to_dict() for f in self.feedback])

    def save_settings(self) -> None:
        write_json(config.SETTINGS_FILE, self.settings)

    def save_all(self) -> None:
        self.save_schemes()
        self.save_questions()
        self.save_feedback()
        self.save_settings()

    def reconcile_questions(self) -> bool:
        changed = False
        for question in self.questions:
            if not question.scheme_id:
                continue
            scheme = self.scheme_by_id(question.scheme_id)
            if scheme is None:
                question.invalidate_classification()
                changed = True
                continue
            for dim_key, label_id in list(question.labels.items()):
                dimension = scheme.dim(dim_key)
                label = dimension.label_by_id(label_id) if dimension and dimension.enabled else None
                if label is None:
                    question.invalidate_classification()
                    changed = True
                    break
                if question.label_names.get(dim_key) != label.name:
                    question.label_names[dim_key] = label.name
                    changed = True
            for dim_key, candidates in question.candidates.items():
                dimension = scheme.dim(dim_key)
                for candidate in candidates:
                    label = dimension.label_by_id(candidate.get("label_id", "")) if dimension else None
                    if label and candidate.get("label_name") != label.name:
                        candidate["label_name"] = label.name
                        changed = True
        if changed:
            self.save_questions()
        return changed

    # ---------------- 分类体系 ---------------- #
    def scheme_by_id(self, sid: str) -> Optional[Scheme]:
        for s in self.schemes:
            if s.id == sid:
                return s
        return None

    def scheme_by_subject(self, subject: str) -> Optional[Scheme]:
        for s in self.schemes:
            if s.subject == subject:
                return s
        return None

    def add_scheme(self, scheme: Scheme) -> Scheme:
        self.schemes.append(scheme)
        self.save_schemes()
        return scheme

    def remove_scheme(self, sid: str) -> bool:
        before = len(self.schemes)
        self.schemes = [s for s in self.schemes if s.id != sid]
        if len(self.schemes) != before:
            self.save_schemes()
            self.reconcile_questions()
            return True
        return False

    def reset_builtin_schemes(self) -> None:
        """恢复内置体系（用户自建体系保留）。"""
        custom = [s for s in self.schemes if not s.builtin]
        self.schemes = [Scheme.from_dict(x) for x in self._seed_schemes()] + custom
        self.save_schemes()
        self.reconcile_questions()

    # ---------------- 错题 ---------------- #
    def question_by_id(self, qid: str) -> Optional[Question]:
        for q in self.questions:
            if q.id == qid:
                return q
        return None

    def add_questions(self, items: List[Question]) -> None:
        previous_count = len(self.questions)
        self.questions.extend(items)
        try:
            self.save_questions()
        except BaseException:
            del self.questions[previous_count:]
            raise

    def add_imported_questions(self, items: List[Question], skip_duplicates: bool = True):
        def fingerprint(question):
            media = ""
            if question.media_path:
                path = Path(question.media_path)
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    media = digest.hexdigest()
                else:
                    media = question.media_path
            return (question.raw_text.strip(), question.annotation.strip(), question.subject,
                    question.qtype, question.priority, media)

        seen = {fingerprint(question) for question in self.questions} if skip_duplicates else set()
        accepted = []
        for question in items:
            identity = fingerprint(question)
            if skip_duplicates and identity in seen:
                continue
            seen.add(identity)
            accepted.append(question)
        if accepted:
            self.add_questions(accepted)
        return len(accepted), len(items) - len(accepted)

    def remove_questions(self, ids: List[str]) -> None:
        idset = set(ids)
        self.questions = [q for q in self.questions if q.id not in idset]
        self.save_questions()

    def clear_questions(self) -> None:
        self.questions = []
        self.save_questions()

    # ---------------- 反馈 ---------------- #
    def add_feedback(self, rec: FeedbackRecord) -> None:
        self.feedback.append(rec)
        self.save_feedback()

    def clear_feedback(self) -> None:
        self.feedback = []
        self.save_feedback()

    # ---------------- 媒体文件归档 ---------------- #
    @staticmethod
    def import_media(src: str) -> str:
        """把用户选择的图片 / 音频复制进工作区，避免原文件移动后预览失效。"""
        src_p = Path(src)
        if not src_p.is_file():
            raise FileNotFoundError("附件文件不存在：{}".format(src))
        config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        dst = config.MEDIA_DIR / src_p.name
        i = 1
        while dst.exists():
            if src_p.resolve() == dst.resolve() or filecmp.cmp(src_p, dst, shallow=False):
                return str(dst.resolve())
            dst = config.MEDIA_DIR / "{}_{}{}".format(src_p.stem, i, src_p.suffix)
            i += 1
        try:
            shutil.copy2(str(src_p), str(dst))
            return str(dst.resolve())
        except OSError as exc:
            log.warning("复制媒体文件失败：%s", exc)
            raise OSError("附件复制到工作区失败：{}".format(exc)) from exc
