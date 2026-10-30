# -*- coding: utf-8 -*-
"""应用上下文：跨页面共享的状态与信号总线。

各页面之间不直接互相持有引用，全部通过本对象的信号解耦通信：
    错题变了      -> questions_changed
    分类体系变了  -> scheme_changed（同时让对应引擎的特征库失效，下次分类自动热重建）
    产生了反馈    -> feedback_changed
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from app import config
from app.core.classify import weights as W
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.models import Question, Scheme
from app.core.storage import Repository, write_json

log = logging.getLogger(__name__)


class AppContext(QObject):
    """全局上下文（单例，由主窗口创建并注入各页面）。"""

    questions_changed = pyqtSignal()
    question_edited = pyqtSignal()
    scheme_changed = pyqtSignal()
    feedback_changed = pyqtSignal()
    status = pyqtSignal(str)
    toast = pyqtSignal(str, str)          # message, kind(info/ok/warn/error)
    task_changed = pyqtSignal(str)
    workspace_reloaded = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.repo = Repository()
        self.active_task = ""
        self.rules = W.WeightRules()
        self._engines: Dict[str, ClassificationEngine] = {}
        self._current_scheme_id: str = self.repo.settings.get("current_scheme_id", "")
        if not self.repo.scheme_by_id(self._current_scheme_id) and self.repo.schemes:
            self._current_scheme_id = self.repo.schemes[0].id

    # ------------------------------------------------------------------ #
    # 分类体系
    # ------------------------------------------------------------------ #
    @property
    def schemes(self) -> List[Scheme]:
        return self.repo.schemes

    def current_scheme(self) -> Optional[Scheme]:
        scheme = self.repo.scheme_by_id(self._current_scheme_id)
        if scheme is None and self.repo.schemes:
            scheme = self.repo.schemes[0]
            self._current_scheme_id = scheme.id
            self.repo.settings["current_scheme_id"] = scheme.id
            self.repo.save_settings()
        return scheme

    def set_current_scheme(self, sid: str) -> None:
        if sid == self._current_scheme_id:
            return
        self._current_scheme_id = sid
        self.repo.settings["current_scheme_id"] = sid
        self.repo.save_settings()
        self.scheme_changed.emit()

    def engine(self, scheme: Optional[Scheme] = None) -> Optional[ClassificationEngine]:
        """取得（并缓存）某套体系的分类引擎。"""
        sc = scheme or self.current_scheme()
        if sc is None:
            return None
        eng = self._engines.get(sc.id)
        if eng is None or eng.scheme is not sc:
            eng = ClassificationEngine(sc, rules=self.rules)
            self._engines[sc.id] = eng
        return eng

    def optimizer(self, scheme: Optional[Scheme] = None) -> Optional[FeedbackOptimizer]:
        eng = self.engine(scheme)
        return FeedbackOptimizer(eng) if eng else None

    def notify_scheme_changed(self, scheme: Optional[Scheme] = None) -> None:
        """用户编辑了分类体系：落盘 + 让特征库失效 + 广播。"""
        sc = scheme or self.current_scheme()
        if sc is not None:
            sc.touch()
            eng = self._engines.get(sc.id)
            if eng is not None:
                eng.invalidate()          # 下次 classify 时自动重建索引与画像
        self.repo.save_schemes()
        if self.repo.reconcile_questions():
            self.questions_changed.emit()
        self.scheme_changed.emit()

    def scheme_for_subject(self, subject: str) -> Optional[Scheme]:
        current = self.current_scheme()
        if current and current.subject in (subject, config.SUBJECT_GENERAL):
            return current
        return self.repo.scheme_by_subject(subject) or current

    # ------------------------------------------------------------------ #
    # 错题
    # ------------------------------------------------------------------ #
    @property
    def questions(self) -> List[Question]:
        return self.repo.questions

    def add_questions(self, items: List[Question]) -> None:
        if not items:
            return
        self.repo.add_questions(items)
        self.questions_changed.emit()

    def add_imported_questions(self, items, skip_duplicates=True):
        counts = self.repo.add_imported_questions(items, skip_duplicates)
        self.questions_changed.emit()
        return counts

    def begin_task(self, name: str) -> bool:
        if self.active_task:
            self.toast.emit("正在{}，请等待结束后再操作".format(self.active_task), "warn")
            return False
        self.active_task = name
        self.task_changed.emit(name)
        return True

    def end_task(self, name: str) -> None:
        if self.active_task == name:
            self.active_task = ""
            self.task_changed.emit("")

    def workspace_restored(self) -> None:
        self.rules = W.WeightRules()
        self._engines.clear()
        self._current_scheme_id = self.repo.settings.get("current_scheme_id", "")
        self.current_scheme()
        self.workspace_reloaded.emit()
        self.scheme_changed.emit()
        self.questions_changed.emit()
        self.feedback_changed.emit()

    def remove_questions(self, ids: List[str]) -> None:
        if not ids:
            return
        self.repo.remove_questions(ids)
        self.questions_changed.emit()

    def clear_questions(self) -> None:
        self.repo.clear_questions()
        self.questions_changed.emit()

    def save_questions(self) -> None:
        self.repo.save_questions()

    def notify_question_edited(self, question: Question) -> None:
        question.invalidate_classification()
        self.repo.save_questions()
        self.question_edited.emit()

    def classified_questions(self) -> List[Question]:
        return [q for q in self.repo.questions if q.classified]

    # ------------------------------------------------------------------ #
    # 反馈
    # ------------------------------------------------------------------ #
    def apply_correction(self, q: Question, dim_key: str, label_id: str) -> bool:
        """应用一次人工修正并触发增量优化。"""
        sc = self.repo.scheme_by_id(q.scheme_id) or self.current_scheme()
        opt = self.optimizer(sc)
        if opt is None:
            return False
        rec = opt.apply_correction(q, dim_key, label_id)
        self.repo.save_questions()
        if rec is not None:
            self.repo.add_feedback(rec)
            self.feedback_changed.emit()
        self.questions_changed.emit()
        return rec is not None

    def reset_optimization(self) -> None:
        """清空全部反馈增量与权重改写，回到出厂状态。"""
        write_json(config.PROFILE_FILE, {})
        self._engines.clear()
        self.rules.reset()
        self.repo.clear_feedback()
        for q in self.repo.questions:
            q.corrected = {}
        self.repo.save_questions()
        self.feedback_changed.emit()
        self.questions_changed.emit()

    # ------------------------------------------------------------------ #
    # 设置
    # ------------------------------------------------------------------ #
    def setting(self, key: str, default=None):
        return self.repo.settings.get(key, default)

    def set_setting(self, key: str, value) -> None:
        self.repo.settings[key] = value
        self.repo.save_settings()

    def theme_name(self) -> str:
        return self.setting("theme", config.DEFAULT_THEME)
