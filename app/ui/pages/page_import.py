# -*- coding: utf-8 -*-
"""功能模块 1：多模态错题导入与预览（需求 4.3.2 第 1 点）。

布局对应需求 4.3.3：中间核心内容展示区在本模式下分为
    「错题列表区」+「文本编辑区 / 图片·音频预览区 / 标注补充区」。

三种导入模式：
    * 单条手动录入
    * 批量导入 TXT / Excel
    * 本地导入 公式图片 / 工科图形题图片 / 英语听力音频（支持整目录扫描）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
                             QPushButton, QSplitter, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from app import config
from app.core.dataio import importer
from app.core.models import Question
from app.ui import icons, theme
from app.ui.pages.base_page import BasePage
from app.ui.widgets import Card, SectionTitle, batch_view_updates, button, hline, label
from app.ui.workers import ImportWorker, stop_worker

# QtMultimedia 在部分精简系统 / 容器里缺音频后端，做成可选依赖
try:
    from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
    _HAS_MEDIA = True
except Exception:                          # noqa: BLE001
    _HAS_MEDIA = False


class ImportPage(BasePage):
    """多模态错题导入与预览页。"""

    title = "多模态错题导入与预览"
    subtitle = "支持单条录入 / TXT·Excel 批量导入 / 公式图片·图形题·听力音频本地导入"

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(ctx, parent)
        self._worker: Optional[ImportWorker] = None
        self._current: Optional[Question] = None
        self._player = None
        self._audio_active_path = ""
        self._audio_error_reported = False
        self._audio_description = ""
        self._retry = None
        self._build()
        ctx.questions_changed.connect(self.refresh_list)
        ctx.workspace_reloaded.connect(self._workspace_reloaded)

    def _workspace_reloaded(self):
        self._stop_play()
        self._current = None
        self.table.clearSelection()
        self.ed_search.clear()
        self._retry = None
        self._clear_detail()
        self.refresh_list()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # ---------------- 左：错题列表区 ---------------- #
        left = Card(margin=10, spacing=8)
        self.sec_list = SectionTitle("错题列表", "尚未导入任何错题")
        left.box.addWidget(self.sec_list)

        filt = QHBoxLayout()
        filt.setSpacing(6)
        self.cb_filter_subject = QComboBox()
        self.cb_filter_subject.addItem("全部学科", "")
        for s in config.SUBJECTS:
            self.cb_filter_subject.addItem(s, s)
        self.cb_filter_subject.currentIndexChanged.connect(self.refresh_list)
        self.cb_filter_qtype = QComboBox()
        self.cb_filter_qtype.addItem("全部题型", "")
        for k, v in config.QTYPE_LABELS.items():
            self.cb_filter_qtype.addItem(v, k)
        self.cb_filter_qtype.currentIndexChanged.connect(self.refresh_list)
        filt.addWidget(self.cb_filter_subject)
        filt.addWidget(self.cb_filter_qtype)
        filt.addStretch(1)
        left.box.addLayout(filt)
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("搜索题干、标注或附件名")
        self.ed_search.setClearButtonEnabled(True)
        self.ed_search.setToolTip("按输入文字匹配，不区分英文大小写；与学科、题型筛选同时生效。")
        self.ed_search.textChanged.connect(self.refresh_list)
        search_row = QHBoxLayout()
        search_row.addWidget(self.ed_search, 1)
        self.btn_clear_search = button("清空", "ghost", "", self.ed_search.clear,
                                       "清空搜索文字，保留学科和题型筛选")
        search_row.addWidget(self.btn_clear_search)
        left.box.addLayout(search_row)
        self.cb_skip_duplicates = QCheckBox("导入时跳过完全重复的题目")
        self.cb_skip_duplicates.setToolTip("题干、标注、学科、题型、复习优先级及附件内容均相同，才视为重复；取消勾选可保留重复项。")
        self.cb_skip_duplicates.setChecked(True)
        left.box.addWidget(self.cb_skip_duplicates)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["题干摘要", "学科", "题型", "附件"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in (1, 2, 3):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_select)
        left.box.addWidget(self.table, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left.box.addWidget(self.progress)
        split.addWidget(left)

        # ---------------- 右：编辑 / 预览 / 标注 ---------------- #
        right = Card(margin=10, spacing=8)
        self.sec_detail = SectionTitle("错题详情", "从左侧选择一条错题，或点击右侧「单条录入」")
        right.box.addWidget(self.sec_detail)

        meta = QHBoxLayout()
        meta.setSpacing(6)
        meta.addWidget(label("学科", "dim"))
        self.cb_subject = QComboBox()
        self.cb_subject.addItems(config.SUBJECTS)
        self.cb_subject.currentIndexChanged.connect(self._on_meta_changed)
        meta.addWidget(self.cb_subject)
        meta.addWidget(label("题型", "dim"))
        self.cb_qtype = QComboBox()
        for k, v in config.QTYPE_LABELS.items():
            self.cb_qtype.addItem(v, k)
        self.cb_qtype.currentIndexChanged.connect(self._on_meta_changed)
        meta.addWidget(self.cb_qtype)
        meta.addStretch(1)
        self.btn_attach = button("关联附件", "", "folder", self._attach_media,
                                 "为当前错题关联公式图片 / 图形题截图 / 听力音频")
        meta.addWidget(self.btn_attach)
        right.box.addLayout(meta)
        self.btn_image_ocr = button("图片文字：识别与校对", "", "image", self._recognize_image,
                                    "识别印刷文字，对照原图修改并应用为题干；支持失败重试")
        right.box.addWidget(self.btn_image_ocr)
        media_actions = QHBoxLayout()
        self.btn_formula_ocr = button("公式图片：识别与校对", "", "image",
                                      self._recognize_formula,
                                      "本地识别 LaTeX，确认后参与公式知识点映射")
        self.btn_audio_transcribe = button("听力音频：转写与校对", "", "audio",
                                           self._transcribe_audio,
                                           "本地 Whisper 转写，确认后参与英文特征提取")
        media_actions.addWidget(self.btn_formula_ocr)
        media_actions.addWidget(self.btn_audio_transcribe)
        right.box.addLayout(media_actions)

        # 文本编辑区
        right.box.addWidget(label("题干文本（可直接编辑；公式请用 $...$ 包裹）", "dim"))
        self.ed_text = QPlainTextEdit()
        self.ed_text.setPlaceholderText(
            "在此录入或修改题干。\n"
            "例：求极限 $\\lim_{x \\to 0}\\frac{\\sin x - x}{x^3}$，用洛必达法则。")
        self.ed_text.textChanged.connect(self._on_text_changed)
        right.box.addWidget(self.ed_text, 3)

        # 图片 / 音频预览区
        self.preview_stack = QStackedWidget()
        self.preview_stack.setMinimumHeight(150)
        self.lb_no_media = label("未关联图片 / 音频附件", "faint")
        self.lb_no_media.setAlignment(Qt.AlignCenter)
        self.preview_stack.addWidget(self.lb_no_media)

        self.lb_image = QLabel()
        self.lb_image.setAlignment(Qt.AlignCenter)
        self.lb_image.setMinimumHeight(150)
        self.preview_stack.addWidget(self.lb_image)

        audio_w = QWidget()
        av = QVBoxLayout(audio_w)
        av.setContentsMargins(0, 8, 0, 8)
        av.setSpacing(8)
        self.lb_audio = label("", "dim")
        self.lb_audio.setAlignment(Qt.AlignCenter)
        self.lb_audio.setWordWrap(True)
        av.addWidget(self.lb_audio)
        arow = QHBoxLayout()
        arow.addStretch(1)
        self.btn_play = button("播放", "", "play", self._toggle_play)
        arow.addWidget(self.btn_play)
        self.btn_open_ext = button("用系统播放器打开", "ghost", "", self._open_external)
        arow.addWidget(self.btn_open_ext)
        arow.addStretch(1)
        av.addLayout(arow)
        self.preview_stack.addWidget(audio_w)
        right.box.addWidget(self.preview_stack, 2)

        # 标注补充区
        right.box.addWidget(hline())
        self.lb_ann_title = label("标注补充区 —— 补充知识点关键词、题型描述、错因分析", "dim")
        right.box.addWidget(self.lb_ann_title)
        self.ed_ann = QPlainTextEdit()
        self.ed_ann.setMaximumHeight(84)
        self.ed_ann.setPlaceholderText(
            "例：这道图形题是斜面受力分析，涉及摩擦因数与正压力分解；错因是漏画了法向分量。")
        self.ed_ann.textChanged.connect(self._on_ann_changed)
        right.box.addWidget(self.ed_ann, 1)
        split.addWidget(right)

        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 5)
        self.root.addWidget(split)
        self._set_editing_enabled(False)
        self.refresh_list()

    # ------------------------------------------------------------------ #
    # 右侧快捷操作区
    # ------------------------------------------------------------------ #
    def quick_actions(self) -> Tuple[str, List[QWidget]]:
        return ("错题来源分散没关系：文本、表格、图片、音频都能直接进来，"
                "系统会自动识别学科与题型。", [
            button("单条录入", "primary", "add", self._add_single,
                   "手动新建一条错题"),
            button("一键导入 TXT", "", "text", lambda: self._import_files("txt"),
                   "支持空行分隔的多条错题，自动识别 GBK / UTF-8 编码"),
            button("一键导入 Excel", "", "import", lambda: self._import_files("excel"),
                   "自动匹配「题干 / 学科 / 题型 / 标注」等列名"),
            button("导入公式·图形图片", "", "image", lambda: self._import_files("image"),
                   "导入后选中题目，点击“图片文字：识别与校对”提取印刷文字"),
            button("导入听力音频", "", "audio", lambda: self._import_files("audio"),
                   "英语听力音频，由 Whisper 本地离线转写"),
            button("导入整个文件夹", "", "folder", self._import_folder,
                   "扫描目录下所有图片与音频并批量导入"),
            button("重试失败的导入", "", "import", self._retry_import,
                   "只重试上次失败或未处理的文件；默认跳过已成功导入的重复题目"),
            hline(),
            button("载入示例错题", "accent", "wand", self._load_samples,
                   "一键灌入 155 条演示错题，快速体验完整流程"),
            button("删除选中", "danger", "remove", self._delete_selected),
            button("清空全部", "danger", "trash", self._clear_all),
        ])

    # ------------------------------------------------------------------ #
    # 列表
    # ------------------------------------------------------------------ #
    def _filtered(self) -> List[Question]:
        s = self.cb_filter_subject.currentData()
        t = self.cb_filter_qtype.currentData()
        query = self.ed_search.text().strip().casefold()
        return [q for q in self.ctx.questions
                if (not s or q.subject == s) and (not t or q.qtype == t)
                and (not query or any(query in text.casefold() for text in (
                    q.raw_text or "", q.annotation or "",
                    (q.media_path or "").replace("\\", "/").rsplit("/", 1)[-1])))]

    def refresh_list(self) -> None:
        self._refresh_list(preserve_editor=False)

    @batch_view_updates("table")
    def _refresh_list(self, preserve_editor: bool) -> None:
        keep = self._current.id if self._current else ""
        items = self._filtered()
        self.table.blockSignals(True)
        self.table.setRowCount(len(items))
        for r, q in enumerate(items):
            it = QTableWidgetItem(q.display_text or "（空题干，仅附件）")
            it.setData(Qt.UserRole, q.id)
            it.setToolTip((q.raw_text or q.annotation or q.media_path)[:400])
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(q.subject))
            self.table.setItem(r, 2, QTableWidgetItem(
                config.QTYPE_LABELS.get(q.qtype, q.qtype)))
            mark = ""
            if q.media_path:
                mark = "图片" if Path(q.media_path).suffix.lower() in config.IMAGE_EXTS else "音频"
            self.table.setItem(r, 3, QTableWidgetItem(mark))
        self.table.blockSignals(False)

        total = len(self.ctx.questions)
        self.sec_list.set_hint(
            "共 {} 条{}".format(total, "（当前筛选 {} 条）".format(len(items))
                               if len(items) != total else "") if total else "尚未导入任何错题")
        if keep:
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).data(Qt.UserRole) == keep:
                    self.table.blockSignals(True)
                    self.table.selectRow(r)
                    self.table.blockSignals(False)
                    if not preserve_editor:
                        self._on_select()
                    return
            if preserve_editor:
                self.table.clearSelection()
                self.sec_detail.set_hint("内容已修改，待重新分类；当前题目不符合筛选条件，可继续编辑")
                return
        self._current = None
        self._set_editing_enabled(False)
        self._clear_detail()

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        qid = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        q = self.ctx.repo.question_by_id(qid)
        if q is None:
            return
        self._current = q
        self._set_editing_enabled(True)
        self._load_detail(q)

    # ------------------------------------------------------------------ #
    # 详情
    # ------------------------------------------------------------------ #
    def _set_editing_enabled(self, on: bool) -> None:
        for w in (self.ed_text, self.ed_ann, self.cb_subject, self.cb_qtype, self.btn_attach,
                  self.btn_image_ocr, self.btn_formula_ocr, self.btn_audio_transcribe):
            w.setEnabled(on)

    def _clear_detail(self) -> None:
        self._stop_play()
        for w, setter in ((self.ed_text, self.ed_text.setPlainText),
                          (self.ed_ann, self.ed_ann.setPlainText)):
            w.blockSignals(True)
            setter("")
            w.blockSignals(False)
        self.preview_stack.setCurrentIndex(0)
        self.btn_formula_ocr.setEnabled(False)
        self.btn_audio_transcribe.setEnabled(False)
        self.sec_detail.set_hint("从左侧选择一条错题，或点击右侧「单条录入」")

    def _load_detail(self, q: Question) -> None:
        for w in (self.ed_text, self.ed_ann, self.cb_subject, self.cb_qtype):
            w.blockSignals(True)
        self.ed_text.setPlainText(q.raw_text or "")
        self.ed_ann.setPlainText(q.annotation or "")
        self.cb_subject.setCurrentText(q.subject)
        idx = self.cb_qtype.findData(q.qtype)
        self.cb_qtype.setCurrentIndex(idx if idx >= 0 else 0)
        for w in (self.ed_text, self.ed_ann, self.cb_subject, self.cb_qtype):
            w.blockSignals(False)

        hint = "来源：{}".format({"manual": "手动录入", "txt": "TXT 导入", "excel": "Excel 导入",
                                 "image": "图片导入", "audio": "音频导入"}.get(q.source, q.source))
        if q.classified:
            hint += " ｜ 已分类：" + "、".join(
                "{}={}".format(config.DIM_LABELS.get(k, k), v)
                for k, v in list(q.label_names.items())[:2])
        if q.image_ocr:
            hint += " ｜ 图片识别：" + q.image_ocr_status
        if q.classified and (q.feature or {}).get("unreviewed_media"):
            hint += " ｜ 本次分类使用未校对的识别内容"
        self.sec_detail.set_hint(hint)
        self.btn_image_ocr.setEnabled(bool(q.media_path) and Path(q.media_path).suffix.lower() in config.IMAGE_EXTS)
        self.btn_formula_ocr.setEnabled(q.qtype == config.QTYPE_FORMULA and
                                        bool(q.media_path) and
                                        Path(q.media_path).suffix.lower() in config.IMAGE_EXTS)
        self.btn_audio_transcribe.setEnabled(bool(q.media_path) and
                                             Path(q.media_path).suffix.lower() in config.AUDIO_EXTS)
        for record, label in ((q.formula_ocr, "公式"), (q.audio_transcript, "听力转写")):
            if record is not None:
                state = "已确认" if record.confirmed else "待校对"
                self.sec_detail.set_hint(self.sec_detail.hint.text() + " ｜ " + label + state)
        self._show_media(q)
        self._update_annotation_hint(q)

    def _update_annotation_hint(self, q: Question) -> None:
        """按题型给出针对性的标注提示——图形题与听力题最依赖人工标注。"""
        base = "标注补充区 —— 补充知识点关键词、题型描述、错因分析"
        if q.qtype == config.QTYPE_FIGURE:
            base += "（图形题建议补充：图中元件、标注参数、考查知识点）"
        elif q.qtype == config.QTYPE_LISTENING:
            base += "（听力题建议补充：场景、关键信息、题型）"
        self.lb_ann_title.setText(base)

    def _show_media(self, q: Question) -> None:
        self._stop_play()
        self._audio_description = ""
        self.lb_audio.clear()
        if not q.media_path:
            self.lb_no_media.setText("未关联图片 / 音频附件")
            self.preview_stack.setCurrentIndex(0)
            return
        p = Path(q.media_path)
        if not p.exists():
            self.lb_no_media.setText("附件文件已丢失：{}".format(q.media_path))
            self.preview_stack.setCurrentIndex(0)
            return
        if p.suffix.lower() in config.IMAGE_EXTS:
            pm = QPixmap(str(p))
            if pm.isNull():
                self.lb_no_media.setText("图片无法解码：{}".format(p.name))
                self.preview_stack.setCurrentIndex(0)
                return
            self.lb_image.setPixmap(pm.scaled(
                QSize(max(self.lb_image.width(), 200), 240),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.preview_stack.setCurrentIndex(1)
        else:
            size_kb = p.stat().st_size / 1024
            self._audio_description = "{}\n{:.0f} KB".format(p.name, size_kb)
            self.lb_audio.setText(self._audio_description)
            self.btn_play.setEnabled(_HAS_MEDIA)
            self.btn_play.setToolTip("" if _HAS_MEDIA else
                                     "当前系统缺少音频后端，请用系统播放器打开")
            self.preview_stack.setCurrentIndex(2)

    # ------------------------------------------------------------------ #
    # 编辑回写
    # ------------------------------------------------------------------ #
    def _on_text_changed(self) -> None:
        if self._current is None:
            return
        self._current.raw_text = self.ed_text.toPlainText()
        self._touch()

    def _on_ann_changed(self) -> None:
        if self._current is None:
            return
        self._current.annotation = self.ed_ann.toPlainText()
        self._touch()

    def _on_meta_changed(self) -> None:
        if self._current is None:
            return
        self._current.subject = self.cb_subject.currentText()
        self._current.qtype = self.cb_qtype.currentData()
        self._update_annotation_hint(self._current)
        self._touch()
        self._refresh_list(preserve_editor=True)

    def _touch(self) -> None:
        if self._current is None:
            return
        self.ctx.notify_question_edited(self._current)
        self.sec_detail.set_hint("内容已修改，待重新分类")
        if self.ed_search.text().strip():
            self._refresh_list(preserve_editor=True)
            return
        row = self.table.currentRow()
        if row >= 0 and self.table.item(row, 0):
            self.table.item(row, 0).setText(self._current.display_text or "（空题干，仅附件）")

    # ------------------------------------------------------------------ #
    # 导入动作
    # ------------------------------------------------------------------ #
    def _add_single(self) -> None:
        q = importer.import_single("", self.cb_filter_subject.currentData() or self.cb_subject.currentText(),
                                   self.cb_filter_qtype.currentData() or self.cb_qtype.currentData())
        self.ed_search.clear()
        self.ctx.add_questions([q])
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole) == q.id:
                self.table.selectRow(r)
                break
        self.ed_text.setFocus()
        self.toast("已新建一条空白错题，请在中间区域录入题干", "ok")

    def _import_files(self, kind: str) -> None:
        filters = {
            "txt": "文本文件 (*.txt);;所有文件 (*)",
            "excel": "表格文件 (*.xlsx *.csv *.tsv);;所有文件 (*)",
            "image": "图片 ({});;所有文件 (*)".format(
                " ".join("*" + e for e in sorted(config.IMAGE_EXTS))),
            "audio": "音频 ({});;所有文件 (*)".format(
                " ".join("*" + e for e in sorted(config.AUDIO_EXTS))),
        }[kind]
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", str(Path.home()), filters)
        if not paths:
            return
        qtype = ""
        if kind == "image":
            qtype = ""       # 交给 guess_qtype 依据文件名与学科自动判别
        elif kind == "audio":
            qtype = config.QTYPE_LISTENING
        self._run_import("media" if kind in ("image", "audio") else kind, paths, qtype)

    def _import_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择包含图片 / 音频的文件夹",
                                             str(Path.home()))
        if not d:
            return
        imgs, auds = importer.scan_dir(d)
        if not imgs and not auds:
            self.toast("该文件夹下没有找到图片或音频文件", "warn")
            return
        self._run_import("media", imgs + auds, "")

    def _run_import(self, kind: str, paths: List[str], qtype: str) -> None:
        if not self.ctx.begin_task("导入"):
            return
        subj = self.cb_filter_subject.currentData() or ""
        self._import_options = (kind, subj, qtype)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self._worker = ImportWorker(kind, paths, subj, qtype, self)
        self._worker.progress.connect(
            lambda d, t, _p: (self.progress.setMaximum(t), self.progress.setValue(d)))
        self._worker.finished_all.connect(self._on_import_done)
        self._worker.finished.connect(lambda: self.ctx.end_task("导入"))
        self._worker.start()
        self.status("正在导入 {} 个文件…".format(len(paths)))

    def _on_import_done(self, items: List[Question], err: str) -> None:
        self.progress.setVisible(False)
        self._retry = None
        if self._worker is not None and self._worker.failed_paths:
            self._retry = (self._import_options, list(self._worker.failed_paths))
        try:
            count, skipped = self.ctx.add_imported_questions(items, self.cb_skip_duplicates.isChecked())
        except OSError as exc:
            if self._worker is not None:
                self._retry = (self._import_options, list(self._worker.paths))
            QMessageBox.warning(self, "导入未保存", "原有错题保持不变。请检查磁盘空间及权限后重试：\n{}".format(exc))
            self.toast("导入结果未保存，可修复磁盘问题后重试", "error")
            return
        summary = "导入结束：新增 {} 条，跳过重复 {} 条".format(count, skipped)
        if err:
            summary += "；部分文件或行未导入，可修正后重试"
            dialog = QMessageBox(QMessageBox.Warning, "导入结果", summary, QMessageBox.Ok, self)
            dialog.setDetailedText(err)
            dialog.exec_()
        self.toast(summary, "warn" if err else "ok")
        self.status(summary)

    def _retry_import(self) -> None:
        if not self._retry:
            self.toast("没有需要重试的导入文件", "info")
            return
        (kind, subject, qtype), paths = self._retry
        index = self.cb_filter_subject.findData(subject)
        self.cb_filter_subject.setCurrentIndex(max(index, 0))
        self._run_import(kind, paths, qtype)

    def _load_samples(self) -> None:
        xlsx = config.SAMPLE_DIR / "sample_questions.xlsx"
        txt = config.SAMPLE_DIR / "sample_questions.txt"
        src = xlsx if xlsx.exists() else txt
        if not src.exists():
            self.toast("未找到示例数据（samples/ 目录缺失）", "warn")
            return
        try:
            items = (importer.import_excel(str(src)) if src.suffix == ".xlsx"
                     else importer.import_txt(str(src)))
        except importer.ImportError_ as exc:
            self.toast("载入示例失败：{}".format(exc), "error")
            return
        count, skipped = self.ctx.add_imported_questions(items, self.cb_skip_duplicates.isChecked())
        self.toast("已载入 {} 条示例错题，跳过重复 {} 条".format(count, skipped), "ok")

    def _attach_media(self) -> None:
        if self._current is None:
            return
        exts = " ".join("*" + e for e in sorted(config.IMAGE_EXTS | config.AUDIO_EXTS))
        p, _ = QFileDialog.getOpenFileName(self, "选择图片或音频",
                                           str(Path.home()), "图片/音频 ({})".format(exts))
        if not p:
            return
        try:
            self._current.media_path = self.ctx.repo.import_media(p)
        except OSError as exc:
            self.toast(str(exc), "error")
            return
        self._current.image_ocr = None
        self._current.formula_ocr = None
        self._current.audio_transcript = None
        ext = Path(p).suffix.lower()
        if ext in config.AUDIO_EXTS:
            self._current.qtype = config.QTYPE_LISTENING
        elif not self._current.qtype or self._current.qtype == config.QTYPE_TEXT:
            from app.core.preprocess import cleaner
            self._current.qtype = cleaner.guess_qtype(
                self._current.raw_text, p, self._current.subject)
        self._touch()
        self._load_detail(self._current)
        self.refresh_list()
        self.toast("附件已关联并复制到工作区", "ok")

    def _recognize_image(self) -> None:
        if self._current is None or not self._current.media_path:
            self.toast("请先选择一条带图片附件的错题", "warn")
            return
        if self.ctx.active_task:
            self.toast("请等待当前任务结束", "warn")
            return
        from app.ui.image_ocr_dialog import ImageOcrDialog
        ImageOcrDialog(self.ctx, self._current, self).exec_()
        self.refresh_list()

    def _recognize_formula(self) -> None:
        if self._current is None or self._current.qtype != config.QTYPE_FORMULA:
            self.toast("请先选择公式类错题", "warn")
            return
        if not self._current.media_path or Path(self._current.media_path).suffix.lower() not in config.IMAGE_EXTS:
            self.toast("请先关联公式图片附件", "warn")
            return
        if self.ctx.active_task:
            self.toast("请等待当前任务结束", "warn")
            return
        from app.ui.media_review_dialog import MediaReviewDialog
        MediaReviewDialog(self.ctx, self._current, "formula", self).exec_()
        self._load_detail(self._current)
        self.refresh_list()

    def _transcribe_audio(self) -> None:
        if self._current is None:
            self.toast("请先选择带音频附件的错题", "warn")
            return
        if not self._current.media_path or Path(self._current.media_path).suffix.lower() not in config.AUDIO_EXTS:
            self.toast("请先关联听力音频附件", "warn")
            return
        if self.ctx.active_task:
            self.toast("请等待当前任务结束", "warn")
            return
        from app.ui.media_review_dialog import MediaReviewDialog
        MediaReviewDialog(self.ctx, self._current, "audio", self).exec_()
        self._load_detail(self._current)
        self.refresh_list()

    def _delete_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.toast("请先在列表中选择要删除的错题", "warn")
            return
        ids = [self.table.item(r.row(), 0).data(Qt.UserRole) for r in rows]
        self.ctx.remove_questions(ids)
        self.toast("已删除 {} 条错题".format(len(ids)), "ok")

    def _clear_all(self) -> None:
        if not self.ctx.questions:
            return
        if QMessageBox.question(
                self, "确认清空",
                "将删除全部 {} 条错题（分类体系与已学到的反馈不受影响），确定吗？".format(
                    len(self.ctx.questions)),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.ctx.clear_questions()
        self.toast("已清空全部错题", "ok")

    # ------------------------------------------------------------------ #
    # 音频播放
    # ------------------------------------------------------------------ #
    def _toggle_play(self) -> None:
        if not _HAS_MEDIA or self._current is None or not self._current.media_path:
            return
        if self._player is None:
            self._player = QMediaPlayer(self)
            self._player.stateChanged.connect(self._on_state)
            self._player.error.connect(self._on_audio_error)
        if self._player.state() == QMediaPlayer.PlayingState:
            self._player.stop()
            return
        self._audio_active_path = self._current.media_path
        self._audio_error_reported = False
        self.lb_audio.setText(self._audio_description)
        self._player.setMedia(QMediaContent(QUrl.fromLocalFile(self._current.media_path)))
        if not self._audio_error_reported:
            self._player.play()

    def _on_audio_error(self, error) -> None:
        if (not error or self._audio_error_reported or self._current is None or
                self._audio_active_path != self._current.media_path):
            return
        if self.sender() is not None and self.sender() is not self._player:
            return
        self._audio_error_reported = True
        reason = self._player.errorString().strip() or "音频格式不受支持，或系统缺少音频播放后端"
        self._player.stop()
        self.lb_audio.setText("{}\n播放失败：{}\n请点击“用系统播放器打开”试听。".format(
            self._audio_description, reason))
        self.toast("音频播放失败：{}；可使用系统播放器打开".format(reason), "warn")

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlayingState
        self.btn_play.setText("停止" if playing else "播放")
        self.btn_play.setIcon(icons.icon("stop" if playing else "play", 16))

    def _stop_play(self) -> None:
        self._audio_active_path = ""
        self._audio_error_reported = False
        if self._player is not None:
            self._player.stop()
            self._player.error.disconnect(self._on_audio_error)
            self._player.stateChanged.disconnect(self._on_state)
            self._player.deleteLater()
            self._player = None

    def _open_external(self) -> None:
        if self._current is None or not self._current.media_path:
            return
        from PyQt5.QtGui import QDesktopServices
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(self._current.media_path)):
            self.lb_audio.setText("{}\n系统播放器未能打开附件，请检查默认播放器设置或更换音频格式。".format(
                self._audio_description))
            self.toast("系统播放器未能打开音频，请检查默认播放器设置或更换格式", "warn")

    def workers(self):
        return [self._worker]

    def shutdown(self) -> None:
        self._stop_play()
        super().shutdown()

    def on_show(self) -> None:
        self.refresh_list()
