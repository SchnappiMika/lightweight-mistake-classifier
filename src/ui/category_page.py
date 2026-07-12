# -*- coding: utf-8 -*-
"""分类体系管理页面"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QTextEdit, QLineEdit, QComboBox,
    QSplitter, QGroupBox, QMessageBox, QFrame, QFileDialog)
from PyQt5.QtCore import Qt
import os, json


class CategoryPage(QWidget):

    def __init__(self, dict_dir=None):
        super().__init__()
        if dict_dir is None:
            dict_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_base", "data", "dictionaries")
        self.dict_dir = dict_dir
        self.current_data = None
        self.current_cat_idx = -1
        self._setup_ui()
        if os.path.exists(dict_dir):
            self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        lbl = QLabel("分类体系管理")
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(lbl)

        top = QHBoxLayout()
        top.addWidget(QLabel("学科:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentTextChanged.connect(self._on_subject_changed)
        top.addWidget(self.subject_combo)
        top.addStretch()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._load_data)
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        left = QFrame()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("分类列表:"))
        self.cat_list = QListWidget()
        self.cat_list.currentRowChanged.connect(self._on_category_selected)
        ll.addWidget(self.cat_list)
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+")
        self.add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(self.add_btn)
        self.del_btn = QPushButton("-")
        self.del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.del_btn)
        ll.addLayout(btn_row)
        splitter.addWidget(left)

        right = QFrame()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("分类名称:"))
        self.name_edit = QLineEdit()
        rl.addWidget(self.name_edit)
        rl.addWidget(QLabel("关键词 (英文逗号分隔):"))
        self.kw_edit = QTextEdit()
        self.kw_edit.setMaximumHeight(100)
        rl.addWidget(self.kw_edit)
        rl.addWidget(QLabel("样本题目:"))
        self.sample_edit = QTextEdit()
        self.sample_edit.setMaximumHeight(80)
        rl.addWidget(self.sample_edit)
        self.save_btn = QPushButton("保存修改")
        self.save_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.save_btn.clicked.connect(self._on_save)
        rl.addWidget(self.save_btn)
        rl.addStretch()
        splitter.addWidget(right)
        layout.addWidget(splitter)

    def _load_data(self):
        """加载所有学科词典"""
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        if os.path.exists(self.dict_dir):
            for fn in sorted(os.listdir(self.dict_dir)):
                if fn.endswith(".json"):
                    fpath = os.path.join(self.dict_dir, fn)
                    with open(fpath, encoding="utf-8") as f:
                        d = json.load(f)
                        self.subject_combo.addItem(d["subject"] + "|" + fn)
        self.subject_combo.blockSignals(False)
        if self.subject_combo.count() > 0:
            self.subject_combo.setCurrentIndex(0)

    def _on_subject_changed(self, subject):
        """学科选择变化时加载对应分类"""
        if not subject or "|" not in subject:
            return
        sub_name, fn = subject.split("|", 1)
        fpath = os.path.join(self.dict_dir, fn)
        if not os.path.exists(fpath):
            self.cat_list.clear()
            self.current_data = None
            return
        with open(fpath, encoding="utf-8") as f:
            self.current_data = json.load(f)
        self.cat_list.blockSignals(True)
        self.cat_list.clear()
        for c in self.current_data["categories"]:
            self.cat_list.addItem(c["name"])
        self.cat_list.blockSignals(False)
        self.current_cat_idx = -1
        self.name_edit.clear()
        self.kw_edit.clear()
        self.sample_edit.clear()

    def _on_category_selected(self, idx):
        """选中分类时加载详情"""
        if idx < 0 or not self.current_data:
            return
        cats = self.current_data["categories"]
        if idx >= len(cats):
            return
        cat = cats[idx]
        self.name_edit.setText(cat.get("name", ""))
        self.kw_edit.setText(", ".join(cat.get("keywords", [])))
        self.sample_edit.setText(cat.get("sample", ""))
        self.current_cat_idx = idx

    def _on_add(self):
        """添加新分类"""
        if not self.current_data:
            QMessageBox.warning(self, "提示", "请先选择一个学科")
            return
        self.current_data["categories"].append({
            "name": "新分类",
            "keywords": [],
            "weight": 1.0,
            "sample": ""
        })
        self.cat_list.blockSignals(True)
        self.cat_list.addItem("新分类")
        self.cat_list.blockSignals(False)
        self.cat_list.setCurrentRow(self.cat_list.count() - 1)

    def _on_delete(self):
        """删除选中分类"""
        if self.current_cat_idx < 0 or not self.current_data:
            return
        reply = QMessageBox.question(self, "????", "??????????????????", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        cats = self.current_data["categories"]
        if self.current_cat_idx >= len(cats):
            return
        del cats[self.current_cat_idx]
        self.cat_list.blockSignals(True)
        self.cat_list.takeItem(self.current_cat_idx)
        self.cat_list.blockSignals(False)
        self.current_cat_idx = -1
        self.name_edit.clear()
        self.kw_edit.clear()
        self.sample_edit.clear()

    def _on_save(self):
        """保存当前编辑的分类"""
        if self.current_cat_idx < 0 or not self.current_data:
            return
        reply = QMessageBox.question(self, "????", "??????????????????", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        cats = self.current_data["categories"]
        if self.current_cat_idx >= len(cats):
            return
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "分类名称不能为空")
            return
        kw_text = self.kw_edit.toPlainText().strip()
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        sample = self.sample_edit.toPlainText().strip()
        existing = cats[self.current_cat_idx]
        cats[self.current_cat_idx] = {
            "name": name,
            "keywords": keywords,
            "weight": existing.get("weight", 1.0),
            "sample": sample
        }
        cur = self.subject_combo.currentText()
        fn = cur.split("|", 1)[1] if "|" in cur else ""
        if fn:
            with open(os.path.join(self.dict_dir, fn), "w", encoding="utf-8") as f:
                json.dump(self.current_data, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "成功", "分类已保存")
