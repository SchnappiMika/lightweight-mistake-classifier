"""公式原图缩放与原始像素坐标选区，不写入附件。"""
from __future__ import annotations

import math

from PyQt5.QtCore import QEvent, QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QScrollArea,
                             QVBoxLayout, QWidget)

from app.ui.widgets import button


class FormulaSelectionCanvas(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(path)
        self.scale = 1.0
        self.crop_box = None
        self._anchor = None
        self._drag_point = None
        self.setCursor(Qt.CrossCursor)
        self.set_scale(1.0)

    def set_scale(self, scale):
        self.scale = max(0.05, min(4.0, float(scale)))
        if self.pixmap.isNull():
            self.setFixedSize(300, 150)
        else:
            self.setFixedSize(max(1, round(self.pixmap.width() * self.scale)),
                              max(1, round(self.pixmap.height() * self.scale)))
        self.update()

    def clear_selection(self):
        self.crop_box = None
        self._anchor = None
        self._drag_point = None
        self.selection_changed.emit()
        self.update()

    def _clamped(self, point):
        return QPoint(max(0, min(point.x(), self.width())),
                      max(0, min(point.y(), self.height())))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.pixmap.isNull():
            self._anchor = self._clamped(event.pos())
            self._drag_point = self._anchor
            self.update()

    def mouseMoveEvent(self, event):
        if self._anchor is not None and event.buttons() & Qt.LeftButton:
            self._drag_point = self._clamped(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._anchor is None:
            return
        endpoint = self._clamped(event.pos())
        left = min(self._anchor.x(), endpoint.x())
        top = min(self._anchor.y(), endpoint.y())
        right = max(self._anchor.x(), endpoint.x())
        bottom = max(self._anchor.y(), endpoint.y())
        if right - left >= 4 and bottom - top >= 4:
            self.crop_box = [max(0, math.floor(left / self.scale)),
                             max(0, math.floor(top / self.scale)),
                             min(self.pixmap.width(), math.ceil(right / self.scale)),
                             min(self.pixmap.height(), math.ceil(bottom / self.scale))]
        self._anchor = None
        self._drag_point = None
        self.selection_changed.emit()
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        if self.pixmap.isNull():
            painter.setPen(QColor("#555555"))
            painter.drawText(self.rect(), Qt.AlignCenter, "图片无法预览，请检查附件。")
            return
        painter.drawPixmap(self.rect(), self.pixmap)
        rect = None
        if self._anchor is not None and self._drag_point is not None:
            rect = QRect(self._anchor, self._drag_point).normalized()
        elif self.crop_box:
            left, top, right, bottom = self.crop_box
            rect = QRect(round(left * self.scale), round(top * self.scale),
                         round((right - left) * self.scale), round((bottom - top) * self.scale))
        if rect is not None:
            painter.fillRect(rect, QColor(40, 130, 235, 35))
            painter.setPen(QPen(QColor("#1681E8"), 2, Qt.DashLine))
            painter.drawRect(rect)


class FormulaCropPanel(QWidget):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("原图缩放"))
        self.zoom = QComboBox()
        self.zoom.addItem("适应宽度", 0)
        for percent in (25, 50, 75, 100, 150, 200):
            self.zoom.addItem("{}%".format(percent), percent / 100)
        controls.addWidget(self.zoom)
        self.reset = button("重置选区", "", "", self._reset)
        controls.addWidget(self.reset)
        layout.addLayout(controls)
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.canvas = FormulaSelectionCanvas(path)
        self.scroll.setWidget(self.canvas)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)
        self.selection_hint = QLabel()
        self.selection_hint.setWordWrap(True)
        layout.addWidget(self.selection_hint)
        self.invert = QCheckBox("深色背景转白底（反色）")
        self.invert.setToolTip("仅对本次送识别的副本反色，原图不变。用于深底浅色公式。")
        layout.addWidget(self.invert)
        row = QHBoxLayout()
        row.addWidget(QLabel("尺寸处理"))
        self.resize_mode = QComboBox()
        self.resize_mode.addItem("自动适配识别尺寸", "fit")
        self.resize_mode.addItem("保留裁剪后的原始尺寸", "original")
        self.resize_mode.setToolTip("自动适配会保持长宽比调整临时图片，建议先框选单条公式。")
        row.addWidget(self.resize_mode)
        layout.addLayout(row)
        self.zoom.currentIndexChanged.connect(self._apply_zoom)
        self.canvas.selection_changed.connect(self._update_hint)
        self._update_hint()
        self._apply_zoom()

    def _reset(self):
        self.canvas.clear_selection()

    def _update_hint(self):
        box = self.canvas.crop_box
        if box:
            self.selection_hint.setText("已选原图区域：({}, {})—({}, {})，{}×{}像素。".format(
                *box, box[2] - box[0], box[3] - box[1]))
        else:
            self.selection_hint.setText("拖动框选单条公式；未选区时使用整图。原附件保持不变。")

    def _apply_zoom(self):
        scale = self.zoom.currentData()
        if not scale:
            width = max(100, self.scroll.viewport().width() - 4)
            scale = min(1.0, width / max(1, self.canvas.pixmap.width()))
        self.canvas.set_scale(scale)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "scroll") and not self.zoom.currentData():
            self._apply_zoom()

    def eventFilter(self, watched, event):
        if watched is self.scroll.viewport() and event.type() == QEvent.Resize:
            if not self.zoom.currentData():
                self._apply_zoom()
        return super().eventFilter(watched, event)

    def options(self):
        return {"crop_box": list(self.canvas.crop_box) if self.canvas.crop_box else None,
                "invert": self.invert.isChecked(),
                "resize_mode": self.resize_mode.currentData()}

    def set_options(self, options):
        """Restore the recognition region when reopening a saved formula draft."""
        from app.core.preprocess.formula_recognition_backend import normalize_options
        try:
            normalized = normalize_options(options)
            box = normalized["crop_box"]
            if box and (box[2] > self.canvas.pixmap.width() or box[3] > self.canvas.pixmap.height()):
                return
        except (ValueError, TypeError):
            return
        self.canvas.crop_box = box
        self.invert.setChecked(normalized["invert"])
        self.resize_mode.setCurrentIndex(self.resize_mode.findData(normalized["resize_mode"]))
        self._update_hint()
        self.canvas.update()
