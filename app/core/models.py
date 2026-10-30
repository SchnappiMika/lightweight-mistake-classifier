# -*- coding: utf-8 -*-
"""核心数据模型。

所有模型均可 ``to_dict`` / ``from_dict`` 与 JSON 互转，配合 storage 层实现
本地离线持久化（不引入数据库依赖，保证轻量化与免配置）。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app import config


def new_id(prefix: str = "id") -> str:
    """生成短唯一 ID。"""
    return "{}_{}".format(prefix, uuid.uuid4().hex[:10])


def now_ts() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# 特征包：多模态预处理层的统一输出契约
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBundle:
    """多模态错题预处理后的标准化特征包。

    不同模态（文本 / 公式 / 图形 / 音频）的预处理单元最终都收敛到本结构，
    从而让下游分级分类层与模态解耦——这是"可插拔预处理单元"架构的关键。
    """

    question_id: str = ""
    subject: str = config.SUBJECT_GENERAL
    qtype: str = config.QTYPE_TEXT

    clean_text: str = ""            # 清洗后的题干正文
    text_tokens: List[str] = field(default_factory=list)      # 核心实词特征
    formula_tokens: List[str] = field(default_factory=list)   # 公式映射出的知识点特征词
    qtype_tokens: List[str] = field(default_factory=list)     # 题型特征词
    subject_tokens: List[str] = field(default_factory=list)   # 学科特征词

    latex: str = ""                 # LaTeX-OCR 转写结果
    ocr_text: str = ""              # 图形内文本 OCR 结果
    transcript: str = ""            # 听力音频转写结果
    figure_parameters: List[Dict[str, Any]] = field(default_factory=list)
    unreviewed_media: List[str] = field(default_factory=list)
    media_elapsed_ms: float = 0.0
    total_process_ms: float = 0.0

    pipeline: List[str] = field(default_factory=list)   # 实际执行过的预处理单元名
    notes: List[str] = field(default_factory=list)      # 降级 / 告警信息（如引擎缺失）
    elapsed_ms: float = 0.0

    def all_tokens(self) -> List[str]:
        return list(self.text_tokens) + list(self.formula_tokens)

    def is_empty(self) -> bool:
        return not (self.text_tokens or self.formula_tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "subject": self.subject,
            "qtype": self.qtype,
            "clean_text": self.clean_text,
            "text_tokens": self.text_tokens,
            "formula_tokens": self.formula_tokens,
            "qtype_tokens": self.qtype_tokens,
            "subject_tokens": self.subject_tokens,
            "latex": self.latex,
            "ocr_text": self.ocr_text,
            "transcript": self.transcript,
            "figure_parameters": self.figure_parameters,
            "unreviewed_media": self.unreviewed_media,
            "media_elapsed_ms": self.media_elapsed_ms,
            "total_process_ms": self.total_process_ms,
            "pipeline": self.pipeline,
            "notes": self.notes,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureBundle":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# 错题
# --------------------------------------------------------------------------- #
@dataclass
class ImageOcrRecord:
    text: str = ""
    draft: str = ""
    lines: List[Dict[str, Any]] = field(default_factory=list)
    media_hash: str = ""
    engine: str = "RapidOCR / PP-OCRv6"
    elapsed_ms: float = 0.0
    confirmed: bool = False
    applied_text: str = ""
    created_at: float = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageOcrRecord":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class MediaReviewRecord:
    text: str = ""
    draft: str = ""
    media_hash: str = ""
    engine: str = ""
    elapsed_ms: float = 0.0
    duration_seconds: float = 0.0
    confirmed: bool = False
    created_at: float = field(default_factory=now_ts)
    error: str = ""
    applied_text: str = ""
    recognition_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaReviewRecord":
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class Question:
    """一条错题记录。"""

    id: str = field(default_factory=lambda: new_id("q"))
    subject: str = config.SUBJECT_GENERAL
    qtype: str = config.QTYPE_TEXT
    raw_text: str = ""              # 原始题干（用户录入 / 文件导入）
    media_path: str = ""            # 图片 / 音频本地路径
    annotation: str = ""            # 用户对图形题、音频题补充的标注信息
    source: str = "manual"          # manual / txt / excel / image / audio
    image_ocr: Optional[ImageOcrRecord] = None
    formula_ocr: Optional[MediaReviewRecord] = None
    audio_transcript: Optional[MediaReviewRecord] = None

    labels: Dict[str, str] = field(default_factory=dict)       # dim -> label_id
    label_names: Dict[str, str] = field(default_factory=dict)  # dim -> label_name（冗余，便于导出）
    scores: Dict[str, float] = field(default_factory=dict)     # dim -> 置信度
    corrected: Dict[str, bool] = field(default_factory=dict)   # dim -> 是否被人工修正过
    candidates: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # dim -> 粗筛候选明细

    priority: str = ""              # 复习优先级人工标记（高/中/低）
    scheme_id: str = ""             # 分类时所用的分类体系
    feature: Optional[Dict[str, Any]] = None   # 缓存的 FeatureBundle
    classified: bool = False
    elapsed_ms: float = 0.0
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)

    # ---- 便捷属性 ---- #
    @property
    def image_ocr_status(self) -> str:
        if self.image_ocr is None:
            return ""
        if not self.image_ocr.confirmed:
            return "待校对应用"
        if self.image_ocr.draft.strip() != self.raw_text.strip():
            return "题干或草稿已修改，待重新应用"
        return "已校对应用"

    @property
    def display_text(self) -> str:
        """列表中展示用的简短摘要。"""
        base = (self.raw_text or self.annotation or "").strip().replace("\n", " ")
        if not base and self.media_path:
            base = "[{}] {}".format(config.QTYPE_LABELS.get(self.qtype, self.qtype),
                                    self.media_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
        return base[:60] + ("…" if len(base) > 60 else "")

    def feature_bundle(self) -> Optional[FeatureBundle]:
        return FeatureBundle.from_dict(self.feature) if self.feature else None

    def invalidate_classification(self) -> None:
        self.labels = {}
        self.label_names = {}
        self.scores = {}
        self.corrected = {}
        self.candidates = {}
        self.scheme_id = ""
        self.feature = None
        self.classified = False
        self.elapsed_ms = 0.0
        self.updated_at = now_ts()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "subject": self.subject, "qtype": self.qtype,
            "raw_text": self.raw_text, "media_path": self.media_path,
            "annotation": self.annotation, "source": self.source,
            "image_ocr": self.image_ocr.to_dict() if self.image_ocr else None,
            "formula_ocr": self.formula_ocr.to_dict() if self.formula_ocr else None,
            "audio_transcript": self.audio_transcript.to_dict() if self.audio_transcript else None,
            "labels": self.labels, "label_names": self.label_names,
            "scores": self.scores, "corrected": self.corrected,
            "candidates": self.candidates, "priority": self.priority,
            "scheme_id": self.scheme_id, "feature": self.feature,
            "classified": self.classified, "elapsed_ms": self.elapsed_ms,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Question":
        values = {key: value for key, value in d.items() if key in cls.__dataclass_fields__}
        if values.get("image_ocr") is not None:
            values["image_ocr"] = ImageOcrRecord.from_dict(values["image_ocr"])
        if values.get("formula_ocr") is not None:
            values["formula_ocr"] = MediaReviewRecord.from_dict(values["formula_ocr"])
        if values.get("audio_transcript") is not None:
            values["audio_transcript"] = MediaReviewRecord.from_dict(values["audio_transcript"])
        return cls(**values)


# --------------------------------------------------------------------------- #
# 分类标签 / 维度 / 体系
# --------------------------------------------------------------------------- #
@dataclass
class CategoryLabel:
    """一个分类标签（如"定积分及其应用"）。

    ``keywords`` 用于第一层关键词粗筛；``samples`` 为标准错题样本，
    与 keywords 一起构成第二层精匹配的标签画像；``weight_override``
    允许用户对该标签单独指定各特征通道权重（零代码可视化配置）。
    """

    id: str = field(default_factory=lambda: new_id("lb"))
    name: str = ""
    dim: str = config.DIM_KNOWLEDGE
    keywords: List[str] = field(default_factory=list)
    keyword_weights: Dict[str, float] = field(default_factory=dict)  # 关键词 -> 权重（默认 1.0）
    samples: List[str] = field(default_factory=list)                 # 标准错题样本
    weight_override: Dict[str, float] = field(default_factory=dict)  # channel -> weight
    prior: float = 1.0              # 先验强度，粗筛召回不足时按此排序补齐
    note: str = ""

    def kw_weight(self, kw: str) -> float:
        return float(self.keyword_weights.get(kw, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "dim": self.dim,
            "keywords": self.keywords, "keyword_weights": self.keyword_weights,
            "samples": self.samples, "weight_override": self.weight_override,
            "prior": self.prior, "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CategoryLabel":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Dimension:
    """一个分类维度（知识点 / 错误类型 / 难度 / 复习优先级 / 用户自定义）。"""

    key: str = config.DIM_KNOWLEDGE
    name: str = "知识点"
    enabled: bool = True
    labels: List[CategoryLabel] = field(default_factory=list)

    def label_by_id(self, lid: str) -> Optional[CategoryLabel]:
        for lb in self.labels:
            if lb.id == lid:
                return lb
        return None

    def label_by_name(self, name: str) -> Optional[CategoryLabel]:
        for lb in self.labels:
            if lb.name == name:
                return lb
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name, "enabled": self.enabled,
                "labels": [x.to_dict() for x in self.labels]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dimension":
        return cls(key=d.get("key", config.DIM_KNOWLEDGE),
                   name=d.get("name", ""),
                   enabled=bool(d.get("enabled", True)),
                   labels=[CategoryLabel.from_dict(x) for x in d.get("labels", [])])


@dataclass
class Scheme:
    """一套分类体系（通常对应一门课程）。

    用户可零代码新建 / 修改；系统在保存后自动重建特征库与匹配规则。
    """

    id: str = field(default_factory=lambda: new_id("sc"))
    name: str = ""
    subject: str = config.SUBJECT_GENERAL
    description: str = ""
    dims: List[Dimension] = field(default_factory=list)
    builtin: bool = False
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    revision: int = 0               # 每次编辑自增，用于触发特征库热更新

    def dim(self, key: str) -> Optional[Dimension]:
        for d in self.dims:
            if d.key == key:
                return d
        return None

    def enabled_dims(self) -> List[Dimension]:
        return [d for d in self.dims if d.enabled and d.labels]

    def find_label(self, lid: str):
        for d in self.dims:
            lb = d.label_by_id(lid)
            if lb:
                return d, lb
        return None, None

    def touch(self) -> None:
        self.updated_at = now_ts()
        self.revision += 1

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "subject": self.subject,
                "description": self.description, "builtin": self.builtin,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "revision": self.revision,
                "dims": [d.to_dict() for d in self.dims]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scheme":
        return cls(id=d.get("id") or new_id("sc"),
                   name=d.get("name", ""),
                   subject=d.get("subject", config.SUBJECT_GENERAL),
                   description=d.get("description", ""),
                   builtin=bool(d.get("builtin", False)),
                   created_at=d.get("created_at", now_ts()),
                   updated_at=d.get("updated_at", now_ts()),
                   revision=int(d.get("revision", 0)),
                   dims=[Dimension.from_dict(x) for x in d.get("dims", [])])


# --------------------------------------------------------------------------- #
# 分类结果
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """粗筛 / 精匹配候选项，携带各通道明细以便界面解释分类依据。"""

    label_id: str
    label_name: str
    coarse_score: float = 0.0
    hit_keywords: List[str] = field(default_factory=list)
    channel_scores: Dict[str, float] = field(default_factory=dict)
    channel_weights: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    fallback: bool = False          # 是否为召回兜底补入

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_id": self.label_id, "label_name": self.label_name,
            "coarse_score": round(self.coarse_score, 4),
            "hit_keywords": self.hit_keywords,
            "channel_scores": {k: round(v, 4) for k, v in self.channel_scores.items()},
            "channel_weights": {k: round(v, 4) for k, v in self.channel_weights.items()},
            "final_score": round(self.final_score, 4),
            "fallback": self.fallback,
        }


@dataclass
class DimResult:
    """单个维度的分类结果。"""

    dim: str
    label_id: str = ""
    label_name: str = ""
    score: float = 0.0
    candidates: List[Candidate] = field(default_factory=list)
    coarse_ms: float = 0.0
    fine_ms: float = 0.0
    scanned: int = 0                # 精匹配实际参与计算的标签数
    total: int = 0                  # 该维度标签总数（用于展示粗筛节省比例）

    @property
    def low_confidence(self) -> bool:
        return self.score < config.LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.dim, "label_id": self.label_id, "label_name": self.label_name,
                "score": round(self.score, 4),
                "candidates": [c.to_dict() for c in self.candidates],
                "coarse_ms": round(self.coarse_ms, 3), "fine_ms": round(self.fine_ms, 3),
                "scanned": self.scanned, "total": self.total}


@dataclass
class ClassifyResult:
    """一条错题在全部维度上的分类结果。"""

    question_id: str
    dims: Dict[str, DimResult] = field(default_factory=dict)
    preprocess_ms: float = 0.0
    classify_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.classify_ms


@dataclass
class FeedbackRecord:
    """一条人工修正反馈，是增量优化的输入。"""

    id: str = field(default_factory=lambda: new_id("fb"))
    question_id: str = ""
    scheme_id: str = ""
    dim: str = ""
    from_label_id: str = ""
    from_label_name: str = ""
    to_label_id: str = ""
    to_label_name: str = ""
    subject: str = ""
    qtype: str = ""
    tokens: List[str] = field(default_factory=list)
    weight_before: Dict[str, float] = field(default_factory=dict)
    weight_after: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "question_id": self.question_id, "scheme_id": self.scheme_id,
            "dim": self.dim, "from_label_id": self.from_label_id,
            "from_label_name": self.from_label_name, "to_label_id": self.to_label_id,
            "to_label_name": self.to_label_name, "subject": self.subject, "qtype": self.qtype,
            "tokens": self.tokens, "weight_before": self.weight_before,
            "weight_after": self.weight_after, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeedbackRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
