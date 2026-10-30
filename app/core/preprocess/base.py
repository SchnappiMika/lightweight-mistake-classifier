# -*- coding: utf-8 -*-
"""可插拔预处理单元的基类与注册表。

架构创新点落地处：传统错题工具是"单一纯文本预处理流水线"，本层把每种模态
做成独立可插拔单元，通过 ``@register`` 注册进注册表，pipeline 按题型分流调度。
新增一种模态（例如手写体识别、视频题）只需新增一个子类，无需改动分类层。
"""
from __future__ import annotations

import abc
import importlib.util
import logging
from typing import Dict, List, Optional, Type

from app.core.models import FeatureBundle, Question

log = logging.getLogger(__name__)

_REGISTRY: Dict[str, "BasePreprocessor"] = {}
_ORDER: List[str] = []


class EngineStatus:
    """外部识别引擎（pix2tex / PaddleOCR / Whisper）的可用性描述。"""

    def __init__(self, name: str, available: bool, detail: str = "",
                 state: str = "", action: str = "") -> None:
        self.name = name
        self.available = available
        self.detail = detail
        self.state = state or ("ready" if available else "unchecked")
        self.action = action

    @property
    def state_label(self) -> str:
        return {"missing": "组件未安装", "unchecked": "组件已安装，模型/语料未验证",
                "loading": "模型正在加载", "ready": "模型/语料已加载",
                "verified": "本地文字识别已验证", "model_missing": "本地模型未准备",
                "failed": "初始化失败", "check_failed": "状态检查失败"}.get(self.state, "状态未知")

    def __repr__(self) -> str:  # pragma: no cover
        return "<EngineStatus {} available={}>".format(self.name, self.available)


def passive_engine_status(name, module_groups, *, checked=False, loaded=False,
                          loading=False, error="", fallback="", ready_detail=""):
    if loading:
        return EngineStatus(name, False, "识别任务正在加载模型；自检不会重复初始化。",
                            "loading", "等待当前任务结束后刷新状态。")
    if loaded:
        return EngineStatus(name, True, ready_detail or "本次运行中模型已成功加载。",
                            "ready", "请用真实文件验证识别效果；加载成功不代表已通过完整离线验收。")
    try:
        installed = any(all(importlib.util.find_spec(module) is not None for module in group)
                        for group in module_groups)
    except Exception as exc:
        return EngineStatus(name, False, "无法读取组件安装信息：{}".format(exc),
                            "check_failed", "检查项目运行环境及错误信息后重试。" + fallback)
    if not installed:
        components = " 或 ".join(" + ".join(group) for group in module_groups)
        detail = "缺少可用的组件组合：{}。".format(components)
        if error:
            detail += "最近错误：{}".format(error)
        return EngineStatus(name, False, detail, "missing",
                            "需在项目运行环境准备识别组件和模型；当前自检不会自动安装或下载。" + fallback)
    if checked:
        return EngineStatus(name, False, "最近初始化未成功：{}".format(error or "未创建模型实例"),
                            "failed", "修复组件或模型问题后重启软件，再用真实文件重试。" + fallback)
    return EngineStatus(name, False, "检测到组件，但尚未验证模型文件、依赖兼容性或识别能力。",
                        "unchecked", "需要准备模型并实际执行识别；本次自检不加载模型，也不触发模型下载。" + fallback)


class BasePreprocessor(abc.ABC):
    """预处理单元契约。

    每个单元只负责往 ``FeatureBundle`` 里补充自己那一份特征，
    由 pipeline 负责组合，单元之间互不感知（松耦合）。
    """

    name: str = "base"
    display_name: str = "基础单元"
    #: 该单元处理的题型；空集合表示对所有题型都执行（如通用文本单元）
    qtypes: frozenset = frozenset()

    def accept(self, q: Question) -> bool:
        return (not self.qtypes) or (q.qtype in self.qtypes)

    @abc.abstractmethod
    def process(self, q: Question, fb: FeatureBundle) -> None:
        """就地填充 ``fb``。实现必须保证：引擎缺失时降级而非抛异常。"""

    def status(self) -> Optional[EngineStatus]:
        """返回外部引擎可用性；纯 Python 单元返回 None。"""
        return None


def register(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
    """装饰器：把预处理单元注册进全局注册表。"""
    inst = cls()
    if inst.name in _REGISTRY:
        log.warning("预处理单元重复注册：%s", inst.name)
    _REGISTRY[inst.name] = inst
    _ORDER.append(inst.name)
    return cls


def get(name: str) -> Optional[BasePreprocessor]:
    return _REGISTRY.get(name)


def all_units() -> List[BasePreprocessor]:
    """按注册顺序返回全部单元。"""
    return [_REGISTRY[n] for n in _ORDER]


def units_for(q: Question) -> List[BasePreprocessor]:
    return [u for u in all_units() if u.accept(q)]


def engine_report() -> List[EngineStatus]:
    """汇总所有外部引擎状态，供界面【环境自检】展示。"""
    out: List[EngineStatus] = []
    for u in all_units():
        try:
            st = u.status()
        except Exception as exc:
            log.warning("读取 %s 引擎状态失败：%s", u.name, exc)
            st = EngineStatus(u.display_name, False, str(exc), "check_failed",
                              "查看错误信息，修复后刷新状态。")
        if st is not None:
            out.append(st)
    return out
