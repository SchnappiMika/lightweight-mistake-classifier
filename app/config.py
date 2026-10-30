# -*- coding: utf-8 -*-
"""全局配置：应用元信息、路径解析、领域常量与算法可调参数。

设计要点：
1. 资源目录通过 ``_base_dir()`` 解析，兼容 PyInstaller 单文件模式（``sys._MEIPASS``），
   保证打包后仍能读到 resources/ 下的词典与默认分类体系。
2. 工作区（可写数据）固定放在用户主目录，避免打包后写入只读的安装目录。
3. 全流程零网络依赖，此处不存在任何云端接口地址。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 应用元信息
# --------------------------------------------------------------------------- #
APP_NAME = "轻量化错题自动分类管理系统"
APP_NAME_EN = "Lightweight Mistake Classifier"
APP_VERSION = "1.0.0"
ORG_NAME = "CollegeMistakeClassifier"


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
def _base_dir() -> Path:
    """返回只读资源根目录，兼容源码运行与 PyInstaller 冻结运行。"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _base_dir()
RESOURCE_DIR = BASE_DIR / "resources"
STOPWORD_DIR = RESOURCE_DIR / "stopwords"
USERDICT_DIR = RESOURCE_DIR / "userdict"
SCHEME_SEED_DIR = RESOURCE_DIR / "schemes"
FORMULA_MAP_FILE = RESOURCE_DIR / "formula_map.json"
SAMPLE_DIR = BASE_DIR / "samples"

# 可写工作区：允许用环境变量覆盖，便携版可指向程序同级目录
WORKSPACE_DIR = Path(os.environ.get("MC_WORKSPACE", str(Path.home() / ".mistake_classifier")))
DATA_DIR = WORKSPACE_DIR / "data"
MEDIA_DIR = WORKSPACE_DIR / "media"
EXPORT_DIR = WORKSPACE_DIR / "exports"
LOG_DIR = WORKSPACE_DIR / "logs"
MODEL_DIR = WORKSPACE_DIR / "model"

SCHEME_FILE = DATA_DIR / "schemes.json"
QUESTION_FILE = DATA_DIR / "questions.json"
FEEDBACK_FILE = DATA_DIR / "feedback_log.json"
WEIGHT_FILE = MODEL_DIR / "weight_rules.json"
PROFILE_FILE = MODEL_DIR / "label_profiles.json"
SETTINGS_FILE = WORKSPACE_DIR / "settings.json"


def ensure_workspace() -> None:
    """首次启动时创建工作区目录树（幂等）。"""
    for d in (WORKSPACE_DIR, DATA_DIR, MEDIA_DIR, EXPORT_DIR, LOG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 领域常量：学科
# --------------------------------------------------------------------------- #
SUBJECT_MATH = "高等数学"
SUBJECT_PHYSICS = "大学物理"
SUBJECT_ENGLISH = "大学英语"
SUBJECT_LINALG = "线性代数"
SUBJECT_PROB = "概率论与数理统计"
SUBJECT_GENERAL = "通用"

SUBJECTS = [
    SUBJECT_MATH,
    SUBJECT_PHYSICS,
    SUBJECT_ENGLISH,
    SUBJECT_LINALG,
    SUBJECT_PROB,
    SUBJECT_GENERAL,
]

# 学科 -> 语言，决定走中文还是英文预处理分支
SUBJECT_LANG = {
    SUBJECT_ENGLISH: "en",
}
DEFAULT_LANG = "zh"


# --------------------------------------------------------------------------- #
# 领域常量：题型（决定预处理分流与特征权重桶）
# --------------------------------------------------------------------------- #
QTYPE_TEXT = "text"           # 纯文本题干
QTYPE_FORMULA = "formula"     # 公式类（高数 / 物理 / 线代 / 概率）
QTYPE_FIGURE = "figure"       # 图形类（物理 / 工科）
QTYPE_LISTENING = "listening"  # 英语听力
QTYPE_READING = "reading"     # 英语阅读 / 完形填空

QTYPES = [QTYPE_TEXT, QTYPE_FORMULA, QTYPE_FIGURE, QTYPE_LISTENING, QTYPE_READING]

QTYPE_LABELS = {
    QTYPE_TEXT: "纯文本题干",
    QTYPE_FORMULA: "公式类",
    QTYPE_FIGURE: "图形类",
    QTYPE_LISTENING: "英语听力",
    QTYPE_READING: "英语阅读/完形",
}

# 需要外部媒体文件的题型
MEDIA_QTYPES = {QTYPE_FORMULA, QTYPE_FIGURE, QTYPE_LISTENING}
IMAGE_QTYPES = {QTYPE_FORMULA, QTYPE_FIGURE}
AUDIO_QTYPES = {QTYPE_LISTENING}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


# --------------------------------------------------------------------------- #
# 领域常量：分类维度（支持多维度自定义归档）
# --------------------------------------------------------------------------- #
DIM_KNOWLEDGE = "knowledge"
DIM_ERROR = "error_type"
DIM_DIFFICULTY = "difficulty"
DIM_PRIORITY = "priority"

DIM_LABELS = {
    DIM_KNOWLEDGE: "知识点",
    DIM_ERROR: "错误类型",
    DIM_DIFFICULTY: "题目难度",
    DIM_PRIORITY: "复习优先级",
}
DEFAULT_DIMS = [DIM_KNOWLEDGE, DIM_ERROR, DIM_DIFFICULTY, DIM_PRIORITY]


# --------------------------------------------------------------------------- #
# 算法参数：双层分级分类
# --------------------------------------------------------------------------- #
COARSE_TOP_K = 5          # 粗筛保留候选标签上限（需求：3-5 个）
COARSE_MIN_K = 3          # 粗筛候选下限，不足时按先验补齐以保障召回率
COARSE_KEYWORD_WEIGHT = 1.0    # 分词精确命中权重
COARSE_SUBSTR_WEIGHT = 0.6     # 原文子串命中权重（覆盖未登录多字关键词）
COARSE_MIN_SCORE = 1e-6        # 低于该分数视为未命中
# 人工修正学到的判别词并入粗筛倒排索引时的权重折扣。
# 机器学来的词精度不如人工挑的关键词，打折可保证它只补召回、不喧宾夺主。
COARSE_LEARNED_WEIGHT = 0.55
# 每个标签最多有多少个"学到的词"进入粗筛索引（只取 delta 层最高的若干个），
# 避免索引被低价值词撑爆、也避免大标签靠词多通吃。
FEEDBACK_KEYWORD_TOPN = 8

# 特征通道
CHANNEL_TEXT = "text"
CHANNEL_FORMULA = "formula"
CHANNEL_QTYPE = "qtype"
CHANNEL_SUBJECT = "subject"
CHANNELS = [CHANNEL_TEXT, CHANNEL_FORMULA, CHANNEL_QTYPE, CHANNEL_SUBJECT]
CHANNEL_LABELS = {
    CHANNEL_TEXT: "文本特征",
    CHANNEL_FORMULA: "公式特征",
    CHANNEL_QTYPE: "题型特征",
    CHANNEL_SUBJECT: "学科特征",
}

# 关键词在标签画像中的放大倍数（关键词比样本更具判别力）
LABEL_KEYWORD_BOOST = 3.0
# 置信度低于该阈值时，标记为"存疑"提示用户复核
LOW_CONFIDENCE_THRESHOLD = 0.15
# 精匹配全 0、只能靠粗筛关键词命中定分类时，置信度的折算系数。
# 取值保证结果落在"低置信、建议复核"区间——这种分类确实只有关键词依据，
# 不该和多特征融合算出来的结果显示成同等可信。
COARSE_ONLY_CONFIDENCE = 0.5


# --------------------------------------------------------------------------- #
# 算法参数：增量反馈优化
# --------------------------------------------------------------------------- #
FEEDBACK_POSITIVE_GAIN = 1.0    # 正样本并入正确标签画像的强度
FEEDBACK_NEGATIVE_DECAY = 0.25  # 错误标签画像中重合词的衰减强度
WEIGHT_LEARNING_RATE = 0.08     # 权重规则在线更新学习率
WEIGHT_MIN = 0.02               # 单通道权重下限，防止通道被完全抹掉


# --------------------------------------------------------------------------- #
# 性能与界面
# --------------------------------------------------------------------------- #
SINGLE_QUESTION_TIME_BUDGET_MS = 1000   # 需求：单条错题全流程 ≤ 1 秒
THEME_LIGHT = "light"
THEME_DARK = "dark"
DEFAULT_THEME = THEME_LIGHT
