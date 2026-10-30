# -*- coding: utf-8 -*-
"""显式准备本地听力模型；不传 --download 时只做说明。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="准备 faster-whisper 本地模型")
    parser.add_argument("--download", action="store_true", help="明确允许从模型仓库下载")
    parser.add_argument("--model", default="base", choices=("tiny", "base"))
    parser.add_argument("--model-dir", default=os.environ.get("MC_MODEL_DIR", ""))
    args = parser.parse_args(argv)
    root = Path(args.model_dir or os.environ.get("MC_MODEL_DIR", "") or
                (Path.home() / ".mistake_classifier" / "models"))
    target = root / "audio"
    target.mkdir(parents=True, exist_ok=True)
    if not args.download:
        print("未下载模型。若确认联网准备本地模型，请重新运行并增加 --download。")
        print("目标目录：{}".format(target))
        return 1
    from faster_whisper import WhisperModel
    print("正在下载/准备 faster-whisper {} 到 {}".format(args.model, target))
    WhisperModel(args.model, device="cpu", compute_type="int8", download_root=str(target))
    print("模型准备完成；运行期将只读取本地目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
