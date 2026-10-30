from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    try:
        from PIL import Image, ImageDraw, ImageFont
        from app.core.preprocess.image_ocr import get_engine, model_paths

        paths = model_paths(verify_hash=True)
        print("本地模型总大小：{:.1f} MiB".format(sum(path.stat().st_size for path in paths.values()) / 1024 ** 2))
        print("模型目录：{}".format(next(iter(paths.values())).parent))
        temporary_root = ROOT / ".startup-check"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ocr-check-", dir=temporary_root) as temporary:
            path = Path(temporary) / "printed-text.png"
            image = Image.new("RGB", (800, 140), "white")
            ImageDraw.Draw(image).text((30, 30), "English question: Choose the answer.",
                                      font=ImageFont.load_default(size=32), fill="black")
            image.save(path)
            result = get_engine().recognize_isolated(str(path))
            if "question" not in result.text.casefold():
                raise ValueError("识别组件未通过英文印刷文字自测")
        print("独立进程本地文字识别自测通过，识别引擎耗时 {:.0f} ms。".format(result.elapsed_ms))
        print("复杂公式、手写文字、图形含义和音频不在本项自测范围内。")
        return 0
    except Exception as exc:
        print("图片文字识别检查失败：{}".format(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
