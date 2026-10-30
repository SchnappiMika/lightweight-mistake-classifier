from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path


def block_network(*args, **kwargs):
    raise OSError("图片文字识别仅使用本地模型，不允许网络连接。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地图片文字识别工作进程")
    parser.add_argument("input_path")
    parser.add_argument("output_path", nargs="?")
    args = parser.parse_args(argv)
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network
    try:
        from app.core.preprocess.image_ocr import ImageTextEngine

        record = ImageTextEngine().recognize(args.input_path)
        payload = record.to_dict()
        if args.output_path:
            Path(args.output_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        if args.output_path:
            Path(args.output_path).write_text(json.dumps({"error": str(exc)}, ensure_ascii=False), encoding="utf-8")
        elif sys.stderr is not None:
            print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
