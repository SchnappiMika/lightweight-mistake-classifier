from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from app.core.preprocess.media_recognition import recognize_isolated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("formula", "audio"))
    parser.add_argument("input_path")
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("--options-file", help="公式选区、反色和缩放选项JSON")
    args = parser.parse_args(argv)
    def block_network(*_args, **_kwargs):
        raise OSError("媒体识别仅使用本地模型，不允许网络连接。")
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network
    try:
        options = json.loads(Path(args.options_file).read_text(encoding="utf-8")) if args.options_file else None
    except (OSError, ValueError) as exc:
        from app.core.models import MediaReviewRecord
        record = MediaReviewRecord(error="无法读取识别选项：{}".format(exc))
    else:
        record = recognize_isolated(args.input_path, args.kind, options)
    payload = json.dumps(record.to_dict(), ensure_ascii=False)
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        sys.stdout.write(payload)
    return 0 if not record.error else 2


if __name__ == "__main__":
    raise SystemExit(main())
