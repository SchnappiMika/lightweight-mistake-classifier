"""Private file protocol for reusing one offline model during a batch.

The client owns the temporary directory and terminates the process on cancel.
No GUI modules or model downloads are used by this entry point.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time


def serve(kind: str, directory: Path, handler=None) -> int:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("媒体任务目录不存在。")
    if handler is None:
        if kind == "image":
            from app.core.preprocess.image_ocr import ImageTextEngine
            engine = ImageTextEngine()
            handler = lambda request: engine.recognize(request["input_path"]).to_dict()
        elif kind == "audio":
            from app.core.preprocess.media_recognition import recognize_isolated
            handler = lambda request: recognize_isolated(request["input_path"], "audio").to_dict()
        else:
            raise ValueError("不支持的常驻媒体类型。")
    index = 1
    while not (directory / "stop").exists():
        source = directory / "request-{:08d}.json".format(index)
        if not source.is_file():
            time.sleep(0.01)
            continue
        try:
            request = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(request, dict) or not isinstance(request.get("input_path"), str):
                raise ValueError("媒体请求必须包含字符串 input_path。")
            if not Path(request["input_path"]).is_absolute():
                raise ValueError("媒体请求必须使用绝对文件路径。")
            payload = handler(request)
            if not isinstance(payload, dict):
                raise ValueError("媒体识别未返回有效记录。")
        except Exception as exc:
            payload = {"text": "", "draft": "", "confirmed": False, "error": str(exc)}
        output = directory / "response-{:08d}.json".format(index)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, output)
        index += 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="复用本地模型的批量媒体工作进程")
    parser.add_argument("kind", choices=("image", "audio"))
    parser.add_argument("directory")
    args = parser.parse_args(argv)
    def block_network(*_args, **_kwargs):
        raise OSError("媒体识别仅使用本地模型，不允许网络连接。")
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network
    return serve(args.kind, Path(args.directory))


if __name__ == "__main__":
    raise SystemExit(main())
