# -*- coding: utf-8 -*-
"""Frozen helper entry: intentionally imports no desktop/Qt modules."""
from __future__ import annotations

import os
import sys


def main(argv=None) -> int:
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return 2
    kind, *arguments = arguments
    if kind == "serve":
        from app.core.preprocess.media_server import main as session_main
        return session_main(arguments)
    if kind == "image":
        from app.core.preprocess.image_ocr_worker import main as image_main
        return image_main(arguments)
    if kind in ("audio", "formula"):
        from app.core.preprocess.media_recognition_worker import main as media_main
        return media_main([kind] + arguments)
    return 2


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
