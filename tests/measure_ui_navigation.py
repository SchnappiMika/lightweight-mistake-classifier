"""在临时工作区测量 155 条内置样例的页面导航，不读取用户题库。"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path


def summarize(values):
    ordered = sorted(values)
    return {"count": len(values), "mean_ms": round(statistics.mean(values), 3),
            "p95_ms": round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3),
            "max_ms": round(max(values), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    with tempfile.TemporaryDirectory(prefix="mc_navigation_") as temporary:
        os.environ["MC_WORKSPACE"] = temporary
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt5.QtWidgets import QApplication
        from app.qt_runtime import configure_qt_plugins
        from app import config
        from app.core.dataio import importer
        from app.core.preprocess import pipeline
        from app.core.classify.engine import ClassificationEngine
        from app.ui import theme
        from app.ui.main_window import MainWindow

        configure_qt_plugins()
        pipeline.warmup()
        app = QApplication([])
        window = MainWindow()
        theme.apply(app, theme.current())
        window.ctx.set_setting("guide_done", True)
        questions = importer.import_excel(str(config.SAMPLE_DIR / "sample_questions.xlsx"))
        window.ctx.add_questions(questions)
        for question in questions:
            scheme = window.ctx.scheme_for_subject(question.subject)
            engine = window.ctx.engine(scheme)
            features = pipeline.preprocess(question)
            result = engine.classify(question, features)
            ClassificationEngine.apply_result(question, result, scheme, features)
        window.ctx.questions_changed.emit()
        window.show()
        window.overlay.hide()
        for index in range(5):
            window._switch(index)
            app.processEvents()
        window._w.wait()
        app.processEvents()

        def measure(force_refresh):
            timings = {index: {"switch_ms": [], "events_ms": [], "on_show_ms": []}
                       for index in range(5)}
            originals = [page.on_show for page in window.pages]
            for index, page in enumerate(window.pages):
                def tracked(index=index):
                    started = time.perf_counter()
                    originals[index]()
                    timings[index]["on_show_ms"].append((time.perf_counter() - started) * 1000)
                page.on_show = tracked
            try:
                for _ in range(args.rounds):
                    for index in range(5):
                        if force_refresh:
                            window._invalidate_pages((index,))
                        started = time.perf_counter()
                        window._switch(index)
                        switched = time.perf_counter()
                        app.processEvents()
                        ended = time.perf_counter()
                        timings[index]["switch_ms"].append((switched - started) * 1000)
                        timings[index]["events_ms"].append((ended - switched) * 1000)
            finally:
                for page, original in zip(window.pages, originals):
                    page.on_show = original
            return {str(index): {key: summarize(values) if values else {"count": 0}
                                 for key, values in measures.items()}
                    for index, measures in timings.items()}

        result = {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "platform": platform.platform(), "python": platform.python_version(),
                  "question_count": len(questions),
                  "mode": "offscreen visible window; temporary workspace; bundled samples",
                  "note": "forced_refresh 每次调用现有 on_show；按需刷新仅在变更后调用。事件耗时不代表真实显示器绘制性能。",
                  "forced_refresh": measure(True), "on_demand": measure(False)}
        window.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
