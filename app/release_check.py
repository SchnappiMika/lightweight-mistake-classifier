"""发布程序内部冒烟入口：所有试运行数据只写入临时工作区。"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="隔离工作区发布版冒烟验证")
    parser.add_argument("report")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument("--formula")
    args = parser.parse_args(argv)
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"frozen": bool(getattr(sys, "frozen", False)),
              "platform": platform.platform(), "python": sys.version,
              "executable": sys.executable, "checks": [], "passed": False,
              "limits": ["本机冒烟，不能替代未装 Python 的干净电脑、macOS 或学生试用验收。"]}
    started = time.perf_counter()
    window = None
    with tempfile.TemporaryDirectory(prefix="mc_release_check_") as temporary:
        os.environ["MC_WORKSPACE"] = temporary
        # 在导入配置、仓库和GUI之前隔离数据，绝不载入用户真实题库。
        report["workspace_isolated"] = True
        try:
            from app import config
            from app.qt_runtime import configure_qt_plugins
            configure_qt_plugins()
            from PyQt5.QtWidgets import QApplication
            from app.ui import theme
            from app.ui.main_window import MainWindow

            application = QApplication.instance() or QApplication([])
            window = MainWindow()
            window.ctx.set_setting("guide_done", True)
            window.overlay.hide()
            theme.apply(application, theme.current())
            window.show()
            application.processEvents()
            for index in range(len(window.pages)):
                window._switch(index)
                application.processEvents()
            report["checks"].append({"name": "gui_navigation", "passed": True, "pages": len(window.pages)})

            from app.core.models import Question
            from app.core.classify.engine import ClassificationEngine
            from app.core.dataio import exporter
            from app.core.dataio.backup import create_backup
            from app.core.preprocess import pipeline

            ctx = window.ctx
            scheme = ctx.repo.scheme_by_subject(config.SUBJECT_MATH)
            question = Question(subject=config.SUBJECT_MATH, raw_text="计算定积分的面积及其应用")
            ctx.add_questions([question])
            bundle = pipeline.preprocess(question)
            engine = ctx.engine(scheme)
            result = engine.classify(question, bundle)
            ClassificationEngine.apply_result(question, result, scheme, bundle)
            ctx.save_questions()
            if not question.classified:
                raise RuntimeError("示例文本未产生有效分类")
            report["checks"].append({"name": "text_classification_and_save", "passed": True,
                                      "labels": question.label_names})

            exported = Path(temporary) / "exports" / "smoke.xlsx"
            exporter.export_excel([question], scheme, str(exported))
            if not exported.is_file():
                raise RuntimeError("未生成 Excel")
            report["checks"].append({"name": "excel_export", "passed": True, "bytes": exported.stat().st_size})
            archive = Path(temporary) / "smoke.zip"
            create_backup(ctx.repo, archive)
            report["checks"].append({"name": "backup", "passed": archive.is_file(),
                                      "bytes": archive.stat().st_size})
            ctx.repo.load_all()
            restored = ctx.repo.question_by_id(question.id)
            if restored is None or restored.label_names != question.label_names:
                raise RuntimeError("重载后分类内容不一致")
            report["checks"].append({"name": "reload", "passed": True})

            if args.image:
                from app.core.preprocess.image_ocr import ImageTextEngine
                record = ImageTextEngine().recognize_isolated(str(Path(args.image).resolve()))
                report["checks"].append({"name": "image_ocr", "passed": bool(record.text),
                                          "text": record.text, "elapsed_ms": record.elapsed_ms})
            for kind, input_path in (("audio", args.audio), ("formula", args.formula)):
                if not input_path:
                    continue
                from app.ui.media_review_dialog import MediaReviewWorker
                received = []
                worker = MediaReviewWorker(kind, str(Path(input_path).resolve()))
                worker.completed.connect(lambda record, error: received.append((record, error)))
                worker.run()
                record, error = received[0] if received else (None, "没有转写结果")
                if error or record is None:
                    raise RuntimeError("{}识别失败：{}".format(kind, error))
                report["checks"].append({"name": "audio_transcription" if kind == "audio" else "formula_recognition", "passed": bool(record.text),
                                          "text": record.text, "elapsed_ms": record.elapsed_ms})
                # Exercise plumbing using model output, without inventing human labels.
                from app.core.preprocess.media_recognition import save_media_review
                original = Path(input_path).resolve()
                attachment = Path(temporary) / "media" / (kind + original.suffix)
                attachment.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, attachment)
                media_question = Question(
                    subject=config.SUBJECT_MATH if kind == "formula" else config.SUBJECT_ENGLISH,
                    qtype=config.QTYPE_FORMULA if kind == "formula" else config.QTYPE_LISTENING,
                    raw_text="", media_path=str(attachment))
                ctx.add_questions([media_question])
                save_media_review(ctx.repo, media_question, record, kind, apply=True)
                media_features = pipeline.preprocess(media_question)
                media_scheme = ctx.repo.scheme_by_subject(media_question.subject)
                media_result = ctx.engine(media_scheme).classify(media_question, media_features)
                ClassificationEngine.apply_result(media_question, media_result, media_scheme, media_features)
                ctx.save_questions()
                media_export = Path(temporary) / "exports" / (kind + ".xlsx")
                exporter.export_excel([media_question], media_scheme, str(media_export))
                media_backup = Path(temporary) / (kind + "_backup.zip")
                create_backup(ctx.repo, media_backup)
                ctx.repo.load_all()
                reloaded = ctx.repo.question_by_id(media_question.id)
                saved = getattr(reloaded, "formula_ocr" if kind == "formula" else "audio_transcript")
                if saved is None or not saved.confirmed or saved.applied_text != record.draft.strip():
                    raise RuntimeError("{}识别确认结果重载不一致".format(kind))
                report["checks"].append({
                    "name": kind + "_review_classify_export_reload", "passed": True,
                    "latex": media_features.latex, "tokens": media_features.text_tokens,
                    "formula_tokens": media_features.formula_tokens,
                    "labels": reloaded.label_names, "export_bytes": media_export.stat().st_size,
                    "backup_bytes": media_backup.stat().st_size,
                    "review_note": "自动应用仅验证数据链路，未经人工校对，不作为准确率或真人验收结论。"})
            window._switch(2)
            application.processEvents()
            screenshot = report_path.with_suffix(".png")
            if window.grab().save(str(screenshot)):
                report["screenshot"] = str(screenshot)
            report["passed"] = all(item["passed"] for item in report["checks"])
        except Exception as exc:
            report["error"] = str(exc)
            report["traceback"] = traceback.format_exc()
        finally:
            if window is not None:
                window.close()
            report["elapsed_ms"] = (time.perf_counter() - started) * 1000
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
