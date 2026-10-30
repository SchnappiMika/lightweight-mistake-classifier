"""Read label training texts without importing questions or copying attachments."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List

from app.core.dataio.importer import COLUMN_ALIASES, ImportError_, _read_text_file


def read_samples(path: str) -> List[str]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".txt":
        lines = [line for line in _read_text_file(source).splitlines()
                 if not line.strip().startswith("#")]
        body = "\n".join(lines).strip()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        if len(paragraphs) > 1:
            samples = paragraphs
        else:
            # Numbered questions often have wrapped stems and option lines.
            # Keep each numbered question intact instead of treating A/B as
            # independent training samples.
            starts = list(re.finditer(r"(?m)^\s*(?:\d{1,3}[.、．)）]|第\s*\d+\s*题)(?!\d)\s*", body))
            if starts and not body[:starts[0].start()].strip():
                boundaries = [match.start() for match in starts] + [len(body)]
                samples = [body[begin:end].strip() for begin, end in zip(boundaries, boundaries[1:])]
            else:
                samples = [line.strip() for line in lines if line.strip()]
    elif suffix in (".xlsx", ".csv", ".tsv"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError_("缺少 pandas 依赖，无法读取样本表格") from exc
        try:
            if suffix in (".csv", ".tsv"):
                table = pd.read_csv(io.StringIO(_read_text_file(source)),
                                    sep="\t" if suffix == ".tsv" else ",",
                                    dtype=str, keep_default_na=False)
            else:
                table = pd.read_excel(source, dtype=str).fillna("")
        except ImportError_:
            raise
        except Exception as exc:
            raise ImportError_("样本表格解析失败：{}".format(exc)) from exc
        aliases = {alias.lower() for alias in COLUMN_ALIASES["raw_text"]}
        aliases.update(("样本", "标准样本", "标准错题样本", "sample", "sample_text"))
        column = next((column for column in table.columns
                       if str(column).strip().lower() in aliases), None)
        if column is None:
            if len(table.columns) == 1:
                column = table.columns[0]
            else:
                raise ImportError_("未找到样本文本列，请使用「题干」或「标准样本」作为列名")
        samples = [str(value).strip() for value in table[column] if str(value).strip()]
    elif suffix == ".xls":
        raise ImportError_("暂不支持旧版 .xls，请另存为 .xlsx 后导入")
    else:
        raise ImportError_("不支持的样本格式，请使用 TXT、XLSX、CSV 或 TSV")
    if not samples:
        raise ImportError_("文件中没有可导入的样本文字")
    # Preserve sample text/ordering while counting each exact sample once.
    return list(dict.fromkeys(samples))
