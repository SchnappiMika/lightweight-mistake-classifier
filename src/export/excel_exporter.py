# -*- coding: utf-8 -*-
"""
Excel export helper module
"""
import os

def export_to_excel(results, filepath, by_subject=True, by_category=False):
    import pandas as pd
    if not results:
        return
    df = pd.DataFrame(results)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        if by_subject and "subject" in df.columns:
            for subj in df["subject"].unique():
                sdf = df[df["subject"] == subj]
                sdf.to_excel(writer, sheet_name=str(subj)[:31], index=False)
        elif by_category and "category" in df.columns:
            for cat in df["category"].unique():
                cdf = df[df["category"] == cat]
                cdf.to_excel(writer, sheet_name=str(cat)[:31], index=False)
        else:
            df.to_excel(writer, sheet_name="All Results", index=False)

def results_summary(results):
    stats = {"total": len(results)}
    if results:
        from collections import Counter
        stats["subjects"] = dict(Counter(r.get("subject", "unknown") for r in results))
        stats["categories"] = dict(Counter(r.get("category", "unknown") for r in results))
        stats["corrected"] = sum(1 for r in results if r.get("corrected"))
    return stats