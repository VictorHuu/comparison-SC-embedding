#!/usr/bin/env python3
"""Generate transfer-v2 control diagnostics from prepared artifacts.

Lightweight control script to validate whether pair-level overlaps are
reasonable before full transfer model runs.
"""

from __future__ import annotations

import csv
import os

DIAG_CSV = "transfer_v2/pair_diagnostics.csv"
OUT_DIR = "transfer"
OUT_CSV = os.path.join(OUT_DIR, "transfer_control_v2_diagnostics.csv")
OUT_MD = os.path.join(OUT_DIR, "report_transfer_control_v2.md")


def read_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(DIAG_CSV):
        raise FileNotFoundError(f"Missing {DIAG_CSV}. Run transfer_v2_prepare.py first.")

    rows = read_csv(DIAG_CSV)
    out = []
    high_ratio = 0

    for r in rows:
        ratio = to_float(r.get("canonical_over_raw_ratio", ""))
        flag = "ok"
        if ratio == ratio and ratio > 50:
            flag = "high_ratio"
            high_ratio += 1

        out.append(
            {
                "train_dataset": r.get("train_dataset", ""),
                "test_dataset": r.get("test_dataset", ""),
                "strict_scope": r.get("strict_scope", ""),
                "n_shared_raw": r.get("n_shared_raw", ""),
                "n_shared_canonical": r.get("n_shared_canonical", ""),
                "canonical_over_raw_ratio": r.get("canonical_over_raw_ratio", ""),
                "quality_flag": flag,
            }
        )

    write_csv(
        OUT_CSV,
        out,
        [
            "train_dataset",
            "test_dataset",
            "strict_scope",
            "n_shared_raw",
            "n_shared_canonical",
            "canonical_over_raw_ratio",
            "quality_flag",
        ],
    )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# report_transfer_control_v2\n\n")
        f.write("## 诊断摘要\n\n")
        f.write(f"- total pairs: {len(rows)}\n")
        f.write(f"- high_ratio pairs (>50): {high_ratio}\n\n")
        f.write("## 输出\n\n")
        f.write(f"- `{OUT_CSV}`\n")
        f.write(f"- `{OUT_MD}`\n")

    print(f"[OK] wrote {OUT_CSV}")
    print(f"[OK] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
