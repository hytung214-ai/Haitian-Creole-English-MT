#!/usr/bin/env python3
"""Apply the existing MT cleaning rules to an arbitrary parallel dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_mt_data import (
    Decision,
    Pair,
    clean_pairs,
    light_decision,
    read_parallel,
    risk_decision,
    write_parallel,
)


DECISIONS = {
    "light_clean": light_decision,
    "risk_clean": risk_decision,
}


def write_removed(
    removed: list[tuple[Pair, tuple[str, ...]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["original_line_no", "reasons", "src", "tgt"])
        for pair, reasons in removed:
            writer.writerow([pair.line_no, ",".join(reasons), pair.src, pair.tgt])


def write_kept_map(kept: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["cleaned_line_no", "original_line_no"])
        for cleaned_idx, pair in enumerate(kept, start=1):
            writer.writerow([cleaned_idx, pair.line_no])


def clean_condition(
    condition: str,
    pairs: list[Pair],
    output_dir: Path,
    report_dir: Path,
    src_lang: str,
    tgt_lang: str,
) -> dict[str, object]:
    decision_fn = DECISIONS[condition]
    kept, removed, reason_counts = clean_pairs(pairs, decision_fn)

    condition_dir = output_dir / condition
    write_parallel(kept, condition_dir / f"test.{src_lang}", condition_dir / f"test.{tgt_lang}")
    write_removed(removed, report_dir / f"{condition}_removed.tsv")
    write_kept_map(kept, report_dir / f"{condition}_kept_map.tsv")

    return {
        "condition": condition,
        "input_pairs": len(pairs),
        "kept_pairs": len(kept),
        "removed_pairs": len(removed),
        "removed_percent": round(100 * len(removed) / len(pairs), 2),
        "reason_counts": dict(Counter(reason_counts)),
        "output_src": str(condition_dir / f"test.{src_lang}"),
        "output_tgt": str(condition_dir / f"test.{tgt_lang}"),
        "removed_report": str(report_dir / f"{condition}_removed.tsv"),
        "kept_map": str(report_dir / f"{condition}_kept_map.tsv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply light_clean and risk_clean rules to a parallel test set."
    )
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--tgt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--src-lang", default="hat")
    parser.add_argument("--tgt-lang", default="eng")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(DECISIONS),
        default=sorted(DECISIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_parallel(args.src, args.tgt)
    summaries = [
        clean_condition(
            condition,
            pairs,
            args.output_dir,
            args.report_dir,
            args.src_lang,
            args.tgt_lang,
        )
        for condition in args.conditions
    ]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / "summary.json").open("w", encoding="utf-8") as out_file:
        json.dump({"input_src": str(args.src), "input_tgt": str(args.tgt), "conditions": summaries}, out_file, indent=2)

    for summary in summaries:
        print(
            f"{summary['condition']}: kept {summary['kept_pairs']} / "
            f"{summary['input_pairs']} ({summary['removed_percent']}% removed)"
        )


if __name__ == "__main__":
    main()
