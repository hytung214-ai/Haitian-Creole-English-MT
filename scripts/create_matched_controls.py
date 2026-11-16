#!/usr/bin/env python3
"""Create matched-size random control datasets for the MT cleaning experiment.

This script is intentionally separate from prepare_mt_data.py. The preparation
script creates full/light/risk data; this script only creates random subsets of
full training data with the same sizes as light_clean and risk_clean.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pair:
    """One aligned sentence pair."""

    src: str
    tgt: str


def read_parallel(src_path: Path, tgt_path: Path) -> list[Pair]:
    """Read aligned source/target files and validate line counts."""

    with src_path.open(encoding="utf-8") as src_file:
        src_lines = src_file.read().splitlines()
    with tgt_path.open(encoding="utf-8") as tgt_file:
        tgt_lines = tgt_file.read().splitlines()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Line-count mismatch: {src_path} has {len(src_lines)} lines, "
            f"but {tgt_path} has {len(tgt_lines)} lines."
        )

    return [Pair(src=src, tgt=tgt) for src, tgt in zip(src_lines, tgt_lines, strict=True)]


def count_parallel(src_path: Path, tgt_path: Path) -> int:
    """Return aligned line count after validating both files."""

    return len(read_parallel(src_path, tgt_path))


def write_parallel(pairs: list[Pair], src_path: Path, tgt_path: Path) -> None:
    """Write aligned source/target pairs."""

    src_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("w", encoding="utf-8") as src_file, tgt_path.open(
        "w", encoding="utf-8"
    ) as tgt_file:
        for pair in pairs:
            src_file.write(pair.src + "\n")
            tgt_file.write(pair.tgt + "\n")


def sample_control(full_pairs: list[Pair], target_size: int, seed: int) -> list[Pair]:
    """Sample a deterministic random subset while preserving original order."""

    if target_size > len(full_pairs):
        raise ValueError(
            f"Target size {target_size} is larger than full data size {len(full_pairs)}."
        )

    rng = random.Random(seed)
    selected_indices = set(rng.sample(range(len(full_pairs)), target_size))
    return [pair for idx, pair in enumerate(full_pairs) if idx in selected_indices]


def make_control(
    name: str,
    full_pairs: list[Pair],
    target_size: int,
    data_dir: Path,
    seed: int,
) -> dict[str, object]:
    """Create one matched-size control dataset and return summary metadata."""

    control_pairs = sample_control(full_pairs, target_size=target_size, seed=seed)
    src_out = data_dir / name / "train.hat"
    tgt_out = data_dir / name / "train.eng"
    write_parallel(control_pairs, src_out, tgt_out)

    return {
        "condition": name,
        "control_type": "random_subset_of_full",
        "matched_to_size": target_size,
        "input_pairs": len(full_pairs),
        "kept_pairs": len(control_pairs),
        "removed_pairs": len(full_pairs) - len(control_pairs),
        "removed_percent": f"{100 * (len(full_pairs) - len(control_pairs)) / len(full_pairs):.2f}",
        "seed": seed,
        "output_src": str(src_out),
        "output_tgt": str(tgt_out),
    }


def write_reports(rows: list[dict[str, object]], report_dir: Path) -> None:
    """Save matched-control metadata for reproducibility."""

    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "matched_control_sizes.csv"
    fieldnames = [
        "condition",
        "control_type",
        "matched_to_size",
        "input_pairs",
        "kept_pairs",
        "removed_pairs",
        "removed_percent",
        "seed",
        "output_src",
        "output_tgt",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = report_dir / "matched_control_sizes.json"
    with json_path.open("w", encoding="utf-8") as out_file:
        json.dump(rows, out_file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create light/risk matched-size random controls from full data."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/data_preparation"),
    )
    parser.add_argument("--light-seed", type=int, default=1135)
    parser.add_argument("--risk-seed", type=int, default=1136)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    full_pairs = read_parallel(data_dir / "full" / "train.hat", data_dir / "full" / "train.eng")
    light_size = count_parallel(
        data_dir / "light_clean" / "train.hat",
        data_dir / "light_clean" / "train.eng",
    )
    risk_size = count_parallel(
        data_dir / "risk_clean" / "train.hat",
        data_dir / "risk_clean" / "train.eng",
    )

    rows = [
        make_control(
            "light_control",
            full_pairs,
            target_size=light_size,
            data_dir=data_dir,
            seed=args.light_seed,
        ),
        make_control(
            "risk_control",
            full_pairs,
            target_size=risk_size,
            data_dir=data_dir,
            seed=args.risk_seed,
        ),
    ]
    write_reports(rows, args.report_dir)

    print("Matched controls created.")
    for row in rows:
        print(
            f"{row['condition']}: kept {row['kept_pairs']} / {row['input_pairs']} "
            f"(seed={row['seed']})"
        )


if __name__ == "__main__":
    main()
