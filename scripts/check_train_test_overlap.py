#!/usr/bin/env python3
"""Check overlap between a parallel training set and a parallel test set."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def normalize(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def build_index(items: list[str]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items, start=1):
        index[item].append(i)
    return index


def write_matches(
    path: Path,
    test_src: list[str],
    test_tgt: list[str],
    train_src: list[str],
    train_tgt: list[str],
    matches: list[tuple[int, list[int], str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(
            [
                "match_type",
                "test_line",
                "train_lines",
                "test_src",
                "test_tgt",
                "first_train_src",
                "first_train_tgt",
            ]
        )
        for test_line, train_lines, match_type in matches:
            first_train_line = train_lines[0]
            writer.writerow(
                [
                    match_type,
                    test_line,
                    ",".join(str(line) for line in train_lines),
                    test_src[test_line - 1],
                    test_tgt[test_line - 1],
                    train_src[first_train_line - 1],
                    train_tgt[first_train_line - 1],
                ]
            )


def collect_matches(
    test_keys: list[str],
    train_index: dict[str, list[int]],
    match_type: str,
) -> list[tuple[int, list[int], str]]:
    matches = []
    for test_idx, key in enumerate(test_keys, start=1):
        if key in train_index:
            matches.append((test_idx, train_index[key], match_type))
    return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-src", type=Path, required=True)
    parser.add_argument("--train-tgt", type=Path, required=True)
    parser.add_argument("--test-src", type=Path, required=True)
    parser.add_argument("--test-tgt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_src = read_lines(args.train_src)
    train_tgt = read_lines(args.train_tgt)
    test_src = read_lines(args.test_src)
    test_tgt = read_lines(args.test_tgt)

    if len(train_src) != len(train_tgt):
        raise ValueError("Training source/target line counts differ.")
    if len(test_src) != len(test_tgt):
        raise ValueError("Test source/target line counts differ.")

    train_pair_exact = [f"{src}\t{tgt}" for src, tgt in zip(train_src, train_tgt, strict=True)]
    test_pair_exact = [f"{src}\t{tgt}" for src, tgt in zip(test_src, test_tgt, strict=True)]
    train_pair_norm = [
        f"{normalize(src)}\t{normalize(tgt)}"
        for src, tgt in zip(train_src, train_tgt, strict=True)
    ]
    test_pair_norm = [
        f"{normalize(src)}\t{normalize(tgt)}"
        for src, tgt in zip(test_src, test_tgt, strict=True)
    ]

    match_sets = {
        "exact_pair": collect_matches(
            test_pair_exact,
            build_index(train_pair_exact),
            "exact_pair",
        ),
        "normalized_pair": collect_matches(
            test_pair_norm,
            build_index(train_pair_norm),
            "normalized_pair",
        ),
        "exact_source": collect_matches(
            test_src,
            build_index(train_src),
            "exact_source",
        ),
        "normalized_source": collect_matches(
            [normalize(x) for x in test_src],
            build_index([normalize(x) for x in train_src]),
            "normalized_source",
        ),
        "exact_target": collect_matches(
            test_tgt,
            build_index(train_tgt),
            "exact_target",
        ),
        "normalized_target": collect_matches(
            [normalize(x) for x in test_tgt],
            build_index([normalize(x) for x in train_tgt]),
            "normalized_target",
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_pairs": len(train_src),
        "test_pairs": len(test_src),
        "matches": {
            name: {
                "test_lines_matched": len(matches),
                "train_occurrences": sum(len(train_lines) for _test_line, train_lines, _kind in matches),
            }
            for name, matches in match_sets.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    for name, matches in match_sets.items():
        write_matches(
            args.output_dir / f"{name}_matches.tsv",
            test_src,
            test_tgt,
            train_src,
            train_tgt,
            matches,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
