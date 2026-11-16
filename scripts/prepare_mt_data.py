#!/usr/bin/env python3
"""Prepare data conditions for the Haitian Creole -> English MT project.

The script implements the data-preparation work described in the v3 proposal and
the bilingual project plan:

1. Validate that the source and target files are aligned.
2. Split the original training data into train/dev/internal-test sets.
3. Keep the provided official test set untouched for final evaluation.
4. Create three comparable training conditions:
   - full: original training data after dev/internal-test splits are removed
   - light_clean: removes clear, low-risk noise
   - risk_clean: starts from light cleaning and removes extra high-risk examples
5. Save cleaning statistics and removed examples for later reporting.

All comments are intentionally in English so the code can be used directly in the
final reproducibility appendix.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_SRC = "Kreyol-MT-main/eng-hat/train.hat"
DEFAULT_TGT = "Kreyol-MT-main/eng-hat/train.eng"
DEFAULT_TEST_SRC = "Kreyol-MT-main/eng-hat/test.hat"
DEFAULT_TEST_TGT = "Kreyol-MT-main/eng-hat/test.eng"


@dataclass(frozen=True)
class Pair:
    """One aligned sentence pair plus its original 1-based line number."""

    line_no: int
    src: str
    tgt: str


@dataclass(frozen=True)
class Decision:
    """Cleaning decision for a sentence pair."""

    keep: bool
    reasons: tuple[str, ...]


def read_parallel(src_path: Path, tgt_path: Path) -> list[Pair]:
    """Read two parallel text files and fail early if their line counts differ."""

    with src_path.open(encoding="utf-8") as src_file:
        src_lines = src_file.read().splitlines()
    with tgt_path.open(encoding="utf-8") as tgt_file:
        tgt_lines = tgt_file.read().splitlines()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Line-count mismatch: {src_path} has {len(src_lines)} lines, "
            f"but {tgt_path} has {len(tgt_lines)} lines."
        )

    return [
        Pair(line_no=i + 1, src=src.strip(), tgt=tgt.strip())
        for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines, strict=True))
    ]


def write_parallel(pairs: Iterable[Pair], src_path: Path, tgt_path: Path) -> int:
    """Write aligned pairs to source and target files and return the line count."""

    src_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src_path.open("w", encoding="utf-8") as src_file, tgt_path.open(
        "w", encoding="utf-8"
    ) as tgt_file:
        for pair in pairs:
            src_file.write(pair.src + "\n")
            tgt_file.write(pair.tgt + "\n")
            count += 1
    return count


def normalize_text(text: str) -> str:
    """Normalize spacing and case for conservative string comparisons."""

    return re.sub(r"\s+", " ", html.unescape(text)).strip().casefold()


def words(text: str) -> list[str]:
    """Return Unicode word-like tokens used by the heuristic filters."""

    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def word_count(text: str) -> int:
    return len(words(text))


def has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def is_punctuation_only(text: str) -> bool:
    """Detect lines that contain no letters or digits after trimming."""

    stripped = text.strip()
    return bool(stripped) and not has_alpha(stripped) and not has_digit(stripped)


def is_number_only(text: str) -> bool:
    """Detect numeric fragments such as prices, IDs, and standalone quantities."""

    stripped = text.strip()
    return bool(stripped) and has_digit(stripped) and not has_alpha(stripped)


def length_ratio(pair: Pair) -> float:
    """Compute a word-count ratio with a safe denominator."""

    src_words = word_count(pair.src)
    tgt_words = word_count(pair.tgt)
    if src_words == 0 or tgt_words == 0:
        return float("inf")
    return max(src_words, tgt_words) / min(src_words, tgt_words)


def contains_markup_or_html(text: str) -> bool:
    """Detect common markup, XML/HTML tags, and encoded HTML entities."""

    return bool(
        re.search(r"<[^>]+>", text)
        or re.search(r"&(?:[a-zA-Z]+|#[0-9]+|#x[0-9a-fA-F]+);", text)
        or "org.ldschurch.psd.mt.tmx" in text
    )


def contains_url_or_email(text: str) -> bool:
    return bool(
        re.search(r"https?://|www\.", text, flags=re.IGNORECASE)
        or re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
    )


def contains_replacement_char(text: str) -> bool:
    return "\ufffd" in text


WEB_NAV_RE = re.compile(
    r"\b("
    r"home|homepage|menu|login|log in|sign in|register|subscribe|"
    r"contact us|about us|privacy policy|terms of use|cookie|cookies|"
    r"read more|learn more|click here|next page|previous page|"
    r"search|sitemap|share|posted by|comments?|download"
    r")\b",
    flags=re.IGNORECASE,
)


PRODUCT_TECH_RE = re.compile(
    r"\b("
    r"technical|parameters?|specifications?|specs?|model|series|product|"
    r"catalog|catalogue|price|cost advantage|supply chain|manufacturer|"
    r"warranty|voltage|power|capacity|dimension|weight|material|"
    r"shipping|packaging|sku|nova sport|bebox"
    r")\b",
    flags=re.IGNORECASE,
)


def is_web_navigation_like(pair: Pair) -> bool:
    """Flag short UI/navigation fragments that are common in web-aligned data."""

    joined = f"{pair.src} {pair.tgt}"
    return bool(WEB_NAV_RE.search(joined)) and max(
        word_count(pair.src), word_count(pair.tgt)
    ) <= 18


def is_product_or_technical_like(pair: Pair) -> bool:
    """Flag catalog and technical-specification fragments seen in the corpus."""

    joined = f"{pair.src} {pair.tgt}"
    if PRODUCT_TECH_RE.search(joined):
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?\s?(?:mm|cm|kg|kw|v|hz|mah)\b", joined, re.I))


def is_title_like(pair: Pair) -> bool:
    """Flag very short title-style fragments without sentence-final punctuation.

    Many valid web-aligned sentences do not end in punctuation, so this rule is
    deliberately narrow. It targets labels and headings rather than every short
    phrase-like translation pair.
    """

    max_words = max(word_count(pair.src), word_count(pair.tgt))
    min_words = min(word_count(pair.src), word_count(pair.tgt))
    if max_words == 0 or max_words > 5 or min_words > 3:
        return False
    terminal_punctuation = pair.src.endswith((".", "?", "!", ":", ";")) or pair.tgt.endswith(
        (".", "?", "!", ":", ";")
    )
    if terminal_punctuation:
        return False
    return bool(has_alpha(pair.src) and has_alpha(pair.tgt))


def has_suspicious_alignment(pair: Pair) -> bool:
    """Flag pairs whose surface form suggests weak alignment."""

    if length_ratio(pair) > 2.5:
        return True
    src_norm = normalize_text(pair.src)
    tgt_norm = normalize_text(pair.tgt)
    if src_norm and tgt_norm and (src_norm in tgt_norm or tgt_norm in src_norm):
        return min(word_count(pair.src), word_count(pair.tgt)) <= 3
    return False


def light_reasons(pair: Pair, seen_pairs: set[tuple[str, str]]) -> list[str]:
    """Return low-risk removal reasons for the light-cleaning condition."""

    reasons: list[str] = []
    src_norm = normalize_text(pair.src)
    tgt_norm = normalize_text(pair.tgt)
    normalized_pair = (src_norm, tgt_norm)

    if not pair.src or not pair.tgt:
        reasons.append("empty_side")
    if src_norm and src_norm == tgt_norm:
        reasons.append("identical_source_target")
    if normalized_pair in seen_pairs:
        reasons.append("duplicate_pair")
    if contains_markup_or_html(pair.src) or contains_markup_or_html(pair.tgt):
        reasons.append("markup_or_html")
    if contains_url_or_email(pair.src) or contains_url_or_email(pair.tgt):
        reasons.append("url_or_email")
    if contains_replacement_char(pair.src) or contains_replacement_char(pair.tgt):
        reasons.append("encoding_replacement_char")
    if length_ratio(pair) > 3.0:
        reasons.append("extreme_length_ratio")
    if max(word_count(pair.src), word_count(pair.tgt)) > 100:
        reasons.append("very_long_segment")
    if is_punctuation_only(pair.src) or is_punctuation_only(pair.tgt):
        reasons.append("punctuation_only")
    if is_number_only(pair.src) or is_number_only(pair.tgt):
        reasons.append("number_only")
    if min(word_count(pair.src), word_count(pair.tgt)) <= 2 and reasons:
        # Very short pairs are not removed by length alone. The extra label is
        # only added when another low-risk noise rule has already fired.
        reasons.append("very_short_bad_fragment")

    return reasons


def risk_reasons(pair: Pair, existing_reasons: list[str]) -> list[str]:
    """Return additional high-risk removal reasons for the risk-aware condition."""

    reasons: list[str] = []
    short_fragment = min(word_count(pair.src), word_count(pair.tgt)) <= 4

    if short_fragment and existing_reasons:
        reasons.append("short_fragment_with_noise")
    if is_web_navigation_like(pair):
        reasons.append("web_navigation_like")
    if is_product_or_technical_like(pair):
        reasons.append("product_or_technical_like")
    if is_title_like(pair):
        reasons.append("title_like_fragment")
    if has_suspicious_alignment(pair):
        reasons.append("suspicious_alignment")

    return reasons


def clean_pairs(
    pairs: Iterable[Pair],
    decision_fn: Callable[[Pair, set[tuple[str, str]]], Decision],
) -> tuple[list[Pair], list[tuple[Pair, tuple[str, ...]]], Counter[str]]:
    """Apply a cleaning decision function and collect removal statistics."""

    kept: list[Pair] = []
    removed: list[tuple[Pair, tuple[str, ...]]] = []
    reason_counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for pair in pairs:
        decision = decision_fn(pair, seen_pairs)
        normalized_pair = (normalize_text(pair.src), normalize_text(pair.tgt))
        if decision.keep:
            kept.append(pair)
            seen_pairs.add(normalized_pair)
        else:
            removed.append((pair, decision.reasons))
            reason_counts.update(decision.reasons)

    return kept, removed, reason_counts


def light_decision(pair: Pair, seen_pairs: set[tuple[str, str]]) -> Decision:
    reasons = light_reasons(pair, seen_pairs)
    return Decision(keep=not reasons, reasons=tuple(reasons))


def risk_decision(pair: Pair, seen_pairs: set[tuple[str, str]]) -> Decision:
    reasons = light_reasons(pair, seen_pairs)
    reasons.extend(risk_reasons(pair, reasons))
    return Decision(keep=not reasons, reasons=tuple(reasons))


def write_removed_examples(
    removed: list[tuple[Pair, tuple[str, ...]]], path: Path, sample_limit: int
) -> None:
    """Save removed examples with line numbers and removal reasons."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["line_no", "reasons", "src", "tgt"])
        for pair, reasons in removed:
            writer.writerow([pair.line_no, ",".join(reasons), pair.src, pair.tgt])

    grouped: dict[str, list[tuple[Pair, tuple[str, ...]]]] = defaultdict(list)
    for pair, reasons in removed:
        for reason in reasons:
            if len(grouped[reason]) < sample_limit:
                grouped[reason].append((pair, reasons))

    sample_path = path.with_name(path.stem + "_samples.tsv")
    with sample_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["primary_reason", "line_no", "all_reasons", "src", "tgt"])
        for reason in sorted(grouped):
            for pair, reasons in grouped[reason]:
                writer.writerow([reason, pair.line_no, ",".join(reasons), pair.src, pair.tgt])


def write_stats_csv(stats: list[dict[str, object]], path: Path) -> None:
    """Write one compact table that can be pasted into the report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "input_pairs",
        "kept_pairs",
        "removed_pairs",
        "removed_percent",
        "output_src",
        "output_tgt",
    ]
    with path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow({key: row[key] for key in fieldnames})


def split_train_dev_test(
    pairs: list[Pair],
    dev_ratio: float,
    internal_test_ratio: float,
    seed: int,
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Create deterministic train/dev/internal-test splits.

    The selected examples are sampled randomly but written in their original file
    order. This keeps the splits reproducible while making output files easier to
    inspect manually.
    """

    if not 0 < dev_ratio < 1:
        raise ValueError("--dev-ratio must be between 0 and 1.")
    if not 0 < internal_test_ratio < 1:
        raise ValueError("--internal-test-ratio must be between 0 and 1.")
    if dev_ratio + internal_test_ratio >= 1:
        raise ValueError("--dev-ratio + --internal-test-ratio must be less than 1.")

    dev_size = round(len(pairs) * dev_ratio)
    internal_test_size = round(len(pairs) * internal_test_ratio)
    if dev_size < 1 or internal_test_size < 1:
        raise ValueError("dev and internal-test splits must each contain at least one pair.")

    rng = random.Random(seed)
    heldout_indices = rng.sample(range(len(pairs)), dev_size + internal_test_size)
    dev_indices = set(heldout_indices[:dev_size])
    internal_test_indices = set(heldout_indices[dev_size:])
    train_pairs = [
        pair
        for idx, pair in enumerate(pairs)
        if idx not in dev_indices and idx not in internal_test_indices
    ]
    dev_pairs = [pair for idx, pair in enumerate(pairs) if idx in dev_indices]
    internal_test_pairs = [
        pair for idx, pair in enumerate(pairs) if idx in internal_test_indices
    ]
    return train_pairs, dev_pairs, internal_test_pairs


def copy_official_test_files(test_src: Path, test_tgt: Path, output_dir: Path) -> None:
    """Copy the official test set without modification."""

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_src, output_dir / "test.hat")
    shutil.copyfile(test_tgt, output_dir / "test.eng")


def build_condition(
    name: str,
    train_pairs: list[Pair],
    output_dir: Path,
    report_dir: Path,
    decision_fn: Callable[[Pair, set[tuple[str, str]]], Decision] | None,
    sample_limit: int,
) -> dict[str, object]:
    """Create one training condition and return its summary statistics."""

    condition_dir = output_dir / name
    if decision_fn is None:
        kept = train_pairs
        removed: list[tuple[Pair, tuple[str, ...]]] = []
        reason_counts: Counter[str] = Counter()
    else:
        kept, removed, reason_counts = clean_pairs(train_pairs, decision_fn)

    src_out = condition_dir / "train.hat"
    tgt_out = condition_dir / "train.eng"
    write_parallel(kept, src_out, tgt_out)

    if removed:
        write_removed_examples(
            removed,
            report_dir / f"{name}_removed.tsv",
            sample_limit=sample_limit,
        )

    removed_count = len(removed)
    input_count = len(train_pairs)
    row = {
        "condition": name,
        "input_pairs": input_count,
        "kept_pairs": len(kept),
        "removed_pairs": removed_count,
        "removed_percent": f"{100 * removed_count / input_count:.2f}",
        "output_src": str(src_out),
        "output_tgt": str(tgt_out),
        "reason_counts": dict(reason_counts),
    }
    return row


def write_command_log(args: argparse.Namespace, path: Path) -> None:
    """Save the exact settings used for reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        out_file.write("# Data Preparation Command Log\n\n")
        out_file.write("Run from the project root:\n\n")
        out_file.write("```bash\n")
        out_file.write(
            "python scripts/prepare_mt_data.py "
            f"--dev-ratio {args.dev_ratio} "
            f"--internal-test-ratio {args.internal_test_ratio} "
            f"--seed {args.seed} "
            f"--output-dir {args.output_dir} --report-dir {args.report_dir}\n"
        )
        out_file.write("```\n\n")
        out_file.write("Settings:\n\n")
        for key, value in sorted(vars(args).items()):
            out_file.write(f"- `{key}`: `{value}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full, light-cleaned, and risk-aware-cleaned MT data."
    )
    parser.add_argument("--src", type=Path, default=Path(DEFAULT_SRC))
    parser.add_argument("--tgt", type=Path, default=Path(DEFAULT_TGT))
    parser.add_argument("--test-src", type=Path, default=Path(DEFAULT_TEST_SRC))
    parser.add_argument("--test-tgt", type=Path, default=Path(DEFAULT_TEST_TGT))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/data_preparation"),
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.10,
        help="Fraction of the original training data to hold out as dev.",
    )
    parser.add_argument(
        "--internal-test-ratio",
        type=float,
        default=0.10,
        help="Fraction of the original training data to hold out as internal test.",
    )
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=None,
        help=(
            "Optional smoke-test cap. When set, only the first N original training "
            "pairs are used before the dev split."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Number of removed examples to save per reason in sample TSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_all = read_parallel(args.src, args.tgt)
    if args.max_train_pairs is not None:
        min_required = 1 / (1 - args.dev_ratio - args.internal_test_ratio)
        if args.max_train_pairs < min_required:
            raise ValueError("--max-train-pairs is too small for the requested split ratios.")
        train_all = train_all[: args.max_train_pairs]

    official_test_pairs = read_parallel(args.test_src, args.test_tgt)
    train_pairs, dev_pairs, internal_test_pairs = split_train_dev_test(
        train_all,
        dev_ratio=args.dev_ratio,
        internal_test_ratio=args.internal_test_ratio,
        seed=args.seed,
    )

    # Shared evaluation files are written once and reused for every condition.
    write_parallel(dev_pairs, args.output_dir / "dev" / "dev.hat", args.output_dir / "dev" / "dev.eng")
    write_parallel(
        internal_test_pairs,
        args.output_dir / "internal_test" / "test.hat",
        args.output_dir / "internal_test" / "test.eng",
    )
    copy_official_test_files(args.test_src, args.test_tgt, args.output_dir / "official_test")

    stats: list[dict[str, object]] = []
    stats.append(
        build_condition(
            "full",
            train_pairs,
            args.output_dir,
            args.report_dir,
            decision_fn=None,
            sample_limit=args.sample_limit,
        )
    )
    stats.append(
        build_condition(
            "light_clean",
            train_pairs,
            args.output_dir,
            args.report_dir,
            decision_fn=light_decision,
            sample_limit=args.sample_limit,
        )
    )
    stats.append(
        build_condition(
            "risk_clean",
            train_pairs,
            args.output_dir,
            args.report_dir,
            decision_fn=risk_decision,
            sample_limit=args.sample_limit,
        )
    )

    inventory = {
        "source_train_pairs": len(train_all),
        "official_test_pairs": len(official_test_pairs),
        "train_pairs_after_split": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "internal_test_pairs": len(internal_test_pairs),
        "dev_seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "internal_test_ratio": args.internal_test_ratio,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_stats_csv(stats, args.report_dir / "condition_sizes.csv")
    with (args.report_dir / "condition_stats.json").open("w", encoding="utf-8") as out_file:
        json.dump({"inventory": inventory, "conditions": stats}, out_file, indent=2)
    write_command_log(args, args.report_dir / "command_log.md")

    print("Data preparation complete.")
    print(json.dumps(inventory, indent=2))
    for row in stats:
        print(
            f"{row['condition']}: kept {row['kept_pairs']} / {row['input_pairs']} "
            f"({row['removed_percent']}% removed)"
        )


if __name__ == "__main__":
    main()
