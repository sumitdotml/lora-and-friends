#!/usr/bin/env python3
"""
Extract suspicious rows from an OpenMath raw dataset using simple phrase heuristics.

Usage:
  uv run python scripts/extract_suspicious_openmath_rows.py
  uv run python scripts/extract_suspicious_openmath_rows.py --input artifacts/raw_datasets/openmath_original_clean/train.jsonl --output-dir artifacts/audits/openmath_original_clean_suspicious_rows_train
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_INPUT = Path("artifacts/raw_datasets/openmath_original_clean/train.jsonl")
DEFAULT_OUTPUT = Path("artifacts/audits/openmath_original_clean_suspicious_rows_train")

SUSPICIOUS_PATTERNS = {
    "cannot_spend_more_than_have": "cannot spend more than",
    "does_not_align_logical_outcome": "doesn't align with the logical outcome",
    "set_value_to_100": "set the value to 100",
    "strictly_interpreting": "strictly interpreting",
    "technically_not_have_money": "would technically not have any money left",
    "must_be_integer": "must be an integer",
    "seems_to_be_looking_for": "seems to be looking for",
}


def parse_args() -> tuple[Path, Path]:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return args.input, args.output_dir


def main() -> None:
    input_path, output_dir = parse_args()
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern_counts = Counter()
    source_counts = Counter()
    suspicious_rows = 0

    out_path = output_dir / "suspicious_rows.jsonl"
    with input_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            solution = row["generated_solution"].lower()
            matched_names = [name for name, pattern in SUSPICIOUS_PATTERNS.items() if pattern in solution]
            if not matched_names:
                continue
            suspicious_rows += 1
            source_counts[row["source"]] += 1
            for name in matched_names:
                pattern_counts[name] += 1
            payload = {
                "row_id": row["row_id"],
                "source": row["source"],
                "matched_patterns": matched_names,
                "problem": row["problem"],
                "generated_solution": row["generated_solution"],
                "expected_answer": row["expected_answer"],
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "suspicious_rows": suspicious_rows,
        "source_counts": dict(source_counts),
        "pattern_counts": dict(pattern_counts),
        "artifacts": {"suspicious_rows_jsonl": str(out_path)},
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
