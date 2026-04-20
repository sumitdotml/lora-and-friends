#!/usr/bin/env python3
"""
Build a deterministic manual-review sample from a training subset.

Usage:
  uv run python scripts/build_manual_audit_sample.py
  uv run python scripts/build_manual_audit_sample.py --input artifacts/subsets/openmath_original_clean/train.jsonl --output-dir artifacts/audits/openmath_original_clean_manual_review_100 --per-source 50
"""

from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

SEED = 20260420
DEFAULT_PER_SOURCE = 5
DEFAULT_INPUT_PATH = Path("artifacts/subsets/openmath_original_clean/train.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/audits/openmath_original_clean_manual_review")


def parse_args() -> tuple[Path, Path, int, int]:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-source", type=int, default=DEFAULT_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    return args.input, args.output_dir, args.per_source, args.seed


def main() -> None:
    input_path, output_dir, per_source, seed = parse_args()
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_source[row["source"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_rows: list[dict] = []
    sampled_counts: dict[str, int] = {}
    for source in sorted(by_source):
        rows = by_source[source]
        take = min(per_source, len(rows))
        picked = rng.sample(rows, take)
        sampled_counts[source] = take
        sampled_rows.extend(
            {
                "row_id": row["row_id"],
                "source": row["source"],
                "problem": row["problem"],
                "generated_solution": row["generated_solution"],
                "expected_answer": row["expected_answer"],
            }
            for row in picked
        )

    rng.shuffle(sampled_rows)

    sample_path = output_dir / "sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for row in sampled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "input_path": str(input_path),
        "per_source": per_source,
        "sample_rows": len(sampled_rows),
        "sampled_counts": sampled_counts,
        "artifacts": {"sample_jsonl": str(sample_path)},
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
