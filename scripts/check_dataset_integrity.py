#!/usr/bin/env python3
"""
Check the frozen training data against GSM8K test and local validation overlap.

Usage:
  uv run --with datasets python scripts/check_dataset_integrity.py
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from argparse import ArgumentParser
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_TRAIN = Path("artifacts/raw_datasets/openmath_original_clean/train.jsonl")
DEFAULT_VAL = Path("artifacts/raw_datasets/openmath_original_clean/val.jsonl")
DEFAULT_OUTPUT = Path("artifacts/audits/contamination_check/report.json")
BENCHMARK_DATASET = "openai/gsm8k"
BENCHMARK_CONFIG = "main"
BENCHMARK_SPLIT = "test"
MAX_EXAMPLES = 25


def parse_args() -> tuple[Path, Path, Path]:
    parser = ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return args.train, args.val, args.output


def canonical_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return " ".join(text.strip().split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def row_ref(row: dict[str, Any], normalized_problem: str) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "source": row["source"],
        "problem_hash": sha256_text(normalized_problem),
    }


def main() -> None:
    train_path, val_path, output_path = parse_args()
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)

    gsm8k_train_rows = [row for row in train_rows if row["source"] == "gsm8k"]
    normalized_train_gsm8k: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gsm8k_train_rows:
        normalized = canonical_problem(row["problem"])
        normalized_train_gsm8k[normalized].append(row_ref(row, normalized))

    benchmark = load_dataset(BENCHMARK_DATASET, BENCHMARK_CONFIG, split=BENCHMARK_SPLIT)
    benchmark_overlaps: list[dict[str, Any]] = []
    for idx, row in enumerate(benchmark):
        normalized = canonical_problem(row["question"])
        matches = normalized_train_gsm8k.get(normalized, [])
        if matches:
            benchmark_overlaps.append(
                {
                    "benchmark_index": idx,
                    "benchmark_question_hash": sha256_text(normalized),
                    "training_matches": matches,
                }
            )

    train_by_row_id = {row["row_id"]: row for row in train_rows}
    val_by_row_id = {row["row_id"]: row for row in val_rows}
    row_id_overlaps = sorted(set(train_by_row_id) & set(val_by_row_id))

    train_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        normalized = canonical_problem(row["problem"])
        train_by_problem[normalized].append(row_ref(row, normalized))

    problem_overlap_by_val_row_id: dict[str, dict[str, Any]] = {}
    for row in val_rows:
        normalized = canonical_problem(row["problem"])
        matches = train_by_problem.get(normalized, [])
        if matches:
            problem_overlap_by_val_row_id[row["row_id"]] = {
                "val_row": row_ref(row, normalized),
                "train_matches": matches[:MAX_EXAMPLES],
                "train_match_count": len(matches),
            }

    benchmark_overlap_count = sum(len(item["training_matches"]) for item in benchmark_overlaps)
    problem_overlap_ids = set(problem_overlap_by_val_row_id)
    train_val_overlapped_val_ids = sorted(set(row_id_overlaps) | problem_overlap_ids)
    train_val_overlap_count = len(train_val_overlapped_val_ids)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "pass" if benchmark_overlap_count == 0 and train_val_overlap_count == 0 else "fail",
        "benchmark": {
            "name": "GSM8K",
            "dataset": BENCHMARK_DATASET,
            "config": BENCHMARK_CONFIG,
            "split": BENCHMARK_SPLIT,
            "rows": len(benchmark),
        },
        "training_source_filter": {"source": "gsm8k", "compared_field": "problem"},
        "normalization_rule": "Unicode NFKC, lowercase, strip leading/trailing whitespace, collapse repeated internal whitespace, preserve punctuation and numeric literals",
        "inputs": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_gsm8k_rows_compared_to_benchmark": len(gsm8k_train_rows),
        },
        "overlap_count": benchmark_overlap_count,
        "overlapping_question_ids_or_hashes": benchmark_overlaps[:MAX_EXAMPLES],
        "overlapping_question_record_count": len(benchmark_overlaps),
        "train_val_overlap": {
            "row_id_overlap_count": len(row_id_overlaps),
            "row_id_overlap_examples": row_id_overlaps[:MAX_EXAMPLES],
            "problem_text_overlap_count": len(problem_overlap_by_val_row_id),
            "problem_text_overlap_examples": list(problem_overlap_by_val_row_id.values())[:MAX_EXAMPLES],
            "overlapped_val_row_count": train_val_overlap_count,
            "overlapped_val_row_id_examples": train_val_overlapped_val_ids[:MAX_EXAMPLES],
        },
        "pass_fail_outcome": {
            "benchmark_contamination": "pass" if benchmark_overlap_count == 0 else "fail",
            "train_val_overlap": "pass" if train_val_overlap_count == 0 else "fail",
        },
        "failure_rule": "If benchmark overlap is greater than 0, rebuild without contaminated rows or change the benchmark anchor before comparison proceeds. If train/val overlap is greater than 0, rebuild the split before training.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
