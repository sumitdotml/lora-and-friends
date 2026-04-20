#!/usr/bin/env python3
"""
Run lightweight quality checks over an OpenMath subset.

Usage:
  uv run python scripts/audit_openmath_quality.py
  uv run python scripts/audit_openmath_quality.py --input artifacts/subsets/openmath_original_clean/train.jsonl --output-dir artifacts/audits/openmath_original_clean_quality_train
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_INPUT = Path("artifacts/subsets/openmath_original_clean/train.jsonl")
DEFAULT_OUTPUT = Path("artifacts/audits/openmath_original_clean_quality_train")

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


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None


def normalize(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\tfrac", "\\frac")
    return text.strip()


def main() -> None:
    input_path, output_dir = parse_args()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    boxed_match = 0
    missing_box = 0
    mismatch_by_source = Counter()
    source_counts = Counter()
    suspicious_counts = Counter()
    suspicious_by_source: dict[str, Counter] = defaultdict(Counter)
    mismatch_examples: list[dict] = []
    suspicious_examples: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            source = row["source"]
            source_counts[source] += 1

            boxed = last_boxed(row["generated_solution"])
            if boxed is None:
                missing_box += 1
                mismatch_by_source[source] += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        {
                            "row_id": row["row_id"],
                            "source": source,
                            "expected_answer": row["expected_answer"],
                            "boxed_answer": None,
                        }
                    )
            elif normalize(boxed) == normalize(row["expected_answer"]):
                boxed_match += 1
            else:
                mismatch_by_source[source] += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        {
                            "row_id": row["row_id"],
                            "source": source,
                            "expected_answer": row["expected_answer"],
                            "boxed_answer": boxed,
                        }
                    )

            solution = row["generated_solution"].lower()
            matched_names = [name for name, pattern in SUSPICIOUS_PATTERNS.items() if pattern in solution]
            if matched_names:
                for name in matched_names:
                    suspicious_counts[name] += 1
                    suspicious_by_source[source][name] += 1
                if len(suspicious_examples) < 10:
                    suspicious_examples.append(
                        {
                            "row_id": row["row_id"],
                            "source": source,
                            "matched_patterns": matched_names,
                            "solution_tail": row["generated_solution"][-400:],
                        }
                    )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "rows": rows,
        "boxed_matches_expected": boxed_match,
        "boxed_match_rate": boxed_match / rows if rows else 0.0,
        "missing_box": missing_box,
        "boxed_mismatches": rows - boxed_match - missing_box,
        "source_counts": dict(source_counts),
        "mismatch_by_source": dict(mismatch_by_source),
        "suspicious_pattern_counts": dict(suspicious_counts),
        "suspicious_pattern_counts_by_source": {
            source: dict(counter) for source, counter in suspicious_by_source.items()
        },
        "mismatch_examples": mismatch_examples,
        "suspicious_examples": suspicious_examples,
    }

    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
