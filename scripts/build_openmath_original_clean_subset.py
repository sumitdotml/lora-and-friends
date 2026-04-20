#!/usr/bin/env python3
"""
Build a strict original-only OpenMath subset from the `gsm8k` and `math` sources in
OpenMathInstruct-2 `train_1M`, using the shared quality gate from the curation scripts.

Usage:
  uv run --with datasets python scripts/build_openmath_original_clean_subset.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "nvidia/OpenMathInstruct-2"
DATASET_SPLIT = "train_1M"
SEED = 20260420
SOURCES = ("gsm8k", "math")
VAL_RATIO = 0.10

RAW_OUTPUT_DIR = Path("artifacts/subsets/openmath_original_clean")
CHAT_OUTPUT_DIR = Path("artifacts/datasets/openmath_original_clean_qwen3_disable_thinking")
SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "Put the final answer in \\boxed{}."
)
MODEL_NAME = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3_disable_thinking"
ANSWER_STYLE = "boxed"

SUSPICIOUS_PATTERNS = {
    "cannot_spend_more_than_have": "cannot spend more than",
    "cannot_be_accurately_completed": "cannot be accurately completed",
    "cannot_be_accurately_provided": "cannot be accurately provided",
    "cannot_directly_calculate_or_deduce": "cannot directly calculate or deduce",
    "does_not_align_logical_outcome": "doesn't align with the logical outcome",
    "i_must_provide_an_answer": "i must provide an answer",
    "need_numerical_or_graphical_methods": "need for numerical or graphical methods",
    "without_further_clarification": "without further clarification",
    "without_further_specific_steps": "without further specific steps",
    "acknowledging_complexity": "acknowledging the complexity",
    "more_sophisticated_mathematical_approach": "more sophisticated mathematical approach",
    "set_value_to_100": "set the value to 100",
    "strictly_interpreting": "strictly interpreting",
    "technically_not_have_money": "would technically not have any money left",
    "must_be_integer": "must be an integer",
    "seems_to_be_looking_for": "seems to be looking for",
}

EXCLUDE_PROBLEM_REGEXES = (
    re.compile(r"^\s*problem\s*:", re.IGNORECASE),
    re.compile(r"\bnew problem\s*:", re.IGNORECASE),
    re.compile(r"\bthe new problem is\s*:", re.IGNORECASE),
    re.compile(r"\bhere is a new problem\s*:", re.IGNORECASE),
    re.compile(r"\bwrite another problem\b", re.IGNORECASE),
    re.compile(r"\banother problem inspired by this one\b", re.IGNORECASE),
    re.compile(r"\binspired by this one\b", re.IGNORECASE),
    re.compile(r"\bconsider the next problem\b", re.IGNORECASE),
    re.compile(r"\blet the students\b", re.IGNORECASE),
    re.compile(r"\bsolve this problem independently\b", re.IGNORECASE),
    re.compile(r"\bcreate a new problem\b", re.IGNORECASE),
    re.compile(r"\breasoning skill for creating a new problem\b", re.IGNORECASE),
    re.compile(r"\blet's modify it to create a new problem\b", re.IGNORECASE),
    re.compile(r"\bthis question is identical to the previous one\b", re.IGNORECASE),
    re.compile(r"\bnote: do not go over the length\b", re.IGNORECASE),
    re.compile(r"\bno similar problem needed\b", re.IGNORECASE),
)


@dataclass
class Row:
    row_id: str
    source: str
    problem: str
    generated_solution: str
    expected_answer: str
    stream_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "source": self.source,
            "problem": self.problem,
            "generated_solution": self.generated_solution,
            "expected_answer": self.expected_answer,
        }

    def to_chat_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "source": self.source,
            "expected_answer": self.expected_answer,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.problem},
                {"role": "assistant", "content": self.generated_solution},
            ],
        }
        

def stable_row_id(source: str, problem: str, expected_answer: str) -> str:
    payload = f"{source}\n{problem}\n{expected_answer}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


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


def problem_has_contamination(problem: str) -> bool:
    return any(pattern.search(problem) for pattern in EXCLUDE_PROBLEM_REGEXES)


def candidate_ok(row: dict) -> tuple[bool, str | None]:
    solution = row["generated_solution"]
    boxed = last_boxed(solution)
    if boxed is None:
        return False, "missing_box"
    if normalize(boxed) != normalize(row["expected_answer"]):
        return False, "boxed_mismatch"

    solution_lower = solution.lower()
    if any(pattern in solution_lower for pattern in SUSPICIOUS_PATTERNS.values()):
        return False, "suspicious_pattern"

    if problem_has_contamination(row["problem"]):
        return False, "problem_pattern"

    return True, None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kept_rows: dict[str, list[Row]] = defaultdict(list)
    reject_reasons: Counter[tuple[str, str]] = Counter()
    total_counts = Counter()

    stream = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    for idx, raw in enumerate(stream):
        source = raw["problem_source"]
        if source not in SOURCES:
            continue
        total_counts[source] += 1
        candidate = {
            "row_id": stable_row_id(source, raw["problem"], raw["expected_answer"]),
            "source": source,
            "problem": raw["problem"],
            "generated_solution": raw["generated_solution"],
            "expected_answer": raw["expected_answer"],
        }
        ok, reason = candidate_ok(candidate)
        if not ok:
            reject_reasons[(source, reason or "unknown")] += 1
            continue
        kept_rows[source].append(
            Row(
                row_id=candidate["row_id"],
                source=source,
                problem=candidate["problem"],
                generated_solution=candidate["generated_solution"],
                expected_answer=candidate["expected_answer"],
                stream_index=idx,
            )
        )

    rng = random.Random(SEED)
    train_rows: list[Row] = []
    val_rows: list[Row] = []
    split_counts = {"train": {}, "val": {}}

    for source in SOURCES:
        rows = kept_rows[source][:]
        rng.shuffle(rows)
        val_count = round(len(rows) * VAL_RATIO)
        val_chunk = rows[:val_count]
        train_chunk = rows[val_count:]
        train_rows.extend(train_chunk)
        val_rows.extend(val_chunk)
        split_counts["train"][source] = len(train_chunk)
        split_counts["val"][source] = len(val_chunk)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    write_jsonl(RAW_OUTPUT_DIR / "train.jsonl", [row.to_dict() for row in train_rows])
    write_jsonl(RAW_OUTPUT_DIR / "val.jsonl", [row.to_dict() for row in val_rows])
    write_jsonl(CHAT_OUTPUT_DIR / "train.jsonl", [row.to_chat_dict() for row in train_rows])
    write_jsonl(CHAT_OUTPUT_DIR / "val.jsonl", [row.to_chat_dict() for row in val_rows])

    raw_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "seed": SEED,
        "sources": list(SOURCES),
        "val_ratio": VAL_RATIO,
        "input_source_counts": dict(total_counts),
        "accepted_source_counts": {source: len(kept_rows[source]) for source in SOURCES},
        "split_source_counts": split_counts,
        "reject_reasons": {f"{source}:{reason}": count for (source, reason), count in reject_reasons.items()},
        "artifacts": {
            "train_jsonl": str(RAW_OUTPUT_DIR / "train.jsonl"),
            "val_jsonl": str(RAW_OUTPUT_DIR / "val.jsonl"),
        },
    }
    chat_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_dir": str(RAW_OUTPUT_DIR),
        "output_dir": str(CHAT_OUTPUT_DIR),
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "answer_style": ANSWER_STYLE,
        "splits": {
            "train": {"rows": len(train_rows)},
            "val": {"rows": len(val_rows)},
        },
        "artifacts": {
            "train_jsonl": str(CHAT_OUTPUT_DIR / "train.jsonl"),
            "val_jsonl": str(CHAT_OUTPUT_DIR / "val.jsonl"),
        },
    }

    (RAW_OUTPUT_DIR / "manifest.json").write_text(json.dumps(raw_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (CHAT_OUTPUT_DIR / "manifest.json").write_text(json.dumps(chat_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "raw_output_dir": str(RAW_OUTPUT_DIR),
                "chat_output_dir": str(CHAT_OUTPUT_DIR),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_source_counts": split_counts["train"],
                "val_source_counts": split_counts["val"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
