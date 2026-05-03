#!/usr/bin/env python3
"""
Compare Qwen3 rendering with and without the current system prompt on the frozen dataset.

Usage:
  uv run python scripts/check_qwen3_render_sanity.py
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "Put the final answer in \\boxed{}."
)

RAW_INPUT_DIR = Path("artifacts/raw_datasets/openmath_original_clean")
OUTPUT_DIR = Path("artifacts/audits/openmath_original_clean_render_sanity")
SAMPLE_ROW_IDS = [
    "d15265ccf80f2ef19132",
    "25147e6a1a089450a310",
    "a4651ec4400479e670bd",
]


@dataclass
class Row:
    row_id: str
    source: str
    problem: str
    generated_solution: str
    expected_answer: str


def load_rows(path: Path) -> list[Row]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            rows.append(
                Row(
                    row_id=raw["row_id"],
                    source=raw["source"],
                    problem=raw["problem"],
                    generated_solution=raw["generated_solution"],
                    expected_answer=raw["expected_answer"],
                )
            )
    return rows


def render_text(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def render_token_count(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return len(rendered["input_ids"])


def messages_with_system(row: Row) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row.problem},
        {"role": "assistant", "content": row.generated_solution},
    ]


def messages_without_system(row: Row) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": row.problem},
        {"role": "assistant", "content": row.generated_solution},
    ]


def summarize_split(tokenizer, rows: list[Row]) -> dict:
    with_system_full = []
    without_system_full = []
    with_system_prompt = []
    without_system_prompt = []

    for row in rows:
        full_with_system = render_token_count(tokenizer, messages_with_system(row))
        full_without_system = render_token_count(tokenizer, messages_without_system(row))
        prompt_with_system = render_token_count(
            tokenizer,
            messages_with_system(row)[:-1],
            add_generation_prompt=True,
        )
        prompt_without_system = render_token_count(
            tokenizer,
            messages_without_system(row)[:-1],
            add_generation_prompt=True,
        )
        with_system_full.append(full_with_system)
        without_system_full.append(full_without_system)
        with_system_prompt.append(prompt_with_system)
        without_system_prompt.append(prompt_without_system)

    full_delta = [a - b for a, b in zip(with_system_full, without_system_full, strict=True)]
    prompt_delta = [a - b for a, b in zip(with_system_prompt, without_system_prompt, strict=True)]

    return {
        "rows": len(rows),
        "full_tokens_with_system": sum(with_system_full),
        "full_tokens_without_system": sum(without_system_full),
        "full_token_delta": sum(full_delta),
        "full_mean_with_system": round(sum(with_system_full) / len(rows), 2),
        "full_mean_without_system": round(sum(without_system_full) / len(rows), 2),
        "full_mean_delta": round(sum(full_delta) / len(rows), 2),
        "full_median_delta": statistics.median(full_delta),
        "prompt_tokens_with_system": sum(with_system_prompt),
        "prompt_tokens_without_system": sum(without_system_prompt),
        "prompt_token_delta": sum(prompt_delta),
        "prompt_mean_with_system": round(sum(with_system_prompt) / len(rows), 2),
        "prompt_mean_without_system": round(sum(without_system_prompt) / len(rows), 2),
        "prompt_mean_delta": round(sum(prompt_delta) / len(rows), 2),
        "prompt_median_delta": statistics.median(prompt_delta),
        "delta_range": {
            "full_min": min(full_delta),
            "full_max": max(full_delta),
            "prompt_min": min(prompt_delta),
            "prompt_max": max(prompt_delta),
        },
    }


def write_samples(tokenizer, rows: list[Row], output_path: Path) -> None:
    selected = {row.row_id: row for row in rows if row.row_id in SAMPLE_ROW_IDS}
    ordered_rows = [selected[row_id] for row_id in SAMPLE_ROW_IDS if row_id in selected]

    with output_path.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            rendered_with_system = render_text(tokenizer, messages_with_system(row))
            rendered_without_system = render_text(tokenizer, messages_without_system(row))
            f.write(f"## {row.row_id} {row.source}\n")
            f.write(f"expected_answer: {row.expected_answer}\n\n")
            f.write("### WITH SYSTEM\n")
            f.write(rendered_with_system)
            f.write("\n\n### WITHOUT SYSTEM\n")
            f.write(rendered_without_system)
            f.write("\n\n" + "=" * 100 + "\n\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    train_rows = load_rows(RAW_INPUT_DIR / "train.jsonl")
    val_rows = load_rows(RAW_INPUT_DIR / "val.jsonl")

    train_summary = summarize_split(tokenizer, train_rows)
    val_summary = summarize_split(tokenizer, val_rows)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": MODEL_NAME,
        "dataset_dir": str(RAW_INPUT_DIR),
        "system_prompt": SYSTEM_PROMPT,
        "decision_hint": (
            "keep_system_prompt_if_overhead_is_small_and_the boxed-answer/style"
            " instruction is still needed at eval time"
        ),
        "train": train_summary,
        "val": val_summary,
        "artifacts": {
            "sample_renders": str(OUTPUT_DIR / "sample_renders.txt"),
        },
    }

    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_samples(tokenizer, train_rows + val_rows, OUTPUT_DIR / "sample_renders.txt")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
