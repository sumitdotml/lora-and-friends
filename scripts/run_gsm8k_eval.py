#!/usr/bin/env python3
"""Evaluate Qwen3 checkpoints on GSM8K under the frozen project contract.

Usage:
  uv run scripts/run_gsm8k_eval.py --self-test
  uv run scripts/run_gsm8k_eval.py --limit 10
  uv run scripts/run_gsm8k_eval.py \
      --checkpoint-path tinker://run-id/sampler_weights/step-3169 \
      --condition attention_only --seed 0 --limit 10
  uv run scripts/run_gsm8k_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_CONDITION = "base"
DEFAULT_SPLIT = "gsm8k_test"
SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "Put the final answer in \\boxed{}."
)
MAX_NEW_TOKENS = 512
BOXED_LOCATOR = re.compile(r"\\boxed\s*\{")
STEP_LOCATOR = re.compile(r"\bstep-(\d+)\b")


@dataclass(frozen=True)
class EvalExample:
    """One GSM8K test example after extracting the reference answer."""

    index: int
    question: str
    reference_answer: str

    @property
    def example_id(self) -> str:
        return f"gsm8k_test_{self.index:04d}"


@dataclass(frozen=True)
class Prediction:
    """One model completion and its deterministic score."""

    example: EvalExample
    prompt_token_count: int
    generated_token_count: int
    generated_text: str
    stop_reason: str | None
    extracted_answer: str | None
    normalized_prediction: str | None
    normalized_reference: str
    correct: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example.example_id,
            "benchmark_index": self.example.index,
            "question": self.example.question,
            "reference_answer": self.example.reference_answer,
            "generated_text": self.generated_text,
            "extracted_answer": self.extracted_answer,
            "normalized_prediction": self.normalized_prediction,
            "normalized_reference": self.normalized_reference,
            "correct": self.correct,
            "prompt_token_count": self.prompt_token_count,
            "generated_token_count": self.generated_token_count,
            "stop_reason": self.stop_reason,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--checkpoint-path",
        help=(
            "Tinker sampler checkpoint path to evaluate "
            "(e.g., tinker://run-id/sampler_weights/final)."
        ),
    )
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument(
        "--seed",
        type=int,
        help="Explicit seed label for metrics/summary; required when --checkpoint-path is set.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def run_id_for(model: str, limit: int | None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = f"-limit-{limit}" if limit is not None else ""
    target_name = target_slug(model)
    prefix = "checkpoint" if model.startswith("tinker://") else "baseline"
    return f"{prefix}-{target_name}-gsm8k-{timestamp}{suffix}"


def eval_target(args: argparse.Namespace) -> str:
    return args.checkpoint_path if args.checkpoint_path is not None else args.model


def target_slug(target: str) -> str:
    """Build a stable, readable slug from a model id or tinker checkpoint path."""

    if target.startswith("tinker://"):
        body = target[len("tinker://") :]
        # this ensures the run-id plus checkpoint identifier to avoid collisions like ".../final".
        if "/" in body:
            run_id, rest = body.split("/", 1)
            label = f"{run_id}-{rest}"
        else:
            label = body
    else:
        label = target.rsplit("/", 1)[-1]
    lowered = label.lower().replace("_", "-").replace(":", "-")
    normalized = re.sub(r"[^a-z0-9-]+", "-", lowered)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized[:80] if normalized else "target"


def parse_step_from_target(target: str) -> int | None:
    """Parse step only when a literal step-<digits> token is present."""

    match = STEP_LOCATOR.search(target)
    if match is None:
        return None
    return int(match.group(1))


def resolved_step(args: argparse.Namespace) -> int | None:
    return parse_step_from_target(eval_target(args))


def validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint_path is not None and args.condition == DEFAULT_CONDITION:
        raise ValueError(
            "--condition is required when --checkpoint-path is set; "
            f"default '{DEFAULT_CONDITION}' is baseline-only."
        )
    if args.checkpoint_path is not None and args.seed is None:
        raise ValueError(
            "--seed is required when --checkpoint-path is set; "
            "record the training seed that produced this checkpoint."
        )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / "predictions.jsonl",
        output_dir / "metrics.jsonl",
        output_dir / "summary.json",
        output_dir / "failure.json",
    ]
    present = [path for path in existing if path.exists()]
    if present and not overwrite:
        names = ", ".join(str(path) for path in present)
        raise FileExistsError(f"output files already exist: {names}")
    if overwrite:
        for path in present:
            path.unlink()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n")


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_state() -> dict[str, Any]:
    status = git_value(["status", "--short"])
    return {
        "sha": git_value(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_reference_answer(answer: str) -> str:
    if "####" not in answer:
        raise ValueError(f"GSM8K answer is missing final marker: {answer!r}")
    return answer.rsplit("####", 1)[1].strip()


def load_gsm8k_examples(limit: int | None) -> list[EvalExample]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    examples: list[EvalExample] = []
    for index, row in enumerate(dataset):
        examples.append(
            EvalExample(
                index=index,
                question=row["question"],
                reference_answer=extract_reference_answer(row["answer"]),
            )
        )
        if limit is not None and len(examples) == limit:
            break
    return examples


def messages_for(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def render_prompt_tokens(tokenizer: Any, question: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages_for(question),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )
    return list(rendered["input_ids"])


def extract_final_boxed(text: str) -> str | None:
    """Return the payload of the final boxed expression, preserving nesting."""

    matches = list(BOXED_LOCATOR.finditer(text))
    if not matches:
        return None

    open_brace = matches[-1].end() - 1
    depth = 0
    for idx in range(open_brace, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : idx]
    return None


def strip_math_wrappers(text: str) -> str:
    pairs = [("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]
    stripped = text.strip()
    for left, right in pairs:
        if stripped.startswith(left) and stripped.endswith(right):
            return stripped[len(left) : len(stripped) - len(right)].strip()
    return stripped


def numeric_value(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def normalize_answer(text: str) -> str:
    stripped = strip_math_wrappers(text)
    stripped = stripped.replace(r"\left", "").replace(r"\right", "")
    stripped = stripped.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    return " ".join(stripped.split())


def answers_match(
    prediction: str | None, reference: str
) -> tuple[bool, str | None, str]:
    normalized_reference = normalize_answer(reference)
    if prediction is None:
        return False, None, normalized_reference

    normalized_prediction = normalize_answer(prediction)
    pred_number = numeric_value(normalized_prediction)
    ref_number = numeric_value(normalized_reference)
    if pred_number is not None and ref_number is not None:
        return pred_number == ref_number, normalized_prediction, normalized_reference

    return (
        normalized_prediction == normalized_reference,
        normalized_prediction,
        normalized_reference,
    )


def score_completion(
    example: EvalExample,
    generated_text: str,
    prompt_tokens: int,
    generated_tokens: int,
    stop_reason: str | None,
) -> Prediction:
    extracted = extract_final_boxed(generated_text)
    correct, normalized_prediction, normalized_reference = answers_match(
        extracted, example.reference_answer
    )
    return Prediction(
        example=example,
        prompt_token_count=prompt_tokens,
        generated_token_count=generated_tokens,
        generated_text=generated_text,
        stop_reason=stop_reason,
        extracted_answer=extracted,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        correct=correct,
    )


async def sample_one(
    sampling_client: Any,
    tokenizer: Any,
    example: EvalExample,
    sampling_params: Any,
) -> Prediction:
    import tinker

    prompt_tokens = render_prompt_tokens(tokenizer, example.question)
    prompt = tinker.types.ModelInput.from_ints(prompt_tokens)
    result = await sampling_client.sample_async(
        prompt=prompt,
        sampling_params=sampling_params,
        num_samples=1,
    )
    sequence = result.sequences[0]
    generated_text = tokenizer.decode(sequence.tokens)
    stop_reason = (
        str(sequence.stop_reason) if sequence.stop_reason is not None else None
    )
    return score_completion(
        example=example,
        generated_text=generated_text,
        prompt_tokens=len(prompt_tokens),
        generated_tokens=len(sequence.tokens),
        stop_reason=stop_reason,
    )


async def run_predictions(
    args: argparse.Namespace,
    examples: list[EvalExample],
) -> list[Prediction]:
    import tinker

    service_client = tinker.ServiceClient()
    if args.checkpoint_path is not None:
        sampling_client = await service_client.create_sampling_client_async(
            model_path=args.checkpoint_path
        )
    else:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=args.model
        )
    tokenizer = sampling_client.get_tokenizer()
    sampling_params = tinker.types.SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )

    predictions: list[Prediction] = []
    concurrency = max(1, args.concurrency)
    for start in range(0, len(examples), concurrency):
        batch = examples[start : start + concurrency]
        batch_predictions = await asyncio.gather(
            *[
                sample_one(sampling_client, tokenizer, example, sampling_params)
                for example in batch
            ]
        )
        predictions.extend(batch_predictions)
        print(f"evaluated {len(predictions)} / {len(examples)}")
    return predictions


def metric_row(
    run_id: str,
    args: argparse.Namespace,
    predictions: list[Prediction],
) -> dict[str, Any]:
    correct = sum(prediction.correct for prediction in predictions)
    total = len(predictions)
    accuracy = correct / total if total else 0.0
    return {
        "run_id": run_id,
        "checkpoint": eval_target(args),
        "condition": args.condition,
        "seed": args.seed,
        "step": resolved_step(args),
        "split": DEFAULT_SPLIT,
        "loss": None,
        "eval_metric": {"name": "gsm8k_accuracy", "value": accuracy},
        "token_count": sum(
            p.prompt_token_count + p.generated_token_count for p in predictions
        ),
        "cost": None,
    }


def summary(
    run_id: str,
    args: argparse.Namespace,
    output_dir: Path,
    predictions: list[Prediction],
) -> dict[str, Any]:
    correct = sum(prediction.correct for prediction in predictions)
    total = len(predictions)
    accuracy = correct / total if total else 0.0
    extraction_failures = sum(
        1 for prediction in predictions if prediction.extracted_answer is None
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": eval_target(args),
        "condition": args.condition,
        "seed": args.seed,
        "step": resolved_step(args),
        "dataset": {
            "name": "GSM8K",
            "source": "openai/gsm8k",
            "config": "main",
            "split": "test",
            "examples": total,
        },
        "primary_metric": {
            "name": "gsm8k_accuracy",
            "value": accuracy,
            "correct": correct,
            "total": total,
        },
        "extraction_failures": extraction_failures,
        "token_count": {
            "prompt": sum(p.prompt_token_count for p in predictions),
            "generated": sum(p.generated_token_count for p in predictions),
            "total": sum(
                p.prompt_token_count + p.generated_token_count for p in predictions
            ),
        },
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "top_p": None,
            "stop": None,
            "enable_thinking": False,
        },
        "artifact_paths": {
            "metrics": str((output_dir / "metrics.jsonl").relative_to(ROOT)),
            "predictions": str((output_dir / "predictions.jsonl").relative_to(ROOT)),
            "summary": str((output_dir / "summary.json").relative_to(ROOT)),
        },
        "versions": {
            "tinker": package_version("tinker"),
            "tinker_cookbook": package_version("tinker-cookbook"),
        },
        "git": git_state(),
        "contracts": {
            "eval_contract": str(Path("docs/freeze/eval_contract.md")),
            "results_schema": str(Path("docs/freeze/results_schema.md")),
            "eval_contract_sha256": sha256_file(ROOT / "docs/freeze/eval_contract.md"),
            "results_schema_sha256": sha256_file(
                ROOT / "docs/freeze/results_schema.md"
            ),
        },
    }


def write_artifacts(
    run_id: str,
    args: argparse.Namespace,
    output_dir: Path,
    predictions: list[Prediction],
) -> None:
    for prediction in predictions:
        append_jsonl(output_dir / "predictions.jsonl", prediction.as_json())
    append_jsonl(output_dir / "metrics.jsonl", metric_row(run_id, args, predictions))
    write_json(
        output_dir / "summary.json", summary(run_id, args, output_dir, predictions)
    )


def run_self_test() -> None:
    example = EvalExample(
        index=0,
        question="What is 20 + 22?",
        reference_answer="42",
    )
    assert (
        extract_final_boxed(r"first \boxed{1} final \boxed{\frac{2}{3}}")
        == r"\frac{2}{3}"
    )
    assert extract_final_boxed(r"nested \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_final_boxed("no boxed answer") is None
    assert answers_match("42.0", "42")[0]
    assert not answers_match(None, "42")[0]
    assert parse_step_from_target("tinker://run/sampler_weights/step-3169") == 3169
    assert parse_step_from_target("tinker://run/sampler_weights/final") is None
    scored = score_completion(
        example=example,
        generated_text=r"Reasoning here. Final answer: \boxed{42.0}",
        prompt_tokens=10,
        generated_tokens=8,
        stop_reason="max_tokens",
    )
    assert scored.correct
    print("self-test passed")


async def async_main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    load_dotenv(ROOT / ".env")
    validate_args(args)
    run_id = args.run_id or run_id_for(eval_target(args), args.limit)
    output_dir = args.output_dir or ROOT / "artifacts/results" / run_id
    prepare_output_dir(output_dir, args.overwrite)

    examples = load_gsm8k_examples(args.limit)
    predictions = await run_predictions(args, examples)
    write_artifacts(run_id, args, output_dir, predictions)
    print(json.dumps(summary(run_id, args, output_dir, predictions), indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
