#!/usr/bin/env python3
"""Run the frozen small LR-selection sweep on Tinker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = (
    ROOT
    / "artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/train.jsonl"
)
VAL_PATH = (
    ROOT
    / "artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/val.jsonl"
)
RAW_MANIFEST_PATH = ROOT / "artifacts/raw_datasets/openmath_original_clean/manifest.json"
RENDERED_MANIFEST_PATH = (
    ROOT
    / "artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/manifest.json"
)
LORA_DEFAULTS_PATH = ROOT / "docs/freeze/lora_defaults.md"
RUN_PROTOCOL_PATH = ROOT / "docs/freeze/run_protocol.md"
RESULTS_SCHEMA_PATH = ROOT / "docs/freeze/results_schema.md"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/results"

MODEL_NAME = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3_disable_thinking"
SEED = 7
LORA_RANK = 8
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
TRAIN_ROWS = 5_000
VAL_ROWS = 500
VALIDATION_EVERY = 125
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
CONDITION_ORDER = ("attention_only", "all_layer")


@dataclass(frozen=True)
class ConditionConfig:
    """One LoRA adapter scope locked for the comparison."""

    train_attn: bool
    train_mlp: bool
    train_unembed: bool

    def as_json(self) -> dict[str, bool]:
        return {
            "train_attn": self.train_attn,
            "train_mlp": self.train_mlp,
            "train_unembed": self.train_unembed,
        }


@dataclass(frozen=True)
class RunSpec:
    """One condition/LR pair in the small selection sweep."""

    run_prefix: str
    condition: str
    learning_rate: float

    @property
    def run_id(self) -> str:
        return f"{self.run_prefix}-{self.condition}-lr-{lr_label(self.learning_rate)}"


CONDITIONS: dict[str, ConditionConfig] = {
    "attention_only": ConditionConfig(
        train_attn=True,
        train_mlp=False,
        train_unembed=False,
    ),
    "all_layer": ConditionConfig(
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen small LR-selection sweep."
    )
    parser.add_argument("--run-prefix", default="lr-select-001")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_ORDER,
        default=list(CONDITION_ORDER),
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=list(LEARNING_RATES),
    )
    parser.add_argument("--train-limit", type=int, default=TRAIN_ROWS)
    parser.add_argument("--val-limit", type=int, default=VAL_ROWS)
    parser.add_argument("--validation-every", type=int, default=VALIDATION_EVERY)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--ttl-seconds", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == limit:
                break
    if len(rows) < limit:
        raise ValueError(f"{path} only has {len(rows)} rows; needed {limit}")
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_state(run_prefix: str | None = None) -> dict[str, Any]:
    status_lines = git_value(["status", "--short"]).splitlines()
    ignored_lines = []
    if run_prefix is not None:
        ignored_prefix = f"?? artifacts/results/{run_prefix}"
        ignored_lines = [
            line for line in status_lines if line.startswith(ignored_prefix)
        ]
        status_lines = [
            line for line in status_lines if not line.startswith(ignored_prefix)
        ]
    return {
        "sha": git_value(["rev-parse", "HEAD"]),
        "dirty": bool(status_lines),
        "status_short": status_lines,
        "ignored_status_short": ignored_lines,
    }


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def lr_label(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("-0", "-")


def tinker_version_note() -> str:
    return (
        f"not exposed by tinker {package_version('tinker')} "
        "create_lora_training_client"
    )


def required_input_paths() -> list[Path]:
    return [
        TRAIN_PATH,
        VAL_PATH,
        RAW_MANIFEST_PATH,
        RENDERED_MANIFEST_PATH,
        LORA_DEFAULTS_PATH,
        RUN_PROTOCOL_PATH,
        RESULTS_SCHEMA_PATH,
    ]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    existing = [
        metrics_path,
        output_dir / "summary.json",
        output_dir / "manifest.json",
        output_dir / "failure.json",
        output_dir / "sample_render.txt",
    ]
    present = [path for path in existing if path.exists()]
    if present and not overwrite:
        names = ", ".join(str(path) for path in present)
        raise FileExistsError(f"output files already exist: {names}")
    if overwrite:
        for path in present:
            path.unlink()
    return metrics_path


def render_tokens(row: dict[str, Any], tokenizer: Any) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    return list(rendered["input_ids"])


def render_prompt_tokens(row: dict[str, Any], tokenizer: Any) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        row["messages"][:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )
    return list(rendered["input_ids"])


def render_text(row: dict[str, Any], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def build_datum(row: dict[str, Any], tokenizer: Any) -> Any:
    """Build one SFT datum with loss masked to assistant answer tokens."""

    import torch
    from tinker_cookbook.supervised.common import datum_from_tokens_weights

    tokens = render_tokens(row, tokenizer)
    prompt_len = len(render_prompt_tokens(row, tokenizer))
    if prompt_len >= len(tokens):
        raise ValueError(
            f"prompt length {prompt_len} leaves no assistant tokens for {row['row_id']}"
        )
    weights = torch.zeros(len(tokens), dtype=torch.float32)
    weights[prompt_len:] = 1.0
    return datum_from_tokens_weights(torch.tensor(tokens, dtype=torch.int64), weights)


def build_datums(rows: list[dict[str, Any]], tokenizer: Any) -> list[Any]:
    return [build_datum(row, tokenizer) for row in rows]


def answer_weight_count(datum: Any) -> float:
    weights = datum.loss_fn_inputs["weights"]
    if hasattr(weights, "sum"):
        return float(weights.sum().item())
    if hasattr(weights, "data"):
        return sum(float(value) for value in weights.data)
    raise TypeError(f"unsupported weight tensor type: {type(weights)!r}")


def datum_token_count(datum: Any) -> int:
    return int(datum.model_input.length)


def mean_nll(output: Any, data: list[Any]) -> float:
    from tinker_cookbook.supervised.common import compute_mean_nll

    logprobs = [x["logprobs"] for x in output.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in data]
    return float(compute_mean_nll(logprobs, weights))


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def metric_row(
    *,
    run_id: str,
    condition: str,
    learning_rate: float,
    step: int,
    split: str,
    loss: float | None,
    token_count: int,
    backend_metrics: dict[str, float] | None,
    eval_metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "checkpoint": None,
        "condition": condition,
        "seed": SEED,
        "step": step,
        "split": split,
        "loss": loss,
        "learning_rate": learning_rate,
        "eval_metric": eval_metric,
        "token_count": token_count,
        "cost": None,
        "backend_metrics": backend_metrics or {},
    }


async def create_training_client(
    service_client: Any,
    spec: RunSpec,
) -> Any:
    config = CONDITIONS[spec.condition]
    return await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME,
        rank=LORA_RANK,
        seed=SEED,
        train_attn=config.train_attn,
        train_mlp=config.train_mlp,
        train_unembed=config.train_unembed,
        user_metadata={
            "project": "lora-and-friends",
            "run_id": spec.run_id,
            "condition": spec.condition,
            "learning_rate": str(spec.learning_rate),
            "phase": "small_lr_selection",
        },
    )


async def run_train_batch(
    training_client: Any,
    batch: list[Any],
    *,
    spec: RunSpec,
    step: int,
    metrics_path: Path,
) -> dict[str, Any]:
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    backend_metrics: list[dict[str, float]] = []
    token_count = 0
    for datum in batch:
        future = await training_client.forward_backward_async(
            [datum], loss_fn="cross_entropy"
        )
        output = await future.result_async()
        loss = mean_nll(output, [datum])
        answer_tokens = answer_weight_count(datum)
        weighted_loss_sum += loss * answer_tokens
        weight_sum += answer_tokens
        token_count += datum_token_count(datum)
        backend_metrics.append(output.metrics)
    row = metric_row(
        run_id=spec.run_id,
        condition=spec.condition,
        learning_rate=spec.learning_rate,
        step=step,
        split="small_train",
        loss=weighted_loss_sum / weight_sum,
        token_count=token_count,
        backend_metrics=aggregate_metrics(backend_metrics),
    )
    append_jsonl(metrics_path, row)
    return row


async def run_optimizer_step(
    training_client: Any,
    *,
    spec: RunSpec,
    step: int,
    token_count: int,
    metrics_path: Path,
) -> dict[str, Any]:
    import tinker

    future = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=spec.learning_rate)
    )
    output = await future.result_async()
    row = metric_row(
        run_id=spec.run_id,
        condition=spec.condition,
        learning_rate=spec.learning_rate,
        step=step,
        split="small_optim",
        loss=None,
        token_count=token_count,
        backend_metrics=output.metrics,
    )
    append_jsonl(metrics_path, row)
    return row


async def run_validation(
    training_client: Any,
    val_datums: list[Any],
    *,
    spec: RunSpec,
    step: int,
    metrics_path: Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    backend_metrics: list[dict[str, float]] = []
    token_count = 0
    for start in range(0, len(val_datums), batch_size):
        chunk = val_datums[start : start + batch_size]
        future = await training_client.forward_async(chunk, loss_fn="cross_entropy")
        output = await future.result_async()
        loss = mean_nll(output, chunk)
        chunk_weight = sum(answer_weight_count(datum) for datum in chunk)
        weighted_loss_sum += loss * chunk_weight
        weight_sum += chunk_weight
        token_count += sum(datum_token_count(datum) for datum in chunk)
        backend_metrics.append(output.metrics)
    loss = weighted_loss_sum / weight_sum
    row = metric_row(
        run_id=spec.run_id,
        condition=spec.condition,
        learning_rate=spec.learning_rate,
        step=step,
        split="small_val",
        loss=loss,
        token_count=token_count,
        backend_metrics=aggregate_metrics(backend_metrics),
        eval_metric={"name": "validation_mean_nll", "value": loss},
    )
    append_jsonl(metrics_path, row)
    return row


async def save_checkpoint(
    training_client: Any,
    *,
    spec: RunSpec,
    ttl_seconds: int,
) -> dict[str, Any]:
    checkpoint_name = f"{spec.run_id}-final"
    future = await training_client.save_state_async(
        checkpoint_name, ttl_seconds=ttl_seconds
    )
    result = await future.result_async()
    return {
        "name": checkpoint_name,
        "path": result.path,
        "ttl_seconds": ttl_seconds,
    }


def lora_config_summary(condition: str) -> dict[str, Any]:
    config = CONDITIONS[condition]
    return {
        "rank": LORA_RANK,
        "lora_alpha": tinker_version_note(),
        "lora_dropout": tinker_version_note(),
        **config.as_json(),
    }


def protocol_mode(args: argparse.Namespace) -> str:
    frozen = (
        args.train_limit == TRAIN_ROWS
        and args.val_limit == VAL_ROWS
        and args.validation_every == VALIDATION_EVERY
        and tuple(args.conditions) == CONDITION_ORDER
        and tuple(args.learning_rates) == LEARNING_RATES
        and args.max_optimizer_steps is None
    )
    return "frozen_small_lr_selection" if frozen else "override"


def build_manifest(
    *,
    spec: RunSpec,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_steps = math.ceil(len(train_rows) / GRADIENT_ACCUMULATION)
    return {
        "run_id": spec.run_id,
        "status": "started",
        "created_at": now_iso(),
        "protocol_mode": protocol_mode(args),
        "git": git_state(spec.run_prefix),
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "package_versions": {
            "tinker": package_version("tinker"),
            "tinker-cookbook": package_version("tinker-cookbook"),
            "transformers": package_version("transformers"),
        },
        "contracts": {
            "lora_defaults": str(LORA_DEFAULTS_PATH.relative_to(ROOT)),
            "lora_defaults_sha256": sha256_file(LORA_DEFAULTS_PATH),
            "run_protocol": str(RUN_PROTOCOL_PATH.relative_to(ROOT)),
            "run_protocol_sha256": sha256_file(RUN_PROTOCOL_PATH),
            "results_schema": str(RESULTS_SCHEMA_PATH.relative_to(ROOT)),
            "results_schema_sha256": sha256_file(RESULTS_SCHEMA_PATH),
        },
        "dataset": {
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "val_path": str(VAL_PATH.relative_to(ROOT)),
            "raw_manifest_sha256": sha256_file(RAW_MANIFEST_PATH),
            "rendered_manifest_sha256": sha256_file(RENDERED_MANIFEST_PATH),
            "train_start_index": 0,
            "train_rows": len(train_rows),
            "train_row_ids": [row["row_id"] for row in train_rows],
            "val_start_index": 0,
            "val_rows": len(val_rows),
            "val_row_ids": [row["row_id"] for row in val_rows],
        },
        "training": {
            "seed": SEED,
            "learning_rate": spec.learning_rate,
            "condition": spec.condition,
            "lora_config": lora_config_summary(spec.condition),
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": GRADIENT_ACCUMULATION,
            "epoch_count": 1,
            "expected_optimizer_steps": total_steps,
            "max_optimizer_steps": args.max_optimizer_steps,
            "validation_every": args.validation_every,
        },
    }


def run_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        RunSpec(
            run_prefix=args.run_prefix,
            condition=condition,
            learning_rate=learning_rate,
        )
        for condition in args.conditions
        for learning_rate in args.learning_rates
    ]


async def run_one_spec(
    service_client: Any,
    spec: RunSpec,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = args.output_root / spec.run_id
    metrics_path = prepare_output_dir(output_dir, args.overwrite)
    manifest = build_manifest(
        spec=spec,
        args=args,
        train_rows=train_rows,
        val_rows=val_rows,
    )
    write_json(output_dir / "manifest.json", manifest)

    if args.dry_run:
        manifest["status"] = "dry_run_pass"
        manifest["finished_at"] = now_iso()
        write_json(output_dir / "manifest.json", manifest)
        return {"run_id": spec.run_id, "status": "dry_run_pass"}

    try:
        print(
            f"starting {spec.run_id}: condition={spec.condition} "
            f"lr={spec.learning_rate}",
            flush=True,
        )
        training_client = await create_training_client(service_client, spec)
        tokenizer = training_client.get_tokenizer()
        train_datums = build_datums(train_rows, tokenizer)
        val_datums = build_datums(val_rows, tokenizer)
        (output_dir / "sample_render.txt").write_text(
            render_text(train_rows[0], tokenizer) + "\n"
        )

        validation_rows = []
        train_token_count = 0
        optim_token_count = 0
        total_steps = math.ceil(len(train_datums) / GRADIENT_ACCUMULATION)
        if args.max_optimizer_steps is not None:
            total_steps = min(total_steps, args.max_optimizer_steps)

        for step in range(1, total_steps + 1):
            start = (step - 1) * GRADIENT_ACCUMULATION
            batch = train_datums[start : start + GRADIENT_ACCUMULATION]
            train_row = await run_train_batch(
                training_client,
                batch,
                spec=spec,
                step=step,
                metrics_path=metrics_path,
            )
            train_token_count += train_row["token_count"]
            optim_row = await run_optimizer_step(
                training_client,
                spec=spec,
                step=step,
                token_count=train_row["token_count"],
                metrics_path=metrics_path,
            )
            optim_token_count += optim_row["token_count"]
            should_validate = step % args.validation_every == 0 or step == total_steps
            if should_validate:
                val_row = await run_validation(
                    training_client,
                    val_datums,
                    spec=spec,
                    step=step,
                    metrics_path=metrics_path,
                )
                validation_rows.append(val_row)
                print(
                    f"{spec.run_id}: validation step={step} "
                    f"nll={val_row['eval_metric']['value']:.6f}",
                    flush=True,
                )

        checkpoint = await save_checkpoint(
            training_client,
            spec=spec,
            ttl_seconds=args.ttl_seconds,
        )
        best_validation = min(
            validation_rows,
            key=lambda row: row["eval_metric"]["value"],
        )
        summary = {
            "run_id": spec.run_id,
            "status": "pass",
            "finished_at": now_iso(),
            "checkpoint": checkpoint,
            "condition": spec.condition,
            "seed": SEED,
            "learning_rate": spec.learning_rate,
            "primary_metric": best_validation["eval_metric"],
            "best_validation_step": best_validation["step"],
            "token_count": {
                "train": train_token_count,
                "optimizer": optim_token_count,
                "validation": sum(row["token_count"] for row in validation_rows),
                "total": train_token_count
                + sum(row["token_count"] for row in validation_rows),
            },
            "artifact_paths": {
                "manifest": display_path(output_dir / "manifest.json"),
                "metrics": display_path(metrics_path),
                "summary": display_path(output_dir / "summary.json"),
                "sample_render": display_path(output_dir / "sample_render.txt"),
            },
        }
        write_json(output_dir / "summary.json", summary)
        manifest["status"] = "pass"
        manifest["finished_at"] = summary["finished_at"]
        write_json(output_dir / "manifest.json", manifest)
        return summary
    except Exception as exc:
        failure = {
            "run_id": spec.run_id,
            "status": "fail",
            "failed_at": now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "failure.json", failure)
        manifest["status"] = "fail"
        manifest["finished_at"] = failure["failed_at"]
        write_json(output_dir / "manifest.json", manifest)
        raise


async def run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    for path in required_input_paths():
        if not path.exists():
            raise FileNotFoundError(path)

    specs = run_specs(args)
    max_train_rows = args.train_limit
    max_val_rows = args.val_limit
    train_rows = read_jsonl(TRAIN_PATH, max_train_rows)
    val_rows = read_jsonl(VAL_PATH, max_val_rows)

    if args.dry_run:
        service_client = None
    else:
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY is not set in the environment or .env")
        import tinker

        service_client = tinker.ServiceClient()
        capabilities = await service_client.get_server_capabilities_async()
        supported_models = [model.model_name for model in capabilities.supported_models]
        if MODEL_NAME not in supported_models:
            raise RuntimeError(
                f"{MODEL_NAME} not in Tinker supported models: {supported_models}"
            )

    results = []
    for spec in specs:
        results.append(
            await run_one_spec(
                service_client,
                spec,
                args,
                train_rows,
                val_rows,
            )
        )
        print(json.dumps(results[-1], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
