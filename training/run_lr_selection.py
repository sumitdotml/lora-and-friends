#!/usr/bin/env python3
"""Run the frozen small LR-selection sweep."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_RESULTS_DIR,
    LORA_DEFAULTS_PATH,
    RAW_MANIFEST_PATH,
    RENDERED_MANIFEST_PATH,
    RESULTS_SCHEMA_PATH,
    ROOT,
    RUN_PROTOCOL_PATH,
    TRAIN_PATH,
    VAL_PATH,
    append_jsonl,
    display_path,
    git_state,
    load_dotenv,
    now_iso,
    package_version,
    prepare_output_files,
    read_jsonl,
    sha256_file,
    write_json,
)
from lora import (
    CONDITION_ORDER,
    LORA_RANK,
    MODEL_NAME,
    RENDERER_NAME,
    SEED,
    create_training_client,
    lora_config_summary,
    require_supported_model,
    save_checkpoint,
)
from sft import (
    aggregate_metrics,
    answer_weight_count,
    build_datums,
    datum_token_count,
    mean_nll,
    render_text,
)


DEFAULT_REQUEST_SHAPE = "single_datum_calls"
REQUEST_SHAPES = ("single_datum_calls", "batched_datums_pipelined")
DEFAULT_EFFECTIVE_BATCH_SIZE = 8
TRAIN_ROWS = 512
VAL_ROWS = 128
VALIDATION_EVERY = 32
LEARNING_RATES = (1e-4, 3e-4, 1e-3)


@dataclass(frozen=True)
class RunSpec:
    """One condition/LR pair in the small selection sweep."""

    run_prefix: str
    condition: str
    learning_rate: float

    @property
    def run_id(self) -> str:
        return f"{self.run_prefix}-{self.condition}-lr-{lr_label(self.learning_rate)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen small LR-selection sweep."
    )
    parser.add_argument("--run-prefix", default="lr-select-001")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_DIR)
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
    parser.add_argument(
        "--request-shape",
        choices=REQUEST_SHAPES,
        default=DEFAULT_REQUEST_SHAPE,
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=DEFAULT_EFFECTIVE_BATCH_SIZE,
    )
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--ttl-seconds", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def lr_label(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("-0", "-")


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
    prepare_output_files(
        output_dir,
        [
            "metrics.jsonl",
            "summary.json",
            "manifest.json",
            "failure.json",
            "sample_render.txt",
        ],
        overwrite=overwrite,
    )
    return output_dir / "metrics.jsonl"


def protocol_mode(args: argparse.Namespace) -> str:
    """Label whether this invocation matches the frozen LR-selection protocol."""

    frozen = (
        args.train_limit == TRAIN_ROWS
        and args.val_limit == VAL_ROWS
        and args.validation_every == VALIDATION_EVERY
        and args.request_shape == DEFAULT_REQUEST_SHAPE
        and args.effective_batch_size == DEFAULT_EFFECTIVE_BATCH_SIZE
        and tuple(args.conditions) == CONDITION_ORDER
        and tuple(args.learning_rates) == LEARNING_RATES
        and args.max_optimizer_steps is None
    )
    return "frozen_small_lr_selection" if frozen else "override"


def run_specs(args: argparse.Namespace) -> list[RunSpec]:
    """Expand CLI choices into the condition/LR runs that will execute."""

    return [
        RunSpec(
            run_prefix=args.run_prefix,
            condition=condition,
            learning_rate=learning_rate,
        )
        for condition in args.conditions
        for learning_rate in args.learning_rates
    ]


def metric_row(
    *,
    spec: RunSpec,
    step: int,
    split: str,
    loss: float | None,
    token_count: int,
    backend_metrics: dict[str, float] | None,
    eval_metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "checkpoint": None,
        "condition": spec.condition,
        "seed": SEED,
        "step": step,
        "split": split,
        "loss": loss,
        "learning_rate": spec.learning_rate,
        "eval_metric": eval_metric,
        "token_count": token_count,
        "cost": None,
        "backend_metrics": backend_metrics or {},
    }


async def run_train_batch(
    training_client: Any,
    batch: list[Any],
    *,
    spec: RunSpec,
    step: int,
    metrics_path: Path,
) -> dict[str, Any]:
    """Run one gradient-accumulation batch and record its weighted train NLL.

    Each datum is sent as a micro-batch of size 1. The optimizer step happens
    separately after all datums in this batch have contributed gradients.
    """

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
        spec=spec,
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
        spec=spec,
        step=step,
        split="small_optim",
        loss=None,
        token_count=token_count,
        backend_metrics=output.metrics,
    )
    append_jsonl(metrics_path, row)
    return row


async def run_pipelined_train_step(
    training_client: Any,
    batch: list[Any],
    *,
    spec: RunSpec,
    step: int,
    metrics_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one batched train/update step using Tinker's pipelined pattern."""

    import tinker

    train_future = await training_client.forward_backward_async(
        batch, loss_fn="cross_entropy"
    )
    optim_future = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=spec.learning_rate)
    )
    train_output = await train_future.result_async()
    optim_output = await optim_future.result_async()
    token_count = sum(datum_token_count(datum) for datum in batch)
    train_row = metric_row(
        spec=spec,
        step=step,
        split="small_train",
        loss=mean_nll(train_output, batch),
        token_count=token_count,
        backend_metrics=train_output.metrics,
    )
    optim_row = metric_row(
        spec=spec,
        step=step,
        split="small_optim",
        loss=None,
        token_count=token_count,
        backend_metrics=optim_output.metrics,
    )
    append_jsonl(metrics_path, train_row)
    append_jsonl(metrics_path, optim_row)
    return train_row, optim_row


async def run_validation(
    training_client: Any,
    val_datums: list[Any],
    *,
    spec: RunSpec,
    step: int,
    metrics_path: Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate the fixed validation slice without applying gradients.

    The final validation metric is weighted by answer-token count, so longer
    answers contribute proportionally more token-level evidence than shorter
    answers.
    """

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
        spec=spec,
        step=step,
        split="small_val",
        loss=loss,
        token_count=token_count,
        backend_metrics=aggregate_metrics(backend_metrics),
        eval_metric={"name": "validation_mean_nll", "value": loss},
    )
    append_jsonl(metrics_path, row)
    return row


def build_manifest(
    *,
    spec: RunSpec,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the reproducibility manifest before the run mutates model state."""

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
            "rank": LORA_RANK,
            "request_shape": args.request_shape,
            "datums_per_forward_backward": (
                1
                if args.request_shape == "single_datum_calls"
                else args.effective_batch_size
            ),
            "forward_backward_calls_per_optimizer_step": (
                args.effective_batch_size
                if args.request_shape == "single_datum_calls"
                else 1
            ),
            "effective_batch_size": args.effective_batch_size,
            "epoch_count": 1,
            "expected_optimizer_steps": math.ceil(
                len(train_rows) / args.effective_batch_size
            ),
            "max_optimizer_steps": args.max_optimizer_steps,
            "validation_every": args.validation_every,
        },
    }


async def run_one_spec(
    service_client: Any,
    spec: RunSpec,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute one condition/LR pair and retain summary or failure artifacts."""

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
        training_client = await create_training_client(
            service_client,
            condition=spec.condition,
            run_id=spec.run_id,
            extra_metadata={
                "learning_rate": str(spec.learning_rate),
                "phase": "small_lr_selection",
            },
        )
        tokenizer = training_client.get_tokenizer()
        train_datums = build_datums(train_rows, tokenizer)
        val_datums = build_datums(val_rows, tokenizer)
        (output_dir / "sample_render.txt").write_text(
            render_text(train_rows[0], tokenizer) + "\n"
        )

        validation_rows = []
        train_token_count = 0
        optim_token_count = 0
        total_steps = math.ceil(len(train_datums) / args.effective_batch_size)
        if args.max_optimizer_steps is not None:
            total_steps = min(total_steps, args.max_optimizer_steps)

        for step in range(1, total_steps + 1):
            start = (step - 1) * args.effective_batch_size
            batch = train_datums[start : start + args.effective_batch_size]
            if args.request_shape == "batched_datums_pipelined":
                train_row, optim_row = await run_pipelined_train_step(
                    training_client,
                    batch,
                    spec=spec,
                    step=step,
                    metrics_path=metrics_path,
                )
            else:
                train_row = await run_train_batch(
                    training_client,
                    batch,
                    spec=spec,
                    step=step,
                    metrics_path=metrics_path,
                )
                optim_row = await run_optimizer_step(
                    training_client,
                    spec=spec,
                    step=step,
                    token_count=train_row["token_count"],
                    metrics_path=metrics_path,
                )
            train_token_count += train_row["token_count"]
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
            checkpoint_name=f"{spec.run_id}-final",
            ttl_seconds=args.ttl_seconds,
        )
        best_validation = min(
            validation_rows,
            key=lambda row: row["eval_metric"]["value"],
        )
        validation_token_count = sum(row["token_count"] for row in validation_rows)
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
                "validation": validation_token_count,
                "total": train_token_count + validation_token_count,
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
    load_dotenv()
    for path in required_input_paths():
        if not path.exists():
            raise FileNotFoundError(path)

    specs = run_specs(args)
    train_rows = read_jsonl(TRAIN_PATH, args.train_limit)
    val_rows = read_jsonl(VAL_PATH, args.val_limit)

    if args.dry_run:
        service_client = None
    else:
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY is not set in the environment or .env")
        import tinker

        service_client = tinker.ServiceClient()
        await require_supported_model(service_client)

    for spec in specs:
        result = await run_one_spec(
            service_client,
            spec,
            args,
            train_rows,
            val_rows,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
