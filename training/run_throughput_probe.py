#!/usr/bin/env python3
"""Compare Tinker training request shapes before the main run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker import AdamParams, Datum, ServiceClient, TrainingClient

from common import (
    DEFAULT_RESULTS_DIR,
    LORA_DEFAULTS_PATH,
    RAW_MANIFEST_PATH,
    RENDERED_MANIFEST_PATH,
    RESULTS_SCHEMA_PATH,
    RUN_PROTOCOL_PATH,
    TRAIN_PATH,
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
)
from sft import (
    aggregate_metrics,
    answer_weight_count,
    build_datums,
    datum_token_count,
    mean_nll,
    render_text,
)


DEFAULT_RUN_ID = "throughput-probe-001"
DEFAULT_OPTIMIZER_STEPS = 16
DEFAULT_EFFECTIVE_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 3e-4
# naming local probe modes, not SDK symbols
REQUEST_SHAPES = ("single_datum_calls", "batched_datums", "batched_datums_pipelined")


@dataclass(frozen=True)
class RequestShape:
    """One local runner request shape tested by the throughput probe."""

    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Tinker forward/backward request shapes."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--condition",
        choices=CONDITION_ORDER,
        default="attention_only",
        help="LoRA condition used for the probe.",
    )
    parser.add_argument(
        "--request-shapes",
        nargs="+",
        choices=REQUEST_SHAPES,
        default=list(REQUEST_SHAPES),
    )
    parser.add_argument("--optimizer-steps", type=int, default=DEFAULT_OPTIMIZER_STEPS)
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=DEFAULT_EFFECTIVE_BATCH_SIZE,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def required_input_paths() -> list[Path]:
    return [
        TRAIN_PATH,
        RAW_MANIFEST_PATH,
        RENDERED_MANIFEST_PATH,
        LORA_DEFAULTS_PATH,
        RUN_PROTOCOL_PATH,
        RESULTS_SCHEMA_PATH,
    ]


def validate_args(args: argparse.Namespace) -> None:
    if args.optimizer_steps < 1:
        raise ValueError("--optimizer-steps must be at least 1")
    if args.effective_batch_size < 1:
        raise ValueError("--effective-batch-size must be at least 1")


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


def build_manifest(
    *,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the retained manifest before the probe contacts Tinker."""

    return {
        "run_id": args.run_id,
        "status": "started",
        "created_at": now_iso(),
        "git": git_state(args.run_id),
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "package_versions": {
            "tinker": package_version("tinker"),
            "tinker-cookbook": package_version("tinker-cookbook"),
            "transformers": package_version("transformers"),
        },
        "contracts": {
            "lora_defaults": display_path(LORA_DEFAULTS_PATH),
            "lora_defaults_sha256": sha256_file(LORA_DEFAULTS_PATH),
            "run_protocol": display_path(RUN_PROTOCOL_PATH),
            "run_protocol_sha256": sha256_file(RUN_PROTOCOL_PATH),
            "results_schema": display_path(RESULTS_SCHEMA_PATH),
            "results_schema_sha256": sha256_file(RESULTS_SCHEMA_PATH),
        },
        "dataset": {
            "train_path": display_path(TRAIN_PATH),
            "raw_manifest_sha256": sha256_file(RAW_MANIFEST_PATH),
            "rendered_manifest_sha256": sha256_file(RENDERED_MANIFEST_PATH),
            "train_start_index": 0,
            "train_rows": len(train_rows),
            "train_row_ids": [row["row_id"] for row in train_rows],
        },
        "probe": {
            "condition": args.condition,
            "lora_config": lora_config_summary(args.condition),
            "rank": LORA_RANK,
            "seed": SEED,
            "learning_rate": args.learning_rate,
            "optimizer_steps_per_shape": args.optimizer_steps,
            "effective_batch_size": args.effective_batch_size,
            "request_shapes": args.request_shapes,
            "checkpoint_saved": False,
        },
    }


def metric_row(
    *,
    run_id: str,
    condition: str,
    request_shape: str,
    step: int,
    loss: float,
    learning_rate: float,
    token_count: int,
    train_call_count: int,
    train_seconds: float,
    optimizer_seconds: float,
    backend_metrics: dict[str, Any],
) -> dict[str, Any]:
    step_seconds = train_seconds + optimizer_seconds
    return {
        "run_id": run_id,
        "checkpoint": None,
        "condition": condition,
        "seed": SEED,
        "step": step,
        "split": "throughput_probe",
        "loss": loss,
        "learning_rate": learning_rate,
        "eval_metric": {
            "name": "seconds_per_optimizer_step",
            "value": step_seconds,
        },
        "token_count": token_count,
        "cost": None,
        "request_shape": request_shape,
        "train_call_count": train_call_count,
        "train_seconds": train_seconds,
        "optimizer_seconds": optimizer_seconds,
        "step_seconds": step_seconds,
        "backend_metrics": backend_metrics,
    }


async def forward_backward_single_datum_calls(
    training_client: TrainingClient,
    batch: list[Datum],
) -> tuple[float, dict[str, float]]:
    """Send one datum per forward/backward call, matching LR selection."""

    weighted_loss_sum = 0.0
    weight_sum = 0.0
    backend_metrics: list[dict[str, float]] = []
    for datum in batch:
        future = await training_client.forward_backward_async(
            [datum], loss_fn="cross_entropy"
        )
        output = await future.result_async()
        answer_tokens = answer_weight_count(datum)
        weighted_loss_sum += mean_nll(output, [datum]) * answer_tokens
        weight_sum += answer_tokens
        backend_metrics.append(output.metrics)
    return weighted_loss_sum / weight_sum, aggregate_metrics(backend_metrics)


async def forward_backward_batched_datums(
    training_client: TrainingClient,
    batch: list[Datum],
) -> tuple[float, dict[str, float]]:
    """Send the whole effective batch in one forward/backward call."""

    future = await training_client.forward_backward_async(
        batch, loss_fn="cross_entropy"
    )
    output = await future.result_async()
    return mean_nll(output, batch), output.metrics


async def optimizer_step(
    training_client: TrainingClient, learning_rate: float
) -> dict[str, float]:
    future = await training_client.optim_step_async(
        AdamParams(learning_rate=learning_rate)
    )
    output = await future.result_async()
    return output.metrics


async def forward_backward_and_optimizer_pipelined(
    training_client: TrainingClient,
    batch: list[Datum],
    learning_rate: float,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Submit one batched train request and one optimizer request before waiting."""

    train_future = await training_client.forward_backward_async(
        batch, loss_fn="cross_entropy"
    )
    optim_future = await training_client.optim_step_async(
        AdamParams(learning_rate=learning_rate)
    )
    train_output = await train_future.result_async()
    optim_output = await optim_future.result_async()
    return mean_nll(train_output, batch), train_output.metrics, optim_output.metrics


async def run_probe_shape(
    service_client: ServiceClient,
    *,
    args: argparse.Namespace,
    request_shape: RequestShape,
    train_rows: list[dict[str, Any]],
    metrics_path: Path,
    sample_render_path: Path,
) -> dict[str, Any]:
    """Measure one request shape with its own Tinker training client."""

    training_client = await create_training_client(
        service_client,
        condition=args.condition,
        run_id=f"{args.run_id}-{request_shape.name}",
        extra_metadata={
            "phase": "throughput_probe",
            "request_shape": request_shape.name,
        },
    )
    tokenizer = training_client.get_tokenizer()
    if not sample_render_path.exists():
        sample_render_path.write_text(render_text(train_rows[0], tokenizer) + "\n")
    train_datums = build_datums(train_rows, tokenizer)

    rows = []
    for step in range(1, args.optimizer_steps + 1):
        start = (step - 1) * args.effective_batch_size
        batch = train_datums[start : start + args.effective_batch_size]
        train_started = time.perf_counter()
        if request_shape.name == "single_datum_calls":
            loss, train_metrics = await forward_backward_single_datum_calls(
                training_client, batch
            )
            train_call_count = len(batch)
            train_seconds = time.perf_counter() - train_started
            optim_started = time.perf_counter()
            optim_metrics = await optimizer_step(training_client, args.learning_rate)
            optimizer_seconds = time.perf_counter() - optim_started
        elif request_shape.name == "batched_datums":
            loss, train_metrics = await forward_backward_batched_datums(
                training_client, batch
            )
            train_call_count = 1
            train_seconds = time.perf_counter() - train_started
            optim_started = time.perf_counter()
            optim_metrics = await optimizer_step(training_client, args.learning_rate)
            optimizer_seconds = time.perf_counter() - optim_started
        else:
            loss, train_metrics, optim_metrics = (
                await forward_backward_and_optimizer_pipelined(
                    training_client,
                    batch,
                    args.learning_rate,
                )
            )
            train_call_count = 1
            train_seconds = time.perf_counter() - train_started
            optimizer_seconds = 0.0

        row = metric_row(
            run_id=args.run_id,
            condition=args.condition,
            request_shape=request_shape.name,
            step=step,
            loss=loss,
            learning_rate=args.learning_rate,
            token_count=sum(datum_token_count(datum) for datum in batch),
            train_call_count=train_call_count,
            train_seconds=train_seconds,
            optimizer_seconds=optimizer_seconds,
            backend_metrics={
                "train": train_metrics,
                "optimizer": optim_metrics,
            },
        )
        append_jsonl(metrics_path, row)
        rows.append(row)
        print(
            f"{request_shape.name}: step={step} "
            f"seconds={row['step_seconds']:.2f}",
            flush=True,
        )

    total_seconds = sum(row["step_seconds"] for row in rows)
    return {
        "request_shape": request_shape.name,
        "status": "pass",
        "optimizer_steps": len(rows),
        "train_call_count": sum(row["train_call_count"] for row in rows),
        "train_calls_per_optimizer_step": (
            sum(row["train_call_count"] for row in rows) / len(rows)
        ),
        "train_tokens": sum(row["token_count"] for row in rows),
        "total_seconds": total_seconds,
        "seconds_per_optimizer_step": total_seconds / len(rows),
    }


async def try_probe_shape(
    service_client: ServiceClient,
    *,
    args: argparse.Namespace,
    request_shape: RequestShape,
    train_rows: list[dict[str, Any]],
    metrics_path: Path,
    sample_render_path: Path,
) -> dict[str, Any]:
    try:
        return await run_probe_shape(
            service_client,
            args=args,
            request_shape=request_shape,
            train_rows=train_rows,
            metrics_path=metrics_path,
            sample_render_path=sample_render_path,
        )
    except Exception as exc:
        return {
            "request_shape": request_shape.name,
            "status": "fail",
            "failed_at": now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def choose_request_shape(results: list[dict[str, Any]]) -> str | None:
    """Pick the fastest passing shape, leaving policy freeze to the docs."""

    passing = [row for row in results if row["status"] == "pass"]
    if not passing:
        return None
    best = min(passing, key=lambda row: row["seconds_per_optimizer_step"])
    return str(best["request_shape"])


async def run(args: argparse.Namespace) -> int:
    validate_args(args)
    load_dotenv()
    for path in required_input_paths():
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_root / args.run_id
    metrics_path = prepare_output_dir(output_dir, args.overwrite)
    needed_rows = args.optimizer_steps * args.effective_batch_size
    train_rows = read_jsonl(TRAIN_PATH, needed_rows)
    manifest = build_manifest(args=args, train_rows=train_rows)
    write_json(output_dir / "manifest.json", manifest)

    if args.dry_run:
        manifest["status"] = "dry_run_pass"
        manifest["finished_at"] = now_iso()
        write_json(output_dir / "manifest.json", manifest)
        print(json.dumps({"run_id": args.run_id, "status": "dry_run_pass"}))
        return 0

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set in the environment or .env")

    service_client = ServiceClient()
    await require_supported_model(service_client)

    results = []
    sample_render_path = output_dir / "sample_render.txt"
    for name in args.request_shapes:
        result = await try_probe_shape(
            service_client,
            args=args,
            request_shape=RequestShape(name),
            train_rows=train_rows,
            metrics_path=metrics_path,
            sample_render_path=sample_render_path,
        )
        results.append(result)

    recommendation = choose_request_shape(results)
    statuses = {row["status"] for row in results}
    if statuses == {"pass"}:
        status = "pass"
    elif "pass" in statuses:
        status = "partial_fail"
    else:
        status = "fail"
    summary = {
        "run_id": args.run_id,
        "status": status,
        "finished_at": now_iso(),
        "condition": args.condition,
        "seed": SEED,
        "learning_rate": args.learning_rate,
        "effective_batch_size": args.effective_batch_size,
        "optimizer_steps_per_shape": args.optimizer_steps,
        "recommended_request_shape": recommendation,
        "request_shapes": results,
        "artifact_paths": {
            "manifest": display_path(output_dir / "manifest.json"),
            "metrics": display_path(metrics_path),
            "summary": display_path(output_dir / "summary.json"),
            "sample_render": display_path(sample_render_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    manifest["status"] = status
    manifest["finished_at"] = summary["finished_at"]
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if "pass" in statuses else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
