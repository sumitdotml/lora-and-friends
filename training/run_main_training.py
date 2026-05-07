#!/usr/bin/env python3
"""Run the frozen main LoRA comparison on Tinker."""

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


MAIN_SEEDS = (0, 1, 2)
PEAK_LR_BY_CONDITION = {
    "attention_only": 3e-4,
    "all_layer": 3e-4,
}
EPOCHS = 2
EFFECTIVE_BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 16
VALIDATION_INTERVAL = 1000
EPOCH_END_GAP_SKIP = 250
WARMUP_FRACTION = 0.03
MIN_LR_RATIO = 0.10
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 0.0
REQUEST_SHAPE = "batched_datums_pipelined"


@dataclass(frozen=True)
class MainRunSpec:
    """One condition/seed run in the final comparison."""

    run_prefix: str
    condition: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.run_prefix}-{self.condition}-seed-{self.seed}"

    @property
    def peak_lr(self) -> float:
        return PEAK_LR_BY_CONDITION[self.condition]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen main Tinker LoRA comparison."
    )
    parser.add_argument("--run-prefix", default="main-001")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_ORDER,
        default=list(CONDITION_ORDER),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(MAIN_SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--ttl-seconds", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def run_specs(args: argparse.Namespace) -> list[MainRunSpec]:
    """Expand CLI choices into the condition/seed runs that will execute."""

    return [
        MainRunSpec(run_prefix=args.run_prefix, condition=condition, seed=seed)
        for condition in args.conditions
        for seed in args.seeds
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


def optimizer_steps_per_epoch(train_rows: int) -> int:
    return math.ceil(train_rows / EFFECTIVE_BATCH_SIZE)


def default_validation_steps(
    *,
    steps_per_epoch: int,
    total_steps: int,
    epochs: int,
) -> list[int]:
    """Return the frozen validation/checkpoint steps for this run shape.

    Epoch-end validations are preferred over nearby interval validations. For
    the frozen full run this produces steps 1000, 2000, 3169, 4000, 5000, 6000,
    and 6338.
    """

    epoch_ends = {
        min(epoch * steps_per_epoch, total_steps) for epoch in range(1, epochs + 1)
    }
    steps = set(epoch_ends)
    for step in range(VALIDATION_INTERVAL, total_steps + 1, VALIDATION_INTERVAL):
        if any(0 < epoch_end - step <= EPOCH_END_GAP_SKIP for epoch_end in epoch_ends):
            continue
        steps.add(step)
    steps.add(total_steps)
    return sorted(step for step in steps if 1 <= step <= total_steps)


def scheduled_lr(
    *,
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
) -> float:
    """Compute the frozen linear-warmup plus cosine-decay learning rate."""

    min_lr = peak_lr * MIN_LR_RATIO
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return peak_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def metric_row(
    *,
    spec: MainRunSpec,
    step: int,
    split: str,
    loss: float | None,
    current_lr: float | None,
    token_count: int,
    checkpoint: str | None,
    backend_metrics: dict[str, float] | None,
    eval_metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "checkpoint": checkpoint,
        "condition": spec.condition,
        "seed": spec.seed,
        "step": step,
        "split": split,
        "loss": loss,
        "learning_rate": current_lr,
        "current_lr": current_lr,
        "peak_lr": spec.peak_lr,
        "eval_metric": eval_metric,
        "token_count": token_count,
        "cost": None,
        "backend_metrics": backend_metrics or {},
    }


async def run_train_step(
    training_client: Any,
    batch: list[Any],
    *,
    spec: MainRunSpec,
    step: int,
    current_lr: float,
    metrics_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one pipelined train/update step and retain train plus optimizer rows."""

    import tinker

    train_future = await training_client.forward_backward_async(
        batch, loss_fn="cross_entropy"
    )
    optim_future = await training_client.optim_step_async(
        tinker.AdamParams(
            learning_rate=current_lr,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=GRAD_CLIP_NORM,
        )
    )
    train_output = await train_future.result_async()
    optim_output = await optim_future.result_async()
    token_count = sum(datum_token_count(datum) for datum in batch)
    train_row = metric_row(
        spec=spec,
        step=step,
        split="main_train",
        loss=mean_nll(train_output, batch),
        current_lr=current_lr,
        token_count=token_count,
        checkpoint=None,
        backend_metrics=train_output.metrics,
    )
    optim_row = metric_row(
        spec=spec,
        step=step,
        split="main_optim",
        loss=None,
        current_lr=current_lr,
        token_count=token_count,
        checkpoint=None,
        backend_metrics=optim_output.metrics,
    )
    append_jsonl(metrics_path, train_row)
    append_jsonl(metrics_path, optim_row)
    return train_row, optim_row


async def evaluate_validation(
    training_client: Any,
    val_datums: list[Any],
    *,
    spec: MainRunSpec,
    step: int,
    current_lr: float,
) -> dict[str, Any]:
    """Measure validation NLL over the frozen validation split."""

    weighted_loss_sum = 0.0
    weight_sum = 0.0
    token_count = 0
    backend_metrics: list[dict[str, float]] = []
    for start in range(0, len(val_datums), VALIDATION_BATCH_SIZE):
        chunk = val_datums[start : start + VALIDATION_BATCH_SIZE]
        future = await training_client.forward_async(chunk, loss_fn="cross_entropy")
        output = await future.result_async()
        chunk_weight = sum(answer_weight_count(datum) for datum in chunk)
        weighted_loss_sum += mean_nll(output, chunk) * chunk_weight
        weight_sum += chunk_weight
        token_count += sum(datum_token_count(datum) for datum in chunk)
        backend_metrics.append(output.metrics)
    loss = weighted_loss_sum / weight_sum
    return metric_row(
        spec=spec,
        step=step,
        split="main_val",
        loss=loss,
        current_lr=current_lr,
        token_count=token_count,
        checkpoint=None,
        backend_metrics=aggregate_metrics(backend_metrics),
        eval_metric={"name": "validation_mean_nll", "value": loss},
    )


def build_manifest(
    *,
    spec: MainRunSpec,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    total_steps: int,
    warmup_steps: int,
    validation_steps: list[int],
) -> dict[str, Any]:
    """Build the reproducibility manifest before model state changes."""

    full_protocol = (
        args.train_limit is None
        and args.val_limit is None
        and args.max_optimizer_steps is None
        and args.epochs == EPOCHS
        and tuple(args.conditions) == CONDITION_ORDER
        and tuple(args.seeds) == MAIN_SEEDS
    )
    return {
        "run_id": spec.run_id,
        "status": "started",
        "created_at": now_iso(),
        "protocol_mode": "frozen_main_comparison" if full_protocol else "override",
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
            "seed": spec.seed,
            "condition": spec.condition,
            "lora_config": lora_config_summary(spec.condition),
            "rank": LORA_RANK,
            "request_shape": REQUEST_SHAPE,
            "datums_per_forward_backward": f"up to {EFFECTIVE_BATCH_SIZE}",
            "forward_backward_calls_per_optimizer_step": 1,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "final_epoch_batch_size": len(train_rows) % EFFECTIVE_BATCH_SIZE
            or EFFECTIVE_BATCH_SIZE,
            "epochs": args.epochs,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch(len(train_rows)),
            "total_optimizer_steps": total_steps,
            "max_optimizer_steps": args.max_optimizer_steps,
        },
        "optimizer": {
            "name": "Tinker AdamParams",
            "peak_lr": spec.peak_lr,
            "min_lr": spec.peak_lr * MIN_LR_RATIO,
            "warmup_fraction": WARMUP_FRACTION,
            "warmup_steps": warmup_steps,
            "schedule": "linear_warmup_cosine_decay",
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-12,
        },
        "validation": {
            "validation_batch_size": VALIDATION_BATCH_SIZE,
            "validation_steps": validation_steps,
            "selection_rule": "lowest validation_mean_nll; exact ties choose later checkpoint",
        },
    }


async def run_one_spec(
    service_client: Any,
    spec: MainRunSpec,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one main comparison training job and retain artifacts."""

    output_dir = args.output_root / spec.run_id
    metrics_path = prepare_output_dir(output_dir, args.overwrite)
    steps_per_epoch = optimizer_steps_per_epoch(len(train_rows))
    total_steps = steps_per_epoch * args.epochs
    if args.max_optimizer_steps is not None:
        total_steps = min(total_steps, args.max_optimizer_steps)
    warmup_steps = round(total_steps * WARMUP_FRACTION)
    validation_steps = default_validation_steps(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        epochs=args.epochs,
    )
    manifest = build_manifest(
        spec=spec,
        args=args,
        train_rows=train_rows,
        val_rows=val_rows,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        validation_steps=validation_steps,
    )
    write_json(output_dir / "manifest.json", manifest)

    if args.dry_run:
        manifest["status"] = "dry_run_pass"
        manifest["finished_at"] = now_iso()
        write_json(output_dir / "manifest.json", manifest)
        return {"run_id": spec.run_id, "status": "dry_run_pass"}

    try:
        print(
            f"starting {spec.run_id}: condition={spec.condition} seed={spec.seed}",
            flush=True,
        )
        training_client = await create_training_client(
            service_client,
            condition=spec.condition,
            run_id=spec.run_id,
            seed=spec.seed,
            extra_metadata={
                "phase": "main_comparison",
                "peak_lr": str(spec.peak_lr),
            },
        )
        tokenizer = training_client.get_tokenizer()
        train_datums = build_datums(train_rows, tokenizer)
        val_datums = build_datums(val_rows, tokenizer)
        (output_dir / "sample_render.txt").write_text(
            render_text(train_rows[0], tokenizer) + "\n"
        )

        train_token_count = 0
        optim_token_count = 0
        validation_token_count = 0
        validation_rows = []
        validation_step_set = set(validation_steps)
        for step in range(1, total_steps + 1):
            step_in_epoch = (step - 1) % steps_per_epoch
            row_start = step_in_epoch * EFFECTIVE_BATCH_SIZE
            batch = train_datums[row_start : row_start + EFFECTIVE_BATCH_SIZE]
            current_lr = scheduled_lr(
                step=step,
                total_steps=total_steps,
                peak_lr=spec.peak_lr,
                warmup_steps=warmup_steps,
            )
            train_row, optim_row = await run_train_step(
                training_client,
                batch,
                spec=spec,
                step=step,
                current_lr=current_lr,
                metrics_path=metrics_path,
            )
            train_token_count += train_row["token_count"]
            optim_token_count += optim_row["token_count"]
            if step in validation_step_set:
                val_row = await evaluate_validation(
                    training_client,
                    val_datums,
                    spec=spec,
                    step=step,
                    current_lr=current_lr,
                )
                checkpoint = await save_checkpoint(
                    training_client,
                    checkpoint_name=f"{spec.run_id}-step-{step}",
                    ttl_seconds=args.ttl_seconds,
                )
                val_row["checkpoint"] = checkpoint["path"]
                append_jsonl(metrics_path, val_row)
                validation_rows.append({**val_row, "checkpoint_metadata": checkpoint})
                validation_token_count += val_row["token_count"]
                print(
                    f"{spec.run_id}: validation step={step} "
                    f"nll={val_row['eval_metric']['value']:.6f}",
                    flush=True,
                )

        best_validation = min(
            validation_rows,
            key=lambda row: (row["eval_metric"]["value"], -row["step"]),
        )
        summary = {
            "run_id": spec.run_id,
            "status": "pass",
            "finished_at": now_iso(),
            "checkpoint": best_validation["checkpoint_metadata"],
            "condition": spec.condition,
            "seed": spec.seed,
            "dataset": {
                "train_path": str(TRAIN_PATH.relative_to(ROOT)),
                "val_path": str(VAL_PATH.relative_to(ROOT)),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
            },
            "primary_metric": best_validation["eval_metric"],
            "best_validation_step": best_validation["step"],
            "learning_rate": {
                "peak_lr": spec.peak_lr,
                "min_lr": spec.peak_lr * MIN_LR_RATIO,
                "warmup_steps": warmup_steps,
            },
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
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    load_dotenv()
    for path in required_input_paths():
        if not path.exists():
            raise FileNotFoundError(path)

    specs = run_specs(args)
    train_rows = read_jsonl(TRAIN_PATH, args.train_limit or 25348)
    val_rows = read_jsonl(VAL_PATH, args.val_limit or 2818)

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
