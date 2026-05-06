"""Run the smallest training check for the two LoRA conditions."""

from __future__ import annotations

import argparse
import asyncio
import os
import traceback
from pathlib import Path
from typing import Any

from common import (
    RAW_MANIFEST_PATH,
    RENDERED_MANIFEST_PATH,
    ROOT,
    TRAIN_PATH,
    VAL_PATH,
    append_jsonl,
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
    CONDITIONS,
    MODEL_NAME,
    RENDERER_NAME,
    SEED,
    create_training_client,
    lora_config_summary,
    require_supported_model,
    save_checkpoint,
)
from sft import build_datums, mean_nll, render_text


DEFAULT_OUTPUT_DIR = ROOT / "artifacts/smoke_pass/001"
LEARNING_RATE = 1e-4
SMOKE_TRAIN_EXAMPLES = {
    "attention_only": 1,
    "all_layer": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the smallest backend check for selected LoRA conditions."
    )
    parser.add_argument("--run-id", default="smoke-001")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_ORDER,
        default=list(CONDITION_ORDER),
        help="LoRA condition names to run.",
    )
    parser.add_argument("--val-examples", type=int, default=2)
    parser.add_argument("--ttl-seconds", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def required_input_paths() -> list[Path]:
    return [TRAIN_PATH, VAL_PATH, RAW_MANIFEST_PATH, RENDERED_MANIFEST_PATH]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    prepare_output_files(
        output_dir,
        ["metrics.jsonl", "summary.json", "sample_render.txt", "failure.json"],
        overwrite=overwrite,
    )
    return output_dir / "metrics.jsonl"


def max_train_examples() -> int:
    return max(SMOKE_TRAIN_EXAMPLES.values())


def condition_artifact_map() -> dict[str, dict[str, bool | int]]:
    """Describe the tiny per-condition shape retained in the manifest."""

    return {
        name: {
            **CONDITIONS[name].as_json(),
            "train_examples": train_examples,
        }
        for name, train_examples in SMOKE_TRAIN_EXAMPLES.items()
    }


def metric_row(
    *,
    run_id: str,
    condition: str,
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
        "eval_metric": eval_metric,
        "token_count": token_count,
        "cost": None,
        "backend_metrics": backend_metrics or {},
    }


async def run_validation(
    training_client: Any,
    val_datums: list[Any],
    *,
    condition: str,
    step: int,
    run_id: str,
    metrics_path: Path,
) -> dict[str, Any]:
    """Run a forward-only validation check for one smoke condition."""

    future = await training_client.forward_async(val_datums, loss_fn="cross_entropy")
    output = await future.result_async()
    loss = mean_nll(output, val_datums)
    row = metric_row(
        run_id=run_id,
        condition=condition,
        step=step,
        split="smoke_val",
        loss=loss,
        token_count=sum(datum.model_input.length for datum in val_datums),
        backend_metrics=output.metrics,
        eval_metric={"name": "validation_mean_nll", "value": loss},
    )
    append_jsonl(metrics_path, row)
    return row


async def run_training_steps(
    training_client: Any,
    train_datums: list[Any],
    *,
    condition: str,
    run_id: str,
    metrics_path: Path,
) -> list[dict[str, Any]]:
    """Run the tiny forward/backward sequence that proves gradients flow."""

    train_metrics = []
    for idx, datum in enumerate(train_datums):
        future = await training_client.forward_backward_async(
            [datum], loss_fn="cross_entropy"
        )
        output = await future.result_async()
        loss = mean_nll(output, [datum])
        row = metric_row(
            run_id=run_id,
            condition=condition,
            step=idx,
            split="smoke_train",
            loss=loss,
            token_count=datum.model_input.length,
            backend_metrics=output.metrics,
        )
        append_jsonl(metrics_path, row)
        train_metrics.append(row)
    return train_metrics


async def run_optimizer_step(
    training_client: Any,
    train_metrics: list[dict[str, Any]],
    *,
    condition: str,
    run_id: str,
    step: int,
    metrics_path: Path,
) -> dict[str, Any]:
    import tinker

    future = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=LEARNING_RATE)
    )
    output = await future.result_async()
    row = metric_row(
        run_id=run_id,
        condition=condition,
        step=step,
        split="smoke_optim",
        loss=None,
        token_count=sum(row["token_count"] for row in train_metrics),
        backend_metrics=output.metrics,
    )
    append_jsonl(metrics_path, row)
    return row


async def run_condition(
    service_client: Any,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    condition: str,
    run_id: str,
    output_dir: Path,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Run one LoRA condition and return its retained summary."""

    metrics_path = output_dir / "metrics.jsonl"
    training_client = await create_training_client(
        service_client,
        condition=condition,
        run_id=run_id,
    )
    tokenizer = training_client.get_tokenizer()
    sample_render_path = output_dir / "sample_render.txt"
    if not sample_render_path.exists():
        sample_render_path.write_text(render_text(train_rows[0], tokenizer) + "\n")

    train_datums = build_datums(
        train_rows[: SMOKE_TRAIN_EXAMPLES[condition]], tokenizer
    )
    val_datums = build_datums(val_rows, tokenizer)
    train_metrics = await run_training_steps(
        training_client,
        train_datums,
        condition=condition,
        run_id=run_id,
        metrics_path=metrics_path,
    )
    await run_optimizer_step(
        training_client,
        train_metrics,
        condition=condition,
        run_id=run_id,
        step=len(train_datums),
        metrics_path=metrics_path,
    )
    val_metric = await run_validation(
        training_client,
        val_datums,
        condition=condition,
        step=len(train_datums),
        run_id=run_id,
        metrics_path=metrics_path,
    )

    checkpoint_name = f"{run_id}-{condition}-final"
    checkpoint = await save_checkpoint(
        training_client,
        checkpoint_name=checkpoint_name,
        ttl_seconds=ttl_seconds,
    )
    return {
        "condition": condition,
        "lora_config": lora_config_summary(condition),
        "train_examples": len(train_datums),
        "optimizer_steps": 1,
        "validation": val_metric,
        "checkpoint": checkpoint,
    }


def build_manifest(
    *,
    run_id: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    ttl_seconds: int,
) -> dict[str, Any]:
    """Build the retained smoke-pass manifest before contacting Tinker."""

    return {
        "run_id": run_id,
        "status": "started",
        "created_at": now_iso(),
        "git": git_state(),
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "package_versions": {
            "tinker": package_version("tinker"),
            "tinker-cookbook": package_version("tinker-cookbook"),
            "transformers": package_version("transformers"),
        },
        "dataset": {
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "val_path": str(VAL_PATH.relative_to(ROOT)),
            "raw_manifest_sha256": sha256_file(RAW_MANIFEST_PATH),
            "rendered_manifest_sha256": sha256_file(RENDERED_MANIFEST_PATH),
            "train_row_ids": [row["row_id"] for row in train_rows],
            "val_row_ids": [row["row_id"] for row in val_rows],
        },
        "smoke_shape": {
            "micro_batch_size": 1,
            "gradient_accumulation_tested_by_all_layer_condition": (
                SMOKE_TRAIN_EXAMPLES["all_layer"]
            ),
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "checkpoint_ttl_seconds": ttl_seconds,
        },
        "conditions": condition_artifact_map(),
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv()
    if not args.dry_run and not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set in the environment or .env")
    for path in required_input_paths():
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    run_id = args.run_id
    metrics_path = prepare_output_dir(output_dir, args.overwrite)
    train_rows = read_jsonl(TRAIN_PATH, max_train_examples())
    val_rows = read_jsonl(VAL_PATH, args.val_examples)

    manifest = build_manifest(
        run_id=run_id,
        train_rows=train_rows,
        val_rows=val_rows,
        ttl_seconds=args.ttl_seconds,
    )
    write_json(output_dir / "manifest.json", manifest)

    if args.dry_run:
        manifest["status"] = "dry_run_pass"
        manifest["finished_at"] = now_iso()
        write_json(output_dir / "manifest.json", manifest)
        return 0

    try:
        import tinker

        service_client = tinker.ServiceClient()
        supported_models = await require_supported_model(service_client)

        condition_summaries = []
        for condition in args.conditions:
            condition_summaries.append(
                await run_condition(
                    service_client,
                    train_rows,
                    val_rows,
                    condition=condition,
                    run_id=run_id,
                    output_dir=output_dir,
                    ttl_seconds=args.ttl_seconds,
                )
            )

        summary = {
            "run_id": run_id,
            "status": "pass",
            "finished_at": now_iso(),
            "supported_model_checked": MODEL_NAME in supported_models,
            "validation_loss_reporting": (
                "manual forward pass controlled by the local training loop"
            ),
            "token_telemetry": (
                "local datum token counts retained; backend cost telemetry not "
                f"observed in tinker {package_version('tinker')} responses"
            ),
            "conditions": condition_summaries,
            "artifact_paths": {
                "manifest": str((output_dir / "manifest.json").relative_to(ROOT)),
                "metrics": str(metrics_path.relative_to(ROOT)),
                "summary": str((output_dir / "summary.json").relative_to(ROOT)),
                "sample_render": str(
                    (output_dir / "sample_render.txt").relative_to(ROOT)
                ),
            },
        }
        write_json(output_dir / "summary.json", summary)
        manifest["status"] = "pass"
        manifest["finished_at"] = summary["finished_at"]
        write_json(output_dir / "manifest.json", manifest)
        return 0
    except Exception as exc:
        failure = {
            "run_id": run_id,
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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
