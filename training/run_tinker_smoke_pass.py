"""Run the smallest Tinker training check for the two LoRA conditions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
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
RAW_MANIFEST_PATH = (
    ROOT / "artifacts/raw_datasets/openmath_original_clean/manifest.json"
)
RENDERED_MANIFEST_PATH = (
    ROOT
    / "artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/smoke_pass/001"

MODEL_NAME = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3_disable_thinking"
SEED = 7
LORA_RANK = 8
LEARNING_RATE = 1e-4


@dataclass(frozen=True)
class ConditionConfig:
    """One LoRA setup to try during the smoke pass."""

    train_attn: bool
    train_mlp: bool
    train_unembed: bool
    train_examples: int

    def as_json(self) -> dict[str, bool | int]:
        return {
            "train_attn": self.train_attn,
            "train_mlp": self.train_mlp,
            "train_unembed": self.train_unembed,
            "train_examples": self.train_examples,
        }


CONDITIONS: dict[str, ConditionConfig] = {
    "attention_only": ConditionConfig(
        train_attn=True,
        train_mlp=False,
        train_unembed=False,
        train_examples=1,
    ),
    "all_layer": ConditionConfig(
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
        train_examples=8,
    ),
}


def condition_artifact_map() -> dict[str, dict[str, bool | int]]:
    return {name: config.as_json() for name, config in CONDITIONS.items()}


def max_train_examples() -> int:
    return max(config.train_examples for config in CONDITIONS.values())


def all_layer_train_examples() -> int:
    return CONDITIONS["all_layer"].train_examples


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_state() -> dict[str, Any]:
    status = git_value(["status", "--short"])
    return {
        "sha": git_value(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def tinker_version_note() -> str:
    return (
        f"not exposed by tinker {package_version('tinker')} create_lora_training_client"
    )


def required_input_paths() -> list[Path]:
    return [TRAIN_PATH, VAL_PATH, RAW_MANIFEST_PATH, RENDERED_MANIFEST_PATH]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to replace it")
    if overwrite:
        for path in [
            metrics_path,
            output_dir / "summary.json",
            output_dir / "sample_render.txt",
            output_dir / "failure.json",
        ]:
            if path.exists():
                path.unlink()
    return metrics_path


def render_tokens(row: dict[str, Any], tokenizer: Any) -> list[int]:
    """Render one row with the frozen Qwen3 no-thinking chat template."""

    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    return list(rendered["input_ids"])


def render_prompt_tokens(row: dict[str, Any], tokenizer: Any) -> list[int]:
    """Render the prompt prefix so answer-only loss can be masked."""

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
    """Build a Tinker SFT datum with loss only on assistant answer tokens."""

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


def mean_nll(output: Any, data: list[Any]) -> float:
    from tinker_cookbook.supervised.common import compute_mean_nll

    logprobs = [x["logprobs"] for x in output.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in data]
    return compute_mean_nll(logprobs, weights)


def metric_row(
    *,
    run_id: str,
    condition_name: str,
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
        "condition": condition_name,
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
    condition_name: str,
    step: int,
    run_id: str,
    metrics_path: Path,
) -> dict[str, Any]:
    future = await training_client.forward_async(val_datums, loss_fn="cross_entropy")
    output = await future.result_async()
    loss = mean_nll(output, val_datums)
    row = metric_row(
        run_id=run_id,
        condition_name=condition_name,
        step=step,
        split="smoke_val",
        loss=loss,
        token_count=sum(datum.model_input.length for datum in val_datums),
        backend_metrics=output.metrics,
        eval_metric={"name": "validation_mean_nll", "value": loss},
    )
    append_jsonl(metrics_path, row)
    return row


async def create_training_client(
    service_client: Any,
    config: ConditionConfig,
    *,
    condition_name: str,
    run_id: str,
) -> Any:
    return await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME,
        rank=LORA_RANK,
        seed=SEED,
        train_attn=config.train_attn,
        train_mlp=config.train_mlp,
        train_unembed=config.train_unembed,
        user_metadata={
            "project": "lora-and-friends",
            "run_id": run_id,
            "condition": condition_name,
        },
    )


def write_sample_render(
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> None:
    sample_render_path = output_dir / "sample_render.txt"
    if not sample_render_path.exists():
        sample_render_path.write_text(render_text(train_rows[0], tokenizer) + "\n")


async def run_training_steps(
    training_client: Any,
    train_datums: list[Any],
    *,
    condition_name: str,
    run_id: str,
    metrics_path: Path,
) -> list[dict[str, Any]]:
    train_metrics = []
    for idx, datum in enumerate(train_datums):
        future = await training_client.forward_backward_async(
            [datum], loss_fn="cross_entropy"
        )
        output = await future.result_async()
        loss = mean_nll(output, [datum])
        row = metric_row(
            run_id=run_id,
            condition_name=condition_name,
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
    condition_name: str,
    run_id: str,
    step: int,
    metrics_path: Path,
) -> dict[str, Any]:
    import tinker

    optim_future = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=LEARNING_RATE)
    )
    optim_result = await optim_future.result_async()
    row = metric_row(
        run_id=run_id,
        condition_name=condition_name,
        step=step,
        split="smoke_optim",
        loss=None,
        token_count=sum(row["token_count"] for row in train_metrics),
        backend_metrics=optim_result.metrics,
    )
    append_jsonl(metrics_path, row)
    return row


async def save_checkpoint(
    training_client: Any,
    *,
    checkpoint_name: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    save_future = await training_client.save_state_async(
        checkpoint_name, ttl_seconds=ttl_seconds
    )
    save_result = await save_future.result_async()
    return {
        "name": checkpoint_name,
        "path": save_result.path,
        "ttl_seconds": ttl_seconds,
    }


def lora_config_summary(config: ConditionConfig) -> dict[str, Any]:
    return {
        "rank": LORA_RANK,
        "lora_alpha": tinker_version_note(),
        "lora_dropout": tinker_version_note(),
        "train_attn": config.train_attn,
        "train_mlp": config.train_mlp,
        "train_unembed": config.train_unembed,
    }


async def run_condition(
    service_client: Any,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    condition_name: str,
    run_id: str,
    output_dir: Path,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Run one LoRA condition and return the summary kept in artifacts."""

    config = CONDITIONS[condition_name]
    metrics_path = output_dir / "metrics.jsonl"
    training_client = await create_training_client(
        service_client,
        config,
        condition_name=condition_name,
        run_id=run_id,
    )
    tokenizer = training_client.get_tokenizer()

    write_sample_render(output_dir, train_rows, tokenizer)
    train_datums = build_datums(train_rows[: config.train_examples], tokenizer)
    val_datums = build_datums(val_rows, tokenizer)
    train_metrics = await run_training_steps(
        training_client,
        train_datums,
        condition_name=condition_name,
        run_id=run_id,
        metrics_path=metrics_path,
    )
    await run_optimizer_step(
        training_client,
        train_metrics,
        condition_name=condition_name,
        run_id=run_id,
        step=len(train_datums),
        metrics_path=metrics_path,
    )
    val_metric = await run_validation(
        training_client,
        val_datums,
        condition_name=condition_name,
        step=len(train_datums),
        run_id=run_id,
        metrics_path=metrics_path,
    )

    checkpoint_name = f"{run_id}-{condition_name}-final"
    checkpoint = await save_checkpoint(
        training_client,
        checkpoint_name=checkpoint_name,
        ttl_seconds=ttl_seconds,
    )

    return {
        "condition": condition_name,
        "lora_config": lora_config_summary(config),
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
                all_layer_train_examples()
            ),
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "checkpoint_ttl_seconds": ttl_seconds,
        },
        "conditions": condition_artifact_map(),
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set in the environment or .env")
    for path in required_input_paths():
        if not path.exists():
            raise FileNotFoundError(path)

    import tinker

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
        service_client = tinker.ServiceClient()
        capabilities = await service_client.get_server_capabilities_async()
        supported_models = [model.model_name for model in capabilities.supported_models]
        if MODEL_NAME not in supported_models:
            raise RuntimeError(
                f"{MODEL_NAME} not in Tinker supported models: {supported_models}"
            )

        condition_summaries = []
        for condition_name in args.conditions:
            condition_summaries.append(
                await run_condition(
                    service_client,
                    train_rows,
                    val_rows,
                    condition_name=condition_name,
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
                "local datum token counts retained; backend cost telemetry not observed "
                f"in tinker {package_version('tinker')} responses"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the smallest Tinker check for selected LoRA conditions."
    )
    parser.add_argument("--run-id", default="smoke-001")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=list(CONDITIONS),
        dest="conditions",
        help="LoRA condition names to run.",
    )
    parser.add_argument("--val-examples", type=int, default=2)
    parser.add_argument("--ttl-seconds", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
