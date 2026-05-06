"""Locked LoRA settings and training-client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common import package_version


MODEL_NAME = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3_disable_thinking"
SEED = 7
LORA_RANK = 8


@dataclass(frozen=True)
class LoraCondition:
    """One locked adapter scope for this project."""

    train_attn: bool
    train_mlp: bool
    train_unembed: bool

    def as_json(self) -> dict[str, bool]:
        return {
            "train_attn": self.train_attn,
            "train_mlp": self.train_mlp,
            "train_unembed": self.train_unembed,
        }


CONDITION_ORDER = ("attention_only", "all_layer")
CONDITIONS: dict[str, LoraCondition] = {
    "attention_only": LoraCondition(
        train_attn=True,
        train_mlp=False,
        train_unembed=False,
    ),
    "all_layer": LoraCondition(
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    ),
}


def tinker_version_note() -> str:
    """Describe LoRA settings the current Tinker API does not expose."""

    return (
        f"not exposed by tinker {package_version('tinker')} "
        "create_lora_training_client"
    )


def lora_config_summary(condition: str) -> dict[str, Any]:
    """Return the retained LoRA config block written into run artifacts."""

    return {
        "rank": LORA_RANK,
        "lora_alpha": tinker_version_note(),
        "lora_dropout": tinker_version_note(),
        **CONDITIONS[condition].as_json(),
    }


async def create_training_client(
    service_client: Any,
    *,
    condition: str,
    run_id: str,
    seed: int = SEED,
    extra_metadata: dict[str, str] | None = None,
) -> Any:
    """Create a Tinker LoRA training client for one locked condition.

    The project intentionally varies only the adapter scope: attention-only
    versus attention plus MLP. Shared settings such as base model, seed, and
    rank live in this module so both runners create comparable training runs.
    """

    config = CONDITIONS[condition]
    metadata = {
        "project": "lora-and-friends",
        "run_id": run_id,
        "condition": condition,
        "seed": str(seed),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME,
        rank=LORA_RANK,
        seed=seed,
        train_attn=config.train_attn,
        train_mlp=config.train_mlp,
        train_unembed=config.train_unembed,
        user_metadata=metadata,
    )


async def require_supported_model(service_client: Any) -> list[str]:
    """Fail before training if Tinker no longer advertises the locked model."""

    capabilities = await service_client.get_server_capabilities_async()
    supported_models = [model.model_name for model in capabilities.supported_models]
    if MODEL_NAME not in supported_models:
        raise RuntimeError(
            f"{MODEL_NAME} not in Tinker supported models: {supported_models}"
        )
    return supported_models


async def save_checkpoint(
    training_client: Any,
    *,
    checkpoint_name: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Save a Tinker checkpoint and normalize the retained checkpoint metadata."""

    save_future = await training_client.save_state_async(
        checkpoint_name, ttl_seconds=ttl_seconds
    )
    save_result = await save_future.result_async()
    return {
        "name": checkpoint_name,
        "path": save_result.path,
        "ttl_seconds": ttl_seconds,
    }
