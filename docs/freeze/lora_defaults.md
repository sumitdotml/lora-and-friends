# LoRA Defaults

**Status**: locked
**Frozen on**: 2026-05-05
**Purpose**: define the adapter defaults used by both LoRA comparison conditions after the Tinker smoke pass.

## Freeze Rule

These defaults are locked before the small LR-selection run and the main comparison.

Do not change these values after result-producing runs start unless a run is invalid. If a change is required, record the reason in `docs/project/LOG.md`, update this file before rerunning, and treat earlier affected results as superseded.

## Locked Shared Defaults

These values apply to both comparison conditions:

- base model: `Qwen/Qwen3-8B`
- backend: Tinker
- observed SDK version at lock time: `tinker==0.18.2`
- rendered dataset: `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/`
- renderer contract: Hugging Face chat template with `enable_thinking=False`
- LoRA rank: `r=8`
- micro-batch size: `1` rendered training example per `forward_backward` call
- gradient accumulation: `8` `forward_backward` calls before one optimizer step
- effective batch size: `8` rendered training examples per optimizer step
- train unembedding: disabled for both conditions with `train_unembed=false`

Backend-owned LoRA fields:

- `lora_alpha`: not exposed by `create_lora_training_client` in `tinker==0.18.2`
- `lora_dropout`: not exposed by `create_lora_training_client` in `tinker==0.18.2`

The public Tinker SDK, official docs, and public GitHub source checked on `2026-05-05` do not specify the backend alpha, scaling, or dropout behavior used when LoRA training clients are created. Treat these values as backend-owned and unknown, not as locally frozen values such as `alpha=16` or `dropout=0.0`.

Learning rate is not locked here. It is selected by the small LR-selection run and recorded in `docs/freeze/run_protocol.md`.

## Locked Conditions

Attention-only LoRA:

- Tinker switches: `train_attn=true`, `train_mlp=false`, `train_unembed=false`
- conceptual module scope: attention projections only, corresponding to `q_proj`, `k_proj`, `v_proj`, and `o_proj`

All-layer LoRA:

- Tinker switches: `train_attn=true`, `train_mlp=true`, `train_unembed=false`
- conceptual module scope: attention and MLP projections, corresponding to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`

## Smoke-Pass Evidence

Observed on `2026-05-04` in `artifacts/smoke_pass/001/`:

- `Qwen/Qwen3-8B` was accepted by Tinker.
- `r=8` was accepted.
- attention-only LoRA completed with `1` train example, `1` optimizer step, and validation mean NLL `1.5048651695251465`.
- all-layer LoRA completed with `8` train examples, `1` optimizer step after `8` forward/backward calls, and validation mean NLL `1.4733978509902954`.
- validation used `2` rows.
- all-layer train token count before optimizer step was `2,461`.
- validation token count per condition was `622`.
- Tinker checkpoint paths were created with a seven-day TTL.
- backend responses exposed metric fields such as `loss:sum` and `clock_cycle:unique`.
- backend responses did not expose cost telemetry in the observed response shape.

Renderer evidence:

- retained sample render: `artifacts/smoke_pass/001/sample_render.txt`
- the smoke runner builds Tinker datums from `AutoTokenizer.apply_chat_template(..., enable_thinking=False)`
- loss is masked to assistant answer tokens after the rendered prompt prefix

Resolved setup issue:

- the first smoke attempt printed `Your Tinker SDK version is outdated. Please upgrade to the latest version.`
- after upgrading to `tinker==0.18.2`, the smoke pass was rerun and the warning did not reappear

## Rationale

The comparison is meant to test adapter scope, not hidden changes in adapter hyperparameters. The shared defaults therefore stay identical across both conditions wherever Tinker exposes a shared setting.

Tinker `0.18.2` exposes layer-family switches rather than raw target-module string lists for this path. The locked contract uses those observed switches directly and keeps the conceptual module lists only as explanation.

`lora_alpha` and `lora_dropout` are not assigned local defaults because the observed Tinker API did not expose them. Recording them as unset is safer than pretending the local runner controls values it cannot pass to the backend.

This is a reproducibility limitation for anyone trying to recreate the run outside Tinker with a different LoRA implementation. It is not a within-Tinker comparison confound as long as both conditions use the same Tinker SDK path, same rank, same base model, same dataset render, and the only intended difference is `train_mlp=false` versus `train_mlp=true`.
