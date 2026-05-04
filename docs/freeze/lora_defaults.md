# LoRA Defaults

**Status**: provisional only  
**Frozen on**: not yet  
**Purpose**: define the provisional adapter defaults needed for the smoke pass, then become the locked adapter contract after the smoke pass.

## Freeze Rule

This file should exist before the smoke pass.

Its contents should only be treated as locked after the smoke pass findings are incorporated and the updated file is committed.

Current interpretation:

- the pre-smoke defaults are defined, so the smoke pass has concrete settings to try
- the final LoRA defaults are not locked yet
- `TODO.md` step 3 being done means only "provisional defaults exist"
- `TODO.md` step 6 is the later lock step that turns this file into the real adapter contract

## Provisional Defaults

Current provisional values:

- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.0`

Provisional batch assumptions for the smoke pass:

- micro-batch size: `1` rendered training example per `forward_backward` call
- gradient accumulation: `8` `forward_backward` calls before one optimizer step
- effective batch size: `8` rendered training examples per optimizer step
- fallback if Tinker rejects this shape: use micro-batch size `1` and gradient accumulation `1` for the smoke pass, then record the backend constraint before locking this file

These are smoke-pass assumptions, not final locked defaults.

Target modules by condition:

- attention-only:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- all-layer:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

Explicit non-scope:

- LR is not owned by this file. LR is selected by the small LR-selection run and frozen in the run protocol.

## Still Provisional

These are not locked yet:

- final batch-size strategy
- final gradient accumulation
- any backend-constrained defaults surfaced by Tinker

## Smoke-Pass Findings

Observed on `2026-05-04` in `artifacts/smoke_pass/001/` with `tinker==0.18.2`:

- `Qwen/Qwen3-8B` was accepted by Tinker.
- attention-only LoRA was accepted as `train_attn=true`, `train_mlp=false`, `train_unembed=false`.
- all-layer LoRA for this project was accepted as `train_attn=true`, `train_mlp=true`, `train_unembed=false`.
- explicit target-module strings such as `q_proj` and `gate_proj` were not part of the observed Tinker API; the current SDK exposes layer-family switches instead.
- `r=8` was accepted.
- `lora_alpha` and `lora_dropout` were not exposed by `create_lora_training_client` in `tinker==0.18.2`.
- micro-batch size `1` and eight forward/backward calls before one optimizer step ran successfully for the all-layer condition.
- validation loss was measured by a local forward pass; Tinker did not automatically emit validation cadence.
- token counts were retained locally from each datum; backend responses did not expose cost telemetry in the observed response shape.

Resolved setup issue:

- the first smoke attempt printed `Your Tinker SDK version is outdated. Please upgrade to the latest version.`
- after upgrading to `tinker==0.18.2`, the smoke pass was rerun and the warning did not reappear

## Intended Scope When Locked

The locked version of this file should define:

- final `r`
- final `lora_alpha`
- final `lora_dropout`
- final batch-size and gradient-accumulation assumptions
- the rationale for why these values are fixed across both conditions
- the target module lists for both conditions

## Open Items Before Lock

- confirm that the provisional values run cleanly on Tinker
- confirm whether Tinker behavior forces any adjustment to effective batch assumptions
- confirm whether any target-module or backend constraint forces a change in the defaults

## Status Transition

When the smoke pass completes:

1. update the values if backend findings require it
2. fill the rationale section
3. change `Status` to locked
4. change `Frozen on` to the lock date
5. add a `docs/project/LOG.md` entry recording the transition
