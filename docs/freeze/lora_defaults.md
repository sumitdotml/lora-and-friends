# LoRA Defaults

**Status**: provisional only  
**Frozen on**: not yet  
**Purpose**: define the provisional adapter defaults needed for the smoke pass, then become the locked adapter contract after the smoke pass.

## Freeze Rule

This file should exist before the smoke pass.

Its contents should only be treated as locked after the smoke pass findings are incorporated and the updated file is committed.

## Provisional Defaults

Current provisional values:

- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.0`

Target modules by arm:

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

- LR is not owned by this file. LR is selected by the pilot sweep and frozen in the run protocol.

## Still Provisional

These are not locked yet:

- batch-size strategy
- gradient accumulation
- any backend-constrained defaults surfaced by Tinker

## Intended Scope When Locked

The locked version of this file should define:

- final `r`
- final `lora_alpha`
- final `lora_dropout`
- final batch-size and gradient-accumulation assumptions
- the rationale for why these values are fixed across both arms
- the target module lists for both arms

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
5. add a `LOG.md` entry recording the transition
