# Run Protocol

**Status**: partial  
**Purpose**: hold the pilot and main-run contracts that freeze later than the schema and eval contract.

## Pilot Sweep

**Frozen on**: not yet

Goal:

- select LR fairly for both arms after the smoke pass

Already fixed:

- pilot seed: `7`
- pilot seed must be held out from the main-run seed set

Still open:

- pilot subset identity
- pilot validation identity
- LR grid
- validation-loss cadence
- per-arm LR selection rule
- pilot budget estimate

## Main Run

**Frozen on**: not yet

Already fixed:

- main seeds: `0`, `1`, `2`

Still open:

- checkpoint-selection rule
- final run shape confirmation
- null-result interpretation rule

## Per-Arm Reduction

**Frozen on**: 2026-04-22

- report arm score as the mean across `3` seeds
- report min/max range alongside the mean
- apply any null-region threshold to the arm mean, not to a single seed

## Budget Reserve

**Frozen on**: 2026-04-22

- reserve `$25` for one correction pass

## Artifact Paths

**Frozen on**: 2026-04-22

- smoke-pass artifacts: `artifacts/smoke_pass/001/`
