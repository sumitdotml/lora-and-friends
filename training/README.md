# Training Scripts

This directory keeps runnable training entrypoints thin and moves repeated mechanics into small helper modules.

## Entrypoints

- `run_smoke_pass.py`: tiny backend check used to prove the renderer, LoRA switches, gradient accumulation, validation loss, and checkpoint saving path work.
- `run_lr_selection.py`: frozen small LR-selection sweep. This is the next script to run manually when ready.

## Shared Modules

- `common.py`: paths, JSON/JSONL writing, hashes, git state, package versions, and output-file checks.
- `sft.py`: Qwen3 chat-template rendering, assistant-only loss masking, Tinker datum creation, token counts, and mean-NLL helpers.
- `lora.py`: locked model name, seed, rank, LoRA condition switches, Tinker training-client creation, supported-model check, and checkpoint saving.

## Terms

- Tinker datum: the object sent to Tinker for one training example. It contains model-input tokens plus loss inputs such as token weights.
- Loss mask: the per-token weights used for SFT. Prompt tokens get weight `0`; assistant answer tokens get weight `1`.
- NLL: negative log-likelihood, the token-level loss used for LR selection. Lower validation NLL is better.
- Frozen mode: a run whose CLI arguments match the retained protocol exactly. Any CLI override is recorded as `override`.

## Manual Commands

These commands are intended to be run manually. Do not auto-run result-producing training commands unless the project owner explicitly asks for it.

To dry-run the LR-selection artifact path without Tinker calls:

```bash
uv run training/run_lr_selection.py \
  --dry-run \
  --run-prefix lr-select-dry-run \
  --output-root /tmp/lora-and-friends-lr-dry-run \
  --conditions attention_only \
  --learning-rates 1e-4 \
  --train-limit 16 \
  --val-limit 4 \
  --validation-every 1 \
  --max-optimizer-steps 1 \
  --overwrite
```

To run the real frozen LR-selection sweep when ready:

```bash
uv run training/run_lr_selection.py --run-prefix lr-select-001
```

Should not use `GSM8K` accuracy to pick the LR. The frozen rule is lowest `validation_mean_nll` per condition on the fixed small validation slice.
