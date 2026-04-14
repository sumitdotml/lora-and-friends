"""
QLoRA Concepts - Crash Course
Understand how quantization makes large model fine-tuning fit on one GPU.
"""

BYTES_PER_PARAM = {
    "float32": 4.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "nf4": 0.5,  # 4-bit = 0.5 bytes per param
}

GEMMA_PARAM_COUNTS = {
    "gemma-2b": 2_000_000_000,
    "gemma-7b": 7_000_000_000,
    "gemma-9b": 9_000_000_000,
    "gemma-27b": 27_000_000_000,
}


def model_memory_gb(param_count: int, dtype: str) -> float:
    """Return approximate GPU memory (GB) for a model in a given dtype."""
    return (param_count * BYTES_PER_PARAM[dtype]) / (1024**3)


def lora_adapter_memory_gb(
    param_count: int, r: int, num_layers: int, num_target_modules: int
) -> float:
    """
    Return approximate memory (GB) for LoRA adapters stored in bfloat16.

    Each target module in each layer gets two matrices:
      A: (r × d)  and  B: (d × r)
    where d ≈ sqrt(param_count / num_layers) as a rough estimate.
    """
    d = int((param_count / num_layers) ** 0.5)
    adapter_params = num_layers * num_target_modules * 2 * r * d
    return (adapter_params * BYTES_PER_PARAM["bfloat16"]) / (1024**3)


# ── TODO(human): implement memory_comparison() ───────────────────────────────


def memory_comparison(
    model_name: str, r: int = 8, num_layers: int = 18, num_target_modules: int = 4
) -> None:
    """
    Print a table comparing GPU memory required to fine-tune `model_name`
    under three strategies:
      1. Full fine-tune in bfloat16  (no LoRA, no quantization)
      2. LoRA only in bfloat16       (LoRA adapters + frozen base in bfloat16)
      3. QLoRA (NF4 + LoRA)          (LoRA adapters in bfloat16 + base in nf4)

    For each strategy print:
      - Base model memory
      - Adapter memory (0 for full fine-tune)
      - Total memory
    """
    param_count = GEMMA_PARAM_COUNTS[model_name]
    adapter_gb = lora_adapter_memory_gb(param_count, r, num_layers, num_target_modules)

    strategies = [
        ("Full fine-tune (bfloat16)", model_memory_gb(param_count, "bfloat16"), 0.0),
        (
            "LoRA only   (bfloat16)",
            model_memory_gb(param_count, "bfloat16"),
            adapter_gb,
        ),
        ("QLoRA       (nf4 + LoRA)", model_memory_gb(param_count, "nf4"), adapter_gb),
    ]

    print(f"\n  {model_name}  (r={r}, {num_target_modules} target modules):")
    print(f"  {'Strategy':<28} {'Base':>7}  {'Adapters':>9}  {'Total':>7}")
    print(f"  {'-' * 56}")
    for name, base_gb, adap_gb in strategies:
        print(
            f"  {name:<28} {base_gb:>6.2f}GB  {adap_gb:>8.4f}GB  {base_gb + adap_gb:>6.2f}GB"
        )


if __name__ == "__main__":
    print("=== Raw dtype memory (no LoRA) ===")
    for model, params in GEMMA_PARAM_COUNTS.items():
        print(f"\n  {model} ({params / 1e9:.0f}B params):")
        for dtype in BYTES_PER_PARAM:
            gb = model_memory_gb(params, dtype)
            fits = {
                8: "RTX 3080",
                16: "RTX 3090/4090",
                24: "RTX 4090",
                40: "A100-40GB",
                80: "A100-80GB",
            }
            fit_msg = next(
                (v for k, v in sorted(fits.items()) if gb <= k), "needs A100-80GB+"
            )
            print(f"    {dtype:10s}: {gb:6.1f} GB  ({fit_msg})")

    print("\n\n=== Fine-tuning strategy comparison ===")
    for model in ["gemma-2b", "gemma-7b"]:
        memory_comparison(model)
