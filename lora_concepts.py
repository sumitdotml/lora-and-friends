"""
LoRA Concepts - Crash Course
Demonstrates the math behind Low-Rank Adaptation.
"""

import numpy as np

# ── Simulated pretrained weight matrix ────────────────────────────────────────
# In a real transformer, d might be 4096. We use 8 for readability.
d = 8
r = 2  # LoRA rank — much smaller than d

W_pretrained = np.random.randn(d, d)  # frozen during fine-tuning

# ── LoRA adapter initialization ────────────────────────────────────────────────
# A: projects input DOWN to rank r  (shape: r × d_in)
# B: projects back UP to d_out      (shape: d_out × r)
A = np.random.randn(r, d) * 0.01  # small random init
B = np.zeros((d, r))  # zero init → ΔW starts at 0

alpha = 4  # scaling hyperparameter
scaling = alpha / r

# ── Forward pass comparison ───────────────────────────────────────────────────
x = np.random.randn(d)  # input vector

# Original pretrained output
out_pretrained = W_pretrained @ x

# LoRA output: W_pretrained stays frozen, adapter adds residual
delta_W = scaling * (B @ A)  # ΔW = (alpha/r) × B × A
W_adapted = W_pretrained + delta_W  # combined weight (done at inference/merge)
out_lora = W_adapted @ x

# Equivalent (and more efficient) — don't materialize ΔW, apply separately:
out_lora_efficient = (W_pretrained @ x) + scaling * (B @ (A @ x))

print("=== LoRA Weight Decomposition ===")
print(f"W shape:     {W_pretrained.shape}  → {d * d} params  (frozen)")
print(f"A shape:     {A.shape}  → {r * d} params  (trainable)")
print(f"B shape:     {B.shape}  → {d * r} params  (trainable)")
print(
    f"Total LoRA params: {r * d + d * r}  vs  {d * d} full  "
    f"({100 * (r * d + d * r) / (d * d):.1f}%)"
)

print(f"\nalpha={alpha}, r={r}, scaling factor={scaling}")
print(f"\nΔW at init (should be ~0): max_abs = {np.abs(delta_W).max():.6f}")

print("\n=== Forward Pass ===")
print(f"Pretrained output:      {out_pretrained.round(4)}")
print(f"LoRA output (merged):   {out_lora.round(4)}")
print(f"LoRA output (efficient):{out_lora_efficient.round(4)}")
print(f"Outputs match: {np.allclose(out_lora, out_lora_efficient)}")

# ── Parameter count at real scale ─────────────────────────────────────────────
print("\n=== Real Gemma Scale (d=2048, r=8) ===")
d_real, r_real = 2048, 8
full_params = d_real * d_real
lora_params = r_real * d_real + d_real * r_real  # A + B
print(f"Full W params:  {full_params:,}")
print(f"LoRA params:    {lora_params:,}  ({100 * lora_params / full_params:.2f}%)")


# ── TODO(human): implement rank_expressiveness() ──────────────────────────────
# See the Learn by Doing section below before implementing this.


def rank_expressiveness(d: int, ranks: list[int]) -> dict[int, float]:
    """
    For each rank r in `ranks`, compute what percentage of a (d x d)
    weight matrix can be represented by a rank-r LoRA decomposition.

    Returns a dict mapping rank → percentage of params vs full matrix.
    """
    # r * d = total parameters in each A & B, hence also multiplied by 2
    percentage = [((r * d * 2) / (d * d)) * 100 for r in ranks]
    return {r: p for r, p in zip(ranks, percentage)}


if __name__ == "__main__":
    result = rank_expressiveness(4096, [1, 2, 4, 8, 16, 32, 64])
    if result:
        print("\n=== Rank vs Parameter % (d=4096) ===")
        for rank, pct in result.items():
            bar = "█" * int(pct * 2)
            print(f"  r={rank:3d}: {pct:5.2f}%  {bar}")
