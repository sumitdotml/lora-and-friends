"""
LoRA Target Modules - Crash Course
Explore which layers in a transformer get LoRA adapters and why.
"""

# ── Simulated named_modules() output for a 3-block Gemma-style model ──────────
# In practice: dict(model.named_modules()) gives you something like this.
# Values are the class name of each module.
GEMMA_MODULES = {
    "embed_tokens": "Embedding",
    "model.norm": "RMSNorm",
    "model.layers.0.self_attn.q_proj": "Linear",
    "model.layers.0.self_attn.k_proj": "Linear",
    "model.layers.0.self_attn.v_proj": "Linear",
    "model.layers.0.self_attn.o_proj": "Linear",
    "model.layers.0.mlp.gate_proj": "Linear",
    "model.layers.0.mlp.up_proj": "Linear",
    "model.layers.0.mlp.down_proj": "Linear",
    "model.layers.0.input_layernorm": "RMSNorm",
    "model.layers.0.post_feedforward_layernorm": "RMSNorm",
    "model.layers.1.self_attn.q_proj": "Linear",
    "model.layers.1.self_attn.k_proj": "Linear",
    "model.layers.1.self_attn.v_proj": "Linear",
    "model.layers.1.self_attn.o_proj": "Linear",
    "model.layers.1.mlp.gate_proj": "Linear",
    "model.layers.1.mlp.up_proj": "Linear",
    "model.layers.1.mlp.down_proj": "Linear",
    "model.layers.1.input_layernorm": "RMSNorm",
    "model.layers.1.post_feedforward_layernorm": "RMSNorm",
    "model.layers.2.self_attn.q_proj": "Linear",
    "model.layers.2.self_attn.k_proj": "Linear",
    "model.layers.2.self_attn.v_proj": "Linear",
    "model.layers.2.self_attn.o_proj": "Linear",
    "model.layers.2.mlp.gate_proj": "Linear",
    "model.layers.2.mlp.up_proj": "Linear",
    "model.layers.2.mlp.down_proj": "Linear",
    "model.layers.2.input_layernorm": "RMSNorm",
    "model.layers.2.post_feedforward_layernorm": "RMSNorm",
    "lm_head": "Linear",
}


def find_linear_modules(modules: dict[str, str]) -> list[str]:
    """
    Given a {full_path: class_name} dict of all model modules,
    return the unique leaf-level names of all Linear layers
    (i.e. the part after the last dot: 'q_proj', 'gate_proj', etc.)
    Exclude 'lm_head' — we never apply LoRA to the output head.

    Example: "model.layers.0.self_attn.q_proj" → "q_proj"
    """
    return sorted(
        {
            module.split(".")[-1]
            for module in modules
            if module != "lm_head" and modules[module] == "Linear"
        }
    )


if __name__ == "__main__":
    linear_names = find_linear_modules(GEMMA_MODULES)
    if linear_names:
        print("=== All Linear layer names (LoRA candidates) ===")
        for name in linear_names:
            print(f"  {name}")

        attn_only = [
            n
            for n in linear_names
            if "_proj" in n and n not in ("gate_proj", "up_proj", "down_proj")
        ]
        all_linear = linear_names

        print("\n=== Config Option A — Attention only ===")
        print(f"  target_modules={attn_only}")
        print(
            f"  Adapter count: {sum(1 for p, t in GEMMA_MODULES.items() if t == 'Linear' and p.split('.')[-1] in attn_only and p != 'lm_head')}"
        )

        print("\n=== Config Option B — Attention + FFN ===")
        print(f"  target_modules={all_linear}")
        print(
            f"  Adapter count: {sum(1 for p, t in GEMMA_MODULES.items() if t == 'Linear' and p.split('.')[-1] in all_linear and p != 'lm_head')}"
        )
