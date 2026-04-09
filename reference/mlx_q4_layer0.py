#!/usr/bin/env python3
"""Run embedding + layer 0 manually in MLX Q4 and dump residual.

This script bypasses MLX's full forward pass to capture the exact
intermediate state after layer 0.  The output is formatted for
direct comparison with nnmetal's --diagnose mode.

Usage:
  python reference/mlx_q4_layer0.py
  python reference/mlx_q4_layer0.py --model /path/to/qwen3-1.7b-q4
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GOLDEN_PROMPT = "The capital of France is"

DEFAULT_MODEL_DIRS = [
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "nnmetal",
        "data",
        "qwen3-1.7b-q4",
    ),
    os.path.expanduser("~/models/qwen3-1.7b-q4"),
]

SEP = "=" * 60


def resolve_model_dir(override: str | None) -> str:
    if override is not None:
        if not os.path.isdir(override):
            print(f"error: not found: {override}", file=sys.stderr)
            sys.exit(1)
        return override
    for d in DEFAULT_MODEL_DIRS:
        r = os.path.realpath(d)
        if os.path.isdir(r):
            return r
    print("error: no Q4 model dir found", file=sys.stderr)
    sys.exit(1)


def build_nnmetal_prompt(tokenizer) -> list[int]:
    """Replicate nnmetal's applyChatTemplate exactly."""
    added = getattr(tokenizer, "added_tokens_encoder", {})
    im_start_id = added.get("<|im_start|>")
    im_end_id = added.get("<|im_end|>")
    if im_start_id is None or im_end_id is None:
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    user_nl = tokenizer.encode("user\n", add_special_tokens=False)
    prompt_toks = tokenizer.encode(GOLDEN_PROMPT, add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    assistant_nl = tokenizer.encode("assistant\n", add_special_tokens=False)

    ids = []
    ids.append(im_start_id)
    ids.extend(user_nl)
    ids.extend(prompt_toks)
    ids.append(im_end_id)
    ids.extend(nl)
    ids.append(im_start_id)
    ids.extend(assistant_nl)
    return ids


def dump_array(label: str, arr, n: int = 16):
    """Print first n values of a numpy array."""
    flat = arr.flatten()
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  shape={arr.shape}  dtype={arr.dtype}")
    print(f"  min={float(flat.min()):.6f}  max={float(flat.max()):.6f}"
          f"  mean={float(flat.mean()):.6f}")
    print(SEP)
    for i in range(min(n, len(flat))):
        print(f"  [{i:4d}] = {flat[i]:.6f}")


def dump_top_logits(logits_1d, label: str, top_n: int = 20):
    indices = np.argsort(-logits_1d)[:top_n]
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  shape={logits_1d.shape}  min={logits_1d.min():.4f}"
          f"  max={logits_1d.max():.4f}  mean={logits_1d.mean():.4f}")
    print(SEP)
    print(f"  {'Rank':>4}  {'Index':>8}  {'Logit':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*12}")
    for rank, idx in enumerate(indices):
        print(f"  {rank:4d}  {idx:8d}  {logits_1d[idx]:12.4f}")


def rms_norm_manual(x, weight, eps=1e-6):
    """Manual RMSNorm in float32 for verification."""
    x32 = x.astype(mx.float32)
    ms = mx.mean(x32 * x32, axis=-1, keepdims=True)
    rms_inv = mx.rsqrt(ms + eps)
    normed = x32 * rms_inv
    return (normed * weight.astype(mx.float32)).astype(x.dtype)


def apply_rope_single(q, k, position, head_dim, rope_theta):
    """Apply RoPE to single-position Q and K tensors.

    q: [num_q_heads, head_dim]
    k: [num_kv_heads, head_dim]
    """
    half_dim = head_dim // 2
    freqs = mx.arange(0, half_dim, dtype=mx.float32)
    freqs = 1.0 / (rope_theta ** (freqs / half_dim))
    t = mx.array([position], dtype=mx.float32)
    angles = t[:, None] * freqs[None, :]  # [1, half_dim]
    cos_vals = mx.cos(angles)  # [1, half_dim]
    sin_vals = mx.sin(angles)  # [1, half_dim]

    def rotate(x):
        # x: [num_heads, head_dim]
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        rx1 = x1 * cos_vals - x2 * sin_vals
        rx2 = x2 * cos_vals + x1 * sin_vals
        return mx.concatenate([rx1, rx2], axis=-1)

    return rotate(q), rotate(k)


def apply_qk_norm(x, weight, eps=1e-6):
    """Per-head RMSNorm.  x: [num_heads, head_dim]."""
    x32 = x.astype(mx.float32)
    ms = mx.mean(x32 * x32, axis=-1, keepdims=True)
    rms_inv = mx.rsqrt(ms + eps)
    normed = x32 * rms_inv
    w = weight.astype(mx.float32)
    return (normed * w[None, :]).astype(x.dtype)


def single_head_attention(q, k, v, head_dim):
    """Single-token self-attention (position 0, seq_len=1).

    At position 0 with seq_len=1, attention is trivial:
      score = q . k^T / sqrt(d_k)
      softmax([score]) = [1.0]
      output = v

    But let's compute it properly anyway.
    q: [1, head_dim], k: [1, head_dim], v: [1, head_dim]
    """
    scale = 1.0 / math.sqrt(head_dim)
    # score: [1, 1]
    score = (q @ k.T) * scale
    attn_weights = mx.softmax(score, axis=-1)
    out = attn_weights @ v  # [1, head_dim]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="MLX Q4 layer-0 manual forward pass",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--full-pass",
        action="store_true",
        help="Also run a full model forward pass for comparison",
    )
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)

    from mlx_lm import load

    print(f"\n{SEP}")
    print(f"  MLX Q4 Layer-0 Manual Forward")
    print(f"  Model: {model_dir}")
    print(SEP)

    model, tokenizer = load(model_dir)
    prompt_ids = build_nnmetal_prompt(tokenizer)
    print(f"\nPrompt ({len(prompt_ids)} tokens): {prompt_ids}")

    # Get model components.
    embed_layer = model.model.embed_tokens
    layer0 = model.model.layers[0]
    final_norm = model.model.norm

    # Config values from the model.
    config = model.model.args if hasattr(model.model, "args") else None
    if config is not None:
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // num_q_heads
        rope_theta = config.rope_theta if hasattr(config, "rope_theta") else 1000000.0
        hidden_size = config.hidden_size
        print(f"\n  Config: hidden={hidden_size}, num_q_heads={num_q_heads},"
              f" num_kv_heads={num_kv_heads}, head_dim={head_dim},"
              f" rope_theta={rope_theta}")
    else:
        # Fallback for Qwen3-1.7B.
        hidden_size = 2048
        num_q_heads = 16
        num_kv_heads = 8
        head_dim = 128
        rope_theta = 1000000.0
        print(f"\n  Using fallback config (Qwen3-1.7B)")

    # ── Step 1: Embedding ──
    # Process only token 0 (first prompt token).
    token_0 = prompt_ids[0]
    tok = mx.array([token_0])
    h = embed_layer(tok)  # [1, hidden_size]
    mx.eval(h)
    h_np = np.array(h.astype(mx.float32))
    dump_array(f"Embedding output for token {token_0}", h_np[0])

    # ── Step 2: Layer 0 — Input LayerNorm ──
    attn_norm = layer0.input_layernorm
    norm_out = rms_norm_manual(h, attn_norm.weight)
    mx.eval(norm_out)
    norm_np = np.array(norm_out.astype(mx.float32))
    dump_array("Layer 0: attn norm output (= QKV input)", norm_np[0])

    # ── Step 3: Q/K/V projections ──
    q_proj = layer0.self_attn.q_proj
    k_proj = layer0.self_attn.k_proj
    v_proj = layer0.self_attn.v_proj

    q = q_proj(norm_out)  # [1, query_dim]
    k = k_proj(norm_out)  # [1, kv_dim]
    v = v_proj(norm_out)  # [1, kv_dim]
    mx.eval(q, k, v)

    q_np = np.array(q.astype(mx.float32))
    k_np = np.array(k.astype(mx.float32))
    v_np = np.array(v.astype(mx.float32))

    dump_array("Layer 0: Q projection output", q_np[0])
    dump_array("Layer 0: K projection output", k_np[0])
    dump_array("Layer 0: V projection output", v_np[0])

    # ── Step 4: Q/K head norms ──
    q_norm_w = layer0.self_attn.q_norm.weight if hasattr(layer0.self_attn, "q_norm") else None
    k_norm_w = layer0.self_attn.k_norm.weight if hasattr(layer0.self_attn, "k_norm") else None

    # Reshape to [num_heads, head_dim].
    q_heads = q.reshape(1, num_q_heads, head_dim)  # [1, num_q_heads, head_dim]
    k_heads = k.reshape(1, num_kv_heads, head_dim)  # [1, num_kv_heads, head_dim]
    v_heads = v.reshape(1, num_kv_heads, head_dim)

    if q_norm_w is not None:
        q_normed = apply_qk_norm(q_heads[0], q_norm_w)
        k_normed = apply_qk_norm(k_heads[0], k_norm_w)
        mx.eval(q_normed, k_normed)
        dump_array("Layer 0: Q after head norm", np.array(q_normed.astype(mx.float32))[0])
        dump_array("Layer 0: K after head norm", np.array(k_normed.astype(mx.float32))[0])
    else:
        q_normed = q_heads[0]
        k_normed = k_heads[0]

    # ── Step 5: RoPE ──
    q_roped, k_roped = apply_rope_single(
        q_normed, k_normed,
        position=0,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )
    mx.eval(q_roped, k_roped)
    dump_array("Layer 0: Q after RoPE (head 0)", np.array(q_roped.astype(mx.float32))[0])
    dump_array("Layer 0: K after RoPE (head 0)", np.array(k_roped.astype(mx.float32))[0])

    # ── Step 6: Attention (trivial at position 0 with seq_len=1) ──
    # For GQA: each KV head serves (num_q_heads / num_kv_heads) query heads.
    heads_per_kv = num_q_heads // num_kv_heads
    attn_outputs = []
    for qh in range(num_q_heads):
        kvh = qh // heads_per_kv
        q_h = q_roped[qh:qh+1]   # [1, head_dim]
        k_h = k_roped[kvh:kvh+1]  # [1, head_dim]
        v_h = v_heads[0, kvh:kvh+1]  # [1, head_dim]
        out_h = single_head_attention(q_h, k_h, v_h, head_dim)
        attn_outputs.append(out_h)

    attn_out = mx.concatenate(attn_outputs, axis=-1)  # [1, query_dim]
    attn_out = attn_out.reshape(1, -1)
    mx.eval(attn_out)
    dump_array("Layer 0: attention output (concatenated)", np.array(attn_out.astype(mx.float32))[0])

    # ── Step 7: O projection + residual ──
    o_proj = layer0.self_attn.o_proj
    o_out = o_proj(attn_out)
    mx.eval(o_out)
    dump_array("Layer 0: O projection output", np.array(o_out.astype(mx.float32))[0])

    h_after_attn = h + o_out  # residual add
    mx.eval(h_after_attn)
    dump_array("Layer 0: residual after attention", np.array(h_after_attn.astype(mx.float32))[0])

    # ── Step 8: FFN norm ──
    ffn_norm = layer0.post_attention_layernorm
    ffn_norm_out = rms_norm_manual(h_after_attn, ffn_norm.weight)
    mx.eval(ffn_norm_out)
    dump_array("Layer 0: FFN norm output", np.array(ffn_norm_out.astype(mx.float32))[0])

    # ── Step 9: MLP (gate + up + SiLU + down) ──
    gate_proj = layer0.mlp.gate_proj
    up_proj = layer0.mlp.up_proj
    down_proj = layer0.mlp.down_proj

    gate_out = gate_proj(ffn_norm_out)
    up_out = up_proj(ffn_norm_out)
    mx.eval(gate_out, up_out)
    dump_array("Layer 0: gate projection output", np.array(gate_out.astype(mx.float32))[0])
    dump_array("Layer 0: up projection output", np.array(up_out.astype(mx.float32))[0])

    # SiLU(gate) * up
    silu_gate = mx.sigmoid(gate_out.astype(mx.float32)) * gate_out.astype(mx.float32)
    mlp_intermediate = silu_gate.astype(gate_out.dtype) * up_out
    mx.eval(mlp_intermediate)
    dump_array("Layer 0: SiLU(gate)*up", np.array(mlp_intermediate.astype(mx.float32))[0])

    down_out = down_proj(mlp_intermediate)
    mx.eval(down_out)
    dump_array("Layer 0: down projection output", np.array(down_out.astype(mx.float32))[0])

    h_after_mlp = h_after_attn + down_out  # residual add
    mx.eval(h_after_mlp)
    h_after_mlp_np = np.array(h_after_mlp.astype(mx.float32))
    dump_array("Layer 0: FINAL residual after MLP (= layer 0 output)", h_after_mlp_np[0])

    # ── Step 10: Compare with MLX's own layer 0 ──
    # Run layer 0 through MLX's own call to check our manual
    # computation matches.
    print(f"\n{SEP}")
    print(f"  Cross-check: MLX layer 0 __call__ vs manual")
    print(SEP)

    # MLX layer call needs a mask.  For single token, mask is None.
    mlx_h_after_layer0 = layer0(h, mask=None)
    mx.eval(mlx_h_after_layer0)
    mlx_np = np.array(mlx_h_after_layer0.astype(mx.float32))

    dump_array("MLX layer0(h) output", mlx_np[0])

    diff = np.abs(h_after_mlp_np - mlx_np)
    print(f"\n  Manual vs MLX layer0: max_diff={diff.max():.8f},"
          f" mean_diff={diff.mean():.8f}")

    # ── Optional: full forward pass for comparison ──
    if args.full_pass:
        print(f"\n{SEP}")
        print(f"  Full forward pass (all 13 tokens)")
        print(SEP)

        input_ids = mx.array([prompt_ids])
        logits = model(input_ids)
        mx.eval(logits)
        logits_np = np.array(logits.astype(mx.float32))

        dump_top_logits(logits_np[0, 0], "Full pass: logits at position 0")
        last = logits_np.shape[1] - 1
        dump_top_logits(logits_np[0, last], f"Full pass: logits at last position ({last})")

    # ── Summary ──
    print(f"\n{SEP}")
    print(f"  KEY VALUES FOR COMPARISON WITH NNMETAL")
    print(SEP)
    print(f"  Embedding[0..7] for token {token_0}:")
    emb = h_np[0]
    for i in range(8):
        print(f"    [{i}] = {emb[i]:.6f}")
    print(f"\n  Residual after layer 0 [0..7]:")
    res = h_after_mlp_np[0]
    for i in range(8):
        print(f"    [{i}] = {res[i]:.6f}")
    print(f"\n  Residual after layer 0 stats:")
    print(f"    min={res.min():.6f}  max={res.max():.6f}  mean={res.mean():.6f}")
    print()


if __name__ == "__main__":
    main()
