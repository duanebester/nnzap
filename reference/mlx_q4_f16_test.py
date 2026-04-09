#!/usr/bin/env python3
"""Test whether MLX Q4 inference degrades when forced to float16.

The Q4 model stores scales/biases as bfloat16 and MLX computes the
entire forward pass in bfloat16.  nnmetal converts everything to
float16 and computes in float16.  This script tests whether float16
computation is the root cause of nnmetal's wrong output.

Three runs:
  1. Native bfloat16 (baseline — should produce correct output)
  2. Model cast to float16 (does output degrade?)
  3. Model cast to float32 (sanity check — should be correct)

Usage:
  python reference/mlx_q4_f16_test.py
  python reference/mlx_q4_f16_test.py --model /path/to/qwen3-1.7b-q4
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import numpy as np

GOLDEN_PROMPT = "The capital of France is"
MAX_TOKENS = 20

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


def dump_top_logits(logits_1d, label: str, top_n: int = 10):
    arr = np.asarray(logits_1d, dtype=np.float32)
    indices = np.argsort(-arr)[:top_n]
    print(f"\n  {label}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
    print(f"  {'Rank':>4}  {'Index':>8}  {'Logit':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*12}")
    for rank, idx in enumerate(indices):
        print(f"  {rank:4d}  {idx:8d}  {arr[idx]:12.4f}")


def generate_greedy(model, tokenizer, prompt_ids, max_tokens):
    from mlx_lm.generate import generate_step

    added = getattr(tokenizer, "added_tokens_encoder", {})
    eos_id = tokenizer.eos_token_id
    im_end_id = added.get("<|im_end|>")
    eos_ids = {eos_id}
    if im_end_id is not None:
        eos_ids.add(im_end_id)

    prompt_tensor = mx.array(prompt_ids)
    output_ids = []
    for _step, (tok, _lp) in enumerate(
        generate_step(prompt_tensor, model, max_tokens=max_tokens + 1),
    ):
        if tok in eos_ids:
            break
        output_ids.append(tok)
        if len(output_ids) >= max_tokens:
            break
    return output_ids


def report_model_dtypes(model, label: str):
    """Print dtype of key model components."""
    print(f"\n  --- {label}: dtype report ---")
    embed = model.model.embed_tokens
    layer0 = model.model.layers[0]
    norm = model.model.norm

    # Embedding
    if hasattr(embed, "weight"):
        w = embed.weight
        if hasattr(w, "dtype"):
            print(f"  embed_tokens.weight dtype: {w.dtype}")
    if hasattr(embed, "scales"):
        print(f"  embed_tokens.scales dtype: {embed.scales.dtype}")

    # Layer 0 q_proj
    q = layer0.self_attn.q_proj
    if hasattr(q, "scales"):
        print(f"  layer0.q_proj.scales dtype: {q.scales.dtype}")
    if hasattr(q, "biases"):
        print(f"  layer0.q_proj.biases dtype: {q.biases.dtype}")

    # Norm
    print(f"  layer0.input_layernorm.weight dtype: {layer0.input_layernorm.weight.dtype}")
    print(f"  final_norm.weight dtype: {norm.weight.dtype}")


def run_test(model, tokenizer, prompt_ids, label: str):
    """Run forward pass and generation, report results."""
    print(f"\n{SEP}")
    print(f"  TEST: {label}")
    print(SEP)

    report_model_dtypes(model, label)

    # Forward pass for logits.
    input_ids = mx.array([prompt_ids])
    logits = model(input_ids)
    mx.eval(logits)
    logits_np = np.array(logits.astype(mx.float32))

    # Position 0 logits.
    dump_top_logits(logits_np[0, 0], f"{label}: logits at position 0")

    # Last position logits.
    last_pos = logits_np.shape[1] - 1
    dump_top_logits(logits_np[0, last_pos], f"{label}: logits at last position ({last_pos})")

    # Check: what's the logit for token 151667 (<think>) at last position?
    think_logit = logits_np[0, last_pos, 151667] if logits_np.shape[2] > 151667 else float("nan")
    top_token = int(np.argmax(logits_np[0, last_pos]))
    top_logit = float(logits_np[0, last_pos, top_token])
    print(f"\n  Token 151667 (<think>) logit at last pos: {think_logit:.4f}")
    print(f"  Top token at last pos: {top_token} (logit {top_logit:.4f})")

    # Generate.
    output_ids = generate_greedy(model, tokenizer, prompt_ids, MAX_TOKENS)
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    print(f"\n  Generated {len(output_ids)} tokens: {output_ids}")
    print(f"  Text: {repr(text)}")

    return output_ids, logits_np


def cast_non_quantized_params(model, dtype):
    """Cast all non-quantized parameters to the given dtype.

    QuantizedLinear and QuantizedEmbedding store weight as uint32
    and scales/biases as float — we cast scales, biases, and all
    other float parameters (norms, etc.) to the target dtype.
    """
    import mlx.nn as nn

    def _cast_module(module):
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            # Cast scales and biases but leave weight (uint32) alone.
            if hasattr(module, "scales") and module.scales.dtype != dtype:
                module.scales = module.scales.astype(dtype)
            if hasattr(module, "biases") and module.biases is not None and module.biases.dtype != dtype:
                module.biases = module.biases.astype(dtype)
        elif isinstance(module, nn.RMSNorm):
            if module.weight.dtype != dtype:
                module.weight = module.weight.astype(dtype)
        elif hasattr(module, "weight") and isinstance(module.weight, mx.array):
            if module.weight.dtype not in (mx.uint32, mx.uint8, mx.int32) and module.weight.dtype != dtype:
                module.weight = module.weight.astype(dtype)
            if hasattr(module, "bias") and module.bias is not None and module.bias.dtype != dtype:
                module.bias = module.bias.astype(dtype)

    def _recurse(mod):
        _cast_module(mod)
        # Recurse into children.
        if hasattr(mod, "layers"):
            for layer in mod.layers:
                _recurse(layer)
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(mod, attr_name)
            except Exception:
                continue
            if isinstance(attr, nn.Module) and attr is not mod:
                _cast_module(attr)
                # Go one more level for attention sub-modules.
                for sub_name in dir(attr):
                    if sub_name.startswith("_"):
                        continue
                    try:
                        sub = getattr(attr, sub_name)
                    except Exception:
                        continue
                    if isinstance(sub, nn.Module) and sub is not attr:
                        _cast_module(sub)

    _recurse(model)
    if hasattr(model, "model"):
        _recurse(model.model)
    mx.eval(model.parameters())


def main():
    parser = argparse.ArgumentParser(
        description="Test MLX Q4 in float16 vs bfloat16",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)

    print(f"\n{SEP}")
    print(f"  MLX Q4 Float16 vs BFloat16 Test")
    print(f"  Model: {model_dir}")
    print(SEP)

    from mlx_lm import load

    # ── Test 1: Native bfloat16 (baseline) ──
    print("\n\nLoading model (native bfloat16)...", file=sys.stderr)
    model_bf16, tokenizer = load(model_dir)
    prompt_ids = build_nnmetal_prompt(tokenizer)
    print(f"Prompt ({len(prompt_ids)} tokens): {prompt_ids}")

    bf16_ids, bf16_logits = run_test(
        model_bf16, tokenizer, prompt_ids, "BFLOAT16 (native)"
    )

    # ── Test 2: Cast to float16 ──
    print("\n\nCasting model to float16...", file=sys.stderr)
    # Reload to get a clean copy.
    model_f16, _ = load(model_dir)
    cast_non_quantized_params(model_f16, mx.float16)

    f16_ids, f16_logits = run_test(
        model_f16, tokenizer, prompt_ids, "FLOAT16 (cast)"
    )

    # ── Test 3: Cast to float32 (sanity) ──
    print("\n\nCasting model to float32...", file=sys.stderr)
    model_f32, _ = load(model_dir)
    cast_non_quantized_params(model_f32, mx.float32)

    f32_ids, f32_logits = run_test(
        model_f32, tokenizer, prompt_ids, "FLOAT32 (cast)"
    )

    # ── Summary ──
    print(f"\n{SEP}")
    print(f"  SUMMARY")
    print(SEP)

    def tokens_str(ids):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        return f"{len(ids)} tokens: {repr(text[:80])}"

    print(f"  BF16:  {tokens_str(bf16_ids)}")
    print(f"  F16:   {tokens_str(f16_ids)}")
    print(f"  F32:   {tokens_str(f32_ids)}")
    print(f"\n  BF16 == F16?  {bf16_ids == f16_ids}")
    print(f"  BF16 == F32?  {bf16_ids == f32_ids}")
    print(f"  F16  == F32?  {f16_ids == f32_ids}")

    # Logit comparison at last position.
    last = bf16_logits.shape[1] - 1
    bf16_last = bf16_logits[0, last].astype(np.float64)
    f16_last = f16_logits[0, last].astype(np.float64)
    f32_last = f32_logits[0, last].astype(np.float64)

    def logit_diff(a, b, name_a, name_b):
        diff = np.abs(a - b)
        max_diff = diff.max()
        mean_diff = diff.mean()
        # Correlation.
        corr = np.corrcoef(a, b)[0, 1]
        print(f"  {name_a} vs {name_b}: max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}  corr={corr:.6f}")

    print(f"\n  Logit differences at last position ({last}):")
    logit_diff(bf16_last, f16_last, "BF16", "F16")
    logit_diff(bf16_last, f32_last, "BF16", "F32")
    logit_diff(f16_last, f32_last, "F16", "F32")

    print()


if __name__ == "__main__":
    main()
