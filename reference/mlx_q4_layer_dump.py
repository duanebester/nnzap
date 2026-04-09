#!/usr/bin/env python3
"""Dump intermediate layer outputs from MLX Q4 model.

Hooks into the MLX model to capture activations at key points
in the forward pass for comparison with nnmetal diagnostics.

Dumps:
  - Embedding output for first and last token
  - Residual after layer 0 (position 0 and last)
  - Residual after all layers (= final norm input)
  - Final norm output (= LM head input)
  - Logits (top 20 at position 0 and last position)

Usage:
  python reference/mlx_q4_layer_dump.py
  python reference/mlx_q4_layer_dump.py --model /path/to/qwen3-1.7b-q4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

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

GOLDEN_PROMPT = "The capital of France is"
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
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  shape={arr.shape}  dtype={arr.dtype}")
    print(f"  min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}")
    print(SEP)
    flat = arr.flatten()
    for i in range(min(n, len(flat))):
        print(f"  [{i:4d}] = {flat[i]:.6f}")


def dump_top_logits(logits_1d, label: str, top_n: int = 20):
    """Print top-N logit values and their token indices."""
    indices = np.argsort(-logits_1d)[:top_n]
    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  shape={logits_1d.shape}  dtype={logits_1d.dtype}")
    print(f"  min={logits_1d.min():.4f}  max={logits_1d.max():.4f}"
          f"  mean={logits_1d.mean():.4f}")
    print(SEP)
    print(f"  {'Rank':>4}  {'Index':>8}  {'Logit':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*12}")
    for rank, idx in enumerate(indices):
        print(f"  {rank:4d}  {idx:8d}  {logits_1d[idx]:12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump MLX Q4 intermediate activations",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)

    import mlx.core as mx
    from mlx_lm import load

    print(f"\n{SEP}")
    print(f"  MLX Q4 Layer Dump")
    print(f"  Model: {model_dir}")
    print(SEP)

    model, tokenizer = load(model_dir)

    # Build prompt (same as nnmetal).
    prompt_ids = build_nnmetal_prompt(tokenizer)
    print(f"\nPrompt ({len(prompt_ids)} tokens): {prompt_ids}")
    print(f"Text: {repr(tokenizer.decode(prompt_ids, skip_special_tokens=False))}")

    # Report model dtype info.
    print(f"\n--- Model info ---")
    embed_layer = model.model.embed_tokens
    layer0 = model.model.layers[0]
    q_proj = layer0.self_attn.q_proj
    print(f"  embed_tokens type: {type(embed_layer).__name__}")
    print(f"  q_proj type: {type(q_proj).__name__}")
    print(f"  q_proj.weight dtype: {q_proj.weight.dtype}")
    print(f"  q_proj.scales dtype: {q_proj.scales.dtype}")
    print(f"  q_proj.bits: {q_proj.bits}")
    print(f"  q_proj.group_size: {q_proj.group_size}")

    norm_w = model.model.layers[0].input_layernorm.weight
    print(f"  input_layernorm.weight dtype: {norm_w.dtype}")
    final_norm_w = model.model.norm.weight
    print(f"  final_norm.weight dtype: {final_norm_w.dtype}")

    # ── Step 1: Embedding output ──
    tok_tensor = mx.array(prompt_ids)
    embed_out = embed_layer(tok_tensor)
    mx.eval(embed_out)
    embed_np = np.array(embed_out.astype(mx.float32))
    print(f"\nEmbedding output shape: {embed_np.shape}, dtype: {embed_out.dtype}")

    dump_array(
        f"Embedding output for token {prompt_ids[0]} (position 0)",
        embed_np[0],
    )

    if len(prompt_ids) > 1:
        dump_array(
            f"Embedding output for token {prompt_ids[-1]} (last position)",
            embed_np[-1],
        )

    # ── Step 2: Manual forward pass, capturing intermediates ──
    # MLX Qwen3 model forward is roughly:
    #   h = embed(tokens)          # [seq_len, hidden]
    #   for layer in layers:
    #       h = layer(h, ...)
    #   h = norm(h)
    #   logits = lm_head(h)
    #
    # We replicate this manually to capture intermediates.
    # The tricky part is the attention mask and cache — for a
    # single forward pass without cache, we can call the model's
    # internal pieces directly.

    # Method: monkey-patch the model's forward to intercept at
    # the right points.  Qwen3 model class has a __call__ that
    # goes through self.model(...) then self.lm_head(...).
    # self.model is a Qwen3Model whose __call__ does:
    #   h = self.embed_tokens(inputs)
    #   mask = create_attention_mask(h, cache)
    #   for layer in self.layers:
    #       h = layer(h, mask, cache)
    #   return self.norm(h)

    captured = {}

    # Patch: intercept between layers and final norm by
    # replacing the norm itself.
    orig_norm_weight = model.model.norm.weight
    orig_norm_forward = type(model.model.norm).__call__

    def patched_norm_call(self_norm, x):
        captured["final_norm_input"] = x
        result = orig_norm_forward(self_norm, x)
        captured["final_norm_output"] = result
        return result

    type(model.model.norm).__call__ = patched_norm_call

    # Also patch layer 0 to capture its output.
    orig_layer0_forward = type(layer0).__call__

    def patched_layer0_call(self_layer, x, *a, **kw):
        captured["layer0_input"] = x
        result = orig_layer0_forward(self_layer, x, *a, **kw)
        captured["layer0_output"] = result
        return result

    # Only patch the instance, not the class (to avoid
    # affecting other layers).  Use bound method trick.
    import types
    layer0.__call__ = types.MethodType(
        lambda self, x, *a, **kw: patched_layer0_call(self, x, *a, **kw),
        layer0,
    )

    # Run full forward pass (single sequence, no cache).
    input_ids = mx.array([prompt_ids])
    logits = model(input_ids)
    mx.eval(logits)

    # Eval captured tensors.
    for k, v in captured.items():
        mx.eval(v)

    # Restore patched norm.
    type(model.model.norm).__call__ = orig_norm_forward
    # Restore layer 0 (delete instance override).
    if hasattr(layer0, "__call__"):
        try:
            del layer0.__call__
        except AttributeError:
            pass

    # ── Dump captured intermediates ──

    if "layer0_input" in captured:
        inp = np.array(captured["layer0_input"].astype(mx.float32))
        ndim = inp.ndim
        print(f"\n  [layer0_input captured: shape={inp.shape}]")
        pos0 = inp[0, 0] if ndim == 3 else inp[0]
        dump_array("Layer 0 input (= embedding), position 0", pos0)
        if ndim == 3 and inp.shape[1] > 1:
            dump_array(
                f"Layer 0 input, last position ({inp.shape[1]-1})",
                inp[0, -1],
            )
    else:
        print("\n  [layer0_input NOT captured]")

    if "layer0_output" in captured:
        out = np.array(captured["layer0_output"].astype(mx.float32))
        ndim = out.ndim
        print(f"\n  [layer0_output captured: shape={out.shape}]")
        pos0 = out[0, 0] if ndim == 3 else out[0]
        dump_array("Layer 0 output (residual after layer 0), position 0", pos0)
        if ndim == 3 and out.shape[1] > 1:
            dump_array(
                f"Layer 0 output, last position ({out.shape[1]-1})",
                out[0, -1],
            )
    else:
        print("\n  [layer0_output NOT captured]")

    if "final_norm_input" in captured:
        fn_in = np.array(captured["final_norm_input"].astype(mx.float32))
        ndim = fn_in.ndim
        print(f"\n  [final_norm_input captured: shape={fn_in.shape}]")
        pos0 = fn_in[0, 0] if ndim == 3 else fn_in[0]
        dump_array("Residual after all 28 layers, position 0", pos0)
        if ndim == 3 and fn_in.shape[1] > 1:
            dump_array(
                f"Residual after all 28 layers, last position ({fn_in.shape[1]-1})",
                fn_in[0, -1],
            )
    else:
        print("\n  [final_norm_input NOT captured]")

    if "final_norm_output" in captured:
        fn_out = np.array(captured["final_norm_output"].astype(mx.float32))
        ndim = fn_out.ndim
        print(f"\n  [final_norm_output captured: shape={fn_out.shape}]")
        pos0 = fn_out[0, 0] if ndim == 3 else fn_out[0]
        dump_array("Final norm output (= LM head input), position 0", pos0)
        if ndim == 3 and fn_out.shape[1] > 1:
            dump_array(
                f"Final norm output, last position ({fn_out.shape[1]-1})",
                fn_out[0, -1],
            )
    else:
        print("\n  [final_norm_output NOT captured]")

    # ── Logits ──
    logits_np = np.array(logits.astype(mx.float32))

    dump_top_logits(
        logits_np[0, 0] if logits_np.ndim == 3 else logits_np[0],
        "Logits at position 0",
    )

    last_pos = logits_np.shape[1] - 1 if logits_np.ndim == 3 else 0
    dump_top_logits(
        logits_np[0, last_pos] if logits_np.ndim == 3 else logits_np[-1],
        f"Logits at last position ({last_pos})",
    )

    # ── Summary ──
    print(f"\n{SEP}")
    print(f"  DONE — compare with: zig build run-bonsai-q4-golden -- --diagnose")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
