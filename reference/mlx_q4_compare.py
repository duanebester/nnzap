#!/usr/bin/env python3
"""Compare MLX Q4 logits with nnmetal.

This script:
1. Encodes the prompt using BOTH nnmetal's template and MLX's template
2. Runs a forward pass with MLX using nnmetal's 13-token prompt
3. Dumps the top-20 logits (values + indices) after the first forward pass
4. Generates 20 tokens greedily and prints them
5. Also dumps the first few embedding values for cross-checking

Usage:
  python reference/mlx_q4_compare.py
  python reference/mlx_q4_compare.py --model /path/to/qwen3-1.7b-q4
  python reference/mlx_q4_compare.py --use-mlx-template
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants — must match bonsai_q4_golden.zig exactly
# ---------------------------------------------------------------------------

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
    """Replicate nnmetal's applyChatTemplate exactly:

    <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n

    No system message, no think block.
    """
    # Get special token IDs
    added = getattr(tokenizer, "added_tokens_encoder", {})
    im_start_id = added.get("<|im_start|>")
    im_end_id = added.get("<|im_end|>")

    if im_start_id is None or im_end_id is None:
        # Fallback: try vocab
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    assert im_start_id is not None, "Cannot find <|im_start|>"
    assert im_end_id is not None, "Cannot find <|im_end|>"

    # Encode sub-parts the same way nnmetal does:
    # nnmetal calls tokenizer.encode() on each piece separately.
    user_nl = tokenizer.encode("user\n", add_special_tokens=False)
    prompt_toks = tokenizer.encode(GOLDEN_PROMPT, add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    assistant_nl = tokenizer.encode("assistant\n", add_special_tokens=False)

    ids = []
    ids.append(im_start_id)       # <|im_start|>
    ids.extend(user_nl)           # user\n
    ids.extend(prompt_toks)       # The capital of France is
    ids.append(im_end_id)         # <|im_end|>
    ids.extend(nl)                # \n
    ids.append(im_start_id)       # <|im_start|>
    ids.extend(assistant_nl)      # assistant\n
    return ids


def build_mlx_prompt(tokenizer) -> list[int]:
    """Standard MLX template with enable_thinking=False."""
    messages = [{"role": "user", "content": GOLDEN_PROMPT}]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    return tokenizer.encode(text)


def build_mlx_thinking_prompt(tokenizer) -> list[int]:
    """Standard MLX template with enable_thinking=True."""
    messages = [{"role": "user", "content": GOLDEN_PROMPT}]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    return tokenizer.encode(text)


def dump_top_logits(logits_1d, label: str, top_n: int = 20):
    """Print top-N logit values and their token indices."""
    assert logits_1d.ndim == 1
    indices = np.argsort(-logits_1d)[:top_n]
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  shape: {logits_1d.shape}, dtype: {logits_1d.dtype}")
    print(f"  min={logits_1d.min():.4f}  max={logits_1d.max():.4f}"
          f"  mean={logits_1d.mean():.4f}")
    print(f"{'=' * 60}")
    print(f"  {'Rank':>4}  {'Index':>8}  {'Logit':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*12}")
    for rank, idx in enumerate(indices):
        print(f"  {rank:4d}  {idx:8d}  {logits_1d[idx]:12.4f}")


def dump_embedding_values(model, token_ids, n_dims: int = 16):
    """Dequantize embedding for first token and dump values."""
    import mlx.core as mx

    first_token = token_ids[0]
    print(f"\n{'=' * 60}")
    print(f"  Embedding lookup for token {first_token}")
    print(f"  (first {n_dims} dims)")
    print(f"{'=' * 60}")

    # Access the embedding layer
    embed_layer = model.model.embed_tokens

    # Do a forward pass through just the embedding
    tok_tensor = mx.array([first_token])
    embed_out = embed_layer(tok_tensor)
    mx.eval(embed_out)
    embed_np = np.array(embed_out.astype(mx.float32)[0])

    print(f"  Embedding shape: {embed_np.shape}, dtype: {embed_np.dtype}")
    for i in range(min(n_dims, embed_np.shape[0])):
        print(f"  [{i:4d}] = {embed_np[i]:.6f}")

    # Also dump embedding for last prompt token
    last_token = token_ids[-1]
    tok_tensor2 = mx.array([last_token])
    embed_out2 = embed_layer(tok_tensor2)
    mx.eval(embed_out2)
    embed_np2 = np.array(embed_out2.astype(mx.float32)[0])
    print(f"\n  Embedding for last prompt token {last_token}:")
    for i in range(min(n_dims, embed_np2.shape[0])):
        print(f"  [{i:4d}] = {embed_np2[i]:.6f}")


def dump_q4_weight_info(model):
    """Print Q4 weight metadata for the first layer's q_proj."""
    print(f"\n{'=' * 60}")
    print(f"  Q4 weight info (layer 0, q_proj)")
    print(f"{'=' * 60}")

    import mlx.core as mx

    layer0 = model.model.layers[0]
    q_proj = layer0.self_attn.q_proj

    # MLX QuantizedLinear stores weight, scales, biases
    # Convert via float32 to avoid bfloat16/numpy issues
    w = q_proj.weight
    s = q_proj.scales
    b = q_proj.biases

    w_np = np.array(w.astype(mx.uint32))
    s_np = np.array(s.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))

    print(f"  weight shape: {w.shape}, dtype: {w.dtype}")
    print(f"  scales shape: {s.shape}, dtype: {s.dtype}")
    print(f"  biases shape: {b.shape}, dtype: {b.dtype}")
    print(f"  group_size: {q_proj.group_size}")
    print(f"  bits: {q_proj.bits}")
    print(f"  weight[:2,:4] (packed uint32):\n    {w_np[:2,:4]}")
    print(f"  scales[:2,:4]:\n    {s_np[:2,:4]}")
    print(f"  biases[:2,:4]:\n    {b_np[:2,:4]}")

    # Manually dequantize first row, first 8 elements
    print(f"\n  Manual dequant row 0, cols 0-7:")
    word0 = int(w_np[0, 0])
    group_size = q_proj.group_size
    print(f"  word0 = 0x{word0 & 0xFFFFFFFF:08x}")
    for ni in range(8):
        nib = (word0 >> (ni * 4)) & 0xF
        scale = float(s_np[0, 0])
        bias = float(b_np[0, 0])
        val = scale * nib + bias
        print(f"    col {ni}: nibble={nib:2d}, "
              f"scale={scale:.6f}, bias={bias:.6f}, "
              f"dequant={val:.6f}")

    # Also do a full dequant of row 0 via MLX's own __call__
    # to get the reference output
    test_input = mx.zeros((1, q_proj.weight.shape[1] * 32 // q_proj.bits))
    test_input = test_input.at[0, 0].add(1.0)
    mx.eval(test_input)
    ref_out = q_proj(test_input)
    mx.eval(ref_out)
    ref_np = np.array(ref_out.astype(mx.float32))
    print(f"\n  q_proj(e_0) first 16 values (column 0 of dequantized W):")
    for i in range(min(16, ref_np.shape[1])):
        print(f"    [{i:4d}] = {ref_np[0, i]:.6f}")


def generate_with_logit_dump(
    model, tokenizer, prompt_ids: list[int], max_tokens: int,
    label: str,
):
    """Run greedy generation, dumping logits at each step."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    print(f"\n{'#' * 60}")
    print(f"# GENERATION: {label}")
    print(f"# Prompt tokens ({len(prompt_ids)}): {prompt_ids}")
    print(f"{'#' * 60}")

    # Decode prompt tokens for inspection
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    print(f"# Prompt text: {repr(prompt_text)}")

    # Run full forward pass on prompt to get first logits
    prompt_mx = mx.array([prompt_ids])
    logits = model(prompt_mx)
    mx.eval(logits)
    logits_np = np.array(logits.astype(mx.float32)[0])  # shape [seq_len, vocab_size]

    # Dump logits for the last prompt position (= first generated token's logits)
    last_logits = logits_np[-1]
    dump_top_logits(last_logits, f"{label}: logits at last prompt position")

    # Also dump logits at a few intermediate positions
    for pos in [0, len(prompt_ids) // 2]:
        if pos < logits_np.shape[0]:
            dump_top_logits(
                logits_np[pos],
                f"{label}: logits at position {pos}",
                top_n=10,
            )

    # Now generate greedily
    prompt_tensor = mx.array(prompt_ids)
    added = getattr(tokenizer, "added_tokens_encoder", {})
    eos_id = tokenizer.eos_token_id
    im_end_id = added.get("<|im_end|>")
    eos_ids = {eos_id}
    if im_end_id is not None:
        eos_ids.add(im_end_id)

    output_ids = []
    for _step, (tok, _lp) in enumerate(
        generate_step(
            prompt_tensor, model, max_tokens=max_tokens + 1,
        ),
    ):
        if tok in eos_ids:
            break
        output_ids.append(tok)
        if len(output_ids) >= max_tokens:
            break

    generated_text = tokenizer.decode(output_ids, skip_special_tokens=False)

    print(f"\n  Generated {len(output_ids)} tokens: {output_ids}")
    print(f"  Text: {repr(generated_text)}")

    # Format as Zig array for easy comparison
    print(f"\n  Zig literal:")
    chunks = []
    for i in range(0, len(output_ids), 10):
        chunk = output_ids[i:i+10]
        chunks.append("    " + ", ".join(f"{t}," for t in chunk))
    print("  const TOKENS = [_]u32{")
    for c in chunks:
        print(c)
    print("  };")

    return output_ids


def main():
    parser = argparse.ArgumentParser(
        description="Compare MLX Q4 logits with nnmetal",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--use-mlx-template",
        action="store_true",
        help="Also run with MLX's native template (enable_thinking=False)",
    )
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)

    print(f"\n{'#' * 60}", file=sys.stderr)
    print(f"# MLX Q4 Comparison Tool", file=sys.stderr)
    print(f"# Model: {model_dir}", file=sys.stderr)
    print(f"{'#' * 60}", file=sys.stderr)

    # Load model + tokenizer
    from mlx_lm import load
    model, tokenizer = load(model_dir)

    # Build prompts
    nnmetal_ids = build_nnmetal_prompt(tokenizer)
    mlx_ids = build_mlx_prompt(tokenizer)
    mlx_think_ids = build_mlx_thinking_prompt(tokenizer)

    print(f"\n--- Prompt comparison ---")
    print(f"nnmetal template ({len(nnmetal_ids)} tokens): {nnmetal_ids}")
    print(f"  text: {repr(tokenizer.decode(nnmetal_ids, skip_special_tokens=False))}")
    print(f"mlx template thinking=False ({len(mlx_ids)} tokens): {mlx_ids}")
    print(f"  text: {repr(tokenizer.decode(mlx_ids, skip_special_tokens=False))}")
    print(f"mlx template thinking=True ({len(mlx_think_ids)} tokens): {mlx_think_ids}")
    print(f"  text: {repr(tokenizer.decode(mlx_think_ids, skip_special_tokens=False))}")

    # Dump Q4 weight info
    dump_q4_weight_info(model)

    # Dump embedding for first token
    dump_embedding_values(model, nnmetal_ids)

    # Generate with nnmetal's template (13 tokens)
    nnmetal_output = generate_with_logit_dump(
        model, tokenizer, nnmetal_ids, MAX_TOKENS,
        "nnmetal-template",
    )

    # Generate with MLX thinking=True template (should also be 13 tokens)
    think_output = generate_with_logit_dump(
        model, tokenizer, mlx_think_ids, MAX_TOKENS,
        "mlx-thinking-true",
    )

    # Optionally also with MLX's native template
    if args.use_mlx_template:
        mlx_output = generate_with_logit_dump(
            model, tokenizer, mlx_ids, MAX_TOKENS,
            "mlx-thinking-false",
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  nnmetal template ({len(nnmetal_ids)} tok prompt) -> "
          f"{len(nnmetal_output)} tok output: {nnmetal_output}")
    print(f"  mlx think=True  ({len(mlx_think_ids)} tok prompt) -> "
          f"{len(think_output)} tok output: {think_output}")
    if args.use_mlx_template:
        print(f"  mlx think=False ({len(mlx_ids)} tok prompt) -> "
              f"{len(mlx_output)} tok output: {mlx_output}")


if __name__ == "__main__":
    main()
