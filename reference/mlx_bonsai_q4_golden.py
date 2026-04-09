#!/usr/bin/env python3
"""MLX golden token capture for Bonsai 1.7B Q4.

Loads the Q4 model via mlx-lm, runs greedy decoding on the
same prompt used by nnmetal's bonsai_q4_golden.zig, and prints
the output token IDs as a Zig array literal for pasting into
the golden test.

This is the ground-truth reference — if MLX and nnmetal
disagree, nnmetal has a bug.

Usage:
  python reference/mlx_bonsai_q4_golden.py
  python reference/mlx_bonsai_q4_golden.py --model /path/to/qwen3-1.7b-q4
  python reference/mlx_bonsai_q4_golden.py --max-tokens 40
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx

# ── Constants (match bonsai_q4_golden.zig exactly) ────────────

GOLDEN_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 20

# Model directory: prefer the nnmetal data/ copy, fall back
# to ~/models/qwen3-1.7b-q4.
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
    """Return the first model directory that exists."""
    if override is not None:
        if not os.path.isdir(override):
            print(
                f"error: model directory not found: "
                f"{override}",
                file=sys.stderr,
            )
            sys.exit(1)
        return override

    for candidate in DEFAULT_MODEL_DIRS:
        resolved = os.path.realpath(candidate)
        if os.path.isdir(resolved):
            return resolved

    print(
        "error: no Q4 model directory found. Tried:\n"
        + "\n".join(f"  {d}" for d in DEFAULT_MODEL_DIRS),
        file=sys.stderr,
    )
    sys.exit(1)


def load_model_and_tokenizer(model_dir: str):
    """Load Qwen3 1.7B Q4 via mlx-lm."""
    from mlx_lm import load

    model, tokenizer = load(model_dir)
    return model, tokenizer


def encode_prompt(tokenizer) -> list[int]:
    """Apply the Qwen3 chat template to the golden prompt.

    Matches bonsai_q4_golden.zig's applyChatTemplate call:
    wraps the user prompt in chat format with
    add_generation_prompt=True and thinking disabled.
    """
    messages = [
        {"role": "user", "content": GOLDEN_PROMPT},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )

    token_ids = tokenizer.encode(prompt_text)
    assert len(token_ids) > 0, "Empty prompt encoding"
    return token_ids


def generate_greedy(
    model,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_token_id: int,
    im_end_token_id: int | None,
) -> list[int]:
    """Greedy decode (temperature=0) and return token IDs.

    Uses mlx_lm's generate_step for parity with how MLX
    benchmarks run internally.
    """
    from mlx_lm.generate import generate_step

    assert max_tokens > 0

    eos_ids = {eos_token_id}
    if im_end_token_id is not None:
        eos_ids.add(im_end_token_id)

    output_ids: list[int] = []

    for _n, (token_id, _logprobs) in enumerate(
        generate_step(
            prompt_tokens,
            model,
            max_tokens=max_tokens + 1,
        ),
    ):
        if token_id in eos_ids:
            break

        output_ids.append(token_id)

        if len(output_ids) >= max_tokens:
            break

    return output_ids


def format_zig_array(token_ids: list[int]) -> str:
    """Format token IDs as a Zig array literal."""
    count = len(token_ids)
    lines = [
        f"const GOLDEN_TOKEN_COUNT: u32 = {count};",
        "",
        "const GOLDEN_TOKENS = [GOLDEN_TOKEN_COUNT]u32{",
    ]

    # Wrap every 10 entries.
    for i in range(0, count, 10):
        chunk = token_ids[i : i + 10]
        parts = []
        for j, tok in enumerate(chunk):
            idx = i + j
            if idx + 1 < count:
                parts.append(f"{tok}, ")
            else:
                parts.append(f"{tok},")
        lines.append("    " + "".join(parts))

    lines.append("};")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MLX golden token capture for "
            "Bonsai 1.7B Q4"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the Q4 model directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate (default: "
        f"{DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)

    print(file=sys.stderr)
    print(
        "\033[1m[mlx] Bonsai 1.7B Q4 Golden Capture"
        "\033[0m",
        file=sys.stderr,
    )
    print(f"model:      {model_dir}", file=sys.stderr)
    print(
        f"prompt:     \"{GOLDEN_PROMPT}\"",
        file=sys.stderr,
    )
    print(
        f"max tokens: {args.max_tokens}",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # ── Load ──

    print(
        "Loading model and tokenizer...",
        file=sys.stderr,
    )
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # ── Encode ──

    prompt_ids = encode_prompt(tokenizer)
    print(
        f"Prompt encoded: {len(prompt_ids)} tokens",
        file=sys.stderr,
    )

    # ── Resolve EOS tokens ──

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None

    # Qwen3 uses <|im_end|> as a second stop token.
    im_end_token_id = None
    if hasattr(tokenizer, "added_tokens_encoder"):
        im_end_token_id = (
            tokenizer.added_tokens_encoder.get(
                "<|im_end|>"
            )
        )

    prompt_tokens = mx.array(prompt_ids)

    # ── Generate ──

    print("Running greedy generation...", file=sys.stderr)

    output_ids = generate_greedy(
        model,
        prompt_tokens,
        max_tokens=args.max_tokens,
        eos_token_id=eos_token_id,
        im_end_token_id=im_end_token_id,
    )

    count = len(output_ids)
    print(
        f"Generated {count}/{args.max_tokens} tokens",
        file=sys.stderr,
    )

    # ── Decode text ──

    generated_text = tokenizer.decode(
        output_ids, skip_special_tokens=False,
    )

    # ── Print results ──

    print(file=sys.stderr)
    print(
        f"Golden output ({count} tokens generated):\n",
    )
    print(format_zig_array(output_ids))
    print(f"\nGenerated text: {generated_text}")

    # Also print raw IDs for quick inspection.
    print(
        f"\nRaw IDs: {output_ids}",
        file=sys.stderr,
    )
    print(file=sys.stderr)


if __name__ == "__main__":
    main()
