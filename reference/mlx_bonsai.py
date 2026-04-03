#!/usr/bin/env python3
"""MLX inference benchmark for Bonsai 1.7B — nnzap comparison.

Mirrors nnzap's nn/examples/bonsai_bench.zig — three phases:

  1. Load the model and tokenizer from a local safetensors directory.
  2. Prefill the prompt (chunked by MLX internally).
  3. Decode tokens one at a time, collecting per-token latencies.

Warmup and measurement are split the same way as the Zig benchmark:
  - 8 warmup decode tokens (timed but excluded from statistics).
  - 64 measured decode tokens (used for throughput and percentiles).

Sampling: greedy (temperature=0) for deterministic, reproducible results.

Prompt (matches bonsai_bench.zig):
  "Explain the theory of general relativity in detail."

Architecture: Qwen3ForCausalLM, 1-bit quantised (g128).

Requires PrismML's MLX fork for 1-bit kernel support:
  pip install "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism"
  pip install mlx-lm

Usage:
  python reference/mlx_bonsai.py
  python reference/mlx_bonsai.py --model /path/to/bonsai-1.7b
  python reference/mlx_bonsai.py --save
  python reference/mlx_bonsai.py --compare "benchmarks/bonsai_bench_*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step

# ── Constants (match bonsai_bench.zig exactly) ────────────────

WARMUP_TOKENS = 8
MEASURE_TOKENS = 64
TOTAL_DECODE_TOKENS = WARMUP_TOKENS + MEASURE_TOKENS

BENCH_PROMPT = (
    "Explain the theory of general relativity in detail."
)

DEFAULT_MODEL_DIR = os.path.expanduser(
    "~/models/bonsai-1.7b"
)


# ── Helpers ───────────────────────────────────────────────────

def nanos_to_us(ns: int) -> float:
    """Convert nanoseconds to microseconds."""
    return ns / 1_000.0


def nanos_to_ms(ns: int) -> float:
    """Convert nanoseconds to milliseconds."""
    return ns / 1_000_000.0


def compute_percentiles(
    latencies_us: list[float],
) -> dict:
    """Compute p50 and p99 using nearest-rank, matching Zig.

    The Zig benchmark sorts the measurement slice and picks:
      p50 = sorted[count / 2]
      p99 = sorted[count * 99 / 100]
    Both use integer (floor) division.
    """
    assert len(latencies_us) > 0

    sorted_us = sorted(latencies_us)
    count = len(sorted_us)

    p50_index = count // 2
    p99_index = (count * 99) // 100

    return {
        "p50_us": sorted_us[p50_index],
        "p99_us": sorted_us[p99_index],
    }


# ── Model loading ─────────────────────────────────────────────

def load_model_and_tokenizer(model_dir: str):
    """Load Bonsai 1.7B via mlx-lm.

    Returns (model, tokenizer) where model is the quantised
    Qwen3 and tokenizer is the HuggingFace fast tokenizer.
    """
    from mlx_lm import load

    model, tokenizer = load(model_dir)
    return model, tokenizer


# ── Tokenisation ──────────────────────────────────────────────

def encode_prompt(tokenizer) -> list[int]:
    """Apply the Qwen3 chat template and return token IDs.

    Matches bonsai_bench.zig's applyChatTemplate call:
    wraps the user prompt in the chat format with
    add_generation_prompt=True and thinking disabled.
    """
    messages = [{"role": "user", "content": BENCH_PROMPT}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )

    token_ids = tokenizer.encode(prompt_text)
    assert len(token_ids) > 0
    return token_ids


# ── Generate (prefill + pipelined decode) ─────────────────────

def run_generate(
    model,
    prompt_tokens: mx.array,
    total_tokens: int,
    eos_token_id: int,
) -> tuple[int, list[tuple[int, int]]]:
    """Prefill + pipelined decode via mlx_lm's generate_step.

    Returns (prefill_ns, decode_timings) where decode_timings
    is a list of (token_id, elapsed_ns) pairs.

    generate_step uses mx.async_eval to pipeline token N+1's
    GPU work while yielding token N — the same code path that
    PrismML benchmarks with.  The wall-clock time between
    successive yields is the effective per-token latency the
    user experiences.

    The first yield boundary marks prefill completion; every
    subsequent yield is one pipelined decode step.
    """
    assert total_tokens > 0

    decode_timings: list[tuple[int, int]] = []

    t_start = time.perf_counter_ns()
    t_prev = t_start
    prefill_ns = 0

    # generate_step handles KV cache creation, chunked
    # prefill, and pipelined decode internally.  With no
    # sampler it defaults to greedy argmax (temperature=0).
    for n, (token_id, _logprobs) in enumerate(
        generate_step(
            prompt_tokens,
            model,
            max_tokens=total_tokens + 1,
        ),
    ):
        t_now = time.perf_counter_ns()

        if n == 0:
            # First yield = prefill complete.  The token
            # itself is the first decode output, but its
            # compute was overlapped with prefill, so we
            # attribute the wall-clock to prefill and start
            # the decode clock fresh.
            prefill_ns = t_now - t_start
        else:
            decode_timings.append(
                (token_id, t_now - t_prev),
            )

        t_prev = t_now

        if token_id == eos_token_id:
            break

        if len(decode_timings) >= total_tokens:
            break

    return prefill_ns, decode_timings


# ── Results computation ───────────────────────────────────────

def compute_results(
    load_ns: int,
    prefill_ns: int,
    prompt_token_count: int,
    decode_timings: list[tuple[int, int]],
    warmup_count: int,
    measure_count: int,
) -> dict:
    """Aggregate statistics from raw timings.

    Mirrors bonsai_bench.zig's computeResults:
      - load_ms from weight-loading wall clock.
      - prefill_tok_per_sec from prompt length / prefill time.
      - decode_tok_per_sec from sum of measured token times.
      - p50/p99 from sorted measured latencies.
    """
    assert prompt_token_count > 0
    assert measure_count > 0

    load_ms = nanos_to_ms(load_ns)
    prefill_ms = nanos_to_ms(prefill_ns)

    prefill_tps = (
        prompt_token_count / (prefill_ms / 1_000.0)
        if prefill_ms > 0.0
        else 0.0
    )

    # Extract measurement-phase latencies (skip warmup).
    generated = len(decode_timings)
    actual_warmup = min(warmup_count, generated)
    actual_measure = min(
        measure_count, generated - actual_warmup,
    )
    assert actual_measure > 0, (
        f"Not enough tokens generated ({generated}) "
        f"for measurement after {actual_warmup} warmup"
    )

    measure_slice = decode_timings[
        actual_warmup : actual_warmup + actual_measure
    ]

    measure_latencies_us = [
        nanos_to_us(ns) for _, ns in measure_slice
    ]
    total_decode_ns = sum(ns for _, ns in measure_slice)
    decode_ms = nanos_to_ms(total_decode_ns)

    decode_tps = (
        actual_measure / (decode_ms / 1_000.0)
        if decode_ms > 0.0
        else 0.0
    )

    percentiles = compute_percentiles(measure_latencies_us)

    return {
        "load_ms": round(load_ms, 1),
        "prefill_tok_per_sec": round(prefill_tps, 1),
        "decode_tok_per_sec": round(decode_tps, 1),
        "decode_p50_us": int(percentiles["p50_us"]),
        "decode_p99_us": int(percentiles["p99_us"]),
        "measured_tokens": actual_measure,
        "prompt_tokens": prompt_token_count,
    }


# ── Output — console summary ─────────────────────────────────

def print_summary(results: dict) -> None:
    """Print benchmark results to stderr, matching Zig output format."""
    assert results["measured_tokens"] > 0
    assert results["prompt_tokens"] > 0

    print(
        f"\n\033[1m--- benchmark results "
        f"----------------------------\033[0m\n"
        f"load:    {results['load_ms']:.1f} ms\n"
        f"prefill: {results['prompt_tokens']} tokens  "
        f"({results['prefill_tok_per_sec']:.1f} tok/s)\n"
        f"decode:  {results['measured_tokens']} tokens  "
        f"({results['decode_tok_per_sec']:.1f} tok/s)\n"
        f"latency: p50={results['decode_p50_us']} us  "
        f"p99={results['decode_p99_us']} us",
        file=sys.stderr,
    )

    peak_bytes = mx.get_peak_memory()
    peak_mb = peak_bytes / (1024 * 1024)
    print(
        f"memory:  {peak_mb:.0f} MB peak",
        file=sys.stderr,
    )
    print(file=sys.stderr)


# ── Output — JSON ─────────────────────────────────────────────

def assemble_json(results: dict) -> dict:
    """Build the full JSON results dict."""
    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return {
        "timestamp_utc": timestamp,
        "framework": "mlx",
        "model": "Bonsai-1.7B",
        "prompt_tokens": results["prompt_tokens"],
        "warmup_tokens": WARMUP_TOKENS,
        "measured_tokens": results["measured_tokens"],
        "load_ms": results["load_ms"],
        "prefill_tok_per_sec": results["prefill_tok_per_sec"],
        "decode_tok_per_sec": results["decode_tok_per_sec"],
        "decode_p50_us": results["decode_p50_us"],
        "decode_p99_us": results["decode_p99_us"],
    }


def save_json(json_dict: dict) -> Path:
    """Write results to benchmarks/bonsai_mlx_<timestamp>.json."""
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)

    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H-%M-%SZ")
    )
    out_path = (
        benchmarks_dir / f"bonsai_mlx_{timestamp}.json"
    )

    with open(out_path, "w") as f:
        json.dump(json_dict, f, indent=2)
        f.write("\n")

    return out_path


# ── Comparison ────────────────────────────────────────────────

def compare_benchmarks(
    mlx_results: dict,
    file_paths: list[str],
) -> None:
    """Print a side-by-side comparison table against nnzap results."""
    benchmarks = {"mlx": mlx_results}

    for path in sorted(file_paths):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"Warning: could not load {path}: {exc}",
                file=sys.stderr,
            )
            continue

        framework = data.get("framework", "nnzap")
        if framework in benchmarks:
            framework = f"{framework} ({Path(path).name})"
        benchmarks[framework] = data

    if len(benchmarks) < 2:
        print(
            "No valid benchmark files to compare against.",
            file=sys.stderr,
        )
        return

    names = list(benchmarks.keys())
    col_w = 16

    sep = "-" * (32 + col_w * len(names))
    header = f"{'Metric':<32}" + "".join(
        f"{n:>{col_w}}" for n in names
    )

    print(file=sys.stderr)
    print(
        f"{'Bonsai 1.7B benchmark comparison':^{len(sep)}}",
        file=sys.stderr,
    )
    print(sep, file=sys.stderr)
    print(header, file=sys.stderr)
    print(sep, file=sys.stderr)

    rows = [
        ("Load (ms)", "load_ms", ".1f"),
        ("Prefill (tok/s)", "prefill_tok_per_sec", ".1f"),
        ("Decode (tok/s)", "decode_tok_per_sec", ".1f"),
        ("Decode p50 (us)", "decode_p50_us", "d"),
        ("Decode p99 (us)", "decode_p99_us", "d"),
        ("Prompt tokens", "prompt_tokens", "d"),
        ("Measured tokens", "measured_tokens", "d"),
    ]

    for label, key, fmt in rows:
        cells = ""
        for name in names:
            bench = benchmarks[name]
            val = bench.get(key, 0)
            cells += f"{val:>{col_w}{fmt}}"
        print(f"{label:<32}{cells}", file=sys.stderr)

    print(sep, file=sys.stderr)
    print(file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MLX Bonsai 1.7B inference benchmark — "
            "mirrors nnzap's bonsai_bench.zig"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_DIR,
        help=(
            "Path to the Bonsai 1.7B model directory "
            f"(default: {DEFAULT_MODEL_DIR})"
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to benchmarks/ as JSON",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help=(
            "Glob pattern for benchmark JSONs to compare "
            "(e.g. 'benchmarks/bonsai_bench_*.json')"
        ),
    )
    args = parser.parse_args()

    # ── Banner ──

    print(file=sys.stderr)
    print(
        "\033[1m[mlx] Bonsai 1.7B Benchmark\033[0m",
        file=sys.stderr,
    )
    print(
        f"model:          {args.model}",
        file=sys.stderr,
    )
    print(
        f"warmup tokens:  {WARMUP_TOKENS}",
        file=sys.stderr,
    )
    print(
        f"measure tokens: {MEASURE_TOKENS}",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # ── Load model ──

    print("Loading model and tokenizer...", file=sys.stderr)

    load_start_ns = time.perf_counter_ns()
    model, tokenizer = load_model_and_tokenizer(args.model)
    load_elapsed_ns = time.perf_counter_ns() - load_start_ns

    print(
        f"Loaded in {nanos_to_ms(load_elapsed_ns):.1f} ms",
        file=sys.stderr,
    )

    # ── Encode prompt ──

    prompt_ids = encode_prompt(tokenizer)
    prompt_token_count = len(prompt_ids)
    assert prompt_token_count > 0

    print(
        f"Prompt: {prompt_token_count} tokens",
        file=sys.stderr,
    )

    # Resolve EOS token ID.
    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None

    prompt_tokens = mx.array(prompt_ids)

    # ── Prefill ──

    print(file=sys.stderr)
    print(
        f"Running generate "
        f"({WARMUP_TOKENS} warmup + "
        f"{MEASURE_TOKENS} measure)...",
        file=sys.stderr,
    )

    # ── Prefill + pipelined decode ──

    prefill_ns, decode_timings = run_generate(
        model,
        prompt_tokens,
        total_tokens=TOTAL_DECODE_TOKENS,
        eos_token_id=eos_token_id,
    )

    generated_count = len(decode_timings)
    warmup_count = min(WARMUP_TOKENS, generated_count)
    measure_count = min(
        MEASURE_TOKENS, generated_count - warmup_count,
    )

    if measure_count == 0:
        print(
            f"error: not enough tokens generated "
            f"({generated_count}) for measurement",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Compute statistics ──

    results = compute_results(
        load_ns=load_elapsed_ns,
        prefill_ns=prefill_ns,
        prompt_token_count=prompt_token_count,
        decode_timings=decode_timings,
        warmup_count=warmup_count,
        measure_count=measure_count,
    )

    # ── Print summary to stderr ──

    print_summary(results)

    # ── JSON to stdout ──

    json_dict = assemble_json(results)
    print(json.dumps(json_dict, indent=2))

    # ── Save ──

    if args.save:
        out_path = save_json(json_dict)
        print(
            f"Results written to {out_path}",
            file=sys.stderr,
        )

    # ── Compare ──

    if args.compare is not None:
        paths = glob.glob(args.compare)
        if not paths:
            print(
                f"Warning: no files matched "
                f"'{args.compare}'",
                file=sys.stderr,
            )
        else:
            compare_benchmarks(json_dict, paths)


if __name__ == "__main__":
    main()
