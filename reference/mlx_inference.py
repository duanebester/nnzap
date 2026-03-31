#!/usr/bin/env python3
"""MLX inference benchmark for nnzap comparison.

Mirrors nnzap's nn/examples/inference_bench.zig — three phases:

  Phase 1 — GPU batched throughput
    Forward-pass 10,240 samples in batches of 64 using MLX's
    Metal GPU backend.  Measures total time and images/sec.

  Phase 2 — GPU single-sample latency
    Forward-pass 1 sample, mx.eval() to force synchronisation,
    repeat 1,000 times.  Reports p50/p99/mean/min/max in us.

  Phase 3 — CPU single-sample latency
    Forward-pass 1 sample using pure NumPy (no MLX, no Metal).
    Repeat 1,000 times.  Reports p50/p99/mean/min/max in us.

Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (raw logits)
Init:         He uniform — w ~ U(-limit, limit), limit = sqrt(6/fan_in), b = 0
Seed:         42

Usage:
  python zap/reference/mlx_inference.py
  python zap/reference/mlx_inference.py --save
  python zap/reference/mlx_inference.py --compare "benchmarks/inference_*.json"

Requires:
  pip install mlx numpy
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np


# ── Constants (match inference_bench.zig exactly) ─────────────

SEED = 42
BATCH_SIZE = 64
WARMUP_ITERS = 100
GPU_THROUGHPUT_BATCHES = 160  # 160 * 64 = 10,240 samples.
GPU_LATENCY_ITERS = 1_000
CPU_LATENCY_ITERS = 1_000

TOTAL_THROUGHPUT_SAMPLES = GPU_THROUGHPUT_BATCHES * BATCH_SIZE

ARCHITECTURE = [
    {"input_size": 784, "output_size": 128, "activation": "relu"},
    {"input_size": 128, "output_size": 64, "activation": "relu"},
    {"input_size": 64, "output_size": 10, "activation": "none"},
]

PARAM_COUNT = (784 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
# = 100,480 + 8,256 + 650 = 109,386


# ── Model ─────────────────────────────────────────────────────

class MnistNet(nn.Module):
    """784 → 128 (ReLU) → 64 (ReLU) → 10 (raw logits).

    Matches nnzap's NetworkLayout exactly.  The last layer
    has no activation — softmax is fused into the loss
    during training, and raw logits are used for inference.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.maximum(self.fc1(x), 0.0)  # ReLU
        x = mx.maximum(self.fc2(x), 0.0)  # ReLU
        x = self.fc3(x)                   # Raw logits
        return x

    def param_count(self) -> int:
        return sum(
            v.size for _, v in mlx.utils.tree_flatten(
                self.parameters(),
            )
        )


# ── Initialisation ────────────────────────────────────────────

def he_init(model: MnistNet) -> None:
    """He uniform init matching nnzap's heInit().

    For each linear layer:
      weights ~ U(-limit, limit),  limit = sqrt(6 / fan_in)
      biases  = 0

    MLX Linear stores weight as [out, in].  We initialise
    the full [out x in] matrix with He uniform.
    """
    for layer in [model.fc1, model.fc2, model.fc3]:
        fan_in = layer.weight.shape[1]
        limit = math.sqrt(6.0 / fan_in)
        layer.weight = mx.random.uniform(
            low=-limit,
            high=limit,
            shape=layer.weight.shape,
        )
        layer.bias = mx.zeros(layer.bias.shape)


# ── Timing helper ─────────────────────────────────────────────

def nanos_to_us(ns: int) -> float:
    """Convert nanosecond delta to microseconds."""
    return ns / 1_000.0


def nanos_to_ms(ns: int) -> float:
    """Convert nanosecond delta to milliseconds."""
    return ns / 1_000_000.0


# ── Statistics ────────────────────────────────────────────────

def compute_stats(latencies_us: list[float]) -> dict:
    """Compute latency statistics from raw measurements.

    Matches inference_bench.zig's computeStats: sorts the
    array and uses nearest-rank percentiles on 0-based indices.
    """
    assert len(latencies_us) > 0

    sorted_us = sorted(latencies_us)
    n = len(sorted_us) - 1  # 0-based max index.

    # Nearest-rank percentile (same as nnzap).
    p50_idx = n * 50 // 100
    p99_idx = n * 99 // 100

    return {
        "iterations": len(sorted_us),
        "mean_us": sum(sorted_us) / len(sorted_us),
        "p50_us": sorted_us[p50_idx],
        "p99_us": sorted_us[p99_idx],
        "min_us": sorted_us[0],
        "max_us": sorted_us[-1],
    }


# ── Phase 1 — GPU batched throughput ─────────────────────────

def warmup_gpu(model: MnistNet, sample: mx.array) -> None:
    """Run warmup forward passes to prime MLX's Metal pipeline.

    MLX lazily compiles kernels on first evaluation — cold
    starts would skew latency measurements.
    """
    assert WARMUP_ITERS > 0

    for _ in range(WARMUP_ITERS):
        logits = model(sample)
        mx.eval(logits)


def bench_gpu_batched(
    model: MnistNet,
    batch: mx.array,
) -> dict:
    """Forward-pass many samples in batches, measuring peak
    GPU throughput with amortised Metal dispatch overhead.
    """
    assert batch.shape == (BATCH_SIZE, 784)
    assert GPU_THROUGHPUT_BATCHES > 0

    total_samples = TOTAL_THROUGHPUT_SAMPLES

    start_ns = time.perf_counter_ns()

    for _ in range(GPU_THROUGHPUT_BATCHES):
        logits = model(batch)
        mx.eval(logits)

    elapsed_ns = time.perf_counter_ns() - start_ns
    elapsed_ms = nanos_to_ms(elapsed_ns)

    images_per_sec = (
        total_samples / (elapsed_ms / 1_000.0)
        if elapsed_ms > 0.0
        else 0.0
    )

    return {
        "total_samples": total_samples,
        "batch_size": BATCH_SIZE,
        "total_ms": round(elapsed_ms, 3),
        "images_per_sec": round(images_per_sec, 1),
    }


# ── Phase 2 — GPU single-sample latency ─────────────────────

def bench_gpu_single(
    model: MnistNet,
    sample: mx.array,
) -> dict:
    """One forward pass per mx.eval(), measuring full Metal
    round-trip for batch=1 — includes graph compilation,
    kernel dispatch, and GPU-to-CPU synchronisation.
    """
    assert sample.shape == (1, 784)
    assert GPU_LATENCY_ITERS > 0

    latencies_us: list[float] = []

    for _ in range(GPU_LATENCY_ITERS):
        start_ns = time.perf_counter_ns()
        logits = model(sample)
        mx.eval(logits)
        elapsed_ns = time.perf_counter_ns() - start_ns
        latencies_us.append(nanos_to_us(elapsed_ns))

    return compute_stats(latencies_us)


# ── Phase 3 — CPU single-sample latency (NumPy) ─────────────

def extract_numpy_weights(model: MnistNet) -> list[dict]:
    """Extract all layer weights and biases as NumPy arrays.

    Returns a list of dicts with 'weight' and 'bias' keys,
    one per linear layer.  MLX stores weight as [out, in],
    and the forward pass computes x @ W.T + b, so we
    transpose here for direct np.dot(x, W.T) + b usage.
    """
    layers = []
    for mlx_layer in [model.fc1, model.fc2, model.fc3]:
        # Transpose: [out, in] → [in, out] for np.dot(x, W).
        weight_t = np.array(mlx_layer.weight).T
        bias = np.array(mlx_layer.bias)
        layers.append({"weight_t": weight_t, "bias": bias})
    return layers


def forward_numpy(
    x: np.ndarray,
    layers: list[dict],
) -> np.ndarray:
    """Pure NumPy forward pass — no MLX, no Metal.

    Implements: x @ W.T + b with ReLU on the first two
    layers, matching the MLX model exactly.
    """
    # Layer 0: 784 → 128 + ReLU.
    x = np.dot(x, layers[0]["weight_t"]) + layers[0]["bias"]
    x = np.maximum(0.0, x)

    # Layer 1: 128 → 64 + ReLU.
    x = np.dot(x, layers[1]["weight_t"]) + layers[1]["bias"]
    x = np.maximum(0.0, x)

    # Layer 2: 64 → 10, raw logits.
    x = np.dot(x, layers[2]["weight_t"]) + layers[2]["bias"]

    return x


def bench_cpu_single(
    np_layers: list[dict],
    sample_np: np.ndarray,
) -> dict:
    """Pure NumPy forward pass — no Metal dispatch at all.

    Reads weights as NumPy arrays and computes matmul + bias
    + ReLU in tight loops.
    """
    assert sample_np.shape == (1, 784)
    assert CPU_LATENCY_ITERS > 0

    latencies_us: list[float] = []

    for _ in range(CPU_LATENCY_ITERS):
        start_ns = time.perf_counter_ns()
        _ = forward_numpy(sample_np, np_layers)
        elapsed_ns = time.perf_counter_ns() - start_ns
        latencies_us.append(nanos_to_us(elapsed_ns))

    return compute_stats(latencies_us)


# ── Printing ──────────────────────────────────────────────────

def print_gpu_batched(result: dict) -> None:
    """Print GPU batched throughput results to stderr."""
    batches = result["total_samples"] // result["batch_size"]
    print(
        f"Phase 1 \u2014 GPU batched throughput\n"
        f"  {result['total_samples']} samples, "
        f"batch={result['batch_size']}, "
        f"{batches} batches\n"
        f"  Time: {result['total_ms']:.1f} ms  "
        f"({result['images_per_sec']:.0f} images/sec)\n",
        file=sys.stderr,
    )


def print_latency(label: str, result: dict) -> None:
    """Print latency statistics to stderr."""
    print(
        f"{label} ({result['iterations']} iterations)\n"
        f"  mean={result['mean_us']:.1f} us  "
        f"p50={result['p50_us']:.1f} us  "
        f"p99={result['p99_us']:.1f} us\n"
        f"  min={result['min_us']:.1f} us  "
        f"max={result['max_us']:.1f} us\n",
        file=sys.stderr,
    )


# ── Comparison ────────────────────────────────────────────────

def compare_benchmarks(
    mlx_results: dict,
    file_paths: list[str],
) -> None:
    """Print comparison table against nnzap benchmark files."""
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

        # Determine framework name from the JSON or filename.
        framework = data.get("framework", "unknown")
        if framework in benchmarks:
            # Append the filename to disambiguate.
            framework = f"{framework} ({Path(path).name})"
        benchmarks[framework] = data

    if len(benchmarks) < 2:
        print(
            "No valid benchmark files to compare against.",
            file=sys.stderr,
        )
        return

    names = list(benchmarks.keys())
    col_w = 14

    sep = "-" * (34 + col_w * len(names))
    header = f"{'Metric':<34}" + "".join(
        f"{n:>{col_w}}" for n in names
    )

    print(
        f"\n{'Inference benchmark comparison':^{len(sep)}}",
        file=sys.stderr,
    )
    print(sep, file=sys.stderr)
    print(header, file=sys.stderr)
    print(sep, file=sys.stderr)

    # GPU batched throughput.
    rows = [
        ("GPU batch total (ms)", "gpu_batched", "total_ms", ".1f"),
        ("GPU batch (img/s)", "gpu_batched", "images_per_sec", ".0f"),
        ("GPU single mean (us)", "gpu_single_sample", "mean_us", ".1f"),
        ("GPU single p50 (us)", "gpu_single_sample", "p50_us", ".1f"),
        ("GPU single p99 (us)", "gpu_single_sample", "p99_us", ".1f"),
        ("GPU single min (us)", "gpu_single_sample", "min_us", ".1f"),
        ("GPU single max (us)", "gpu_single_sample", "max_us", ".1f"),
        ("CPU single mean (us)", "cpu_single_sample", "mean_us", ".1f"),
        ("CPU single p50 (us)", "cpu_single_sample", "p50_us", ".1f"),
        ("CPU single p99 (us)", "cpu_single_sample", "p99_us", ".1f"),
        ("CPU single min (us)", "cpu_single_sample", "min_us", ".1f"),
        ("CPU single max (us)", "cpu_single_sample", "max_us", ".1f"),
    ]

    for label, section, key, fmt in rows:
        cells = ""
        for name in names:
            bench = benchmarks[name]
            section_data = bench.get(section, {})
            val = section_data.get(key, 0)
            cells += f"{val:>{col_w}{fmt}}"
        print(f"{label:<34}{cells}", file=sys.stderr)

    print(sep, file=sys.stderr)
    print(file=sys.stderr)


# ── JSON assembly ─────────────────────────────────────────────

def assemble_results(
    gpu_batched: dict,
    gpu_single: dict,
    cpu_single: dict,
) -> dict:
    """Assemble the full JSON results dict."""
    return {
        "framework": "mlx",
        "timestamp_utc": (
            datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "architecture": ARCHITECTURE,
        "param_count": PARAM_COUNT,
        "gpu_batched": gpu_batched,
        "gpu_single_sample": gpu_single,
        "cpu_single_sample": cpu_single,
    }


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MLX inference benchmark — mirrors nnzap's "
            "inference_bench.zig"
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
            "(e.g. 'benchmarks/inference_*.json')"
        ),
    )
    args = parser.parse_args()

    # ── Banner ──

    print(file=sys.stderr)
    print(
        "MLX inference benchmark",
        file=sys.stderr,
    )
    print(
        "=======================",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # ── Model setup ──

    mx.random.seed(SEED)

    model = MnistNet()
    he_init(model)
    mx.eval(model.parameters())

    param_count = model.param_count()
    assert param_count == PARAM_COUNT, (
        f"Expected {PARAM_COUNT} params, got {param_count}"
    )

    print(
        f"Architecture: 784 -> 128 (relu) -> "
        f"64 (relu) -> 10 (none)",
        file=sys.stderr,
    )
    print(
        f"Parameters:   {param_count:,}",
        file=sys.stderr,
    )
    print(
        f"Seed:         {SEED}",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # ── Input data (random, same as nnzap) ──
    # nnzap fills input buffers with random floats in [0, 1).
    # We do the same — the actual values don't matter for
    # benchmarking, only the shapes and computation paths.

    mx.random.seed(SEED)
    batch_input = mx.random.uniform(
        shape=(BATCH_SIZE, 784),
    )
    single_input = batch_input[0:1, :]  # Shape [1, 784].
    mx.eval(batch_input, single_input)

    # NumPy copy for Phase 3 (CPU path).
    single_input_np = np.array(single_input)

    # Extract NumPy weights before benchmarking (not timed).
    np_layers = extract_numpy_weights(model)

    # ── Warmup ──

    print(
        f"Warming up ({WARMUP_ITERS} iterations)... ",
        end="",
        flush=True,
        file=sys.stderr,
    )
    warmup_gpu(model, single_input)
    print("done.", file=sys.stderr)
    print(file=sys.stderr)

    # ── Phase 1 — GPU batched throughput ──

    gpu_batched = bench_gpu_batched(model, batch_input)
    print_gpu_batched(gpu_batched)

    # ── Phase 2 — GPU single-sample latency ──

    gpu_single = bench_gpu_single(model, single_input)
    print_latency(
        "Phase 2 \u2014 GPU single-sample",
        gpu_single,
    )

    # ── Phase 3 — CPU single-sample latency (NumPy) ──

    cpu_single = bench_cpu_single(np_layers, single_input_np)
    print_latency(
        "Phase 3 \u2014 CPU single-sample",
        cpu_single,
    )

    # ── Assemble results ──

    results = assemble_results(
        gpu_batched,
        gpu_single,
        cpu_single,
    )

    # ── Output JSON to stdout ──

    print(json.dumps(results, indent=4))

    # ── Save ──

    if args.save:
        benchmarks_dir = Path("benchmarks")
        benchmarks_dir.mkdir(exist_ok=True)

        timestamp = (
            datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H-%M-%SZ")
        )
        out_path = (
            benchmarks_dir
            / f"inference_mlx_{timestamp}.json"
        )

        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        print(
            f"Benchmark saved: {out_path}",
            file=sys.stderr,
        )
        print(file=sys.stderr)

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
            compare_benchmarks(results, paths)


if __name__ == "__main__":
    main()
