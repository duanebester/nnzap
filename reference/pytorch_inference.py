#!/usr/bin/env python3
"""PyTorch inference benchmark mirroring nnmetal's inference_bench.zig.

Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (raw logits)
Init:         He uniform — w ~ U(-limit, limit), limit = sqrt(6/fan_in), b = 0
Seed:         42

Three benchmark phases:

  Phase 1 — GPU batched throughput
    Forward-pass 10,240 samples in batches of 64 on MPS.
    Measures total time and images/sec.

  Phase 2 — GPU single-sample latency
    Forward-pass 1 sample on MPS, synchronise, repeat 1,000 times.
    Reports p50/p99/mean/min/max in microseconds.

  Phase 3 — CPU single-sample latency
    Forward-pass 1 sample on CPU, repeat 1,000 times.
    Reports p50/p99/mean/min/max in microseconds.

Usage:
  python zap/reference/pytorch_inference.py
  python zap/reference/pytorch_inference.py --save
  python zap/reference/pytorch_inference.py --compare "benchmarks/inference_*.json"

Requires:
  pip install torch
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


# ── Constants (match inference_bench.zig) ─────────────────────

WARMUP_ITERS = 100
GPU_THROUGHPUT_BATCHES = 160
BATCH_SIZE = 64
GPU_LATENCY_ITERS = 1_000
CPU_LATENCY_ITERS = 1_000
SEED = 42

TOTAL_THROUGHPUT_SAMPLES = GPU_THROUGHPUT_BATCHES * BATCH_SIZE  # 10,240

ARCHITECTURE = [
    {"input_size": 784, "output_size": 128, "activation": "relu"},
    {"input_size": 128, "output_size": 64, "activation": "relu"},
    {"input_size": 64, "output_size": 10, "activation": "none"},
]

EXPECTED_PARAM_COUNT = 109_386


# ── Lazy torch import ─────────────────────────────────────────
# Deferred so the module-level constants above are available
# even if torch is not installed (useful for --help).

def _import_torch():
    """Import torch once and cache in module globals."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ── Architecture ──────────────────────────────────────────────

def _build_model_class():
    """Build the MnistNet class after torch is imported.

    Kept in a factory so the top-level module doesn't
    require torch at import time.
    """
    torch, nn, F = _import_torch()

    class MnistNet(nn.Module):
        """784 → 128 (ReLU) → 64 (ReLU) → 10 (raw logits).

        Matches nnmetal's NetworkLayout exactly.  The last
        layer has no activation — softmax is fused into
        the loss during training.
        """

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(784, 128)   # Layer 0: relu.
            self.fc2 = nn.Linear(128, 64)    # Layer 1: relu.
            self.fc3 = nn.Linear(64, 10)     # Layer 2: none.

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

    return MnistNet


# ── Initialisation ────────────────────────────────────────────

def he_init(model) -> None:
    """He uniform init matching nnmetal's heInit().

    For each linear layer:
      weights ~ U(-limit, limit),  limit = sqrt(6 / fan_in)
      biases  = 0
    """
    torch, nn, _ = _import_torch()

    for module in model.modules():
        if isinstance(module, nn.Linear):
            fan_in = module.in_features
            limit = math.sqrt(6.0 / fan_in)
            nn.init.uniform_(module.weight, -limit, limit)
            nn.init.zeros_(module.bias)


# ── MPS availability ─────────────────────────────────────────

def mps_available() -> bool:
    """Check whether MPS (Apple GPU) is available."""
    torch, _, _ = _import_torch()
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


# ── Phase 1 — GPU batched throughput ─────────────────────────

def bench_gpu_batched(
    model,
    device,
) -> dict:
    """Forward-pass TOTAL_THROUGHPUT_SAMPLES samples in
    batches of BATCH_SIZE on MPS.  Returns timing dict.

    Warmup runs WARMUP_ITERS batches first to prime the
    Metal shader cache and avoid JIT-compilation skew.
    """
    torch, _, _ = _import_torch()

    assert TOTAL_THROUGHPUT_SAMPLES > 0
    assert BATCH_SIZE > 0

    # Pre-allocate a single random input batch, reused
    # every iteration (we measure throughput, not data
    # loading).
    input_batch = torch.rand(
        BATCH_SIZE, 784,
        device=device, dtype=torch.float32,
    )

    # Warmup: prime Metal shader compilation.
    for _ in range(WARMUP_ITERS):
        _ = model(input_batch)
    torch.mps.synchronize()

    # Timed run.
    start_ns = time.perf_counter_ns()

    for _ in range(GPU_THROUGHPUT_BATCHES):
        _ = model(input_batch)
    torch.mps.synchronize()

    elapsed_ns = time.perf_counter_ns() - start_ns
    elapsed_ms = elapsed_ns / 1_000_000.0

    images_per_sec = (
        TOTAL_THROUGHPUT_SAMPLES / (elapsed_ms / 1_000.0)
        if elapsed_ms > 0.0
        else 0.0
    )

    return {
        "total_samples": TOTAL_THROUGHPUT_SAMPLES,
        "batch_size": BATCH_SIZE,
        "total_ms": round(elapsed_ms, 3),
        "images_per_sec": round(images_per_sec, 1),
    }


# ── Phase 2 — GPU single-sample latency ─────────────────────

def bench_gpu_single(
    model,
    device,
) -> dict:
    """Forward-pass 1 sample on MPS, synchronise, measure
    round-trip.  Repeat GPU_LATENCY_ITERS times.
    """
    torch, _, _ = _import_torch()

    assert GPU_LATENCY_ITERS > 0

    single_input = torch.rand(
        1, 784,
        device=device, dtype=torch.float32,
    )

    # Warmup (the model should already be warm from
    # Phase 1, but guard against being called standalone).
    for _ in range(WARMUP_ITERS):
        _ = model(single_input)
    torch.mps.synchronize()

    latencies_ns: list[int] = []

    for _ in range(GPU_LATENCY_ITERS):
        start_ns = time.perf_counter_ns()
        _ = model(single_input)
        torch.mps.synchronize()
        elapsed_ns = time.perf_counter_ns() - start_ns
        latencies_ns.append(elapsed_ns)

    return compute_latency_stats(
        latencies_ns, GPU_LATENCY_ITERS,
    )


# ── Phase 3 — CPU single-sample latency ─────────────────────

def bench_cpu_single(model) -> dict:
    """Forward-pass 1 sample on CPU, repeat CPU_LATENCY_ITERS
    times.  Uses a separate model copy on torch.device("cpu").
    """
    torch, _, _ = _import_torch()

    assert CPU_LATENCY_ITERS > 0

    cpu_device = torch.device("cpu")
    cpu_model = _clone_model_to(model, cpu_device)
    cpu_model.eval()

    single_input = torch.rand(
        1, 784,
        device=cpu_device, dtype=torch.float32,
    )

    # Warmup on CPU.
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = cpu_model(single_input)

    latencies_ns: list[int] = []

    with torch.no_grad():
        for _ in range(CPU_LATENCY_ITERS):
            start_ns = time.perf_counter_ns()
            _ = cpu_model(single_input)
            elapsed_ns = time.perf_counter_ns() - start_ns
            latencies_ns.append(elapsed_ns)

    return compute_latency_stats(
        latencies_ns, CPU_LATENCY_ITERS,
    )


# ── Statistics ────────────────────────────────────────────────

def compute_latency_stats(
    latencies_ns: list[int],
    count: int,
) -> dict:
    """Compute p50/p99/mean/min/max from nanosecond samples.

    Uses nearest-rank percentile on 0-based indices to
    match nnmetal's computeStats().
    """
    assert len(latencies_ns) == count
    assert count > 0

    # Convert to microseconds.
    latencies_us = [ns / 1_000.0 for ns in latencies_ns]
    latencies_us.sort()

    total = sum(latencies_us)
    n = count - 1  # Last valid 0-based index.
    p50_idx = n * 50 // 100
    p99_idx = n * 99 // 100

    return {
        "iterations": count,
        "mean_us": round(total / count, 2),
        "p50_us": round(latencies_us[p50_idx], 2),
        "p99_us": round(latencies_us[p99_idx], 2),
        "min_us": round(latencies_us[0], 2),
        "max_us": round(latencies_us[-1], 2),
    }


# ── Helpers ───────────────────────────────────────────────────

def _clone_model_to(model, device):
    """Deep-copy a model onto a different device."""
    import copy
    clone = copy.deepcopy(model)
    clone.to(device)
    return clone


# ── Printing (to stderr, matching nnmetal's debug output) ──────

def print_header(param_count: int) -> None:
    """Print the architecture summary banner."""
    print(file=sys.stderr)
    print(
        "PyTorch inference benchmark",
        file=sys.stderr,
    )
    print(
        "===========================",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(
        "+-----------------------------------+",
        file=sys.stderr,
    )
    print(
        "|     PyTorch Network Layout        |",
        file=sys.stderr,
    )
    print(
        "+-----------------------------------+",
        file=sys.stderr,
    )
    print(
        "|  Layer 0:  784 -> 128   (relu   ) |",
        file=sys.stderr,
    )
    print(
        "|  Layer 1:  128 -> 64    (relu   ) |",
        file=sys.stderr,
    )
    print(
        "|  Layer 2:   64 -> 10    (none   ) |",
        file=sys.stderr,
    )
    print(
        "+-----------------------------------+",
        file=sys.stderr,
    )
    print(
        f"|  Total params: {param_count:<16} |",
        file=sys.stderr,
    )
    print(
        "+-----------------------------------+",
        file=sys.stderr,
    )
    print(file=sys.stderr)


def print_gpu_batched(result: dict) -> None:
    """Print Phase 1 summary to stderr."""
    batches = (
        result["total_samples"] // result["batch_size"]
    )
    print(
        "Phase 1 \u2014 GPU batched throughput",
        file=sys.stderr,
    )
    print(
        f"  {result['total_samples']} samples, "
        f"batch={result['batch_size']}, "
        f"{batches} batches",
        file=sys.stderr,
    )
    print(
        f"  Time: {result['total_ms']:.1f} ms  "
        f"({result['images_per_sec']:.0f} images/sec)",
        file=sys.stderr,
    )
    print(file=sys.stderr)


def print_latency(label: str, result: dict) -> None:
    """Print a latency-phase summary to stderr."""
    print(
        f"{label} ({result['iterations']} iterations)",
        file=sys.stderr,
    )
    print(
        f"  mean={result['mean_us']:.1f} us  "
        f"p50={result['p50_us']:.1f} us  "
        f"p99={result['p99_us']:.1f} us",
        file=sys.stderr,
    )
    print(
        f"  min={result['min_us']:.1f} us  "
        f"max={result['max_us']:.1f} us",
        file=sys.stderr,
    )
    print(file=sys.stderr)


def print_skipped(label: str) -> None:
    """Note that a GPU phase was skipped."""
    print(
        f"{label}: skipped (MPS unavailable)",
        file=sys.stderr,
    )
    print(file=sys.stderr)


# ── Comparison ────────────────────────────────────────────────

def compare_with_nnmetal(
    pytorch_results: dict,
    nnmetal_path: str,
) -> None:
    """Print side-by-side comparison with an nnmetal
    inference benchmark JSON.
    """
    with open(nnmetal_path) as f:
        nnmetal = json.load(f)

    sep = "-" * 58
    print(file=sys.stderr)
    print(
        f"{'Comparison with nnmetal':^58}",
        file=sys.stderr,
    )
    print(f"  nnmetal file: {nnmetal_path}", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(
        f"{'Metric':<30} {'nnmetal':>12} {'PyTorch':>12}",
        file=sys.stderr,
    )
    print(sep, file=sys.stderr)

    # GPU batched throughput.
    nz_gpu = nnmetal.get("gpu_batched")
    pt_gpu = pytorch_results.get("gpu_batched")

    if nz_gpu and pt_gpu:
        print(
            f"{'GPU batch total (ms)':<30} "
            f"{nz_gpu['total_ms']:>12.1f} "
            f"{pt_gpu['total_ms']:>12.1f}",
            file=sys.stderr,
        )
        print(
            f"{'GPU batch (img/s)':<30} "
            f"{nz_gpu['images_per_sec']:>12.0f} "
            f"{pt_gpu['images_per_sec']:>12.0f}",
            file=sys.stderr,
        )
    elif nz_gpu:
        print(
            f"{'GPU batch (img/s)':<30} "
            f"{nz_gpu['images_per_sec']:>12.0f} "
            f"{'n/a':>12}",
            file=sys.stderr,
        )

    # GPU single-sample latency.
    nz_gs = nnmetal.get("gpu_single_sample")
    pt_gs = pytorch_results.get("gpu_single_sample")

    if nz_gs and pt_gs:
        print(
            f"{'GPU single p50 (us)':<30} "
            f"{nz_gs['p50_us']:>12.1f} "
            f"{pt_gs['p50_us']:>12.1f}",
            file=sys.stderr,
        )
        print(
            f"{'GPU single p99 (us)':<30} "
            f"{nz_gs['p99_us']:>12.1f} "
            f"{pt_gs['p99_us']:>12.1f}",
            file=sys.stderr,
        )

    # CPU single-sample latency.
    nz_cs = nnmetal.get("cpu_single_sample")
    pt_cs = pytorch_results.get("cpu_single_sample")

    if nz_cs and pt_cs:
        print(
            f"{'CPU single p50 (us)':<30} "
            f"{nz_cs['p50_us']:>12.1f} "
            f"{pt_cs['p50_us']:>12.1f}",
            file=sys.stderr,
        )
        print(
            f"{'CPU single p99 (us)':<30} "
            f"{nz_cs['p99_us']:>12.1f} "
            f"{pt_cs['p99_us']:>12.1f}",
            file=sys.stderr,
        )
        print(
            f"{'CPU single mean (us)':<30} "
            f"{nz_cs['mean_us']:>12.1f} "
            f"{pt_cs['mean_us']:>12.1f}",
            file=sys.stderr,
        )

    print(sep, file=sys.stderr)
    print(file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PyTorch inference benchmark "
            "(mirrors nnmetal inference_bench.zig)"
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save JSON results to benchmarks/ directory",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help=(
            "Glob pattern for nnmetal benchmark JSON files "
            "to compare against"
        ),
    )
    args = parser.parse_args()

    torch, _, _ = _import_torch()

    # ── Seed ──

    torch.manual_seed(SEED)

    # ── Model ──

    MnistNet = _build_model_class()
    model = MnistNet()
    he_init(model)

    param_count = model.param_count()
    assert param_count == EXPECTED_PARAM_COUNT, (
        f"Expected {EXPECTED_PARAM_COUNT} params, "
        f"got {param_count}"
    )

    print_header(param_count)

    has_mps = mps_available()

    if has_mps:
        mps_device = torch.device("mps")
        print(
            f"MPS: available (device={mps_device})",
            file=sys.stderr,
        )
    else:
        mps_device = None
        print("MPS: unavailable", file=sys.stderr)

    print(file=sys.stderr)

    # ── Phase 1 — GPU batched throughput ──

    gpu_batched = None

    if has_mps:
        gpu_model = model.to(mps_device)
        gpu_model.eval()

        with torch.no_grad():
            gpu_batched = bench_gpu_batched(
                gpu_model, mps_device,
            )
        print_gpu_batched(gpu_batched)
    else:
        print_skipped(
            "Phase 1 \u2014 GPU batched throughput",
        )

    # ── Phase 2 — GPU single-sample latency ──

    gpu_single = None

    if has_mps:
        # gpu_model is already on MPS and in eval mode.
        with torch.no_grad():
            gpu_single = bench_gpu_single(
                gpu_model, mps_device,
            )
        print_latency(
            "Phase 2 \u2014 GPU single-sample",
            gpu_single,
        )
    else:
        print_skipped(
            "Phase 2 \u2014 GPU single-sample latency",
        )

    # ── Phase 3 — CPU single-sample latency ──

    cpu_single = bench_cpu_single(model)
    print_latency(
        "Phase 3 \u2014 CPU single-sample",
        cpu_single,
    )

    # ── Assemble results ──

    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    results = {
        "framework": "pytorch",
        "timestamp_utc": timestamp,
        "architecture": ARCHITECTURE,
        "param_count": param_count,
        "gpu_batched": gpu_batched,
        "gpu_single_sample": gpu_single,
        "cpu_single_sample": cpu_single,
    }

    # ── Output JSON ──

    json_str = json.dumps(results, indent=4)

    if args.save:
        benchmarks_dir = Path("benchmarks")
        benchmarks_dir.mkdir(exist_ok=True)

        file_ts = (
            datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H-%M-%SZ")
        )
        out_path = (
            benchmarks_dir
            / f"inference_pytorch_{file_ts}.json"
        )

        with open(out_path, "w") as f:
            f.write(json_str)
            f.write("\n")

        print(
            f"Benchmark saved: {out_path}",
            file=sys.stderr,
        )
        print(file=sys.stderr)
    else:
        print(json_str)

    # ── Compare ──

    if args.compare is not None:
        paths = sorted(glob.glob(args.compare))
        if not paths:
            print(
                f"Warning: no files matched "
                f"'{args.compare}'",
                file=sys.stderr,
            )
        else:
            # Use the most recent nnmetal benchmark.
            compare_with_nnmetal(results, paths[-1])
    else:
        # Auto-detect nnmetal inference benchmarks.
        nnmetal_files = sorted(
            glob.glob("benchmarks/inference_2*.json"),
        )
        if nnmetal_files:
            compare_with_nnmetal(results, nnmetal_files[-1])


if __name__ == "__main__":
    main()
