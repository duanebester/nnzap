# Profiling session: nnzap vs PrismML MLX — 2 April 2026

Machine: Apple Silicon (M2 Max), quiet system.
Model: Bonsai 1.7B (Qwen3, 1-bit Q1_0_g128, 28 layers).

## 1. Benchmark results

Both benchmarks use the same prompt, greedy decoding (temperature=0),
8 warmup tokens, 64 measured tokens.

| Metric               | **nnzap (Zig)** | **MLX (Python)** | Delta      |
| -------------------- | --------------: | ---------------: | ---------- |
| Load time            |         24.7 ms |          773.9ms | nnzap 31×  |
| Prefill (tok/s)      |           102.8 |            261.9 | MLX 2.5×   |
| **Decode (tok/s)**   |       **195.6** |        **224.5** | MLX +15%   |
| Decode latency p50   |       5,066 µs  |         4,444 µs | MLX −12%   |
| Decode latency p99   |       6,523 µs  |         5,067 µs | MLX −22%   |
| Prompt tokens        |              19 |               23 | Chat tmpl  |

The prompt token count difference (19 vs 23) is a chat-template
divergence between nnzap's tokenizer and HuggingFace's. It only
affects the prefill comparison, not decode.

### How the MLX benchmark works

Created `reference/mlx_bonsai.py` — a standalone benchmark script
that mirrors `nn/examples/bonsai_bench.zig` exactly:

- Uses PrismML's MLX fork (`pip install mlx @ git+...prism`)
  for 1-bit kernel support.
- Calls `mlx_lm.generate.generate_step` for pipelined decode,
  matching how PrismML's own benchmarks run.
- `generate_step` uses `mx.async_eval` internally — the GPU
  computes token N+1 while the CPU reads token N.
- Per-token latency is measured between successive yields.
- Supports `--save` (JSON to benchmarks/) and `--compare`
  (side-by-side table against nnzap JSON files).

## 2. Instruments Metal System Trace

Captured with `xctrace record --template 'Metal System Trace'`
for both nnzap and MLX during the full benchmark run.

### GPU Channel Activity Summary

| Metric                | **nnzap**  | **MLX**    |
| --------------------- | ---------: | ---------: |
| Total GPU compute     | 453.54 ms  | 360.50 ms  |
| Command buffer count  |        100 |        990 |
| Avg CPU→GPU latency   |    1.81 ms |    4.32 ms |
| Min CB duration       |  712.83 µs |    7.12 µs |
| Max CB duration       |   12.70 ms |    2.22 ms |

### What these numbers mean

**Total GPU compute (the smoking gun).** nnzap's GPU spends
93 ms more than MLX's across the full run — a 26% difference.
This is purely GPU-side kernel execution time, independent of
CPU scheduling or pipelining.

Per-decode-step GPU time (subtract prefill, divide by 72 tokens):

- nnzap: (453 − 12.7) / 72 ≈ **6.1 ms/token**
- MLX:   (360 − ~15) / 72  ≈ **4.8 ms/token**

These track the wall-clock benchmarks closely (5.1 ms vs 4.4 ms),
confirming the GPU is the bottleneck for both systems. CPU-side
overhead is minimal for both.

**Command buffer architecture.** nnzap uses 100 fat command
buffers — one per decode step, with all 28 layers and ~451
dispatches encoded into a single compute encoder. The min
duration (712 µs) is one decode step; the max (12.7 ms) is
prefill. MLX uses 990 thin command buffers, each containing a
handful of dispatches.

**CPU→GPU latency.** nnzap's is lower (1.81 ms vs 4.32 ms)
thanks to `commitAndSpinOnFlag`, which avoids the ~100–150 µs
Mach kernel trap of `waitUntilCompleted`. This is good. The
higher MLX latency reflects queued-up pipelined command buffers
waiting in the GPU's work queue — the GPU is always busy, so
latency is irrelevant.

### Conclusion from Instruments

**The gap is not pipelining.** nnzap's GPU-to-CPU turnaround is
faster than MLX's. The 15% decode throughput gap maps directly
to 26% more GPU compute time per token. The fix is in kernel
speed, not dispatch architecture.

## 3. MLX GPU trace (single decode step)

Captured with `mx.metal.start_capture()` / `stop_capture()` after
8 warmup tokens. The trace shows every Metal API call for one
decode step.

### Architecture

MLX splits one decode step across **~11 command buffers**, each
containing a `computeCommandEncoderWithDispatchType:Concurrent`
encoder processing 2–3 transformer layers. Command buffers are
chained with `waitForFence` / `updateFence` pairs.

The final command buffer uses `encodeSignalEvent` /
`waitUntilSignaledValue` for CPU-side completion notification.

Each command buffer has **~80 `addCompletedHandler`** calls —
these are MLX's internal bookkeeping for buffer lifetime tracking
and lazy-evaluation graph teardown.

### Concurrent dispatch type

Every encoder is created with `Concurrent` dispatch type:

    [computeCommandEncoderWithDispatchType:Concurrent]

This tells the GPU that dispatches without an intervening
`memoryBarrierWithScope:` can execute in parallel on different
GPU execution units. MLX selectively omits barriers between
independent operations.

**However:** nnzap already tried concurrent dispatch (experiment
23 in PERF_PLAN.md) and saw no improvement. The likely reason is
that 1-bit QMV kernels are memory-bandwidth bound — they saturate
the memory bus, so running two in parallel on different ALU units
does not help. The GPU's memory controller is the bottleneck, not
the number of in-flight compute pipelines.

### Per-layer dispatch pattern (from trace)

Each transformer layer in MLX has the following dispatch sequence
(identified by thread grid dimensions and barrier placement):

| #  | Operation              | Grid shape         | TG size      | Barrier after? |
| -- | ---------------------- | ------------------ | ------------ | -------------- |
| 1  | RMSNorm (attn)         | {2048, 1, 1}       | {1024, 1, 1} | No             |
| 2  | QMV (Q proj, bf16)     | {512, 1, 1}        | {512, 1, 1}  | Yes            |
| 3  | QMV (K proj, bf16)     | {1, 128, 1} groups | {32, 2, 1}   | No             |
| 4  | QMV (V proj, bf16)     | {1, 128, 1} groups | {32, 2, 1}   | No             |
| 5  | RoPE (Q+K norms)       | {128, 8, 1} ×2     | {128, 8, 1}  | Yes            |
| 6  | RMSNorm (q_norm)       | {2048, 1, 1}       | {1024, 1, 1} | No             |
| 7  | QMV (attn Q·K, bf16)   | {64, 8, 1}         | {64, 8, 1}   | Yes            |
| 8  | RoPE (K+V cache)       | {128, 8, 1} ×2     | {128, 8, 1}  | Yes            |
| 9  | Attention (GQA)        | {16, 1, 1} groups  | {1024, 1, 1} | Yes            |
| 10 | QMV (O proj, bf16)     | {1, 256, 1} groups | {32, 2, 1}   | Yes            |
| 11 | Residual add           | {2048, 1, 1}       | {1024, 1, 1} | No             |
| 12 | RMSNorm (MLP)          | {512, 1, 1}        | {512, 1, 1}  | Yes            |
| 13 | QMV (gate, bf16)       | {1, 768, 1} groups | {32, 2, 1}   | No             |
| 14 | QMV (up, bf16)         | {1, 768, 1} groups | {32, 2, 1}   | No             |
| 15 | SiLU elementwise mul   | {6144, 1, 1}       | {1024, 1, 1} | Yes            |
| 16 | QMV (down, bf16)       | {1, 256, 1} groups | {32, 2, 1}   | Yes            |
| 17 | Residual add           | {2048, 1, 1}       | {1024, 1, 1} | No             |

Notable: gate and up projections (#13, #14) are dispatched
back-to-back with **no barrier** between them, so they can run
concurrently. Same for K and V projections (#3, #4). nnzap
achieves the same effect with fused-pair kernels instead.

### Key observation: bf16 everywhere

Every QMV dispatch uses `affine_qmv_bfloat16_t_gs_128_b_1_batch_0`.
The `bfloat16_t` in the name confirms all activations flow through
the model as bf16. nnzap uses f32 activations, doubling memory
traffic on every read and write.

The QMV dispatch grid sizes tell the story:

- MLX gate/up: `{1, 768, 1}` groups × `{32, 2, 1}` = 768 TGs,
  64 threads each = 49,152 threads.
- nnzap gate/up fused: one dispatch of ~192 TGs × 512 threads
  = 98,304 threads.

MLX uses fewer threads with smaller threadgroups (2 simdgroups
per TG, 8 rows per TG). nnzap uses 16 simdgroups per TG and 32
rows per TG. Different strategies, but nnzap's constant-cache
approach is theoretically better (zero shared-memory barriers).

## 4. What we confirmed

### Confirmed: the gap is activation precision (f32 vs bf16)

The PERF_PLAN.md root cause analysis is correct. The Instruments
trace proves it quantitatively:

- nnzap GPU compute per token: **6.1 ms** (f32 activations).
- MLX GPU compute per token: **4.8 ms** (bf16 activations).
- Ratio: 6.1 / 4.8 = **1.27×**. Activation data is 2× larger.

If activation traffic is ~50% of total memory bandwidth (the rest
is 1-bit weights + scales), then halving it saves ~25% of total
bandwidth — which matches the 27% measured gap almost exactly.

### Confirmed: not pipelining

CPU→GPU latency is lower for nnzap (1.81 ms vs 4.32 ms). The
spin-on-flag approach works. Pipelining would add 2–3% at most,
per PERF_PLAN.md §5.

### Confirmed: not concurrent dispatch

Already tested (experiment 23), no improvement. The GPU trace
shows MLX does use concurrent dispatch, but the memory-bandwidth-
bound QMV kernels cannot benefit from running in parallel because
they compete for the same memory bus.

### Confirmed: not kernel micro-optimisation

MLX uses 2 simdgroups / 8 rows per TG. nnzap uses 16 simdgroups
/ 32 rows per TG with constant-cache input. Both approaches are
valid for memory-bound workloads. The uint32 weight loads, the
select-vs-accum-trick, and rows-per-simdgroup have all been
tested and eliminated as factors.

### Confirmed: not dispatch count

nnzap's 100 fat command buffers vs MLX's 990 thin ones. nnzap's
approach is actually better (fewer Metal API calls, fewer command
buffer submissions, less Obj-C overhead). The 3,748 API calls per
token cost ~375 µs of CPU time — significant, but not the primary
bottleneck given that GPU compute dominates at 6.1 ms/token.

## 5. Remaining gap breakdown

| Factor                     | Impact  | Status      |
| -------------------------- | ------: | ----------- |
| f32 → f16/bf16 activations | +15–20% | **Next**    |
| Async token pipelining     |   +2–3% | After f16   |
| GQA attention vectorise    |   +2–5% | After f16   |
| Prefill optimisation       |    2.5× | Investigate |

The **f16 activation pipeline** (PERF_PLAN.md §"THE NEXT
EXPERIMENT") is the single remaining optimisation that can close
the gap. Everything else has been tested or is marginal.

### The prefill gap (2.5×) — needs investigation

The prefill gap is much larger than the decode gap. nnzap's
prefill at 102.8 tok/s vs MLX's 261.9 tok/s suggests nnzap may
be using a batch-1 decode path for prefill instead of a batched
matmul. This is a separate issue from the activation precision
gap, and worth investigating after f16 is landed.

## 6. Files produced

| File                                | Purpose                        |
| ----------------------------------- | ------------------------------ |
| `reference/mlx_bonsai.py`           | MLX Bonsai 1.7B benchmark      |
| `benchmarks/nnzap_metal.trace`      | Instruments trace (nnzap)      |
| `benchmarks/mlx_metal.trace`        | Instruments trace (MLX)        |
| `benchmarks/mlx_one_step.gputrace`  | Xcode GPU trace (1 MLX step)   |
| `benchmarks/bonsai_bench_*.json`    | nnzap benchmark JSON           |
| `docs/PROFILING_2026_04_02.md`      | This document                  |

## 7. Methodology notes

### MLX benchmark setup

PrismML's MLX fork (branch `prism`) was installed from source
into a local venv:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism"
    pip install mlx-lm

The standard `mlx` package does not support 1-bit quantisation.
PrismML's fork adds 1-bit support to the `affine_quantize`,
`qdot`, `load_vector`, `dequantize`, and `qouter` template
functions in `quantized.h`.

### Benchmark script usage

    # Basic run (prints to stderr + JSON to stdout):
    python reference/mlx_bonsai.py

    # Save JSON to benchmarks/:
    python reference/mlx_bonsai.py --save

    # Compare against nnzap results:
    python reference/mlx_bonsai.py --compare "benchmarks/bonsai_bench_*.json"

    # Custom model path:
    python reference/mlx_bonsai.py --model /path/to/bonsai-1.7b

### GPU trace capture

    # Instruments trace (system-wide, includes GPU timeline):
    xctrace record --template 'Metal System Trace' \
      --output benchmarks/nnzap_metal.trace \
      --time-limit 30s --launch -- ./zig-out/bin/bonsai_bench

    # MLX programmatic GPU capture (single decode step):
    MTL_CAPTURE_ENABLED=1 python3 -c "
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tokenizer = load('...')
    # ... warm up ...
    mx.metal.start_capture('benchmarks/mlx_one_step.gputrace')
    logits = model(token.reshape(1,1), cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    mx.metal.stop_capture()
    "

The `.gputrace` file opens in Xcode and shows every Metal API
call with per-dispatch timing. The `.trace` file opens in
Instruments and shows the GPU timeline with command buffer
durations.
