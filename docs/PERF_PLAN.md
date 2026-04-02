# Performance plan: closing the gap with PrismML MLX fork

## Competitive benchmark (M2 Max, quiet system)

| Runtime                 | Decode (tok/s) | Notes                     |
| ----------------------- | -------------: | ------------------------- |
| **nnzap** (Zig + Metal) |        **169** | After uint32 weight loads |
| **PrismML MLX fork**    |        **224** | Custom 1-bit MLX fork     |

Gap: **~25%**. Target: **224 tok/s**.

PrismML's fork extends Apple's MLX framework with `bits=1` support
in the existing `affine_qmv` kernel template. They get years of MLX
Metal kernel infrastructure for free. We wrote everything from
scratch in ~22K lines of Zig. The gap is not a design flaw — it is
a short list of concrete, measurable differences.

## Experiment log (completed)

| #   | Optimisation                                       | Result                    | Status                                                         |
| --- | -------------------------------------------------- | ------------------------- | -------------------------------------------------------------- |
| 1A  | 4 rows per simdgroup                               | Flat (~163 tok/s)         | ❌ Kernel is memory-bandwidth bound, not ALU bound             |
| 1B  | uint32 weight loads (qmv_const + fused_pair_const) | **+3.9%** (163→169 tok/s) | ✅ Committed                                                   |
| 1C  | Algebraic qdot trick (accum + bias×sum_x)          | Flat (~163.5 tok/s)       | ❌ Metal compiler already optimises `select` to single ternary |
| —   | uint32 for qmv_const_multigroup (down proj)        | Test failure              | ❌ Alignment: K=5504 → byte_start not 4-byte aligned           |
| —   | Fused RoPE QK                                      | +0.5% (169.8 tok/s)       | ❌ Below 5% threshold                                          |
| —   | Fused residual+RMSNorm                             | Regressed 8%              | ❌ Exp 13                                                      |
| —   | Fused qk_norm_rope                                 | Regressed 8%              | ❌ Exp 17                                                      |
| —   | Concurrent dispatch                                | No improvement            | ❌ Exp 23                                                      |
| —   | Obj-C overhead reduction                           | Marginal                  | ❌ Exp 9                                                       |

**Conclusion: kernel-level micro-optimisations are exhausted.**
The system runs at ~8% memory bandwidth utilisation. The GPU
finishes each kernel fast and stalls on the 366 barriers. Wider
loads and fewer ALU instructions do not help when the bottleneck
is latency across serialisation points, not compute or bandwidth
within a single kernel.

**The next step-change is f16 activations.** Everything else is
at the margins. This is the single remaining optimisation that
can close the gap to 224 tok/s.

## Root cause analysis

Five factors explain the original 35% gap, ordered by impact.

### 1. f32 activations — the single biggest factor (~15–20%)

MLX kernel names: `affine_qmv_bfloat16_t_gs_128_b_1_batch_0`.
Their entire activation pipeline runs in **bfloat16**. We run
**f32 everywhere**:

```
transformer.metal L7–10:
  // Every kernel here operates on f32 activations.
```

This means **2× the memory traffic** on every activation read and
write — every RMSNorm, every SiLU, every residual add, and
critically, every QMV input vector load. For `qmv_const` with
K=2048, our input vector is 8 KB (f32) vs 4 KB (bf16). That is
half the constant-cache pressure and half the bandwidth for every
projection dispatch across all 28 layers.

1-bit inference is solidly **memory-bandwidth bound** — the
weights are 1 bit per element, so activation traffic dominates
the memory bus. Halving that traffic is a direct throughput win.

### 2. Graph-level kernel fusion (~5–10%)

Our per-token decode involves:

| Metric          |     nnzap | MLX (estimated) |
| --------------- | --------: | --------------: |
| GPU dispatches  |   **451** |        ~100–150 |
| Memory barriers |   **366** |          ~50–80 |
| Obj-C API calls | **3,748** |        ~500–800 |

MLX uses lazy evaluation with graph compilation. Operations like
RMSNorm→QMV, ResidualAdd→RMSNorm, and SiLU→mul are candidates
for fusion. We encode them as separate dispatches with full-scope
`memoryBarrierWithScope:` barriers between each.

At ~100 ns per Obj-C message send, our 3,748 API calls cost
**~375 μs of pure CPU overhead per token** — roughly 6% of the
~6 ms token budget at 169 tok/s.

Multiple fusion experiments have been tried and all regressed or
were marginal. This factor is real but hard to exploit without
a fundamentally different dispatch architecture.

### 3. The qdot accumulation trick — NOT a real factor

MLX uses `accum + bias * sum_x` instead of `select(-x, x, bit)`.
We tested this (experiment 25): flat. The Metal compiler already
optimises `select` to an efficient ternary instruction. The
algebraic rewrite produces identical machine code. Cross this off.

### 4. More rows per simdgroup — NOT a real factor

MLX uses 4 rows/simdgroup vs our 2. We tested this (experiment
21): flat. The kernel is memory-bandwidth bound, not ALU bound.
Processing more rows per simdgroup does not help because the
bottleneck is memory latency, not compute. Cross this off.

### 5. No CPU/GPU pipelining (~2–3%)

We block on `commitAndWait` every single token. MLX's async
evaluation naturally pipelines GPU compute with CPU scheduling.
Worth doing after f16 activations, but small absolute gain.

## Detailed per-token dispatch audit

### Per decoder block (16 dispatches, 13 barriers)

Attention half (11 dispatches, 8 barriers):

1. RMSNorm (attn_norm) → barrier
2. QMV (Q projection) |
3. QMV fused pair (K+V) → barrier
4. RMSNorm (Q head norm) |
5. RMSNorm (K head norm) → barrier
6. RoPE (Q) |
7. RoPE (K) → barrier
8. KV cache update → barrier
9. GQA attention → barrier
10. QMV (O projection) → barrier
11. Residual add → barrier

MLP half (5 dispatches, 5 barriers): 12. RMSNorm (ffn_norm) → barrier 13. QMV fused pair (gate+up) → barrier 14. SiLU elementwise mul → barrier 15. QMV (down projection) → barrier 16. Residual add → barrier

Full forward pass (28 layers):
Dispatches: 16 × 28 + 3 = 451
Barriers: 13 × 28 + 2 = 366
setBuffer: 72 × 28 + 9 ≈ 2,025
setBytes: 16 × 28 + 2 ≈ 450
Total Metal API calls: ~3,748

All 366 barriers use global `memoryBarrierWithScope:` (all
buffers) even when only one or two specific buffers have a
RAW dependency. All 366 are necessary for correctness.

### QMV projection dimensions (Bonsai 1.7B)

| Projection     |      M |    K | Kernel used            | Rows/TG |
| -------------- | -----: | ---: | ---------------------- | ------: |
| Q proj         |  2,048 | 2048 | `qmv_const`            |      32 |
| K+V fused      |  1,024 | 2048 | `qmv_fused_pair_const` |      32 |
| O proj         |  2,048 | 2048 | `qmv_const`            |      32 |
| gate+up fused  |  6,144 | 2048 | `qmv_fused_pair_const` |      32 |
| down proj      |  2,048 | 5504 | `qmv_const_multigroup` |      32 |
| LM head (tied) | 151669 | 2048 | `qmv_const`            |      32 |

The LM head is the single largest dispatch: 151,669 / 32 ≈ 4,740
threadgroups. This one dispatch is a significant fraction of total
per-token compute.

## Activation precision map

| Data                                             | Precision                    | Bandwidth cost |
| ------------------------------------------------ | ---------------------------- | -------------- |
| Activations (residual, Q/K/V, MLP intermediates) | **f32**                      | 2× vs f16      |
| KV cache                                         | **f16**                      | Optimal        |
| Norm scales                                      | **f16**                      | Optimal        |
| Projection weights                               | **1-bit** + f16 group scales | Optimal        |
| Logits output                                    | **f32**                      | Required       |

Switching activations to f16 would halve bandwidth for:

- RMSNorm input/output (28 × 4 = 112 dispatches)
- SiLU elementwise mul (28 dispatches)
- Residual adds (28 × 2 = 56 dispatches)
- QMV input vector loads (all ~197+ dispatches)
- GQA attention Q input and output

QMV accumulation must stay f32 internally (intermediate precision).
Only the loads and stores change to f16.

## GQA attention analysis

Current kernel: `gqa_attention` in transformer.metal.

- Thread group: 256 threads, one per query head.
- Scratch buffer: **device memory** (not threadgroup).
- Softmax: classical two-pass (find max, then exp+sum+normalize).
- Dot product: **fully scalar** — one element at a time.
- V accumulation: **fully scalar** — loops over seq_len per dim.
- KV cache reads: scalar `half` → `float` conversions.

Opportunities (after f16 is done):

- `half4` loads would reduce KV cache memory transactions 4×.
- Online softmax (FlashAttention-style) would fuse 3 passes into 1.
- These matter more at longer sequence lengths.

## >>> THE NEXT EXPERIMENT: f16 activations <<<

> **This is the single most important optimisation remaining.**
> Expected: **+15–20%** (169 → 195–205 tok/s).
> This is what closes the gap to MLX's 224 tok/s.
>
> **Do not skip this. Do not try more kernel micro-optimisations
> instead. Those are exhausted. This is the move.**

### Why this works

The system is at ~8% memory bandwidth utilisation. Each kernel
finishes fast and stalls on barriers. f16 activations help by:

1. **Halving activation buffer sizes.** Every RMSNorm, SiLU,
   residual_add, and QMV input/output is currently f32. With
   f16, each of the 451 dispatches moves half as many bytes.
   Less data per kernel → faster completion → less stall time.

2. **Halving the QMV constant-cache footprint.** The input
   vector for qmv_const is 2048 × 4 = 8 KB (f32). With f16
   it is 4 KB. This leaves more constant-cache space for the
   hardware to prefetch weight data, and improves cache hit
   rates across all 16 simdgroups sharing the same input.

3. **Matching MLX's architecture.** The PrismML fork's kernel
   names are `affine_qmv_bfloat16_t_...`. Their entire
   activation pipeline is bf16. This is the #1 difference.

### This is NOT the same as experiment 12

Experiment 12 tried **f16 shared memory inside qmv_fast** —
storing the input vector as f16 in threadgroup shared memory
and accumulating in f16. That failed because intermediate
accumulation in f16 loses precision across 2048 elements.

**The f16 activation pipeline is fundamentally different:**

- Activations are stored as f16 in **device-memory buffers**
  between kernels (norm_out, q, k, v, gate, up, mlp_out, etc.)
- QMV loads f16 input, converts to f32 immediately, and
  accumulates in f32. The `float(half_input[i])` cast happens
  once per element in the inner loop. **No f16 accumulation.**
- RMSNorm reads f16 input, accumulates sum-of-squares in f32,
  writes f16 output. The f32 intermediate is in registers only.
- The residual buffer can stay f32 if precision is a concern
  (see implementation strategy below).

### Implementation strategy: incremental, one kernel at a time

Do NOT try to convert all kernels at once. The correct approach
is to convert one kernel pair at a time, keeping an f32↔f16
conversion at the boundary, then expanding until the entire
activation pipeline is f16.

**Step 1: f16 RMSNorm output (smallest change, validates path)**

Create a new kernel `rms_norm_f16out` that reads f32 residual
and writes f16 norm_out:

```
kernel void rms_norm_f16out(
    device const float* x,         // f32 residual input
    device half*        out,       // f16 norm output  ← CHANGED
    device const half*  scale,
    constant uint&      hidden_size,
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
```

The norm_out buffer changes from `Buffer` (f32) to `HalfBuffer`
(f16) in `transformer.zig`. The QMV kernels that read norm_out
must also change their input type to `device const half*` (or
`constant half*` for the constant-cache variants).

This one change converts the RMSNorm→QMV boundary to f16, which
covers the two hottest kernel pairs per block:

- attn_norm → Q proj (+ K/V fused pair)
- ffn_norm → gate/up fused pair

That is 2 × 28 = 56 dispatch pairs where the activation buffer
traffic is halved.

**Step 2: f16 QMV input in qmv_const**

Change the QMV constant-cache input from `constant float*` to
`constant half*`. The inner loop changes:

```
// Before (f32):
float x0 = input[base_col + 0];
float x1 = input[base_col + 1];
...

// After (f16 → f32 on load):
float x0 = float(input[base_col + 0]);
float x1 = float(input[base_col + 1]);
...
```

Accumulation stays f32. Only the load type changes. The constant
cache now holds 4 KB instead of 8 KB for the input vector.

Apply to: `qmv_const`, `qmv_fused_pair_const`,
`qmv_const_multigroup`, and all their dispatch call sites.

**Step 3: f16 QMV output**

Change QMV output buffers from f32 to f16. The QMV kernel
already accumulates in f32 internally; just cast the output:

```
// In qmv_const, change the output write:
if (lane == 0) output[row0] = half(acc0);  // was: float
```

Change `device float* output` to `device half* output` in the
kernel signature. Change the corresponding Zig buffer from
`Buffer` to `HalfBuffer`.

This converts Q, K, V, O, gate, up, down, and lm_head outputs
to f16. The kernels that consume these (QK-norm, RoPE, SiLU,
residual_add, GQA attention, sampling) must also read f16.

**Step 4: f16 lightweight kernels**

Convert `silu_elementwise_mul`, `rope`, `kv_cache_update`, and
`residual_add` to f16 input/output. These are small elementwise
kernels — the change is trivial (swap `float` for `half` in the
signature, cast to f32 for any arithmetic, cast back).

`kv_cache_update` currently converts f32→f16 at write time.
With f16 input, it becomes a straight copy (or even a no-op if
the projection output IS the cache write location).

**Step 5: f16 GQA attention Q/output**

The GQA kernel reads f32 Q and writes f32 output. Change both
to f16. The KV cache is already f16. The Q·K dot product and
V accumulation stay f32 internally.

**Step 6 (optional): f16 residual stream**

The residual buffer is the one place where f16 might cause
precision issues, because the residual is a running sum across
28 layers. Options:
a. Keep residual as f32 (safest). Each kernel that reads/writes
the residual does an f32↔f16 cast at the boundary. The
residual itself stays f32.
b. Use f16 residual with Kahan summation in residual_add.
c. Use f16 residual and accept the tiny precision loss (the
model was trained in bf16 anyway, so it is robust to this).

Start with option (a). If benchmarks show the f32 residual is
not a bottleneck, stop there. The residual is only 2048 floats
= 8 KB per token — the bandwidth cost of keeping it f32 is
negligible compared to the activation buffers it feeds.

### Files to modify

Metal kernels (nn/src/shaders/):

- `transformer.metal`: rms_norm, silu_elementwise_mul, rope,
  kv_cache_update, gqa_attention, embedding_lookup, residual_add
  — add f16 variants or change signatures.
- `compute.metal`: qmv_const, qmv_fused_pair_const,
  qmv_const_multigroup — change `constant float* input` to
  `constant half* input` and `device float* output` to
  `device half* output`. Inner accumulation stays f32.

Zig dispatch (nn/src/):

- `transformer.zig`: change activation buffer types from
  `Buffer` (f32) to `HalfBuffer` (f16). Update all
  `setBuffer` calls. Change `decode_activation_elements`
  sizing to account for f16 (half the bytes).
- `metal.zig`: add new pipeline states for f16 kernel
  variants (e.g., `rms_norm_f16out`, `qmv_const_f16`).
- `model.zig`: change buffer allocation sizes (halved for
  f16 activations).

### Precision rules

- QMV internal accumulation: ALWAYS f32.
- RMSNorm sum-of-squares: ALWAYS f32.
- Softmax in GQA attention: ALWAYS f32.
- Residual buffer: f32 (safest) or f16 (test carefully).
- Everything else: f16 loads and stores are fine.

### What the engine agent should do

This is a multi-step optimisation. Each step should be a
separate experiment:

1. **Experiment A**: rms_norm_f16out + qmv_const with half input.
   Convert the norm_out buffer to f16. Add new kernels, new
   pipelines, update dispatch. check → test → bench.
   Expected: measurable improvement from halving the norm_out
   buffer traffic across 56 dispatch pairs.

2. **Experiment B**: f16 QMV output. Convert Q, K, V, O, gate,
   up, down, mlp_out buffers to f16. Update all downstream
   kernels. check → test → bench.

3. **Experiment C**: f16 lightweight kernels + GQA. Complete the
   f16 pipeline. check → test → bench.

Each experiment is independently valuable. Even step 1 alone
should show a measurable win.

## Later phases (after f16 is done)

### Async token pipelining (+2–3%)

Replace `commitAndWait` with a completion handler and double-
buffered logits. While the CPU samples token N, the GPU can
start token N+1's embedding lookup and first few layers.

Files: `transformer.zig` (generation loop).
Risk: Low — the sampling overhead is small (~50–100 μs), so the
win is modest. But it is free throughput.

### GQA attention vectorisation (+2–5% at long contexts)

Use `half4` loads in the GQA attention kernel for both Q·K dot
product and V accumulation. The inner loops are currently fully
scalar. With `half4`, load 4 KV cache elements per memory
transaction, convert to float4, dot with float4 from Q.
Reduces KV memory transactions by 4×.

Replace the three-pass softmax (max → exp+sum → normalize) with
a single-pass online softmax. Reduces passes over the scores
buffer from 3 to 1.

Files: `transformer.metal` (gqa_attention kernel).

## Expected cumulative impact

| Optimisation           | Expected | Effort | nnzap tok/s |
| ---------------------- | -------: | ------ | ----------: |
| Baseline (post uint32) |        — | —      |         169 |
| **f16 activations**    |  +15–20% | Medium |     195–205 |
| Async pipelining       |    +2–3% | Medium |     199–211 |
| GQA vectorise          |    +2–5% | Medium |     203–222 |

Conservative estimate with f16 alone: **195–205 tok/s**.
Full roadmap: **203–222 tok/s** — within striking distance of
MLX's 224.

## MLX kernel architecture reference

Extracted from the PrismML fork's compiled `mlx.metallib` and
upstream MLX source (`ml-explore/mlx` on GitHub).

### Weight format

MLX affine quantization at 1 bit:

- Weights packed into **uint32** words (32 weights per word).
- Per-group **scale** and **bias** (both same type as activation).
- Dequantisation: `w_real = scale * w_int + bias`.
- For {-1, +1} mapping: scale=2, bias=-1.
- Group sizes: 32, 64, or 128. Bonsai uses 128.

Our Q1_0_g128 format:

- Weights packed into **uint8** bytes (8 weights per byte).
- Per-group **f16 scale** only (no bias).
- Dequantisation: `bit=1 → +scale, bit=0 → -scale`.
- Mathematically equivalent to affine with scale=2s, bias=-s.

### Kernel template: affine_qmv

MLX template parameters: `<T, group_size, bits, batch>`.
For Bonsai 1-bit: `<bfloat16_t, 128, 1, 0>`.

Key constants:

- 2 simdgroups per threadgroup.
- 4 output rows per simdgroup → 8 rows per threadgroup.
- pack_factor = 32 (32 weights per uint32 for bits=1).
- Accumulation always in float (not half).
- Reduction via `simd_sum` (hardware intrinsic).

Inner loop structure:

1. Load block of x values, pre-scale by powers of 2.
   Compute sum(x) for bias correction.
2. For each of 4 rows: load packed uint32, call qdot.
   qdot masks without shifting — the pre-scaled x values
   compensate. Result: `scale * accum + bias * sum_x`.
3. After K-dimension loop: `simd_sum` reduction.
   Lane 0 writes output.

No threadgroup shared memory for the matvec path. The input
vector stays in registers, reused across 4 rows. Only the
tiled matmul (qmm) path uses threadgroup memory.

### Key architectural differences from nnzap

| Aspect              | MLX (PrismML)          | nnzap (current)       |
| ------------------- | ---------------------- | --------------------- |
| Activation type     | **bfloat16**           | f32 ← FIX THIS        |
| Weight word size    | uint32 (32/load)       | **uint32** ✅ done    |
| Rows per simdgroup  | 4                      | 2 (tried 4, flat)     |
| Simdgroups per TG   | 2                      | **16**                |
| Rows per TG         | 8                      | **32**                |
| Input vector source | registers (device mem) | **constant cache**    |
| Threadgroup memory  | none                   | **none** (const path) |
| Inner loop strategy | accum + bias\*sum      | select (equivalent)   |
| Graph fusion        | **yes** (lazy eval)    | none (eager)          |
| GPU/CPU overlap     | **yes** (async eval)   | commitAndWait         |

Our constant-cache approach and 32-rows-per-TG are advantages.
The remaining gap is almost entirely **activation precision**.

## What NOT to do

- Do not attempt more kernel inner-loop micro-optimisations.
  Phases 1A, 1C, and multiple fusion experiments have shown
  the kernel compute is not the bottleneck.
- Do not attempt uint32 loads for qmv_const_multigroup.
  The alignment issue (byte_start not 4-byte aligned for
  K=5504) makes this impractical without a lane reassignment
  that adds complexity for marginal gain on 28 dispatches.
- Do not rewrite the entire QMV kernel from scratch.
- Do not remove existing kernel variants during development.
  Add new f16 variants alongside existing f32 ones, validate,
  then remove the old ones.
- Do not change the weight packing format (Q1_0_g128).
