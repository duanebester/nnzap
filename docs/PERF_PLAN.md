# Performance plan: closing the gap with PrismML MLX fork

## Competitive benchmark (M2 Max, quiet system)

| Runtime                    | Decode (tok/s) | Notes                        |
| -------------------------- | -------------: | ---------------------------- |
| **nnzap** (Zig + Metal)   |    **146–166** | 166 peak, 146 under load     |
| **PrismML MLX fork**      |        **224** | Custom 1-bit MLX fork        |

Gap: **~35%**. Target: **224 tok/s**.

PrismML's fork extends Apple's MLX framework with `bits=1` support
in the existing `affine_qmv` kernel template. They get years of MLX
Metal kernel infrastructure for free. We wrote everything from
scratch in ~22K lines of Zig. The gap is not a design flaw — it is
a short list of concrete, measurable differences.

## Root cause analysis

Five factors explain the 35% gap, ordered by impact.

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

| Metric              | nnzap     | MLX (estimated) |
| ------------------- | --------: | --------------: |
| GPU dispatches      |   **451** |        ~100–150 |
| Memory barriers     |   **366** |          ~50–80 |
| Obj-C API calls     | **3,748** |        ~500–800 |

MLX uses lazy evaluation with graph compilation. Operations like
RMSNorm→QMV, ResidualAdd→RMSNorm, and SiLU→mul are candidates
for fusion. We encode them as separate dispatches with full-scope
`memoryBarrierWithScope:` barriers between each.

At ~100 ns per Obj-C message send, our 3,748 API calls cost
**~375 μs of pure CPU overhead per token** — roughly 6% of the
~6 ms token budget at 166 tok/s.

### 3. The qdot accumulation trick (~3–5%)

MLX's quantized matvec exploits an algebraic identity:

```
dot(x, dequant(w)) = scale * dot(x, w_int) + bias * sum(x)
```

For 1-bit affine (mapping {0,1} → {-1,+1}: scale=2, bias=-1):
  accum = sum of x[i] where bit=1 (no negation needed).
  result = 2 * accum - sum_x.

Our approach does `select(-x, x, bit)` for every element — a
conditional negation per bit. MLX just accumulates the set bits
and applies a single correction. Fewer ALU ops per element.

MLX also packs into **uint32** (32 weights per load) vs our
**uint8** (8 weights per load) — 4× fewer weight memory
instructions in the hot loop.

### 4. More rows per simdgroup: 4 vs 2 (~2–4%)

MLX `qmv_fast_impl`: 4 output rows per simdgroup.
Our `qmv_const`: 2 output rows per simdgroup.

The input vector is loaded once and reused across all rows.
Processing 4 rows doubles the compute-to-bandwidth ratio for
weight reads at zero extra input cost.

### 5. No CPU/GPU pipelining (~2–3%)

We block on `commitAndWait` every single token:

```
transformer.zig L1878–1891:
  device.commitAndWait(cmd);
  // CPU idle until GPU finishes entire forward pass.
  // Then: sample token, THEN create next command buffer.
```

MLX's async evaluation naturally pipelines GPU compute with CPU
scheduling. The GPU never waits for the CPU to finish sampling.

## Detailed per-token dispatch audit

### Per decoder block (16 dispatches, 13 barriers)

Attention half (11 dispatches, 8 barriers):
  1. RMSNorm (attn_norm)         → barrier
  2. QMV (Q projection)          |
  3. QMV fused pair (K+V)        → barrier
  4. RMSNorm (Q head norm)       |
  5. RMSNorm (K head norm)       → barrier
  6. RoPE (Q)                    |
  7. RoPE (K)                    → barrier
  8. KV cache update             → barrier
  9. GQA attention               → barrier
  10. QMV (O projection)         → barrier
  11. Residual add               → barrier

MLP half (5 dispatches, 5 barriers):
  12. RMSNorm (ffn_norm)         → barrier
  13. QMV fused pair (gate+up)   → barrier
  14. SiLU elementwise mul       → barrier
  15. QMV (down projection)      → barrier
  16. Residual add               → barrier

Full forward pass (28 layers):
  Dispatches: 16 × 28 + 3 = 451
  Barriers:   13 × 28 + 2 = 366
  setBuffer:  72 × 28 + 9 ≈ 2,025
  setBytes:   16 × 28 + 2 ≈ 450
  Total Metal API calls: ~3,748

All 366 barriers use global `memoryBarrierWithScope:` (all
buffers) even when only one or two specific buffers have a
RAW dependency. All 366 are necessary for correctness — every
one guards a true read-after-write on the same buffer. But the
scope is broader than needed.

### QMV projection dimensions (Bonsai 1.7B)

| Projection       | M      | K    | Kernel used            | Rows/TG |
| ---------------- | -----: | ---: | ---------------------- | ------: |
| Q proj           |  2,048 | 2048 | `qmv_const`            |      32 |
| K+V fused        |  1,024 | 2048 | `qmv_fused_pair_const` |      32 |
| O proj           |  2,048 | 2048 | `qmv_const`            |      32 |
| gate+up fused    |  6,144 | 2048 | `qmv_fused_pair_const` |      32 |
| down proj        |  2,048 | 6144 | `qmv_fast_multigroup`  |      16 |
| LM head (tied)   | 151669 | 2048 | `qmv_const`            |      32 |

The `qmv_const` kernels use Metal's constant address space for
the input vector (zero threadgroup memory, zero barriers for the
input load). The `down_proj` uses shared-memory `qmv_fast_multigroup`
because K=6144 exceeds the constant-cache size limit.

The LM head is the single largest dispatch: 151,669 / 32 ≈ 4,740
threadgroups. This one dispatch is a significant fraction of total
per-token compute.

## Activation precision map

| Data                         | Precision     | Bandwidth cost |
| ---------------------------- | ------------- | -------------- |
| Activations (residual, Q/K/V, MLP intermediates) | **f32** | 2× vs f16 |
| KV cache                     | **f16**       | Optimal        |
| Norm scales                  | **f16**       | Optimal        |
| Projection weights           | **1-bit** + f16 group scales | Optimal |
| Logits output                | **f32**       | Required       |

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

Opportunities:
  - `half4` loads would reduce KV cache memory transactions 4×.
  - Online softmax (FlashAttention-style) would fuse 3 passes into 1.
  - `simd_sum` for dot product reduction within a thread.
  - Hybrid: use threadgroup memory when seq_len × 4 ≤ 32 KB,
    fall back to device memory for longer contexts.

At short contexts (early decode), this kernel is cheap. At longer
contexts it becomes increasingly important.

## Prioritised optimisation roadmap

### Tier 1 — Kernel-level wins (low effort, high confidence)

#### 1A. Four rows per simdgroup in qmv_const (+3–5%)

Change `qmv_const` from 2 → 4 rows per simdgroup. The 16
simdgroups become 8, each processing 4 rows. Total 32 rows per
threadgroup stays the same. The inner loop gets two more
accumulators (`sig2`, `sig3`) and two more byte loads per
iteration, but the 8 float reads from constant cache are shared
across all 4 rows (register reuse, not extra bandwidth).

Files: `compute.metal` (kernel), `transformer.zig` (dispatch).
Validation: all 71 tests pass, bench shows improvement.

#### 1B. uint32 weight loads in hot loop (+2–4%)

Load 4 packed bytes at once as a `uint32` instead of 1 byte at
a time. Process 32 weight elements per inner-loop iteration
instead of 8. This reduces the inner loop trip count by 4× and
cuts weight load instructions by 4×. The unrolled body processes
32 elements with 32 `select` + accumulate operations.

For K=2048 with 32 SIMD lanes: each lane has 64 columns = 8
bytes = 2 uint32 loads instead of 8 byte loads. Inner loop goes
from 8 iterations to 2.

Files: `compute.metal` (kernel inner loop).
Validation: numerical correctness against CPU reference.

#### 1C. Accumulate-and-correct instead of select (+2–3%)

Replace `select(-x, x, bit)` per element with:
  accum += x when bit=1 (else nothing).
  result = 2 * accum - sum_x (applied once after the loop).

This eliminates the conditional negation entirely. The inner
loop becomes: mask the packed uint32, test each bit, add x
if set. Then a single correction: `result = scale * (2 * accum
- sum_x)` where `sum_x = simd_sum(lane_sum_x)` is computed
once during vector loading.

This is the MLX `qdot` approach — algebraically equivalent to
our `select` approach, but with fewer ALU instructions.

Files: `compute.metal` (new kernel variant or refactored loop).
Validation: numerical correctness against CPU reference.

### Tier 2 — Activation precision (medium effort, biggest payoff)

#### 2A. f16 activation pipeline (+15–20%)

Switch activation buffers from f32 to f16. This is the single
highest-impact change and touches every kernel:

Metal kernels to modify:
  - `rms_norm`: input/output f16, accumulation stays f32.
  - `silu_elementwise_mul`: input/output f16.
  - `residual_add`: in-place f16.
  - `rope`: in-place f16.
  - `embedding_lookup`: output f16 (currently f32).
  - `gqa_attention`: Q input f16, output f16.
  - `kv_cache_update`: input becomes f16 (KV cache already f16).
  - `qmv` / `qmv_const` / all variants: input vector f16,
    output f16. Internal accumulation stays f32.

Zig dispatch to modify:
  - `transformer.zig`: change `Buffer` (f32) to `HalfBuffer`
    (f16) for all activation scratch. Allocation sizes halve.
  - `model.zig`: buffer allocation changes.
  - `metal.zig`: ensure `dispatch1D` / `dispatchCustom` work
    with HalfBuffer arguments.

Risk: Precision loss in residual stream accumulation. Mitigate
by keeping the residual buffer in f32 and converting at kernel
boundaries, or by using the Kahan summation trick. Test against
CPU reference and verify generation quality.

This change halves activation memory traffic across the entire
pipeline — 451 dispatches all benefit. At the memory-bandwidth
wall, this is the most direct path to closing the gap.

### Tier 3 — Pipeline-level wins (medium effort)

#### 3A. Async token pipelining (+2–3%)

Replace `commitAndWait` with a completion handler and double-
buffered logits. While the CPU samples token N, the GPU can
start token N+1's embedding lookup and first few layers.

Requires:
  - Two logits buffers (double-buffered).
  - MTLSharedEvent or completion handler instead of commitAndWait.
  - Careful ordering: next token's embedding depends on sampling
    result, so only the embedding lookup must wait — the rest
    of the pipeline can be pre-encoded.

Files: `transformer.zig` (generation loop).
Risk: Low — the sampling overhead is small (~50–100 μs), so the
win is modest. But it is free throughput.

#### 3B. Reduce barrier scope (+1–2%)

Switch from `memoryBarrierWithScope:` (global) to
`memoryBarrierWithResources:count:` (targeted). Pass only the
specific buffer(s) that have a RAW dependency.

Example: after RoPE(Q) + RoPE(K), only `k` needs to be visible
to `kv_cache_update`. The global barrier synchronizes every
buffer unnecessarily.

On Apple Silicon unified memory the practical difference may be
small, but it is architecturally correct and removes any hidden
serialization of unrelated buffer writes.

Files: `transformer.zig` (barrier calls), `metal.zig` (helper).

#### 3C. Kernel fusion: RMSNorm + QMV (+2–3%)

Fuse the RMSNorm output directly into the QMV input. Instead of:
  dispatch RMSNorm → barrier → dispatch QMV (reads norm output)

Create a fused kernel that:
  1. Cooperatively computes RMSNorm in threadgroup memory.
  2. Uses the normalized vector as the QMV input.

This eliminates one dispatch + one barrier + one full activation
buffer write/read roundtrip. With 4 RMSNorm→QMV pairs per block
(attn_norm→Q, ffn_norm→gate+up, plus Q/K norms→RoPE), fusing
the first two saves 2 × 28 = 56 dispatches and 56 barriers.

Complexity: High. The RMSNorm is 256 threads (one threadgroup),
while QMV is 512 threads with ceil(M/32) threadgroups. The fused
kernel would need a different structure — perhaps RMSNorm as a
preamble within each QMV threadgroup, reading from a shared
normalized vector.

### Tier 4 — GQA attention (important at longer contexts)

#### 4A. Vectorised KV reads (+2–5% at long contexts)

Use `half4` loads in the GQA attention kernel for both Q·K dot
product and V accumulation. The inner loops are currently fully
scalar:

```
transformer.metal L322–328:
  for (uint d = 0; d < head_dim; d++) {
      dot_product += q[d] * float(k[t * head_dim + d]);
  }
```

With `half4`:
  Load 4 KV cache elements per memory transaction.
  Convert to float4, dot with float4 from Q.
  Reduces KV memory transactions by 4×.

Files: `transformer.metal` (gqa_attention kernel).

#### 4B. Online softmax (+2–3% at long contexts)

Replace the three-pass softmax (max → exp+sum → normalize) with
a single-pass online softmax that maintains running max and sum.
This reduces passes over the scores buffer from 3 to 1, cutting
device-memory bandwidth for the softmax phase by 3×.

Files: `transformer.metal` (gqa_attention kernel).

### Tier 5 — Down-projection special case

#### 5A. Optimise qmv_fast_multigroup for K=6144 (+1–2%)

The down projection (M=2048, K=6144) is the only per-block QMV
that cannot use the constant-cache path. It falls back to
`qmv_fast_multigroup` with 24 KB of threadgroup shared memory.
This is called 28 times per token.

Options:
  - Increase to 4 rows per simdgroup (same as 1A but for this
    variant).
  - Use a split-K approach: divide K=6144 into two dispatches
    of K=3072, each fitting the constant cache, then add the
    partial results.
  - Hybrid: load the first 2048 elements from constant cache,
    the remaining 4096 from shared memory.

## Expected cumulative impact

| Optimisation            | Expected | Effort | nnzap tok/s |
| ----------------------- | -------: | ------ | ----------: |
| Baseline                |       — | —      |         166 |
| 1A. 4 rows/simdgroup    |   +3–5% | Low    |     171–174 |
| 1B. uint32 weight loads |   +2–4% | Low    |     175–181 |
| 1C. accum-and-correct   |   +2–3% | Low    |     178–186 |
| **2A. f16 activations** | +15–20% | Medium |     205–224 |
| 3A. Async pipelining    |   +2–3% | Medium |     209–230 |
| 3B. Targeted barriers   |   +1–2% | Low    |     211–235 |
| 3C. Fused RMSNorm+QMV   |   +2–3% | High   |     216–242 |
| 4A+4B. GQA vectorise    |   +2–5% | Medium |     220–254 |

Conservative estimate with Tier 1 + Tier 2: **205–224 tok/s**.
Full roadmap: **220–254 tok/s** — matching or exceeding MLX.

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

| Aspect                | MLX (PrismML)          | nnzap                |
| --------------------- | ---------------------- | -------------------- |
| Activation type       | **bfloat16**           | f32                  |
| Weight word size      | **uint32** (32/load)   | uint8 (8/load)       |
| Rows per simdgroup    | **4**                  | 2                    |
| Simdgroups per TG     | 2                      | **16**               |
| Rows per TG           | 8                      | **32**               |
| Input vector source   | registers (device mem) | **constant cache**   |
| Threadgroup memory    | none                   | **none** (const path)|
| Inner loop strategy   | accum + bias*sum       | select(-x, x, bit)  |
| Graph fusion          | **yes** (lazy eval)    | none (eager)         |
| GPU/CPU overlap       | **yes** (async eval)   | commitAndWait        |

Note: our constant-cache approach and 32-rows-per-TG are
advantages. The gap comes from activation precision, wider
weight loads, and graph-level overhead — not from fundamentally
worse kernel design.

## Implementation order for the engine agent

The engine agent should tackle these optimisations in this order.
Each is a single experiment. Validate with check → test → bench
after each change.

### Experiment sequence

1. **1A: 4 rows per simdgroup** — modify `qmv_const` and
   `qmv_fused_pair_const` inner loops. Add `sig2`/`sig3`
   accumulators. Change rows per simdgroup from 2 to 4.
   Update `qmv_const_multigroup` and `qmv_fused_pair_const`
   similarly. Adjust dispatch in `transformer.zig` if the
   rows-per-TG calculation changes.

2. **1B: uint32 weight loads** — refactor the inner loop to
   load `*(device const uint32_t*)(packed_bits + offset)`
   instead of byte-by-byte. Unroll 32 `select` operations
   per uint32 (or use the accum-and-correct approach from 1C).
   This can be combined with 1C in a single experiment.

3. **1C: accum-and-correct** — change the accumulation strategy.
   Pre-compute `lane_sum_x` during input loading. In the inner
   loop, accumulate only where bit=1: `accum += x * bit`. After
   the loop: `result = scale * (2 * accum - lane_sum_x)`.
   Use `simd_sum` on both `accum` and `lane_sum_x`.

4. **2A: f16 activations** — the big one. Start with a single
   kernel (e.g., RMSNorm) to validate the f16 path, then expand
   to all kernels. Change activation buffers from `Buffer` to
   `HalfBuffer`. Update all `setBuffer` calls. Keep QMV internal
   accumulation in f32. Test generation quality carefully.

5. **3A: async pipelining** — after f16 is stable, add double-
   buffered logits and replace `commitAndWait` with a completion
   handler.

6. **4A: vectorised GQA** — use `half4` loads in gqa_attention.
   Independent of the other changes.

### What NOT to do

- Do not attempt kernel fusion (3C) before Tier 1 and Tier 2
  are done. The complexity is high and the payoff is smaller.
- Do not rewrite the entire QMV kernel from scratch. Make
  incremental changes to the existing variants.
- Do not remove existing kernel variants. Add new ones alongside
  for A/B comparison, then remove the old ones after validation.
- Do not change the weight packing format (Q1_0_g128). The
  model files use this format and the safetensors loader depends
  on it. Kernel-level repacking at load time is acceptable if
  needed for uint32 alignment.
