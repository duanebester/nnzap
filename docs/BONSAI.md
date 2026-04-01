# Bonsai 1-bit: implementation plan

Bring [PrismML's 1-bit Bonsai](https://prismml.com/news/bonsai-8b)
language models to nnzap — inference only, Metal compute, zero-copy
unified memory.

## Why Bonsai

Bonsai is a family of end-to-end 1-bit language models (1.7B, 4B, 8B)
based on the Qwen3 architecture. Every weight — embeddings, attention
projections, MLP projections, LM head — is a single bit. The 8B
model is 1.15 GB. It runs at 131 tok/s on an M4 Pro via MLX
(Python + C++ + Metal).

The model is interesting to nnzap for three reasons:

1. **Unified memory is the killer feature for 1-bit.** The entire
   model fits trivially in shared memory. No copies, no transfers.
   This is the regime where Apple Silicon dominates.

2. **1-bit matmul is simpler than general matmul.** The inner loop
   replaces multiplication with conditional accumulation — a
   `select(0, x, bit)` per weight. The kernel complexity is lower
   than the tiled f32 matmul nnzap already has.

3. **All dimensions are known at compile time.** Qwen3-8B has a
   fixed architecture. Every buffer offset, dispatch grid size, and
   assertion bound can be resolved at comptime. Zero runtime overhead.

A Zig + Metal implementation with zero-copy buffers, comptime dispatch,
and no Python overhead has a realistic shot at matching or exceeding
MLX's throughput — especially on the token generation path where
per-token overhead matters most.

## Target architecture: Qwen3

All three Bonsai sizes share the same transformer architecture. Only
the dimensions change.

```
┌──────────────────────────────────────────────────────────────────┐
│  Token IDs                                                       │
│  ┌───────────────────────────────────────────────────┐           │
│  │  Embedding lookup (1-bit packed, vocab × hidden)  │           │
│  └───────────────────────┬───────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────┐           │
│  │  Transformer Block  ×  num_layers                 │           │
│  │  ┌─────────────────────────────────────────────┐  │           │
│  │  │  RMSNorm (f16 scales)                       │  │           │
│  │  │  GQA Attention (1-bit Q/K/V/O projections)  │  │           │
│  │  │    • RoPE positional encoding               │  │           │
│  │  │    • KV cache append                        │  │           │
│  │  │    • Scaled dot-product + causal mask        │  │           │
│  │  │  Residual add                               │  │           │
│  │  │  RMSNorm (f16 scales)                       │  │           │
│  │  │  SwiGLU MLP (1-bit gate/up/down)            │  │           │
│  │  │    • gate = silu(x @ W_gate)                │  │           │
│  │  │    • up   = x @ W_up                        │  │           │
│  │  │    • down = (gate ⊙ up) @ W_down            │  │           │
│  │  │  Residual add                               │  │           │
│  │  └─────────────────────────────────────────────┘  │           │
│  └───────────────────────┬───────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────┐           │
│  │  RMSNorm (final)                                  │           │
│  │  LM Head (1-bit, hidden → vocab)                  │           │
│  │  Softmax / sampling → next token                  │           │
│  └───────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

### Dimensions

Values from `config.json` in each
[`prism-ml/Bonsai-*-mlx-1bit`](https://huggingface.co/prism-ml)
repository (retrieved 2025-07-17):

| Property            | 1.7B    | 4B      | 8B      |
| ------------------- | ------- | ------- | ------- |
| Layers              | 28      | 36      | 36      |
| Hidden size         | 2048    | 2560    | 4096    |
| Intermediate (MLP)  | 6144    | 9728    | 12288   |
| Query heads         | 16      | 32      | 32      |
| KV heads            | 8       | 8       | 8       |
| Head dim            | 128     | 128     | 128     |
| Vocab size          | 151669  | 151669  | 151669  |
| Context length      | 32768   | 32768   | 65536   |
| RoPE theta          | 1000000 | 5000000 | 1000000 |
| Tie word embeddings | true    | true    | false   |
| MLX safetensors     | 269 MB  | 629 MB  | 1.28 GB |

The 1.7B is the implementation target. Every line of code transfers
directly to 4B and 8B — only the comptime config constants change.

### Architecture details

Discovered from the actual model files — not in the original
Qwen3 documentation:

- **QK norms.** Each attention layer has `q_norm` and `k_norm`
  (RMSNorm applied to Q and K _after_ projection, before RoPE).
  These are f32 weight vectors of size `head_dim` (128).
- **Tied word embeddings.** The 1.7B and 4B models set
  `tie_word_embeddings: true` — the embedding table and the LM
  head share the same weight tensor. The 8B does not tie them.
- **YaRN RoPE scaling.** All three models use `rope_type: "yarn"`
  with `factor: 4.0` to extend context beyond the base training
  length (`original_max_position_embeddings` = 8192 for 1.7B/4B,
  16384 for 8B).
- **Decoupled head dim.** For the 4B model, `hidden_size` (2560)
  ≠ `num_query_heads * head_dim` (32 × 128 = 4096). The Q
  projection up-sizes from 2560 to 4096. The comptime assertion
  `hidden_size == query_dim` must be relaxed for 4B.
- **No attention bias.** `attention_bias: false` and `no_bias: true`
  for all three models — no bias terms in Q/K/V/O projections.

## Weight format: 1-bit with group scales

Each weight is a single bit: `0` maps to `−scale`, `1` maps to
`+scale`. Every group of 128 weights shares one scale factor.

### Packing

8 weights per `u8`, 32 per `u32`, LSB-first:

```
byte:   [b7 b6 b5 b4 b3 b2 b1 b0]
weight:  w7 w6 w5 w4 w3 w2 w1 w0
```

### Dequantization

The MLX safetensors format stores each quantized projection as
three tensors — `.weight` (packed uint8 bits), `.scales` (f16),
and `.biases` (f16) — using an affine encoding:

```
w_reconstructed[i] = mlx_scale * bit[i] + mlx_bias
```

where `mlx_scale = 2 * s` and `mlx_bias = −s` for some learned
scale `s`. At load time we convert to symmetric form:

```
s = mlx_scale / 2
w_reconstructed[i] = if (bit[i] == 1) +s else −s
```

This is the same encoding our `qmv` and `qmm` kernels already
use. The conversion is a single division per group at load time.

For reference, the GGUF format stores the same information more
compactly (one f16 scale per group, no bias — 1.125 vs 1.25 bits
per weight), but the GGUF container format is significantly more
complex to parse than safetensors (see "Safetensors parser" below).

### Memory layout for a weight matrix [M × K]

```
┌─────────────────────────────────────────────────────────┐
│  Packed bits: ceil(M * K / 8) bytes                     │
│  [byte_0] [byte_1] ... [byte_{ceil(M*K/8) - 1}]        │
├─────────────────────────────────────────────────────────┤
│  Scales: ceil(M * K / 128) × f16                        │
│  [scale_0] [scale_1] ... [scale_{num_groups - 1}]       │
└─────────────────────────────────────────────────────────┘

num_groups = divCeil(M * K, 128)
packed_bytes = divCeil(M * K, 8)
total_bytes = packed_bytes + num_groups * 2
```

## Incremental implementation plan

The plan has five steps. Each step produces a working, testable
artifact. No step depends on code that has not been tested in a
prior step.

### Step 0 — 1-bit MNIST MLP ✅

**Goal.** Validate the 1-bit quantization and inference kernels
against known-good results, using the existing MNIST training
pipeline.

**What exists.** nnzap trains a 784→128→64→10 MLP to 97.24% accuracy
on MNIST with f32 weights. f16 inference weights are already
supported via `HalfBuffer` and the `f32_to_f16` kernel.

**What was built.**

1. `PackedBuffer` — a Metal shared buffer that stores 1-bit packed
   weights with per-group f16 scales in a single allocation.
   `setPackedBuffer` binds the same MTLBuffer twice at consecutive
   indices with different byte offsets.
2. `f32_to_1bit` kernel — quantize trained f32 weights to 1-bit
   Q1_0_g128 format on the GPU. One thread per 128-weight group.
3. `qmv` kernel — 1-bit matrix-vector multiply using simdgroups.
   Requires K % 32 == 0 (see implementation note below).
4. `qmm` kernel — 1-bit tiled matrix-matrix multiply (16×16 output
   tiles, K tiles of 128). Used for all MNIST inference.
5. `forwardInfer1Bit` — per-layer inference path in `Network` that
   dispatches `qmm` → `bias_add` → activation.
6. `quantizeWeightsTo1Bit` — dispatches `f32_to_1bit` per layer,
   copies biases (f32) into a separate contiguous buffer on the CPU.
7. `mnist_1bit.zig` — integration test with CPU reference
   verification.

**Verification results.**

| Metric                   | Value                           |
| ------------------------ | ------------------------------- |
| f32 inference accuracy   | 97.24% (10,000 test images)     |
| 1-bit inference accuracy | 82.56% (14.68pp below f32)      |
| CPU reference check      | PASS (all 10 logits < 0.05 err) |
| Unit tests               | 20/20 pass                      |

The 14.68pp accuracy drop is unsurprising but we have no literature
to call it "expected". This is naive post-training quantization:
binarize the sign, take mean absolute value as scale. No
straight-through estimators, no learned scales, no calibration
data — the techniques that Bonsai and BitNet use to make 1-bit
work. The accuracy number is a sanity check (well above 10%
random chance, confirming the kernels compute something
meaningful), not a benchmark. **The CPU reference match is the
real correctness proof** — it is independent of model accuracy.

**Implementation note: `qmv` requires K % 32 == 0.** The simdgroup
kernel partitions K across 32 lanes, so each lane must process an
integer number of columns. MNIST layer 0 has K=784; 784 / 32 = 24.5
— not even. The `qmm` tiled kernel handles arbitrary dimensions
correctly (with bounds checks), so Step 0 uses `qmm` for all batch
sizes. The `qmv` kernel is wired up and ready for transformer
dimensions where K ≥ 128 and K % 128 == 0.

**Implementation note: weight layout is [K × N], not [N × K].** The
original plan assumed each row of W is one output neuron (`[N × K]`).
The existing nnzap matmul convention stores weights as
`[in_size × out_size]` (`[K × N]`), and `f32_to_1bit` quantizes
them in-place in that order. The `qmm` kernel was adapted to index
into `[K × N]` packed storage, avoiding any weight transposition.
The `qmv` kernel assumes `[N × K]` (row per output neuron) and
will need the GGUF weights in that layout — which they naturally
are, since GGUF stores each tensor in its own contiguous region.

**Deliverables.**

| File                           | Contents                                            |
| ------------------------------ | --------------------------------------------------- |
| `nn/src/metal.zig`             | `PackedBuffer`, `setPackedBuffer`, `dispatchCustom` |
| `nn/src/shaders/compute.metal` | `f32_to_1bit`, `qmv`, `qmm` kernels                 |
| `nn/src/network.zig`           | `forwardInfer1Bit`, `quantizeWeightsTo1Bit`         |
| `nn/src/layout.zig`            | `divCeil`, packed size helpers, comptime offsets    |
| `nn/src/root.zig`              | Re-exports: `PackedBuffer`, `divCeil`               |
| `nn/examples/mnist_1bit.zig`   | Integration test + CPU reference                    |
| `nn/build.zig`                 | `run-1bit` build step                               |

### Step 1 — transformer primitives ✅

**Goal.** Implement and test every kernel needed by a Qwen3
transformer block, in isolation, before combining them.

**What exists.** Step 0 validated the 1-bit quantization and
inference kernels (`f32_to_1bit`, `qmv`, `qmm`) against
known-good MNIST results.

**What was built.**

1. `TransformerConfig` — comptime type (same pattern as
   `NetworkLayout`) that resolves all buffer sizes, weight
   offsets, KV cache sizes, and activation scratch requirements
   at compile time for a given `TransformerDesc`. Three model
   configs validated: `Bonsai1_7B`, `Bonsai4B`, `Bonsai8B`.
2. `TransformerPipelines` — compiles `transformer.metal` as a
   separate shader library and holds pre-compiled pipeline
   states for all transformer kernels.
3. Dispatch helpers — standalone functions (no `self`, Rule 20)
   that bind buffers and dispatch each kernel. Each takes raw
   `objc.Object` buffer handles and dimension structs matching
   the Metal shader structs.
4. Eight Metal kernels in `transformer.metal`:

| Kernel                 | What it does                             | Test strategy                                    |
| ---------------------- | ---------------------------------------- | ------------------------------------------------ |
| `rms_norm`             | x / sqrt(mean(x²) + eps) \* scale        | 2 tokens × 8 hidden, CPU ref, < 1e-3 tolerance   |
| `silu`                 | x \* sigmoid(x), elementwise             | 8 values spanning [-3, 4], CPU ref, < 1e-5       |
| `silu_elementwise_mul` | fused silu(gate) ⊙ up for SwiGLU         | 8 gate/up pairs, CPU ref, < 1e-5                 |
| `rope`                 | Rotary position embeddings (in-place)    | 2 heads × 8 dim, pos=3, CPU sin/cos ref, < 1e-4  |
| `kv_cache_update`      | Append f32 K/V to f16 cache at position  | 2 KV heads × 4 dim, written values + zeros check |
| `gqa_attention`        | Grouped query attention with causal mask | 4 QH, 2 KVH, 8 dim, 4 tokens, CPU ref, < 1e-3    |
| `embedding_lookup`     | Gather + dequantize from 1-bit table     | 4 vocab × 16 hidden, 0xAA pattern, CPU ref       |
| `residual_add`         | In-place a[i] += b[i]                    | 8 elements, exact match < 1e-6                   |

Each kernel is a standalone function with primitive arguments
(no `self`), tested independently with a Metal command buffer
that writes results to a shared buffer and compares on CPU.

**Verification results.**

| Test                                         | Result |
| -------------------------------------------- | ------ |
| `TransformerConfig comptime validation`      | PASS   |
| `rms_norm matches CPU reference`             | PASS   |
| `silu matches CPU reference`                 | PASS   |
| `silu_elementwise_mul matches CPU reference` | PASS   |
| `rope matches CPU reference`                 | PASS   |
| `kv_cache_update writes correct values`      | PASS   |
| `gqa_attention matches CPU reference`        | PASS   |
| `embedding_lookup matches CPU dequant`       | PASS   |
| `residual_add matches CPU reference`         | PASS   |

All 29/29 project tests pass (20 existing + 9 new).

**Implementation note: separate shader library.** The transformer
kernels live in `transformer.metal`, compiled as a separate
`MTLLibrary` from the training kernels in `compute.metal`. This
avoids polluting the training pipeline compilation with transformer
symbols and keeps compilation times independent. The
`TransformerPipelines` struct holds the library handle and all
pipeline states.

**Implementation note: `silu_elementwise_mul` fused kernel.** The
plan called for separate `silu` and elementwise multiply steps.
A fused `silu_elementwise_mul` kernel was added to halve memory
traffic in the SwiGLU MLP path (`output[i] = silu(gate[i]) *
up[i]`). The standalone `silu` kernel is also available for
cases where SiLU is needed independently.

**Implementation note: `residual_add` kernel.** Not in the
original kernel table but essential for the transformer block —
adds the sublayer output back to the residual stream in-place.
Added as a simple elementwise kernel.

**Implementation note: GQA attention uses device memory for
scores.** The attention scratch buffer (scores) is stored in
device memory rather than threadgroup memory. This removes
the 32 KB threadgroup memory limit on sequence length,
supporting arbitrarily long contexts. A
`threadgroup_barrier(mem_flags::mem_device)` ensures visibility
within the threadgroup.

**Implementation note: 4B model relaxes hidden_size == query_dim.**
The Bonsai 4B model has `hidden_size` (2560) ≠ `query_dim`
(32 × 128 = 4096). The comptime assertion was relaxed —
`hidden_size == query_dim` is not required. The Q projection
handles the size change.

**Deliverables.**

| File                               | Contents                                               |
| ---------------------------------- | ------------------------------------------------------ |
| `nn/src/shaders/transformer.metal` | 8 kernels: rms_norm, silu, silu_elementwise_mul, rope, |
|                                    | kv_cache_update, gqa_attention, embedding_lookup,      |
|                                    | residual_add                                           |
| `nn/src/transformer.zig`           | `TransformerConfig`, `TransformerPipelines`, 3 model   |
|                                    | configs, dispatch helpers, CPU references, 9 tests     |
| `nn/src/root.zig`                  | Re-exports: transformer module, configs, pipelines     |

### Step 1.5 — single-block integration test ✅

**Goal.** Verify that the transformer kernels compose correctly
through a single decoder block, before wiring 24+ blocks together.

The jump from "6 kernels pass individual tests" to "24 blocks
produce correct text" is where buffer layout bugs, residual-add
wiring mistakes, and f16/f32 promotion mismatches hide. A
single-block test catches these cheaply.

**What was built.**

1. `TinyBlock` test config: 2 query heads, 2 KV heads, 128 hidden
   dim, 256 intermediate dim, 1 layer, 8-token context. Dimensions
   chosen so every `qmv` dispatch has K ≥ 128 (group_size) and
   K % 32 == 0 (simd lane requirement).
2. `forwardBlock` — GPU encoder function that encodes one full
   decoder block into a compute command encoder:
   RMSNorm → Q/K/V projection (qmv) → RoPE → KV cache append →
   GQA attention → O projection (qmv) → residual add →
   RMSNorm → gate/up projection (qmv) → SiLU⊙mul →
   down projection (qmv) → residual add.
3. `cpuForwardBlock` — sequential CPU reference for the same block,
   split into `cpuAttentionProjections`, `cpuAttentionGather`, and
   `cpuForwardBlockMLP` helpers (Rule 5: 70-line limit).
4. `dispatchQMV` — standalone dispatch helper for the 1-bit
   matrix-vector multiply kernel (`qmv` from `compute.metal`),
   following the same pattern as the existing transformer dispatch
   helpers.
5. `cpuQMV` — CPU reference for 1-bit matrix-vector multiply,
   using the same dequantization logic as `cpuEmbeddingDequant`.
6. Integration test that pre-fills the KV cache with 3 positions,
   runs the block for position 3 on GPU, and compares against the
   CPU reference.

This test exercises every buffer hand-off between kernels:

- RMSNorm output feeds into Q/K/V projection input.
- Q and K pass through RoPE before the dot product.
- KV cache stores f16 values; attention loads and promotes to f32.
- Attention output passes through O projection, then adds back to
  the pre-attention residual (not the post-RMSNorm intermediate).
- The MLP residual adds to the post-attention output, not the
  post-RMSNorm intermediate.
- Memory barriers between dependent dispatches ensure write
  visibility within the same compute encoder.

**Verification results.**

| Test                                         | Result |
| -------------------------------------------- | ------ |
| `single-block forward matches CPU reference` | PASS   |

Maximum absolute error: 0.0039 (at element 98 where values
reach ~6000). Maximum relative error: < 1e-6. Tolerance is
5e-3 (not 1e-3) because errors accumulate through the full
block: two RMSNorms, seven qmv projections, RoPE, f32→f16→f32
KV cache round-trip, softmax, and two residual adds.

All 30/30 project tests pass (29 existing + 1 new).

**Implementation note: dimensions bumped from plan.** The
original plan called for 64 hidden dim / 128 intermediate dim.
These were increased to 128 / 256 because the `qmv` kernel
indexes scales per-row as `row * ceil(K/group_size)`, which
only matches the flat group layout when K ≥ group_size (128).
With K=64 the scale indexing was misaligned, producing NaN
outputs. The larger dimensions satisfy the constraint while
remaining small enough for stack-allocated CPU reference arrays.

**Implementation note: `ForwardBlockArgs` struct.** All buffer
handles for one decoder block are bundled into a single struct
rather than passed as individual parameters. This prevents
buffer-binding order mistakes at the call site (Rule 19:
explicit options) and makes the `forwardBlock` signature
manageable.

**Implementation note: `bufferBarrier` between dispatches.**
Metal compute encoders can reorder dispatches unless a memory
barrier is inserted. `memoryBarrierWithScope:` with
`MTLBarrierScope.buffers` ensures preceding buffer writes are
visible to subsequent reads within the same encoder. One
barrier is inserted between each dependent pair of dispatches
(e.g. RMSNorm output → qmv input).

**Implementation note: function splitting.** The attention half
was split into `encodeAttentionProjections` (RMSNorm → QKV →
RoPE → KV cache) and `encodeAttentionGather` (GQA → O proj →
residual add). The CPU reference was split similarly:
`cpuAttentionProjections`, `cpuProjectQKV`, `cpuKVCacheWrite`,
and `cpuAttentionGather`. All new functions are under 70 lines.

**Deliverables.**

| File                     | Contents                                |
| ------------------------ | --------------------------------------- |
| `nn/src/transformer.zig` | `TinyBlock` config, `ForwardBlockArgs`, |
|                          | `forwardBlock`, `dispatchQMV`,          |
|                          | `QMVDims`, CPU reference functions,     |
|                          | test helpers, integration test          |

### Step 2 — Bonsai 1.7B inference ✅

**Goal.** Load the Bonsai 1.7B safetensors model, run a prompt
through the full 28-block transformer, and produce correct text
output verified against a reference implementation.

Step 2 is broken into six sub-phases. Each produces a working,
testable artifact. Sub-phases 2a, 2b, and 2e have no mutual
dependencies and can be built in any order (or in parallel).
Sub-phase 2c depends on 2a. Sub-phase 2d depends on 2b and 2c.
Sub-phase 2f depends on 2d and 2e.

```
    2a (safetensors)    2b (QK norms)    2e (tokenizer)
         │                   │                │
         ▼                   │                │
    2c (model loader) ◄──────┘                │
         │                                    │
         ▼                                    │
    2d (multi-block forward)                  │
         │                                    │
         ▼                                    ▼
    2f (generation loop + sampling) ◄─────────┘
```

#### Step 2a — safetensors parser ✅

**Goal.** Parse the MLX safetensors binary format and mmap tensor
data directly into Metal-visible memory (zero-copy from disk to
GPU).

**What was built.**

1. `SafetensorsFile` — top-level parser struct with `init(path)`
   (opens file, mmaps it, parses JSON header) and
   `initFromBytes(bytes)` (for testing without a real file).
   `deinit()` unmaps the file. `getTensor(name)` performs a
   linear scan lookup by tensor name.
2. `TensorDescriptor` — holds name (slice into mmap'd JSON
   header), `Dtype`, shape (`[MAX_DIMS]u32` bounded array with
   `rank`), and data (byte slice into mmap'd data region).
   Helpers: `elementCount()`, `sizeBytes()`.
3. `Dtype` — enum with `.u8`, `.f16`, `.f32`, `.bf16` variants,
   `sizeBytes()` and `fromString()` methods.
4. Fixed-capacity array of `MAX_TENSORS = 1024` descriptors — no
   dynamic allocation after init.
5. JSON header parsed with `std.json.parseFromSlice` using a
   256 KiB stack-backed `FixedBufferAllocator` — freed
   automatically when the parsing helper returns.
6. `std.posix.mmap` / `munmap` for zero-copy file access.
7. Parsing split into small helpers: `readHeaderLength`,
   `parseJsonHeader`, `parseTensorEntry`, `parseDtype`,
   `parseShape`, `parseDataSlice`, `extractOffsetPair`,
   `extractOffset` — all under 70 lines.

**Verification results.**

| Test                                   | Result |
| -------------------------------------- | ------ |
| `parse safetensors header from bytes`  | PASS   |
| `getTensor returns correct descriptor` | PASS   |
| `data offsets are validated`           | PASS   |
| `metadata key is skipped`              | PASS   |
| `Dtype.fromString round-trips`         | PASS   |
| `Dtype.sizeBytes`                      | PASS   |

All 6 safetensors tests pass. Tests use in-memory byte arrays
(via `initFromBytes`) with a `buildTestFile` comptime helper
that constructs minimal valid safetensors buffers — no
dependency on actual model files.

**Implementation note: `initFromBytes` for testing.** A second
init path accepts a raw byte slice instead of a file path.
It stores a zero-length mmap sentinel so `deinit` knows not
to call `munmap`. Tensor data slices point into the caller's
byte array, which must outlive the `SafetensorsFile`.

**Implementation note: `__metadata__` key is skipped.** The
JSON header may contain a `__metadata__` key with arbitrary
string values (not a tensor descriptor). The parser detects
this key by name and skips it, avoiding a parse error on the
missing `dtype`/`shape`/`data_offsets` fields.

**Deliverables.**

| File                     | Contents                                     |
| ------------------------ | -------------------------------------------- |
| `nn/src/safetensors.zig` | Parser, mmap, tensor descriptor struct,      |
|                          | `buildTestFile` helper, 6 tests              |
| `nn/src/root.zig`        | Re-exports: `safetensors`, `SafetensorsFile` |

#### Step 2b — QK norms in forward block ✅

**Goal.** Wire QK norms into the existing `forwardBlock` so that
it matches the real Qwen3 architecture.

**What exists.** The current `encodeAttentionProjections` goes
RMSNorm → Q/K/V proj → RoPE. The real Bonsai model applies
per-head RMSNorm to Q and K _after_ projection and _before_
RoPE: RMSNorm → Q/K/V proj → **Q norm → K norm** → RoPE.

**What was built.**

1. Added `q_norm_scale` and `k_norm_scale` fields (f16 buffers,
   `head_dim` elements each) to `ForwardBlockArgs`.
2. Inserted two `dispatchRMSNorm` calls in
   `encodeAttentionProjections` between the QKV projections and
   RoPE. Each reuses the existing `rms_norm` kernel with
   `hidden_size = head_dim` and `num_tokens = num_query_heads`
   (or `num_kv_heads` for K), treating each head as a separate
   "token" row. The same `head_dim`-sized scale vector is
   broadcast across all heads. In-place: input == output buffer.
3. Extracted `encodeRoPEAndKVCache` helper from
   `encodeAttentionProjections` to keep both functions under
   70 lines after the QK norm additions.
4. Added `q_norm` and `k_norm` fields to `BlockWeightSlices`.
5. Updated `cpuAttentionProjections` with matching CPU QK norm
   calls (`cpuRMSNorm` with `hidden_size = head_dim`,
   `num_tokens = num_heads`), and extracted
   `cpuRoPEAndKVCache` helper similarly.
6. Updated single-block integration test with QK norm scale
   buffers (deterministic values) wired into both
   `ForwardBlockArgs` and `BlockWeightSlices`.
7. Added standalone `rms_norm_heads` test (4 heads × 8 dim).

**Verification results.**

| Test                                   | Result |
| -------------------------------------- | ------ |
| `rms_norm_heads matches CPU reference` | PASS   |
| `single-block forward matches CPU ref` | PASS   |

All 37/37 project tests pass (30 existing + 1 rms_norm_heads +
6 safetensors).

**Implementation note: no new Metal kernel needed.** The
existing `rms_norm` kernel already handles per-head
normalisation — its comment says "set hidden_size = head_dim
and num_tokens = num_heads." Each threadgroup normalises one
head independently. This avoids adding a separate
`rms_norm_heads` kernel and keeps the shader count unchanged.

**Implementation note: in-place dispatch is safe.** The
`rms_norm` kernel reads all `x[i]` values in the first loop
(partial sum of squares), synchronises via
`threadgroup_barrier`, then writes `out[i]` in the second
loop. When `input == output` (same buffer), the barrier
guarantees all reads complete before any writes begin.

**Implementation note: `encodeRoPEAndKVCache` extraction.**
Adding 2 QK norm dispatches + 1 barrier pushed
`encodeAttentionProjections` over 70 lines. The RoPE and
KV cache update portion was extracted into a new helper,
keeping both functions under the limit.

**Deliverables.**

| File                     | Contents                       |
| ------------------------ | ------------------------------ |
| `nn/src/transformer.zig` | Updated `ForwardBlockArgs`,    |
|                          | `encodeAttentionProjections`,  |
|                          | `encodeRoPEAndKVCache` (new),  |
|                          | `BlockWeightSlices`, CPU refs, |
|                          | updated + new tests            |

#### Step 2c — model loader (weights → GPU buffers) ✅

**Goal.** Load a Bonsai 1.7B safetensors model from disk into
pre-allocated Metal buffers, ready for inference.

**Depends on:** Step 2a (safetensors parser).

**What to build.**

1. `model.zig` — a `Model` struct parameterised by
   `TransformerConfig` that owns all GPU resources:
   - Per-layer `PackedBuffer` arrays for the 7 projections
     (Q/K/V/O/gate/up/down) — or a single large mmap'd buffer
     with comptime-known offsets per layer.
   - Per-layer f16 norm scale buffers (attn_norm, ffn_norm,
     q_norm, k_norm).
   - Embedding `PackedBuffer` (and LM head, if untied).
   - Final RMSNorm scale buffer.
   - KV cache: one f16 buffer per layer, pre-allocated for
     `max_context_length × kv_dim × 2` (K and V interleaved
     or separate — match `kv_cache_update` kernel layout).
   - Activation scratch buffers (pre-allocated for the larger
     of decode and prefill regimes).
2. Tensor name mapping — translate safetensors tensor names
   (e.g., `model.layers.5.self_attn.q_proj.weight`) to the
   correct layer index and buffer slot using string parsing on
   the layer index.
3. Scale conversion — MLX affine encoding stores
   `(mlx_scale, mlx_bias)` per group. Convert at load time:
   `symmetric_scale = mlx_scale / 2`. One f16 division per
   group, performed on the CPU during loading.
4. Zero-copy where possible — for packed weight bytes, create
   `MTLBuffer` with `newBufferWithBytesNoCopy` pointing into
   the mmap'd safetensors data region. For scales that need
   conversion, allocate a separate buffer and copy+convert.
5. `deinit` releases all Metal buffers and unmaps the file.

**Verification.**

| Test                        | Strategy                                                |
| --------------------------- | ------------------------------------------------------- |
| All buffers allocated       | Assert every `PackedBuffer` and norm buffer is non-null |
|                             | after loading 1.7B model.                               |
| Buffer sizes match config   | Assert `q_proj` byte length ==                          |
|                             | `Bonsai1_7B.q_proj_bytes` for every layer.              |
| Weight data is non-trivial  | Spot-check: sum of first 1024 packed bytes > 0          |
|                             | (not all zeros).                                        |
| Scale conversion is correct | Load one group's MLX scale+bias in Python, compute      |
|                             | symmetric scale, compare to loaded f16 value.           |
| KV cache is zeroed          | Assert all KV cache bytes are zero after init.          |

**Deliverables.**

| File               | Contents                                |
| ------------------ | --------------------------------------- |
| `nn/src/model.zig` | `Model` struct, loader, tensor mapping, |
|                    | scale conversion, KV cache allocation   |

#### Step 2d — multi-block forward pass ✅

**Goal.** Run a single token through the full 28-block
transformer and produce a logits vector over the vocabulary.

**Depends on:** Steps 2b (QK norms) and 2c (model loader).

**What to build.**

1. `forwardDecode` — encode the full single-token decode path
   into one Metal command buffer:
   - Embedding lookup (1-bit, via `embedding_lookup` kernel).
   - 28 × `forwardBlock` (with QK norms from Step 2b).
   - Final RMSNorm.
   - LM head projection. With `tie_word_embeddings=true`, the
     LM head reuses the embedding `PackedBuffer`. The embedding
     is stored as `[vocab_size × hidden_size]`, which is exactly
     the `[N × K]` layout `qmv` expects — so `qmv` with
     `M=vocab_size, K=hidden_size` produces the logits directly.
   - Output: `[vocab_size]` f32 logits buffer.
2. Construct `ForwardBlockArgs` for each layer from `Model`
   buffers — a helper that indexes into the per-layer buffer
   arrays and fills the args struct.
3. Barrier discipline — one `bufferBarrier` between each block
   (the residual output of block _i_ feeds the input of block
   _i+1_). Inside each block, barriers are already handled by
   `forwardBlock`.

**Verification.**

| Test                          | Strategy                                            |
| ----------------------------- | --------------------------------------------------- |
| Logits are finite             | Load 1.7B, embed token 0, run 28 blocks + LM head.  |
|                               | Assert no NaN/Inf in the logits buffer.             |
| Logits are non-degenerate     | Assert `max(logits) - min(logits) > 1.0` — the      |
|                               | model is not outputting a flat distribution.        |
| Argmax is plausible           | For a known prompt token, check that the argmax     |
|                               | token is a real vocabulary entry (not padding/UNK). |
| Dispatch count matches budget | Assert the command buffer encodes exactly           |
|                               | `1 + 28×18 + 2 = 507` dispatches (embedding +       |
|                               | 28 blocks × 18 kernels + final norm + LM head).     |

**Deliverables.**

| File                     | Contents                               |
| ------------------------ | -------------------------------------- |
| `nn/src/transformer.zig` | `forwardDecode`, per-layer args helper |

#### Step 2e — BPE tokenizer ✅

**Goal.** Encode text to token IDs and decode token IDs back to
text, using the HuggingFace `tokenizer.json` from the MLX
repository.

**No dependencies** on other Step 2 sub-phases. Can be built
and tested in isolation.

**What to build.**

1. `tokenizer.zig` — parse the HuggingFace `tokenizer.json`
   file using `std.json`:
   - Vocabulary: map of token string → token ID (151669 entries).
   - Reverse vocabulary: array of token ID → token string (for
     decode).
   - BPE merge rules: ordered list of `(token_a, token_b)` pairs,
     applied greedily from highest priority to lowest.
   - Special tokens: `<|im_start|>`, `<|im_end|>`,
     `<|endoftext|>`, etc.
2. `encode(text) → []u32` — apply BPE merges:
   - Split input into UTF-8 characters (respecting byte-level
     BPE conventions).
   - Iteratively merge the highest-priority adjacent pair until
     no more merges apply.
   - Handle special tokens: scan for special token strings
     before BPE splitting.
3. `decode(token_ids) → []u8` — concatenate token strings.
   Handle byte-level tokens (Qwen3 uses byte-fallback encoding
   for rare characters).
4. Chat template helper — hardcode Qwen3's format:
   `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`

**Verification.**

| Test                | Strategy                                                 |
| ------------------- | -------------------------------------------------------- |
| Vocabulary size     | Assert 151669 tokens loaded.                             |
| Round-trip identity | `decode(encode("Hello, world!"))` == `"Hello, world!"`   |
| Known token IDs     | Compare `encode("Hello")` against Python `AutoTokenizer` |
|                     | output for exact ID match.                               |
| Special tokens      | Assert `encode("<\|im_start\|>")` returns the single     |
|                     | special token ID, not BPE-split fragments.               |
| Chat template       | Assert `apply_chat_template("Hi")` produces the expected |
|                     | `<\|im_start\|>user\nHi<\|im_end\|>\n...` IDs.           |
| UTF-8 edge cases    | Encode/decode strings with emoji, CJK, and mixed scripts |
|                     | — no panics, no data loss.                               |

Estimated size: ~800–1200 lines. BPE merge ordering, UTF-8
boundary handling, special token detection, and the Qwen3 chat
template are each straightforward in isolation but interact in
ways that add up. Budget for the upper end.

**Deliverables.**

| File                   | Contents                                  |
| ---------------------- | ----------------------------------------- |
| `nn/src/tokenizer.zig` | BPE encode/decode, vocab loader, chat tpl |

#### Step 2f — autoregressive generation + sampling ✅

**Goal.** Generate text from a prompt, end to end. Verify greedy
output matches a reference implementation.

**Depends on:** Steps 2d (multi-block forward) and 2e (tokenizer).

**What to build.**

1. `generate` function — the autoregressive loop:
   - Tokenize the prompt.
   - **Prefill**: process all prompt tokens at once using `qmm`
     (matrix-matrix) for each 1-bit projection. This requires
     a `forwardPrefill` path that mirrors `forwardDecode` but
     dispatches `qmm` instead of `qmv`, and processes `[M × K]`
     activations where `M = prompt_length`. RMSNorm, RoPE, and
     KV cache update kernels need to handle M > 1 (or chunk
     the prompt into `max_prefill_length`-sized pieces and run
     `forwardDecode` per chunk — simpler, slower, acceptable
     for Step 2).
   - **Decode**: generate one token at a time. Dispatch `qmv`
     (matrix-vector) for each projection. Read logits, sample,
     append token, advance position.
   - Stream decoded tokens to stdout as they are generated.
2. Sampling strategies:
   - **Greedy** (temperature=0): argmax of logits. Used for
     verification.
   - **Temperature**: divide logits by T before softmax.
   - **Top-k**: zero out all logits except the k largest.
   - **Top-p** (nucleus): sort logits, zero out tokens whose
     cumulative probability exceeds p.
   - A `SamplingParams` struct with all three knobs.
3. `bonsai.zig` CLI entry point:
   - Parse command-line args: model path, prompt, max tokens,
     temperature, top-k, top-p.
   - Load model (Step 2c).
   - Load tokenizer (Step 2e).
   - Run generation loop.
   - Print timing: prompt tok/s, generation tok/s.

For the initial implementation, **chunked prefill** (reusing
the decode path, one token at a time) is acceptable. True
batched prefill with `qmm` is a performance optimisation that
can be added after correctness is verified.

**Verification.**

| Test                | Strategy                                                    |
| ------------------- | ----------------------------------------------------------- |
| Greedy decode match | Run `"The capital of France is"` with temperature=0 through |
|                     | both nnzap and MLX (or llama.cpp). First 20 output tokens   |
|                     | must be identical.                                          |
| EOS handling        | Assert generation stops when `<\|im_end\|>` or              |
|                     | `<\|endoftext\|>` is emitted.                               |
| Sampling diversity  | With temperature=0.7, top_p=0.9, generate 100 tokens twice  |
|                     | — outputs should differ (non-deterministic).                |
| Timing              | Print tok/s. Ballpark target: > 30 tok/s on M-series for    |
|                     | 1.7B decode (loose — correctness first).                    |

**Deliverables.**

| File                     | Contents                                 |
| ------------------------ | ---------------------------------------- |
| `nn/src/transformer.zig` | `generate`, `forwardPrefill` (or chunked |
|                          | decode), sampling functions              |
| `nn/examples/bonsai.zig` | CLI: load model, run prompt, print text  |
| `nn/build.zig`           | `run-bonsai` build step                  |

#### Step 2g — weight-loading correctness fixes ✅

**Symptom.** The full pipeline ran end to end at ~33 tok/s, but
output was degenerate (repeated `!` tokens regardless of prompt).

**Root cause.** Two dtype mismatches between what the code assumed
and what the MLX Bonsai safetensors actually contain:

| Bug                 | Expected | Actual  | Effect                                                                                                                                                                         |
| ------------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Norm weights        | F32      | **F16** | `loadNormScale` reinterpreted pairs of F16 bytes as single F32 floats → garbage RMSNorm scales (~1e-11 instead of ~0.08). Every activation was corrupted from the first layer. |
| Packed weight dtype | U8       | **U32** | Assertion mismatch only — raw bytes are identical on little-endian, so the memcpy was correct. But `std.debug.assert` would fire in Debug/ReleaseSafe builds.                  |

Both assertions were stripped in `ReleaseFast`, so the pipeline
ran without crashing — but norm corruption silently destroyed
all signal through the network.

**Fixes** (`model.zig`):

1. `loadNormScale` — detect dtype at runtime. When F16, direct
   `@memcpy` (no conversion needed). When F32, narrow to F16
   as before. Assert `dtype == .f16 or dtype == .f32`.
2. `loadPackedProjection` — widen the assertion to accept U32:
   `weight_desc.dtype == .u8 or weight_desc.dtype == .u32`.

**Result.** Greedy decode of `"What is the capital of France?"`
now produces `"The capital of France is Paris."` — correct and
coherent. **36.8 tok/s decode** on M-series.

### Step 3 — Bonsai 8B and performance

**Goal.** Scale to the 8B model and tune Metal kernels for peak
throughput.

**What to build.**

1. Change the `TransformerConfig` constants — all code from Step 2
   transfers directly.
2. Profile token generation throughput (tok/s) on M-series hardware.
3. Tune `qmv` kernel: try `qmv_fast` (simdgroup reductions for
   aligned cases) and `qmv_quad` (quadgroup reductions for small K).
4. Tune `qmm` kernel for prefill: tiled GEMM with 1-bit block loader
   that dequantizes into shared memory.
5. Benchmark against llama.cpp Metal and MLX on the same hardware
   with the same model and prompt.

**Deliverables.**

| File                               | Contents              |
| ---------------------------------- | --------------------- |
| `nn/src/shaders/transformer.metal` | Tuned kernel variants |
| `benchmarks/bonsai_*.json`         | Throughput benchmarks |

## Comptime transformer config

The `TransformerConfig` follows the same pattern as `NetworkLayout`:
all buffer sizes and offsets are resolved at compile time.

```
pub fn TransformerConfig(comptime cfg: TransformerDesc) type {
    return struct {

        // -- Architecture constants --

        pub const vocab_size: u32 = cfg.vocab_size;
        pub const hidden_size: u32 = cfg.hidden_size;
        pub const intermediate_size: u32 = cfg.intermediate_size;
        pub const num_layers: u32 = cfg.num_layers;
        pub const num_query_heads: u32 = cfg.num_query_heads;
        pub const num_kv_heads: u32 = cfg.num_kv_heads;
        pub const head_dim: u32 = cfg.head_dim;
        pub const max_context_length: u32 = cfg.max_context_length;
        pub const max_prefill_length: u32 = cfg.max_prefill_length;
        pub const group_size: u32 = 128;

        // -- Derived constants --

        pub const query_dim: u32 = num_query_heads * head_dim;
        pub const kv_dim: u32 = num_kv_heads * head_dim;
        pub const heads_per_kv_group: u32 =
            num_query_heads / num_kv_heads;

        // -- Comptime assertions --

        comptime {
            // GQA requires query heads to be a multiple of KV heads.
            std.debug.assert(num_query_heads % num_kv_heads == 0);
            // Hidden size must equal total query dimension.
            std.debug.assert(hidden_size == query_dim);
            // Group size must divide head dim evenly.
            std.debug.assert(head_dim % group_size == 0
                or group_size % head_dim == 0);
        }

        // -- 1-bit packed sizes (bytes) --

        /// Packed bytes for a weight matrix [rows × cols].
        fn packedBytes(
            comptime rows: u32,
            comptime cols: u32,
        ) u32 {
            const total_weights = rows * cols;
            const packed = divCeil(total_weights, 8);
            const num_groups = divCeil(total_weights, group_size);
            const scale_bytes = num_groups * 2; // f16 scales
            return packed + scale_bytes;
        }

        // Per-layer packed weight sizes (bytes).
        pub const q_proj_bytes: u32 =
            packedBytes(hidden_size, query_dim);
        pub const k_proj_bytes: u32 =
            packedBytes(hidden_size, kv_dim);
        pub const v_proj_bytes: u32 =
            packedBytes(hidden_size, kv_dim);
        pub const o_proj_bytes: u32 =
            packedBytes(query_dim, hidden_size);
        pub const gate_proj_bytes: u32 =
            packedBytes(hidden_size, intermediate_size);
        pub const up_proj_bytes: u32 =
            packedBytes(hidden_size, intermediate_size);
        pub const down_proj_bytes: u32 =
            packedBytes(intermediate_size, hidden_size);

        pub const attention_weight_bytes: u32 =
            q_proj_bytes + k_proj_bytes
            + v_proj_bytes + o_proj_bytes;
        pub const mlp_weight_bytes: u32 =
            gate_proj_bytes + up_proj_bytes + down_proj_bytes;
        pub const layer_weight_bytes: u32 =
            attention_weight_bytes + mlp_weight_bytes;
        pub const total_layer_weight_bytes: u32 =
            layer_weight_bytes * num_layers;

        // Embedding + LM head.
        pub const embedding_bytes: u32 =
            packedBytes(vocab_size, hidden_size);
        pub const lm_head_bytes: u32 =
            packedBytes(hidden_size, vocab_size);

        // RMSNorm scales: f16, two per layer + one final.
        pub const norm_scales_per_layer: u32 = hidden_size * 2;
        pub const total_norm_scale_count: u32 =
            norm_scales_per_layer * num_layers + hidden_size;

        // Total model weight bytes. u64 for future-proofing —
        // the 8B model is ~1.2 × 10⁹ bytes, which fits in u32
        // but leaves little headroom for larger models.
        pub const total_weight_bytes: u64 =
            @as(u64, embedding_bytes)
            + total_layer_weight_bytes
            + lm_head_bytes
            + total_norm_scale_count * 2; // f16

        // -- KV cache sizes (f16 elements) --
        // f16 KV cache halves memory vs f32 with negligible
        // quality loss.  Use u64 because 65k ctx × 1024 kv_dim
        // × 2 × 36 layers exceeds u32 range.

        pub const kv_cache_elements_per_layer: u64 =
            @as(u64, max_context_length) * kv_dim * 2; // K and V
        pub const total_kv_cache_elements: u64 =
            kv_cache_elements_per_layer * num_layers;
        pub const total_kv_cache_bytes: u64 =
            total_kv_cache_elements * 2; // f16 = 2 bytes

        // -- Activation scratch --
        // Decode (M=1) and prefill (M=max_prefill_length) have
        // different scratch requirements.  Pre-allocate for the
        // larger of the two so no allocation happens after init.

        /// Single-token decode scratch (f32 elements).
        pub const decode_activation_elements: u32 = blk: {
            var max: u32 = hidden_size;
            if (query_dim > max) max = query_dim;
            if (intermediate_size > max) max = intermediate_size;
            // Attention scores: one row per query head.
            const attn_scores =
                num_query_heads * max_context_length;
            if (attn_scores > max) max = attn_scores;
            break :blk max;
        };

        /// Prefill scratch (f32 elements).  During prefill,
        /// activations are [seq_len × hidden_size], not
        /// [1 × hidden_size].  Chunked prefill reuses the same
        /// scratch buffer in chunks of max_prefill_length.
        pub const prefill_activation_elements: u64 = blk: {
            const p: u64 = max_prefill_length;
            var max: u64 = p * hidden_size;
            const qp = p * query_dim;
            if (qp > max) max = qp;
            const ip = p * intermediate_size;
            if (ip > max) max = ip;
            // Attention scores: [num_query_heads × seq × seq].
            // For chunked prefill, seq = max_prefill_length.
            const attn = @as(u64, num_query_heads)
                * p * max_context_length;
            if (attn > max) max = attn;
            break :blk max;
        };

        /// Total scratch to pre-allocate (the larger regime).
        pub const max_activation_elements: u64 = blk: {
            const d: u64 = decode_activation_elements;
            const p: u64 = prefill_activation_elements;
            break :blk if (p > d) p else d;
        };
    };
}
```

### Model configs

```
const Bonsai1_7B = TransformerConfig(.{
    .vocab_size = 151669,
    .hidden_size = 2048,
    .intermediate_size = 6144,
    .num_layers = 28,
    .num_query_heads = 16,
    .num_kv_heads = 8,
    .head_dim = 128,
    .max_context_length = 32768,
    .max_prefill_length = 512,
    .rope_theta = 1000000.0,
    .tie_word_embeddings = true,
});

const Bonsai4B = TransformerConfig(.{
    .vocab_size = 151669,
    .hidden_size = 2560,
    .intermediate_size = 9728,
    .num_layers = 36,
    .num_query_heads = 32,
    .num_kv_heads = 8,
    .head_dim = 128,
    .max_context_length = 32768,
    .max_prefill_length = 512,
    .rope_theta = 5000000.0,
    .tie_word_embeddings = true,
});

const Bonsai8B = TransformerConfig(.{
    .vocab_size = 151669,
    .hidden_size = 4096,
    .intermediate_size = 12288,
    .num_layers = 36,
    .num_query_heads = 32,
    .num_kv_heads = 8,
    .head_dim = 128,
    .max_context_length = 65536,
    .max_prefill_length = 512,
    .rope_theta = 1000000.0,
    .tie_word_embeddings = false,
});
```

## PackedBuffer

A new buffer type alongside `Buffer` and `HalfBuffer`, storing 1-bit
packed weights with per-group f16 scales.

```
pub const PackedBuffer = struct {
    obj: objc.Object,
    packed_count: u32,  // number of 1-bit weight elements
    group_size: u32,    // weights per scale group (128)
    byte_len: u32,      // total allocation in bytes

    /// Number of packed bytes (ceil(packed_count / 8)).
    pub fn packedBytes(self: PackedBuffer) u32 {
        return divCeil(self.packed_count, 8);
    }

    /// Number of scale groups (ceil(packed_count / group_size)).
    pub fn numGroups(self: PackedBuffer) u32 {
        return divCeil(self.packed_count, self.group_size);
    }

    /// Byte offset where the f16 scale array begins.
    pub fn scaleOffset(self: PackedBuffer) u32 {
        return self.packedBytes();
    }

    // ... init, deinit, metalBuffer follow Buffer's pattern.
};
```

### Metal binding layout

When binding a `PackedBuffer` to a compute encoder, `setPackedBuffer`
binds the same MTLBuffer twice at consecutive indices:

```
setPackedBuffer(encoder, packed_buf, 0)
  → buffer(0) = packed bits  (uint8_t*)   offset 0
  → buffer(1) = scales       (half*)      offset packedBytes()
```

The bits and scales live in the same `MTLBuffer`. The encoder binds
the same buffer twice with different offsets to avoid an extra
allocation.

For `qmv`, the full binding layout is:

```
buffer(0) = packed bits     (uint8_t*)
buffer(1) = scales          (half*)    — same MTLBuffer, offset by packedBytes()
buffer(2) = input vector    (float*)
buffer(3) = output vector   (float*)
buffer(4) = dimensions      (QMVDims via setBytes)
```

For `qmm`, the full binding layout is:

```
buffer(0) = A activations   (float*)   — [M × K]
buffer(1) = packed bits     (uint8_t*)
buffer(2) = scales          (half*)    — same MTLBuffer, offset by packedBytes()
buffer(3) = output          (float*)   — [M × N]
buffer(4) = dimensions      (QMMDims via setBytes)
```

## Core Metal kernels

### 1-bit matrix-vector multiply (`qmv`)

Used during token generation (one token at a time, M=1). This is the
hot path — the kernel that determines tok/s.

The key insight: **1-bit matmul replaces multiplication with
conditional accumulation.** If the bit is 1, add the activation
value. If 0, add nothing. Then apply the group scale.

#### Scalar reference (what the kernel computes)

```
// One output element: dot product of packed 1-bit row with f32 input.
float result = 0.0;
for (uint g = 0; g < groups_per_row; g++) {
    float scale = scales[row * groups_per_row + g];
    float set_accum = 0.0;   // Sum of input[col] where bit=1.
    float group_sum = 0.0;   // Sum of input[col] for all cols.
    for (uint i = 0; i < group_size / 8; i++) {
        uint8_t bits = packed[offset + i];
        for (uint b = 0; b < 8; b++) {
            uint col = g * group_size + i * 8 + b;
            group_sum += input[col];
            set_accum += select(0.0, input[col], bool(bits & (1 << b)));
        }
    }
    // Q1_0_g128: bit=1 → +scale, bit=0 → −scale.
    // = scale * set_accum + (-scale) * (group_sum - set_accum)
    // = scale * (2 * set_accum - group_sum)
    result += scale * (2.0 * set_accum - group_sum);
}
output[row] = result;
```

#### Tiled kernel design

The scalar loop above runs in O(K) per output element. The GPU
kernel parallelises across two axes:

```
Output dimension M (rows):
  Each row is independent.  Map one simdgroup (32 threads) per row.
  Two simdgroups per threadgroup → 2 rows per threadgroup.
  Grid: ceil(M / 2) threadgroups.

Input dimension K (columns):
  Each simdgroup partitions K across its 32 threads.
  Thread t processes columns [t*chunk, (t+1)*chunk) where
  chunk = K / 32.  K must be a multiple of 128 (group_size),
  and 128/32 = 4, so each thread processes complete groups.

Per-thread accumulation:
  Each thread loads its chunk of the input vector into registers,
  then iterates over the corresponding packed bytes and scales.
  The inner loop processes 8 weights per byte:

    uint8_t wb = packed_bits[byte_idx];
    accum += select(0.0f, x[0], bool(wb & 0x01));
    accum += select(0.0f, x[1], bool(wb & 0x02));
    accum += select(0.0f, x[2], bool(wb & 0x04));
    ...
    accum += select(0.0f, x[7], bool(wb & 0x80));

  No branching — select compiles to a conditional move.

Reduction:
  After the inner loop, each thread holds a partial sum.
  Use simd_sum(accum) to reduce across the 32 threads in the
  simdgroup.  Thread 0 writes the final result to output[row].
  No threadgroup shared memory needed — simd_sum is a hardware
  intrinsic on Apple GPUs.

Scale application:
  The scale is applied per-group during accumulation (not after
  reduction), because each group has a different scale.  Each
  thread tracks its own group boundaries and applies the
  2*scale*set_accum - scale*group_sum identity locally.
```

#### Dispatch variants (future optimisation — Step 3)

| Variant    | When                     | Difference                                  |
| ---------- | ------------------------ | ------------------------------------------- |
| `qmv`      | General case             | Boundary checks, any K                      |
| `qmv_fast` | K % 512 == 0, M % 8 == 0 | No bounds checks, unrolled                  |
| `qmv_quad` | K ≤ 128 (e.g. head_dim)  | 4-thread quadgroup, not 32-thread simdgroup |

Step 0 implements `qmv` only. The fast/quad variants are tuning
work for Step 3 after profiling reveals which projections are
bottlenecked.

### 1-bit matrix-matrix multiply (`qmm`)

Used during prefill (processing the full prompt, M >> 1) and for
MNIST inference at all batch sizes (since `qmv` requires K % 32
== 0). Where `qmv` computes one output element per simdgroup,
`qmm` computes a tile of output elements using shared memory —
same idea as the existing `matmul_tiled` in `compute.metal`, but
with a custom block loader that dequantizes 1-bit tiles on the fly.

**Weight layout.** The existing nnzap matmul stores weights as
`[K × N]` (in_size × out_size). The `qmm` kernel follows this
convention: W is `[K × N]` packed, and the flat bit index for
position (k, n) is `k * N + n`. Scale groups are assigned by
flat position: `group_idx = flat_bit / group_size`.

For the transformer path (Step 1+), GGUF stores each weight
tensor in `[N × K]` order (row per output neuron). The `qmv`
kernel already assumes this layout. A layout flag or a separate
`qmm_transposed` kernel will resolve this when needed.

#### Tiling strategy

```
Output tile:  16 × 16 (same as existing matmul_tiled)
K tiles:      step through K in chunks of 128 (= group_size)
              This aligns tile boundaries with scale group
              boundaries, so each tile load touches exactly one
              scale value per row.

Tile load (A — activations, f32):
  Standard cooperative load into threadgroup memory.
  16 rows × 128 cols = 2048 floats = 8 KB per tile.

Tile load (W — 1-bit packed weights, [K × N]):
  Each thread loads packed bytes, dequantizes into threadgroup
  memory as f32.  The tile is stored transposed in shared memory
  as tile_W[n_local][k_local] so the inner loop can index
  tile_W[local_col][k] for the dot product:

    uint flat_bit = w_k * N + w_n;
    uint byte_idx = flat_bit / 8;
    uint bit_pos  = flat_bit % 8;
    bool bit_set  = (packed_bits[byte_idx] >> bit_pos) & 1;
    uint g_idx    = flat_bit / group_size;
    float scale   = float(scales[g_idx]);
    tile_W[tile_n][tile_k] = select(-scale, +scale, bit_set);

  128 cols × 16 rows = 2048 floats = 8 KB per tile.
  Packed source: 2048 bits = 256 bytes per tile (32× compression).

Compute:
  After both tiles are in shared memory, the multiply is a
  standard 16×16×128 tile matmul — identical to the existing
  kernel's inner loop.

threadgroup_barrier(mem_flags::mem_threadgroup);
```

Dequantization happens into threadgroup memory as f32, not f16.
The overhead is 8 KB × 2 tiles = 16 KB of threadgroup memory per
threadgroup, well within Metal's 32 KB limit. f32 avoids
precision loss during the accumulation across K tiles.

#### Why not dequantize to f16 in shared memory?

f16 tiles would halve shared memory usage (8 KB total) and
double the effective tile size. But the 1-bit → f16 → f32
promotion chain introduces rounding at the dequantization step
that accumulates across K. Since the weights are already
maximally quantized (1 bit), preserving f32 precision in the
accumulation is worth the extra shared memory.

### RMSNorm

```
// Per-token: normalize, then scale.
float sum_sq = 0.0;
for (uint i = 0; i < hidden_size; i++) {
    sum_sq += input[i] * input[i];
}
float rms = rsqrt(sum_sq / float(hidden_size) + eps);
for (uint i = 0; i < hidden_size; i++) {
    output[i] = input[i] * rms * float(scale[i]); // scale is f16
}
```

### RoPE

Rotary position embeddings applied to Q and K after projection:

```
// For each pair (x[2i], x[2i+1]) at position pos:
float cos_val = cos(pos * theta_i);
float sin_val = sin(pos * theta_i);
output[2i]     = x[2i] * cos_val − x[2i+1] * sin_val;
output[2i + 1] = x[2i] * sin_val + x[2i+1] * cos_val;

// where theta_i = 1 / (rope_theta ^ (2i / head_dim))
```

### SiLU

```
output[i] = input[i] * (1.0 / (1.0 + exp(-input[i])));
// Equivalent to: input[i] * sigmoid(input[i])
```

### GQA attention

Grouped query attention with causal masking. Each KV head is shared
by `heads_per_kv_group` query heads:

```
// For each query head h:
kv_head = h / heads_per_kv_group;
scores[t] = dot(Q[h], K[kv_head][t]) / sqrt(head_dim);
// Apply causal mask: scores[t] = -inf for t > current_pos.
attention_weights = softmax(scores[0..current_pos+1]);
output[h] = sum(attention_weights[t] * V[kv_head][t]);
```

### Precision at the KV cache boundary

The KV cache stores f16 values to halve memory. Activations
flowing through the transformer are f32. The conversion boundary
must be explicit:

```
Projection outputs (f32)
  │
  ▼
KV cache write: f32 → f16 (truncation happens here)
  │
  ▼
KV cache storage (f16)
  │
  ▼
Attention read: f16 → f32 (promotion happens here)
  │
  ▼
Dot product Q·K and weighted sum over V (f32 accumulation)
```

The `kv_cache_update` kernel converts f32 projections to f16 on
write. The `gqa_attention` kernel loads f16 K/V values and
promotes to f32 before the dot product. This matches llama.cpp's
behaviour — critical for greedy decode verification where a
single divergent token at position 47 means a precision mismatch,
not a logic bug.

The f16 truncation is lossy but bounded: the maximum relative
error per value is 2^−10 ≈ 0.001. Over a full context of 32k
tokens, the accumulated error in attention scores remains small
because softmax normalisation dampens it.

## Safetensors parser

We load weights from the MLX safetensors format
(`prism-ml/Bonsai-*-mlx-1bit`), not GGUF. Safetensors is simpler
to parse and the MLX repos include `config.json` and
`tokenizer.json` as standard HuggingFace files.

### Why safetensors over GGUF

|                   | Safetensors (MLX)                                              | GGUF                                                                    |
| ----------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Parser complexity | ~80 lines. 8-byte header + JSON + raw data.                    | ~400 lines. Magic, version, 13 KV types, tensor descriptors, alignment. |
| Metadata          | `config.json` — standard JSON, parsed by `std.json`.           | Baked into binary with its own KV schema.                               |
| Tokenizer         | `tokenizer.json` — standard HF format, separate file.          | Embedded in binary as metadata KV arrays.                               |
| 1-bit encoding    | Affine (scale + bias per group). Convert to symmetric at load. | Symmetric (one scale per group). Simpler encoding but harder container. |
| Zero-copy mmap    | Tensors are contiguous and aligned.                            | Same — data region is mmap-able.                                        |

The GGUF format is 5× more code to parse for zero benefit. The
21 MB size difference (affine vs symmetric encoding) is irrelevant
when the whole model fits in RAM.

### File layout

```
┌───────────────────────────────────────────────┐
│  Header length: u64 little-endian (8 bytes)   │
├───────────────────────────────────────────────┤
│  JSON header (header_length bytes)            │
│  {                                            │
│    "tensor_name": {                           │
│      "dtype": "U8" | "F16" | "F32",          │
│      "shape": [dim0, dim1, ...],              │
│      "data_offsets": [start, end]             │
│    },                                         │
│    ...                                        │
│    "__metadata__": { ... }                     │
│  }                                            │
├───────────────────────────────────────────────┤
│  Tensor data (contiguous, aligned)            │
│  mmap this region directly into               │
│  Metal shared buffers.                        │
└───────────────────────────────────────────────┘
```

The tensor data region starts at byte `8 + header_length`. Each
tensor's `data_offsets` are relative to this start. The region
can be mmap'd and passed directly to
`MTLDevice.newBufferWithBytesNoCopy` — true zero-copy from disk
to GPU.

### Tensor naming convention

Each 1-bit projection has three tensors (`.weight`, `.scales`,
`.biases`). Non-quantized tensors (norms, QK norms) are plain
f32 or f16.

```
model.embed_tokens.weight        → embedding (packed uint8)
model.embed_tokens.scales        → embedding scales (f16)
model.embed_tokens.biases        → embedding biases (f16)
model.layers.{i}.input_layernorm.weight    → pre-attn RMSNorm (f32)
model.layers.{i}.self_attn.q_proj.weight   → Q projection (packed)
model.layers.{i}.self_attn.q_proj.scales   → Q scales (f16)
model.layers.{i}.self_attn.q_proj.biases   → Q biases (f16)
model.layers.{i}.self_attn.k_proj.*        → K projection
model.layers.{i}.self_attn.v_proj.*        → V projection
model.layers.{i}.self_attn.o_proj.*        → O projection
model.layers.{i}.self_attn.q_norm.weight   → Q RMSNorm (f32)
model.layers.{i}.self_attn.k_norm.weight   → K RMSNorm (f32)
model.layers.{i}.post_attention_layernorm.weight → pre-MLP RMSNorm
model.layers.{i}.mlp.gate_proj.*           → SwiGLU gate
model.layers.{i}.mlp.up_proj.*             → SwiGLU up
model.layers.{i}.mlp.down_proj.*           → SwiGLU down
model.norm.weight                          → final RMSNorm (f32)
```

When `tie_word_embeddings` is true (1.7B, 4B), the LM head
reuses `model.embed_tokens.weight` — no separate `lm_head`
tensor exists. When false (8B), a separate `lm_head.*` tensor
set is present.

### Scale conversion at load time

The MLX affine encoding stores `(mlx_scale, mlx_bias)` per group.
At load time, convert to the symmetric form our kernels expect:

```
symmetric_scale = mlx_scale / 2
// Then: bit=1 → +symmetric_scale, bit=0 → −symmetric_scale
```

This is one f16 division per group — negligible at load time.
The bias tensor is not needed after conversion.

## BPE tokenizer

The MLX repository includes `tokenizer.json` in standard
HuggingFace format. The tokenizer needs two operations:

1. **Encode** (text → token IDs): apply BPE merges greedily, handling
   UTF-8, special tokens, and the chat template.
2. **Decode** (token ID → text): look up the token string by index.

For Step 2, a minimal tokenizer is sufficient. Chat template
formatting can be hardcoded for Qwen3's format:

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

The `tokenizer.json` file contains the full vocabulary (151669
tokens), merge rules, and special token definitions in a single
JSON structure. Parsing it with `std.json` is straightforward —
no custom binary format to handle.

Estimated size: ~800–1200 lines of Zig. BPE merge ordering, UTF-8
boundary handling, special token detection, and the Qwen3 chat
template are each straightforward in isolation but interact in
ways that add up. Budget for the upper end.

## Autoregressive generation loop

```
fn generate(model, tokenizer, prompt, max_tokens) void {
    var token_ids = tokenizer.encode(prompt);
    var position: u32 = 0;

    // Prefill: process all prompt tokens at once.
    // Uses qmm (matrix-matrix) for each 1-bit projection.
    var logits = model.prefill(token_ids, &position);

    // Decode: generate one token at a time.
    // Uses qmv (matrix-vector) for each 1-bit projection.
    for (0..max_tokens) |_| {
        const next_token = sample(logits, temperature, top_k, top_p);
        if (next_token == eos_token) break;
        print(tokenizer.decode(next_token));
        logits = model.decode_one(next_token, &position);
    }
}
```

The prefill and decode paths share all the same transformer logic.
The only difference is which matmul kernel is dispatched: `qmm`
(M > 1) for prefill, `qmv` (M = 1) for decode.

## Back-of-envelope performance sketch

### Memory

KV cache dominates memory at long contexts. Using f16 for KV cache
(standard practice — halves memory vs f32 with negligible quality
loss).

Bonsai 1.7B on Apple Silicon (unified memory):

```
                             4k ctx       32k ctx (max)
Model weights (packed):      ~300 MB      ~300 MB
KV cache (f16):              ~768 MB      ~6 GB
  = 16 kv_heads × 128 dim × 2 (K+V) × 24 layers × ctx × 2 bytes
Activation scratch:            ~1 MB        ~1 MB
Total:                      ~1.1 GB      ~6.3 GB
```

Bonsai 8B on Apple Silicon (unified memory):

```
                             8k ctx       32k ctx      65k ctx (max)
Model weights (packed):     ~1150 MB     ~1150 MB      ~1150 MB
KV cache (f16):             ~1100 MB     ~4500 MB      ~9000 MB
  = 8 kv_heads × 128 dim × 2 (K+V) × 36 layers × ctx × 2 bytes
Activation scratch:            ~2 MB        ~2 MB         ~2 MB
Total:                      ~2.3 GB      ~5.7 GB      ~10.2 GB
```

These match PrismML's published estimates (8k: ~2.5 GB, 32k: ~5.9 GB,
65k: ~10.5 GB). The 1.7B at 4k context is the development target —
fits comfortably on any Apple Silicon Mac. Full 65k context on the
8B model requires 16+ GB unified memory.

### Arithmetic intensity

Token generation is **memory-bandwidth bound**, not compute bound.
For each token, the model reads all weights once and performs one
mat-vec per projection. The arithmetic intensity is:

```
Weight reads per token:   ~1.15 GB  (8B model, packed)
FLOPs per token:          ~16 GFLOP (8B params × 2 ops/param)
Arithmetic intensity:     ~14 FLOP/byte

M4 Pro memory bandwidth:  ~273 GB/s
Theoretical max tok/s:    273 / 1.15 ≈ 237 tok/s  (bandwidth ceiling)
```

MLX achieves 131 tok/s on M4 Pro — about 55% of the bandwidth
ceiling. The gap is overhead: Python, command buffer dispatch, kernel
launch latency. A Zig implementation that minimises this overhead
has room to close the gap.

### Kernel dispatch budget

Per token, the 8B model needs:

```
1 embedding lookup
36 × (RMSNorm + Q/K/V projections + Q norm + K norm + RoPE(Q)
       + RoPE(K) + KV cache + attention + O projection + residual
       + RMSNorm + gate/up projections + SiLU⊙mul + down projection
       + residual)
1 final RMSNorm
1 LM head projection
1 sampling step

≈ 36 × 18 + 4 = 652 kernel dispatches per token
```

All 652 dispatches should be encoded into a single Metal command
buffer. At ~1 μs per dispatch encode, the command buffer overhead
is ~0.65 ms — negligible compared to the ~7.5 ms per token at
131 tok/s.

## Verification strategy

Correctness is verified at every step, from individual kernels up to
full model output.

### Kernel-level

Each kernel has a dedicated test that:

1. Prepares input in a Metal shared buffer.
2. Dispatches the kernel.
3. Reads the output buffer on CPU.
4. Compares against a CPU reference implementation.
5. Asserts maximum absolute error < tolerance.

The tolerance depends on precision:

- f32 accumulation: < 1e-5
- f16 scales dequantized to f32: < 1e-3
- 1-bit quantized (inherent loss): compare against dequantized
  reference, not original f32
- `qmm` GPU vs CPU reference: < 0.05 (FP32 non-associativity from
  different accumulation order in tiled vs sequential matmul)

**Step 0 results.** The `mnist_1bit.zig` integration test runs a
full CPU reference check: dequantize all three layers' packed
weights, compute the forward pass with sequential matmul on the
CPU, and compare all 10 output logits against the GPU `qmm`
output. All logits matched within 0.05 tolerance, confirming
`f32_to_1bit` and `qmm` kernel correctness.

### Model-level

1. **Greedy decode match.** Run the same prompt with temperature=0
   through nnzap and llama.cpp. Output tokens must be identical.
2. **Perplexity.** Evaluate on a standard dataset (e.g., WikiText-2)
   and compare perplexity to the published Bonsai numbers.
3. **Throughput.** Measure tok/s on standardised prompts (128-token
   prompt, 128-token generation) and compare to MLX and llama.cpp.

## What this is not

This plan is inference-only. It does not cover:

- **Training or fine-tuning.** Bonsai models are pre-trained by
  PrismML. 1-bit training requires specialised techniques
  (straight-through estimators, learned scales) that are out of scope.
- **Speculative decoding.** A potential future optimisation.
- **Batched serving.** The initial target is single-sequence
  generation. Continuous batching is a separate project.
- **GGUF.** We use safetensors (MLX format), not GGUF.
- **Non-Qwen3 architectures.** The transformer implementation is
  tailored to Qwen3. Supporting Llama, Mistral, etc. is possible
  later but not a goal of this plan.

## File inventory

When all five steps are complete, the new and modified files are:

```
nn/src/
├── metal.zig              (modified: PackedBuffer, setPackedBuffer, dispatchCustom)
├── layout.zig             (modified: divCeil, packed size helpers, comptime offsets)
├── network.zig            (modified: forwardInfer1Bit, quantizeWeightsTo1Bit)
├── root.zig               (modified: re-exports PackedBuffer, divCeil)
├── transformer.zig        (new: TransformerConfig, forward pass, generation,
│                                 sampling, forwardDecode, forwardPrefill)
├── safetensors.zig        (new: safetensors parser, mmap, tensor descriptors)
├── model.zig              (new: Model struct, weight loader, tensor name mapping,
│                                 scale conversion, KV cache + scratch allocation)
├── tokenizer.zig          (new: BPE tokenizer from tokenizer.json, chat template)
├── shaders/
│   ├── compute.metal      (modified: f32_to_1bit, qmv, qmm kernels)
│   └── transformer.metal  (new: rms_norm, rms_norm_heads, silu, rope,
│                                 gqa_attention, kv_cache_update,
│                                 embedding_lookup, residual_add)
└── examples/
    ├── mnist_1bit.zig      (new: 1-bit integration test + CPU reference)
    └── bonsai.zig          (new: CLI inference demo + timing)
```

Step 0 added ~1,130 lines of Zig and ~330 lines of MSL.
Estimated remaining new code (Step 2):

| Sub-phase | File(s)                         | Estimated lines |
| --------- | ------------------------------- | --------------- |
| 2a        | `safetensors.zig`               | ~150 Zig        |
| 2b        | `transformer.zig/metal`         | ~120 Zig + MSL  |
| 2c        | `model.zig`                     | ~400 Zig        |
| 2d        | `transformer.zig`               | ~200 Zig        |
| 2e        | `tokenizer.zig`                 | ~800–1200 Zig   |
| 2f        | `transformer.zig`, `bonsai.zig` | ~400 Zig        |
| **Total** |                                 | ~2100–2500 Zig  |
