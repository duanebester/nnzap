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

| Property           | 1.7B    | 4B       | 8B       |
| ------------------ | ------- | -------- | -------- |
| Layers             | 24      | 36       | 36       |
| Hidden size        | 2048    | 3584     | 4096     |
| Intermediate (MLP) | 5632    | 9728     | 11008    |
| Query heads        | 16      | 28       | 32       |
| KV heads           | 16      | 4        | 8        |
| Head dim           | 128     | 128      | 128      |
| Vocab size         | 151936  | 151936   | 151936   |
| Context length     | 32768   | 65536    | 65536    |
| Model size on disk | ~0.3 GB | ~0.65 GB | ~1.15 GB |

The 1.7B is the implementation target. Every line of code transfers
directly to 4B and 8B — only the comptime config constants change.

## Weight format: Q1_0_g128

Each weight is a single bit: `0` maps to `−scale`, `1` maps to
`+scale`. Every group of 128 weights shares one f16 scale factor.

### Packing

8 weights per `u8`, 32 per `u32`, LSB-first:

```
byte:   [b7 b6 b5 b4 b3 b2 b1 b0]
weight:  w7 w6 w5 w4 w3 w2 w1 w0
```

### Dequantization

GGUF format stores one f16 scale per group (1.125 bits per weight):

```
w_reconstructed[i] = if (bit[i] == 1) +scale else −scale
```

MLX format stores both scale and bias per group (1.25 bits per weight)
as an affine encoding:

```
mlx_scale = 2 * original_scale
mlx_bias  = −original_scale
w_reconstructed[i] = mlx_scale * bit[i] + mlx_bias
```

We target the GGUF `Q1_0_g128` format. It is simpler (one value per
group instead of two), more compact, and the GGUF container is a
straightforward binary format with no external dependencies.

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

### Step 0 — 1-bit MNIST MLP

**Goal.** Validate the 1-bit `qmv` kernel against known-good results,
using the existing MNIST training pipeline.

**What exists.** nnzap trains a 784→128→64→10 MLP to 97.85% accuracy
on MNIST with f32 weights. f16 inference weights are already
supported via `HalfBuffer` and the `f32_to_f16` kernel.

**What to build.**

1. `PackedBuffer` — a Metal shared buffer that stores 1-bit packed
   weights with per-group f16 scales.
2. `f32_to_1bit` kernel — quantize trained f32 weights to 1-bit
   Q1_0_g128 format on the GPU.
3. `qmv` kernel — 1-bit matrix-vector multiply (the core operation).
4. `qmm` kernel — 1-bit matrix-matrix multiply (for batched
   inference).
5. A fused 1-bit inference path in `Network` that uses the packed
   weights.
6. Tests: compare 1-bit inference accuracy to f32 and f16 baselines.

**Verification.** Run 10,000 MNIST test images through both the f32
path and the 1-bit path. The 1-bit accuracy will be lower (expected —
the MNIST MLP was not trained for 1-bit), but the kernel correctness
can be verified by comparing against a CPU reference implementation
that dequantizes and multiplies in f32.

**Deliverables.**

| File                           | Contents                            |
| ------------------------------ | ----------------------------------- |
| `nn/src/metal.zig`             | `PackedBuffer` struct               |
| `nn/src/shaders/compute.metal` | `f32_to_1bit`, `qmv`, `qmm` kernels |
| `nn/src/network.zig`           | 1-bit inference path                |
| `nn/src/layout.zig`            | Packed buffer size helpers          |

### Step 1 — transformer primitives

**Goal.** Implement and test every kernel needed by a Qwen3
transformer block, in isolation, before combining them.

**What to build.**

| Kernel             | What it does                             | Test strategy                                    |
| ------------------ | ---------------------------------------- | ------------------------------------------------ |
| `rms_norm`         | x / sqrt(mean(x²) + eps) \* scale        | Compare to CPU reference, 5 decimal places       |
| `silu`             | x \* sigmoid(x), elementwise             | Compare to CPU reference                         |
| `rope`             | Apply rotary position embeddings         | Compare to known sin/cos table values            |
| `gqa_attention`    | Grouped query attention with causal mask | Small 2-head, 4-token case vs CPU                |
| `kv_cache_update`  | Append K/V to cache at position          | Verify written values, unwritten slots unchanged |
| `embedding_lookup` | Gather rows from packed 1-bit table      | Verify against CPU dequant + gather              |

Each kernel is a standalone function with primitive arguments
(no `self`), tested independently with a Metal command buffer
that writes results to a shared buffer and compares on CPU.

**Deliverables.**

| File                               | Contents                               |
| ---------------------------------- | -------------------------------------- |
| `nn/src/shaders/transformer.metal` | All transformer kernels                |
| `nn/src/transformer.zig`           | Dispatch helpers + `TransformerConfig` |

### Step 1.5 — single-block integration test

**Goal.** Verify that the transformer kernels compose correctly
through a single decoder block, before wiring 24+ blocks together.

The jump from "6 kernels pass individual tests" to "24 blocks
produce correct text" is where buffer layout bugs, residual-add
wiring mistakes, and f16/f32 promotion mismatches hide. A
single-block test catches these cheaply.

**What to build.**

1. A tiny test config: 2 query heads, 2 KV heads, 64 hidden dim,
   128 intermediate dim, 1 layer, 4-token context. Small enough
   that a CPU reference is trivial to write.
2. A CPU reference function that runs one full decoder block:
   RMSNorm → Q/K/V projection → RoPE → KV cache append →
   attention scores → softmax → weighted V sum → O projection →
   residual add → RMSNorm → gate/up projection → SiLU →
   elementwise mul → down projection → residual add.
3. A GPU test that encodes the same block into a command buffer,
   runs it, and compares the output against the CPU reference.

This test exercises every buffer hand-off between kernels:

- RMSNorm output feeds into Q/K/V projection input.
- Q and K pass through RoPE before the dot product.
- KV cache stores f16 values; attention loads and promotes to f32.
- Attention output passes through O projection, then adds back to
  the pre-attention residual (not the post-RMSNorm intermediate).
- The MLP residual adds to the post-attention output, not the
  post-RMSNorm intermediate.

**Verification.** Maximum absolute error between GPU and CPU
reference < 1e-3 across all output elements. The tolerance is
looser than individual kernel tests because errors accumulate
through the block.

**Deliverables.**

| File                     | Contents                            |
| ------------------------ | ----------------------------------- |
| `nn/src/transformer.zig` | `forwardBlock` helper + test config |
| (test in same file)      | CPU reference + GPU comparison test |

### Step 2 — Bonsai 1.7B inference

**Goal.** Load the Bonsai 1.7B GGUF file, run a prompt through the
full transformer, and produce correct text output.

**What to build.**

1. **GGUF parser** — read the binary header, metadata key-value pairs,
   and tensor descriptors. Map packed weight data directly into
   `PackedBuffer` instances (zero-copy from mmap).
2. **BPE tokenizer** — decode the vocabulary and merges from the GGUF
   metadata. Implement encode (text → token IDs) and decode
   (token IDs → text).
3. **Transformer forward pass** — wire the Step 1 kernels into a
   sequential 24-block forward pass. Encode all 24 blocks + LM head
   into a single Metal command buffer per token.
4. **KV cache management** — pre-allocate the full KV cache at init
   (context_length × num_kv_heads × head_dim × 2 × num_layers
   × sizeof(f16)). See "Precision at the KV cache boundary" below.
5. **Autoregressive generation loop** — embed → 24 blocks → norm →
   LM head → sample → append → repeat.
6. **Sampling** — temperature scaling, top-k filtering, top-p
   (nucleus) sampling.

**Verification.** Run the same prompt through both nnzap and
llama.cpp (using the same GGUF file and temperature=0 greedy
decoding). The output tokens must match exactly.

**Deliverables.**

| File                     | Contents                                |
| ------------------------ | --------------------------------------- |
| `nn/src/gguf.zig`        | GGUF binary format parser               |
| `nn/src/tokenizer.zig`   | BPE tokenizer (encode + decode)         |
| `nn/src/transformer.zig` | Transformer forward pass + generation   |
| `nn/examples/bonsai.zig` | CLI: load model, run prompt, print text |

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
    .vocab_size = 151936,
    .hidden_size = 2048,
    .intermediate_size = 5632,
    .num_layers = 24,
    .num_query_heads = 16,
    .num_kv_heads = 16,
    .head_dim = 128,
    .max_context_length = 32768,
    .max_prefill_length = 512,
});

const Bonsai4B = TransformerConfig(.{
    .vocab_size = 151936,
    .hidden_size = 3584,
    .intermediate_size = 9728,
    .num_layers = 36,
    .num_query_heads = 28,
    .num_kv_heads = 4,
    .head_dim = 128,
    .max_context_length = 65536,
    .max_prefill_length = 512,
});

const Bonsai8B = TransformerConfig(.{
    .vocab_size = 151936,
    .hidden_size = 4096,
    .intermediate_size = 11008,
    .num_layers = 36,
    .num_query_heads = 32,
    .num_kv_heads = 8,
    .head_dim = 128,
    .max_context_length = 65536,
    .max_prefill_length = 512,
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

When binding a `PackedBuffer` to a compute encoder for `qmv`:

```
buffer(0) = packed bits     (uint8_t*)
buffer(1) = scales          (half*)    — same MTLBuffer, offset by packedBytes()
buffer(2) = input vector    (float*)
buffer(3) = output vector   (float*)
buffer(4) = dimensions      (QMVDims via setBytes)
```

The bits and scales live in the same `MTLBuffer`. The encoder binds
the same buffer twice with different offsets to avoid an extra
allocation.

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

Used during prefill (processing the full prompt, M >> 1). Where
`qmv` computes one output element per simdgroup, `qmm` computes a
tile of output elements using shared memory — same idea as the
existing `matmul_tiled` in `compute.metal`, but with a custom block
loader that dequantizes 1-bit tiles on the fly.

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

Tile load (B — 1-bit packed weights):
  Each thread loads packed bytes, dequantizes into threadgroup
  memory as f32:

    uint8_t wb = packed[byte_offset];
    float scale = scales[group_idx];
    // Dequantize 8 weights per byte into shared memory.
    tile_B[col + 0] = (wb & 0x01) ? +scale : -scale;
    tile_B[col + 1] = (wb & 0x02) ? +scale : -scale;
    ...
    tile_B[col + 7] = (wb & 0x80) ? +scale : -scale;

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

## GGUF parser

[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) is
a single-file binary format. The parser needs to handle:

```
┌──────────────────────────────────┐
│  Header                         │
│  magic: "GGUF" (4 bytes)        │
│  version: u32 (= 3)             │
│  tensor_count: u64              │
│  metadata_kv_count: u64         │
├──────────────────────────────────┤
│  Metadata key-value pairs       │
│  (architecture, vocab, merges,  │
│   tokenizer config, etc.)       │
├──────────────────────────────────┤
│  Tensor descriptors             │
│  (name, shape, type, offset)    │
├──────────────────────────────────┤
│  Alignment padding              │
├──────────────────────────────────┤
│  Tensor data                    │
│  (packed weights, scales, etc.) │
│  mmap this region directly into │
│  Metal shared buffers.          │
└──────────────────────────────────┘
```

The tensor data region can be mmap'd and the resulting pointer passed
directly to `MTLDevice.newBufferWithBytesNoCopy` — true zero-copy
from disk to GPU. This avoids reading the entire file into memory and
then copying it into Metal buffers.

### Key metadata fields

| Key                             | Value (Bonsai 8B)  |
| ------------------------------- | ------------------ |
| `general.architecture`          | `qwen3`            |
| `qwen3.block_count`             | `36`               |
| `qwen3.embedding_length`        | `4096`             |
| `qwen3.feed_forward_length`     | `11008`            |
| `qwen3.attention.head_count`    | `32`               |
| `qwen3.attention.head_count_kv` | `8`                |
| `qwen3.rope.freq_base`          | `1000000.0`        |
| `tokenizer.ggml.model`          | `gpt2` (BPE)       |
| `tokenizer.ggml.tokens`         | `[151936 strings]` |
| `tokenizer.ggml.merges`         | `[merge rules]`    |

### Tensor naming convention

```
token_embd.weight              → embedding
blk.{i}.attn_norm.weight       → pre-attention RMSNorm scale
blk.{i}.attn_q.weight          → Q projection
blk.{i}.attn_k.weight          → K projection
blk.{i}.attn_v.weight          → V projection
blk.{i}.attn_output.weight     → O projection
blk.{i}.ffn_norm.weight        → pre-MLP RMSNorm scale
blk.{i}.ffn_gate.weight        → SwiGLU gate projection
blk.{i}.ffn_up.weight          → SwiGLU up projection
blk.{i}.ffn_down.weight        → SwiGLU down projection
output_norm.weight              → final RMSNorm scale
output.weight                   → LM head
```

## BPE tokenizer

The GGUF file embeds the full vocabulary and merge table. The
tokenizer needs two operations:

1. **Encode** (text → token IDs): apply BPE merges greedily, handling
   UTF-8, special tokens, and the chat template.
2. **Decode** (token ID → text): look up the token string by index.

For Step 2, a minimal tokenizer is sufficient. Chat template
formatting can be hardcoded for Qwen3's format:

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

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
36 × (RMSNorm + Q/K/V/O projections + RoPE + attention + RMSNorm
       + gate/up/down projections + SiLU + elementwise mul)
1 final RMSNorm
1 LM head projection
1 sampling step

≈ 36 × 11 + 4 = 400 kernel dispatches per token
```

All 400 dispatches should be encoded into a single Metal command
buffer. At ~1 μs per dispatch encode, the command buffer overhead
is ~0.4 ms — negligible compared to the ~7.5 ms per token at
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
- **GGUF writing.** We only read GGUF files, never write them.
- **Non-Qwen3 architectures.** The transformer implementation is
  tailored to Qwen3. Supporting Llama, Mistral, etc. is possible
  later but not a goal of this plan.

## File inventory

When all five steps are complete, the new and modified files are:

```
nn/src/
├── metal.zig              (modified: PackedBuffer)
├── layout.zig             (modified: packed size helpers, divCeil)
├── network.zig            (modified: 1-bit MNIST inference path)
├── transformer.zig        (new: TransformerConfig, forward pass, generation)
├── gguf.zig               (new: GGUF binary parser)
├── tokenizer.zig          (new: BPE tokenizer)
├── shaders/
│   ├── compute.metal      (modified: f32_to_1bit, qmv, qmm kernels)
│   └── transformer.metal  (new: rms_norm, silu, rope, gqa_attention,
│                                 kv_cache_update, embedding_lookup)
└── examples/
    └── bonsai.zig          (new: CLI inference demo)
```

Estimated total new code: ~3,200 lines of Zig, ~1,200 lines of MSL.
