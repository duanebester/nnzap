//! Transformer primitives — Step 1 of the Bonsai implementation plan.
//!
//! Provides:
//!   - `TransformerConfig`: comptime type that resolves all buffer sizes
//!     and offsets for a Qwen3-style transformer.
//!   - `TransformerPipelines`: compiled Metal compute pipelines for
//!     transformer kernels (rms_norm, silu, rope, gqa_attention,
//!     kv_cache_update, embedding_lookup, residual_add).
//!   - Dispatch helpers: standalone functions that bind buffers and
//!     dispatch each kernel.
//!
//! All dispatch helpers accept raw `objc.Object` buffer handles and
//! dimension structs (matching the Metal shader structs).  Higher-level
//! code (Step 1.5+) wraps these with typed buffers.

const std = @import("std");
const objc = @import("objc");
const metal = @import("metal.zig");
const layout = @import("layout.zig");

const log = std.log.scoped(.transformer);
const divCeil = layout.divCeil;

// ============================================================================
// TransformerDesc — input configuration for TransformerConfig
// ============================================================================

/// Describes a Qwen3-style transformer architecture.  All fields are
/// required; no defaults.  Pass this to `TransformerConfig` to get a
/// comptime type with all derived sizes and offsets.
pub const TransformerDesc = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_layers: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_context_length: u32,
    max_prefill_length: u32,
    rope_theta: f32,
    tie_word_embeddings: bool,
};

// ============================================================================
// TransformerConfig — comptime type with all derived constants
// ============================================================================

/// Resolves all buffer sizes, weight counts, and activation scratch
/// requirements at compile time for a given transformer description.
/// Follows the same pattern as `NetworkLayout`: zero runtime cost,
/// every constant is a `pub const`.
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
        pub const max_context_length: u32 =
            cfg.max_context_length;
        pub const max_prefill_length: u32 =
            cfg.max_prefill_length;
        pub const rope_theta: f32 = cfg.rope_theta;
        pub const tie_word_embeddings: bool =
            cfg.tie_word_embeddings;
        pub const group_size: u32 = 128;

        // -- Derived constants --

        pub const query_dim: u32 =
            num_query_heads * head_dim;
        pub const kv_dim: u32 = num_kv_heads * head_dim;
        pub const heads_per_kv_group: u32 =
            num_query_heads / num_kv_heads;

        // -- Comptime assertions --

        comptime {
            // GQA: query heads must be a multiple of KV heads.
            std.debug.assert(
                num_query_heads % num_kv_heads == 0,
            );
            // Group size must divide head dim or vice versa.
            std.debug.assert(
                head_dim % group_size == 0 or group_size % head_dim == 0,
            );
            // RoPE operates on pairs — head dim must be even.
            std.debug.assert(head_dim % 2 == 0);
            // Basic sanity: no zero-sized dimensions.
            std.debug.assert(vocab_size > 0);
            std.debug.assert(hidden_size > 0);
            std.debug.assert(intermediate_size > 0);
            std.debug.assert(num_layers > 0);
            std.debug.assert(max_context_length > 0);
            std.debug.assert(max_prefill_length > 0);
            std.debug.assert(
                max_prefill_length <= max_context_length,
            );
            // Note: hidden_size != query_dim is valid (4B model
            // has hidden=2560, query_dim=4096).  The Q projection
            // handles the size change.
        }

        // -- 1-bit packed sizes (bytes) --

        /// Total packed bytes (bits + f16 scales) for a weight
        /// matrix with `rows × cols` elements.
        fn packedBytes(
            comptime rows: u32,
            comptime cols: u32,
        ) u32 {
            const total = rows * cols;
            const bit_bytes = divCeil(total, 8);
            const num_groups = divCeil(total, group_size);
            const scale_bytes = num_groups * 2; // f16
            return bit_bytes + scale_bytes;
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
            q_proj_bytes + k_proj_bytes + v_proj_bytes + o_proj_bytes;
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
            if (tie_word_embeddings) 0 else packedBytes(hidden_size, vocab_size);

        // RMSNorm scales: f16, two per layer (attn + ffn) + final.
        // QK norms: f16, two per layer (q_norm + k_norm), each
        // head_dim elements.  Stored as f16 at inference time
        // (converted from f32 at load time — negligible loss for
        // values near 1.0).
        pub const norm_scales_per_layer: u32 =
            hidden_size * 2 + head_dim * 2;
        pub const total_norm_scale_count: u32 =
            norm_scales_per_layer * num_layers + hidden_size;

        // Total model weight bytes.  u64 for future-proofing —
        // the 8B model exceeds 1 GB.
        pub const total_weight_bytes: u64 =
            @as(u64, embedding_bytes) + total_layer_weight_bytes + lm_head_bytes + total_norm_scale_count * 2; // f16

        // -- KV cache sizes (f16 elements) --
        // u64 because large contexts can exceed u32 range.

        pub const kv_cache_elements_per_layer: u64 =
            @as(u64, max_context_length) * kv_dim * 2;
        pub const total_kv_cache_elements: u64 =
            kv_cache_elements_per_layer * num_layers;
        pub const total_kv_cache_bytes: u64 =
            total_kv_cache_elements * 2; // f16

        // -- Activation scratch (f32 elements) --
        // Pre-allocate for the larger of decode and prefill.

        pub const decode_activation_elements: u32 = blk: {
            var max: u32 = hidden_size;
            if (query_dim > max) max = query_dim;
            if (intermediate_size > max) {
                max = intermediate_size;
            }
            const attn_scores =
                num_query_heads * max_context_length;
            if (attn_scores > max) max = attn_scores;
            break :blk max;
        };

        pub const prefill_activation_elements: u64 = blk: {
            const p: u64 = max_prefill_length;
            var max: u64 = p * hidden_size;
            const qp = p * query_dim;
            if (qp > max) max = qp;
            const ip = p * intermediate_size;
            if (ip > max) max = ip;
            const attn = @as(u64, num_query_heads) * p * max_context_length;
            if (attn > max) max = attn;
            break :blk max;
        };

        pub const max_activation_elements: u64 = blk: {
            const d: u64 = decode_activation_elements;
            const p: u64 = prefill_activation_elements;
            break :blk if (p > d) p else d;
        };
    };
}

// ============================================================================
// Model configs — validate comptime logic across all sizes
// ============================================================================

pub const Bonsai1_7B = TransformerConfig(.{
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

pub const Bonsai4B = TransformerConfig(.{
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

pub const Bonsai8B = TransformerConfig(.{
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

/// Tiny test config for Step 1.5 single-block integration test.
/// Small enough that all activation scratch fits in stack arrays
/// for the CPU reference.  Dimensions chosen so that:
///   - every qmv dispatch has K % 32 == 0 (simd lane requirement),
///   - every qmv dispatch has K >= group_size (128), because qmv
///     indexes scales per-row as `row * ceil(K/group_size)`, which
///     only matches the flat group layout when K >= group_size.
const TinyBlock = TransformerConfig(.{
    .vocab_size = 32,
    .hidden_size = 128,
    .intermediate_size = 256,
    .num_layers = 1,
    .num_query_heads = 2,
    .num_kv_heads = 2,
    .head_dim = 64,
    .max_context_length = 8,
    .max_prefill_length = 4,
    .rope_theta = 10000.0,
    .tie_word_embeddings = true,
});

// ============================================================================
// Metal extern structs — must match shader struct layouts exactly
// ============================================================================

/// RMSNorm kernel dimensions.
pub const RMSNormDims = extern struct {
    hidden_size: u32,
    num_tokens: u32,
    eps: f32,
};

/// RoPE kernel dimensions.
pub const RoPEDims = extern struct {
    num_heads: u32,
    head_dim: u32,
    position: u32,
    rope_theta: f32,
};

/// KV cache update dimensions.
pub const KVUpdateDims = extern struct {
    num_kv_heads: u32,
    head_dim: u32,
    position: u32,
    max_context_length: u32,
};

/// Grouped query attention dimensions.
pub const GQADims = extern struct {
    num_query_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_context_length: u32,
    heads_per_kv_group: u32,
};

/// 1-bit embedding lookup dimensions.
pub const EmbedDims = extern struct {
    vocab_size: u32,
    hidden_size: u32,
    num_tokens: u32,
    group_size: u32,
};

/// 1-bit matrix-vector multiply dimensions.  Must match Metal's
/// QMVDims in compute.metal.  M = output rows (weight matrix
/// rows), K = input dimension (weight matrix columns).
pub const QMVDims = extern struct {
    M: u32,
    K: u32,
    group_size: u32,
};

// ============================================================================
// TransformerPipelines — compiled Metal kernels
// ============================================================================

/// Holds pre-compiled Metal compute pipelines for all transformer
/// kernels.  Compiles `transformer.metal` as a separate library
/// from the training kernels in `compute.metal`.
pub const TransformerPipelines = struct {
    rms_norm: metal.ComputePipeline,
    silu: metal.ComputePipeline,
    silu_elementwise_mul: metal.ComputePipeline,
    rope: metal.ComputePipeline,
    kv_cache_update: metal.ComputePipeline,
    gqa_attention: metal.ComputePipeline,
    embedding_lookup: metal.ComputePipeline,
    residual_add: metal.ComputePipeline,
    library: objc.Object,

    /// Compile the transformer shader library and create all
    /// pipeline states.  Uses in-place initialisation (Rule 13).
    pub fn init(
        self: *TransformerPipelines,
        device_obj: objc.Object,
    ) !void {
        std.debug.assert(device_obj.value != null);

        const lib = try compileTransformerLibrary(
            device_obj,
        );

        self.* = .{
            .library = lib,
            .rms_norm = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "rms_norm",
            ),
            .silu = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "silu",
            ),
            .silu_elementwise_mul = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "silu_elementwise_mul",
            ),
            .rope = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "rope",
            ),
            .kv_cache_update = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "kv_cache_update",
            ),
            .gqa_attention = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "gqa_attention",
            ),
            .embedding_lookup = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "embedding_lookup",
            ),
            .residual_add = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "residual_add",
            ),
        };
    }
};

// ============================================================================
// Dispatch helpers — standalone, no self (Rule 20)
// ============================================================================

/// Bind a raw `objc.Object` (MTLBuffer) to a compute encoder.
fn setRawBuffer(
    encoder: objc.Object,
    buffer: objc.Object,
    offset_bytes: u32,
    index: u32,
) void {
    std.debug.assert(index <= 31);
    encoder.msgSend(void, "setBuffer:offset:atIndex:", .{
        buffer.value,
        @as(c_ulong, offset_bytes),
        @as(c_ulong, index),
    });
}

/// Dispatch RMSNorm: output = input * rsqrt(mean(input^2) + eps)
/// * scale.  One threadgroup (256 threads) per token.
pub fn dispatchRMSNorm(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    input_buffer: objc.Object,
    scale_buffer: objc.Object,
    output_buffer: objc.Object,
    dims: RMSNormDims,
) void {
    std.debug.assert(dims.hidden_size > 0);
    std.debug.assert(dims.num_tokens > 0);

    setRawBuffer(encoder, input_buffer, 0, 0);
    setRawBuffer(encoder, scale_buffer, 0, 1);
    setRawBuffer(encoder, output_buffer, 0, 2);
    metal.setBytes(encoder, RMSNormDims, &dims, 3);

    const grid = metal.MTLSize{
        .width = @as(c_ulong, dims.num_tokens),
        .height = 1,
        .depth = 1,
    };
    const group = metal.MTLSize{
        .width = 256,
        .height = 1,
        .depth = 1,
    };
    device.dispatchCustom(encoder, pipeline, grid, group);
}

/// Dispatch SiLU: output[i] = input[i] * sigmoid(input[i]).
pub fn dispatchSiLU(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    input_buffer: objc.Object,
    output_buffer: objc.Object,
    count: u32,
) void {
    std.debug.assert(count > 0);

    setRawBuffer(encoder, input_buffer, 0, 0);
    setRawBuffer(encoder, output_buffer, 0, 1);
    metal.setBytes(encoder, u32, &count, 2);

    device.dispatch1D(encoder, pipeline, count);
}

/// Dispatch fused SiLU + elementwise multiply for SwiGLU:
/// output[i] = silu(gate[i]) * up[i].
pub fn dispatchSiLUElementwiseMul(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    gate_buffer: objc.Object,
    up_buffer: objc.Object,
    output_buffer: objc.Object,
    count: u32,
) void {
    std.debug.assert(count > 0);

    setRawBuffer(encoder, gate_buffer, 0, 0);
    setRawBuffer(encoder, up_buffer, 0, 1);
    setRawBuffer(encoder, output_buffer, 0, 2);
    metal.setBytes(encoder, u32, &count, 3);

    device.dispatch1D(encoder, pipeline, count);
}

/// Dispatch RoPE: apply rotary position embeddings in-place.
/// One thread per (head, pair) — total = num_heads * head_dim / 2.
pub fn dispatchRoPE(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    data_buffer: objc.Object,
    dims: RoPEDims,
) void {
    std.debug.assert(dims.num_heads > 0);
    std.debug.assert(dims.head_dim >= 2);
    std.debug.assert(dims.head_dim % 2 == 0);

    setRawBuffer(encoder, data_buffer, 0, 0);
    metal.setBytes(encoder, RoPEDims, &dims, 1);

    const count = dims.num_heads * (dims.head_dim / 2);
    device.dispatch1D(encoder, pipeline, count);
}

/// Dispatch KV cache update: write f32 projections into f16 cache
/// at the given position.  One thread per element.
pub fn dispatchKVCacheUpdate(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    k_proj_buffer: objc.Object,
    v_proj_buffer: objc.Object,
    k_cache_buffer: objc.Object,
    v_cache_buffer: objc.Object,
    dims: KVUpdateDims,
) void {
    std.debug.assert(dims.num_kv_heads > 0);
    std.debug.assert(dims.head_dim > 0);
    std.debug.assert(
        dims.position < dims.max_context_length,
    );

    setRawBuffer(encoder, k_proj_buffer, 0, 0);
    setRawBuffer(encoder, v_proj_buffer, 0, 1);
    setRawBuffer(encoder, k_cache_buffer, 0, 2);
    setRawBuffer(encoder, v_cache_buffer, 0, 3);
    metal.setBytes(encoder, KVUpdateDims, &dims, 4);

    const count = dims.num_kv_heads * dims.head_dim;
    device.dispatch1D(encoder, pipeline, count);
}

/// Dispatch grouped query attention (decode path, M=1).
/// One threadgroup (256 threads) per query head.
pub fn dispatchGQAAttention(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    q_buffer: objc.Object,
    k_cache_buffer: objc.Object,
    v_cache_buffer: objc.Object,
    output_buffer: objc.Object,
    scratch_buffer: objc.Object,
    dims: GQADims,
) void {
    std.debug.assert(dims.num_query_heads > 0);
    std.debug.assert(dims.seq_len > 0);
    std.debug.assert(
        dims.seq_len <= dims.max_context_length,
    );
    std.debug.assert(
        dims.heads_per_kv_group == dims.num_query_heads / dims.num_kv_heads,
    );

    setRawBuffer(encoder, q_buffer, 0, 0);
    setRawBuffer(encoder, k_cache_buffer, 0, 1);
    setRawBuffer(encoder, v_cache_buffer, 0, 2);
    setRawBuffer(encoder, output_buffer, 0, 3);
    setRawBuffer(encoder, scratch_buffer, 0, 4);
    metal.setBytes(encoder, GQADims, &dims, 5);

    const grid = metal.MTLSize{
        .width = @as(c_ulong, dims.num_query_heads),
        .height = 1,
        .depth = 1,
    };
    const group = metal.MTLSize{
        .width = 256,
        .height = 1,
        .depth = 1,
    };
    device.dispatchCustom(encoder, pipeline, grid, group);
}

/// Dispatch 1-bit embedding lookup: gather + dequantize rows
/// from a packed embedding table.  One thread per output element.
pub fn dispatchEmbeddingLookup(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    token_ids_buffer: objc.Object,
    packed_bits_buffer: objc.Object,
    scales_buffer: objc.Object,
    output_buffer: objc.Object,
    dims: EmbedDims,
) void {
    std.debug.assert(dims.num_tokens > 0);
    std.debug.assert(dims.hidden_size > 0);
    std.debug.assert(dims.vocab_size > 0);

    setRawBuffer(encoder, token_ids_buffer, 0, 0);
    setRawBuffer(encoder, packed_bits_buffer, 0, 1);
    setRawBuffer(encoder, scales_buffer, 0, 2);
    setRawBuffer(encoder, output_buffer, 0, 3);
    metal.setBytes(encoder, EmbedDims, &dims, 4);

    const count = dims.num_tokens * dims.hidden_size;
    device.dispatch1D(encoder, pipeline, count);
}

/// Dispatch residual add: residual[i] += addition[i], in-place.
pub fn dispatchResidualAdd(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    residual_buffer: objc.Object,
    addition_buffer: objc.Object,
    count: u32,
) void {
    std.debug.assert(count > 0);

    setRawBuffer(encoder, residual_buffer, 0, 0);
    setRawBuffer(encoder, addition_buffer, 0, 1);
    metal.setBytes(encoder, u32, &count, 2);

    device.dispatch1D(encoder, pipeline, count);
}

/// Dispatch fused pair of 1-bit matrix-vector multiplies.
/// Computes output_a = W_a × input AND output_b = W_b × input
/// in a single kernel dispatch, sharing the input vector in
/// threadgroup memory.  Both weight matrices must have the
/// same M, K, and group_size.  Saves one full dispatch
/// overhead (~7 Obj-C msgSend) per call vs two dispatchQMV.
fn dispatchQMVFusedPair(
    device: *const metal.Device,
    encoder: objc.Object,
    packed_a: metal.PackedBuffer,
    packed_b: metal.PackedBuffer,
    input_buffer: objc.Object,
    output_a: objc.Object,
    output_b: objc.Object,
    dims: QMVDims,
) void {
    std.debug.assert(dims.M > 0);
    std.debug.assert(dims.K > 0);
    std.debug.assert(dims.K % 32 == 0);
    std.debug.assert(dims.group_size > 0);
    std.debug.assert(dims.K % dims.group_size == 0);
    std.debug.assert(dims.K <= 6144);
    std.debug.assert(dims.K >= 256);
    // Single-group requirement: same as qmv_fast.
    std.debug.assert(dims.K / 32 <= dims.group_size);
    std.debug.assert(packed_a.packed_count > 0);
    std.debug.assert(packed_b.packed_count > 0);

    // buffer(0,1): packed A (bits + scales).
    metal.setPackedBuffer(encoder, packed_a, 0);
    // buffer(2): shared input vector [K] f32.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): output A [M] f32.
    setRawBuffer(encoder, output_a, 0, 3);
    // buffer(4,5): packed B (bits + scales).
    metal.setPackedBuffer(encoder, packed_b, 4);
    // buffer(6): output B [M] f32.
    setRawBuffer(encoder, output_b, 0, 6);
    // buffer(7): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 7);

    // 16 rows per threadgroup, 512 threads (same as qmv_fast).
    const threadgroups = (dims.M + 15) / 16;
    const grid = metal.MTLSize{
        .width = @as(c_ulong, threadgroups),
        .height = 1,
        .depth = 1,
    };
    const group = metal.MTLSize{
        .width = 512,
        .height = 1,
        .depth = 1,
    };
    device.dispatchCustom(
        encoder,
        device.qmv_fused_pair,
        grid,
        group,
    );
}

/// Dispatch 1-bit matrix-vector multiply (qmv): output = W_1bit × input.
/// W is [M × K] in Q1_0_g128 packed format.  Input is [K] f32,
/// output is [M] f32.  Uses the qmv pipeline from compute.metal
/// (on Device, not TransformerPipelines).
///
/// Dispatch: threadgroups = ceil(M / 2), threads_per_group = 64.
/// Two simdgroups per threadgroup → two output rows per threadgroup.
pub fn dispatchQMV(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    packed_buffer: metal.PackedBuffer,
    input_buffer: objc.Object,
    output_buffer: objc.Object,
    dims: QMVDims,
) void {
    std.debug.assert(dims.M > 0);
    std.debug.assert(dims.K > 0);
    // qmv partitions K across 32 simd lanes — K must divide evenly.
    std.debug.assert(dims.K % 32 == 0);
    std.debug.assert(dims.group_size > 0);
    std.debug.assert(packed_buffer.packed_count > 0);

    // Use qmv_fast variants when K is group-aligned, fits
    // in shared memory (6144 floats = 24 KB < 32 KB limit),
    // and K/32 >= 8 (each lane processes at least one byte).
    // For K/32 > group_size, lanes span multiple groups —
    // use qmv_fast_multigroup which tracks group boundaries.
    const aligned = (dims.K % dims.group_size == 0) and
        (dims.K <= 6144) and (dims.K >= 256);
    const single_group = dims.K / 32 <= dims.group_size;
    const actual_pipeline = if (aligned and single_group)
        device.qmv_fast
    else if (aligned)
        device.qmv_fast_multigroup
    else
        pipeline;

    // buffer(0,1): packed bits + f16 scales (same MTLBuffer,
    // two bindings at different offsets).
    metal.setPackedBuffer(encoder, packed_buffer, 0);
    // buffer(2): input vector [K] f32.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): output vector [M] f32.
    setRawBuffer(encoder, output_buffer, 0, 3);
    // buffer(4): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 4);

    // qmv_fast variants: 16 rows/tg (16 simdgroups of 32).
    // qmv fallback: 2 rows/tg (2 simdgroups of 32).
    const rows_per_tg: u32 = if (aligned) 16 else 2;
    const threads_per_tg: u32 = if (aligned) 512 else 64;
    const threadgroups = (dims.M + rows_per_tg - 1) / rows_per_tg;
    const grid = metal.MTLSize{
        .width = @as(c_ulong, threadgroups),
        .height = 1,
        .depth = 1,
    };
    const group = metal.MTLSize{
        .width = @as(c_ulong, threads_per_tg),
        .height = 1,
        .depth = 1,
    };
    device.dispatchCustom(encoder, actual_pipeline, grid, group);
}

// ============================================================================
// Single-block forward (Step 1.5)
// ============================================================================

/// All buffer handles needed to encode one decoder block.
/// Packed weight buffers store 1-bit weights + f16 scales.
/// Activation and cache buffers are raw `objc.Object` handles,
/// following the convention of the individual dispatch helpers.
pub const ForwardBlockArgs = struct {
    // Packed weight buffers (1-bit, 7 projections).
    q_proj: metal.PackedBuffer,
    k_proj: metal.PackedBuffer,
    v_proj: metal.PackedBuffer,
    o_proj: metal.PackedBuffer,
    gate_proj: metal.PackedBuffer,
    up_proj: metal.PackedBuffer,
    down_proj: metal.PackedBuffer,
    // Norm scale buffers (f16).
    attn_norm_scale: objc.Object,
    ffn_norm_scale: objc.Object,
    // QK norm scale buffers (f16, head_dim elements each).
    q_norm_scale: objc.Object,
    k_norm_scale: objc.Object,
    // Activation scratch buffers (f32).
    residual: objc.Object, // [hidden_size] — modified in place.
    norm_out: objc.Object, // [hidden_size] — scratch.
    q: objc.Object, // [query_dim]
    k: objc.Object, // [kv_dim]
    v: objc.Object, // [kv_dim]
    attn_out: objc.Object, // [query_dim]
    proj_out: objc.Object, // [hidden_size] — O and down output.
    attn_scratch: objc.Object, // [num_query_heads × max_ctx]
    gate: objc.Object, // [intermediate_size]
    up: objc.Object, // [intermediate_size]
    mlp_out: objc.Object, // [intermediate_size] — SiLU output.
    // KV cache (f16).
    k_cache: objc.Object,
    v_cache: objc.Object,
    // Sequence state.
    position: u32,
    seq_len: u32, // = position + 1 for single-token decode.
};

/// Encode one full decoder block (attention + MLP) into a
/// compute command encoder.  `Config` is a comptime
/// `TransformerConfig` type providing all dimension constants.
///
/// The block follows Qwen3/Bonsai architecture:
///   RMSNorm → Q/K/V proj → Q norm → K norm → RoPE →
///   KV cache → attention → O proj → residual add →
///   RMSNorm → gate/up proj → SiLU⊙mul → down proj →
///   residual add.
///
/// Buffer memory barriers are inserted between dependent
/// dispatches to ensure write visibility within the encoder.
pub fn forwardBlock(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    args: ForwardBlockArgs,
) void {
    comptime {
        std.debug.assert(Config.hidden_size > 0);
        std.debug.assert(Config.num_layers > 0);
    }
    std.debug.assert(args.seq_len > 0);
    std.debug.assert(
        args.position < Config.max_context_length,
    );
    std.debug.assert(
        args.seq_len <= Config.max_context_length,
    );

    encodeAttentionHalf(
        Config,
        device,
        encoder,
        pipelines,
        args,
    );
    encodeMLPHalf(
        Config,
        device,
        encoder,
        pipelines,
        args,
    );
}

// ============================================================================
// Full-model forward decode (Step 2d)
// ============================================================================

/// All buffer handles needed to encode a single-token decode
/// pass through the full transformer: embedding lookup →
/// N × forwardBlock → final RMSNorm → LM head projection.
///
/// Per-layer arrays are slices of length `Config.num_layers`,
/// borrowed from the model's fixed-size arrays.  Activation
/// scratch buffers are shared across all layers.
pub const ForwardDecodeArgs = struct {
    // Model-level packed weights.
    embedding: metal.PackedBuffer,
    lm_head: metal.PackedBuffer, // = embedding when tied.
    final_norm_scale: objc.Object, // f16 HalfBuffer.

    // Per-layer packed weight buffers (indexed by layer).
    q_proj: []const metal.PackedBuffer,
    k_proj: []const metal.PackedBuffer,
    v_proj: []const metal.PackedBuffer,
    o_proj: []const metal.PackedBuffer,
    gate_proj: []const metal.PackedBuffer,
    up_proj: []const metal.PackedBuffer,
    down_proj: []const metal.PackedBuffer,

    // Per-layer norm scale buffers (f16, indexed by layer).
    attn_norm: []const metal.HalfBuffer,
    ffn_norm: []const metal.HalfBuffer,
    q_norm: []const metal.HalfBuffer,
    k_norm: []const metal.HalfBuffer,

    // Per-layer KV caches (f16, indexed by layer).
    k_cache: []const metal.HalfBuffer,
    v_cache: []const metal.HalfBuffer,

    // Activation scratch (shared across all layers, f32).
    residual: objc.Object,
    norm_out: objc.Object,
    q: objc.Object,
    k: objc.Object,
    v: objc.Object,
    attn_out: objc.Object,
    proj_out: objc.Object,
    attn_scratch: objc.Object,
    gate: objc.Object,
    up: objc.Object,
    mlp_out: objc.Object,

    // Forward pass I/O.
    token_ids: objc.Object, // u32 buffer — written by CPU.
    logits: objc.Object, // [vocab_size] f32 output.

    // Sequence state.
    token_id: u32,
    position: u32,
};

/// Encode a full single-token decode pass into one compute
/// command encoder:
///   embedding lookup → N × forwardBlock → final RMSNorm →
///   LM head QMV → [vocab_size] f32 logits.
///
/// Writes the token ID into the `token_ids` buffer (CPU →
/// shared memory) before any GPU work is dispatched.  The
/// caller must not read `logits` until the command buffer
/// has completed (Rule 26).
///
/// Dispatch count: 1 + N×18 + 2 (embedding + N blocks ×
/// 18 kernels + final norm + LM head).
/// For Bonsai1_7B: 1 + 28×18 + 2 = 507.
pub fn forwardDecode(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    args: ForwardDecodeArgs,
) void {
    comptime {
        std.debug.assert(Config.num_layers > 0);
        std.debug.assert(Config.vocab_size > 0);
        std.debug.assert(Config.hidden_size > 0);
    }
    std.debug.assert(
        args.position < Config.max_context_length,
    );
    std.debug.assert(
        args.q_proj.len == Config.num_layers,
    );
    std.debug.assert(
        args.k_cache.len == Config.num_layers,
    );

    writeTokenId(args.token_ids, args.token_id);
    encodeEmbeddingLookup(
        Config,
        device,
        encoder,
        pipelines,
        args,
    );
    bufferBarrier(encoder);
    encodeAllBlocks(Config, device, encoder, pipelines, args);
    encodeFinalNormAndLMHead(
        Config,
        device,
        encoder,
        pipelines,
        args,
    );
}

/// Write a single u32 token ID into position 0 of the
/// token_ids buffer.  Safe before any GPU work is
/// dispatched on the same command buffer.
fn writeTokenId(
    token_ids_buf: objc.Object,
    token_id: u32,
) void {
    std.debug.assert(token_ids_buf.value != null);
    const ptr = token_ids_buf.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const u32_ptr: [*]u32 = @ptrCast(@alignCast(ptr));
    u32_ptr[0] = token_id;
}

/// Dispatch 1-bit embedding lookup from a PackedBuffer.
/// Binds the same MTLBuffer at two offsets: bits at 0,
/// scales at scaleOffset.  Output goes to the residual
/// buffer (the block chain's running state).
fn encodeEmbeddingLookup(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    args: ForwardDecodeArgs,
) void {
    std.debug.assert(args.embedding.packed_count > 0);
    std.debug.assert(args.token_ids.value != null);

    setRawBuffer(encoder, args.token_ids, 0, 0);
    setRawBuffer(encoder, args.embedding.obj, 0, 1);
    setRawBuffer(
        encoder,
        args.embedding.obj,
        args.embedding.scaleOffset(),
        2,
    );
    setRawBuffer(encoder, args.residual, 0, 3);
    metal.setBytes(encoder, EmbedDims, &.{
        .vocab_size = Config.vocab_size,
        .hidden_size = Config.hidden_size,
        .num_tokens = 1,
        .group_size = Config.group_size,
    }, 4);

    // One token × hidden_size output elements.
    device.dispatch1D(
        encoder,
        pipelines.embedding_lookup,
        Config.hidden_size,
    );
}

/// Encode all N decoder blocks sequentially.  Each block
/// ends with a bufferBarrier (inside forwardBlock), so the
/// residual output of block i is visible to block i+1
/// without additional barriers here.
fn encodeAllBlocks(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    args: ForwardDecodeArgs,
) void {
    std.debug.assert(
        args.q_proj.len == Config.num_layers,
    );
    std.debug.assert(args.position + 1 > 0);

    const seq_len: u32 = args.position + 1;
    for (0..Config.num_layers) |i| {
        const idx: u32 = @intCast(i);
        const block_args = blockArgsFromDecode(
            args,
            idx,
            seq_len,
        );
        forwardBlock(
            Config,
            device,
            encoder,
            pipelines,
            block_args,
        );
    }
}

/// Encode final RMSNorm on the residual, then dispatch
/// the LM head QMV to produce [vocab_size] f32 logits.
fn encodeFinalNormAndLMHead(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    args: ForwardDecodeArgs,
) void {
    std.debug.assert(args.lm_head.packed_count > 0);
    std.debug.assert(args.final_norm_scale.value != null);

    dispatchRMSNorm(
        device,
        encoder,
        pipelines.rms_norm,
        args.residual,
        args.final_norm_scale,
        args.norm_out,
        .{
            .hidden_size = Config.hidden_size,
            .num_tokens = 1,
            .eps = 1e-6,
        },
    );
    bufferBarrier(encoder);
    dispatchQMV(
        device,
        encoder,
        device.qmv,
        args.lm_head,
        args.norm_out,
        args.logits,
        .{
            .M = Config.vocab_size,
            .K = Config.hidden_size,
            .group_size = Config.group_size,
        },
    );
}

/// Construct a `ForwardBlockArgs` for one layer by indexing
/// into the per-layer slices of `ForwardDecodeArgs`.
/// Activation scratch buffers are shared across all layers.
fn blockArgsFromDecode(
    args: ForwardDecodeArgs,
    layer_index: u32,
    seq_len: u32,
) ForwardBlockArgs {
    std.debug.assert(layer_index < args.q_proj.len);
    std.debug.assert(seq_len > 0);

    const i = layer_index;
    return .{
        .q_proj = args.q_proj[i],
        .k_proj = args.k_proj[i],
        .v_proj = args.v_proj[i],
        .o_proj = args.o_proj[i],
        .gate_proj = args.gate_proj[i],
        .up_proj = args.up_proj[i],
        .down_proj = args.down_proj[i],
        .attn_norm_scale = args.attn_norm[i].obj,
        .ffn_norm_scale = args.ffn_norm[i].obj,
        .q_norm_scale = args.q_norm[i].obj,
        .k_norm_scale = args.k_norm[i].obj,
        .residual = args.residual,
        .norm_out = args.norm_out,
        .q = args.q,
        .k = args.k,
        .v = args.v,
        .attn_out = args.attn_out,
        .proj_out = args.proj_out,
        .attn_scratch = args.attn_scratch,
        .gate = args.gate,
        .up = args.up,
        .mlp_out = args.mlp_out,
        .k_cache = args.k_cache[i].obj,
        .v_cache = args.v_cache[i].obj,
        .position = args.position,
        .seq_len = seq_len,
    };
}

/// Insert a buffer memory barrier on the compute encoder.
/// Ensures preceding buffer writes are visible to subsequent
/// reads within the same encoder.
fn bufferBarrier(encoder: objc.Object) void {
    std.debug.assert(encoder.value != null);
    // MTLBarrierScope.buffers = 1 (NSUInteger).
    encoder.msgSend(
        void,
        "memoryBarrierWithScope:",
        .{@as(c_ulong, 1)},
    );
}

/// Encode the attention half of a decoder block:
/// RMSNorm → Q/K/V proj → RoPE → KV cache → GQA
/// attention → O proj → residual add.
fn encodeAttentionHalf(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    a: ForwardBlockArgs,
) void {
    encodeAttentionProjections(
        Config,
        device,
        encoder,
        pipelines,
        a,
    );
    encodeAttentionGather(
        Config,
        device,
        encoder,
        pipelines,
        a,
    );
}

/// Encode RMSNorm → Q/K/V projections → QK norms.
/// RoPE and KV cache update follow in a separate helper
/// to stay within the 70-line function limit.
fn encodeAttentionProjections(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    a: ForwardBlockArgs,
) void {
    const kv_qmv = QMVDims{
        .M = Config.kv_dim,
        .K = Config.hidden_size,
        .group_size = Config.group_size,
    };
    dispatchRMSNorm(
        device,
        encoder,
        pipelines.rms_norm,
        a.residual,
        a.attn_norm_scale,
        a.norm_out,
        .{
            .hidden_size = Config.hidden_size,
            .num_tokens = 1,
            .eps = 1e-6,
        },
    );
    bufferBarrier(encoder);
    dispatchQMV(device, encoder, device.qmv, a.q_proj, a.norm_out, a.q, .{
        .M = Config.query_dim,
        .K = Config.hidden_size,
        .group_size = Config.group_size,
    });
    dispatchQMV(device, encoder, device.qmv, a.k_proj, a.norm_out, a.k, kv_qmv);
    dispatchQMV(device, encoder, device.qmv, a.v_proj, a.norm_out, a.v, kv_qmv);
    bufferBarrier(encoder);
    // QK norms: per-head RMSNorm on Q and K before RoPE.
    // Reuses rms_norm with hidden_size = head_dim, treating
    // each head as a separate "token" row.
    dispatchRMSNorm(device, encoder, pipelines.rms_norm, a.q, a.q_norm_scale, a.q, .{
        .hidden_size = Config.head_dim,
        .num_tokens = Config.num_query_heads,
        .eps = 1e-6,
    });
    dispatchRMSNorm(device, encoder, pipelines.rms_norm, a.k, a.k_norm_scale, a.k, .{
        .hidden_size = Config.head_dim,
        .num_tokens = Config.num_kv_heads,
        .eps = 1e-6,
    });
    bufferBarrier(encoder);
    encodeRoPEAndKVCache(
        Config,
        device,
        encoder,
        pipelines,
        a,
    );
}

/// Encode RoPE on Q and K, then write K/V to the cache.
fn encodeRoPEAndKVCache(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    a: ForwardBlockArgs,
) void {
    std.debug.assert(
        a.position < Config.max_context_length,
    );
    std.debug.assert(Config.head_dim % 2 == 0);
    dispatchRoPE(device, encoder, pipelines.rope, a.q, .{
        .num_heads = Config.num_query_heads,
        .head_dim = Config.head_dim,
        .position = a.position,
        .rope_theta = Config.rope_theta,
    });
    dispatchRoPE(device, encoder, pipelines.rope, a.k, .{
        .num_heads = Config.num_kv_heads,
        .head_dim = Config.head_dim,
        .position = a.position,
        .rope_theta = Config.rope_theta,
    });
    bufferBarrier(encoder);
    dispatchKVCacheUpdate(
        device,
        encoder,
        pipelines.kv_cache_update,
        a.k,
        a.v,
        a.k_cache,
        a.v_cache,
        .{
            .num_kv_heads = Config.num_kv_heads,
            .head_dim = Config.head_dim,
            .position = a.position,
            .max_context_length = Config.max_context_length,
        },
    );
    bufferBarrier(encoder);
}

/// Encode GQA attention → O projection → residual add.
/// Reads from Q buffer and KV cache written by
/// `encodeAttentionProjections`.
fn encodeAttentionGather(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    a: ForwardBlockArgs,
) void {
    dispatchGQAAttention(
        device,
        encoder,
        pipelines.gqa_attention,
        a.q,
        a.k_cache,
        a.v_cache,
        a.attn_out,
        a.attn_scratch,
        .{
            .num_query_heads = Config.num_query_heads,
            .num_kv_heads = Config.num_kv_heads,
            .head_dim = Config.head_dim,
            .seq_len = a.seq_len,
            .max_context_length = Config.max_context_length,
            .heads_per_kv_group = Config.heads_per_kv_group,
        },
    );
    bufferBarrier(encoder);
    dispatchQMV(
        device,
        encoder,
        device.qmv,
        a.o_proj,
        a.attn_out,
        a.proj_out,
        .{
            .M = Config.hidden_size,
            .K = Config.query_dim,
            .group_size = Config.group_size,
        },
    );
    bufferBarrier(encoder);
    dispatchResidualAdd(
        device,
        encoder,
        pipelines.residual_add,
        a.residual,
        a.proj_out,
        Config.hidden_size,
    );
    bufferBarrier(encoder);
}

/// Encode the MLP half of a decoder block:
/// RMSNorm → gate/up proj → SiLU⊙mul → down proj →
/// residual add.
fn encodeMLPHalf(
    comptime Config: type,
    device: *const metal.Device,
    encoder: objc.Object,
    pipelines: *const TransformerPipelines,
    a: ForwardBlockArgs,
) void {
    const mlp_qmv = QMVDims{
        .M = Config.intermediate_size,
        .K = Config.hidden_size,
        .group_size = Config.group_size,
    };
    dispatchRMSNorm(
        device,
        encoder,
        pipelines.rms_norm,
        a.residual,
        a.ffn_norm_scale,
        a.norm_out,
        .{
            .hidden_size = Config.hidden_size,
            .num_tokens = 1,
            .eps = 1e-6,
        },
    );
    bufferBarrier(encoder);
    // Gate and up projections share the same input and dims.
    // Use the fused pair kernel when K meets qmv_fast
    // requirements — saves one full dispatch overhead.
    const can_fuse = comptime (Config.hidden_size % Config.group_size == 0) and
        (Config.hidden_size <= 6144) and
        (Config.hidden_size >= 256) and
        (Config.hidden_size / 32 <= Config.group_size);
    if (can_fuse) {
        dispatchQMVFusedPair(
            device,
            encoder,
            a.gate_proj,
            a.up_proj,
            a.norm_out,
            a.gate,
            a.up,
            mlp_qmv,
        );
    } else {
        dispatchQMV(device, encoder, device.qmv, a.gate_proj, a.norm_out, a.gate, mlp_qmv);
        dispatchQMV(device, encoder, device.qmv, a.up_proj, a.norm_out, a.up, mlp_qmv);
    }
    bufferBarrier(encoder);
    dispatchSiLUElementwiseMul(
        device,
        encoder,
        pipelines.silu_elementwise_mul,
        a.gate,
        a.up,
        a.mlp_out,
        Config.intermediate_size,
    );
    bufferBarrier(encoder);
    dispatchQMV(device, encoder, device.qmv, a.down_proj, a.mlp_out, a.proj_out, .{
        .M = Config.hidden_size,
        .K = Config.intermediate_size,
        .group_size = Config.group_size,
    });
    bufferBarrier(encoder);
    dispatchResidualAdd(
        device,
        encoder,
        pipelines.residual_add,
        a.residual,
        a.proj_out,
        Config.hidden_size,
    );
    bufferBarrier(encoder);
}

// ============================================================================
// Sampling and autoregressive generation (Step 2f)
// ============================================================================

/// Parameters controlling token sampling from logits.
/// temperature=0 selects greedy (argmax).  Set top_k=0 and
/// top_p=1.0 to disable filtering.
pub const SamplingParams = struct {
    temperature: f32 = 0.0,
    top_k: u32 = 0,
    top_p: f32 = 1.0,
    seed: u64 = 42,
};

/// Timing and count results from a generate call.
pub const GenerateResult = struct {
    tokens_generated: u32,
    prefill_ns: u64,
    decode_ns: u64,
};

/// Options for the autoregressive generation loop.
/// Bundled into a struct to avoid mixing up slice parameters
/// at the call site (Rule 19).
pub const GenerateOpts = struct {
    prompt_ids: []const u32,
    params: SamplingParams,
    eos_ids: []const u32,
    output_tokens: []u32,
    /// Scratch buffer for top-k/top-p sorting.  Length must
    /// be >= vocab_size.  May be empty for greedy decoding.
    scratch: []f32,
    /// Index scratch for top-p sorting.  Same length
    /// requirement as `scratch`.
    indices: []u32,
};

/// Find the index of the maximum value in a logits slice.
/// Returns 0 for a single-element slice.
pub fn argmax(logits: []const f32) u32 {
    std.debug.assert(logits.len > 0);
    std.debug.assert(logits.len <= 0xFFFF_FFFF);

    var best_idx: u32 = 0;
    var best_val: f32 = logits[0];
    for (logits[1..], 1..) |val, i| {
        if (val > best_val) {
            best_val = val;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

/// Apply temperature scaling to logits in place.
/// Divides each logit by `temperature`.  Caller must ensure
/// temperature > 0 — use argmax directly for greedy decoding.
fn applyTemperature(
    logits: []f32,
    temperature: f32,
) void {
    std.debug.assert(temperature > 0.0);
    std.debug.assert(logits.len > 0);

    const inv_t: f32 = 1.0 / temperature;
    for (logits) |*l| {
        l.* *= inv_t;
    }
}

/// Softmax in place: exp(x - max) / sum(exp(x - max)).
/// Converts logits to a probability distribution.
pub fn softmaxInPlace(logits: []f32) void {
    std.debug.assert(logits.len > 0);

    var max_val: f32 = logits[0];
    for (logits[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    var sum: f32 = 0.0;
    for (logits) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    std.debug.assert(sum > 0.0);
    const inv_sum: f32 = 1.0 / sum;
    for (logits) |*v| {
        v.* *= inv_sum;
    }
}

/// Keep only the top-k largest logits; set the rest to
/// negative infinity.  Requires a scratch buffer of length
/// >= logits.len for sorting a copy.
fn applyTopK(
    logits: []f32,
    k: u32,
    scratch: []f32,
) void {
    std.debug.assert(k > 0);
    std.debug.assert(logits.len > 0);
    std.debug.assert(scratch.len >= logits.len);
    if (k >= logits.len) return; // Nothing to filter.

    // Copy logits into scratch and sort descending to find
    // the k-th largest value as a threshold.
    @memcpy(scratch[0..logits.len], logits);
    std.sort.pdq(
        f32,
        scratch[0..logits.len],
        {},
        sortF32Desc,
    );

    const threshold: f32 = scratch[k - 1];
    const neg_inf: f32 = -std.math.inf(f32);
    for (logits) |*l| {
        if (l.* < threshold) l.* = neg_inf;
    }
}

/// Keep only tokens whose cumulative probability mass does
/// not exceed `p` (nucleus sampling).  Applies a temporary
/// softmax to scratch to determine the probability cutoff,
/// then filters logits below the threshold.
fn applyTopP(
    logits: []f32,
    p: f32,
    scratch: []f32,
    indices: []u32,
) void {
    std.debug.assert(p > 0.0);
    std.debug.assert(p <= 1.0);
    std.debug.assert(logits.len > 0);
    std.debug.assert(scratch.len >= logits.len);
    std.debug.assert(indices.len >= logits.len);
    const len: u32 = @intCast(logits.len);

    // Build softmax probabilities in scratch.
    @memcpy(scratch[0..logits.len], logits);
    softmaxInPlace(scratch[0..logits.len]);

    // Initialize index array and sort by probability
    // descending.
    for (0..len) |i| {
        indices[i] = @intCast(i);
    }
    const ctx = SortByValueCtx{
        .values = scratch[0..logits.len],
    };
    std.sort.pdq(
        u32,
        indices[0..len],
        ctx,
        SortByValueCtx.descending,
    );

    // Walk sorted indices, accumulating probability mass.
    // Once cumulative >= p, filter all remaining tokens.
    const neg_inf: f32 = -std.math.inf(f32);
    var cumulative: f32 = 0.0;
    var threshold_reached = false;
    for (indices[0..len]) |idx| {
        if (threshold_reached) {
            logits[idx] = neg_inf;
        } else {
            cumulative += scratch[idx];
            if (cumulative >= p) {
                threshold_reached = true;
            }
        }
    }
}

/// Context for sorting u32 indices by their corresponding
/// f32 values in a separate slice.
const SortByValueCtx = struct {
    values: []const f32,

    fn descending(
        ctx: SortByValueCtx,
        a: u32,
        b: u32,
    ) bool {
        return ctx.values[a] > ctx.values[b];
    }
};

/// Compare f32 descending for pdq sort.
fn sortF32Desc(_: void, a: f32, b: f32) bool {
    return a > b;
}

/// Sample a token from logits using the given parameters.
/// Modifies the logits slice in place (temperature, top-k,
/// top-p filtering).  For greedy (temperature=0), returns
/// argmax without modifying logits.
pub fn sampleToken(
    logits: []f32,
    params: SamplingParams,
    scratch: []f32,
    indices: []u32,
    rng: std.Random,
) u32 {
    std.debug.assert(logits.len > 0);

    // Greedy: just return the argmax.
    if (params.temperature == 0.0) {
        return argmax(logits);
    }

    std.debug.assert(params.temperature > 0.0);

    // Temperature scaling.
    applyTemperature(logits, params.temperature);

    // Top-k filtering.
    if (params.top_k > 0) {
        applyTopK(logits, params.top_k, scratch);
    }

    // Top-p (nucleus) filtering.
    if (params.top_p < 1.0) {
        applyTopP(
            logits,
            params.top_p,
            scratch,
            indices,
        );
    }

    // Convert to probabilities and sample.
    softmaxInPlace(logits);
    return sampleFromDistribution(logits, rng);
}

/// Draw a token from a probability distribution using the
/// inverse-CDF method.  `probs` must sum to ~1.0.
fn sampleFromDistribution(
    probs: []const f32,
    rng: std.Random,
) u32 {
    std.debug.assert(probs.len > 0);

    const r: f32 = rng.float(f32);
    var cumulative: f32 = 0.0;
    for (probs, 0..) |p, i| {
        cumulative += p;
        if (cumulative > r) return @intCast(i);
    }
    // Rounding tail: return last token.
    return @intCast(probs.len - 1);
}

/// Read f32 logits from a raw Metal buffer handle.
/// The caller must ensure no GPU work is in flight on
/// this buffer (Rule 26: commitAndWait before reading).
fn readLogitsSlice(
    logits_obj: objc.Object,
    count: u32,
) []f32 {
    std.debug.assert(logits_obj.value != null);
    std.debug.assert(count > 0);

    const ptr = logits_obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const float_ptr: [*]f32 = @ptrCast(
        @alignCast(ptr),
    );
    return float_ptr[0..@as(usize, count)];
}

/// Run the autoregressive generation loop: chunked prefill
/// followed by single-token decode.  Returns timing info
/// and the number of tokens generated.
///
/// `opts.output_tokens` receives the generated token IDs
/// (not including the prompt).  Its length caps the maximum
/// number of generated tokens.
///
/// Chunked prefill processes prompt tokens one at a time
/// through forwardDecode (correctness-first; true batched
/// prefill with qmm is a Step 3 optimisation).
pub fn generate(
    comptime Config: type,
    device: *const metal.Device,
    pipelines: *const TransformerPipelines,
    args: *ForwardDecodeArgs,
    opts: GenerateOpts,
) GenerateResult {
    std.debug.assert(opts.prompt_ids.len > 0);
    std.debug.assert(opts.output_tokens.len > 0);
    std.debug.assert(
        opts.prompt_ids.len < Config.max_context_length,
    );

    var prng = std.Random.DefaultPrng.init(
        opts.params.seed,
    );
    const rng = prng.random();

    // Chunked prefill: one token at a time.
    const t0 = std.time.nanoTimestamp();
    chunkedPrefill(
        Config,
        device,
        pipelines,
        args,
        opts.prompt_ids,
    );
    const t1 = std.time.nanoTimestamp();

    // Sample first token from prefill logits.
    const logits = readLogitsSlice(
        args.logits,
        Config.vocab_size,
    );
    var next_token = sampleToken(
        logits,
        opts.params,
        opts.scratch,
        opts.indices,
        rng,
    );

    // Decode loop: sample, store, advance.
    var position: u32 = @intCast(opts.prompt_ids.len);
    var count: u32 = 0;

    while (count < opts.output_tokens.len) {
        if (isEosToken(next_token, opts.eos_ids)) break;
        if (position >= Config.max_context_length) break;

        opts.output_tokens[count] = next_token;
        count += 1;

        next_token = decodeOneToken(
            Config,
            device,
            pipelines,
            args,
            next_token,
            position,
            opts.params,
            opts.scratch,
            opts.indices,
            rng,
        );
        position += 1;
    }
    const t2 = std.time.nanoTimestamp();

    return .{
        .tokens_generated = count,
        .prefill_ns = @intCast(t1 - t0),
        .decode_ns = @intCast(t2 - t1),
    };
}

/// Process all prompt tokens through forwardDecode, one at
/// a time (chunked prefill).  Advances the KV cache so that
/// position = prompt_ids.len after completion.
fn chunkedPrefill(
    comptime Config: type,
    device: *const metal.Device,
    pipelines: *const TransformerPipelines,
    args: *ForwardDecodeArgs,
    prompt_ids: []const u32,
) void {
    std.debug.assert(prompt_ids.len > 0);
    std.debug.assert(
        prompt_ids.len < Config.max_context_length,
    );

    for (prompt_ids, 0..) |token_id, i| {
        args.token_id = token_id;
        args.position = @intCast(i);

        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);
        forwardDecode(
            Config,
            device,
            enc,
            pipelines,
            args.*,
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }
}

/// Run one decode step: forwardDecode + commitAndWait +
/// sample from logits.  Returns the sampled token ID.
fn decodeOneToken(
    comptime Config: type,
    device: *const metal.Device,
    pipelines: *const TransformerPipelines,
    args: *ForwardDecodeArgs,
    token_id: u32,
    position: u32,
    params: SamplingParams,
    scratch: []f32,
    indices: []u32,
    rng: std.Random,
) u32 {
    std.debug.assert(
        position < Config.max_context_length,
    );
    std.debug.assert(token_id < Config.vocab_size);

    args.token_id = token_id;
    args.position = position;

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    forwardDecode(
        Config,
        device,
        enc,
        pipelines,
        args.*,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    const logits = readLogitsSlice(
        args.logits,
        Config.vocab_size,
    );
    return sampleToken(
        logits,
        params,
        scratch,
        indices,
        rng,
    );
}

/// Check whether a token ID is in the EOS set.
fn isEosToken(
    token_id: u32,
    eos_ids: []const u32,
) bool {
    std.debug.assert(eos_ids.len > 0);
    for (eos_ids) |eos| {
        if (token_id == eos) return true;
    }
    return false;
}

// ============================================================================
// Shader compilation (private)
// ============================================================================

fn compileTransformerLibrary(
    device: objc.Object,
) !objc.Object {
    const source = @embedFile("shaders/transformer.metal");

    // Embedded source must not be empty — a missing or truncated
    // file would compile "successfully" with zero kernels.
    comptime {
        std.debug.assert(source.len > 0);
    }

    const NSString = objc.getClass("NSString") orelse
        return error.ClassNotFound;
    const source_ns = NSString.msgSend(
        objc.Object,
        "stringWithUTF8String:",
        .{@as([*:0]const u8, source.ptr)},
    );

    var error_ptr: ?*anyopaque = null;
    const lib_raw = device.msgSend(
        ?*anyopaque,
        "newLibraryWithSource:options:error:",
        .{
            source_ns.value,
            @as(?*anyopaque, null),
            &error_ptr,
        },
    ) orelse {
        if (error_ptr) |err| {
            const err_obj = objc.Object.fromId(err);
            const desc = err_obj.msgSend(
                objc.Object,
                "localizedDescription",
                .{},
            );
            const c_str = desc.msgSend(
                [*:0]const u8,
                "UTF8String",
                .{},
            );
            log.err(
                "Transformer shader compilation failed: {s}",
                .{c_str},
            );
        }
        return error.MetalShaderCompilationFailed;
    };

    return objc.Object.fromId(lib_raw);
}

// ============================================================================
// CPU reference implementations (for tests)
// ============================================================================

/// CPU RMSNorm: output = input * rsqrt(mean(input^2) + eps) * scale.
fn cpuRMSNorm(
    input: []const f32,
    scale: []const f16,
    output: []f32,
    hidden_size: u32,
    num_tokens: u32,
    eps: f32,
) void {
    std.debug.assert(
        input.len >= num_tokens * hidden_size,
    );
    std.debug.assert(
        output.len >= num_tokens * hidden_size,
    );

    const h: usize = hidden_size;
    for (0..num_tokens) |t| {
        const off = t * h;
        var sum_sq: f32 = 0.0;
        for (0..h) |i| {
            const val = input[off + i];
            sum_sq += val * val;
        }
        const rms = 1.0 / @sqrt(
            sum_sq / @as(f32, @floatFromInt(h)) + eps,
        );
        for (0..h) |i| {
            const s: f32 = @floatCast(scale[i]);
            output[off + i] = input[off + i] * rms * s;
        }
    }
}

/// CPU SiLU: output = input * sigmoid(input).
fn cpuSiLU(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    for (input, output) |x, *o| {
        o.* = x / (1.0 + @exp(-x));
    }
}

/// CPU fused SiLU + mul: output = silu(gate) * up.
fn cpuSiLUElementwiseMul(
    gate: []const f32,
    up: []const f32,
    output: []f32,
) void {
    std.debug.assert(gate.len == up.len);
    std.debug.assert(gate.len == output.len);
    for (gate, up, output) |g, u, *o| {
        const sigmoid_g = 1.0 / (1.0 + @exp(-g));
        o.* = g * sigmoid_g * u;
    }
}

/// CPU RoPE: apply rotary position embeddings in-place.
fn cpuRoPE(
    data: []f32,
    num_heads: u32,
    head_dim: u32,
    position: u32,
    rope_theta: f32,
) void {
    std.debug.assert(data.len >= num_heads * head_dim);
    std.debug.assert(head_dim % 2 == 0);

    const half_dim = head_dim / 2;
    for (0..num_heads) |h| {
        for (0..half_dim) |p| {
            const exponent: f32 =
                @as(f32, @floatFromInt(2 * p)) / @as(f32, @floatFromInt(head_dim));
            const theta_i = 1.0 / std.math.pow(f32, rope_theta, exponent);
            const angle =
                @as(f32, @floatFromInt(position)) * theta_i;
            const cos_val = @cos(angle);
            const sin_val = @sin(angle);

            const base = h * head_dim + p * 2;
            const x0 = data[base];
            const x1 = data[base + 1];
            data[base] = x0 * cos_val - x1 * sin_val;
            data[base + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// CPU softmax (in-place) over the first `len` elements.
fn cpuSoftmax(scores: []f32, len: usize) void {
    std.debug.assert(len > 0);
    std.debug.assert(scores.len >= len);

    var max_val: f32 = scores[0];
    for (scores[1..len]) |s| {
        if (s > max_val) max_val = s;
    }

    var sum: f32 = 0.0;
    for (scores[0..len]) |*s| {
        s.* = @exp(s.* - max_val);
        sum += s.*;
    }
    for (scores[0..len]) |*s| {
        s.* /= sum;
    }
}

/// CPU attention for a single query head.
fn cpuAttentionHead(
    q_head: []const f32,
    k_cache_f16: [*]const f16,
    v_cache_f16: [*]const f16,
    output_head: []f32,
    kv_head: u32,
    head_dim: u32,
    seq_len: u32,
    max_ctx: u32,
) void {
    std.debug.assert(q_head.len == head_dim);
    std.debug.assert(output_head.len == head_dim);
    std.debug.assert(seq_len <= 64); // Stack array limit.

    const hd: usize = head_dim;
    const inv_sqrt_d = 1.0 / @sqrt(
        @as(f32, @floatFromInt(head_dim)),
    );
    const kv_stride: usize = max_ctx * hd;

    // Compute scaled dot-product attention scores.
    var scores: [64]f32 = undefined;
    for (0..seq_len) |t| {
        var dot: f32 = 0.0;
        for (0..hd) |d| {
            const k_val: f32 = @floatCast(
                k_cache_f16[kv_head * kv_stride + t * hd + d],
            );
            dot += q_head[d] * k_val;
        }
        scores[t] = dot * inv_sqrt_d;
    }

    // Softmax over valid positions.
    cpuSoftmax(&scores, seq_len);

    // Weighted sum of V.
    for (0..hd) |d| {
        var accum: f32 = 0.0;
        for (0..seq_len) |t| {
            const v_val: f32 = @floatCast(
                v_cache_f16[kv_head * kv_stride + t * hd + d],
            );
            accum += scores[t] * v_val;
        }
        output_head[d] = accum;
    }
}

/// CPU grouped query attention — dispatches per-head.
fn cpuGQAAttention(
    q: []const f32,
    k_cache_f16: [*]const f16,
    v_cache_f16: [*]const f16,
    output: []f32,
    num_query_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_ctx: u32,
    heads_per_kv_group: u32,
) void {
    std.debug.assert(
        q.len >= num_query_heads * head_dim,
    );
    std.debug.assert(
        output.len >= num_query_heads * head_dim,
    );

    const hd: usize = head_dim;
    for (0..num_query_heads) |h| {
        const kv_head: u32 =
            @as(u32, @intCast(h)) / heads_per_kv_group;
        cpuAttentionHead(
            q[h * hd ..][0..hd],
            k_cache_f16,
            v_cache_f16,
            output[h * hd ..][0..hd],
            kv_head,
            head_dim,
            seq_len,
            max_ctx,
        );
    }
}

/// CPU 1-bit embedding dequantize + gather.
fn cpuEmbeddingDequant(
    token_ids: []const u32,
    packed_bits: []const u8,
    scales_f16: [*]const f16,
    output: []f32,
    hidden_size: u32,
    group_size_param: u32,
) void {
    std.debug.assert(
        output.len >= token_ids.len * hidden_size,
    );

    const hs: usize = hidden_size;
    for (token_ids, 0..) |row, t| {
        for (0..hs) |d| {
            const flat_bit: usize = row * hs + d;
            const byte_idx = flat_bit / 8;
            const bit_pos: u3 =
                @intCast(flat_bit % 8);
            const bit_set =
                (packed_bits[byte_idx] >> bit_pos) & 1;
            const g_idx = flat_bit / group_size_param;
            const scale: f32 =
                @floatCast(scales_f16[g_idx]);
            output[t * hs + d] =
                if (bit_set == 1) scale else -scale;
        }
    }
}

/// CPU 1-bit matrix-vector multiply: output[row] = dot(W[row], input).
/// W is [M × K] in Q1_0_g128 packed format (row-major).
/// Uses the same dequantization as cpuEmbeddingDequant: bit=1 → +scale,
/// bit=0 → −scale, then dot product with the input vector.
fn cpuQMV(
    packed_bits: []const u8,
    scales_f16: []const f16,
    input: []const f32,
    output: []f32,
    M: u32,
    K: u32,
    group_size_param: u32,
) void {
    std.debug.assert(input.len >= K);
    std.debug.assert(output.len >= M);
    std.debug.assert(packed_bits.len >= (M * K + 7) / 8);
    std.debug.assert(group_size_param > 0);

    const k: usize = K;
    const gs: usize = group_size_param;

    for (0..M) |row| {
        var accum: f32 = 0.0;
        for (0..k) |col| {
            const flat_bit: usize = row * k + col;
            const byte_idx = flat_bit / 8;
            const bit_pos: u3 = @intCast(flat_bit % 8);
            const bit_set =
                (packed_bits[byte_idx] >> bit_pos) & 1;
            const g_idx = flat_bit / gs;
            const scale: f32 = @floatCast(scales_f16[g_idx]);
            const weight: f32 =
                if (bit_set == 1) scale else -scale;
            accum += weight * input[col];
        }
        output[row] = accum;
    }
}

// ============================================================================
// CPU block forward (Step 1.5 — test reference)
// ============================================================================

/// Packed weight slices for one decoder block (CPU side).
/// Each pair (bits, scales) comes from a PackedBuffer's
/// underlying Metal shared memory.
const BlockWeightSlices = struct {
    q_bits: []const u8,
    q_scales: []const f16,
    k_bits: []const u8,
    k_scales: []const f16,
    v_bits: []const u8,
    v_scales: []const f16,
    o_bits: []const u8,
    o_scales: []const f16,
    gate_bits: []const u8,
    gate_scales: []const f16,
    up_bits: []const u8,
    up_scales: []const f16,
    down_bits: []const u8,
    down_scales: []const f16,
    attn_norm: []const f16,
    ffn_norm: []const f16,
    q_norm: []const f16,
    k_norm: []const f16,
};

/// CPU reference for one full decoder block.  Dispatches
/// to the attention and MLP halves sequentially.
fn cpuForwardBlock(
    comptime Config: type,
    residual: []f32,
    weights: *const BlockWeightSlices,
    k_cache: []f16,
    v_cache: []f16,
    position: u32,
    seq_len: u32,
) void {
    std.debug.assert(residual.len >= Config.hidden_size);
    std.debug.assert(seq_len > 0);

    cpuForwardBlockAttention(
        Config,
        residual,
        weights,
        k_cache,
        v_cache,
        position,
        seq_len,
    );
    cpuForwardBlockMLP(Config, residual, weights);
}

/// CPU attention half: RMSNorm → Q/K/V proj → RoPE →
/// KV cache update → GQA attention → O proj → residual add.
fn cpuForwardBlockAttention(
    comptime Config: type,
    residual: []f32,
    w: *const BlockWeightSlices,
    k_cache: []f16,
    v_cache: []f16,
    position: u32,
    seq_len: u32,
) void {
    const QD = Config.query_dim;

    // RMSNorm → Q/K/V → RoPE → KV cache.
    var q: [QD]f32 = undefined;
    cpuAttentionProjections(
        Config,
        residual,
        w,
        k_cache,
        v_cache,
        &q,
        position,
    );

    // GQA attention → O projection → residual add.
    cpuAttentionGather(
        Config,
        residual,
        w,
        &q,
        k_cache,
        v_cache,
        seq_len,
    );
}

/// CPU RMSNorm → Q/K/V projections → QK norms → RoPE →
/// KV cache update.  Writes the rotated Q vector to
/// `q_out` for use by `cpuAttentionGather`.
fn cpuAttentionProjections(
    comptime Config: type,
    residual: []const f32,
    w: *const BlockWeightSlices,
    k_cache: []f16,
    v_cache: []f16,
    q_out: []f32,
    position: u32,
) void {
    const H = Config.hidden_size;
    const KVD = Config.kv_dim;
    const HD = Config.head_dim;

    // RMSNorm.
    var norm_out: [H]f32 = undefined;
    cpuRMSNorm(
        residual[0..H],
        w.attn_norm,
        &norm_out,
        H,
        1,
        1e-6,
    );

    // Q/K/V projections.
    var k: [KVD]f32 = undefined;
    var v_proj: [KVD]f32 = undefined;
    cpuProjectQKV(Config, &norm_out, w, q_out, &k, &v_proj);

    // QK norms: per-head RMSNorm before RoPE.
    cpuRMSNorm(
        q_out,
        w.q_norm,
        q_out,
        HD,
        Config.num_query_heads,
        1e-6,
    );
    cpuRMSNorm(
        &k,
        w.k_norm,
        &k,
        HD,
        Config.num_kv_heads,
        1e-6,
    );

    // RoPE and KV cache update.
    cpuRoPEAndKVCache(
        Config,
        q_out,
        &k,
        &v_proj,
        k_cache,
        v_cache,
        position,
    );
}

/// CPU RoPE on Q and K, then write K/V to cache.
/// Separated from `cpuAttentionProjections` to keep
/// function bodies under 70 lines (Rule 5).
fn cpuRoPEAndKVCache(
    comptime Config: type,
    q_out: []f32,
    k: []f32,
    v_proj: []const f32,
    k_cache: []f16,
    v_cache: []f16,
    position: u32,
) void {
    const HD = Config.head_dim;
    const MAX_CTX = Config.max_context_length;

    std.debug.assert(q_out.len >= Config.query_dim);
    std.debug.assert(k.len >= Config.kv_dim);

    cpuRoPE(
        q_out,
        Config.num_query_heads,
        HD,
        position,
        Config.rope_theta,
    );
    cpuRoPE(
        k,
        Config.num_kv_heads,
        HD,
        position,
        Config.rope_theta,
    );

    // KV cache update (f32 → f16).
    cpuKVCacheWrite(
        k,
        v_proj,
        k_cache,
        v_cache,
        Config.num_kv_heads,
        HD,
        MAX_CTX,
        position,
    );
}

/// CPU Q/K/V projections: three 1-bit matrix-vector multiplies
/// from the same norm output.  Separated from the parent to
/// keep function bodies under 70 lines (Rule 5).
fn cpuProjectQKV(
    comptime Config: type,
    norm_out: []const f32,
    w: *const BlockWeightSlices,
    q_out: []f32,
    k_out: []f32,
    v_out: []f32,
) void {
    const H = Config.hidden_size;
    const QD = Config.query_dim;
    const KVD = Config.kv_dim;
    const GS = Config.group_size;

    cpuQMV(
        w.q_bits,
        w.q_scales,
        norm_out,
        q_out[0..QD],
        QD,
        H,
        GS,
    );
    cpuQMV(
        w.k_bits,
        w.k_scales,
        norm_out,
        k_out[0..KVD],
        KVD,
        H,
        GS,
    );
    cpuQMV(
        w.v_bits,
        w.v_scales,
        norm_out,
        v_out[0..KVD],
        KVD,
        H,
        GS,
    );
}

/// CPU KV cache write: convert f32 K/V projections to f16
/// and store at the given position in the cache.
fn cpuKVCacheWrite(
    k_proj: []const f32,
    v_proj: []const f32,
    k_cache: []f16,
    v_cache: []f16,
    num_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    position: u32,
) void {
    std.debug.assert(k_proj.len >= num_kv_heads * head_dim);
    std.debug.assert(v_proj.len >= num_kv_heads * head_dim);

    const hd: usize = head_dim;
    for (0..num_kv_heads) |h| {
        for (0..hd) |d| {
            const idx = h * max_ctx * hd + position * hd + d;
            k_cache[idx] = @floatCast(k_proj[h * hd + d]);
            v_cache[idx] = @floatCast(v_proj[h * hd + d]);
        }
    }
}

/// CPU GQA attention → O projection → residual add.
/// Reads Q from `q_in` and K/V from the cache.
fn cpuAttentionGather(
    comptime Config: type,
    residual: []f32,
    w: *const BlockWeightSlices,
    q_in: []const f32,
    k_cache: []const f16,
    v_cache: []const f16,
    seq_len: u32,
) void {
    const H = Config.hidden_size;
    const QD = Config.query_dim;
    const HD = Config.head_dim;
    const GS = Config.group_size;
    const MAX_CTX = Config.max_context_length;

    var attn_out: [QD]f32 = undefined;
    cpuGQAAttention(
        q_in,
        k_cache.ptr,
        v_cache.ptr,
        &attn_out,
        Config.num_query_heads,
        HD,
        seq_len,
        MAX_CTX,
        Config.heads_per_kv_group,
    );

    // O projection.
    var o_out: [H]f32 = undefined;
    cpuQMV(
        w.o_bits,
        w.o_scales,
        &attn_out,
        &o_out,
        H,
        QD,
        GS,
    );

    // Residual add.
    for (residual[0..H], o_out[0..H]) |*r, o| {
        r.* += o;
    }
}

/// CPU MLP half: RMSNorm → gate/up proj → SiLU⊙mul →
/// down proj → residual add.
fn cpuForwardBlockMLP(
    comptime Config: type,
    residual: []f32,
    w: *const BlockWeightSlices,
) void {
    const H = Config.hidden_size;
    const I = Config.intermediate_size;
    const GS = Config.group_size;

    // RMSNorm.
    var norm_out: [H]f32 = undefined;
    cpuRMSNorm(
        residual[0..H],
        w.ffn_norm,
        &norm_out,
        H,
        1,
        1e-6,
    );

    // Gate and up projections.
    var gate_out: [I]f32 = undefined;
    cpuQMV(
        w.gate_bits,
        w.gate_scales,
        &norm_out,
        &gate_out,
        I,
        H,
        GS,
    );
    var up_out: [I]f32 = undefined;
    cpuQMV(
        w.up_bits,
        w.up_scales,
        &norm_out,
        &up_out,
        I,
        H,
        GS,
    );

    // Fused SiLU + elementwise multiply.
    var mlp_hidden: [I]f32 = undefined;
    cpuSiLUElementwiseMul(&gate_out, &up_out, &mlp_hidden);

    // Down projection.
    var down_out: [H]f32 = undefined;
    cpuQMV(
        w.down_bits,
        w.down_scales,
        &mlp_hidden,
        &down_out,
        H,
        I,
        GS,
    );

    // Residual add.
    for (residual[0..H], down_out[0..H]) |*r, d| {
        r.* += d;
    }
}

// ============================================================================
// Test helpers
// ============================================================================

/// Compare two f32 slices element-by-element.  Prints the first
/// mismatched element's index, GPU value, CPU value, and error
/// before failing the test.
fn expectClose(
    gpu: []const f32,
    cpu: []const f32,
    tolerance: f32,
    label: []const u8,
) !void {
    try std.testing.expectEqual(gpu.len, cpu.len);
    for (gpu, cpu, 0..) |g, c, i| {
        const err = @abs(g - c);
        if (err > tolerance) {
            std.debug.print(
                "{s}[{d}]: gpu={d:.6} cpu={d:.6}" ++ " err={d:.8}\n",
                .{ label, i, g, c, err },
            );
        }
        try std.testing.expect(err <= tolerance);
    }
}

/// Get a typed f16 slice from a HalfBuffer's underlying Metal
/// shared memory.  Valid only while the buffer is live and no
/// GPU operation is in flight.
fn halfBufferSlice(buf: metal.HalfBuffer) []f16 {
    const raw = buf.obj.msgSend(
        [*]u8,
        "contents",
        .{},
    );
    const aligned: [*]align(@alignOf(f16)) u8 =
        @alignCast(raw);
    const ptr: [*]f16 = @ptrCast(aligned);
    return ptr[0..buf.len];
}

/// Create a raw Metal shared buffer and copy byte data into it.
/// Returns the MTLBuffer as an `objc.Object`.  The caller is
/// responsible for releasing it (or letting it leak in tests).
fn createRawBuffer(
    device_obj: objc.Object,
    bytes: []const u8,
) objc.Object {
    std.debug.assert(bytes.len > 0);
    const buf = device_obj.msgSend(
        objc.Object,
        "newBufferWithLength:options:",
        .{
            @as(c_ulong, bytes.len),
            metal.MTLResourceOptions.storage_shared,
        },
    );
    const raw = buf.msgSend([*]u8, "contents", .{});
    @memcpy(raw[0..bytes.len], bytes);
    return buf;
}

/// Create a Metal shared buffer from a u32 slice.
fn createU32Buffer(
    device_obj: objc.Object,
    data: []const u32,
) objc.Object {
    return createRawBuffer(
        device_obj,
        std.mem.sliceAsBytes(data),
    );
}

/// Create a Metal shared buffer from an f16 slice.
fn createF16Buffer(
    device_obj: objc.Object,
    data: []const f16,
) objc.Object {
    return createRawBuffer(
        device_obj,
        std.mem.sliceAsBytes(data),
    );
}

/// Verify a KV cache slice: correct values at the written
/// position, zeros everywhere else.
fn verifyKVCacheSlice(
    cache_f16: []const f16,
    proj_f32: []const f32,
    kv_heads: u32,
    hd: u32,
    max_ctx: u32,
    pos: u32,
) !void {
    std.debug.assert(
        cache_f16.len >= kv_heads * max_ctx * hd,
    );
    std.debug.assert(proj_f32.len >= kv_heads * hd);

    for (0..kv_heads) |head| {
        for (0..max_ctx) |t| {
            for (0..hd) |d| {
                const idx =
                    head * max_ctx * hd + t * hd + d;
                if (t == pos) {
                    // Written position: f32→f16→f32.
                    const proj_idx = head * hd + d;
                    const expected: f32 = @floatCast(
                        @as(f16, @floatCast(
                            proj_f32[proj_idx],
                        )),
                    );
                    const actual: f32 =
                        @floatCast(cache_f16[idx]);
                    try std.testing.expectApproxEqAbs(
                        expected,
                        actual,
                        1e-3,
                    );
                } else {
                    // Unwritten: must be zero.
                    try std.testing.expectEqual(
                        @as(f16, 0.0),
                        cache_f16[idx],
                    );
                }
            }
        }
    }
}

/// Populate KV cache positions 0..seq_len with deterministic
/// test values.  Fills both K and V caches.
fn populateTestKVCache(
    k_f16: []f16,
    v_f16: []f16,
    num_kvh: u32,
    seq_len: u32,
    hd: u32,
    max_ctx: u32,
) void {
    std.debug.assert(
        k_f16.len >= num_kvh * max_ctx * hd,
    );
    std.debug.assert(
        v_f16.len >= num_kvh * max_ctx * hd,
    );

    @memset(k_f16, @as(f16, 0.0));
    @memset(v_f16, @as(f16, 0.0));

    for (0..num_kvh) |h| {
        for (0..seq_len) |t| {
            for (0..hd) |d| {
                const idx =
                    h * max_ctx * hd + t * hd + d;
                const seed: f32 = @floatFromInt(
                    h * seq_len * hd + t * hd + d + 1,
                );
                k_f16[idx] = @floatCast(seed * 0.05);
                v_f16[idx] = @floatCast(seed * 0.025);
            }
        }
    }
}

/// Allocate a PackedBuffer and fill it with a deterministic
/// bit pattern and scale values derived from `seed`.  Used
/// to create reproducible weight matrices for integration tests.
fn createTestPackedBuffer(
    device_obj: objc.Object,
    num_weights: u32,
    seed: u8,
) !metal.PackedBuffer {
    std.debug.assert(num_weights > 0);

    var buf = try metal.PackedBuffer.init(
        device_obj,
        num_weights,
        128,
    );

    const raw: [*]u8 = @ptrCast(
        buf.obj.msgSend(*anyopaque, "contents", .{}),
    );
    const packed_len = buf.packedBytes();

    // Fill bits with a repeating pattern derived from seed.
    // Using modulo 37 (prime) to avoid alignment artifacts.
    for (0..packed_len) |i| {
        raw[i] = seed +% @as(u8, @intCast(i % 37));
    }

    // Fill scales with small varied values.
    const num_groups = buf.numGroups();
    const scale_ptr: [*]f16 = @ptrCast(
        @alignCast(raw + packed_len),
    );
    for (0..num_groups) |i| {
        const val: f32 = 0.1 +
            @as(f32, @floatFromInt(seed % 10)) * 0.01 +
            @as(f32, @floatFromInt(i)) * 0.005;
        scale_ptr[i] = @floatCast(val);
    }

    return buf;
}

/// Get the packed bits slice from a PackedBuffer's Metal
/// shared memory.  Valid only while no GPU write is in flight.
fn packedBufBits(buf: metal.PackedBuffer) []const u8 {
    const raw: [*]const u8 = @ptrCast(
        buf.obj.msgSend(*anyopaque, "contents", .{}),
    );
    return raw[0..buf.packedBytes()];
}

/// Get the f16 scale slice from a PackedBuffer's Metal
/// shared memory.  Scales start at byte offset packedBytes().
fn packedBufScales(
    buf: metal.PackedBuffer,
) []const f16 {
    const raw: [*]u8 = @ptrCast(
        buf.obj.msgSend(*anyopaque, "contents", .{}),
    );
    const offset = buf.packedBytes();
    const ptr: [*]const f16 = @ptrCast(
        @alignCast(raw + offset),
    );
    return ptr[0..buf.numGroups()];
}

// ============================================================================
// Tests
// ============================================================================

test "TransformerConfig comptime validation" {
    // Goal: Verify all three model configs compile and produce
    // expected derived values.  No GPU needed — pure comptime.

    // 1.7B: hidden_size == query_dim (16 × 128 = 2048).
    try std.testing.expectEqual(
        @as(u32, 2048),
        Bonsai1_7B.query_dim,
    );
    try std.testing.expectEqual(
        @as(u32, 1024),
        Bonsai1_7B.kv_dim,
    );
    try std.testing.expectEqual(
        @as(u32, 2),
        Bonsai1_7B.heads_per_kv_group,
    );
    // Tied embeddings → lm_head_bytes is zero.
    try std.testing.expectEqual(
        @as(u32, 0),
        Bonsai1_7B.lm_head_bytes,
    );
    try std.testing.expect(
        Bonsai1_7B.total_weight_bytes > 0,
    );

    // 4B: hidden_size (2560) != query_dim (32 × 128 = 4096).
    // This validates the relaxed assertion.
    try std.testing.expectEqual(
        @as(u32, 4096),
        Bonsai4B.query_dim,
    );
    try std.testing.expectEqual(
        @as(u32, 2560),
        Bonsai4B.hidden_size,
    );
    try std.testing.expectEqual(
        @as(u32, 4),
        Bonsai4B.heads_per_kv_group,
    );

    // 8B: not tied → lm_head_bytes > 0.
    try std.testing.expect(Bonsai8B.lm_head_bytes > 0);

    // KV cache sanity: 1.7B should be < 8 GB.
    const kv_mb = Bonsai1_7B.total_kv_cache_bytes / (1024 * 1024);
    try std.testing.expect(kv_mb > 0);
    try std.testing.expect(kv_mb < 8192);
}

test "rms_norm matches CPU reference" {
    // Goal: Verify GPU RMSNorm produces the same output as a
    // sequential CPU reference, within f16 scale precision.
    // Method: 2 tokens × 8 hidden, known input and scale values.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const hidden: u32 = 8;
    const tokens: u32 = 2;
    const count = tokens * hidden;

    // Input: sequential [0.1, 0.2, ..., 1.6].
    var input_buf = try metal.Buffer.init(device.obj, count);
    defer input_buf.deinit();

    for (input_buf.asSlice(), 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }

    // Scale: 8 f16 values with varied magnitudes.
    const scale_vals = [_]f32{
        1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 1.25, 3.0,
    };
    var scale_buf = try metal.HalfBuffer.init(
        device.obj,
        hidden,
    );
    defer scale_buf.deinit();

    const scale_f16 = halfBufferSlice(scale_buf);
    for (scale_f16, scale_vals[0..]) |*v, sv| {
        v.* = @floatCast(sv);
    }

    var output_buf = try metal.Buffer.init(
        device.obj,
        count,
    );
    defer output_buf.deinit();

    // Dispatch GPU kernel.
    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchRMSNorm(
        &device,
        enc,
        pipelines.rms_norm,
        input_buf.obj,
        scale_buf.obj,
        output_buf.obj,
        .{
            .hidden_size = hidden,
            .num_tokens = tokens,
            .eps = 1e-6,
        },
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // CPU reference.
    var cpu_output: [count]f32 = undefined;
    cpuRMSNorm(
        input_buf.asSlice(),
        scale_f16,
        &cpu_output,
        hidden,
        tokens,
        1e-6,
    );

    try expectClose(
        output_buf.asSlice(),
        &cpu_output,
        1e-3,
        "rms_norm",
    );
}

test "silu matches CPU reference" {
    // Goal: Verify silu(x) = x * sigmoid(x) against CPU reference.
    // Method: 8 values spanning negative, zero, and positive.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const count: u32 = 8;
    const vals = [_]f32{
        -3.0, -1.5, -0.5, 0.0, 0.25, 1.0, 2.0, 4.0,
    };

    var input_buf = try metal.Buffer.init(device.obj, count);
    defer input_buf.deinit();
    @memcpy(input_buf.asSlice(), &vals);

    var output_buf = try metal.Buffer.init(device.obj, count);
    defer output_buf.deinit();

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchSiLU(
        &device,
        enc,
        pipelines.silu,
        input_buf.obj,
        output_buf.obj,
        count,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    var cpu_out: [count]f32 = undefined;
    cpuSiLU(&vals, &cpu_out);

    try expectClose(
        output_buf.asSlice(),
        &cpu_out,
        1e-5,
        "silu",
    );
}

test "silu_elementwise_mul matches CPU reference" {
    // Goal: Verify fused silu(gate) * up matches CPU reference.
    // Method: 8 gate/up pairs with varied magnitudes.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const count: u32 = 8;
    const gate_vals = [_]f32{
        -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0,
    };
    const up_vals = [_]f32{
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    };

    var gate_buf = try metal.Buffer.init(device.obj, count);
    defer gate_buf.deinit();
    @memcpy(gate_buf.asSlice(), &gate_vals);

    var up_buf = try metal.Buffer.init(device.obj, count);
    defer up_buf.deinit();
    @memcpy(up_buf.asSlice(), &up_vals);

    var output_buf = try metal.Buffer.init(device.obj, count);
    defer output_buf.deinit();

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchSiLUElementwiseMul(
        &device,
        enc,
        pipelines.silu_elementwise_mul,
        gate_buf.obj,
        up_buf.obj,
        output_buf.obj,
        count,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    var cpu_out: [count]f32 = undefined;
    cpuSiLUElementwiseMul(&gate_vals, &up_vals, &cpu_out);

    try expectClose(
        output_buf.asSlice(),
        &cpu_out,
        1e-5,
        "silu_elementwise_mul",
    );
}

test "rope matches CPU reference" {
    // Goal: Verify RoPE rotation matches CPU sin/cos reference.
    // Method: 2 heads × 8 dim, position=3, theta=10000.
    // In-place modification — GPU and CPU start from same data.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const num_heads: u32 = 2;
    const head_dim: u32 = 8;
    const position: u32 = 3;
    const theta: f32 = 10000.0;
    const count = num_heads * head_dim;

    // Sequential input values.
    var vals: [count]f32 = undefined;
    for (&vals, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }

    // CPU reference (operates on a copy of the same data).
    var cpu_out: [count]f32 = vals;
    cpuRoPE(&cpu_out, num_heads, head_dim, position, theta);

    // GPU: in-place modification.
    var data_buf = try metal.Buffer.init(device.obj, count);
    defer data_buf.deinit();
    @memcpy(data_buf.asSlice(), &vals);

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchRoPE(
        &device,
        enc,
        pipelines.rope,
        data_buf.obj,
        .{
            .num_heads = num_heads,
            .head_dim = head_dim,
            .position = position,
            .rope_theta = theta,
        },
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    try expectClose(
        data_buf.asSlice(),
        &cpu_out,
        1e-4,
        "rope",
    );
}

test "kv_cache_update writes correct values" {
    // Goal: Verify KV cache update writes f32→f16 at the target
    // position and leaves other positions unchanged (zeros).
    // Method: 2 KV heads × 4 dim, max_ctx=8, write at pos=2.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const kv_heads: u32 = 2;
    const hd: u32 = 4;
    const max_ctx: u32 = 8;
    const pos: u32 = 2;
    const proj_count = kv_heads * hd;
    const cache_count = kv_heads * max_ctx * hd;

    // K/V projections: known f32 values.
    var k_proj = try metal.Buffer.init(
        device.obj,
        proj_count,
    );
    defer k_proj.deinit();

    var v_proj = try metal.Buffer.init(
        device.obj,
        proj_count,
    );
    defer v_proj.deinit();

    for (k_proj.asSlice(), 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 0.5;
    }
    for (v_proj.asSlice(), 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 0.25;
    }

    // KV caches: zero-initialised f16.
    var k_cache = try metal.HalfBuffer.init(
        device.obj,
        cache_count,
    );
    defer k_cache.deinit();

    var v_cache = try metal.HalfBuffer.init(
        device.obj,
        cache_count,
    );
    defer v_cache.deinit();

    @memset(halfBufferSlice(k_cache), @as(f16, 0.0));
    @memset(halfBufferSlice(v_cache), @as(f16, 0.0));

    // Dispatch.
    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchKVCacheUpdate(
        &device,
        enc,
        pipelines.kv_cache_update,
        k_proj.obj,
        v_proj.obj,
        k_cache.obj,
        v_cache.obj,
        .{
            .num_kv_heads = kv_heads,
            .head_dim = hd,
            .position = pos,
            .max_context_length = max_ctx,
        },
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // Verify K cache.
    try verifyKVCacheSlice(
        halfBufferSlice(k_cache),
        k_proj.asSlice(),
        kv_heads,
        hd,
        max_ctx,
        pos,
    );

    // Verify V cache.
    try verifyKVCacheSlice(
        halfBufferSlice(v_cache),
        v_proj.asSlice(),
        kv_heads,
        hd,
        max_ctx,
        pos,
    );
}

test "gqa_attention matches CPU reference" {
    // Goal: Verify grouped query attention with 4 query heads,
    // 2 KV heads, head_dim=8, seq_len=4 against a sequential
    // CPU reference.  Tests the h / heads_per_kv_group mapping.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const num_qh: u32 = 4;
    const num_kvh: u32 = 2;
    const hd: u32 = 8;
    const seq_len: u32 = 4;
    const max_ctx: u32 = 16;
    const hpkg: u32 = num_qh / num_kvh;
    const q_count = num_qh * hd;
    const cache_count = num_kvh * max_ctx * hd;
    const scratch_count = num_qh * max_ctx;

    // Q: sequential values, one row per query head.
    var q_buf = try metal.Buffer.init(device.obj, q_count);
    defer q_buf.deinit();

    for (q_buf.asSlice(), 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }

    // KV caches: deterministic test pattern.
    var k_cache = try metal.HalfBuffer.init(
        device.obj,
        cache_count,
    );
    defer k_cache.deinit();

    var v_cache = try metal.HalfBuffer.init(
        device.obj,
        cache_count,
    );
    defer v_cache.deinit();

    populateTestKVCache(
        halfBufferSlice(k_cache),
        halfBufferSlice(v_cache),
        num_kvh,
        seq_len,
        hd,
        max_ctx,
    );

    // Output + scratch.
    var out_buf = try metal.Buffer.init(
        device.obj,
        q_count,
    );
    defer out_buf.deinit();

    var scratch_buf = try metal.Buffer.init(
        device.obj,
        scratch_count,
    );
    defer scratch_buf.deinit();

    // Dispatch GPU.
    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchGQAAttention(
        &device,
        enc,
        pipelines.gqa_attention,
        q_buf.obj,
        k_cache.obj,
        v_cache.obj,
        out_buf.obj,
        scratch_buf.obj,
        .{
            .num_query_heads = num_qh,
            .num_kv_heads = num_kvh,
            .head_dim = hd,
            .seq_len = seq_len,
            .max_context_length = max_ctx,
            .heads_per_kv_group = hpkg,
        },
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // CPU reference.
    const k_f16 = halfBufferSlice(k_cache);
    const v_f16 = halfBufferSlice(v_cache);
    var cpu_out: [q_count]f32 = undefined;
    cpuGQAAttention(
        q_buf.asSlice(),
        k_f16.ptr,
        v_f16.ptr,
        &cpu_out,
        num_qh,
        hd,
        seq_len,
        max_ctx,
        hpkg,
    );

    try expectClose(
        out_buf.asSlice(),
        &cpu_out,
        1e-3,
        "gqa_attention",
    );
}

test "embedding_lookup matches CPU dequant" {
    // Goal: Verify 1-bit embedding gather against CPU dequant.
    // Method: 4 vocab × 16 hidden (64 elements, < one 128 group),
    // look up tokens [1, 3].

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const vocab: u32 = 4;
    const hidden: u32 = 16;
    const num_tok: u32 = 2;
    const gs: u32 = 128;
    // 4 vocab × 16 hidden = 64 weights.
    // packed_byte_count = ceil(64 / 8) = 8.
    // num_groups = ceil(64 / 128) = 1.
    const packed_byte_count = 8;
    const num_groups = 1;

    // Packed bits: 0xAA = 10101010 pattern.
    var bits: [packed_byte_count]u8 = undefined;
    for (&bits) |*b| {
        b.* = 0xAA;
    }

    // One scale group, scale = 0.5.
    var scale_data = [_]f16{@as(f16, 0.5)} ** num_groups;

    // Token IDs: look up rows 1 and 3.
    const token_ids = [_]u32{ 1, 3 };

    // Create Metal buffers.
    const ids_buf = createU32Buffer(device.obj, &token_ids);
    defer ids_buf.msgSend(void, "release", .{});

    const bits_buf = createRawBuffer(device.obj, &bits);
    defer bits_buf.msgSend(void, "release", .{});

    const scale_buf = createF16Buffer(
        device.obj,
        &scale_data,
    );
    defer scale_buf.msgSend(void, "release", .{});

    const out_count = num_tok * hidden;
    var output_buf = try metal.Buffer.init(
        device.obj,
        out_count,
    );
    defer output_buf.deinit();

    // Dispatch.
    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchEmbeddingLookup(
        &device,
        enc,
        pipelines.embedding_lookup,
        ids_buf,
        bits_buf,
        scale_buf,
        output_buf.obj,
        .{
            .vocab_size = vocab,
            .hidden_size = hidden,
            .num_tokens = num_tok,
            .group_size = gs,
        },
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // CPU reference.
    var cpu_out: [out_count]f32 = undefined;
    cpuEmbeddingDequant(
        &token_ids,
        &bits,
        @as([*]const f16, &scale_data),
        &cpu_out,
        hidden,
        gs,
    );

    try expectClose(
        output_buf.asSlice(),
        &cpu_out,
        1e-3,
        "embedding_lookup",
    );
}

test "residual_add matches CPU reference" {
    // Goal: Verify in-place residual[i] += addition[i].
    // Method: 8-element vectors with known values.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const count: u32 = 8;
    const residual_vals = [_]f32{
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    };
    const addition_vals = [_]f32{
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    };

    var residual_buf = try metal.Buffer.init(
        device.obj,
        count,
    );
    defer residual_buf.deinit();
    @memcpy(residual_buf.asSlice(), &residual_vals);

    var addition_buf = try metal.Buffer.init(
        device.obj,
        count,
    );
    defer addition_buf.deinit();
    @memcpy(addition_buf.asSlice(), &addition_vals);

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchResidualAdd(
        &device,
        enc,
        pipelines.residual_add,
        residual_buf.obj,
        addition_buf.obj,
        count,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // CPU reference: simple element-wise addition.
    var expected: [count]f32 = undefined;
    for (
        &expected,
        residual_vals[0..],
        addition_vals[0..],
    ) |*e, r, a| {
        e.* = r + a;
    }

    try expectClose(
        residual_buf.asSlice(),
        &expected,
        1e-6,
        "residual_add",
    );
}

test "single-block forward matches CPU reference" {
    // Goal: Verify that all transformer kernels compose correctly
    // through one full decoder block (attention + MLP), catching
    // buffer wiring bugs, residual-add mistakes, and f16/f32
    // promotion mismatches that hide between individual tests.
    //
    // Method: TinyBlock config (2 heads, 64 hidden, 128 inter-
    // mediate), deterministic weights and scales, pre-filled KV
    // cache for 3 positions, process position 3 on GPU, compare
    // against sequential CPU reference.  Tolerance < 1e-3.

    const H = TinyBlock.hidden_size; // 64
    const QD = TinyBlock.query_dim; // 64
    const KVD = TinyBlock.kv_dim; // 64
    const I = TinyBlock.intermediate_size; // 128
    const HD = TinyBlock.head_dim; // 32
    const NQH = TinyBlock.num_query_heads; // 2
    const NKVH = TinyBlock.num_kv_heads; // 2
    const MAX_CTX = TinyBlock.max_context_length; // 8

    // Comptime sanity: qmv requires K % 32 == 0.
    comptime {
        std.debug.assert(H % 32 == 0);
        std.debug.assert(I % 32 == 0);
    }

    // ── Device and pipeline setup ────────────────────
    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    // ── Packed weight buffers (7 projections) ────────────
    // Each seed byte creates a distinct bit pattern so that
    // a buffer-binding mismatch produces a visible error.
    var q_packed = try createTestPackedBuffer(
        device.obj,
        QD * H,
        0xA5,
    );
    defer q_packed.deinit();

    var k_packed = try createTestPackedBuffer(
        device.obj,
        KVD * H,
        0x5A,
    );
    defer k_packed.deinit();

    var v_packed = try createTestPackedBuffer(
        device.obj,
        KVD * H,
        0x33,
    );
    defer v_packed.deinit();

    var o_packed = try createTestPackedBuffer(
        device.obj,
        H * QD,
        0xCC,
    );
    defer o_packed.deinit();

    var gate_packed = try createTestPackedBuffer(
        device.obj,
        I * H,
        0x55,
    );
    defer gate_packed.deinit();

    var up_packed = try createTestPackedBuffer(
        device.obj,
        I * H,
        0xAA,
    );
    defer up_packed.deinit();

    var down_packed = try createTestPackedBuffer(
        device.obj,
        H * I,
        0x69,
    );
    defer down_packed.deinit();

    // ── Norm scale buffers (f16) ───────────────────
    var attn_norm_buf = try metal.HalfBuffer.init(
        device.obj,
        H,
    );
    defer attn_norm_buf.deinit();

    var ffn_norm_buf = try metal.HalfBuffer.init(
        device.obj,
        H,
    );
    defer ffn_norm_buf.deinit();

    const attn_norm_f16 = halfBufferSlice(attn_norm_buf);
    for (attn_norm_f16, 0..) |*v, i| {
        v.* = @floatCast(
            @as(f32, 0.5) +
                @as(f32, @floatFromInt(i)) * 0.01,
        );
    }
    const ffn_norm_f16 = halfBufferSlice(ffn_norm_buf);
    for (ffn_norm_f16, 0..) |*v, i| {
        v.* = @floatCast(
            @as(f32, 0.8) +
                @as(f32, @floatFromInt(i)) * 0.005,
        );
    }

    // QK norm scale buffers (f16, head_dim elements each).
    var q_norm_scale_buf = try metal.HalfBuffer.init(
        device.obj,
        HD,
    );
    defer q_norm_scale_buf.deinit();

    var k_norm_scale_buf = try metal.HalfBuffer.init(
        device.obj,
        HD,
    );
    defer k_norm_scale_buf.deinit();

    const q_norm_f16 = halfBufferSlice(q_norm_scale_buf);
    for (q_norm_f16, 0..) |*v, i| {
        v.* = @floatCast(
            @as(f32, 0.9) +
                @as(f32, @floatFromInt(i)) * 0.003,
        );
    }
    const k_norm_f16 = halfBufferSlice(k_norm_scale_buf);
    for (k_norm_f16, 0..) |*v, i| {
        v.* = @floatCast(
            @as(f32, 0.85) +
                @as(f32, @floatFromInt(i)) * 0.004,
        );
    }

    // ── KV cache (f16) — pre-fill positions 0..2 ────────
    const kv_elems = NKVH * MAX_CTX * HD;
    var k_cache_buf = try metal.HalfBuffer.init(
        device.obj,
        kv_elems,
    );
    defer k_cache_buf.deinit();

    var v_cache_buf = try metal.HalfBuffer.init(
        device.obj,
        kv_elems,
    );
    defer v_cache_buf.deinit();

    const k_cache_f16 = halfBufferSlice(k_cache_buf);
    const v_cache_f16 = halfBufferSlice(v_cache_buf);
    populateTestKVCache(
        k_cache_f16,
        v_cache_f16,
        NKVH,
        3,
        HD,
        MAX_CTX,
    );

    // CPU-side copy of pre-filled KV cache.
    var cpu_k_cache: [kv_elems]f16 = undefined;
    var cpu_v_cache: [kv_elems]f16 = undefined;
    @memcpy(&cpu_k_cache, k_cache_f16);
    @memcpy(&cpu_v_cache, v_cache_f16);

    // ── Activation scratch buffers (f32) ─────────────────
    var residual_buf = try metal.Buffer.init(
        device.obj,
        H,
    );
    defer residual_buf.deinit();

    var norm_out_buf = try metal.Buffer.init(
        device.obj,
        H,
    );
    defer norm_out_buf.deinit();

    var q_buf = try metal.Buffer.init(device.obj, QD);
    defer q_buf.deinit();

    var k_buf = try metal.Buffer.init(device.obj, KVD);
    defer k_buf.deinit();

    var v_buf = try metal.Buffer.init(device.obj, KVD);
    defer v_buf.deinit();

    var attn_out_buf = try metal.Buffer.init(
        device.obj,
        QD,
    );
    defer attn_out_buf.deinit();

    var proj_out_buf = try metal.Buffer.init(
        device.obj,
        H,
    );
    defer proj_out_buf.deinit();

    var scratch_buf = try metal.Buffer.init(
        device.obj,
        NQH * MAX_CTX,
    );
    defer scratch_buf.deinit();

    var gate_buf = try metal.Buffer.init(device.obj, I);
    defer gate_buf.deinit();

    var up_buf = try metal.Buffer.init(device.obj, I);
    defer up_buf.deinit();

    var mlp_out_buf = try metal.Buffer.init(
        device.obj,
        I,
    );
    defer mlp_out_buf.deinit();

    // ── Fill residual with deterministic values ──────────
    for (residual_buf.asSlice(), 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 0.05;
    }
    var cpu_residual: [H]f32 = undefined;
    @memcpy(&cpu_residual, residual_buf.asSlice());

    // ── GPU: encode and run forwardBlock ─────────────────
    const position: u32 = 3;
    const seq_len: u32 = position + 1; // 4 valid KV entries.

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    forwardBlock(TinyBlock, &device, enc, &pipelines, .{
        .q_proj = q_packed,
        .k_proj = k_packed,
        .v_proj = v_packed,
        .o_proj = o_packed,
        .gate_proj = gate_packed,
        .up_proj = up_packed,
        .down_proj = down_packed,
        .attn_norm_scale = attn_norm_buf.obj,
        .ffn_norm_scale = ffn_norm_buf.obj,
        .q_norm_scale = q_norm_scale_buf.obj,
        .k_norm_scale = k_norm_scale_buf.obj,
        .residual = residual_buf.obj,
        .norm_out = norm_out_buf.obj,
        .q = q_buf.obj,
        .k = k_buf.obj,
        .v = v_buf.obj,
        .attn_out = attn_out_buf.obj,
        .proj_out = proj_out_buf.obj,
        .attn_scratch = scratch_buf.obj,
        .gate = gate_buf.obj,
        .up = up_buf.obj,
        .mlp_out = mlp_out_buf.obj,
        .k_cache = k_cache_buf.obj,
        .v_cache = v_cache_buf.obj,
        .position = position,
        .seq_len = seq_len,
    });
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // ── CPU reference ────────────────────────
    const weights = BlockWeightSlices{
        .q_bits = packedBufBits(q_packed),
        .q_scales = packedBufScales(q_packed),
        .k_bits = packedBufBits(k_packed),
        .k_scales = packedBufScales(k_packed),
        .v_bits = packedBufBits(v_packed),
        .v_scales = packedBufScales(v_packed),
        .o_bits = packedBufBits(o_packed),
        .o_scales = packedBufScales(o_packed),
        .gate_bits = packedBufBits(gate_packed),
        .gate_scales = packedBufScales(gate_packed),
        .up_bits = packedBufBits(up_packed),
        .up_scales = packedBufScales(up_packed),
        .down_bits = packedBufBits(down_packed),
        .down_scales = packedBufScales(down_packed),
        .attn_norm = attn_norm_f16,
        .ffn_norm = ffn_norm_f16,
        .q_norm = q_norm_f16,
        .k_norm = k_norm_f16,
    };
    cpuForwardBlock(
        TinyBlock,
        &cpu_residual,
        &weights,
        &cpu_k_cache,
        &cpu_v_cache,
        position,
        seq_len,
    );

    // ── Compare GPU vs CPU output ────────────────────
    // Tolerance is 5e-3 (not 1e-3) because errors accumulate
    // through the full block: two RMSNorms, seven qmv projections,
    // RoPE, f32→f16→f32 KV cache round-trip, softmax, and two
    // residual adds.  Relative error is < 1e-6; the absolute error
    // grows with output magnitude (values reach ~6000).
    try expectClose(
        residual_buf.asSlice(),
        &cpu_residual,
        5e-3,
        "single_block_forward",
    );
}

test "rms_norm_heads matches CPU reference" {
    // Goal: Verify per-head RMSNorm dispatched via the
    // rms_norm kernel (hidden_size = head_dim, num_tokens =
    // num_heads) matches the CPU reference.
    //
    // Method: 4 heads × 8 dim, deterministic input and
    // scale, compare GPU vs CPU within 1e-3.

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    const num_heads: u32 = 4;
    const head_dim: u32 = 8;
    const count = num_heads * head_dim;

    // Input: 32 f32 values with varied magnitudes.
    var input_buf = try metal.Buffer.init(
        device.obj,
        count,
    );
    defer input_buf.deinit();

    for (input_buf.asSlice(), 0..) |*v, i| {
        const fi: f32 = @floatFromInt(i);
        v.* = @sin(fi * 0.7) * 2.0 + 0.5;
    }

    // Scale: head_dim f16 values, broadcast across heads.
    var scale_buf = try metal.HalfBuffer.init(
        device.obj,
        head_dim,
    );
    defer scale_buf.deinit();

    const scale_f16 = halfBufferSlice(scale_buf);
    for (scale_f16, 0..) |*v, i| {
        v.* = @floatCast(
            @as(f32, 0.9) +
                @as(f32, @floatFromInt(i)) * 0.03,
        );
    }

    var output_buf = try metal.Buffer.init(
        device.obj,
        count,
    );
    defer output_buf.deinit();

    // GPU: treat each head as a "token" row.
    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchRMSNorm(
        &device,
        enc,
        pipelines.rms_norm,
        input_buf.obj,
        scale_buf.obj,
        output_buf.obj,
        .{
            .hidden_size = head_dim,
            .num_tokens = num_heads,
            .eps = 1e-6,
        },
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // CPU reference with same per-head interpretation.
    var cpu_output: [count]f32 = undefined;
    cpuRMSNorm(
        input_buf.asSlice(),
        scale_f16,
        &cpu_output,
        head_dim,
        num_heads,
        1e-6,
    );

    try expectClose(
        output_buf.asSlice(),
        &cpu_output,
        1e-3,
        "rms_norm_heads",
    );
}

test "forwardDecode full path matches CPU reference" {
    // Goal: Verify that embedding → 1 block → final RMSNorm →
    // LM head composes correctly into a single command buffer
    // and matches a sequential CPU reference.
    //
    // Method: TinyBlock config (1 layer, vocab=32, hidden=128),
    // deterministic weights, single token at position 0.
    // Tolerance < 1e-2 (accumulated rounding over the full path).

    const V = TinyBlock.vocab_size; // 32
    const H = TinyBlock.hidden_size; // 128
    const QD = TinyBlock.query_dim; // 128
    const KVD = TinyBlock.kv_dim; // 128
    const I = TinyBlock.intermediate_size; // 256
    const HD = TinyBlock.head_dim; // 64
    const NQH = TinyBlock.num_query_heads; // 2
    const MAX_CTX = TinyBlock.max_context_length; // 8
    const GS = TinyBlock.group_size; // 128

    comptime {
        std.debug.assert(H % 32 == 0);
        std.debug.assert(I % 32 == 0);
        std.debug.assert(TinyBlock.tie_word_embeddings);
    }

    // ── Device and pipeline setup ────────────────────
    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    // ── Embedding (also serves as LM head when tied) ──

    var emb_packed = try createTestPackedBuffer(
        device.obj,
        V * H,
        0xB7,
    );
    defer emb_packed.deinit();

    // ── Per-layer packed weight buffers (1 layer) ────

    var q_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * QD,
            0xA5,
        ),
    };
    defer q_proj_arr[0].deinit();

    var k_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * KVD,
            0x5A,
        ),
    };
    defer k_proj_arr[0].deinit();

    var v_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * KVD,
            0x33,
        ),
    };
    defer v_proj_arr[0].deinit();

    var o_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            QD * H,
            0xCC,
        ),
    };
    defer o_proj_arr[0].deinit();

    var gate_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * I,
            0x55,
        ),
    };
    defer gate_proj_arr[0].deinit();

    var up_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * I,
            0xAA,
        ),
    };
    defer up_proj_arr[0].deinit();

    var down_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            I * H,
            0x69,
        ),
    };
    defer down_proj_arr[0].deinit();

    // ── Per-layer norm scale buffers (f16) ───────────

    var attn_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, H),
    };
    defer attn_norm_arr[0].deinit();

    var ffn_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, H),
    };
    defer ffn_norm_arr[0].deinit();

    var q_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, HD),
    };
    defer q_norm_arr[0].deinit();

    var k_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, HD),
    };
    defer k_norm_arr[0].deinit();

    // Final norm.

    var final_norm_buf = try metal.HalfBuffer.init(
        device.obj,
        H,
    );
    defer final_norm_buf.deinit();

    // Fill all norm scales with deterministic values.
    for (halfBufferSlice(attn_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.5) +
                @as(f32, @floatFromInt(i)) * 0.01,
        );
    }
    for (halfBufferSlice(ffn_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.8) +
                @as(f32, @floatFromInt(i)) * 0.005,
        );
    }
    for (halfBufferSlice(q_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.6) +
                @as(f32, @floatFromInt(i)) * 0.008,
        );
    }
    for (halfBufferSlice(k_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.7) +
                @as(f32, @floatFromInt(i)) * 0.006,
        );
    }
    const final_norm_f16 = halfBufferSlice(final_norm_buf);
    for (final_norm_f16, 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 1.0) +
                @as(f32, @floatFromInt(i)) * 0.002,
        );
    }

    // ── KV caches (f16, zeroed — no history at pos 0) ──
    const kv_elems: u32 = KVD * MAX_CTX;

    var k_cache_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, kv_elems),
    };
    defer k_cache_arr[0].deinit();

    var v_cache_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, kv_elems),
    };
    defer v_cache_arr[0].deinit();

    // ── Activation scratch buffers (f32) ─────────────

    var residual_buf = try metal.Buffer.init(device.obj, H);
    defer residual_buf.deinit();

    var norm_out_buf = try metal.Buffer.init(
        device.obj,
        H,
    );
    defer norm_out_buf.deinit();

    var q_buf = try metal.Buffer.init(device.obj, QD);
    defer q_buf.deinit();

    var k_buf = try metal.Buffer.init(device.obj, KVD);
    defer k_buf.deinit();

    var v_buf = try metal.Buffer.init(device.obj, KVD);
    defer v_buf.deinit();

    var attn_out_buf = try metal.Buffer.init(
        device.obj,
        QD,
    );
    defer attn_out_buf.deinit();

    var proj_out_buf = try metal.Buffer.init(
        device.obj,
        H,
    );
    defer proj_out_buf.deinit();

    const scratch_len: u32 = NQH * MAX_CTX;

    var attn_scratch_buf = try metal.Buffer.init(
        device.obj,
        scratch_len,
    );
    defer attn_scratch_buf.deinit();

    var gate_buf = try metal.Buffer.init(device.obj, I);
    defer gate_buf.deinit();

    var up_buf = try metal.Buffer.init(device.obj, I);
    defer up_buf.deinit();

    var mlp_out_buf = try metal.Buffer.init(
        device.obj,
        I,
    );
    defer mlp_out_buf.deinit();

    // ── Forward pass I/O buffers ─────────────────────
    // Token IDs: 1 slot (f32/u32 are both 4 bytes).

    var token_ids_buf = try metal.Buffer.init(
        device.obj,
        1,
    );
    defer token_ids_buf.deinit();

    var logits_buf = try metal.Buffer.init(
        device.obj,
        V,
    );
    defer logits_buf.deinit();

    // ── GPU: forwardDecode at position 0, token 1 ────
    const token_id: u32 = 1;
    const position: u32 = 0;
    const seq_len: u32 = position + 1;

    const decode_args = ForwardDecodeArgs{
        .embedding = emb_packed,
        .lm_head = emb_packed, // Tied weights.
        .final_norm_scale = final_norm_buf.obj,
        .q_proj = &q_proj_arr,
        .k_proj = &k_proj_arr,
        .v_proj = &v_proj_arr,
        .o_proj = &o_proj_arr,
        .gate_proj = &gate_proj_arr,
        .up_proj = &up_proj_arr,
        .down_proj = &down_proj_arr,
        .attn_norm = &attn_norm_arr,
        .ffn_norm = &ffn_norm_arr,
        .q_norm = &q_norm_arr,
        .k_norm = &k_norm_arr,
        .k_cache = &k_cache_arr,
        .v_cache = &v_cache_arr,
        .residual = residual_buf.obj,
        .norm_out = norm_out_buf.obj,
        .q = q_buf.obj,
        .k = k_buf.obj,
        .v = v_buf.obj,
        .attn_out = attn_out_buf.obj,
        .proj_out = proj_out_buf.obj,
        .attn_scratch = attn_scratch_buf.obj,
        .gate = gate_buf.obj,
        .up = up_buf.obj,
        .mlp_out = mlp_out_buf.obj,
        .token_ids = token_ids_buf.obj,
        .logits = logits_buf.obj,
        .token_id = token_id,
        .position = position,
    };

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    forwardDecode(
        TinyBlock,
        &device,
        enc,
        &pipelines,
        decode_args,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // ── Verify GPU logits are finite ─────────────────
    const gpu_logits = logits_buf.asSlice();
    std.debug.assert(gpu_logits.len == V);
    for (gpu_logits) |l| {
        try std.testing.expect(!std.math.isNan(l));
        try std.testing.expect(!std.math.isInf(l));
    }

    // ── Verify logits are non-degenerate ─────────────
    // A flat distribution means all logits are identical —
    // the model is not differentiating between tokens.
    var min_val: f32 = gpu_logits[0];
    var max_val: f32 = gpu_logits[0];
    for (gpu_logits) |l| {
        if (l < min_val) min_val = l;
        if (l > max_val) max_val = l;
    }
    try std.testing.expect(max_val - min_val > 0.001);

    // ── CPU reference: embed → block → norm → LM head ──
    const emb_bits = packedBufBits(emb_packed);
    const emb_scales = packedBufScales(emb_packed);

    // Step 1: embedding lookup for token 1.
    var cpu_residual: [H]f32 = undefined;
    cpuEmbeddingDequant(
        &[_]u32{token_id},
        emb_bits,
        emb_scales.ptr,
        &cpu_residual,
        H,
        GS,
    );

    // Step 2: one decoder block.
    const block_w = BlockWeightSlices{
        .q_bits = packedBufBits(q_proj_arr[0]),
        .q_scales = packedBufScales(q_proj_arr[0]),
        .k_bits = packedBufBits(k_proj_arr[0]),
        .k_scales = packedBufScales(k_proj_arr[0]),
        .v_bits = packedBufBits(v_proj_arr[0]),
        .v_scales = packedBufScales(v_proj_arr[0]),
        .o_bits = packedBufBits(o_proj_arr[0]),
        .o_scales = packedBufScales(o_proj_arr[0]),
        .gate_bits = packedBufBits(gate_proj_arr[0]),
        .gate_scales = packedBufScales(gate_proj_arr[0]),
        .up_bits = packedBufBits(up_proj_arr[0]),
        .up_scales = packedBufScales(up_proj_arr[0]),
        .down_bits = packedBufBits(down_proj_arr[0]),
        .down_scales = packedBufScales(down_proj_arr[0]),
        .attn_norm = halfBufferSlice(attn_norm_arr[0]),
        .ffn_norm = halfBufferSlice(ffn_norm_arr[0]),
        .q_norm = halfBufferSlice(q_norm_arr[0]),
        .k_norm = halfBufferSlice(k_norm_arr[0]),
    };

    // CPU gets its own zeroed KV caches (GPU wrote to
    // the Metal buffers during forwardDecode).
    var cpu_k_cache = [_]f16{0} ** (KVD * MAX_CTX);
    var cpu_v_cache = [_]f16{0} ** (KVD * MAX_CTX);

    cpuForwardBlock(
        TinyBlock,
        &cpu_residual,
        &block_w,
        &cpu_k_cache,
        &cpu_v_cache,
        position,
        seq_len,
    );

    // Step 3: final RMSNorm.
    var cpu_norm_out: [H]f32 = undefined;
    cpuRMSNorm(
        &cpu_residual,
        final_norm_f16,
        &cpu_norm_out,
        H,
        1,
        1e-6,
    );

    // Step 4: LM head (qmv with embedding weights).
    var cpu_logits: [V]f32 = undefined;
    cpuQMV(
        emb_bits,
        emb_scales,
        &cpu_norm_out,
        &cpu_logits,
        V,
        H,
        GS,
    );

    // ── Compare GPU vs CPU logits ────────────────────
    // Wider tolerance: accumulated f16 rounding through
    // embedding + full block + final norm + LM head.
    try expectClose(
        gpu_logits,
        &cpu_logits,
        1e-2,
        "forwardDecode",
    );
}

// ============================================================================
// Sampling and generation tests (Step 2f)
// ============================================================================

test "argmax returns index of maximum value" {
    // Goal: argmax on a hand-crafted slice returns the correct
    // index.  Covers single element, first-max, middle-max,
    // last-max, and negative values.

    const single = [_]f32{7.0};
    try std.testing.expectEqual(@as(u32, 0), argmax(&single));

    const first_max = [_]f32{ 9.0, 1.0, 2.0, 3.0 };
    try std.testing.expectEqual(
        @as(u32, 0),
        argmax(&first_max),
    );

    const mid_max = [_]f32{ 1.0, 2.0, 8.0, 3.0 };
    try std.testing.expectEqual(
        @as(u32, 2),
        argmax(&mid_max),
    );

    const last_max = [_]f32{ 1.0, 2.0, 3.0, 10.0 };
    try std.testing.expectEqual(
        @as(u32, 3),
        argmax(&last_max),
    );

    const neg = [_]f32{ -5.0, -2.0, -8.0, -1.0 };
    try std.testing.expectEqual(
        @as(u32, 3),
        argmax(&neg),
    );
}

test "softmaxInPlace produces valid distribution" {
    // Goal: softmax converts logits to probabilities that
    // sum to ~1.0 and preserve the relative ordering.
    //
    // Method: apply softmaxInPlace to a known logit vector,
    // check sum ≈ 1.0 and that the largest logit maps to
    // the largest probability.

    var logits = [_]f32{ 1.0, 3.0, 2.0, 0.5 };
    softmaxInPlace(&logits);

    // All probabilities must be positive.
    for (logits) |p| {
        try std.testing.expect(p > 0.0);
        try std.testing.expect(p <= 1.0);
    }

    // Sum must be approximately 1.0.
    var sum: f32 = 0.0;
    for (logits) |p| sum += p;
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        sum,
        1e-5,
    );

    // Index 1 had the largest logit (3.0) — it should have
    // the highest probability.
    try std.testing.expect(logits[1] > logits[0]);
    try std.testing.expect(logits[1] > logits[2]);
    try std.testing.expect(logits[1] > logits[3]);
}

test "sampleToken greedy returns argmax" {
    // Goal: with temperature=0, sampleToken must return the
    // same index as argmax regardless of the random seed.

    var logits1 = [_]f32{ 0.1, 5.0, 2.0, 3.0 };
    var logits2 = [_]f32{ 0.1, 5.0, 2.0, 3.0 };
    var scratch = [_]f32{0.0} ** 4;
    var indices = [_]u32{0} ** 4;

    const greedy = SamplingParams{
        .temperature = 0.0,
        .top_k = 0,
        .top_p = 1.0,
        .seed = 123,
    };

    var prng = std.Random.DefaultPrng.init(999);
    const result = sampleToken(
        &logits1,
        greedy,
        &scratch,
        &indices,
        prng.random(),
    );
    const expected = argmax(&logits2);

    try std.testing.expectEqual(expected, result);
}

test "applyTemperature scales logits" {
    // Goal: dividing logits by temperature=2.0 halves each
    // value.

    var logits = [_]f32{ 4.0, 8.0, -2.0, 6.0 };
    applyTemperature(&logits, 2.0);

    try std.testing.expectApproxEqAbs(
        @as(f32, 2.0),
        logits[0],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 4.0),
        logits[1],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -1.0),
        logits[2],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 3.0),
        logits[3],
        1e-6,
    );
}

test "applyTopK zeroes out low logits" {
    // Goal: with k=2, only the two largest logits survive;
    // the rest become -inf.

    var logits = [_]f32{ 1.0, 5.0, 3.0, 2.0, 4.0 };
    var scratch = [_]f32{0.0} ** 5;

    applyTopK(&logits, 2, &scratch);

    // The two largest are 5.0 (idx 1) and 4.0 (idx 4).
    try std.testing.expectApproxEqAbs(
        @as(f32, 5.0),
        logits[1],
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 4.0),
        logits[4],
        1e-6,
    );

    // Everything else is -inf.
    const neg_inf: f32 = -std.math.inf(f32);
    try std.testing.expectEqual(neg_inf, logits[0]);
    try std.testing.expectEqual(neg_inf, logits[2]);
    try std.testing.expectEqual(neg_inf, logits[3]);
}

test "applyTopP filters by cumulative probability" {
    // Goal: with p=0.8, tokens are kept until cumulative
    // probability >= 0.8, then remaining tokens are set to
    // -inf.
    //
    // Method: logits are chosen so softmax gives a skewed
    // distribution.  Token 2 (logit=10.0) dominates ~99%
    // of mass, so top-p=0.8 should keep only token 2 and
    // filter the rest.

    var logits = [_]f32{ 0.0, 0.0, 10.0, 0.0 };
    var scratch = [_]f32{0.0} ** 4;
    var indices = [_]u32{0} ** 4;

    applyTopP(&logits, 0.8, &scratch, &indices);

    // Token 2 should survive (it holds > 80% mass).
    try std.testing.expect(logits[2] > -100.0);

    // The other tokens should be filtered to -inf.
    const neg_inf: f32 = -std.math.inf(f32);
    try std.testing.expectEqual(neg_inf, logits[0]);
    try std.testing.expectEqual(neg_inf, logits[1]);
    try std.testing.expectEqual(neg_inf, logits[3]);
}

test "isEosToken detects stop tokens" {
    const eos_ids = [_]u32{ 100, 200, 300 };

    try std.testing.expect(isEosToken(100, &eos_ids));
    try std.testing.expect(isEosToken(200, &eos_ids));
    try std.testing.expect(isEosToken(300, &eos_ids));
    try std.testing.expect(!isEosToken(50, &eos_ids));
    try std.testing.expect(!isEosToken(150, &eos_ids));
}

test "generate loop produces tokens and respects EOS" {
    // Goal: run the full generate loop (chunked prefill +
    // decode) with TinyBlock and verify:
    //   1. At least one token is generated.
    //   2. Generation stops at EOS.
    //   3. Timing values are non-zero.
    //
    // Method: TinyBlock config (1 layer, vocab=32, hidden=128),
    // random weights, greedy decode.  We inject an artificial
    // EOS token — since weights are deterministic, greedy
    // decode will loop on the same token, but we verify the
    // mechanics work.

    const V = TinyBlock.vocab_size; // 32
    const H = TinyBlock.hidden_size; // 128
    const QD = TinyBlock.query_dim; // 128
    const KVD = TinyBlock.kv_dim; // 128
    const I = TinyBlock.intermediate_size; // 256
    const HD = TinyBlock.head_dim; // 64
    const NQH = TinyBlock.num_query_heads; // 2
    const MAX_CTX = TinyBlock.max_context_length; // 8

    comptime {
        std.debug.assert(TinyBlock.tie_word_embeddings);
    }

    // ── Device and pipeline setup ────────────────────
    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    // ── Embedding (also LM head when tied) ───────────

    var emb_packed = try createTestPackedBuffer(
        device.obj,
        V * H,
        0xB7,
    );
    defer emb_packed.deinit();

    // ── Per-layer packed weight buffers ───────────────

    var q_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * QD,
            0xA5,
        ),
    };
    defer q_proj_arr[0].deinit();

    var k_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * KVD,
            0x5A,
        ),
    };
    defer k_proj_arr[0].deinit();

    var v_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * KVD,
            0x33,
        ),
    };
    defer v_proj_arr[0].deinit();

    var o_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            QD * H,
            0xCC,
        ),
    };
    defer o_proj_arr[0].deinit();

    var gate_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * I,
            0x55,
        ),
    };
    defer gate_proj_arr[0].deinit();

    var up_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            H * I,
            0xAA,
        ),
    };
    defer up_proj_arr[0].deinit();

    var down_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(
            device.obj,
            I * H,
            0x69,
        ),
    };
    defer down_proj_arr[0].deinit();

    // ── Per-layer norm scale buffers (f16) ───────────

    var attn_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, H),
    };
    defer attn_norm_arr[0].deinit();

    var ffn_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, H),
    };
    defer ffn_norm_arr[0].deinit();

    var q_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, HD),
    };
    defer q_norm_arr[0].deinit();

    var k_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, HD),
    };
    defer k_norm_arr[0].deinit();

    var final_norm_buf = try metal.HalfBuffer.init(
        device.obj,
        H,
    );
    defer final_norm_buf.deinit();

    // Fill norms with deterministic values.
    for (halfBufferSlice(attn_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.5) +
                @as(f32, @floatFromInt(i)) * 0.01,
        );
    }
    for (halfBufferSlice(ffn_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.8) +
                @as(f32, @floatFromInt(i)) * 0.005,
        );
    }
    for (halfBufferSlice(q_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.6) +
                @as(f32, @floatFromInt(i)) * 0.008,
        );
    }
    for (halfBufferSlice(k_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.7) +
                @as(f32, @floatFromInt(i)) * 0.006,
        );
    }
    for (halfBufferSlice(final_norm_buf), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 1.0) +
                @as(f32, @floatFromInt(i)) * 0.002,
        );
    }

    // ── KV caches (f16, zeroed) ──────────────────────
    const kv_elems: u32 = KVD * MAX_CTX;

    var k_cache_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, kv_elems),
    };
    defer k_cache_arr[0].deinit();

    var v_cache_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, kv_elems),
    };
    defer v_cache_arr[0].deinit();

    // ── Activation scratch buffers (f32) ─────────────

    var residual_buf = try metal.Buffer.init(device.obj, H);
    defer residual_buf.deinit();

    var norm_out_buf = try metal.Buffer.init(device.obj, H);
    defer norm_out_buf.deinit();

    var q_buf = try metal.Buffer.init(device.obj, QD);
    defer q_buf.deinit();

    var k_buf = try metal.Buffer.init(device.obj, KVD);
    defer k_buf.deinit();

    var v_buf = try metal.Buffer.init(device.obj, KVD);
    defer v_buf.deinit();

    var attn_out_buf = try metal.Buffer.init(device.obj, QD);
    defer attn_out_buf.deinit();

    var proj_out_buf = try metal.Buffer.init(device.obj, H);
    defer proj_out_buf.deinit();

    const scratch_len: u32 = NQH * MAX_CTX;

    var attn_scratch_buf = try metal.Buffer.init(
        device.obj,
        scratch_len,
    );
    defer attn_scratch_buf.deinit();

    var gate_buf = try metal.Buffer.init(device.obj, I);
    defer gate_buf.deinit();

    var up_buf = try metal.Buffer.init(device.obj, I);
    defer up_buf.deinit();

    var mlp_out_buf = try metal.Buffer.init(device.obj, I);
    defer mlp_out_buf.deinit();

    var token_ids_buf = try metal.Buffer.init(device.obj, 4);
    defer token_ids_buf.deinit();

    var logits_buf = try metal.Buffer.init(device.obj, V);
    defer logits_buf.deinit();

    // ── Build ForwardDecodeArgs ──────────────────────

    var decode_args = ForwardDecodeArgs{
        .embedding = emb_packed,
        .lm_head = emb_packed,
        .final_norm_scale = final_norm_buf.obj,
        .q_proj = &q_proj_arr,
        .k_proj = &k_proj_arr,
        .v_proj = &v_proj_arr,
        .o_proj = &o_proj_arr,
        .gate_proj = &gate_proj_arr,
        .up_proj = &up_proj_arr,
        .down_proj = &down_proj_arr,
        .attn_norm = &attn_norm_arr,
        .ffn_norm = &ffn_norm_arr,
        .q_norm = &q_norm_arr,
        .k_norm = &k_norm_arr,
        .k_cache = &k_cache_arr,
        .v_cache = &v_cache_arr,
        .residual = residual_buf.obj,
        .norm_out = norm_out_buf.obj,
        .q = q_buf.obj,
        .k = k_buf.obj,
        .v = v_buf.obj,
        .attn_out = attn_out_buf.obj,
        .proj_out = proj_out_buf.obj,
        .attn_scratch = attn_scratch_buf.obj,
        .gate = gate_buf.obj,
        .up = up_buf.obj,
        .mlp_out = mlp_out_buf.obj,
        .token_ids = token_ids_buf.obj,
        .logits = logits_buf.obj,
        .token_id = 0,
        .position = 0,
    };

    // ── Sampling scratch ─────────────────────────────
    var sample_scratch: [V]f32 = undefined;
    var sample_indices: [V]u32 = undefined;

    // ── Prompt: two tokens ───────────────────────────
    const prompt = [_]u32{ 1, 3 };
    var output_tokens: [4]u32 = undefined;

    // Use a fake EOS that is unlikely to appear immediately
    // so we exercise the decode loop.  We cap output at 4
    // tokens anyway (max_context_length=8, prompt=2, so at
    // most 6 decode slots, but we request 4).
    const fake_eos = [_]u32{31}; // Last token in vocab.

    const result = generate(
        TinyBlock,
        &device,
        &pipelines,
        &decode_args,
        .{
            .prompt_ids = &prompt,
            .params = .{
                .temperature = 0.0,
                .top_k = 0,
                .top_p = 1.0,
                .seed = 42,
            },
            .eos_ids = &fake_eos,
            .output_tokens = &output_tokens,
            .scratch = &sample_scratch,
            .indices = &sample_indices,
        },
    );

    // ── Verify results ───────────────────────────────

    // At least one token should have been generated (greedy
    // decode on deterministic weights won't produce EOS=31
    // on the first step).
    try std.testing.expect(result.tokens_generated > 0);
    try std.testing.expect(
        result.tokens_generated <= output_tokens.len,
    );

    // All generated tokens must be valid vocab IDs.
    for (output_tokens[0..result.tokens_generated]) |tok| {
        try std.testing.expect(tok < V);
    }

    // Timing: prefill and decode should both take nonzero ns.
    try std.testing.expect(result.prefill_ns > 0);
    try std.testing.expect(result.decode_ns > 0);
}

test "generate stops on EOS token" {
    // Goal: verify that if greedy decode immediately produces
    // the EOS token, generate returns 0 tokens_generated.
    //
    // Method: same TinyBlock setup, but set EOS to whatever
    // token greedy decode produces first.  We discover this
    // by running one decode step, then use that token as EOS.

    const V = TinyBlock.vocab_size;
    const H = TinyBlock.hidden_size;
    const QD = TinyBlock.query_dim;
    const KVD = TinyBlock.kv_dim;
    const I = TinyBlock.intermediate_size;
    const HD = TinyBlock.head_dim;
    const NQH = TinyBlock.num_query_heads;
    const MAX_CTX = TinyBlock.max_context_length;

    var device: metal.Device = undefined;
    try device.init();

    var pipelines: TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    // Shared packed buffer setup (identical seed 0xB7).
    var emb_packed = try createTestPackedBuffer(
        device.obj,
        V * H,
        0xB7,
    );
    defer emb_packed.deinit();

    var q_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(device.obj, H * QD, 0xA5),
    };
    defer q_proj_arr[0].deinit();

    var k_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(device.obj, H * KVD, 0x5A),
    };
    defer k_proj_arr[0].deinit();

    var v_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(device.obj, H * KVD, 0x33),
    };
    defer v_proj_arr[0].deinit();

    var o_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(device.obj, QD * H, 0xCC),
    };
    defer o_proj_arr[0].deinit();

    var gate_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(device.obj, H * I, 0x55),
    };
    defer gate_proj_arr[0].deinit();

    var up_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(device.obj, H * I, 0xAA),
    };
    defer up_proj_arr[0].deinit();

    var down_proj_arr = [1]metal.PackedBuffer{
        try createTestPackedBuffer(device.obj, I * H, 0x69),
    };
    defer down_proj_arr[0].deinit();

    var attn_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, H),
    };
    defer attn_norm_arr[0].deinit();

    var ffn_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, H),
    };
    defer ffn_norm_arr[0].deinit();

    var q_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, HD),
    };
    defer q_norm_arr[0].deinit();

    var k_norm_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, HD),
    };
    defer k_norm_arr[0].deinit();

    var final_norm_buf = try metal.HalfBuffer.init(
        device.obj,
        H,
    );
    defer final_norm_buf.deinit();

    // Fill norms (same values as the generate-loop test).
    for (halfBufferSlice(attn_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.5) +
                @as(f32, @floatFromInt(i)) * 0.01,
        );
    }
    for (halfBufferSlice(ffn_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.8) +
                @as(f32, @floatFromInt(i)) * 0.005,
        );
    }
    for (halfBufferSlice(q_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.6) +
                @as(f32, @floatFromInt(i)) * 0.008,
        );
    }
    for (halfBufferSlice(k_norm_arr[0]), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 0.7) +
                @as(f32, @floatFromInt(i)) * 0.006,
        );
    }
    for (halfBufferSlice(final_norm_buf), 0..) |*s, i| {
        s.* = @floatCast(
            @as(f32, 1.0) +
                @as(f32, @floatFromInt(i)) * 0.002,
        );
    }

    const kv_elems: u32 = KVD * MAX_CTX;

    var k_cache_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, kv_elems),
    };
    defer k_cache_arr[0].deinit();

    var v_cache_arr = [1]metal.HalfBuffer{
        try metal.HalfBuffer.init(device.obj, kv_elems),
    };
    defer v_cache_arr[0].deinit();

    var residual_buf = try metal.Buffer.init(device.obj, H);
    defer residual_buf.deinit();

    var norm_out_buf = try metal.Buffer.init(device.obj, H);
    defer norm_out_buf.deinit();

    var q_buf = try metal.Buffer.init(device.obj, QD);
    defer q_buf.deinit();

    var k_buf = try metal.Buffer.init(device.obj, KVD);
    defer k_buf.deinit();

    var v_buf = try metal.Buffer.init(device.obj, KVD);
    defer v_buf.deinit();

    var attn_out_buf = try metal.Buffer.init(device.obj, QD);
    defer attn_out_buf.deinit();

    var proj_out_buf = try metal.Buffer.init(device.obj, H);
    defer proj_out_buf.deinit();

    var attn_scratch_buf = try metal.Buffer.init(
        device.obj,
        NQH * MAX_CTX,
    );
    defer attn_scratch_buf.deinit();

    var gate_buf = try metal.Buffer.init(device.obj, I);
    defer gate_buf.deinit();

    var up_buf = try metal.Buffer.init(device.obj, I);
    defer up_buf.deinit();

    var mlp_out_buf = try metal.Buffer.init(device.obj, I);
    defer mlp_out_buf.deinit();

    var token_ids_buf = try metal.Buffer.init(device.obj, 4);
    defer token_ids_buf.deinit();

    var logits_buf = try metal.Buffer.init(device.obj, V);
    defer logits_buf.deinit();

    // ── First pass: discover what token greedy produces ──

    var decode_args = ForwardDecodeArgs{
        .embedding = emb_packed,
        .lm_head = emb_packed,
        .final_norm_scale = final_norm_buf.obj,
        .q_proj = &q_proj_arr,
        .k_proj = &k_proj_arr,
        .v_proj = &v_proj_arr,
        .o_proj = &o_proj_arr,
        .gate_proj = &gate_proj_arr,
        .up_proj = &up_proj_arr,
        .down_proj = &down_proj_arr,
        .attn_norm = &attn_norm_arr,
        .ffn_norm = &ffn_norm_arr,
        .q_norm = &q_norm_arr,
        .k_norm = &k_norm_arr,
        .k_cache = &k_cache_arr,
        .v_cache = &v_cache_arr,
        .residual = residual_buf.obj,
        .norm_out = norm_out_buf.obj,
        .q = q_buf.obj,
        .k = k_buf.obj,
        .v = v_buf.obj,
        .attn_out = attn_out_buf.obj,
        .proj_out = proj_out_buf.obj,
        .attn_scratch = attn_scratch_buf.obj,
        .gate = gate_buf.obj,
        .up = up_buf.obj,
        .mlp_out = mlp_out_buf.obj,
        .token_ids = token_ids_buf.obj,
        .logits = logits_buf.obj,
        .token_id = 0,
        .position = 0,
    };

    var scratch: [V]f32 = undefined;
    var idx_scratch: [V]u32 = undefined;

    // Run a single prefill + 1 decode to find the greedy
    // token produced from prompt [1].
    const prompt_discover = [_]u32{1};
    var out_discover: [1]u32 = undefined;
    const no_eos = [_]u32{0xFFFF_FFFF};

    const r1 = generate(
        TinyBlock,
        &device,
        &pipelines,
        &decode_args,
        .{
            .prompt_ids = &prompt_discover,
            .params = .{
                .temperature = 0.0,
                .top_k = 0,
                .top_p = 1.0,
                .seed = 42,
            },
            .eos_ids = &no_eos,
            .output_tokens = &out_discover,
            .scratch = &scratch,
            .indices = &idx_scratch,
        },
    );
    try std.testing.expectEqual(@as(u32, 1), r1.tokens_generated);
    const greedy_token = out_discover[0];
    try std.testing.expect(greedy_token < V);

    // ── Second pass: set EOS to that token → 0 generated ──
    // Reset KV caches (zero-fill for clean state).
    const k_cache_f16 = halfBufferSlice(k_cache_arr[0]);
    const v_cache_f16 = halfBufferSlice(v_cache_arr[0]);
    @memset(k_cache_f16, @as(f16, 0.0));
    @memset(v_cache_f16, @as(f16, 0.0));

    // Reset activation buffers too.
    @memset(residual_buf.asSlice(), 0.0);

    decode_args.token_id = 0;
    decode_args.position = 0;

    const eos_set = [_]u32{greedy_token};
    const prompt2 = [_]u32{1};
    var out2: [4]u32 = undefined;

    const r2 = generate(
        TinyBlock,
        &device,
        &pipelines,
        &decode_args,
        .{
            .prompt_ids = &prompt2,
            .params = .{
                .temperature = 0.0,
                .top_k = 0,
                .top_p = 1.0,
                .seed = 42,
            },
            .eos_ids = &eos_set,
            .output_tokens = &out2,
            .scratch = &scratch,
            .indices = &idx_scratch,
        },
    );

    // The first sampled token should be the greedy_token
    // which is EOS — so zero tokens are emitted.
    try std.testing.expectEqual(
        @as(u32, 0),
        r2.tokens_generated,
    );
}
