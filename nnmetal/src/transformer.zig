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
const specialized_qmv = @import("specialized_qmv.zig");

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

/// Fused RoPE K + KV cache update dimensions.
pub const FusedRoPEKVDims = extern struct {
    num_kv_heads: u32,
    head_dim: u32,
    position: u32,
    max_context_length: u32,
    rope_theta: f32,
};

/// Fused per-head RMSNorm + RoPE dimensions.
pub const FusedNormRoPEDims = extern struct {
    num_heads: u32,
    head_dim: u32,
    position: u32,
    eps: f32,
    rope_theta: f32,
};

/// Fused K-norm + RoPE K + KV cache dimensions.
pub const FusedKNormRoPEKVDims = extern struct {
    num_kv_heads: u32,
    head_dim: u32,
    position: u32,
    max_context_length: u32,
    eps: f32,
    rope_theta: f32,
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
    rms_norm_f16out: metal.ComputePipeline,
    silu: metal.ComputePipeline,
    silu_elementwise_mul: metal.ComputePipeline,
    rope: metal.ComputePipeline,
    kv_cache_update: metal.ComputePipeline,
    gqa_attention: metal.ComputePipeline,
    embedding_lookup: metal.ComputePipeline,
    residual_add: metal.ComputePipeline,
    rms_norm_f16: metal.ComputePipeline,
    rope_f16: metal.ComputePipeline,
    kv_cache_update_f16in: metal.ComputePipeline,
    fused_rope_k_kv_cache_f16: metal.ComputePipeline,
    fused_norm_rope_f16: metal.ComputePipeline,
    fused_k_norm_rope_kv_cache_f16: metal.ComputePipeline,
    gqa_attention_f16io: metal.ComputePipeline,
    gqa_attention_f16io_tg: metal.ComputePipeline,
    silu_elementwise_mul_f16: metal.ComputePipeline,
    residual_add_f16: metal.ComputePipeline,
    set_completion_flag: metal.ComputePipeline,
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
            .rms_norm_f16out = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "rms_norm_f16out",
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
            .rms_norm_f16 = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "rms_norm_f16",
            ),
            .rope_f16 = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "rope_f16",
            ),
            .kv_cache_update_f16in = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "kv_cache_update_f16in",
            ),
            .fused_rope_k_kv_cache_f16 = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "fused_rope_k_kv_cache_update_f16",
            ),
            .fused_norm_rope_f16 = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "fused_norm_rope_f16",
            ),
            .fused_k_norm_rope_kv_cache_f16 = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "fused_k_norm_rope_kv_cache_f16",
            ),
            .gqa_attention_f16io = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "gqa_attention_f16io",
            ),
            .gqa_attention_f16io_tg = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "gqa_attention_f16io_tg",
            ),
            .silu_elementwise_mul_f16 = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "silu_elementwise_mul_f16",
            ),
            .residual_add_f16 = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "residual_add_f16",
            ),
            .set_completion_flag = try metal.ComputePipeline.init(
                device_obj,
                lib,
                "set_completion_flag",
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

/// Dispatch fused RoPE K + KV cache update.
/// Applies RoPE to K and writes both K and V directly to
/// their caches in a single dispatch.  K is NOT written
/// back to the k_proj buffer — only to k_cache.
/// Saves 1 dispatch + 1 barrier per block vs separate
/// rope_k + kv_cache_update.
fn dispatchFusedRoPEKVCache(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    k_proj_buffer: objc.Object,
    v_proj_buffer: objc.Object,
    k_cache_buffer: objc.Object,
    v_cache_buffer: objc.Object,
    dims: FusedRoPEKVDims,
) void {
    std.debug.assert(dims.num_kv_heads > 0);
    std.debug.assert(dims.head_dim >= 2);
    std.debug.assert(dims.head_dim % 2 == 0);
    std.debug.assert(
        dims.position < dims.max_context_length,
    );

    setRawBuffer(encoder, k_proj_buffer, 0, 0);
    setRawBuffer(encoder, v_proj_buffer, 0, 1);
    setRawBuffer(encoder, k_cache_buffer, 0, 2);
    setRawBuffer(encoder, v_cache_buffer, 0, 3);
    metal.setBytes(
        encoder,
        FusedRoPEKVDims,
        &dims,
        4,
    );

    // One thread per RoPE pair; each thread also copies
    // the corresponding 2 V elements.
    const count = dims.num_kv_heads * (dims.head_dim / 2);
    device.dispatch1D(encoder, pipeline, count);
}

/// Dispatch fused per-head RMSNorm + RoPE (f16 in-place).
/// One threadgroup per head, 256 threads per group.
/// Replaces separate Q norm + RoPE Q dispatches.
fn dispatchFusedNormRoPE(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    data_buffer: objc.Object,
    scale_buffer: objc.Object,
    dims: FusedNormRoPEDims,
) void {
    std.debug.assert(dims.num_heads > 0);
    std.debug.assert(dims.head_dim >= 2);
    std.debug.assert(dims.head_dim % 2 == 0);

    setRawBuffer(encoder, data_buffer, 0, 0);
    setRawBuffer(encoder, scale_buffer, 0, 1);
    metal.setBytes(
        encoder,
        FusedNormRoPEDims,
        &dims,
        2,
    );

    const grid = metal.MTLSize{
        .width = @as(c_ulong, dims.num_heads),
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

/// Dispatch fused K-norm + RoPE K + KV cache update.
/// One threadgroup per KV head, 256 threads per group.
/// Replaces separate K norm + RoPE K + KV cache dispatches.
fn dispatchFusedKNormRoPEKVCache(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    k_proj_buffer: objc.Object,
    v_proj_buffer: objc.Object,
    k_cache_buffer: objc.Object,
    v_cache_buffer: objc.Object,
    k_scale_buffer: objc.Object,
    dims: FusedKNormRoPEKVDims,
) void {
    std.debug.assert(dims.num_kv_heads > 0);
    std.debug.assert(dims.head_dim >= 2);
    std.debug.assert(dims.head_dim % 2 == 0);
    std.debug.assert(
        dims.position < dims.max_context_length,
    );

    setRawBuffer(encoder, k_proj_buffer, 0, 0);
    setRawBuffer(encoder, v_proj_buffer, 0, 1);
    setRawBuffer(encoder, k_cache_buffer, 0, 2);
    setRawBuffer(encoder, v_cache_buffer, 0, 3);
    setRawBuffer(encoder, k_scale_buffer, 0, 4);
    metal.setBytes(
        encoder,
        FusedKNormRoPEKVDims,
        &dims,
        5,
    );

    const grid = metal.MTLSize{
        .width = @as(c_ulong, dims.num_kv_heads),
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

/// Dispatch grouped query attention (decode path, M=1).
/// One threadgroup (256 threads) per query head.
/// Dispatch the threadgroup-scores GQA attention kernel.
/// Same buffer layout as dispatchGQAAttention but buffer(4)
/// (scratch) is skipped — the kernel stores scores in
/// threadgroup memory instead.  Valid when seq_len <= 7168.
fn dispatchGQAAttentionTG(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    q_buffer: objc.Object,
    k_cache_buffer: objc.Object,
    v_cache_buffer: objc.Object,
    output_buffer: objc.Object,
    dims: GQADims,
) void {
    std.debug.assert(dims.num_query_heads > 0);
    std.debug.assert(dims.seq_len > 0);
    std.debug.assert(
        dims.seq_len <= dims.max_context_length,
    );
    std.debug.assert(
        dims.heads_per_kv_group ==
            dims.num_query_heads / dims.num_kv_heads,
    );
    // The TG variant only works when seq_len fits in
    // threadgroup memory (7168 floats for scores).
    std.debug.assert(dims.seq_len <= 7168);

    setRawBuffer(encoder, q_buffer, 0, 0);
    setRawBuffer(encoder, k_cache_buffer, 0, 1);
    setRawBuffer(encoder, v_cache_buffer, 0, 2);
    setRawBuffer(encoder, output_buffer, 0, 3);
    // Skip buffer(4) — no scratch needed.
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

    // For K <= 2048, use constant-cache variant: zero shared
    // memory for max occupancy, 2 rows/simdgroup (32 rows/tg).
    // Otherwise, fall back to shared-memory variants.
    const use_const = dims.K <= 2048;
    const rows_per_tg: u32 = if (use_const) 32 else 16;
    const threadgroups = (dims.M + rows_per_tg - 1) /
        rows_per_tg;
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
    const pipeline = if (use_const)
        device.qmv_fused_pair_const
    else
        device.qmv_fused_pair;
    device.dispatchCustom(encoder, pipeline, grid, group);
}

/// Dispatch fused-pair QMV with f16 input, f32 output.
/// Input is [K] f16, outputs are [M] f32.
fn dispatchQMVFusedPairf16in(
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
    std.debug.assert(dims.K / 32 <= dims.group_size);
    std.debug.assert(packed_a.packed_count > 0);
    std.debug.assert(packed_b.packed_count > 0);

    // buffer(0,1): packed A (bits + scales).
    metal.setPackedBuffer(encoder, packed_a, 0);
    // buffer(2): shared input vector [K] f16.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): output A [M] f32.
    setRawBuffer(encoder, output_a, 0, 3);
    // buffer(4,5): packed B (bits + scales).
    metal.setPackedBuffer(encoder, packed_b, 4);
    // buffer(6): output B [M] f32.
    setRawBuffer(encoder, output_b, 0, 6);
    // buffer(7): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 7);

    const use_const = dims.K <= 2048;
    const rows_per_tg: u32 = if (use_const) 32 else 16;
    const threadgroups = (dims.M + rows_per_tg - 1) /
        rows_per_tg;
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
    const pipeline = if (use_const)
        device.qmv_fused_pair_const_f16in
    else
        device.qmv_fused_pair;
    device.dispatchCustom(encoder, pipeline, grid, group);
}

/// Dispatch fused-pair QMV with f16 input, f16 output.
/// Input is [K] f16, outputs are [M] f16.
fn dispatchQMVFusedPairf16io(
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
    std.debug.assert(dims.K / 32 <= dims.group_size);
    std.debug.assert(packed_a.packed_count > 0);
    std.debug.assert(packed_b.packed_count > 0);

    // buffer(0,1): packed A (bits + scales).
    metal.setPackedBuffer(encoder, packed_a, 0);
    // buffer(2): shared input vector [K] f16.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): output A [M] f16.
    setRawBuffer(encoder, output_a, 0, 3);
    // buffer(4,5): packed B (bits + scales).
    metal.setPackedBuffer(encoder, packed_b, 4);
    // buffer(6): output B [M] f16.
    setRawBuffer(encoder, output_b, 0, 6);
    // buffer(7): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 7);

    const use_const = dims.K <= 2048;
    const use_spec =
        if (device.spec_qmv_fused_pair_f16io) |s|
            s.max_threads_per_group >= 512 and
                dims.K == device.spec_hidden_K
        else
            false;
    const rows_per_tg: u32 = if (use_spec or use_const)
        32
    else
        16;
    const threadgroups = (dims.M + rows_per_tg - 1) /
        rows_per_tg;
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
    const pipeline = if (use_spec)
        device.spec_qmv_fused_pair_f16io.?
    else if (use_const)
        device.qmv_fused_pair_const_f16io
    else
        device.qmv_fused_pair;
    device.dispatchCustom(encoder, pipeline, grid, group);
}

/// Dispatch fused gate/up QMV + SiLU + elementwise multiply.
/// output[row] = silu(gate_row) * up_row.  Eliminates the
/// separate SiLU dispatch and two barriers per block.
/// Uses the specialized pipeline when available; falls
/// back to the generic fused pair + separate SiLU.
fn dispatchQMVFusedPairSiLUf16io(
    device: *const metal.Device,
    encoder: objc.Object,
    packed_a: metal.PackedBuffer,
    packed_b: metal.PackedBuffer,
    input_buffer: objc.Object,
    output: objc.Object,
    dims: QMVDims,
) void {
    std.debug.assert(dims.M > 0);
    std.debug.assert(dims.K > 0);
    std.debug.assert(dims.K % 32 == 0);
    std.debug.assert(dims.group_size > 0);
    std.debug.assert(dims.K % dims.group_size == 0);
    std.debug.assert(dims.K <= 6144);
    std.debug.assert(dims.K >= 256);
    std.debug.assert(dims.K / 32 <= dims.group_size);
    std.debug.assert(packed_a.packed_count > 0);
    std.debug.assert(packed_b.packed_count > 0);

    // buffer(0,1): packed A / gate (bits + scales).
    metal.setPackedBuffer(encoder, packed_a, 0);
    // buffer(2): shared input vector [K] f16.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): fused output [M] f16.
    setRawBuffer(encoder, output, 0, 3);
    // buffer(4,5): packed B / up (bits + scales).
    metal.setPackedBuffer(encoder, packed_b, 4);
    // buffer(7): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 7);

    // 1 row per simdgroup × 16 simdgroups = 16 rows/TG.
    const rows_per_tg: u32 = 16;
    const threadgroups = (dims.M + rows_per_tg - 1) /
        rows_per_tg;
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
        device.spec_qmv_fused_pair_silu_f16io.?,
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
    // The multigroup kernel uses byte-aligned lane assignment
    // (distributing K/8 bytes evenly across 32 lanes) so it
    // handles any K that is group-aligned, even when K % 256
    // != 0 (e.g. K=5504 for the Bonsai down projection).
    const aligned = (dims.K % dims.group_size == 0) and
        (dims.K <= 6144) and (dims.K >= 256);
    const single_group = dims.K / 32 <= dims.group_size;
    // Use constant-cache variants when the input vector fits
    // in Metal's 64 KB constant address space (K*4 <= 64 KB).
    // Zero shared memory allows maximum GPU occupancy.
    // Single-group: qmv_const (2 rows/sg, 32 rows/tg).
    // Multi-group:  qmv_const_multigroup (same layout).
    const fits_const = dims.K * 4 <= 65536;
    const use_const_sg = aligned and single_group and
        fits_const;
    const use_const_mg = aligned and !single_group and
        fits_const;
    const actual_pipeline = if (use_const_sg)
        device.qmv_const
    else if (use_const_mg)
        device.qmv_const_multigroup
    else if (aligned and single_group)
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

    // Constant-cache variants: 32 rows/tg (16 sgs × 2 rows).
    // Shared-memory fast: 16 rows/tg (16 simdgroups of 32).
    // Fallback qmv: 2 rows/tg (2 simdgroups of 32).
    const use_const = use_const_sg or use_const_mg;
    const rows_per_tg: u32 = if (use_const)
        32
    else if (aligned)
        16
    else
        2;
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

/// Dispatch 1-bit QMV with f16 input, f32 output.
/// W is [M × K] in Q1_0_g128 packed format.  Input is [K] f16,
/// output is [M] f32.  Selects constant-cache variants when K
/// fits in 64 KB, falls back to the base qmv_f16in kernel.
fn dispatchQMVf16in(
    device: *const metal.Device,
    encoder: objc.Object,
    packed_buffer: metal.PackedBuffer,
    input_buffer: objc.Object,
    output_buffer: objc.Object,
    dims: QMVDims,
) void {
    std.debug.assert(dims.M > 0);
    std.debug.assert(dims.K > 0);
    std.debug.assert(dims.K % 32 == 0);
    std.debug.assert(dims.group_size > 0);
    std.debug.assert(packed_buffer.packed_count > 0);

    const aligned = (dims.K % dims.group_size == 0) and
        (dims.K <= 6144) and (dims.K >= 256);
    const single_group = dims.K / 32 <= dims.group_size;
    const fits_const = dims.K * 4 <= 65536;
    const use_const_sg = aligned and single_group and
        fits_const;
    const use_const_mg = aligned and !single_group and
        fits_const;
    const use_spec = if (device.spec_qmv_f16in) |s|
        s.max_threads_per_group >= 512 and
            dims.K == device.spec_hidden_K
    else
        false;
    const actual_pipeline = if (use_spec)
        device.spec_qmv_f16in.?
    else if (use_const_sg)
        device.qmv_const_f16in
    else if (use_const_mg)
        device.qmv_const_multigroup_f16in
    else
        device.qmv_f16in;

    // buffer(0,1): packed bits + f16 scales.
    metal.setPackedBuffer(encoder, packed_buffer, 0);
    // buffer(2): input vector [K] f16.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): output vector [M] f32.
    setRawBuffer(encoder, output_buffer, 0, 3);
    // buffer(4): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 4);

    const use_const = use_spec or use_const_sg or use_const_mg;
    const rows_per_tg: u32 = if (use_const)
        32
    else if (aligned)
        16
    else
        2;
    const threads_per_tg: u32 = if (use_spec or aligned)
        512
    else
        64;
    const threadgroups = (dims.M + rows_per_tg - 1) /
        rows_per_tg;
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

/// Dispatch 1-bit QMV with f16 input, f16 output.
/// W is [M × K] in Q1_0_g128 packed format.  Input is [K] f16,
/// output is [M] f16.  Selects constant-cache variants when K
/// fits in 64 KB, falls back to the base qmv_f16io kernel.
fn dispatchQMVf16io(
    device: *const metal.Device,
    encoder: objc.Object,
    packed_buffer: metal.PackedBuffer,
    input_buffer: objc.Object,
    output_buffer: objc.Object,
    dims: QMVDims,
) void {
    std.debug.assert(dims.M > 0);
    std.debug.assert(dims.K > 0);
    std.debug.assert(dims.K % 32 == 0);
    std.debug.assert(dims.group_size > 0);
    std.debug.assert(packed_buffer.packed_count > 0);

    const aligned = (dims.K % dims.group_size == 0) and
        (dims.K <= 6144) and (dims.K >= 256);
    const single_group = dims.K / 32 <= dims.group_size;
    const fits_const = dims.K * 4 <= 65536;
    const use_const_sg = aligned and single_group and
        fits_const;
    const use_const_mg = aligned and !single_group and
        fits_const;
    // Prefer specialized pipeline when available and K matches.
    const use_spec = if (device.spec_qmv_f16io) |s|
        s.max_threads_per_group >= 512 and
            dims.K == device.spec_hidden_K
    else
        false;
    const actual_pipeline = if (use_spec)
        device.spec_qmv_f16io.?
    else if (use_const_sg)
        device.qmv_const_f16io
    else if (use_const_mg)
        device.qmv_const_multigroup_f16io
    else
        device.qmv_f16io;

    // buffer(0,1): packed bits + f16 scales.
    metal.setPackedBuffer(encoder, packed_buffer, 0);
    // buffer(2): input vector [K] f16.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): output vector [M] f16.
    setRawBuffer(encoder, output_buffer, 0, 3);
    // buffer(4): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 4);

    const use_const = use_spec or use_const_sg or use_const_mg;
    const rows_per_tg: u32 = if (use_const)
        32
    else if (aligned)
        16
    else
        2;
    const threads_per_tg: u32 = if (use_spec or aligned)
        512
    else
        64;
    const threadgroups = (dims.M + rows_per_tg - 1) /
        rows_per_tg;
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

/// Dispatch a 1-bit QMV with fused f32 residual accumulate.
/// Instead of writing f16 to an intermediate buffer, the kernel
/// adds its f32 result directly to the residual: residual[row] += acc.
/// Eliminates one dispatch (residual_add_f16) and one barrier per
/// call site.  Buffer layout matches qmv_f16io except buffer(3) is
/// the f32 residual instead of an f16 output.
fn dispatchQMVf16ioResadd(
    device: *const metal.Device,
    encoder: objc.Object,
    packed_buffer: metal.PackedBuffer,
    input_buffer: objc.Object,
    residual_buffer: objc.Object,
    dims: QMVDims,
) void {
    std.debug.assert(dims.M > 0);
    std.debug.assert(dims.K > 0);
    std.debug.assert(dims.K % 32 == 0);
    std.debug.assert(dims.group_size > 0);
    std.debug.assert(packed_buffer.packed_count > 0);

    const aligned = (dims.K % dims.group_size == 0) and
        (dims.K <= 6144) and (dims.K >= 256);
    const single_group = dims.K / 32 <= dims.group_size;
    const fits_const = dims.K * 4 <= 65536;
    const use_const_sg = aligned and single_group and
        fits_const;
    const use_const_mg = aligned and !single_group and
        fits_const;
    // Prefer specialized pipelines when K matches.
    const use_spec_sg =
        if (device.spec_qmv_f16io_resadd) |s|
            s.max_threads_per_group >= 512 and
                use_const_sg and
                dims.K == device.spec_hidden_K
        else
            false;
    const use_spec_mg =
        if (device.spec_qmv_mg_f16io_resadd) |s|
            s.max_threads_per_group >= 512 and
                use_const_mg and
                dims.K == device.spec_inter_K
        else
            false;
    const actual_pipeline = if (use_spec_sg)
        device.spec_qmv_f16io_resadd.?
    else if (use_spec_mg)
        device.spec_qmv_mg_f16io_resadd.?
    else if (use_const_sg)
        device.qmv_const_f16io_resadd
    else if (use_const_mg)
        device.qmv_const_multigroup_f16io_resadd
    else
        device.qmv_f16io_resadd;

    // buffer(0,1): packed bits + f16 scales.
    metal.setPackedBuffer(encoder, packed_buffer, 0);
    // buffer(2): input vector [K] f16.
    setRawBuffer(encoder, input_buffer, 0, 2);
    // buffer(3): residual vector [M] f32 (accumulated in-place).
    setRawBuffer(encoder, residual_buffer, 0, 3);
    // buffer(4): QMVDims.
    metal.setBytes(encoder, QMVDims, &dims, 4);

    const use_spec = use_spec_sg or use_spec_mg;
    const use_const = use_spec or use_const_sg or use_const_mg;
    const rows_per_tg: u32 = if (use_const)
        32
    else if (aligned)
        16
    else
        2;
    const threads_per_tg: u32 = if (use_spec or aligned)
        512
    else
        64;
    const threadgroups = (dims.M + rows_per_tg - 1) /
        rows_per_tg;
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
    device.dispatchCustom(
        encoder,
        actual_pipeline,
        grid,
        group,
    );
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
    // Activation scratch buffers (f16, except residual which is f32).
    residual: objc.Object, // [hidden_size] f32 — modified in place.
    norm_out: objc.Object, // [hidden_size] f16 — scratch.
    q: objc.Object, // [query_dim] f16
    k: objc.Object, // [kv_dim] f16
    v: objc.Object, // [kv_dim] f16
    attn_out: objc.Object, // [query_dim] f16
    proj_out: objc.Object, // [hidden_size] f16 — O and down output.
    attn_scratch: objc.Object, // [num_query_heads × max_ctx] f32
    gate: objc.Object, // [intermediate_size] f16
    up: objc.Object, // [intermediate_size] f16
    mlp_out: objc.Object, // [intermediate_size] f16 — SiLU output.
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

    // GPU completion flag for spin-wait (Rule 26).
    // The GPU writes 1 here when done; the CPU spin-reads
    // it to avoid waitUntilCompleted Mach kernel trap.
    flag_buf: objc.Object,
    flag_ptr: *volatile u32,

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
        pipelines.rms_norm_f16out,
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
    dispatchQMVf16in(
        device,
        encoder,
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

/// Dispatch the GPU completion flag kernel: atomic-write 1
/// to flag[0].  Must be the last dispatch before endEncoding
/// so the CPU can spin-wait on the flag instead of calling
/// waitUntilCompleted (~100-150 us Mach trap saved).
///
/// A bufferBarrier must precede this dispatch to ensure all
/// preceding kernel writes are visible before the flag is
/// set.
///
/// Dispatch: 1 threadgroup, 1 thread.
fn dispatchCompletionFlag(
    device: *const metal.Device,
    encoder: objc.Object,
    pipeline: metal.ComputePipeline,
    flag_buf: objc.Object,
) void {
    std.debug.assert(encoder.value != null);
    std.debug.assert(flag_buf.value != null);

    setRawBuffer(encoder, flag_buf, 0, 0);

    const one = metal.MTLSize{
        .width = 1,
        .height = 1,
        .depth = 1,
    };
    device.dispatchCustom(
        encoder,
        pipeline,
        one,
        one,
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
        pipelines.rms_norm_f16out,
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
    dispatchQMVf16io(device, encoder, a.q_proj, a.norm_out, a.q, .{
        .M = Config.query_dim,
        .K = Config.hidden_size,
        .group_size = Config.group_size,
    });
    // K and V projections share input (norm_out) and dims
    // (kv_dim x hidden_size). Fuse into one dispatch to
    // halve input vector loads and save dispatch overhead.
    const can_fuse_kv = comptime (Config.hidden_size % Config.group_size == 0) and
        (Config.hidden_size <= 6144) and
        (Config.hidden_size >= 256) and
        (Config.hidden_size / 32 <= Config.group_size);
    if (can_fuse_kv) {
        dispatchQMVFusedPairf16io(
            device,
            encoder,
            a.k_proj,
            a.v_proj,
            a.norm_out,
            a.k,
            a.v,
            kv_qmv,
        );
    } else {
        dispatchQMVf16io(device, encoder, a.k_proj, a.norm_out, a.k, kv_qmv);
        dispatchQMVf16io(device, encoder, a.v_proj, a.norm_out, a.v, kv_qmv);
    }
    bufferBarrier(encoder);
    // Fused Q norm + RoPE Q: per-head RMSNorm then RoPE
    // in a single dispatch.  Saves 1 dispatch vs separate.
    dispatchFusedNormRoPE(
        device,
        encoder,
        pipelines.fused_norm_rope_f16,
        a.q,
        a.q_norm_scale,
        .{
            .num_heads = Config.num_query_heads,
            .head_dim = Config.head_dim,
            .position = a.position,
            .eps = 1e-6,
            .rope_theta = Config.rope_theta,
        },
    );
    // Fused K norm + RoPE K + KV cache: per-head RMSNorm,
    // then RoPE, then write to k_cache + copy V to v_cache.
    // Saves 2 dispatches + 2 barriers vs separate path.
    dispatchFusedKNormRoPEKVCache(
        device,
        encoder,
        pipelines.fused_k_norm_rope_kv_cache_f16,
        a.k,
        a.v,
        a.k_cache,
        a.v_cache,
        a.k_norm_scale,
        .{
            .num_kv_heads = Config.num_kv_heads,
            .head_dim = Config.head_dim,
            .position = a.position,
            .max_context_length = Config.max_context_length,
            .eps = 1e-6,
            .rope_theta = Config.rope_theta,
        },
    );
    // Single barrier: wait for both fused dispatches before
    // GQA attention reads Q and k/v caches.
    bufferBarrier(encoder);
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
    // Q RoPE: applied in-place (Q is read by GQA attention).
    dispatchRoPE(device, encoder, pipelines.rope_f16, a.q, .{
        .num_heads = Config.num_query_heads,
        .head_dim = Config.head_dim,
        .position = a.position,
        .rope_theta = Config.rope_theta,
    });
    // Fused RoPE K + KV cache update: applies RoPE to K and
    // writes both K and V directly to their caches.  Saves
    // 1 dispatch + 1 barrier vs separate rope_k + cache_update.
    dispatchFusedRoPEKVCache(
        device,
        encoder,
        pipelines.fused_rope_k_kv_cache_f16,
        a.k,
        a.v,
        a.k_cache,
        a.v_cache,
        .{
            .num_kv_heads = Config.num_kv_heads,
            .head_dim = Config.head_dim,
            .position = a.position,
            .max_context_length = Config.max_context_length,
            .rope_theta = Config.rope_theta,
        },
    );
    // Single barrier: wait for both Q RoPE and fused K/V
    // cache write before GQA attention reads Q and k/v caches.
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
    const dims = GQADims{
        .num_query_heads = Config.num_query_heads,
        .num_kv_heads = Config.num_kv_heads,
        .head_dim = Config.head_dim,
        .seq_len = a.seq_len,
        .max_context_length = Config.max_context_length,
        .heads_per_kv_group = Config.heads_per_kv_group,
    };

    // Use threadgroup-scores kernel when seq_len fits in
    // threadgroup memory (avoids the scratch buffer
    // round-trip).  Fall back to the scratch-based kernel
    // for longer sequences.
    // Use threadgroup-scores kernel when seq_len fits in
    // threadgroup memory (avoids the scratch buffer
    // round-trip).  Fall back to the scratch-based kernel
    // for longer sequences.
    const TG_SEQ_LIMIT: u32 = 1024;
    if (a.seq_len <= TG_SEQ_LIMIT) {
        dispatchGQAAttentionTG(
            device,
            encoder,
            pipelines.gqa_attention_f16io_tg,
            a.q,
            a.k_cache,
            a.v_cache,
            a.attn_out,
            dims,
        );
    } else {
        dispatchGQAAttention(
            device,
            encoder,
            pipelines.gqa_attention_f16io,
            a.q,
            a.k_cache,
            a.v_cache,
            a.attn_out,
            a.attn_scratch,
            dims,
        );
    }
    bufferBarrier(encoder);
    // Fused QMV + residual accumulate: the O-projection result
    // is added directly to the f32 residual (residual[row] += acc)
    // instead of writing f16 to proj_out then dispatching
    // residual_add_f16.  Saves one dispatch + one barrier.
    dispatchQMVf16ioResadd(
        device,
        encoder,
        a.o_proj,
        a.attn_out,
        a.residual,
        .{
            .M = Config.hidden_size,
            .K = Config.query_dim,
            .group_size = Config.group_size,
        },
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
        pipelines.rms_norm_f16out,
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
    // Prefer the fused gate+up+SiLU kernel when the
    // specialized pipeline is available — saves 2 dispatches
    // and 2 barriers per block by computing
    // silu(gate) * up in the QMV output stage.
    const can_fuse = comptime (Config.hidden_size % Config.group_size == 0) and
        (Config.hidden_size <= 6144) and
        (Config.hidden_size >= 256) and
        (Config.hidden_size / 32 <= Config.group_size);
    const use_fused_silu = can_fuse and
        device.spec_qmv_fused_pair_silu_f16io != null and
        device.spec_qmv_fused_pair_silu_f16io.?
            .max_threads_per_group >= 512 and
        mlp_qmv.K == device.spec_hidden_K;
    if (use_fused_silu) {
        // Fused gate+up+SiLU: output goes directly to
        // mlp_out, skipping gate/up buffers entirely.
        dispatchQMVFusedPairSiLUf16io(
            device,
            encoder,
            a.gate_proj,
            a.up_proj,
            a.norm_out,
            a.mlp_out,
            mlp_qmv,
        );
    } else if (can_fuse) {
        dispatchQMVFusedPairf16io(
            device,
            encoder,
            a.gate_proj,
            a.up_proj,
            a.norm_out,
            a.gate,
            a.up,
            mlp_qmv,
        );
        bufferBarrier(encoder);
        dispatchSiLUElementwiseMul(
            device,
            encoder,
            pipelines.silu_elementwise_mul_f16,
            a.gate,
            a.up,
            a.mlp_out,
            Config.intermediate_size,
        );
    } else {
        dispatchQMVf16io(
            device,
            encoder,
            a.gate_proj,
            a.norm_out,
            a.gate,
            mlp_qmv,
        );
        dispatchQMVf16io(
            device,
            encoder,
            a.up_proj,
            a.norm_out,
            a.up,
            mlp_qmv,
        );
        bufferBarrier(encoder);
        dispatchSiLUElementwiseMul(
            device,
            encoder,
            pipelines.silu_elementwise_mul_f16,
            a.gate,
            a.up,
            a.mlp_out,
            Config.intermediate_size,
        );
    }
    bufferBarrier(encoder);
    // Fused QMV + residual accumulate: the down-projection
    // result is added directly to the f32 residual instead of
    // writing f16 to proj_out then dispatching residual_add_f16.
    // Saves one dispatch + one barrier.
    dispatchQMVf16ioResadd(
        device,
        encoder,
        a.down_proj,
        a.mlp_out,
        a.residual,
        .{
            .M = Config.hidden_size,
            .K = Config.intermediate_size,
            .group_size = Config.group_size,
        },
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
    /// Optional buffer for per-token decode latencies
    /// (nanoseconds).  When non-null, must be at least as
    /// long as `output_tokens`.  Element [i] receives the
    /// wall-clock time for the decode step that runs while
    /// output_tokens[i] is the current token (GPU forward
    /// pass + CPU sampling).
    per_token_ns: ?[]u64 = null,
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
pub fn applyTemperature(
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
pub fn applyTopK(
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
pub fn applyTopP(
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
    if (opts.per_token_ns) |buf| {
        std.debug.assert(buf.len >= opts.output_tokens.len);
    }

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

        const t_start = std.time.nanoTimestamp();
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
        const t_end = std.time.nanoTimestamp();

        // Record per-token timing if buffer provided.
        if (opts.per_token_ns) |buf| {
            buf[count] = @intCast(t_end - t_start);
        }

        count += 1;
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
    std.debug.assert(args.flag_buf.value != null);

    for (prompt_ids, 0..) |token_id, i| {
        args.token_id = token_id;
        args.position = @intCast(i);

        const cmd = device.beginCommandBufferUnretained();
        const enc = device.beginCompute(cmd);
        forwardDecode(
            Config,
            device,
            enc,
            pipelines,
            args.*,
        );
        bufferBarrier(enc);
        dispatchCompletionFlag(
            device,
            enc,
            pipelines.set_completion_flag,
            args.flag_buf,
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndSpinOnFlag(cmd, args.flag_ptr);
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
    std.debug.assert(args.flag_buf.value != null);

    args.token_id = token_id;
    args.position = position;

    const cmd = device.beginCommandBufferUnretained();
    const enc = device.beginCompute(cmd);
    forwardDecode(
        Config,
        device,
        enc,
        pipelines,
        args.*,
    );
    bufferBarrier(enc);
    dispatchCompletionFlag(
        device,
        enc,
        pipelines.set_completion_flag,
        args.flag_buf,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndSpinOnFlag(cmd, args.flag_ptr);

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
pub fn isEosToken(
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


test {
    _ = @import("transformer_test.zig");
}
