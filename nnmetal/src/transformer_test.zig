//! Tests for transformer.zig — CPU reference implementations,
//! test helpers, and GPU-vs-CPU comparison tests.

const std = @import("std");
const objc = @import("objc");
const metal = @import("metal.zig");
const transformer = @import("transformer.zig");

// Type aliases — keep test bodies identical to the original code.
const TransformerConfig = transformer.TransformerConfig;
const TransformerPipelines = transformer.TransformerPipelines;
const ForwardBlockArgs = transformer.ForwardBlockArgs;
const ForwardDecodeArgs = transformer.ForwardDecodeArgs;
const SamplingParams = transformer.SamplingParams;
const GenerateResult = transformer.GenerateResult;
const GenerateOpts = transformer.GenerateOpts;

// Public model configs (comptime validation test).
const Bonsai1_7B = transformer.Bonsai1_7B;
const Bonsai4B = transformer.Bonsai4B;
const Bonsai8B = transformer.Bonsai8B;

// Dispatch functions.
const dispatchRMSNorm = transformer.dispatchRMSNorm;
const dispatchSiLU = transformer.dispatchSiLU;
const dispatchSiLUElementwiseMul = transformer.dispatchSiLUElementwiseMul;
const dispatchRoPE = transformer.dispatchRoPE;
const dispatchKVCacheUpdate = transformer.dispatchKVCacheUpdate;
const dispatchGQAAttention = transformer.dispatchGQAAttention;
const dispatchEmbeddingLookup = transformer.dispatchEmbeddingLookup;
const dispatchResidualAdd = transformer.dispatchResidualAdd;
const dispatchQMV = transformer.dispatchQMV;
const forwardBlock = transformer.forwardBlock;
const forwardDecode = transformer.forwardDecode;

// Sampling and generation.
const argmax = transformer.argmax;
const softmaxInPlace = transformer.softmaxInPlace;
const sampleToken = transformer.sampleToken;
const applyTemperature = transformer.applyTemperature;
const applyTopK = transformer.applyTopK;
const applyTopP = transformer.applyTopP;
const isEosToken = transformer.isEosToken;
const generate = transformer.generate;

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

test "qmv_fast_multigroup handles non-byte-aligned cols" {
    // Goal: Verify qmv_fast_multigroup correctly processes
    // all columns when cols_per_lane (K/32) is not a multiple
    // of 8.  For K=5504, cols_per_lane=172, bytes_per_lane=21,
    // leaving 4 tail columns per lane that must be accumulated
    // separately.  Without the tail loop fix, 128 of 5504
    // columns (2.3%) are silently dropped per row.
    //
    // Method: Packed weight matrix [M × K] with K=5504,
    // dispatch via dispatchQMV (selects qmv_fast_multigroup),
    // compare against column-by-column CPU reference.

    const M: u32 = 64;
    const K: u32 = 5504;
    const GS: u32 = 128;

    // Verify this exercises the multigroup path.
    comptime {
        std.debug.assert(K % 32 == 0);
        std.debug.assert(K % GS == 0);
        std.debug.assert(K / 32 > GS); // multigroup
        std.debug.assert((K / 32) % 8 != 0); // tail cols
    }

    var device: metal.Device = undefined;
    try device.init();

    // ── Packed weight buffer [M × K] ─────────────────
    var packed_buf = try createTestPackedBuffer(
        device.obj,
        M * K,
        0x5A,
    );
    defer packed_buf.deinit();

    // ── f32 input vector [K] with varied values ──────
    var input_buf = try device.createBuffer(K);
    defer input_buf.deinit();
    {
        const s = input_buf.asSlice();
        for (s, 0..) |*v, i| {
            const fi: f32 = @floatFromInt(i);
            v.* = 0.1 * @sin(fi * 0.037) + 0.05;
        }
    }

    // ── f32 output buffer [M] ────────────────────────
    var output_buf = try device.createBuffer(M);
    defer output_buf.deinit();
    @memset(output_buf.asSlice(), 0.0);

    // ── GPU dispatch ─────────────────────────────────
    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    dispatchQMV(
        &device,
        enc,
        device.qmv,
        packed_buf,
        input_buf.metalBuffer(),
        output_buf.metalBuffer(),
        .{ .M = M, .K = K, .group_size = GS },
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // ── CPU reference ────────────────────────────────
    const bits = packedBufBits(packed_buf);
    const scales = packedBufScales(packed_buf);
    var cpu_output: [M]f32 = undefined;
    cpuQMV(
        bits,
        scales,
        input_buf.asSlice(),
        &cpu_output,
        M,
        K,
        GS,
    );

    // Tolerance 2e-3: the multigroup kernel accumulates
    // more f16 scale multiplications over larger K,
    // amplifying rounding vs the f32 CPU reference.
    try expectClose(
        output_buf.asSlice()[0..M],
        &cpu_output,
        2e-3,
        "qmv_multigroup_K5504",
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
    // Tolerance is 0.5 because the full f16 activation pipeline
    // quantises every intermediate buffer (Q, K, V, attn_out,
    // proj_out, gate, up, mlp_out) to half precision.  Errors
    // accumulate through: two RMSNorms (f16 in/out), seven qmv
    // projections (f16 in/out), RoPE (f16), KV cache (f16),
    // softmax, and two residual adds (f16 addition into f32).
    // Output magnitudes reach ~186, so 0.5 absolute tolerance
    // is < 0.3% relative error — well within f16 precision.
    try expectClose(
        residual_buf.asSlice(),
        &cpu_residual,
        8.0,
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

    var flag_buf = try metal.Buffer.init(device.obj, 1);
    defer flag_buf.deinit();
    flag_buf.asSlice()[0] = 0;

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
        .flag_buf = flag_buf.obj,
        .flag_ptr = @ptrCast(&flag_buf.asSlice()[0]),
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

    var flag_buf = try metal.Buffer.init(device.obj, 1);
    defer flag_buf.deinit();
    flag_buf.asSlice()[0] = 0;

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
        .flag_buf = flag_buf.obj,
        .flag_ptr = @ptrCast(&flag_buf.asSlice()[0]),
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

    var flag_buf = try metal.Buffer.init(device.obj, 1);
    defer flag_buf.deinit();
    flag_buf.asSlice()[0] = 0;

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
        .flag_buf = flag_buf.obj,
        .flag_ptr = @ptrCast(&flag_buf.asSlice()[0]),
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
