const std = @import("std");
const objc = @import("objc");
const metal = @import("metal.zig");
const transformer = @import("transformer.zig");
const model = @import("model.zig");

const Model = model.Model;
const convertScalesAffineToSymmetric = model.convertScalesAffineToSymmetric;
const convertF32ToF16 = model.convertF32ToF16;

/// Tiny test config with 2 layers for verifying per-layer array
/// behaviour.  Dimensions chosen so that:
///   - every qmv dispatch has K % 32 == 0 (simd lane requirement),
///   - K >= group_size (128) for correct scale indexing.
const TestConfig = transformer.TransformerConfig(.{
    .vocab_size = 32,
    .hidden_size = 128,
    .intermediate_size = 256,
    .num_layers = 2,
    .num_query_heads = 2,
    .num_kv_heads = 2,
    .head_dim = 64,
    .max_context_length = 8,
    .max_prefill_length = 4,
    .rope_theta = 10000.0,
    .tie_word_embeddings = true,
});

test "Model allocates all buffers" {
    // Goal: Verify that init allocates all expected Metal buffers
    // with correct sizes, and that deinit releases cleanly.
    //
    // Method: Create a Model(TestConfig), check buffer counts and
    // element sizes against Config constants, then deinit.

    const Config = TestConfig;
    const device_obj = objc.Object.fromId(
        metal.MTLCreateSystemDefaultDevice() orelse
            return error.MetalNotAvailable,
    );

    var m: Model(Config) = undefined;
    try m.init(device_obj);
    defer m.deinit();

    // Embedding.
    try std.testing.expectEqual(
        Config.vocab_size * Config.hidden_size,
        m.embedding.packed_count,
    );
    try std.testing.expectEqual(
        @as(u32, 128),
        m.embedding.group_size,
    );

    // Final norm.
    try std.testing.expectEqual(
        Config.hidden_size,
        m.final_norm.len,
    );

    // Per-layer packed projections.
    for (0..Config.num_layers) |i| {
        try std.testing.expectEqual(
            Config.hidden_size * Config.query_dim,
            m.q_proj[i].packed_count,
        );
        try std.testing.expectEqual(
            Config.hidden_size * Config.kv_dim,
            m.k_proj[i].packed_count,
        );
        try std.testing.expectEqual(
            Config.hidden_size * Config.kv_dim,
            m.v_proj[i].packed_count,
        );
        try std.testing.expectEqual(
            Config.query_dim * Config.hidden_size,
            m.o_proj[i].packed_count,
        );
        try std.testing.expectEqual(
            Config.hidden_size * Config.intermediate_size,
            m.gate_proj[i].packed_count,
        );
        try std.testing.expectEqual(
            Config.hidden_size * Config.intermediate_size,
            m.up_proj[i].packed_count,
        );
        try std.testing.expectEqual(
            Config.intermediate_size * Config.hidden_size,
            m.down_proj[i].packed_count,
        );
    }

    // Per-layer norm scales.
    for (0..Config.num_layers) |i| {
        try std.testing.expectEqual(
            Config.hidden_size,
            m.attn_norm[i].len,
        );
        try std.testing.expectEqual(
            Config.hidden_size,
            m.ffn_norm[i].len,
        );
        try std.testing.expectEqual(
            Config.head_dim,
            m.q_norm[i].len,
        );
        try std.testing.expectEqual(
            Config.head_dim,
            m.k_norm[i].len,
        );
    }

    // Per-layer KV caches.
    const kv_cache_len: u32 =
        Config.max_context_length * Config.kv_dim;
    for (0..Config.num_layers) |i| {
        try std.testing.expectEqual(
            kv_cache_len,
            m.k_cache[i].len,
        );
        try std.testing.expectEqual(
            kv_cache_len,
            m.v_cache[i].len,
        );
    }

    // Activation scratch.
    try std.testing.expectEqual(
        Config.hidden_size,
        m.residual.len,
    );
    try std.testing.expectEqual(
        Config.hidden_size,
        m.norm_out.len,
    );
    try std.testing.expectEqual(
        Config.query_dim,
        m.q.len,
    );
    try std.testing.expectEqual(
        Config.kv_dim,
        m.k.len,
    );
    try std.testing.expectEqual(
        Config.kv_dim,
        m.v.len,
    );
    try std.testing.expectEqual(
        Config.query_dim,
        m.attn_out.len,
    );
    try std.testing.expectEqual(
        Config.hidden_size,
        m.proj_out.len,
    );
    try std.testing.expectEqual(
        Config.num_query_heads * Config.max_context_length,
        m.attn_scratch.len,
    );
    try std.testing.expectEqual(
        Config.intermediate_size,
        m.gate.len,
    );
    try std.testing.expectEqual(
        Config.intermediate_size,
        m.up.len,
    );
    try std.testing.expectEqual(
        Config.intermediate_size,
        m.mlp_out.len,
    );
}

test "forwardBlockArgs returns correct handles" {
    // Goal: Verify that forwardBlockArgs indexes into the correct
    // per-layer arrays and shares activation buffers across layers.
    //
    // Method: Init model with 2 layers, call forwardBlockArgs for
    // layers 0 and 1, verify different weight buffers but identical
    // activation buffer pointers.

    const Config = TestConfig;
    const device_obj = objc.Object.fromId(
        metal.MTLCreateSystemDefaultDevice() orelse
            return error.MetalNotAvailable,
    );

    var m: Model(Config) = undefined;
    try m.init(device_obj);
    defer m.deinit();

    const args0 = m.forwardBlockArgs(0, 0, 1);
    const args1 = m.forwardBlockArgs(1, 0, 1);

    // Per-layer buffers must differ between layers.
    try std.testing.expect(
        args0.q_proj.obj.value != args1.q_proj.obj.value,
    );
    try std.testing.expect(
        args0.k_proj.obj.value != args1.k_proj.obj.value,
    );
    try std.testing.expect(
        args0.gate_proj.obj.value !=
            args1.gate_proj.obj.value,
    );
    try std.testing.expect(
        args0.attn_norm_scale.value !=
            args1.attn_norm_scale.value,
    );
    try std.testing.expect(
        args0.k_cache.value != args1.k_cache.value,
    );
    try std.testing.expect(
        args0.v_cache.value != args1.v_cache.value,
    );

    // Activation scratch must be shared (same pointer).
    try std.testing.expectEqual(
        args0.residual.value,
        args1.residual.value,
    );
    try std.testing.expectEqual(
        args0.norm_out.value,
        args1.norm_out.value,
    );
    try std.testing.expectEqual(
        args0.q.value,
        args1.q.value,
    );
    try std.testing.expectEqual(
        args0.attn_scratch.value,
        args1.attn_scratch.value,
    );
    try std.testing.expectEqual(
        args0.gate.value,
        args1.gate.value,
    );
    try std.testing.expectEqual(
        args0.mlp_out.value,
        args1.mlp_out.value,
    );

    // Sequence state must match what was requested.
    try std.testing.expectEqual(@as(u32, 0), args0.position);
    try std.testing.expectEqual(@as(u32, 1), args0.seq_len);
    try std.testing.expectEqual(@as(u32, 0), args1.position);
    try std.testing.expectEqual(@as(u32, 1), args1.seq_len);
}

test "scale conversion divides by two" {
    // Goal: Verify that convertScalesAffineToSymmetric correctly
    // halves each f16 scale value.
    //
    // Method: Hand-picked f16 values covering normal, small, and
    // negative cases.  Verify output ≈ input / 2.

    const source = [_]f16{
        @as(f16, 1.0),
        @as(f16, 2.0),
        @as(f16, 0.5),
        @as(f16, -1.0),
        @as(f16, 0.125),
    };
    var target: [5]f16 = undefined;
    convertScalesAffineToSymmetric(&source, &target);

    // f16 has ~3 decimal digits of precision; 1e-4 tolerance
    // is well within that.
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.5),
        @as(f32, @floatCast(target[0])),
        1e-4,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        @as(f32, @floatCast(target[1])),
        1e-4,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.25),
        @as(f32, @floatCast(target[2])),
        1e-4,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, -0.5),
        @as(f32, @floatCast(target[3])),
        1e-4,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 0.0625),
        @as(f32, @floatCast(target[4])),
        1e-4,
    );
}

test "f32 to f16 conversion" {
    // Goal: Verify convertF32ToF16 correctly narrows f32 to f16.
    //
    // Method: Convert known values and check round-trip accuracy.

    const source = [_]f32{ 1.0, -0.5, 0.0, 3.14159, 100.0 };
    var target: [5]f16 = undefined;
    convertF32ToF16(&source, &target);

    // f16 precision: values near 1.0 have ~1e-3 precision;
    // 100.0 has ~0.1 precision.  Use 0.05 tolerance.
    for (source, target) |s, t| {
        try std.testing.expectApproxEqAbs(
            s,
            @as(f32, @floatCast(t)),
            0.05,
        );
    }
}
