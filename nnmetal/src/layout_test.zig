const std = @import("std");
const layout = @import("layout.zig");
const LayerDesc = layout.LayerDesc;
const NetworkLayout = layout.NetworkLayout;

test "MNIST-like layout" {
    const arch = [_]LayerDesc{
        .{ .in = 784, .out = 128, .act = .relu },
        .{ .in = 128, .out = 64, .act = .relu },
        .{ .in = 64, .out = 10, .act = .tanh_act },
    };
    const Layout = NetworkLayout(&arch);

    // Weight counts: 784*128=100352, 128*64=8192, 64*10=640.
    try std.testing.expectEqual(
        @as(u32, 100352),
        Layout.weight_counts[0],
    );
    try std.testing.expectEqual(
        @as(u32, 8192),
        Layout.weight_counts[1],
    );
    try std.testing.expectEqual(
        @as(u32, 640),
        Layout.weight_counts[2],
    );

    // Bias counts: 128, 64, 10.
    try std.testing.expectEqual(
        @as(u32, 128),
        Layout.bias_counts[0],
    );

    // Total params: 100352+128 + 8192+64 + 640+10 = 109386.
    try std.testing.expectEqual(
        @as(u32, 109386),
        Layout.param_count,
    );

    // Offsets.
    try std.testing.expectEqual(
        @as(u32, 0),
        Layout.weight_offsets[0],
    );
    try std.testing.expectEqual(
        @as(u32, 100480), // 100352+128
        Layout.weight_offsets[1],
    );
    try std.testing.expectEqual(
        @as(u32, 100352),
        Layout.bias_offsets[0],
    );

    // Activation sizes.
    try std.testing.expectEqual(
        @as(u32, 128),
        Layout.max_activation_size,
    );
    try std.testing.expectEqual(
        @as(u32, 784),
        Layout.input_size,
    );
    try std.testing.expectEqual(
        @as(u32, 10),
        Layout.output_size,
    );
}

test "tiny layout" {
    const arch = [_]LayerDesc{
        .{ .in = 2, .out = 4, .act = .relu },
        .{ .in = 4, .out = 1, .act = .sigmoid },
    };
    const Layout = NetworkLayout(&arch);

    try std.testing.expectEqual(
        @as(u32, 2),
        Layout.num_layers,
    );
    try std.testing.expectEqual(
        @as(u32, 8 + 4 + 4 + 1), // 17
        Layout.param_count,
    );
    try std.testing.expectEqual(
        @as(u32, 4),
        Layout.max_activation_size,
    );
}

test "getWeightSlice and getBiasSlice" {
    const arch = [_]LayerDesc{
        .{ .in = 2, .out = 4, .act = .relu },
        .{ .in = 4, .out = 1, .act = .sigmoid },
    };
    const Layout = NetworkLayout(&arch);

    // Total params: 2*4 + 4 + 4*1 + 1 = 17.
    var params: [Layout.param_count]f32 = undefined;

    // Fill weights for layer 0 (2*4=8 floats at offset 0).
    const w0 = Layout.getWeightSlice(&params, 0);
    try std.testing.expectEqual(
        @as(usize, 8),
        w0.len,
    );
    for (w0, 0..) |*v, i| {
        v.* = @floatFromInt(i);
    }

    // Fill biases for layer 0 (4 floats at offset 8).
    const b0 = Layout.getBiasSlice(&params, 0);
    try std.testing.expectEqual(
        @as(usize, 4),
        b0.len,
    );

    // Layer 1 weights (4*1=4 floats at offset 12).
    const w1 = Layout.getWeightSlice(&params, 1);
    try std.testing.expectEqual(
        @as(usize, 4),
        w1.len,
    );

    // Layer 1 biases (1 float at offset 16).
    const b1 = Layout.getBiasSlice(&params, 1);
    try std.testing.expectEqual(
        @as(usize, 1),
        b1.len,
    );

    // Verify slices don't overlap and cover the full buffer.
    try std.testing.expectEqual(
        Layout.weight_offsets[0],
        0,
    );
    try std.testing.expectEqual(
        Layout.bias_offsets[0],
        8,
    );
    try std.testing.expectEqual(
        Layout.weight_offsets[1],
        12,
    );
    try std.testing.expectEqual(
        Layout.bias_offsets[1],
        16,
    );
}

// Verify that a well-formed multi-layer architecture passes
// comptime adjacency validation. A mismatched architecture
// (e.g., layer 0 output=16 feeding layer 1 input=8) would
// produce a @compileError and cannot be tested at runtime —
// that is by design. This test documents the positive case.
test "comptime adjacency validation" {
    // Each layer's output matches the next layer's input.
    // Instantiation succeeds because adjacency checks pass.
    const arch = [_]LayerDesc{
        .{ .in = 8, .out = 16, .act = .relu },
        .{ .in = 16, .out = 4, .act = .sigmoid },
        .{ .in = 4, .out = 1, .act = .none },
    };
    const Layout = NetworkLayout(&arch);

    try std.testing.expectEqual(
        @as(u32, 3),
        Layout.num_layers,
    );
    try std.testing.expectEqual(
        @as(u32, 8),
        Layout.input_size,
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        Layout.output_size,
    );

    // Params: 8*16+16 + 16*4+4 + 4*1+1 = 144+68+5 = 217.
    try std.testing.expectEqual(
        @as(u32, 217),
        Layout.param_count,
    );
}

test "packed weight sizes — MNIST layout" {
    const arch = [_]LayerDesc{
        .{ .in = 784, .out = 128, .act = .relu },
        .{ .in = 128, .out = 64, .act = .relu },
        .{ .in = 64, .out = 10, .act = .tanh_act },
    };
    const Layout = NetworkLayout(&arch);

    // Layer 0: 784 × 128 = 100352 weights.
    // Packed bit bytes: ceil(100352 / 8) = 12544.
    // Scale groups: ceil(100352 / 128) = 784.
    // Scale bytes: 784 × 2 = 1568.
    // Total: 12544 + 1568 = 14112.
    try std.testing.expectEqual(
        @as(u32, 14112),
        Layout.packed_weight_bytes[0],
    );

    // Layer 1: 128 × 64 = 8192 weights.
    // Bit bytes: 8192 / 8 = 1024.
    // Groups: 8192 / 128 = 64.
    // Scale bytes: 64 × 2 = 128.
    // Total: 1024 + 128 = 1152.
    try std.testing.expectEqual(
        @as(u32, 1152),
        Layout.packed_weight_bytes[1],
    );

    // Layer 2: 64 × 10 = 640 weights.
    // Bit bytes: ceil(640 / 8) = 80.
    // Groups: ceil(640 / 128) = 5.
    // Scale bytes: 5 × 2 = 10.
    // Total: 80 + 10 = 90.
    try std.testing.expectEqual(
        @as(u32, 90),
        Layout.packed_weight_bytes[2],
    );

    // Offsets: 0, 14112, 14112+1152=15264.
    try std.testing.expectEqual(
        @as(u32, 0),
        Layout.packed_weight_offsets[0],
    );
    try std.testing.expectEqual(
        @as(u32, 14112),
        Layout.packed_weight_offsets[1],
    );
    try std.testing.expectEqual(
        @as(u32, 15264),
        Layout.packed_weight_offsets[2],
    );

    // Total: 14112 + 1152 + 90 = 15354.
    try std.testing.expectEqual(
        @as(u32, 15354),
        Layout.total_packed_weight_bytes,
    );

    // Per-layer helpers.
    try std.testing.expectEqual(
        @as(u32, 12544),
        Layout.packedBitBytes(0),
    );
    try std.testing.expectEqual(
        @as(u32, 784),
        Layout.numScaleGroups(0),
    );
    try std.testing.expectEqual(
        @as(u32, 12544),
        Layout.scaleByteOffset(0),
    );
}

test "divCeil" {
    const divCeil_fn = layout.divCeil;
    try std.testing.expectEqual(
        @as(u32, 1),
        divCeil_fn(1, 1),
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        divCeil_fn(7, 8),
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        divCeil_fn(8, 8),
    );
    try std.testing.expectEqual(
        @as(u32, 2),
        divCeil_fn(9, 8),
    );
    try std.testing.expectEqual(
        @as(u32, 16),
        divCeil_fn(128, 8),
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        divCeil_fn(128, 128),
    );
    try std.testing.expectEqual(
        @as(u32, 5),
        divCeil_fn(640, 128),
    );
}
