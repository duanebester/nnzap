//! Comptime network layout
//!
//! Define a neural network architecture at compile time. All buffer sizes,
//! weight/bias offsets, and activation sizes are resolved by the compiler.
//! At runtime we just index into flat shared buffers with known offsets.

const std = @import("std");

// -- Hard limits (Rule 4) --

const MAX_LAYERS: u32 = 64;
const MAX_PARAMS_PER_NETWORK: u32 = 16 * 1024 * 1024; // 16M parameters

/// Ceiling division: divCeil(a, b) = ceil(a / b).
/// Both operands must be positive.  Uses @divFloor to show intent
/// explicitly (Rule 18).
pub fn divCeil(a: u32, b: u32) u32 {
    std.debug.assert(b > 0);
    std.debug.assert(a > 0 or b > 0); // At least one positive.
    return @as(
        u32,
        @intCast(std.math.divCeil(u32, a, b) catch unreachable),
    );
}

/// Supported activation functions.
pub const Activation = enum {
    relu,
    tanh_act,
    sigmoid,
    none,
};

/// Describes one dense (fully-connected) layer.
pub const LayerDesc = struct {
    in: u32,
    out: u32,
    act: Activation = .relu,
};

/// Comptime-resolved network layout.
/// All fields are known at compile time -- zero runtime cost.
pub fn NetworkLayout(comptime arch: []const LayerDesc) type {
    const n = arch.len;

    return struct {
        // A network must have at least one layer.
        comptime {
            std.debug.assert(n > 0);
        }

        // Layer count must not exceed the hard cap.
        comptime {
            std.debug.assert(n <= MAX_LAYERS);
        }

        // Comptime adjacency validation: layer i's output must
        // match layer i+1's input (Rule 14). This is a
        // compile-time safety net — mismatched architectures
        // cannot be tested at runtime because they produce a
        // comptime error.
        comptime {
            for (0..n - 1) |i| {
                if (arch[i].out != arch[i + 1].in) {
                    @compileError(
                        std.fmt.comptimePrint(
                            "Layer {d} output ({d}) != " ++ "layer {d} input ({d}).",
                            .{
                                i,
                                arch[i].out,
                                i + 1,
                                arch[i + 1].in,
                            },
                        ),
                    );
                }
            }
        }

        pub const num_layers: u32 = n;
        pub const layers: [n]LayerDesc = arch[0..n].*;

        // -- Parameter counts --

        pub const weight_counts: [n]u32 = blk: {
            var counts: [n]u32 = undefined;
            for (0..n) |i| {
                counts[i] = @as(u32, arch[i].in) * @as(u32, arch[i].out);
            }
            break :blk counts;
        };

        pub const bias_counts: [n]u32 = blk: {
            var counts: [n]u32 = undefined;
            for (0..n) |i| {
                counts[i] = @as(u32, arch[i].out);
            }
            break :blk counts;
        };

        /// Total number of trainable parameters (weights + biases).
        pub const param_count: u32 = blk: {
            var total: u32 = 0;
            for (0..n) |i| {
                total += weight_counts[i] + bias_counts[i];
            }
            break :blk total;
        };

        // The network must have at least some parameters.
        comptime {
            std.debug.assert(param_count > 0);
        }

        // Total parameters must not exceed the hard cap.
        comptime {
            std.debug.assert(param_count <= MAX_PARAMS_PER_NETWORK);
        }

        // -- Offsets into the flat parameter buffer --

        /// weight_offsets[i] = start index of layer i's weights
        /// in the param buffer.
        pub const weight_offsets: [n]u32 = blk: {
            var offsets: [n]u32 = undefined;
            var offset: u32 = 0;
            for (0..n) |i| {
                offsets[i] = offset;
                offset += weight_counts[i] + bias_counts[i];
            }
            break :blk offsets;
        };

        /// bias_offsets[i] = start index of layer i's biases
        /// in the param buffer.
        pub const bias_offsets: [n]u32 = blk: {
            var offsets: [n]u32 = undefined;
            for (0..n) |i| {
                offsets[i] = weight_offsets[i] + weight_counts[i];
            }
            break :blk offsets;
        };

        // -- Activation buffer sizes --

        /// activation_sizes[i] = number of floats for layer i's
        /// output.
        pub const activation_sizes: [n]u32 = blk: {
            var sizes: [n]u32 = undefined;
            for (0..n) |i| {
                sizes[i] = @as(u32, arch[i].out);
            }
            break :blk sizes;
        };

        /// Largest single activation buffer needed (for
        /// double-buffering).
        pub const max_activation_size: u32 = blk: {
            var max: u32 = 0;
            for (0..n) |i| {
                if (activation_sizes[i] > max) {
                    max = activation_sizes[i];
                }
            }
            break :blk max;
        };

        /// Input size of the first layer.
        pub const input_size: u32 = @as(u32, arch[0].in);

        /// Output size of the last layer.
        pub const output_size: u32 =
            @as(u32, arch[n - 1].out);

        // -- 1-bit packed weight sizes (Q1_0_g128 format) --
        // Group size: 128 weights share one f16 scale.

        pub const PACK_GROUP_SIZE: u32 = 128;

        /// Packed bytes for a single weight matrix [rows × cols].
        /// Layout: ceil(rows*cols / 8) packed bytes
        ///       + ceil(rows*cols / 128) × 2 scale bytes (f16).
        pub fn packedWeightBytes(
            comptime rows: u32,
            comptime cols: u32,
        ) u32 {
            comptime {
                std.debug.assert(rows > 0);
                std.debug.assert(cols > 0);
            }
            const total_weights =
                @as(u32, rows) * @as(u32, cols);
            const bit_bytes = std.math.divCeil(
                u32,
                total_weights,
                8,
            ) catch unreachable;
            const num_groups = std.math.divCeil(
                u32,
                total_weights,
                PACK_GROUP_SIZE,
            ) catch unreachable;
            const scale_bytes = num_groups * 2; // f16 = 2 bytes.
            return bit_bytes + scale_bytes;
        }

        /// Per-layer packed weight sizes in bytes.
        pub const packed_weight_bytes: [n]u32 = blk: {
            var sizes: [n]u32 = undefined;
            for (0..n) |i| {
                sizes[i] = packedWeightBytes(
                    arch[i].in,
                    arch[i].out,
                );
            }
            break :blk sizes;
        };

        /// Per-layer packed byte offsets into a flat packed
        /// buffer.
        pub const packed_weight_offsets: [n]u32 = blk: {
            var offsets: [n]u32 = undefined;
            var offset: u32 = 0;
            for (0..n) |i| {
                offsets[i] = offset;
                offset += packed_weight_bytes[i];
            }
            break :blk offsets;
        };

        /// Total packed weight bytes across all layers
        /// (no biases — biases remain f32).
        pub const total_packed_weight_bytes: u32 = blk: {
            var total: u32 = 0;
            for (0..n) |i| {
                total += packed_weight_bytes[i];
            }
            break :blk total;
        };

        /// Number of packed bit-bytes for a layer's weight
        /// matrix.
        pub fn packedBitBytes(comptime layer: u32) u32 {
            comptime {
                std.debug.assert(layer < n);
            }
            return std.math.divCeil(
                u32,
                weight_counts[layer],
                8,
            ) catch unreachable;
        }

        /// Number of scale groups for a layer's weight matrix.
        pub fn numScaleGroups(comptime layer: u32) u32 {
            comptime {
                std.debug.assert(layer < n);
            }
            return std.math.divCeil(
                u32,
                weight_counts[layer],
                PACK_GROUP_SIZE,
            ) catch unreachable;
        }

        /// Byte offset where scales begin within a layer's
        /// packed region (= packedBitBytes for that layer).
        pub fn scaleByteOffset(comptime layer: u32) u32 {
            return packedBitBytes(layer);
        }

        // -- Slice helpers for indexing into flat param buffers --

        /// Return the weight slice for layer `layer` from a flat
        /// parameter buffer. The offset and length are
        /// comptime-known — this compiles to pointer arithmetic.
        pub fn getWeightSlice(
            params: []f32,
            comptime layer: u32,
        ) []f32 {
            comptime {
                std.debug.assert(layer < n);
            }
            const offset: usize = weight_offsets[layer]; // Widen to usize for slice indexing.
            const len: usize = weight_counts[layer];
            std.debug.assert(params.len >= offset + len);
            return params[offset..][0..len];
        }

        /// Return the bias slice for layer `layer` from a flat
        /// parameter buffer. The offset and length are
        /// comptime-known — this compiles to pointer arithmetic.
        pub fn getBiasSlice(
            params: []f32,
            comptime layer: u32,
        ) []f32 {
            comptime {
                std.debug.assert(layer < n);
            }
            const offset: usize = bias_offsets[layer]; // Widen to usize for slice indexing.
            const len: usize = bias_counts[layer];
            std.debug.assert(params.len >= offset + len);
            return params[offset..][0..len];
        }

        // -- Debug / logging --

        pub fn printSummary() void {
            std.debug.print(
                "\n+-----------------------------------+\n",
                .{},
            );
            std.debug.print(
                "|      nnmetal Network Layout       |\n",
                .{},
            );
            std.debug.print(
                "+-----------------------------------+\n",
                .{},
            );
            for (0..n) |i| {
                std.debug.print(
                    "|  Layer {d}: {d:>4} -> {d:<4}" ++ "  ({s:<8}) |\n",
                    .{
                        i,
                        arch[i].in,
                        arch[i].out,
                        @tagName(arch[i].act),
                    },
                );
            }
            std.debug.print(
                "+-----------------------------------+\n",
                .{},
            );
            std.debug.print(
                "|  Total params: {d:<16} |\n",
                .{param_count},
            );
            std.debug.print(
                "|  Max activation: {d:<14} |\n",
                .{max_activation_size},
            );
            std.debug.print(
                "+-----------------------------------+\n\n",
                .{},
            );
        }
    };
}

// -- Tests --

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
    const divCeil_fn = @import("layout.zig").divCeil;
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
