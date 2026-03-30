//! Comptime network layout
//!
//! Define a neural network architecture at compile time. All buffer sizes,
//! weight/bias offsets, and activation sizes are resolved by the compiler.
//! At runtime we just index into flat shared buffers with known offsets.

const std = @import("std");

// -- Hard limits (Rule 4) --

const MAX_LAYERS: u32 = 64;
const MAX_PARAMS_PER_NETWORK: u32 = 16 * 1024 * 1024; // 16M parameters

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
                "|       nnzap Network Layout        |\n",
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
