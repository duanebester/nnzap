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

test {
    _ = @import("layout_test.zig");
}
