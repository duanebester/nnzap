//! nnzap — GPU-accelerated neural network library for Apple Silicon
//!
//! Uses Metal compute shaders + Zig comptime for zero-copy, zero-overhead
//! neural network training and inference on Apple Silicon unified memory.
//!
//! Architecture:
//!   Comptime: Network shapes, offsets, buffer sizes resolved at compile time.
//!   Runtime:  Metal shared buffers (zero-copy) + compute shaders for the math.
//!   Overlap:  Double-buffered activations let CPU prep the next batch while
//!             the GPU crunches the current one.

pub const metal = @import("metal.zig");
pub const layout = @import("layout.zig");
pub const network = @import("network.zig");
pub const mnist = @import("mnist.zig");
pub const benchmark = @import("benchmark.zig");

// Re-exports for convenience
pub const Device = metal.Device;
pub const Buffer = metal.Buffer;
pub const MultiBuffered = metal.MultiBuffered;
pub const NetworkLayout = layout.NetworkLayout;
pub const LayerDesc = layout.LayerDesc;
pub const Activation = layout.Activation;
pub const Network = network.Network;
pub const Mnist = mnist.Mnist;
pub const Benchmark = benchmark.Benchmark;

test {
    @import("std").testing.refAllDecls(@This());
}
