//! nn — GPU-accelerated neural network library for Apple Silicon
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
pub const transformer = @import("transformer.zig");
pub const safetensors = @import("safetensors.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const model = @import("model.zig");
pub const specialized_qmv = @import("specialized_qmv.zig");

// Re-exports for convenience
pub const Device = metal.Device;
pub const Buffer = metal.Buffer;
pub const PackedBuffer = metal.PackedBuffer;
pub const MultiBuffered = metal.MultiBuffered;
pub const NetworkLayout = layout.NetworkLayout;
pub const LayerDesc = layout.LayerDesc;
pub const Activation = layout.Activation;
pub const divCeil = layout.divCeil;
pub const Network = network.Network;
pub const Mnist = mnist.Mnist;
pub const Benchmark = benchmark.Benchmark;
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerPipelines = transformer.TransformerPipelines;
pub const Bonsai1_7B = transformer.Bonsai1_7B;
pub const Bonsai4B = transformer.Bonsai4B;
pub const Bonsai8B = transformer.Bonsai8B;
pub const SamplingParams = transformer.SamplingParams;
pub const GenerateResult = transformer.GenerateResult;
pub const GenerateOpts = transformer.GenerateOpts;
pub const SafetensorsFile = safetensors.SafetensorsFile;
pub const Tokenizer = tokenizer.Tokenizer;
pub const Model = model.Model;

test {
    @import("std").testing.refAllDecls(@This());
}
