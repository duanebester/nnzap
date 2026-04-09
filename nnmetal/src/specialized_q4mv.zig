// specialized_q4mv.zig — Compile-time specialized Q4 QMV kernels
//
// Generates Metal shader source with model-specific K and
// group_size baked in as constexpr, then compiles and installs
// the resulting pipelines on the Device at init time.
//
// Q4 MLX format: 4-bit unsigned nibbles packed into uint32,
// with per-group raw BF16 scales and biases (affine quant).
// GPU kernels convert BF16→F32 inline and BF16-round all
// output writes to match MLX's BF16 arithmetic precision.

const std = @import("std");
const metal = @import("metal.zig");
const objc = @import("objc");
const log = std.log.scoped(.specialized_q4mv);

/// Generate the full Metal source for specialized Q4 QMV kernels.
pub fn shaderSource(
    comptime hidden_K: u32,
    comptime inter_K: u32,
    comptime group_size: u32,
) [:0]const u8 {
    return std.fmt.comptimePrint(
        "#define SPEC_HIDDEN_K {d}\n" ++
            "#define SPEC_INTER_K {d}\n" ++
            "#define SPEC_GS {d}\n",
        .{ hidden_K, inter_K, group_size },
    ) ++ @embedFile("shaders/q4mv_bf16_specialized.metal");
}

/// Compile the specialized Q4 QMV library and install the
/// resulting pipelines on the Device.
pub fn initOnDevice(
    device: *metal.Device,
    comptime hidden_K: u32,
    comptime inter_K: u32,
    comptime group_size: u32,
) !void {
    // Compile-time validation.
    comptime {
        std.debug.assert(hidden_K % 32 == 0);
        std.debug.assert(inter_K % 32 == 0);
        std.debug.assert(group_size > 0);
        std.debug.assert(hidden_K >= 256);
        std.debug.assert(inter_K >= 256);
    }

    const source = comptime shaderSource(
        hidden_K,
        inter_K,
        group_size,
    );

    const device_obj = device.obj;
    std.debug.assert(device_obj.value != null);

    const lib = try metal.compileLibraryFromSource(
        device_obj,
        source.ptr,
    );

    // Create pipeline states for each specialized kernel.
    device.spec_q4mv_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_f16io",
        );

    device.spec_q4mv_f16io_resadd =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_f16io_resadd",
        );

    device.spec_q4mv_fused_pair_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_fused_pair_f16io",
        );

    device.spec_q4mv_f16in =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_f16in",
        );

    device.spec_q4mv_mg_f16io_resadd =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_mg_f16io_resadd",
        );

    device.spec_q4mv_fused_pair_silu_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_fused_pair_silu_f16io",
        );

    device.spec_q4mv_fused_norm_pair_silu_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_fused_norm_pair_silu_f16io",
        );

    device.spec_q4mv_fused_norm_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_fused_norm_f16io",
        );

    device.spec_q4mv_fused_norm_pair_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "q4mv_spec_fused_norm_pair_f16io",
        );

    // Record which K values these pipelines target so
    // dispatch functions can verify dims.K matches.
    device.spec_q4_hidden_K = hidden_K;
    device.spec_q4_inter_K = inter_K;

    // Log max threads so we can diagnose fallbacks when
    // register pressure from unrolling exceeds 512.
    log.info(
        "Specialized Q4 QMV pipelines compiled: " ++
            "hidden_K={d}, inter_K={d}, gs={d}.",
        .{ hidden_K, inter_K, group_size },
    );
    log.info(
        "  f16io={d}  resadd={d}  fused={d}" ++
            "  f16in={d}  mg_resadd={d} max_tpg",
        .{
            device.spec_q4mv_f16io.?
                .max_threads_per_group,
            device.spec_q4mv_f16io_resadd.?
                .max_threads_per_group,
            device.spec_q4mv_fused_pair_f16io.?
                .max_threads_per_group,
            device.spec_q4mv_f16in.?
                .max_threads_per_group,
            device.spec_q4mv_mg_f16io_resadd.?
                .max_threads_per_group,
        },
    );
}
