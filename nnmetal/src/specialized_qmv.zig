// specialized_qmv.zig — Compile-time specialized QMV kernels
//
// Generates Metal shader source with model-specific K and
// group_size baked in as constexpr, then compiles and installs
// the resulting pipelines on the Device at init time.

const std = @import("std");
const metal = @import("metal.zig");
const objc = @import("objc");
const log = std.log.scoped(.specialized_qmv);

/// Generate the full Metal source for specialized QMV kernels.
/// Prepends #define macros for the model's hidden_size K,
/// intermediate_size K, and group_size before the template
/// source from qmv_specialized.metal.
///
/// Returns a comptime null-terminated string suitable for
/// passing to compileLibraryFromSource.
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
    ) ++ @embedFile("shaders/qmv_specialized.metal");
}

/// Compile the specialized QMV library and install the
/// resulting pipelines on the Device.  Call this once during
/// transformer initialisation, after Device.init().
///
/// The comptime parameters come from the TransformerConfig:
///   hidden_K    = Config.hidden_size  (Q/K/V/O/gate/up)
///   inter_K     = Config.intermediate_size  (down proj)
///   group_size  = Config.group_size  (always 128)
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
    device.spec_qmv_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "qmv_spec_f16io",
        );

    device.spec_qmv_f16io_resadd =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "qmv_spec_f16io_resadd",
        );

    device.spec_qmv_fused_pair_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "qmv_spec_fused_pair_f16io",
        );

    device.spec_qmv_f16in =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "qmv_spec_f16in",
        );

    device.spec_qmv_mg_f16io_resadd =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "qmv_spec_mg_f16io_resadd",
        );

    device.spec_qmv_fused_pair_silu_f16io =
        try metal.ComputePipeline.init(
            device_obj,
            lib,
            "qmv_spec_fused_pair_silu_f16io",
        );

    // Record which K values these pipelines target so
    // dispatch functions can verify dims.K matches.
    device.spec_hidden_K = hidden_K;
    device.spec_inter_K = inter_K;

    // Log max threads so we can diagnose fallbacks when
    // register pressure from unrolling exceeds 512.
    log.info(
        "Specialized QMV pipelines compiled: " ++
            "hidden_K={d}, inter_K={d}, gs={d}.",
        .{ hidden_K, inter_K, group_size },
    );
    log.info(
        "  f16io={d}  resadd={d}  fused={d}" ++
            "  f16in={d}  mg_resadd={d} max_tpg",
        .{
            device.spec_qmv_f16io.?.max_threads_per_group,
            device.spec_qmv_f16io_resadd.?
                .max_threads_per_group,
            device.spec_qmv_fused_pair_f16io.?
                .max_threads_per_group,
            device.spec_qmv_f16in.?
                .max_threads_per_group,
            device.spec_qmv_mg_f16io_resadd.?
                .max_threads_per_group,
        },
    );
}
