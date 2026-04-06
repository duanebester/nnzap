//! Bonsai researcher — configures the generic
//! toolbox for engine code optimisation.
//!
//! Thin wrapper: all tool logic lives in toolbox.zig.
//!
//! Usage:
//!   zig build
//!   ./zig-out/bin/bonsai_researcher <tool> [args...]

const toolbox = @import("toolbox.zig");

// ============================================================
// Configuration
// ============================================================

const config = toolbox.ToolboxConfig{
    .name = "bonsai",
    .project_root = "../nn",
    .write_scope = &.{
        "nn/src/transformer.zig",
        "nn/src/network.zig",
        "nn/src/layout.zig",
        "nn/src/model.zig",
        "nn/src/metal.zig",
        "nn/src/safetensors.zig",
        "nn/src/tokenizer.zig",
        "nn/src/shaders/transformer.metal",
        "nn/src/shaders/compute.metal",
        "nn/examples/bonsai.zig",
        "nn/examples/bonsai_bench.zig",
    },
    .read_scope = &.{
        "nn/src/",
        "nn/examples/",
        "src/",
        "programs/",
        "docs/",
        "data/",
        "benchmarks/",
        ".bonsai_history/",
        ".engine_snapshots/",
    },
    .read_files = &.{
        "README.md",
        "CLAUDE.md",
        "nn/build.zig",
        "nn/build.zig.zon",
        "build.zig",
        "build.zig.zon",
    },
    .engine_files = &.{
        "nn/src/transformer.zig",
        "nn/src/network.zig",
        "nn/src/layout.zig",
        "nn/src/model.zig",
        "nn/src/metal.zig",
        "nn/src/safetensors.zig",
        "nn/src/tokenizer.zig",
        "nn/src/shaders/transformer.metal",
        "nn/src/shaders/compute.metal",
        "nn/examples/bonsai.zig",
        "nn/examples/bonsai_bench.zig",
    },
    .check_command = &.{ "zig", "build" },
    .test_command = &.{ "zig", "build", "test" },
    .bench_command = &.{
        "zig",
        "build",
        "run-bonsai-bench",
        "-Doptimize=ReleaseFast",
    },
    .extra_bench = &.{},
    .bench_dir = "nn/benchmarks",
    .bench_prefixes = &.{"bonsai_bench_"},
    .history_dir = ".bonsai_history",
    .snapshot_dir = ".engine_snapshots",
};

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    toolbox.run(&config);
}
