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
    .project_root = "../nnmetal",
    .write_scope = &.{
        "nnmetal/src/",
        "nnmetal/examples/",
    },
    .read_scope = &.{
        "nnmetal/",
        "labrat/src/",
        "labrat/programs/",
        "docs/",
        "benchmarks/",
        ".bonsai_history/",
    },
    .read_files = &.{
        "README.md",
        "CLAUDE.md",
        "labrat/build.zig",
        "labrat/build.zig.zon",
        "nnmetal/build.zig",
        "nnmetal/build.zig.zon",
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
    .history_dir = ".bonsai_history",
};

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    toolbox.run(&config);
}
