//! Bonsai Q4 researcher — configures the generic
//! toolbox for Q4 engine code optimisation.
//!
//! Thin wrapper: all tool logic lives in toolbox.zig.
//!
//! Usage:
//!   zig build
//!   ./zig-out/bin/bonsai_q4_researcher <tool> [args...]

const toolbox = @import("toolbox.zig");

// ============================================================
// Configuration
// ============================================================

const config = toolbox.ToolboxConfig{
    .name = "bonsai_q4",
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
        ".bonsai_q4_history/",
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
    .test_command = &.{
        "zig",
        "build",
        "run-bonsai-q4-golden",
        "-Doptimize=ReleaseFast",
    },
    .bench_command = &.{
        "zig",
        "build",
        "run-bonsai-q4-bench",
        "-Doptimize=ReleaseFast",
    },
    .reference_bench_command = &.{
        "../reference/.venv/bin/python",
        "../reference/mlx_bonsai.py",
    },
    .reference_cwd = "../nnmetal",
    .extra_bench = &.{},
    .history_dir = ".bonsai_q4_history",
};

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    toolbox.run(&config);
}
