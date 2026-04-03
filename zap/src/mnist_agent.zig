//! MNIST training agent — thin profile on agent_core.
//!
//! Configures the generic agent loop for MNIST
//! hyperparameter optimisation.  All tool logic lives
//! in the mnist_research toolbox binary.
//!
//! Usage:
//!   export ANTHROPIC_API_KEY=sk-ant-...
//!   zig build
//!   ./zig-out/bin/mnist_agent

const std = @import("std");
const core = @import("agent_core.zig");

// ============================================================
// Tool mapping table
//
// Maps LLM tool names to mnist_research CLI subcommands.
// Seven tools, all simple: six no_input, one json_payload.
// ============================================================

const mnist_tools = [_]core.ToolMapping{
    .{
        .tool_name = "config_show",
        .subcommand = "config-show",
        .shape = .no_input,
    },
    .{
        .tool_name = "config_set",
        .subcommand = "config-set",
        .shape = .json_payload,
    },
    .{
        .tool_name = "config_backup",
        .subcommand = "config-backup",
        .shape = .no_input,
    },
    .{
        .tool_name = "config_restore",
        .subcommand = "config-restore",
        .shape = .no_input,
    },
    .{
        .tool_name = "train",
        .subcommand = "train",
        .shape = .no_input,
    },
    .{
        .tool_name = "benchmark_latest",
        .subcommand = "benchmark-latest",
        .shape = .no_input,
    },
    .{
        .tool_name = "benchmark_compare",
        .subcommand = "benchmark-compare",
        .shape = .no_input,
    },
};

// ============================================================
// Profile configuration
// ============================================================

const config = core.AgentConfig{
    .name = "mnist",
    .toolbox_path = "./zig-out/bin/mnist_research",
    .history_dir = ".mnist_history",
    .system_prompt_path = "programs/mnist_system.md",
    .tool_schemas_path = "programs/mnist_tools.json",
    .tool_map = &mnist_tools,
    .persist_tools = &.{"train"},
    .build_context_fn = &buildMnistContext,
};

// ============================================================
// Context builder
//
// MNIST context is simple: experiment history plus a
// short instruction.  No engineering rules, no source
// code context, no summaries file.
// ============================================================

fn buildMnistContext(
    arena: std.mem.Allocator,
    cfg: *const core.AgentConfig,
) []const u8 {
    const history = core.buildHistorySummary(
        arena,
        cfg.history_dir,
    );
    const has_history = history.len > 0;

    const intro = if (has_history)
        "Here is the experiment history from " ++
            "previous agent runs (compact summary " ++
            "— one row per experiment):\n\n"
    else
        "No previous experiment history. " ++
            "This is the first run.\n\n";

    const suffix =
        "\nBegin optimizing MNIST test accuracy. " ++
        "Start by calling config_show to see " ++
        "the current configuration.";

    if (has_history) {
        return std.fmt.allocPrint(
            arena,
            "{s}{s}{s}",
            .{ intro, history, suffix },
        ) catch "Begin optimizing.";
    }
    return std.fmt.allocPrint(
        arena,
        "{s}{s}",
        .{ intro, suffix },
    ) catch "Begin optimizing.";
}

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    core.run(&config) catch |err| {
        core.api.log(
            "\nFATAL: agent crashed: {s}\n",
            .{@errorName(err)},
        );
        std.process.exit(1);
    };
}
