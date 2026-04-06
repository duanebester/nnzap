//! MNIST training agent — thin profile on agent_core.
//!
//! Configures the generic agent loop for MNIST
//! hyperparameter optimisation.  All tool logic lives
//! in the mnist_research toolbox binary.
//!
//! This profile demonstrates extending the standard
//! toolset with domain-specific tools.  The four
//! config_* tools are MNIST-specific (implemented via
//! custom_dispatch in mnist_research.zig); every other
//! tool is provided by the generic toolbox.
//!
//! Usage:
//!   export ANTHROPIC_API_KEY=sk-ant-...
//!   zig build
//!   ./zig-out/bin/mnist_agent

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("agent_core.zig");
const api = core.api;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_FILE_SIZE: usize = 2 * 1024 * 1024;
const HISTORY_DIR: []const u8 = ".mnist_history";

// ============================================================
// Tool definitions (comptime)
//
// Each ToolDef carries its own schema metadata.  The
// agent_core helpers toolMappings() and toolSchemas()
// derive the dispatch table and JSON at comptime.
//
// The first four tools (config_*) are MNIST-specific
// — they manipulate hyperparameters in main.zig via
// custom_dispatch in mnist_research.zig.  Everything
// else is the standard generic toolset provided by
// toolbox.zig.
// ============================================================

const mnist_tools = [_]core.ToolDef{
    // ============================================================
    // Domain-specific tools (MNIST hyperparameter config)
    //
    // These are the extension point.  A new domain would
    // replace these with its own domain-specific tools
    // and wire them through custom_dispatch.
    // ============================================================
    .{
        .name = "config_show",
        .subcommand = "config-show",
        .description = "Show current hyperparameters " ++
            "from main.zig. Returns JSON with " ++
            "architecture, learning_rate, " ++
            "optimizer, batch_size, epochs, seed.",
    },
    .{
        .name = "config_set",
        .subcommand = "config-set",
        .description = "Modify hyperparameters in " ++
            "main.zig. Pass key=value pairs as " ++
            "the settings array. Keys: lr " ++
            "(float), batch (int, must divide " ++
            "50000), epochs (int, max 30), seed " ++
            "(int), optimizer (sgd or adam), " ++
            "arch (layer spec like " ++
            "784:256:relu,256:10:none), beta1 " ++
            "(float), beta2 (float), epsilon " ++
            "(float).",
        .properties = &.{.{
            .name = "settings",
            .description = "Key=value pairs, " ++
                "e.g. optimizer=adam, lr=0.001",
            .type = .string_array,
            .required = true,
        }},
        .shape_override = .json_payload,
    },
    .{
        .name = "config_backup",
        .subcommand = "config-backup",
        .description = "Backup main.zig before " ++
            "making changes. Always call before " ++
            "config_set so you can revert.",
    },
    .{
        .name = "config_restore",
        .subcommand = "config-restore",
        .description = "Restore main.zig from " ++
            "backup. Use after an experiment " ++
            "made things worse.",
    },
    // ============================================================
    // Standard toolset (generic — provided by toolbox.zig)
    // ============================================================

    // ---- Snapshot management ----
    .{
        .name = "snapshot",
        .subcommand = "snapshot",
        .description = "Save engine source files " ++
            "as a restore point. Always call " ++
            "before editing.",
    },
    .{
        .name = "snapshot_list",
        .subcommand = "snapshot-list",
        .description = "List all saved snapshots " ++
            "with timestamps.",
    },
    .{
        .name = "rollback",
        .subcommand = "rollback",
        .description = "Restore engine files from " ++
            "a specific snapshot.",
        .properties = &.{.{
            .name = "id",
            .description = "Snapshot ID from " ++
                "snapshot or snapshot_list",
        }},
    },
    .{
        .name = "rollback_latest",
        .subcommand = "rollback-latest",
        .description = "Restore from the most " ++
            "recent snapshot. Use when " ++
            "check/test/bench fails.",
    },
    .{
        .name = "diff",
        .subcommand = "diff",
        .description = "Show source file changes " ++
            "since a snapshot.",
        .properties = &.{.{
            .name = "id",
            .description = "Snapshot ID to diff " ++
                "against",
        }},
    },
    // ---- Build / test / bench ----
    .{
        .name = "check",
        .subcommand = "check",
        .description = "Compile-only validation " ++
            "(~2s). Run after every edit. STOP " ++
            "if this fails.",
    },
    .{
        .name = "test",
        .subcommand = "test",
        .description = "Run full test suite for " ++
            "numerical correctness. STOP if " ++
            "this fails.",
    },
    .{
        .name = "train",
        .subcommand = "bench",
        .description = "Build and run MNIST " ++
            "training benchmark (~10s). Returns " ++
            "JSON with final_test_accuracy_pct, " ++
            "throughput_images_per_sec, " ++
            "total_training_ms, per-epoch " ++
            "validation metrics, and test results.",
    },
    .{
        .name = "bench_compare",
        .subcommand = "bench-compare",
        .description = "Compare all saved benchmark " ++
            "runs side by side. Shows test " ++
            "accuracy, throughput, optimizer, " ++
            "learning rate, and architecture " ++
            "for each past run.",
    },
    .{
        .name = "bench_latest",
        .subcommand = "bench-latest",
        .description = "Output the most recent " ++
            "benchmark result as JSON.",
    },
    .{
        .name = "history",
        .subcommand = "history",
        .description = "Return the last N full " ++
            "experiment benchmark records as a " ++
            "JSON array. Use this for detailed " ++
            "per-epoch data, config, etc. The " ++
            "initial summary only shows key " ++
            "metrics.",
        .properties = &.{.{
            .name = "count",
            .description = "Number of recent " ++
                "records to return " ++
                "(default 5, max 20)",
            .type = .integer,
            .required = false,
        }},
    },
    // ---- File inspection ----
    .{
        .name = "show",
        .subcommand = "show",
        .description = "View a source file as " ++
            "structured JSON with line numbers.",
        .properties = &.{.{
            .name = "file",
            .description = "Source file path, " ++
                "e.g. nn/src/network.zig",
        }},
    },
    .{
        .name = "show_function",
        .subcommand = "show-function",
        .description = "Extract a specific function " ++
            "from a source file with line " ++
            "numbers.",
        .properties = &.{
            .{
                .name = "file",
                .description = "Source file path",
            },
            .{
                .name = "function_name",
                .description = "Function name to " ++
                    "extract",
            },
        },
    },
    .{
        .name = "read_file",
        .subcommand = "read-file",
        .description = "Read raw contents of a " ++
            "project file. Locked to project " ++
            "directory.",
        .properties = &.{.{
            .name = "path",
            .description = "File path relative " ++
                "to project root, " ++
                "e.g. nn/examples/mnist.zig",
        }},
    },
    // ---- File mutation ----
    .{
        .name = "write_file",
        .subcommand = "write-file",
        .description = "Replace entire contents " ++
            "of an engine source file. Use " ++
            "edit_file for small changes instead.",
        .properties = &.{
            .{
                .name = "path",
                .description = "Engine file path, " ++
                    "e.g. nn/examples/mnist.zig",
            },
            .{
                .name = "content",
                .description = "Complete new file " ++
                    "contents",
            },
        },
    },
    .{
        .name = "edit_file",
        .subcommand = "edit-file",
        .description = "Targeted find-and-replace " ++
            "in an engine source file. More " ++
            "efficient than write_file for " ++
            "small edits. The old_content must " ++
            "match exactly.",
        .properties = &.{
            .{
                .name = "path",
                .description = "Engine file path, " ++
                    "e.g. nn/examples/mnist.zig",
            },
            .{
                .name = "old_content",
                .description = "Exact text to find " ++
                    "(must match exactly)",
            },
            .{
                .name = "new_content",
                .description = "Replacement text",
            },
        },
    },
    // ---- Filesystem / shell ----
    .{
        .name = "list_directory",
        .subcommand = "list-dir",
        .description = "List files and " ++
            "subdirectories in a project " ++
            "directory.",
        .properties = &.{.{
            .name = "path",
            .description = "Directory path " ++
                "relative to project root, " ++
                "e.g. nn/src/ or nn/examples",
        }},
    },
    .{
        .name = "cwd",
        .subcommand = "cwd",
        .description = "Return the absolute path " ++
            "of the working directory used by " ++
            "run_command and list_directory. " ++
            "Useful for orienting yourself in " ++
            "the filesystem.",
    },
    .{
        .name = "run_command",
        .subcommand = "run-cmd",
        .description = "Execute a shell command " ++
            "in the project root directory via " ++
            "/bin/sh. 120s timeout. Use for " ++
            "grep, wc, head, tail, find, cat, " ++
            "etc. Do NOT run long-lived " ++
            "processes. A non-zero exit code " ++
            "does NOT mean the tool failed.",
        .properties = &.{.{
            .name = "command",
            .description = "Shell command to " ++
                "execute. Use single quotes " ++
                "for patterns.",
        }},
        .shape_override = .json_payload,
    },
    // ---- Git / bookkeeping ----
    .{
        .name = "commit",
        .subcommand = "commit",
        .description = "Git commit all current " ++
            "changes. Call after a successful " ++
            "KEEP decision to preserve the " ++
            "optimization. The message should " ++
            "summarize what was optimized and " ++
            "the result.",
        .properties = &.{.{
            .name = "message",
            .description = "Commit message " ++
                "summarizing the optimization",
        }},
        .shape_override = .json_payload,
    },
    .{
        .name = "add_summary",
        .subcommand = "add-summary",
        .description = "Record a concise summary " ++
            "of what was tried and why it " ++
            "succeeded or failed. Call after " ++
            "every rollback or KEEP. Summaries " ++
            "are injected into every future " ++
            "experiment.",
        .properties = &.{.{
            .name = "summary",
            .description = "Concise summary: " ++
                "what was tried, the result, " ++
                "and why it succeeded or failed.",
        }},
        .shape_override = .json_payload,
    },
};

// ============================================================
// Profile configuration
// ============================================================

const config = core.AgentConfig{
    .name = "mnist",
    .toolbox_path = "./zig-out/bin/mnist_research",
    .history_dir = HISTORY_DIR,
    .system_prompt_path = "programs/mnist_system.md",
    .tool_schemas = core.toolSchemas(&mnist_tools),
    .tool_map = core.toolMappings(&mnist_tools),
    .persist_tools = &.{"train"},
    .history_fields = &.{
        .{ .json_key = "throughput_images_per_sec", .label = "throughput" },
        .{ .json_key = "final_test_accuracy_pct", .label = "acc%" },
        .{ .json_key = "total_training_ms", .label = "time_ms" },
    },
    .build_context_fn = &buildMnistContext,
};

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    core.run(&config) catch |err| {
        api.log(
            "\nFATAL: agent crashed: {s}\n",
            .{@errorName(err)},
        );
        api.log(
            "Hint: if the agent edited source files " ++
                "before crashing, run:\n" ++
                "  ./zig-out/bin/mnist_research " ++
                "rollback-latest\n",
            .{},
        );
        std.process.exit(1);
    };
}

// ============================================================
// Context builder
//
// Assembles the first user message from:
//   1. Orientation (working directory, git state).
//   2. Engineering rules from CLAUDE.md.
//   3. Compact benchmark history.
//   4. Agent-written summaries from prior runs.
// ============================================================

fn buildMnistContext(
    arena: Allocator,
    cfg: *const core.AgentConfig,
) []const u8 {
    const orientation = core.buildOrientation(arena);
    const rules = loadEngineeringRules(arena);
    const history = core.buildHistorySummary(
        arena,
        cfg,
    );
    const summaries = core.buildSummariesSection(
        arena,
        cfg.history_dir,
    );
    const has_history = history.len > 0;

    const rules_section =
        "## Engineering rules (CLAUDE.md)\n\n" ++
        "You MUST follow these rules when editing " ++
        "source code. They are non-negotiable " ++
        "— assertion density, function length " ++
        "limits, naming, explicit control flow, " ++
        "and all other rules apply to every line " ++
        "you write.\n\n";

    const hist_section = if (has_history)
        "## Benchmark history (compact)\n\n" ++
            "Use the history tool for full " ++
            "experiment details.\n\n"
    else
        "No previous benchmark history. " ++
            "This is the first run.\n\n";

    const suffix =
        "\n\n## Begin\n\n" ++
        "Optimise MNIST test accuracy and " ++
        "throughput. Start by calling " ++
        "config_show to see the current " ++
        "configuration, then snapshot to " ++
        "create a restore point.";

    return std.fmt.allocPrint(
        arena,
        "{s}\n{s}{s}\n\n{s}{s}{s}{s}",
        .{
            orientation,
            rules_section,
            rules,
            hist_section,
            if (has_history) history else "",
            summaries,
            suffix,
        },
    ) catch "Begin optimizing.";
}

// ============================================================
// Engineering rules loader
// ============================================================

/// Load CLAUDE.md engineering rules from disk.
/// Returns the file contents or a fallback message.
fn loadEngineeringRules(arena: Allocator) []const u8 {
    const path = "CLAUDE.md";
    const fs_path = core.resolveToFs(
        arena,
        path,
    ) orelse {
        api.log(
            "WARNING: cannot resolve {s}\n",
            .{path},
        );
        return "(CLAUDE.md not found — " ++
            "follow standard engineering practices)";
    };
    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch |err| {
        api.log(
            "WARNING: cannot open {s}: {s}\n",
            .{ path, @errorName(err) },
        );
        return "(CLAUDE.md not found — " ++
            "follow standard engineering practices)";
    };
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_FILE_SIZE,
    ) catch |err| {
        api.log(
            "WARNING: cannot read {s}: {s}\n",
            .{ path, @errorName(err) },
        );
        return "(CLAUDE.md too large or unreadable)";
    };

    if (content.len == 0) {
        api.log(
            "WARNING: {s} is empty.\n",
            .{path},
        );
        return "(CLAUDE.md is empty)";
    }

    api.log(
        "Engineering rules: {d} KB ({s})\n",
        .{ content.len / 1024, path },
    );
    return content;
}
