//! Bonsai research agent — optimises Bonsai 1.7B
//! inference on Apple Silicon.
//!
//! Thin profile on agent_core.  Configures the generic
//! agent loop for Bonsai inference optimisation.  All
//! tool logic lives in the bonsai_researcher toolbox
//! binary.
//!
//! Usage:
//!   export ANTHROPIC_API_KEY=sk-ant-...
//!   zig build
//!   ./zig-out/bin/bonsai_agent

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("agent_core.zig");
const api = core.api;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_FILE_SIZE: usize = 2 * 1024 * 1024;
const HISTORY_DIR: []const u8 = ".bonsai_history";

/// Source files comprising the hot path.  Kept here so
/// the context builder can include them in a future
/// revision (Rule 8 — amortise upfront).
const CODE_CONTEXT_FILES = [_][]const u8{
    "nnmetal/src/shaders/transformer.metal",
    "nnmetal/src/shaders/compute.metal",
    "nnmetal/src/transformer.zig",
    "nnmetal/src/metal.zig",
    "nnmetal/src/model.zig",
};

// ============================================================
// Tool definitions (comptime)
//
// Each ToolDef declares the LLM-facing schema and maps
// to a CLI subcommand on the bonsai_researcher toolbox
// binary.  toolMappings() and toolSchemas() derive the
// dispatch table and JSON schemas at comptime.
// ============================================================

const bonsai_tools = [_]core.ToolDef{
    // ---- Git experiment management ----
    .{
        .name = "git_start",
        .subcommand = "git-start",
        .description = "Create a new experiment " ++
            "branch from main. Call before making " ++
            "any edits.",
        .properties = &.{.{
            .name = "name",
            .description = "Short experiment " ++
                "description, e.g. " ++
                "simd-rms-norm",
        }},
        .shape_override = .json_payload,
    },
    .{
        .name = "git_diff",
        .subcommand = "git-diff",
        .description = "Show uncommitted changes " ++
            "(git diff). Use to review edits " ++
            "before committing.",
    },
    .{
        .name = "git_finish",
        .subcommand = "git-finish",
        .description = "Merge the current " ++
            "experiment branch into main. " ++
            "Call after a successful experiment " ++
            "is committed.",
    },
    .{
        .name = "git_abandon",
        .subcommand = "git-abandon",
        .description = "Discard uncommitted " ++
            "changes and switch back to main. " ++
            "Call after a failed experiment.",
    },
    // ---- Build / test / bench ----
    .{
        .name = "check",
        .subcommand = "check",
        .description = "Compile-only validation " ++
            "(~2s). Run after every edit. STOP if " ++
            "this fails.",
    },
    .{
        .name = "test",
        .subcommand = "test",
        .description = "Run full test suite for " ++
            "numerical correctness. STOP if this " ++
            "fails.",
    },
    .{
        .name = "bench",
        .subcommand = "bench",
        .description = "Bonsai 1.7B inference " ++
            "benchmark (~5s). Returns JSON with " ++
            "decode_tok_per_sec, " ++
            "prefill_tok_per_sec, and " ++
            "decode_p99_us.",
    },
    .{
        .name = "bench_compare",
        .subcommand = "bench-compare",
        .description = "Compare all benchmark " ++
            "results side by side.",
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
        .description = "View an engine source file " ++
            "as structured JSON with line numbers.",
        .properties = &.{.{
            .name = "file",
            .description = "Source file path, " ++
                "e.g. nnmetal/src/metal.zig",
        }},
    },
    .{
        .name = "show_function",
        .subcommand = "show-function",
        .description = "Extract a specific function " ++
            "from a source file with line numbers.",
        .properties = &.{
            .{
                .name = "file",
                .description = "Source file path",
            },
            .{
                .name = "function_name",
                .description = "Function name to extract",
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
            .description = "File path relative to " ++
                "project root, " ++
                "e.g. nnmetal/src/metal.zig",
        }},
    },
    // ---- File mutation ----
    .{
        .name = "write_file",
        .subcommand = "write-file",
        .description = "Replace entire contents of " ++
            "an engine source file. Use edit_file " ++
            "for small changes instead.",
        .properties = &.{
            .{
                .name = "path",
                .description = "Engine file path, " ++
                    "e.g. nnmetal/src/metal.zig",
            },
            .{
                .name = "content",
                .description = "Complete new file contents",
            },
        },
    },
    .{
        .name = "edit_file",
        .subcommand = "edit-file",
        .description = "Targeted find-and-replace " ++
            "in an engine source file. More " ++
            "efficient than write_file for small " ++
            "edits. The old_content must match " ++
            "exactly.",
        .properties = &.{
            .{
                .name = "path",
                .description = "Engine file path, " ++
                    "e.g. nnmetal/src/metal.zig",
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
            "subdirectories in a project directory.",
        .properties = &.{.{
            .name = "path",
            .description = "Directory path relative " ++
                "to project root, e.g. nnmetal/src/ " ++
                "or nnmetal/src/shaders",
        }},
    },
    .{
        .name = "cwd",
        .subcommand = "cwd",
        .description = "Return the absolute path of " ++
            "the working directory used by " ++
            "run_command and list_directory. " ++
            "Useful for orienting yourself in " ++
            "the filesystem.",
    },
    .{
        .name = "run_command",
        .subcommand = "run-cmd",
        .description = "Execute a shell command in " ++
            "the project root directory via " ++
            "/bin/sh. 120s timeout. Use for grep, " ++
            "wc, head, tail, find, cat, etc. Do " ++
            "NOT run long-lived processes. A " ++
            "non-zero exit code does NOT mean " ++
            "the tool failed.",
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
            "changes. Call after a successful KEEP " ++
            "decision to preserve the optimization." ++
            " The message should summarize what " ++
            "was optimized and the throughput " ++
            "improvement.",
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
            "every experiment. Summaries " ++
            "are injected into every future " ++
            "experiment.",
        .properties = &.{.{
            .name = "summary",
            .description = "Concise summary: what " ++
                "was tried, the throughput " ++
                "result, and why it succeeded " ++
                "or failed.",
        }},
        .shape_override = .json_payload,
    },
};

// ============================================================
// Profile configuration
// ============================================================

const config = core.AgentConfig{
    .name = "bonsai",
    .toolbox_path = "./zig-out/bin/bonsai_researcher",
    .history_dir = HISTORY_DIR,
    .system_prompt_path = "programs/bonsai_system.md",
    .tool_schemas = core.toolSchemas(&bonsai_tools),
    .tool_map = core.toolMappings(&bonsai_tools),
    .persist_tools = &.{"bench"},
    .history_fields = &.{
        .{ .json_key = "decode_tok_per_sec", .label = "tok/s" },
        .{ .json_key = "prefill_tok_per_sec", .label = "prefill" },
        .{ .json_key = "decode_p99_us", .label = "p99_us" },
    },
    .max_turns_per_experiment = 50,
    .build_context_fn = &buildBonsaiContext,
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
                "  git reset --hard HEAD && " ++
                "git checkout main\n",
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

fn buildBonsaiContext(
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
        "Optimise engine throughput. " ++
        "Start by calling git_start to create an " ++
        "experiment branch, then read the source " ++
        "code to understand the baseline.";

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
    ) catch "Begin optimising.";
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
