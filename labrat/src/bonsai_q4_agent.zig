//! Bonsai Q4 research agent — optimises Bonsai 1.7B
//! Q4 inference on Apple Silicon.
//!
//! Thin profile on agent_core.  Configures the generic
//! agent loop for Bonsai Q4 inference optimisation.
//! All tool logic lives in the bonsai_q4_researcher
//! toolbox binary.
//!
//! Usage:
//!   export ANTHROPIC_API_KEY=sk-ant-...
//!   zig build
//!   ./zig-out/bin/bonsai_q4_agent

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("agent_core.zig");
const api = core.api;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_FILE_SIZE: usize = 2 * 1024 * 1024;
const HISTORY_DIR: []const u8 = ".bonsai_q4_history";

/// Source files comprising the hot path.  Kept here so
/// the context builder can include them in a future
/// revision (Rule 8 — amortise upfront).
const CODE_CONTEXT_FILES = [_][]const u8{
    "nnmetal/src/shaders/transformer.metal",
    "nnmetal/src/shaders/compute.metal",
    "nnmetal/src/shaders/q4mv_bf16_specialized.metal",
    "nnmetal/src/transformer.zig",
    "nnmetal/src/metal.zig",
    "nnmetal/src/model.zig",
};

// ============================================================
// Tool definitions (comptime)
// ============================================================

const bonsai_q4_tools = [_]core.ToolDef{
    // ---- Experiment lifecycle ----
    .{
        .name = "experiment_start",
        .subcommand = "experiment-start",
        .description = "Create a new experiment " ++
            "branch from main. Call before making " ++
            "any edits.",
        .properties = &.{.{
            .name = "name",
            .description = "Short experiment " ++
                "description, e.g. " ++
                "q4-simd-rms-norm",
        }},
    },
    .{
        .name = "diff",
        .subcommand = "diff",
        .description = "Show uncommitted changes " ++
            "(git diff). Use to review edits " ++
            "before finishing an experiment.",
    },
    .{
        .name = "experiment_finish",
        .subcommand = "experiment-finish",
        .description = "Conclude the current " ++
            "experiment. Pass decision " ++
            "(keep or abandon) and a summary " ++
            "of what was tried. If keep: " ++
            "commits and merges to main. " ++
            "If abandon: discards changes " ++
            "and returns to main.",
        .properties = &.{
            .{
                .name = "decision",
                .description = "keep or abandon",
            },
            .{
                .name = "summary",
                .description = "Concise summary: " ++
                    "what was tried, the result, " ++
                    "and why it succeeded or failed.",
            },
        },
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
        .description = "Run golden output test: " ++
            "loads the real Bonsai 1.7B Q4 model, " ++
            "generates from a known prompt with " ++
            "greedy decoding, and checks that " ++
            "output tokens match a hardcoded " ++
            "expected sequence. Tests the ENTIRE " ++
            "pipeline end-to-end. STOP if this " ++
            "fails.",
    },
    .{
        .name = "bench",
        .subcommand = "bench",
        .description = "Bonsai 1.7B Q4 inference " ++
            "benchmark (~5s). Returns JSON with " ++
            "decode_tok_per_sec, " ++
            "prefill_tok_per_sec, and " ++
            "decode_p99_us.",
    },
    .{
        .name = "history",
        .subcommand = "history",
        .description = "Return the last N " ++
            "experiment records as a JSON array. " ++
            "Each record includes decision, " ++
            "summary, and benchmark metrics.",
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
    },
};

// ============================================================
// Profile configuration
// ============================================================

const config = core.AgentConfig{
    .name = "bonsai_q4",
    .toolbox_path = "./zig-out/bin/bonsai_q4_researcher",
    .history_dir = HISTORY_DIR,
    .system_prompt_path = "programs/q4_system.md",
    .tool_schemas = core.toolSchemas(&bonsai_q4_tools),
    .tool_map = core.toolMappings(&bonsai_q4_tools),
    .history_fields = &.{
        .{ .json_key = "decode_tok_per_sec", .label = "tok/s" },
        .{ .json_key = "prefill_tok_per_sec", .label = "prefill" },
        .{ .json_key = "decode_p99_us", .label = "p99_us" },
    },
    .max_turns_per_experiment = 80,
    .build_context_fn = &buildBonsaiQ4Context,
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
// ============================================================

fn buildBonsaiQ4Context(
    arena: Allocator,
    cfg: *const core.AgentConfig,
) []const u8 {
    const orientation = core.buildOrientation(arena);
    const rules = loadEngineeringRules(arena);
    const history = core.buildHistorySummary(
        arena,
        cfg,
    );
    const has_history = history.len > 0;
    const outline = buildHotPathOutline(arena);
    const last_bench = loadLastBench(arena);

    const rules_section =
        "## Engineering rules (CLAUDE.md)\n\n" ++
        "You MUST follow these rules when editing " ++
        "source code. They are non-negotiable " ++
        "— assertion density, function length " ++
        "limits, naming, explicit control flow, " ++
        "and all other rules apply to every line " ++
        "you write.\n\n";

    const hist_section = if (has_history)
        "## Experiment history\n\n" ++
            "Each record includes decision, " ++
            "summary, and benchmark metrics. " ++
            "Use the history tool for full " ++
            "details.\n\n"
    else
        "No previous experiment history. " ++
            "This is the first run.\n\n";

    const suffix =
        "\n\n## Begin\n\n" ++
        "Optimise Q4 engine throughput. " ++
        "Call experiment_start with your " ++
        "hypothesis, then edit. The " ++
        "codebase scope section in the " ++
        "system prompt tells you exactly " ++
        "which line ranges to read.";

    return std.fmt.allocPrint(
        arena,
        "{s}\n{s}{s}\n\n{s}{s}{s}{s}{s}",
        .{
            orientation,
            rules_section,
            rules,
            hist_section,
            if (has_history) history else "",
            outline,
            last_bench,
            suffix,
        },
    ) catch "Begin optimising.";
}

// ============================================================
// Hot-path outline builder
// ============================================================

fn buildHotPathOutline(arena: Allocator) []const u8 {
    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "\n## Hot-path map\n\n" ++
            "Function names and line numbers for " ++
            "the engine hot path. Use run_command " ++
            "with sed to read specific functions.\n" ++
            "NOTE: run_command cwd is labrat/, so " ++
            "prefix paths with ../ — e.g. " ++
            "sed -n '100,150p' " ++
            "../nnmetal/src/transformer.zig\n",
    ) catch return "";

    for (CODE_CONTEXT_FILES) |rel_path| {
        appendFileOutline(
            arena,
            &buf,
            rel_path,
        );
    }

    return buf.items;
}

/// Append a single file's function outline to buf.
fn appendFileOutline(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    rel_path: []const u8,
) void {
    const fs_path = core.resolveToFs(
        arena,
        rel_path,
    ) orelse return;
    const content = std.fs.cwd().readFileAlloc(
        arena,
        fs_path,
        MAX_FILE_SIZE,
    ) catch return;
    if (content.len == 0) return;

    // Count lines.
    var line_count: u32 = 0;
    for (content) |c| {
        if (c == '\n') line_count += 1;
    }
    if (content.len > 0 and
        content[content.len - 1] != '\n')
    {
        line_count += 1;
    }

    // Extract short filename for the header.
    const short_name = shortName(rel_path);
    buf.appendSlice(arena, "\n### ") catch return;
    buf.appendSlice(arena, short_name) catch return;
    const header_suffix = std.fmt.allocPrint(
        arena,
        " ({d} lines)\n",
        .{line_count},
    ) catch return;
    buf.appendSlice(arena, header_suffix) catch return;

    // Scan for function signatures.
    var line_number: u32 = 0;
    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        line_number += 1;
        appendFunctionLine(
            arena,
            buf,
            line,
            line_number,
        );
    }
}

/// If line contains a function signature, append a
/// formatted outline entry to buf.
fn appendFunctionLine(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    line: []const u8,
    line_number: u32,
) void {
    // Skip blank or very short lines.
    if (line.len < 4) return;

    // Zig: look for "fn " in the line.
    if (std.mem.indexOf(u8, line, "fn ")) |fn_pos| {
        const is_pub = fn_pos >= 4 and
            std.mem.eql(
                u8,
                line[fn_pos - 4 .. fn_pos],
                "pub ",
            );
        const name_start = fn_pos + 3;
        const rest = line[name_start..];
        const name_end = std.mem.indexOfScalar(
            u8,
            rest,
            '(',
        ) orelse return;
        if (name_end == 0) return;
        const name = rest[0..name_end];

        const entry = std.fmt.allocPrint(
            arena,
            "  L{d:<5} {s}fn {s}\n",
            .{
                line_number,
                if (is_pub)
                    @as([]const u8, "pub ")
                else
                    @as([]const u8, ""),
                name,
            },
        ) catch return;
        buf.appendSlice(arena, entry) catch {};
        return;
    }

    // Metal: look for "kernel " at start of line.
    if (std.mem.startsWith(u8, line, "kernel ")) {
        const after_kernel = line[7..];
        const space_pos = std.mem.indexOfScalar(
            u8,
            after_kernel,
            ' ',
        ) orelse return;
        const name_part =
            after_kernel[space_pos + 1 ..];
        const paren_pos = std.mem.indexOfScalar(
            u8,
            name_part,
            '(',
        ) orelse return;
        if (paren_pos == 0) return;
        const name = name_part[0..paren_pos];

        const entry = std.fmt.allocPrint(
            arena,
            "  L{d:<5} kernel {s}\n",
            .{ line_number, name },
        ) catch return;
        buf.appendSlice(arena, entry) catch {};
    }
}

/// Extract the short filename from a path.
fn shortName(path: []const u8) []const u8 {
    if (std.mem.indexOf(u8, path, "src/")) |pos| {
        return path[pos + 4 ..];
    }
    if (std.mem.lastIndexOfScalar(
        u8,
        path,
        '/',
    )) |pos| {
        return path[pos + 1 ..];
    }
    return path;
}

// ============================================================
// Last benchmark loader
// ============================================================

fn loadLastBench(arena: Allocator) []const u8 {
    const bench_path = core.resolveToFs(
        arena,
        HISTORY_DIR ++ "/_last_bench.json",
    ) orelse return "";
    const raw = std.fs.cwd().readFileAlloc(
        arena,
        bench_path,
        64 * 1024,
    ) catch return "";
    if (raw.len == 0) return "";

    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        raw,
        .{},
    ) catch return "";
    const obj = switch (parsed) {
        .object => |o| o,
        else => return "",
    };

    const decode = core.getNumStr(
        arena,
        obj,
        "decode_tok_per_sec",
    );
    const prefill = core.getNumStr(
        arena,
        obj,
        "prefill_tok_per_sec",
    );
    const p99 = core.getNumStr(
        arena,
        obj,
        "decode_p99_us",
    );

    return std.fmt.allocPrint(
        arena,
        "\n## Last benchmark\n\n" ++
            "decode: {s} tok/s | prefill: {s} tok/s" ++
            " | p99: {s} us\n" ++
            "Target: 224 tok/s\n",
        .{ decode, prefill, p99 },
    ) catch "";
}

// ============================================================
// Engineering rules loader
// ============================================================

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
