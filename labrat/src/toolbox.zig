//! Generic research toolbox — shared CLI tool
//! implementations for AI agent research binaries.
//!
//! Provides sandboxed file I/O, experiment branching,
//! build/test/bench dispatch, and experiment history
//! tracking.
//!
//! Domain-specific binaries configure this module via
//! ToolboxConfig and optionally handle custom tools
//! through the custom_dispatch callback.
//!
//! Tools provided:
//!   check               Compile-only validation
//!   test                Run test suite
//!   bench               Run primary benchmark
//!   show                Show a source file
//!   show-function       Extract a function body
//!   read-file           Read a project file
//!   write-file          Write to a project file
//!   edit-file           Search/replace edit
//!   list-dir            List directory contents
//!   cwd                 Print working directory
//!   run-cmd             Run a shell command
//!   experiment-start    Create experiment branch
//!   diff                Show uncommitted changes
//!   experiment-finish   Conclude an experiment
//!   history             Show recent experiments

const std = @import("std");
const Allocator = std.mem.Allocator;
const tools = @import("tools.zig");

// ============================================================
// Configuration
// ============================================================

/// Domain-specific configuration for the generic toolbox.
/// Each research domain provides one of these to
/// configure build commands, file scopes, and benchmark
/// handling.
pub const ToolboxConfig = struct {
    /// Display name for log messages.
    name: []const u8,

    /// Working directory for build/test/bench commands.
    project_root: []const u8,

    /// Filesystem root prefix for resolving monorepo-relative
    /// paths.  Defaults to ".." (one directory above zap/).
    fs_root: []const u8 = "..",

    /// Files the agent may write (monorepo-relative).
    write_scope: []const []const u8,

    /// Directory prefixes the agent may read.
    read_scope: []const []const u8,

    /// Individual files the agent may read outside
    /// the prefix scope.
    read_files: []const []const u8,

    /// Build check command (compile-only validation).
    check_command: []const []const u8,

    /// Test command.
    test_command: []const []const u8,

    /// Primary benchmark command.
    bench_command: []const []const u8,

    /// Additional benchmark commands keyed by
    /// subcommand name.
    extra_bench: []const ExtraBench = &.{},

    /// History directory (monorepo-relative).
    history_dir: []const u8,

    /// Optional domain-specific tool dispatch.
    /// Return true if the command was handled.
    custom_dispatch: ?*const fn (
        std.mem.Allocator,
        []const u8,
        []const []const u8,
    ) anyerror!bool = null,
};

/// An extra benchmark command beyond the primary one.
pub const ExtraBench = struct {
    tool_name: []const u8,
    command: []const []const u8,
};

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_FUNCTION_NAME: u32 = 256;
const MAX_PATH_LEN: u32 = 512;
const MAX_HISTORY_SIZE: usize = 2 * 1024 * 1024;
const MAX_COMMAND_OUTPUT: usize = 1 * 1024 * 1024;

// ============================================================
// Internal result types
// ============================================================

const FunctionBounds = struct {
    start_line: u32,
    end_line: u32,
};

// ============================================================
// JSON input helpers (for -f flag subcommands)
// ============================================================

/// Read and parse a JSON input file passed via -f flag.
/// Returns the parsed JSON object, or null on failure.
/// Deletes the input file after reading (cleanup).
fn readJsonInput(
    arena: Allocator,
    args: []const []const u8,
) ?std.json.ObjectMap {
    if (args.len < 2) return null;
    if (!tools.eql(args[0], "-f")) return null;

    const path = args[1];
    const content = tools.readFile(arena, path) catch
        return null;

    // Clean up the temp file after reading.
    std.fs.cwd().deleteFile(path) catch {};

    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        content,
        .{},
    ) catch return null;

    return switch (parsed) {
        .object => |o| o,
        else => null,
    };
}

/// Extract a string field from a JSON object.
fn getJsonString(
    obj: std.json.ObjectMap,
    key: []const u8,
) ?[]const u8 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .string => |s| s,
        else => null,
    };
}

// ============================================================
// Entry point
// ============================================================

/// Run the toolbox with the given configuration.
/// Parses CLI args, dispatches to the matching tool,
/// and handles errors.  Does not return on failure.
pub fn run(config: *const ToolboxConfig) void {
    std.debug.assert(config.write_scope.len > 0);
    std.debug.assert(config.read_scope.len > 0);
    std.debug.assert(config.bench_command.len > 0);

    var arena_state = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const args = std.process.argsAlloc(arena) catch {
        std.process.exit(1);
    };
    std.debug.assert(args.len >= 1);

    if (args.len < 2) {
        toolHelp(config) catch {};
        std.process.exit(1);
    }

    const cmd = args[1];
    const rest: []const []const u8 = if (args.len > 2)
        args[2..]
    else
        &.{};

    dispatch(config, arena, cmd, rest) catch |err| {
        tools.writeJsonError(err) catch {};
        std.process.exit(1);
    };
}

fn dispatch(
    config: *const ToolboxConfig,
    arena: Allocator,
    cmd: []const u8,
    args: []const []const u8,
) !void {
    std.debug.assert(cmd.len > 0);
    std.debug.assert(cmd.len < MAX_PATH_LEN);

    if (tools.eql(cmd, "help")) {
        return toolHelp(config);
    }
    if (tools.eql(cmd, "check")) {
        return toolCheck(config, arena);
    }
    if (tools.eql(cmd, "test")) {
        return toolTest(config, arena);
    }
    if (tools.eql(cmd, "bench")) {
        return toolBench(config, arena);
    }
    if (tools.eql(cmd, "show")) {
        return toolShow(config, arena, args);
    }
    if (tools.eql(cmd, "show-function")) {
        return toolShowFunction(config, arena, args);
    }
    if (tools.eql(cmd, "read-file")) {
        return toolReadFile(config, arena, args);
    }
    if (tools.eql(cmd, "write-file")) {
        return toolWriteFile(config, arena, args);
    }
    if (tools.eql(cmd, "edit-file")) {
        return toolEditFile(config, arena, args);
    }
    if (tools.eql(cmd, "list-dir")) {
        return toolListDir(config, arena, args);
    }
    if (tools.eql(cmd, "cwd")) return toolCwd(arena);
    if (tools.eql(cmd, "run-cmd")) {
        return toolRunCmd(arena, args);
    }
    if (tools.eql(cmd, "experiment-start")) {
        return toolExperimentStart(
            config,
            arena,
            args,
        );
    }
    if (tools.eql(cmd, "diff")) {
        return toolDiff(arena);
    }
    if (tools.eql(cmd, "experiment-finish")) {
        return toolExperimentFinish(
            config,
            arena,
            args,
        );
    }
    if (tools.eql(cmd, "history")) {
        return toolHistory(config, arena, args);
    }

    // Extra benchmark commands from config.
    for (config.extra_bench) |extra| {
        if (tools.eql(cmd, extra.tool_name)) {
            return runExtraBench(config, arena, extra);
        }
    }

    // Domain-specific dispatch.
    if (config.custom_dispatch) |custom| {
        if (try custom(arena, cmd, args)) return;
    }

    try toolHelp(config);
    std.process.exit(1);
}

// ============================================================
// Tool: help
// ============================================================

fn toolHelp(config: *const ToolboxConfig) !void {
    _ = config;
    try tools.writeStdout(
        "{\"tools\": [" ++
            "\"check\", \"test\", " ++
            "\"bench\", \"show\", " ++
            "\"show-function\", \"read-file\", " ++
            "\"write-file\", \"edit-file\", " ++
            "\"list-dir\", \"cwd\", \"run-cmd\", " ++
            "\"experiment-start\", " ++
            "\"diff\", " ++
            "\"experiment-finish\", " ++
            "\"history\"]}\n",
    );
}

// ============================================================
// Tools: check / test — build validation
// ============================================================

/// Run a build command and report success/failure as
/// JSON.  Used by check and test tools.
fn runAndReport(
    config: *const ToolboxConfig,
    arena: Allocator,
    argv: []const []const u8,
    label: []const u8,
    result_key: []const u8,
    error_key: []const u8,
) !void {
    std.debug.print(
        "{s}: running {s}...\n",
        .{ config.name, label },
    );

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = argv,
        .cwd = config.project_root,
        .max_output_bytes = tools.MAX_OUTPUT_BYTES,
    }) catch |err| {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"spawn_failed: " ++
                "{s}\"}}\n",
            .{@errorName(err)},
        );
        try tools.writeStdout(json);
        return;
    };

    const success = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (success) {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"ok\", " ++
                "\"{s}\": true}}\n",
            .{result_key},
        );
        try tools.writeStdout(json);
        return;
    }

    const escaped = try tools.truncateAndEscape(
        arena,
        result.stderr,
        4000,
    );
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"{s}\": false, " ++
            "\"{s}\": \"{s}\"}}\n",
        .{ result_key, error_key, escaped },
    );
    std.debug.assert(json.len > 0);
    try tools.writeStdout(json);
}

fn toolCheck(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    try runAndReport(
        config,
        arena,
        config.check_command,
        "check",
        "compiled",
        "errors",
    );
}

fn toolTest(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    try runAndReport(
        config,
        arena,
        config.test_command,
        "test",
        "passed",
        "output",
    );
}

// ============================================================
// Tool: bench — run primary benchmark
// ============================================================

fn toolBench(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    std.debug.print(
        "{s}: running bench...\n",
        .{config.name},
    );

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = config.bench_command,
        .cwd = config.project_root,
        .max_output_bytes = tools.MAX_OUTPUT_BYTES,
    }) catch |err| {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"spawn_failed: " ++
                "{s}\"}}\n",
            .{@errorName(err)},
        );
        try tools.writeStdout(json);
        return;
    };

    if (result.stderr.len > 0) {
        tools.stderr_file.writeAll(
            result.stderr,
        ) catch {};
    }

    const success = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (!success) {
        try tools.writeBuildError(
            arena,
            result.stderr,
            2000,
        );
        return;
    }

    // Capture stdout from the benchmark command.
    if (result.stdout.len > 0) {
        cacheBenchResult(config, arena, result.stdout);
        try tools.writeStdout(result.stdout);
    } else {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"no benchmark " ++
                "output\"}\n",
        );
    }
}

/// Cache the latest benchmark JSON to a temp file
/// so experiment_finish can read it.
fn cacheBenchResult(
    config: *const ToolboxConfig,
    arena: Allocator,
    json: []const u8,
) void {
    std.debug.assert(json.len > 0);

    const path_rel = std.fmt.allocPrint(
        arena,
        "{s}/_last_bench.json",
        .{config.history_dir},
    ) catch return;
    const fs_path = tools.resolveToFs(
        arena,
        config.fs_root,
        path_rel,
    ) catch return;
    const file = std.fs.cwd().createFile(
        fs_path,
        .{},
    ) catch return;
    defer file.close();
    file.writeAll(json) catch {};
}

/// Run an extra benchmark command (always captures
/// stdout).  Does not cache results — only the primary
/// bench tool caches to _last_bench.json.
fn runExtraBench(
    config: *const ToolboxConfig,
    arena: Allocator,
    extra: ExtraBench,
) !void {
    std.debug.print(
        "{s}: running {s}...\n",
        .{ config.name, extra.tool_name },
    );

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = extra.command,
        .cwd = config.project_root,
        .max_output_bytes = tools.MAX_OUTPUT_BYTES,
    }) catch |err| {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"spawn_failed: " ++
                "{s}\"}}\n",
            .{@errorName(err)},
        );
        try tools.writeStdout(json);
        return;
    };

    if (result.stderr.len > 0) {
        tools.stderr_file.writeAll(
            result.stderr,
        ) catch {};
    }

    const success = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (!success) {
        try tools.writeBuildError(
            arena,
            result.stderr,
            2000,
        );
        return;
    }

    if (result.stdout.len > 0) {
        try tools.writeStdout(result.stdout);
    } else {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"no benchmark " ++
                "output\"}\n",
        );
    }
}

// ============================================================
// Tool: show — display contents of an engine source file
// ============================================================

fn toolShow(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"usage: " ++
                "show <file>\"}\n",
        );
        return;
    }

    const path = args[0];
    std.debug.assert(path.len > 0);

    if (!isAllowedReadPath(config, path)) {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"read not allowed " ++
                "for '{s}'\"}}\n",
            .{path},
        );
        try tools.writeStdout(err_msg);
        return;
    }

    const path_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        path,
    );
    const content = tools.readFile(
        arena,
        path_fs,
    ) catch {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"cannot read " ++
                "'{s}'\"}}\n",
            .{path},
        );
        try tools.writeStdout(err_msg);
        return;
    };
    const lines = countLines(content);
    const escaped = try tools.jsonEscape(
        arena,
        content,
    );

    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"file\": \"{s}\", " ++
            "\"lines\": {d}, " ++
            "\"content\": \"{s}\"}}\n",
        .{ path, lines, escaped },
    );
    std.debug.assert(json.len > 0);
    try tools.writeStdout(json);
}

// ============================================================
// Tool: show-function — extract a function body
// ============================================================

fn toolShowFunction(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 2) {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"usage: " ++
                "show-function " ++
                "<file> <function_name>\"}\n",
        );
        return;
    }

    const path = args[0];
    const function_name = args[1];
    std.debug.assert(path.len > 0);
    std.debug.assert(function_name.len > 0);

    if (!isAllowedReadPath(config, path)) {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"read not allowed " ++
                "for '{s}'\"}}\n",
            .{path},
        );
        try tools.writeStdout(err_msg);
        return;
    }

    const path_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        path,
    );
    const content = tools.readFile(
        arena,
        path_fs,
    ) catch {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"cannot read " ++
                "'{s}'\"}}\n",
            .{path},
        );
        try tools.writeStdout(err_msg);
        return;
    };
    const bounds = findFunctionBounds(
        content,
        function_name,
    );

    if (bounds == null) {
        try writeFunctionNotFound(
            arena,
            path,
            function_name,
        );
        return;
    }

    const body = extractLineRange(
        arena,
        content,
        bounds.?.start_line,
        bounds.?.end_line,
    );
    try emitFunctionJson(
        arena,
        path,
        function_name,
        bounds.?,
        body,
    );
}

/// Emit a "function not found" JSON error.
fn writeFunctionNotFound(
    arena: Allocator,
    file: []const u8,
    function_name: []const u8,
) !void {
    std.debug.assert(file.len > 0);
    std.debug.assert(function_name.len > 0);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"error\", " ++
            "\"error\": \"function '{s}' not " ++
            "found in {s}\"}}\n",
        .{ function_name, file },
    );
    try tools.writeStdout(json);
}

/// Emit the show-function success JSON.
fn emitFunctionJson(
    arena: Allocator,
    file: []const u8,
    function_name: []const u8,
    bounds: FunctionBounds,
    body: []const u8,
) !void {
    std.debug.assert(file.len > 0);
    std.debug.assert(function_name.len > 0);

    const escaped = try tools.jsonEscape(arena, body);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"file\": \"{s}\", " ++
            "\"function\": \"{s}\", " ++
            "\"start_line\": {d}, " ++
            "\"end_line\": {d}, " ++
            "\"content\": \"{s}\"}}\n",
        .{
            file,
            function_name,
            bounds.start_line,
            bounds.end_line,
            escaped,
        },
    );
    try tools.writeStdout(json);
}

// ============================================================
// Function-finding logic
// ============================================================

/// Find the start and end line numbers of a named
/// function.  Uses a simple brace-depth parser: find the
/// line containing the function signature, then track
/// brace depth until it returns to zero.  Line numbers
/// are 1-based.
///
/// Supports both Zig-style (`fn name(`) and C/Metal-style
/// (`void name(`, `kernel void name(`) declarations.  The
/// Zig pattern is tried first; the C/Metal fallback only
/// fires if the Zig pattern finds nothing.
fn findFunctionBounds(
    content: []const u8,
    name: []const u8,
) ?FunctionBounds {
    if (name.len == 0) return null;
    if (name.len > MAX_FUNCTION_NAME) return null;

    // Primary pattern: Zig-style "fn <name>(".
    var zig_buf: [MAX_FUNCTION_NAME + 8]u8 = undefined;
    const zig_pattern = std.fmt.bufPrint(
        &zig_buf,
        "fn {s}(",
        .{name},
    ) catch return null;

    // Fallback pattern: C/Metal-style " <name>(" for
    // kernel/void declarations.
    var c_buf: [MAX_FUNCTION_NAME + 4]u8 = undefined;
    const c_pattern = std.fmt.bufPrint(
        &c_buf,
        " {s}(",
        .{name},
    ) catch return null;

    // First pass: try Zig pattern.
    const zig_result = scanForFunction(
        content,
        zig_pattern,
    );
    if (zig_result != null) return zig_result;

    // Second pass: try C/Metal pattern.
    return scanForFunction(content, c_pattern);
}

/// Scan content for a function matching the given
/// pattern, then track brace depth to find the closing
/// brace.  Returns null if the pattern is not found.
///
/// Key invariant: we only terminate when `body_entered`
/// is true AND `depth` returns to zero.  This correctly
/// handles multi-line signatures where several lines
/// appear between `fn name(` and the opening `{`.
fn scanForFunction(
    content: []const u8,
    pattern: []const u8,
) ?FunctionBounds {
    std.debug.assert(pattern.len > 0);
    std.debug.assert(content.len <= tools.MAX_FILE_SIZE);

    var line_number: u32 = 0;
    var start_line: u32 = 0;
    var depth: i32 = 0;
    var found = false;
    var body_entered = false;

    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        line_number += 1;

        if (!found) {
            // Search for the function signature.
            if (std.mem.indexOf(
                u8,
                line,
                pattern,
            )) |_| {
                found = true;
                start_line = line_number;
            } else {
                continue;
            }
        }

        // Count braces on this line.
        depth = countBracesOnLine(line, depth);

        // Track whether we have entered the function
        // body (seen at least one opening brace).
        if (depth > 0) body_entered = true;

        // Function body ends when depth returns to zero
        // after having gone positive (the body's closing
        // brace balances the opening brace).
        if (body_entered and depth == 0) {
            return FunctionBounds{
                .start_line = start_line,
                .end_line = line_number,
            };
        }
    }

    // Handle unterminated functions at end of file.
    if (found and start_line > 0) {
        return FunctionBounds{
            .start_line = start_line,
            .end_line = line_number,
        };
    }
    return null;
}

/// Count braces on a single line and return updated
/// depth.  Ignores braces inside string literals (basic
/// handling for double-quoted strings and single-quoted
/// chars).
fn countBracesOnLine(
    line: []const u8,
    depth: i32,
) i32 {
    std.debug.assert(line.len <= tools.MAX_FILE_SIZE);
    var current_depth = depth;
    var in_string = false;
    var prev_char: u8 = 0;

    for (line) |c| {
        if (c == '"' and prev_char != '\\') {
            in_string = !in_string;
        }
        if (!in_string) {
            if (c == '{') current_depth += 1;
            if (c == '}') current_depth -= 1;
        }
        prev_char = c;
    }

    std.debug.assert(current_depth >= -1);
    return current_depth;
}

/// Extract lines [start..end] (1-based, inclusive) from
/// content.  Returns the raw text of those lines.
fn extractLineRange(
    arena: Allocator,
    content: []const u8,
    start_line: u32,
    end_line: u32,
) []const u8 {
    std.debug.assert(start_line > 0);
    std.debug.assert(end_line >= start_line);

    var buf: std.ArrayList(u8) = .empty;
    var line_number: u32 = 0;

    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        line_number += 1;
        if (line_number < start_line) continue;
        if (line_number > end_line) break;
        if (buf.items.len > 0) {
            buf.append(arena, '\n') catch return "";
        }
        buf.appendSlice(arena, line) catch return "";
    }

    return buf.items;
}

// ============================================================
// Timestamp formatting
// ============================================================

/// Format the current UTC time as an ISO-8601-like string
/// suitable for directory names: "2025-01-15T14-30-00".
/// Colons are replaced with dashes for filesystem safety.
fn formatTimestamp(buf: *[32]u8) []const u8 {
    const ts: u64 = @intCast(std.time.timestamp());
    std.debug.assert(ts > 0);

    const es = std.time.epoch.EpochSeconds{
        .secs = ts,
    };
    const epoch_day = es.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = es.getDaySeconds();

    const year = year_day.year;
    const month = month_day.month.numeric();
    const day: u32 =
        @as(u32, month_day.day_index) + 1;
    const hour = day_secs.getHoursIntoDay();
    const minute = day_secs.getMinutesIntoHour();
    const second = day_secs.getSecondsIntoMinute();

    std.debug.assert(month >= 1);
    std.debug.assert(month <= 12);
    return std.fmt.bufPrint(
        buf,
        "{d:0>4}-{d:0>2}-{d:0>2}T" ++
            "{d:0>2}-{d:0>2}-{d:0>2}",
        .{
            year,
            month,
            day,
            hour,
            minute,
            second,
        },
    ) catch "0000-00-00T00-00-00";
}

/// Format the current UTC time as ISO 8601 with colons:
/// "2025-01-15T14:30:00Z".  Used for JSONL records.
fn formatTimestampUtc(buf: *[32]u8) []const u8 {
    const ts: u64 = @intCast(std.time.timestamp());
    std.debug.assert(ts > 0);

    const es = std.time.epoch.EpochSeconds{
        .secs = ts,
    };
    const epoch_day = es.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = es.getDaySeconds();

    const year = year_day.year;
    const month = month_day.month.numeric();
    const day: u32 =
        @as(u32, month_day.day_index) + 1;
    const hour = day_secs.getHoursIntoDay();
    const minute = day_secs.getMinutesIntoHour();
    const second = day_secs.getSecondsIntoMinute();

    std.debug.assert(month >= 1);
    std.debug.assert(month <= 12);
    return std.fmt.bufPrint(
        buf,
        "{d:0>4}-{d:0>2}-{d:0>2}T" ++
            "{d:0>2}:{d:0>2}:{d:0>2}Z",
        .{
            year,
            month,
            day,
            hour,
            minute,
            second,
        },
    ) catch "0000-00-00T00:00:00Z";
}

// ============================================================
// Line counting
// ============================================================

/// Count the number of lines in a text buffer.
fn countLines(content: []const u8) u32 {
    std.debug.assert(
        content.len <= tools.MAX_FILE_SIZE,
    );
    if (content.len == 0) return 0;

    var count: u32 = 0;
    for (content) |c| {
        if (c == '\n') count += 1;
    }

    // Count the last line if it doesn't end with
    // newline.
    if (content[content.len - 1] != '\n') {
        count += 1;
    }
    std.debug.assert(count > 0);
    return count;
}

// ============================================================
// Tool: read-file — read a project file by relative path
// ============================================================

fn toolReadFile(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try tools.writeJsonError(
            error.MissingArgument,
        );
        std.process.exit(1);
    }
    const path = args[0];
    std.debug.assert(path.len > 0);
    if (!isAllowedReadPath(config, path)) {
        const msg = try std.fmt.allocPrint(
            arena,
            "Error: read not allowed for '{s}'. " ++
                "Only project files are accessible.",
            .{path},
        );
        try tools.writeStdout(msg);
        std.process.exit(1);
    }
    const fs_path = try tools.resolveToFs(arena, config.fs_root, path);
    const content = tools.readFile(
        arena,
        fs_path,
    ) catch {
        const msg = try std.fmt.allocPrint(
            arena,
            "Error: cannot read '{s}'",
            .{path},
        );
        try tools.writeStdout(msg);
        std.process.exit(1);
    };
    try tools.writeStdout(content);
}

// ============================================================
// Tool: write-file — write content to a project file
// ============================================================

fn toolWriteFile(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    const obj = readJsonInput(arena, args) orelse {
        try tools.writeJsonError(error.InvalidInput);
        std.process.exit(1);
    };
    const path = getJsonString(
        obj,
        "path",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'path' field",
        );
        std.process.exit(1);
    };
    const content = getJsonString(
        obj,
        "content",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'content' field",
        );
        std.process.exit(1);
    };
    std.debug.assert(path.len > 0);
    if (!isAllowedWritePath(config, path)) {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "Error: write not allowed to '{s}'",
            .{path},
        );
        try tools.writeStdout(err_msg);
        std.process.exit(1);
    }
    const fs_path = try tools.resolveToFs(arena, config.fs_root, path);
    const file = try std.fs.cwd().createFile(
        fs_path,
        .{},
    );
    defer file.close();
    try file.writeAll(content);
    const ok_msg = try std.fmt.allocPrint(
        arena,
        "OK: wrote {d} bytes to {s}",
        .{ content.len, path },
    );
    try tools.writeStdout(ok_msg);
}

// ============================================================
// Tool: edit-file — search/replace edit on a project file
// ============================================================

fn toolEditFile(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    const obj = readJsonInput(arena, args) orelse {
        try tools.writeJsonError(error.InvalidInput);
        std.process.exit(1);
    };
    const path = getJsonString(
        obj,
        "path",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'path' field",
        );
        std.process.exit(1);
    };
    const old_text = getJsonString(
        obj,
        "old_content",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'old_content'",
        );
        std.process.exit(1);
    };
    const new_text = getJsonString(
        obj,
        "new_content",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'new_content'",
        );
        std.process.exit(1);
    };
    std.debug.assert(path.len > 0);
    if (!isAllowedWritePath(config, path)) {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "Error: edit not allowed for '{s}'",
            .{path},
        );
        try tools.writeStdout(err_msg);
        std.process.exit(1);
    }
    if (old_text.len == 0) {
        try tools.stdout_file.writeAll(
            "Error: old_content is empty",
        );
        std.process.exit(1);
    }
    try applyEdit(config, arena, path, old_text, new_text);
}

/// Apply a search/replace edit: find old_text in the file
/// at path, verify it is unique, and replace with
/// new_text.
fn applyEdit(
    config: *const ToolboxConfig,
    arena: Allocator,
    path: []const u8,
    old_text: []const u8,
    new_text: []const u8,
) !void {
    std.debug.assert(path.len > 0);
    std.debug.assert(old_text.len > 0);

    const fs_path = try tools.resolveToFs(arena, config.fs_root, path);
    const current = try tools.readFile(arena, fs_path);
    const pos = std.mem.indexOf(
        u8,
        current,
        old_text,
    ) orelse {
        const preview = tools.truncate(old_text, 100);
        const err_msg = try std.fmt.allocPrint(
            arena,
            "Error: old_content not found in " ++
                "{s}. Searched for: \"{s}...\" " ++
                "({d} chars).",
            .{ path, preview, old_text.len },
        );
        try tools.writeStdout(err_msg);
        std.process.exit(1);
    };
    // Check for ambiguous matches.
    const after_first = pos + old_text.len;
    if (after_first < current.len) {
        if (std.mem.indexOf(
            u8,
            current[after_first..],
            old_text,
        ) != null) {
            try tools.stdout_file.writeAll(
                "Error: old_content matches " ++
                    "multiple locations. Make " ++
                    "it more specific.",
            );
            std.process.exit(1);
        }
    }
    const new_file = try std.fmt.allocPrint(
        arena,
        "{s}{s}{s}",
        .{
            current[0..pos],
            new_text,
            current[after_first..],
        },
    );
    const file = try std.fs.cwd().createFile(
        fs_path,
        .{},
    );
    defer file.close();
    try file.writeAll(new_file);
    const ok_msg = try std.fmt.allocPrint(
        arena,
        "OK: edited {s} (-{d} +{d} chars, " ++
            "{d} bytes total)",
        .{
            path,
            old_text.len,
            new_text.len,
            new_file.len,
        },
    );
    try tools.writeStdout(ok_msg);
}

// ============================================================
// Tool: list-dir — list directory contents
// ============================================================

fn toolListDir(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try tools.writeJsonError(
            error.MissingArgument,
        );
        std.process.exit(1);
    }
    const path = args[0];
    std.debug.assert(path.len > 0);
    if (!isAllowedReadPath(config, path)) {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "Error: listing not allowed for '{s}'",
            .{path},
        );
        try tools.writeStdout(err_msg);
        std.process.exit(1);
    }
    const fs_path = try tools.resolveToFs(arena, config.fs_root, path);
    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "/bin/ls", "-la", fs_path },
        .max_output_bytes = MAX_COMMAND_OUTPUT,
    }) catch |err| {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "Error: ls failed: {s}",
            .{@errorName(err)},
        );
        try tools.writeStdout(err_msg);
        std.process.exit(1);
    };
    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (result.stdout.len > 0) {
        try tools.writeStdout(result.stdout);
    }
    if (!ok) std.process.exit(1);
}

// ============================================================
// Tool: cwd — print working directory
// ============================================================

fn toolCwd(arena: Allocator) !void {
    const abs = try tools.cwdAbsolute(arena);
    try tools.writeStdout(abs);
}

// ============================================================
// Tool: run-cmd — run a shell command
// ============================================================

fn toolRunCmd(
    arena: Allocator,
    args: []const []const u8,
) !void {
    const obj = readJsonInput(arena, args) orelse {
        try tools.writeJsonError(error.InvalidInput);
        std.process.exit(1);
    };
    const command = getJsonString(
        obj,
        "command",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'command' field",
        );
        std.process.exit(1);
    };
    if (command.len == 0) {
        try tools.stdout_file.writeAll(
            "Error: empty command",
        );
        std.process.exit(1);
    }
    try runShellCommand(arena, command);
}

/// Execute a shell command via /bin/sh -c and write
/// combined stdout/stderr to our stdout.
fn runShellCommand(
    arena: Allocator,
    command: []const u8,
) !void {
    std.debug.assert(command.len > 0);
    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "/bin/sh", "-c", command },
        .max_output_bytes = MAX_COMMAND_OUTPUT,
    }) catch |err| {
        const err_msg = try std.fmt.allocPrint(
            arena,
            "Error: spawn failed: {s}",
            .{@errorName(err)},
        );
        try tools.writeStdout(err_msg);
        std.process.exit(1);
    };
    const exit_code: ?u32 = switch (result.term) {
        .Exited => |code| @as(?u32, code),
        else => null,
    };
    if (result.stdout.len > 0) {
        try tools.stdout_file.writeAll(result.stdout);
    }
    if (result.stderr.len > 0) {
        if (result.stdout.len > 0) {
            try tools.stdout_file.writeAll(
                "\n--- stderr ---\n",
            );
        }
        try tools.stdout_file.writeAll(result.stderr);
    }
    if (exit_code) |code| {
        if (code != 0) {
            const exit_msg = try std.fmt.allocPrint(
                arena,
                "\n(exit code {d})",
                .{code},
            );
            try tools.writeStdout(exit_msg);
        }
    }
}

// ============================================================
// Tool: experiment-start — create an experiment branch
// ============================================================

fn toolExperimentStart(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    _ = config;
    const obj = readJsonInput(arena, args) orelse {
        try tools.writeJsonError(error.InvalidInput);
        std.process.exit(1);
    };
    const raw_name = getJsonString(
        obj,
        "name",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'name' field",
        );
        std.process.exit(1);
    };
    if (raw_name.len == 0) {
        try tools.stdout_file.writeAll(
            "Error: empty experiment name",
        );
        std.process.exit(1);
    }

    var ts_buf: [32]u8 = undefined;
    const ts = formatTimestamp(&ts_buf);
    const safe_name = sanitizeBranchName(
        arena,
        raw_name,
    );
    const branch = try std.fmt.allocPrint(
        arena,
        "experiment/{s}-{s}",
        .{ ts, safe_name },
    );

    // Switch to main first, then create the branch.
    const cmd = try std.fmt.allocPrint(
        arena,
        "git checkout main && " ++
            "git checkout -b '{s}'",
        .{branch},
    );
    try runShellCommand(arena, cmd);
}

// ============================================================
// Tool: diff — show uncommitted changes
// ============================================================

fn toolDiff(arena: Allocator) !void {
    try runShellCommand(
        arena,
        "git --no-pager diff",
    );
}

// ============================================================
// Tool: experiment-finish — conclude an experiment
// ============================================================

/// Single tool to conclude an experiment. Replaces
/// the former git_finish/git_abandon/commit/add_summary
/// multi-step workflow.
///
/// JSON input: {"decision": "keep"|"abandon", "summary": "..."}
///
/// Steps:
///   1. Read current git branch (error if on main).
///   2. Read cached bench result from _last_bench.json.
///   3. Append one enriched record to experiments.jsonl.
///   4. If keep: git add, commit, merge to main.
///   5. If abandon: git reset --hard, checkout main.
///   6. Clean up _last_bench.json.
fn toolExperimentFinish(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    const obj = readJsonInput(arena, args) orelse {
        try tools.writeJsonError(error.InvalidInput);
        std.process.exit(1);
    };
    const decision = getJsonString(
        obj,
        "decision",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'decision' field",
        );
        std.process.exit(1);
    };
    const summary = getJsonString(
        obj,
        "summary",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'summary' field",
        );
        std.process.exit(1);
    };

    const is_keep = tools.eql(decision, "keep");
    const is_abandon = tools.eql(decision, "abandon");
    if (!is_keep and !is_abandon) {
        try tools.stdout_file.writeAll(
            "Error: decision must be " ++
                "'keep' or 'abandon'",
        );
        std.process.exit(1);
    }
    if (summary.len == 0) {
        try tools.stdout_file.writeAll(
            "Error: empty summary",
        );
        std.process.exit(1);
    }

    // 1. Read current branch.
    const branch = captureGitBranch(arena) catch {
        try tools.stdout_file.writeAll(
            "Error: cannot determine current branch",
        );
        std.process.exit(1);
    };
    if (tools.eql(branch, "main")) {
        try tools.stdout_file.writeAll(
            "Error: already on main, " ++
                "no experiment to finish",
        );
        std.process.exit(1);
    }

    // 2. Read cached bench result (optional).
    const bench_json = readLastBench(config, arena);

    // 3. Write enriched record to experiments.jsonl.
    const exp_num =
        countExperimentLines(config, arena) + 1;
    appendEnrichedExperiment(
        config,
        arena,
        exp_num,
        branch,
        decision,
        summary,
        bench_json,
    );

    // 4/5. Git operations.
    if (is_keep) {
        const msg_path_rel = try std.fmt.allocPrint(
            arena,
            "{s}/_commit_msg.txt",
            .{config.history_dir},
        );
        const msg_path = try tools.resolveToFs(
            arena,
            config.fs_root,
            msg_path_rel,
        );
        const msg_file = try std.fs.cwd().createFile(
            msg_path,
            .{},
        );
        try msg_file.writeAll(summary);
        msg_file.close();

        const cmd = try std.fmt.allocPrint(
            arena,
            "git add -A && " ++
                "git commit -F '{s}' && " ++
                "rm -f '{s}' && " ++
                "git checkout main && " ++
                "git merge '{s}'",
            .{ msg_path, msg_path, branch },
        );
        try runShellCommand(arena, cmd);
    } else {
        try runShellCommand(
            arena,
            "git reset --hard HEAD && " ++
                "git checkout main",
        );
    }

    // 6. Clean up cached bench result.
    cleanLastBench(config, arena);

    const out = try std.fmt.allocPrint(
        arena,
        "Experiment #{d} {s}.",
        .{ exp_num, decision },
    );
    try tools.writeStdout(out);
}

/// Read the cached benchmark result from
/// {history_dir}/_last_bench.json.  Returns null
/// if the file does not exist or is empty.
fn readLastBench(
    config: *const ToolboxConfig,
    arena: Allocator,
) ?[]const u8 {
    const path_rel = std.fmt.allocPrint(
        arena,
        "{s}/_last_bench.json",
        .{config.history_dir},
    ) catch return null;
    const fs_path = tools.resolveToFs(
        arena,
        config.fs_root,
        path_rel,
    ) catch return null;
    const content = tools.readFile(
        arena,
        fs_path,
    ) catch return null;
    if (content.len == 0) return null;
    return content;
}

/// Count existing experiment lines in experiments.jsonl.
fn countExperimentLines(
    config: *const ToolboxConfig,
    arena: Allocator,
) u32 {
    const path_rel = std.fmt.allocPrint(
        arena,
        "{s}/experiments.jsonl",
        .{config.history_dir},
    ) catch return 0;
    const fs_path = tools.resolveToFs(
        arena,
        config.fs_root,
        path_rel,
    ) catch return 0;
    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch return 0;
    defer file.close();
    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch return 0;
    var count: u32 = 0;
    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        if (line.len >= 2) count += 1;
    }
    return count;
}

/// Append one enriched experiment record to
/// experiments.jsonl.  Merges experiment metadata
/// with bench results into a single JSON line.
fn appendEnrichedExperiment(
    config: *const ToolboxConfig,
    arena: Allocator,
    experiment_number: u32,
    branch: []const u8,
    decision: []const u8,
    summary: []const u8,
    bench_json: ?[]const u8,
) void {
    std.debug.assert(experiment_number > 0);
    std.debug.assert(branch.len > 0);
    std.debug.assert(summary.len > 0);

    var ts_buf: [32]u8 = undefined;
    const ts = formatTimestampUtc(&ts_buf);

    const trimmed = truncateAtSentence(summary, 500);

    var buf: std.ArrayList(u8) = .empty;

    // Experiment metadata.
    const num_str = std.fmt.allocPrint(
        arena,
        "{d}",
        .{experiment_number},
    ) catch return;
    buf.appendSlice(
        arena,
        "{\"experiment\":",
    ) catch return;
    buf.appendSlice(arena, num_str) catch return;
    buf.appendSlice(
        arena,
        ",\"branch\":\"",
    ) catch return;
    buf.appendSlice(arena, branch) catch return;
    buf.appendSlice(
        arena,
        "\",\"decision\":\"",
    ) catch return;
    buf.appendSlice(arena, decision) catch return;
    buf.appendSlice(
        arena,
        "\",\"summary\":\"",
    ) catch return;

    // Escape summary text.
    for (trimmed) |c| {
        switch (c) {
            '"' => buf.appendSlice(
                arena,
                "\\\"",
            ) catch return,
            '\\' => buf.appendSlice(
                arena,
                "\\\\",
            ) catch return,
            '\n' => buf.append(arena, ' ') catch return,
            '\r' => {},
            else => buf.append(arena, c) catch return,
        }
    }
    buf.append(arena, '"') catch return;

    // Merge bench fields if available.
    if (bench_json) |bj| {
        mergeBenchFields(arena, &buf, bj);
    }

    // Timestamp and close.
    buf.appendSlice(
        arena,
        ",\"timestamp_utc\":\"",
    ) catch return;
    buf.appendSlice(arena, ts) catch return;
    buf.appendSlice(arena, "\"}\n") catch return;

    // Append to experiments.jsonl.
    const path_rel = std.fmt.allocPrint(
        arena,
        "{s}/experiments.jsonl",
        .{config.history_dir},
    ) catch return;
    const fs_path = tools.resolveToFs(
        arena,
        config.fs_root,
        path_rel,
    ) catch return;
    const file = std.fs.cwd().createFile(
        fs_path,
        .{ .truncate = false },
    ) catch return;
    defer file.close();
    file.seekFromEnd(0) catch return;
    file.writeAll(buf.items) catch return;
}

/// Parse bench JSON and append each field (except
/// timestamp_utc) to the output buffer as ",key:value".
fn mergeBenchFields(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    bench_json: []const u8,
) void {
    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        bench_json,
        .{},
    ) catch return;
    const obj = switch (parsed) {
        .object => |o| o,
        else => return,
    };
    var iter = obj.iterator();
    while (iter.next()) |entry| {
        // Skip timestamp — we generate our own.
        if (tools.eql(
            entry.key_ptr.*,
            "timestamp_utc",
        )) continue;

        buf.appendSlice(arena, ",\"") catch return;
        buf.appendSlice(
            arena,
            entry.key_ptr.*,
        ) catch return;
        buf.appendSlice(arena, "\":") catch return;

        switch (entry.value_ptr.*) {
            .string => |s| {
                buf.append(arena, '"') catch return;
                buf.appendSlice(arena, s) catch return;
                buf.append(arena, '"') catch return;
            },
            .float => |f| {
                const fs = std.fmt.allocPrint(
                    arena,
                    "{d:.1}",
                    .{f},
                ) catch return;
                buf.appendSlice(arena, fs) catch return;
            },
            .integer => |i| {
                const is = std.fmt.allocPrint(
                    arena,
                    "{d}",
                    .{i},
                ) catch return;
                buf.appendSlice(arena, is) catch return;
            },
            else => {
                buf.appendSlice(
                    arena,
                    "null",
                ) catch return;
            },
        }
    }
}

/// Delete the cached _last_bench.json file.
fn cleanLastBench(
    config: *const ToolboxConfig,
    arena: Allocator,
) void {
    const path_rel = std.fmt.allocPrint(
        arena,
        "{s}/_last_bench.json",
        .{config.history_dir},
    ) catch return;
    const fs_path = tools.resolveToFs(
        arena,
        config.fs_root,
        path_rel,
    ) catch return;
    std.fs.cwd().deleteFile(fs_path) catch {};
}

// ============================================================
// Git helpers
// ============================================================

/// Run `git rev-parse --abbrev-ref HEAD` and return
/// the current branch name.
fn captureGitBranch(
    arena: Allocator,
) ![]const u8 {
    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        },
        .max_output_bytes = MAX_COMMAND_OUTPUT,
    }) catch return error.GitFailed;

    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (!ok) return error.GitFailed;

    const trimmed = std.mem.trimRight(
        u8,
        result.stdout,
        "\n\r \t",
    );
    if (trimmed.len == 0) return error.GitFailed;
    return trimmed;
}

/// Sanitize user input for use as a git branch name.
/// Replaces non-alphanumeric characters with dashes
/// and collapses consecutive dashes.
fn sanitizeBranchName(
    arena: Allocator,
    raw: []const u8,
) []const u8 {
    std.debug.assert(raw.len > 0);
    const max_len: usize = 60;
    const limit = @min(raw.len, max_len);

    var buf: std.ArrayList(u8) = .empty;
    var prev_dash = false;

    for (raw[0..limit]) |c| {
        const keep = (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9');
        if (keep) {
            buf.append(arena, c) catch break;
            prev_dash = false;
        } else if (!prev_dash) {
            buf.append(arena, '-') catch break;
            prev_dash = true;
        }
    }

    // Trim trailing dash.
    if (buf.items.len > 0 and
        buf.items[buf.items.len - 1] == '-')
    {
        buf.items.len -= 1;
    }

    if (buf.items.len == 0) return "unnamed";
    return buf.items;
}

/// Truncate text at a sentence boundary (. ! ? or ))
/// within max_len characters.  Falls back to hard cut.
fn truncateAtSentence(
    text: []const u8,
    max_len: usize,
) []const u8 {
    std.debug.assert(max_len > 0);
    if (text.len <= max_len) return text;
    var pos: usize = max_len;
    while (pos > 0) {
        pos -= 1;
        const c = text[pos];
        const is_ender = c == '.' or c == '!' or
            c == '?' or c == ')';
        if (!is_ender) continue;
        if (pos + 1 >= max_len) {
            return text[0 .. pos + 1];
        }
        const next = text[pos + 1];
        if (next == ' ' or next == '\n' or
            next == '\r')
        {
            return text[0 .. pos + 1];
        }
    }
    return text[0..max_len];
}

// ============================================================
// Tool: history — show recent experiment entries
// ============================================================

fn toolHistory(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    const count: u32 = blk: {
        if (args.len > 0) {
            break :blk std.fmt.parseInt(
                u32,
                args[0],
                10,
            ) catch 5;
        }
        break :blk 5;
    };
    const capped = @min(count, 20);
    if (capped == 0) {
        try tools.writeStdout("[]");
        return;
    }
    try emitHistoryJson(config, arena, capped);
}

/// Read the history file and emit the last N entries as
/// a JSON array.
fn emitHistoryJson(
    config: *const ToolboxConfig,
    arena: Allocator,
    max_entries: u32,
) !void {
    std.debug.assert(max_entries > 0);
    std.debug.assert(max_entries <= 20);

    const hist_path_rel = std.fmt.allocPrint(
        arena,
        "{s}/experiments.jsonl",
        .{config.history_dir},
    ) catch {
        try tools.writeStdout("[]");
        return;
    };
    const fs_path = tools.resolveToFs(
        arena,
        config.fs_root,
        hist_path_rel,
    ) catch {
        try tools.writeStdout("[]");
        return;
    };
    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch {
        try tools.writeStdout("[]");
        return;
    };
    defer file.close();
    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch {
        try tools.writeStdout("[]");
        return;
    };
    if (content.len == 0) {
        try tools.writeStdout("[]");
        return;
    }
    var lines: [512][]const u8 = undefined;
    var line_count: u32 = 0;
    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        if (line.len < 2) continue;
        if (line_count < 512) {
            lines[line_count] = line;
            line_count += 1;
        }
    }
    if (line_count == 0) {
        try tools.writeStdout("[]");
        return;
    }
    const start = if (line_count > max_entries)
        line_count - max_entries
    else
        0;
    var buf: std.ArrayList(u8) = .empty;
    try buf.append(arena, '[');
    for (lines[start..line_count], 0..) |line, i| {
        if (i > 0) {
            try buf.appendSlice(arena, ",\n");
        }
        try buf.appendSlice(arena, line);
    }
    try buf.append(arena, ']');
    try tools.writeStdout(buf.items);
}

// ============================================================
// Path permission guards
// ============================================================

fn isAllowedReadPath(
    config: *const ToolboxConfig,
    path: []const u8,
) bool {
    std.debug.assert(path.len > 0);
    if (std.mem.indexOf(u8, path, "..") != null) {
        return false;
    }
    if (path[0] == '/') return false;
    for (config.read_files) |file| {
        if (tools.eql(path, file)) return true;
    }
    for (config.read_scope) |prefix| {
        if (tools.startsWith(path, prefix)) {
            return true;
        }
    }
    for (config.write_scope) |file| {
        if (tools.eql(path, file)) return true;
    }
    return false;
}

fn isAllowedWritePath(
    config: *const ToolboxConfig,
    path: []const u8,
) bool {
    std.debug.assert(path.len > 0);
    if (std.mem.indexOf(u8, path, "..") != null) {
        return false;
    }
    if (path[0] == '/') return false;
    for (config.write_scope) |prefix| {
        if (tools.startsWith(path, prefix)) {
            return true;
        }
    }
    return false;
}
