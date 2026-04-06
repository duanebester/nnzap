//! Generic research toolbox — shared CLI tool
//! implementations for AI agent research binaries.
//!
//! Provides sandboxed file I/O, snapshot/rollback,
//! build/test/bench dispatch, benchmark management,
//! git commit, and experiment history tracking.
//!
//! Domain-specific binaries configure this module via
//! ToolboxConfig and optionally handle custom tools
//! through the custom_dispatch callback.
//!
//! Tools provided:
//!   snapshot            Save engine files to snapshot
//!   snapshot-list       List all snapshots
//!   rollback            Restore from a snapshot
//!   rollback-latest     Restore most recent snapshot
//!   diff                Diff current vs snapshot
//!   check               Compile-only validation
//!   test                Run test suite
//!   bench               Run primary benchmark
//!   bench-compare       Compare benchmark results
//!   bench-list          List benchmark files
//!   bench-latest        Output latest benchmark
//!   bench-clean         Delete old benchmark files
//!   show                Show engine source file
//!   show-function       Extract a function body
//!   read-file           Read a project file
//!   write-file          Write to a project file
//!   edit-file           Search/replace edit
//!   list-dir            List directory contents
//!   cwd                 Print working directory
//!   run-cmd             Run a shell command
//!   commit              Git add and commit
//!   add-summary         Record experiment summary
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

    /// Files to include in snapshots.
    engine_files: []const []const u8,

    /// Build check command (compile-only validation).
    check_command: []const []const u8,

    /// Test command.
    test_command: []const []const u8,

    /// Primary benchmark command.
    bench_command: []const []const u8,

    /// Additional benchmark commands keyed by
    /// subcommand name.
    extra_bench: []const ExtraBench = &.{},

    /// When true, the bench command writes JSON to a
    /// file in bench_dir.  The toolbox cleans old
    /// files, runs the command, then reads the new
    /// file.  When false (default), the bench command
    /// writes JSON to stdout directly.
    bench_output_file: bool = false,

    /// Benchmark directory (monorepo-relative).
    bench_dir: []const u8,

    /// Benchmark filename prefixes.
    bench_prefixes: []const []const u8,

    /// History directory (monorepo-relative).
    history_dir: []const u8,

    /// Snapshot directory (monorepo-relative).
    snapshot_dir: []const u8,

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

const MAX_SNAPSHOTS: u32 = 128;
const MAX_DIFF_LINES: u32 = 200_000;
const MAX_FUNCTION_NAME: u32 = 256;
const MAX_PATH_LEN: u32 = 512;
const TIMESTAMP_LEN: u32 = 19; // "2025-01-15T14-30-00"
const MAX_HISTORY_SIZE: usize = 2 * 1024 * 1024;
const MAX_COMMAND_OUTPUT: usize = 1 * 1024 * 1024;

// ============================================================
// Internal result types
// ============================================================

const DiffFileResult = struct {
    changed: bool,
    additions: u32,
    deletions: u32,
};

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
    std.debug.assert(config.engine_files.len > 0);
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
    if (tools.eql(cmd, "snapshot")) {
        return toolSnapshot(config, arena);
    }
    if (tools.eql(cmd, "snapshot-list")) {
        return toolSnapshotList(config, arena);
    }
    if (tools.eql(cmd, "rollback")) {
        return toolRollback(config, arena, args);
    }
    if (tools.eql(cmd, "rollback-latest")) {
        return toolRollbackLatest(config, arena);
    }
    if (tools.eql(cmd, "diff")) {
        return toolDiff(config, arena, args);
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
    if (tools.eql(cmd, "bench-compare")) {
        return toolBenchCompare(config, arena);
    }
    if (tools.eql(cmd, "bench-list")) {
        return toolBenchList(config, arena);
    }
    if (tools.eql(cmd, "bench-latest")) {
        return toolBenchLatest(config, arena);
    }
    if (tools.eql(cmd, "bench-clean")) {
        return toolBenchClean(config, arena);
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
    if (tools.eql(cmd, "commit")) {
        return toolCommit(config, arena, args);
    }
    if (tools.eql(cmd, "add-summary")) {
        return toolAddSummary(config, arena, args);
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
            "\"snapshot\", \"snapshot-list\", " ++
            "\"rollback\", \"rollback-latest\", " ++
            "\"diff\", \"check\", \"test\", " ++
            "\"bench\", \"bench-compare\", " ++
            "\"bench-list\", \"bench-latest\", " ++
            "\"bench-clean\", \"show\", " ++
            "\"show-function\", \"read-file\", " ++
            "\"write-file\", \"edit-file\", " ++
            "\"list-dir\", \"cwd\", \"run-cmd\", " ++
            "\"commit\", \"add-summary\", " ++
            "\"history\"]}\n",
    );
}

// ============================================================
// Tool: snapshot — copy engine sources to timestamped dir
// ============================================================

fn toolSnapshot(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    var ts_buf: [32]u8 = undefined;
    const timestamp = formatTimestamp(&ts_buf);
    std.debug.assert(timestamp.len == TIMESTAMP_LEN);

    const snap_dir = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ config.snapshot_dir, timestamp },
    );
    const snap_dir_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        snap_dir,
    );

    // Create the snapshot root directory.
    try std.fs.cwd().makePath(snap_dir_fs);

    const count = try snapshotCopyFiles(
        config,
        arena,
        snap_dir_fs,
    );
    std.debug.assert(count <= config.engine_files.len);

    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"snapshot_id\": \"{s}\", " ++
            "\"files\": {d}}}\n",
        .{ timestamp, count },
    );
    try tools.writeStdout(json);
}

/// Copy each engine file into the snapshot directory,
/// preserving the relative path structure.
fn snapshotCopyFiles(
    config: *const ToolboxConfig,
    arena: Allocator,
    snap_dir_fs: []const u8,
) !u32 {
    std.debug.assert(snap_dir_fs.len > 0);
    var count: u32 = 0;

    for (config.engine_files) |file| {
        std.debug.assert(file.len > 0);

        // Ensure parent directories exist inside snapshot.
        const dest = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ snap_dir_fs, file },
        );
        if (std.fs.path.dirname(dest)) |parent| {
            try std.fs.cwd().makePath(parent);
        }

        const file_fs = try tools.resolveToFs(
            arena,
            config.fs_root,
            file,
        );
        tools.copyFile(file_fs, dest) catch |err| {
            std.debug.print(
                "{s}: skip {s}: {s}\n",
                .{
                    config.name,
                    file,
                    @errorName(err),
                },
            );
            continue;
        };
        count += 1;
    }
    return count;
}

// ============================================================
// Tool: snapshot-list — enumerate all snapshot directories
// ============================================================

fn toolSnapshotList(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    const dirs = try listSnapshotDirs(config, arena);
    var buf: std.ArrayList(u8) = .empty;

    try buf.appendSlice(arena, "{\"status\": \"ok\", ");
    try buf.appendSlice(arena, "\"snapshots\": [");

    for (dirs, 0..) |name, i| {
        if (i > 0) try buf.appendSlice(arena, ", ");
        try buf.appendSlice(arena, "\"");
        try buf.appendSlice(arena, name);
        try buf.appendSlice(arena, "\"");
    }

    try buf.appendSlice(arena, "]}\n");
    std.debug.assert(buf.items.len > 0);
    try tools.writeStdout(buf.items);
}

// ============================================================
// Tool: rollback — restore engine files from a snapshot
// ============================================================

fn toolRollback(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"usage: rollback " ++
                "<snapshot_id>\"}\n",
        );
        return;
    }

    const snapshot_id = args[0];
    std.debug.assert(snapshot_id.len > 0);
    try restoreSnapshot(config, arena, snapshot_id);
}

fn toolRollbackLatest(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    const latest = try findLatestSnapshot(
        config,
        arena,
    );
    std.debug.assert(latest.len > 0);
    try restoreSnapshot(config, arena, latest);
}

/// Restore all engine files from a named snapshot.
fn restoreSnapshot(
    config: *const ToolboxConfig,
    arena: Allocator,
    snapshot_id: []const u8,
) !void {
    std.debug.assert(snapshot_id.len > 0);

    const snap_dir = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ config.snapshot_dir, snapshot_id },
    );
    const snap_dir_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        snap_dir,
    );

    // Verify the snapshot directory exists.
    std.fs.cwd().access(snap_dir_fs, .{}) catch {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"snapshot not " ++
                "found: {s}\"}}\n",
            .{snapshot_id},
        );
        try tools.writeStdout(json);
        return;
    };

    var count: u32 = 0;
    for (config.engine_files) |file| {
        const source = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ snap_dir_fs, file },
        );
        const file_fs = try tools.resolveToFs(
            arena,
            config.fs_root,
            file,
        );
        tools.copyFile(source, file_fs) catch continue;
        count += 1;
    }

    std.debug.assert(count <= config.engine_files.len);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"snapshot_id\": \"{s}\", " ++
            "\"files_restored\": {d}}}\n",
        .{ snapshot_id, count },
    );
    try tools.writeStdout(json);
}

// ============================================================
// Tool: diff — compare current files against a snapshot
// ============================================================

fn toolDiff(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"usage: diff " ++
                "<snapshot_id>\"}\n",
        );
        return;
    }

    const snapshot_id = args[0];
    std.debug.assert(snapshot_id.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    const header = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"snapshot_id\": \"{s}\", " ++
            "\"files\": [\n",
        .{snapshot_id},
    );
    try buf.appendSlice(arena, header);

    for (config.engine_files, 0..) |file, i| {
        if (i > 0) try buf.appendSlice(arena, ",\n");
        const entry = try formatDiffEntry(
            config,
            arena,
            snapshot_id,
            file,
        );
        try buf.appendSlice(arena, entry);
    }

    try buf.appendSlice(arena, "\n]}\n");
    try tools.writeStdout(buf.items);
}

/// Run diff(1) on a single file, return JSON fragment.
fn formatDiffEntry(
    config: *const ToolboxConfig,
    arena: Allocator,
    snapshot_id: []const u8,
    file: []const u8,
) ![]const u8 {
    std.debug.assert(snapshot_id.len > 0);
    std.debug.assert(file.len > 0);

    const result = diffOneFile(
        config,
        arena,
        snapshot_id,
        file,
    );

    if (result.changed) {
        return std.fmt.allocPrint(
            arena,
            "  {{\"file\": \"{s}\", " ++
                "\"changed\": true, " ++
                "\"additions\": {d}, " ++
                "\"deletions\": {d}}}",
            .{
                file,
                result.additions,
                result.deletions,
            },
        );
    }

    return std.fmt.allocPrint(
        arena,
        "  {{\"file\": \"{s}\", " ++
            "\"changed\": false}}",
        .{file},
    );
}

/// Shell out to diff(1) for a single file pair.
/// Returns change counts; never fails (defaults to
/// changed if something goes wrong).
fn diffOneFile(
    config: *const ToolboxConfig,
    arena: Allocator,
    snapshot_id: []const u8,
    file: []const u8,
) DiffFileResult {
    const snap_rel = std.fmt.allocPrint(
        arena,
        "{s}/{s}/{s}",
        .{ config.snapshot_dir, snapshot_id, file },
    ) catch return .{
        .changed = true,
        .additions = 0,
        .deletions = 0,
    };

    const snap_path = tools.resolveToFs(
        arena,
        config.fs_root,
        snap_rel,
    ) catch return .{
        .changed = true,
        .additions = 0,
        .deletions = 0,
    };

    const file_fs = tools.resolveToFs(
        arena,
        config.fs_root,
        file,
    ) catch return .{
        .changed = true,
        .additions = 0,
        .deletions = 0,
    };

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &[_][]const u8{
            "diff", "-u", snap_path, file_fs,
        },
        .max_output_bytes = tools.MAX_OUTPUT_BYTES,
    }) catch return .{
        .changed = true,
        .additions = 0,
        .deletions = 0,
    };

    const code: u8 = switch (result.term) {
        .Exited => |c| c,
        else => return .{
            .changed = true,
            .additions = 0,
            .deletions = 0,
        },
    };

    // diff exit codes: 0 = identical, 1 = different,
    // 2+ = error.
    if (code == 0) {
        return .{
            .changed = false,
            .additions = 0,
            .deletions = 0,
        };
    }

    const changes = countDiffChanges(result.stdout);
    return .{
        .changed = true,
        .additions = changes[0],
        .deletions = changes[1],
    };
}

/// Count additions (+) and deletions (-) in unified diff
/// output, skipping the --- and +++ header lines.
fn countDiffChanges(output: []const u8) [2]u32 {
    std.debug.assert(
        output.len <= tools.MAX_OUTPUT_BYTES,
    );
    var additions: u32 = 0;
    var deletions: u32 = 0;

    var iter = std.mem.splitScalar(
        u8,
        output,
        '\n',
    );
    while (iter.next()) |line| {
        if (line.len == 0) continue;
        // Skip unified diff headers.
        if (tools.startsWith(line, "---")) continue;
        if (tools.startsWith(line, "+++")) continue;
        if (line[0] == '+') additions += 1;
        if (line[0] == '-') deletions += 1;
    }

    std.debug.assert(additions <= MAX_DIFF_LINES);
    std.debug.assert(deletions <= MAX_DIFF_LINES);
    return .{ additions, deletions };
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

/// Run the primary benchmark command.
/// Handles both stdout-capture and file-based output
/// depending on config.bench_output_file.
fn toolBench(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    if (config.bench_output_file) {
        const bench_fs = try tools.resolveToFs(
            arena,
            config.fs_root,
            config.bench_dir,
        );
        _ = tools.cleanBenchmarkFiles(
            bench_fs,
            config.bench_prefixes,
        );
    }

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

    if (config.bench_output_file) {
        // Find the new benchmark file on disk.
        const bench_fs = try tools.resolveToFs(
            arena,
            config.fs_root,
            config.bench_dir,
        );
        const name = tools.findOneBenchmark(
            arena,
            bench_fs,
            config.bench_prefixes,
        );
        if (name) |n| {
            const path = try std.fmt.allocPrint(
                arena,
                "{s}/{s}",
                .{ bench_fs, n },
            );
            const content = try tools.readFile(
                arena,
                path,
            );
            try tools.writeStdout(content);
        } else {
            try tools.writeStdout(
                "{\"status\": \"error\", " ++
                    "\"error\": \"no benchmark " ++
                    "produced\"}\n",
            );
        }
    } else {
        // Capture stdout from the benchmark command.
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
}

/// Run an extra benchmark command (always captures
/// stdout).
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
// Tool: bench-compare — compare all benchmark results
// ============================================================

fn toolBenchCompare(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    const bench_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        config.bench_dir,
    );
    try tools.toolBenchCompare(
        arena,
        bench_fs,
        config.bench_prefixes,
    );
}

// ============================================================
// Tool: bench-list — list benchmark files
// ============================================================

/// List all benchmark JSON filenames.
fn toolBenchList(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    const bench_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        config.bench_dir,
    );
    const names = try tools.listBenchmarkFiles(
        arena,
        bench_fs,
        config.bench_prefixes,
    );
    var buf: std.ArrayList(u8) = .empty;

    try buf.appendSlice(arena, "{\"files\": [");
    for (names, 0..) |name, i| {
        if (i > 0) {
            try buf.appendSlice(arena, ", ");
        }
        try buf.appendSlice(arena, "\"");
        try buf.appendSlice(arena, name);
        try buf.appendSlice(arena, "\"");
    }
    try buf.appendSlice(arena, "]}\n");
    try tools.writeStdout(buf.items);
}

// ============================================================
// Tool: bench-latest — output latest benchmark
// ============================================================

/// Output the most recent benchmark result as JSON.
fn toolBenchLatest(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    const bench_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        config.bench_dir,
    );
    const names = try tools.listBenchmarkFiles(
        arena,
        bench_fs,
        config.bench_prefixes,
    );
    if (names.len == 0) return error.NoBenchmarkFiles;

    const latest = names[names.len - 1];
    const path = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ bench_fs, latest },
    );
    const content = try tools.readFile(arena, path);
    try tools.writeStdout(content);
}

// ============================================================
// Tool: bench-clean — delete old benchmark files
// ============================================================

/// Delete all benchmark JSON files.
fn toolBenchClean(
    config: *const ToolboxConfig,
    arena: Allocator,
) !void {
    const bench_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        config.bench_dir,
    );
    const count = tools.cleanBenchmarkFiles(
        bench_fs,
        config.bench_prefixes,
    );
    var buf: [128]u8 = undefined;
    const json = std.fmt.bufPrint(
        &buf,
        "{{\"status\": \"ok\", " ++
            "\"files_deleted\": {d}}}\n",
        .{count},
    ) catch unreachable;
    try tools.writeStdout(json);
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

    const short_name = args[0];
    const path = resolveEnginePath(config, short_name);
    if (path == null) {
        try writeFileNotFound(
            config,
            arena,
            short_name,
        );
        return;
    }

    const path_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        path.?,
    );
    const content = try tools.readFile(arena, path_fs);
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
        .{ short_name, lines, escaped },
    );
    std.debug.assert(json.len > 0);
    try tools.writeStdout(json);
}

/// Emit a "file not found" JSON error with the list of
/// valid file names the user can choose from.
fn writeFileNotFound(
    config: *const ToolboxConfig,
    arena: Allocator,
    short_name: []const u8,
) !void {
    std.debug.assert(short_name.len > 0);
    std.debug.assert(config.engine_files.len > 0);

    var names_buf: std.ArrayList(u8) = .empty;
    for (config.engine_files, 0..) |file, i| {
        if (i > 0) {
            names_buf.appendSlice(
                arena,
                ", ",
            ) catch {};
        }
        names_buf.appendSlice(arena, file) catch {};
    }

    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"error\", " ++
            "\"error\": \"unknown file: {s}. " ++
            "Valid: {s}\"}}\n",
        .{ short_name, names_buf.items },
    );
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

    const short_name = args[0];
    const function_name = args[1];
    std.debug.assert(function_name.len > 0);

    const path = resolveEnginePath(config, short_name);
    if (path == null) {
        try writeFileNotFound(
            config,
            arena,
            short_name,
        );
        return;
    }

    const path_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        path.?,
    );
    const content = try tools.readFile(arena, path_fs);
    const bounds = findFunctionBounds(
        content,
        function_name,
    );

    if (bounds == null) {
        try writeFunctionNotFound(
            arena,
            short_name,
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
        short_name,
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

// ============================================================
// Snapshot directory helpers
// ============================================================

/// List all snapshot directory names, sorted ascending.
fn listSnapshotDirs(
    config: *const ToolboxConfig,
    arena: Allocator,
) ![]const []const u8 {
    const snap_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        config.snapshot_dir,
    );
    var dir = std.fs.cwd().openDir(
        snap_fs,
        .{ .iterate = true },
    ) catch return &.{};
    defer dir.close();

    var names: std.ArrayList([]const u8) = .empty;
    var iter = dir.iterate();
    var count: u32 = 0;

    while (try iter.next()) |entry| {
        if (count >= MAX_SNAPSHOTS) break;
        if (entry.kind != .directory) continue;
        const dupe = try arena.dupe(u8, entry.name);
        try names.append(arena, dupe);
        count += 1;
    }

    tools.sortStrings(names.items);
    std.debug.assert(names.items.len <= MAX_SNAPSHOTS);
    return names.items;
}

/// Find the lexicographically latest snapshot (most
/// recent).  Fails with error.NoSnapshots if none exist.
fn findLatestSnapshot(
    config: *const ToolboxConfig,
    arena: Allocator,
) ![]const u8 {
    const dirs = try listSnapshotDirs(config, arena);
    std.debug.assert(config.snapshot_dir.len > 0);
    if (dirs.len == 0) return error.NoSnapshots;
    return dirs[dirs.len - 1];
}

// ============================================================
// Engine file resolution
// ============================================================

/// Map a short filename ("metal.zig", "compute.metal",
/// etc.) to its full relative path ("nn/src/metal.zig",
/// "nn/src/shaders/compute.metal", etc.).
/// Also accepts full paths like "nn/src/metal.zig".
fn resolveEnginePath(
    config: *const ToolboxConfig,
    short_name: []const u8,
) ?[]const u8 {
    std.debug.assert(short_name.len > 0);

    // First, check for exact match against full paths.
    for (config.engine_files) |path| {
        if (tools.eql(path, short_name)) return path;
    }

    // Then, check if any path ends with the short name.
    for (config.engine_files) |path| {
        if (std.mem.endsWith(u8, path, short_name)) {
            return path;
        }
    }

    std.debug.assert(config.engine_files.len > 0);
    return null;
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
// Tool: commit — git add -A and commit
// ============================================================

fn toolCommit(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    const obj = readJsonInput(arena, args) orelse {
        try tools.writeJsonError(error.InvalidInput);
        std.process.exit(1);
    };
    const msg = getJsonString(
        obj,
        "message",
    ) orelse {
        try tools.stdout_file.writeAll(
            "Error: missing 'message' field",
        );
        std.process.exit(1);
    };
    if (msg.len == 0) {
        try tools.stdout_file.writeAll(
            "Error: empty commit message",
        );
        std.process.exit(1);
    }
    // Write commit message to temp file to avoid
    // shell quoting issues.
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
    try msg_file.writeAll(msg);
    msg_file.close();

    const cmd = try std.fmt.allocPrint(
        arena,
        "git add -A && git commit -F '{s}' " ++
            "&& rm -f '{s}'",
        .{ msg_path, msg_path },
    );
    try runShellCommand(arena, cmd);
}

// ============================================================
// Tool: add-summary — record an experiment summary
// ============================================================

fn toolAddSummary(
    config: *const ToolboxConfig,
    arena: Allocator,
    args: []const []const u8,
) !void {
    const obj = readJsonInput(arena, args) orelse {
        try tools.writeJsonError(error.InvalidInput);
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
    if (summary.len == 0) {
        try tools.stdout_file.writeAll(
            "Error: empty summary",
        );
        std.process.exit(1);
    }
    const next_id =
        countSummaryLines(config, arena) + 1;
    try appendSummary(config, arena, next_id, summary);
    const out = try std.fmt.allocPrint(
        arena,
        "Summary #{d} recorded.",
        .{next_id},
    );
    try tools.writeStdout(out);
}

/// Count existing summary lines in the summaries file.
fn countSummaryLines(
    config: *const ToolboxConfig,
    arena: Allocator,
) u32 {
    const sum_path_rel = std.fmt.allocPrint(
        arena,
        "{s}/summaries.jsonl",
        .{config.history_dir},
    ) catch return 0;
    const fs_path = tools.resolveToFs(
        arena,
        config.fs_root,
        sum_path_rel,
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

/// Append a JSON summary line to the summaries file.
fn appendSummary(
    config: *const ToolboxConfig,
    arena: Allocator,
    experiment_number: u32,
    text: []const u8,
) !void {
    std.debug.assert(text.len > 0);
    std.debug.assert(experiment_number > 0);

    var ts_buf: [32]u8 = undefined;
    const ts = formatTimestamp(&ts_buf);

    // Truncate at sentence boundary (max 500 chars).
    const trimmed = truncateAtSentence(text, 500);

    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\"experiment\":");
    const num_str = try std.fmt.allocPrint(
        arena,
        "{d}",
        .{experiment_number},
    );
    try buf.appendSlice(arena, num_str);
    try buf.appendSlice(arena, ",\"timestamp\":\"");
    try buf.appendSlice(arena, ts);
    try buf.appendSlice(
        arena,
        "\",\"summary\":\"",
    );
    for (trimmed) |c| {
        switch (c) {
            '"' => try buf.appendSlice(
                arena,
                "\\\"",
            ),
            '\\' => try buf.appendSlice(
                arena,
                "\\\\",
            ),
            '\n' => try buf.append(arena, ' '),
            '\r' => {},
            else => try buf.append(arena, c),
        }
    }
    try buf.appendSlice(arena, "\"}\n");

    const sum_path_rel = try std.fmt.allocPrint(
        arena,
        "{s}/summaries.jsonl",
        .{config.history_dir},
    );
    const fs_path = try tools.resolveToFs(
        arena,
        config.fs_root,
        sum_path_rel,
    );
    const file = std.fs.cwd().createFile(
        fs_path,
        .{ .truncate = false },
    ) catch return;
    defer file.close();
    file.seekFromEnd(0) catch return;
    file.writeAll(buf.items) catch return;
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
    for (config.write_scope) |allowed| {
        if (tools.eql(path, allowed)) return true;
    }
    return false;
}
