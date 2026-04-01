//! nnzap engine_research — AI agent toolbox for engine code.
//!
//! A CLI binary that outputs JSON on stdout, designed for
//! AI coding agents to safely modify and benchmark the core
//! nnzap library code (Metal GPU dispatch, shader code,
//! buffer management, network forward/backward passes).
//!
//! Unlike autoresearch (which only modifies hyperparameters
//! in main.zig), this tool targets the engine itself:
//!   src/metal.zig, src/network.zig, src/layout.zig,
//!   src/shaders/compute.metal, src/main.zig
//!
//! Tools:
//!   help                         Show available tools
//!   snapshot                     Save engine source files
//!   snapshot-list                List all snapshots
//!   rollback <snapshot_id>       Restore from snapshot
//!   rollback-latest              Restore most recent snap
//!   diff <snapshot_id>           Diff against snapshot
//!   check                        Compile-only validation
//!   test                         Run zig build test
//!   bench                        Full training benchmark
//!   bench-compare                Compare benchmark results
//!   show <file>                  Show source file contents
//!   show-function <file> <fn>    Extract a function body
//!
//! Output contract:
//!   stdout  → JSON (parsed by agent)
//!   stderr  → human-readable diagnostics
//!   exit 0  → success
//!   exit 1  → failure (stdout still valid JSON with error)
//!
//! Build:
//!   Add to build.zig, then: zig build
//!
//! Run:
//!   ./zig-out/bin/engine_research <tool> [args...]

const std = @import("std");
const Allocator = std.mem.Allocator;
const tools = @import("tools.zig");

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const ENGINE_FILES = [_][]const u8{
    "nn/src/transformer.zig",
    "nn/src/model.zig",
    "nn/src/metal.zig",
    "nn/src/safetensors.zig",
    "nn/src/tokenizer.zig",
    "nn/src/shaders/transformer.metal",
    "nn/src/shaders/compute.metal",
    "nn/examples/bonsai.zig",
    "nn/examples/bonsai_bench.zig",
};

const SNAPSHOT_DIR = ".engine_snapshots";
const MAX_SNAPSHOTS: u32 = 128;
const MAX_DIFF_LINES: u32 = 200_000;
const MAX_FUNCTION_NAME: u32 = 256;
const MAX_PATH_LEN: u32 = 512;
const TIMESTAMP_LEN: u32 = 19; // "2025-01-15T14-30-00"

// Compile-time validation (Rule 14 — comptime all the things).
comptime {
    std.debug.assert(ENGINE_FILES.len > 0);
    std.debug.assert(ENGINE_FILES.len <= 16);
    std.debug.assert(TIMESTAMP_LEN == 19);
    // Bonsai files: 9 source files covering the
    // transformer inference stack.
    std.debug.assert(ENGINE_FILES.len == 9);
}

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
// Entry point
// ============================================================

pub fn main() !void {
    var arena_state = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const args = try std.process.argsAlloc(arena);
    std.debug.assert(args.len >= 1);

    if (args.len < 2) {
        try toolHelp();
        std.process.exit(1);
    }

    const cmd = args[1];
    const rest: []const []const u8 = if (args.len > 2)
        args[2..]
    else
        &.{};

    dispatch(arena, cmd, rest) catch |err| {
        try tools.writeJsonError(err);
        std.process.exit(1);
    };
}

fn dispatch(
    arena: Allocator,
    cmd: []const u8,
    args: []const []const u8,
) !void {
    std.debug.assert(cmd.len > 0);
    std.debug.assert(cmd.len < MAX_PATH_LEN);

    if (tools.eql(cmd, "help")) return toolHelp();
    if (tools.eql(cmd, "snapshot")) {
        return toolSnapshot(arena);
    }
    if (tools.eql(cmd, "snapshot-list")) {
        return toolSnapshotList(arena);
    }
    if (tools.eql(cmd, "rollback")) {
        return toolRollback(arena, args);
    }
    if (tools.eql(cmd, "rollback-latest")) {
        return toolRollbackLatest(arena);
    }
    if (tools.eql(cmd, "diff")) {
        return toolDiff(arena, args);
    }
    if (tools.eql(cmd, "check")) return toolCheck(arena);
    if (tools.eql(cmd, "test")) return toolTest(arena);
    if (tools.eql(cmd, "bench")) return toolBench(arena);
    if (tools.eql(cmd, "bench-infer")) {
        return toolBenchInfer(arena);
    }
    if (tools.eql(cmd, "bench-compare")) {
        return toolBenchCompare(arena);
    }
    if (tools.eql(cmd, "show")) {
        return toolShow(arena, args);
    }
    if (tools.eql(cmd, "show-function")) {
        return toolShowFunction(arena, args);
    }

    try toolHelp();
    std.process.exit(1);
}

// ============================================================
// Tool: help
// ============================================================

fn toolHelp() !void {
    try tools.writeStdout(
        \\{
        \\  "tools": [
        \\    {
        \\      "name": "snapshot",
        \\      "description": "Save engine files to snapshot",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "snapshot-list",
        \\      "description": "List all snapshots",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "rollback",
        \\      "description": "Restore from a snapshot",
        \\      "args": ["<snapshot_id>"]
        \\    },
        \\    {
        \\      "name": "rollback-latest",
        \\      "description": "Restore most recent snapshot",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "diff",
        \\      "description": "Diff current vs snapshot",
        \\      "args": ["<snapshot_id>"]
        \\    },
        \\    {
        \\      "name": "check",
        \\      "description": "Compile-only (zig build)",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "test",
        \\      "description": "Run zig build test",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "bench",
        \\      "description": "Full training benchmark",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "bench-infer",
        \\      "description": "Inference benchmark (GPU batched, GPU single, CPU single latency). Returns JSON.",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "bench-compare",
        \\      "description": "Compare benchmark results",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "show",
        \\      "description": "Show engine source file",
        \\      "args": ["<file>"]
        \\    },
        \\    {
        \\      "name": "show-function",
        \\      "description": "Extract a function body",
        \\      "args": ["<file>", "<function_name>"]
        \\    }
        \\  ]
        \\}
        \\
    );
}

// ============================================================
// Tool: snapshot — copy engine sources to timestamped dir
// ============================================================

fn toolSnapshot(arena: Allocator) !void {
    var ts_buf: [32]u8 = undefined;
    const timestamp = formatTimestamp(&ts_buf);
    std.debug.assert(timestamp.len == TIMESTAMP_LEN);

    const snap_dir = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ SNAPSHOT_DIR, timestamp },
    );
    const snap_dir_fs = try tools.resolveToFs(
        arena,
        snap_dir,
    );

    // Create the snapshot root directory.
    try std.fs.cwd().makePath(snap_dir_fs);

    const count = try snapshotCopyFiles(
        arena,
        snap_dir_fs,
    );
    std.debug.assert(count <= ENGINE_FILES.len);

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
    arena: Allocator,
    snap_dir_fs: []const u8,
) !u32 {
    std.debug.assert(snap_dir_fs.len > 0);
    var count: u32 = 0;

    for (&ENGINE_FILES) |file| {
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

        const file_fs = try tools.resolveToFs(arena, file);
        tools.copyFile(file_fs, dest) catch |err| {
            std.debug.print(
                "engine_research: skip {s}: {s}\n",
                .{ file, @errorName(err) },
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

fn toolSnapshotList(arena: Allocator) !void {
    const dirs = try listSnapshotDirs(arena);
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
    try restoreSnapshot(arena, snapshot_id);
}

fn toolRollbackLatest(arena: Allocator) !void {
    const latest = try findLatestSnapshot(arena);
    std.debug.assert(latest.len > 0);
    try restoreSnapshot(arena, latest);
}

/// Restore all engine files from a named snapshot.
fn restoreSnapshot(
    arena: Allocator,
    snapshot_id: []const u8,
) !void {
    std.debug.assert(snapshot_id.len > 0);

    const snap_dir = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ SNAPSHOT_DIR, snapshot_id },
    );
    const snap_dir_fs = try tools.resolveToFs(
        arena,
        snap_dir,
    );

    // Verify the snapshot directory exists.
    std.fs.cwd().access(snap_dir_fs, .{}) catch {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"snapshot not found: " ++
                "{s}\"}}\n",
            .{snapshot_id},
        );
        try tools.writeStdout(json);
        return;
    };

    var count: u32 = 0;
    for (&ENGINE_FILES) |file| {
        const source = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ snap_dir_fs, file },
        );
        const file_fs = try tools.resolveToFs(arena, file);
        tools.copyFile(source, file_fs) catch continue;
        count += 1;
    }

    std.debug.assert(count <= ENGINE_FILES.len);
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
            "\"snapshot_id\": \"{s}\", \"files\": [\n",
        .{snapshot_id},
    );
    try buf.appendSlice(arena, header);

    for (&ENGINE_FILES, 0..) |file, i| {
        if (i > 0) try buf.appendSlice(arena, ",\n");
        const entry = try formatDiffEntry(
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
    arena: Allocator,
    snapshot_id: []const u8,
    file: []const u8,
) ![]const u8 {
    std.debug.assert(snapshot_id.len > 0);
    std.debug.assert(file.len > 0);

    const result = diffOneFile(arena, snapshot_id, file);

    if (result.changed) {
        return std.fmt.allocPrint(
            arena,
            "  {{\"file\": \"{s}\", " ++
                "\"changed\": true, " ++
                "\"additions\": {d}, " ++
                "\"deletions\": {d}}}",
            .{ file, result.additions, result.deletions },
        );
    }

    return std.fmt.allocPrint(
        arena,
        "  {{\"file\": \"{s}\", \"changed\": false}}",
        .{file},
    );
}

/// Shell out to diff(1) for a single file pair.
/// Returns change counts; never fails (defaults to changed
/// if something goes wrong).
fn diffOneFile(
    arena: Allocator,
    snapshot_id: []const u8,
    file: []const u8,
) DiffFileResult {
    const snap_rel = std.fmt.allocPrint(
        arena,
        "{s}/{s}/{s}",
        .{ SNAPSHOT_DIR, snapshot_id, file },
    ) catch return .{
        .changed = true,
        .additions = 0,
        .deletions = 0,
    };

    const snap_path = tools.resolveToFs(
        arena,
        snap_rel,
    ) catch return .{
        .changed = true,
        .additions = 0,
        .deletions = 0,
    };

    const file_fs = tools.resolveToFs(
        arena,
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

    // diff exit codes: 0 = identical, 1 = different, 2+ = error.
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
    std.debug.assert(output.len <= tools.MAX_OUTPUT_BYTES);
    var additions: u32 = 0;
    var deletions: u32 = 0;

    var iter = std.mem.splitScalar(u8, output, '\n');
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
// Tool: check — compile-only validation (zig build)
// ============================================================

fn toolCheck(arena: Allocator) !void {
    std.debug.print(
        "engine_research: running zig build...\n",
        .{},
    );

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &[_][]const u8{ "zig", "build" },
        .cwd = "../" ++ tools.NN_DIR,
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
        try tools.writeStdout(
            "{\"status\": \"ok\", " ++
                "\"compiled\": true}\n",
        );
        return;
    }

    // Compilation failed — include error output.
    try writeCheckFailure(arena, result.stderr);
}

/// Format and emit a check-failure JSON response with
/// compiler error output.
fn writeCheckFailure(
    arena: Allocator,
    stderr_output: []const u8,
) !void {
    std.debug.assert(
        stderr_output.len <= tools.MAX_OUTPUT_BYTES,
    );
    const max_err: u32 = 4000;
    const truncated = if (stderr_output.len > max_err)
        stderr_output[stderr_output.len - max_err ..]
    else
        stderr_output;

    const escaped = try tools.jsonEscape(arena, truncated);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"compiled\": false, " ++
            "\"errors\": \"{s}\"}}\n",
        .{escaped},
    );
    std.debug.assert(json.len > 0);
    try tools.writeStdout(json);
}

// ============================================================
// Tool: test — run zig build test
// ============================================================

fn toolTest(arena: Allocator) !void {
    std.debug.print(
        "engine_research: running zig build test...\n",
        .{},
    );

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &[_][]const u8{
            "zig", "build", "test",
        },
        .cwd = "../" ++ tools.NN_DIR,
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
        try tools.writeStdout(
            "{\"status\": \"ok\", " ++
                "\"passed\": true}\n",
        );
        return;
    }

    try writeTestFailure(arena, result.stderr);
}

/// Format and emit a test-failure JSON response.
fn writeTestFailure(
    arena: Allocator,
    stderr_output: []const u8,
) !void {
    std.debug.assert(
        stderr_output.len <= tools.MAX_OUTPUT_BYTES,
    );
    const max_err: u32 = 4000;
    const truncated = if (stderr_output.len > max_err)
        stderr_output[stderr_output.len - max_err ..]
    else
        stderr_output;

    const escaped = try tools.jsonEscape(arena, truncated);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"passed\": false, " ++
            "\"output\": \"{s}\"}}\n",
        .{escaped},
    );
    std.debug.assert(json.len > 0);
    try tools.writeStdout(json);
}

// ============================================================
// Tool: bench — full training benchmark
// ============================================================

fn toolBench(arena: Allocator) !void {
    // Bonsai bench must run in ReleaseFast — debug mode
    // is unusably slow (~0.5 tok/s vs ~37 tok/s).
    try runBonsaiBench(arena);
}

fn toolBenchInfer(arena: Allocator) !void {
    // bench and bench-infer are the same for bonsai:
    // there is only one benchmark (decode throughput).
    try runBonsaiBench(arena);
}

/// Run the bonsai inference benchmark and emit its
/// JSON output to stdout.  The benchmark binary
/// writes structured JSON to stdout directly.
fn runBonsaiBench(arena: Allocator) !void {
    std.debug.print(
        "engine_research: running bonsai bench " ++
            "(ReleaseFast)...\n",
        .{},
    );

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &[_][]const u8{
            "zig",
            "build",
            "run-bonsai-bench",
            "-Doptimize=ReleaseFast",
        },
        .cwd = "../" ++ tools.NN_DIR,
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

    // Forward diagnostic output to stderr.
    if (result.stderr.len > 0) {
        tools.stderr_file.writeAll(result.stderr) catch {};
    }

    const success = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (!success) {
        try writeBenchError(arena, result.stderr);
        return;
    }

    // The bonsai bench binary writes JSON to stdout.
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

/// Locate the new benchmark file and write it to stdout.
fn emitBenchmarkResult(
    arena: Allocator,
    bench_fs: []const u8,
) !void {
    std.debug.assert(bench_fs.len > 0);

    const bench_name = tools.findOneBenchmark(
        arena,
        bench_fs,
    );

    if (bench_name) |name| {
        const path = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ bench_fs, name },
        );
        const content = try tools.readFile(arena, path);
        std.debug.assert(content.len > 0);
        try tools.writeStdout(content);
    } else {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"no benchmark " ++
                "produced\"}\n",
        );
    }
}

/// Format and emit a bench-failure JSON response.
fn writeBenchError(
    arena: Allocator,
    stderr_output: []const u8,
) !void {
    std.debug.assert(
        stderr_output.len <= tools.MAX_OUTPUT_BYTES,
    );
    const max_err: u32 = 2000;
    const truncated = if (stderr_output.len > max_err)
        stderr_output[stderr_output.len - max_err ..]
    else
        stderr_output;

    const escaped = try tools.jsonEscape(arena, truncated);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"error\", " ++
            "\"error\": \"build_failed\", " ++
            "\"output\": \"{s}\"}}\n",
        .{escaped},
    );
    std.debug.assert(json.len > 0);
    try tools.writeStdout(json);
}

// ============================================================
// Tool: bench-compare — compare all benchmark results
// ============================================================

fn toolBenchCompare(arena: Allocator) !void {
    const bench_fs = try tools.resolveToFs(
        arena,
        tools.NN_DIR ++ "/" ++ tools.BENCH_DIR,
    );
    try tools.toolBenchCompare(arena, bench_fs);
}

// ============================================================
// Tool: show — display contents of an engine source file
// ============================================================

fn toolShow(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"usage: show <file>\"}\n",
        );
        return;
    }

    const short_name = args[0];
    const path = resolveEnginePath(short_name);
    if (path == null) {
        try writeFileNotFound(arena, short_name);
        return;
    }

    const path_fs = try tools.resolveToFs(arena, path.?);
    const content = try tools.readFile(arena, path_fs);
    const lines = countLines(content);
    const escaped = try tools.jsonEscape(arena, content);

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
    arena: Allocator,
    short_name: []const u8,
) !void {
    std.debug.assert(short_name.len > 0);
    std.debug.assert(ENGINE_FILES.len > 0);

    var names_buf: std.ArrayList(u8) = .empty;
    for (&ENGINE_FILES, 0..) |file, i| {
        if (i > 0) {
            names_buf.appendSlice(arena, ", ") catch {};
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
// Tool: show-function — extract a function body from source
// ============================================================

fn toolShowFunction(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 2) {
        try tools.writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"usage: show-function " ++
                "<file> <function_name>\"}\n",
        );
        return;
    }

    const short_name = args[0];
    const function_name = args[1];
    std.debug.assert(function_name.len > 0);

    const path = resolveEnginePath(short_name);
    if (path == null) {
        try writeFileNotFound(arena, short_name);
        return;
    }

    const path_fs = try tools.resolveToFs(arena, path.?);
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

/// Find the start and end line numbers of a named function.
/// Uses a simple brace-depth parser: find the line containing
/// the function signature, then track brace depth until it
/// returns to zero.  Line numbers are 1-based.
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

/// Scan content for a function matching the given pattern,
/// then track brace depth to find the closing brace.
/// Returns null if the pattern is not found.
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

    var iter = std.mem.splitScalar(u8, content, '\n');
    while (iter.next()) |line| {
        line_number += 1;

        if (!found) {
            // Search for the function signature.
            if (std.mem.indexOf(u8, line, pattern)) |_| {
                found = true;
                start_line = line_number;
            } else {
                continue;
            }
        }

        // Count braces on this line.
        depth = countBracesOnLine(line, depth);

        // Track whether we have entered the function body
        // (seen at least one opening brace).
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

/// Count braces on a single line and return updated depth.
/// Ignores braces inside string literals (basic handling
/// for double-quoted strings and single-quoted chars).
fn countBracesOnLine(line: []const u8, depth: i32) i32 {
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

    var iter = std.mem.splitScalar(u8, content, '\n');
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

    const es = std.time.epoch.EpochSeconds{ .secs = ts };
    const epoch_day = es.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = es.getDaySeconds();

    const year = year_day.year;
    const month = month_day.month.numeric();
    const day: u32 = @as(u32, month_day.day_index) + 1;
    const hour = day_secs.getHoursIntoDay();
    const minute = day_secs.getMinutesIntoHour();
    const second = day_secs.getSecondsIntoMinute();

    std.debug.assert(month >= 1);
    std.debug.assert(month <= 12);
    return std.fmt.bufPrint(
        buf,
        "{d:0>4}-{d:0>2}-{d:0>2}T" ++
            "{d:0>2}-{d:0>2}-{d:0>2}",
        .{ year, month, day, hour, minute, second },
    ) catch "0000-00-00T00-00-00";
}

// ============================================================
// Snapshot directory helpers
// ============================================================

/// List all snapshot directory names, sorted ascending.
fn listSnapshotDirs(
    arena: Allocator,
) ![]const []const u8 {
    const snap_fs = try tools.resolveToFs(
        arena,
        SNAPSHOT_DIR,
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

/// Find the lexicographically latest snapshot (most recent).
/// Fails with error.NoSnapshots if none exist.
fn findLatestSnapshot(
    arena: Allocator,
) ![]const u8 {
    const dirs = try listSnapshotDirs(arena);
    std.debug.assert(SNAPSHOT_DIR.len > 0);
    if (dirs.len == 0) return error.NoSnapshots;
    return dirs[dirs.len - 1];
}

// ============================================================
// Engine file resolution
// ============================================================

/// Map a short filename ("metal.zig", "compute.metal", etc.)
/// to its full relative path ("nn/src/metal.zig",
/// "nn/src/shaders/compute.metal", etc.).
/// Also accepts full paths like "nn/src/metal.zig".
fn resolveEnginePath(
    short_name: []const u8,
) ?[]const u8 {
    std.debug.assert(short_name.len > 0);

    // First, check for exact match against full paths.
    for (&ENGINE_FILES) |path| {
        if (tools.eql(path, short_name)) return path;
    }

    // Then, check if any path ends with the short name.
    for (&ENGINE_FILES) |path| {
        if (std.mem.endsWith(u8, path, short_name)) {
            return path;
        }
    }

    std.debug.assert(ENGINE_FILES.len > 0);
    return null;
}

// ============================================================
// Line counting
// ============================================================

/// Count the number of lines in a text buffer.
fn countLines(content: []const u8) u32 {
    std.debug.assert(content.len <= tools.MAX_FILE_SIZE);
    if (content.len == 0) return 0;

    var count: u32 = 0;
    for (content) |c| {
        if (c == '\n') count += 1;
    }

    // Count the last line if it doesn't end with newline.
    if (content[content.len - 1] != '\n') {
        count += 1;
    }
    std.debug.assert(count > 0);
    return count;
}
