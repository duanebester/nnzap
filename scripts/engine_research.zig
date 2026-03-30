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

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const ENGINE_FILES = [_][]const u8{
    "src/metal.zig",
    "src/network.zig",
    "src/shaders/compute.metal",
    "src/layout.zig",
    "src/main.zig",
};

const SNAPSHOT_DIR = ".engine_snapshots";
const BENCH_DIR = "benchmarks";
const MAX_FILE_SIZE: u32 = 2 * 1024 * 1024; // 2 MB.
const MAX_OUTPUT_BYTES: u32 = 4 * 1024 * 1024; // 4 MB.
const MAX_SNAPSHOTS: u32 = 128;
const MAX_DIFF_LINES: u32 = 200_000;
const MAX_FUNCTION_NAME: u32 = 256;
const MAX_PATH_LEN: u32 = 512;
const TIMESTAMP_LEN: u32 = 19; // "2025-01-15T14-30-00"

// Compile-time validation (Rule 14 — comptime all the things).
comptime {
    std.debug.assert(ENGINE_FILES.len > 0);
    std.debug.assert(ENGINE_FILES.len <= 16);
    std.debug.assert(MAX_FILE_SIZE > 0);
    std.debug.assert(MAX_OUTPUT_BYTES >= MAX_FILE_SIZE);
    std.debug.assert(TIMESTAMP_LEN == 19);
}

// ============================================================
// Benchmark JSON types — mirrors benchmark.zig output
// (needed for bench-compare)
// ============================================================

const ArchLayer = struct {
    input_size: u64 = 0,
    output_size: u64 = 0,
    activation: []const u8 = "",
};

const EpochEntry = struct {
    epoch: u64 = 0,
    train_loss: f64 = 0,
    duration_ms: f64 = 0,
    validation_loss: ?f64 = null,
    validation_accuracy_pct: ?f64 = null,
};

const TestResultJson = struct {
    correct: u64 = 0,
    total: u64 = 0,
    accuracy_pct: f64 = 0,
    duration_ms: f64 = 0,
};

const BenchConfig = struct {
    architecture: []const ArchLayer = &.{},
    param_count: u64 = 0,
    batch_size: u64 = 0,
    learning_rate: f64 = 0,
    learning_rate_decay: f64 = 0,
    optimizer: []const u8 = "",
    loss_function: []const u8 = "",
    num_epochs: u64 = 0,
    seed: u64 = 0,
    train_samples: u64 = 0,
    validation_samples: u64 = 0,
    test_samples: u64 = 0,
};

const BenchResult = struct {
    timestamp_utc: []const u8 = "",
    config: BenchConfig = .{},
    final_train_loss: f64 = 0,
    final_validation_loss: ?f64 = null,
    final_validation_accuracy_pct: ?f64 = null,
    final_test_accuracy_pct: ?f64 = null,
    epochs: []const EpochEntry = &.{},
    test_result: ?TestResultJson = null,
    total_training_ms: f64 = 0,
    throughput_images_per_sec: f64 = 0,
};

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
        try writeJsonError(err);
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

    if (eql(cmd, "help")) return toolHelp();
    if (eql(cmd, "snapshot")) return toolSnapshot(arena);
    if (eql(cmd, "snapshot-list")) {
        return toolSnapshotList(arena);
    }
    if (eql(cmd, "rollback")) {
        return toolRollback(arena, args);
    }
    if (eql(cmd, "rollback-latest")) {
        return toolRollbackLatest(arena);
    }
    if (eql(cmd, "diff")) return toolDiff(arena, args);
    if (eql(cmd, "check")) return toolCheck(arena);
    if (eql(cmd, "test")) return toolTest(arena);
    if (eql(cmd, "bench")) return toolBench(arena);
    if (eql(cmd, "bench-compare")) {
        return toolBenchCompare(arena);
    }
    if (eql(cmd, "show")) return toolShow(arena, args);
    if (eql(cmd, "show-function")) {
        return toolShowFunction(arena, args);
    }

    try toolHelp();
    std.process.exit(1);
}

// ============================================================
// Tool: help
// ============================================================

fn toolHelp() !void {
    try writeStdout(
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

    // Create the snapshot root directory.
    try std.fs.cwd().makePath(snap_dir);

    const count = try snapshotCopyFiles(arena, snap_dir);
    std.debug.assert(count <= ENGINE_FILES.len);

    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"snapshot_id\": \"{s}\", " ++
            "\"files\": {d}}}\n",
        .{ timestamp, count },
    );
    try writeStdout(json);
}

/// Copy each engine file into the snapshot directory,
/// preserving the relative path structure.
fn snapshotCopyFiles(
    arena: Allocator,
    snap_dir: []const u8,
) !u32 {
    std.debug.assert(snap_dir.len > 0);
    var count: u32 = 0;

    for (&ENGINE_FILES) |file| {
        std.debug.assert(file.len > 0);

        // Ensure parent directories exist inside snapshot.
        const dest = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ snap_dir, file },
        );
        if (std.fs.path.dirname(dest)) |parent| {
            try std.fs.cwd().makePath(parent);
        }

        copyFile(file, dest) catch |err| {
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
    try writeStdout(buf.items);
}

// ============================================================
// Tool: rollback — restore engine files from a snapshot
// ============================================================

fn toolRollback(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try writeStdout(
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

    // Verify the snapshot directory exists.
    std.fs.cwd().access(snap_dir, .{}) catch {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"snapshot not found: " ++
                "{s}\"}}\n",
            .{snapshot_id},
        );
        try writeStdout(json);
        return;
    };

    var count: u32 = 0;
    for (&ENGINE_FILES) |file| {
        const source = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ snap_dir, file },
        );
        copyFile(source, file) catch continue;
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
    try writeStdout(json);
}

// ============================================================
// Tool: diff — compare current files against a snapshot
// ============================================================

fn toolDiff(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try writeStdout(
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
    try writeStdout(buf.items);
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
    const snap_path = std.fmt.allocPrint(
        arena,
        "{s}/{s}/{s}",
        .{ SNAPSHOT_DIR, snapshot_id, file },
    ) catch return .{
        .changed = true,
        .additions = 0,
        .deletions = 0,
    };

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &[_][]const u8{
            "diff", "-u", snap_path, file,
        },
        .max_output_bytes = MAX_OUTPUT_BYTES,
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
    std.debug.assert(output.len <= MAX_OUTPUT_BYTES);
    var additions: u32 = 0;
    var deletions: u32 = 0;

    var iter = std.mem.splitScalar(u8, output, '\n');
    while (iter.next()) |line| {
        if (line.len == 0) continue;
        // Skip unified diff headers.
        if (startsWith(line, "---")) continue;
        if (startsWith(line, "+++")) continue;
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
        .max_output_bytes = MAX_OUTPUT_BYTES,
    }) catch |err| {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"spawn_failed: " ++
                "{s}\"}}\n",
            .{@errorName(err)},
        );
        try writeStdout(json);
        return;
    };

    const success = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (success) {
        try writeStdout(
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
    std.debug.assert(stderr_output.len <= MAX_OUTPUT_BYTES);
    const max_err: u32 = 4000;
    const truncated = if (stderr_output.len > max_err)
        stderr_output[stderr_output.len - max_err ..]
    else
        stderr_output;

    const escaped = try jsonEscape(arena, truncated);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"compiled\": false, " ++
            "\"errors\": \"{s}\"}}\n",
        .{escaped},
    );
    std.debug.assert(json.len > 0);
    try writeStdout(json);
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
        .max_output_bytes = MAX_OUTPUT_BYTES,
    }) catch |err| {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"spawn_failed: " ++
                "{s}\"}}\n",
            .{@errorName(err)},
        );
        try writeStdout(json);
        return;
    };

    const success = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (success) {
        try writeStdout(
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
    std.debug.assert(stderr_output.len <= MAX_OUTPUT_BYTES);
    const max_err: u32 = 4000;
    const truncated = if (stderr_output.len > max_err)
        stderr_output[stderr_output.len - max_err ..]
    else
        stderr_output;

    const escaped = try jsonEscape(arena, truncated);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"passed\": false, " ++
            "\"output\": \"{s}\"}}\n",
        .{escaped},
    );
    std.debug.assert(json.len > 0);
    try writeStdout(json);
}

// ============================================================
// Tool: bench — full training benchmark
// ============================================================

fn toolBench(arena: Allocator) !void {
    // Clean old benchmarks so we find exactly the new one.
    _ = cleanBenchmarkFiles();

    std.debug.print(
        "engine_research: running zig build run...\n",
        .{},
    );

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &[_][]const u8{ "zig", "build", "run" },
        .max_output_bytes = MAX_OUTPUT_BYTES,
    }) catch |err| {
        const json = try std.fmt.allocPrint(
            arena,
            "{{\"status\": \"error\", " ++
                "\"error\": \"spawn_failed: " ++
                "{s}\"}}\n",
            .{@errorName(err)},
        );
        try writeStdout(json);
        return;
    };

    // Forward training output to stderr for diagnostics.
    if (result.stderr.len > 0) {
        stderr_file.writeAll(result.stderr) catch {};
    }

    const success = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (!success) {
        try writeBenchError(arena, result.stderr);
        return;
    }

    // Find and output the benchmark JSON.
    try emitBenchmarkResult(arena);
}

/// Locate the new benchmark file and write it to stdout.
fn emitBenchmarkResult(arena: Allocator) !void {
    const bench_name = findOneBenchmark(arena);
    std.debug.assert(BENCH_DIR.len > 0);

    if (bench_name) |name| {
        const path = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ BENCH_DIR, name },
        );
        const content = try readFile(arena, path);
        std.debug.assert(content.len > 0);
        try writeStdout(content);
    } else {
        try writeStdout(
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
    std.debug.assert(stderr_output.len <= MAX_OUTPUT_BYTES);
    const max_err: u32 = 2000;
    const truncated = if (stderr_output.len > max_err)
        stderr_output[stderr_output.len - max_err ..]
    else
        stderr_output;

    const escaped = try jsonEscape(arena, truncated);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"error\", " ++
            "\"error\": \"build_failed\", " ++
            "\"output\": \"{s}\"}}\n",
        .{escaped},
    );
    std.debug.assert(json.len > 0);
    try writeStdout(json);
}

// ============================================================
// Tool: bench-compare — compare all benchmark results
// ============================================================

fn toolBenchCompare(arena: Allocator) !void {
    const names = try listBenchmarkFiles(arena);
    var buf: std.ArrayList(u8) = .empty;

    try buf.appendSlice(arena, "[\n");
    for (names, 0..) |name, i| {
        if (i > 0) try buf.appendSlice(arena, ",\n");
        const entry = formatCompareEntry(arena, name);
        if (entry) |json| {
            try buf.appendSlice(arena, json);
        } else |_| {
            const fallback = try std.fmt.allocPrint(
                arena,
                "  {{\"file\": \"{s}\", " ++
                    "\"error\": \"parse_failed\"}}",
                .{name},
            );
            try buf.appendSlice(arena, fallback);
        }
    }

    try buf.appendSlice(arena, "\n]\n");
    std.debug.assert(buf.items.len > 0);
    try writeStdout(buf.items);
}

/// Parse one benchmark JSON and format a comparison entry.
fn formatCompareEntry(
    arena: Allocator,
    name: []const u8,
) ![]const u8 {
    std.debug.assert(name.len > 0);

    const path = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ BENCH_DIR, name },
    );
    const content = try readFile(arena, path);
    const parsed = try std.json.parseFromSlice(
        BenchResult,
        arena,
        content,
        .{},
    );
    const r = parsed.value;
    const c = r.config;

    const arch_str = buildArchString(arena, c.architecture);
    const test_acc = r.final_test_accuracy_pct orelse 0.0;
    const val_acc =
        r.final_validation_accuracy_pct orelse 0.0;

    std.debug.assert(BENCH_DIR.len > 0);
    return std.fmt.allocPrint(
        arena,
        "  {{" ++
            "\"file\": \"{s}\", " ++
            "\"test_accuracy_pct\": {d:.2}, " ++
            "\"val_accuracy_pct\": {d:.2}, " ++
            "\"throughput\": {d:.0}, " ++
            "\"training_time_ms\": {d:.0}, " ++
            "\"optimizer\": \"{s}\", " ++
            "\"learning_rate\": {d}, " ++
            "\"batch_size\": {d}, " ++
            "\"num_epochs\": {d}, " ++
            "\"param_count\": {d}, " ++
            "\"architecture\": \"{s}\"" ++
            "}}",
        .{
            name,
            test_acc,
            val_acc,
            r.throughput_images_per_sec,
            r.total_training_ms,
            c.optimizer,
            c.learning_rate,
            c.batch_size,
            c.num_epochs,
            c.param_count,
            arch_str,
        },
    );
}

/// Build a human-readable architecture string like
/// "784->128->10" from the benchmark config layers.
fn buildArchString(
    arena: Allocator,
    layers: []const ArchLayer,
) []const u8 {
    if (layers.len == 0) return "(empty)";
    std.debug.assert(layers.len < 64);
    var buf: std.ArrayList(u8) = .empty;

    for (layers) |layer| {
        if (buf.items.len > 0) {
            buf.appendSlice(arena, "->") catch {
                return "?";
            };
        }
        var num_buf: [16]u8 = undefined;
        const s = std.fmt.bufPrint(
            &num_buf,
            "{d}",
            .{layer.output_size},
        ) catch return "?";
        buf.appendSlice(arena, s) catch return "?";
    }

    std.debug.assert(buf.items.len > 0);
    return buf.items;
}

// ============================================================
// Tool: show — display contents of an engine source file
// ============================================================

fn toolShow(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 1) {
        try writeStdout(
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

    const content = try readFile(arena, path.?);
    const lines = countLines(content);
    const escaped = try jsonEscape(arena, content);

    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"file\": \"{s}\", " ++
            "\"lines\": {d}, " ++
            "\"content\": \"{s}\"}}\n",
        .{ short_name, lines, escaped },
    );
    std.debug.assert(json.len > 0);
    try writeStdout(json);
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
    try writeStdout(json);
}

// ============================================================
// Tool: show-function — extract a function body from source
// ============================================================

fn toolShowFunction(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len < 2) {
        try writeStdout(
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

    const content = try readFile(arena, path.?);
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
    try writeStdout(json);
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

    const escaped = try jsonEscape(arena, body);
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
    try writeStdout(json);
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
    std.debug.assert(content.len <= MAX_FILE_SIZE);

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
    std.debug.assert(line.len <= MAX_FILE_SIZE);
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
    var dir = std.fs.cwd().openDir(
        SNAPSHOT_DIR,
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

    sortStrings(names.items);
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
/// to its full relative path ("src/metal.zig",
/// "src/shaders/compute.metal", etc.).
/// Also accepts full paths like "src/metal.zig".
fn resolveEnginePath(
    short_name: []const u8,
) ?[]const u8 {
    std.debug.assert(short_name.len > 0);

    // First, check for exact match against full paths.
    for (&ENGINE_FILES) |path| {
        if (eql(path, short_name)) return path;
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
    std.debug.assert(content.len <= MAX_FILE_SIZE);
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

// ============================================================
// Benchmark file helpers
// ============================================================

fn listBenchmarkFiles(
    arena: Allocator,
) ![]const []const u8 {
    var dir = std.fs.cwd().openDir(
        BENCH_DIR,
        .{ .iterate = true },
    ) catch return &.{};
    defer dir.close();

    var names: std.ArrayList([]const u8) = .empty;
    var iter = dir.iterate();

    while (try iter.next()) |entry| {
        if (isBenchmarkFile(entry.name)) {
            const dupe = try arena.dupe(u8, entry.name);
            try names.append(arena, dupe);
        }
    }

    // Selection sort — correct for small N.
    sortStrings(names.items);
    std.debug.assert(BENCH_DIR.len > 0);
    return names.items;
}

fn findOneBenchmark(arena: Allocator) ?[]const u8 {
    var dir = std.fs.cwd().openDir(
        BENCH_DIR,
        .{ .iterate = true },
    ) catch return null;
    defer dir.close();

    var iter = dir.iterate();
    while (iter.next() catch null) |entry| {
        if (isBenchmarkFile(entry.name)) {
            return arena.dupe(u8, entry.name) catch null;
        }
    }
    return null;
}

fn cleanBenchmarkFiles() u32 {
    var dir = std.fs.cwd().openDir(
        BENCH_DIR,
        .{ .iterate = true },
    ) catch return 0;
    defer dir.close();

    var count: u32 = 0;
    var iter = dir.iterate();
    while (iter.next() catch null) |entry| {
        if (isBenchmarkFile(entry.name)) {
            dir.deleteFile(entry.name) catch {};
            count += 1;
        }
    }
    std.debug.assert(BENCH_DIR.len > 0);
    return count;
}

fn isBenchmarkFile(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "mnist_") and
        std.mem.endsWith(u8, name, ".json");
}

// ============================================================
// Sorting
// ============================================================

/// Selection sort for small string arrays.
fn sortStrings(items: [][]const u8) void {
    if (items.len < 2) return;
    for (0..items.len - 1) |i| {
        var min_idx = i;
        for (i + 1..items.len) |j| {
            if (std.mem.order(
                u8,
                items[j],
                items[min_idx],
            ) == .lt) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            const tmp = items[i];
            items[i] = items[min_idx];
            items[min_idx] = tmp;
        }
    }
}

// ============================================================
// File I/O
// ============================================================

fn readFile(
    arena: Allocator,
    path: []const u8,
) ![]const u8 {
    std.debug.assert(path.len > 0);
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try file.readToEndAlloc(
        arena,
        MAX_FILE_SIZE,
    );
}

fn copyFile(source: []const u8, dest: []const u8) !void {
    std.debug.assert(source.len > 0);
    std.debug.assert(dest.len > 0);
    try std.fs.cwd().copyFile(
        source,
        std.fs.cwd(),
        dest,
        .{},
    );
}

// ============================================================
// Stdout / stderr I/O
// ============================================================

const stdout_file = std.fs.File{
    .handle = std.posix.STDOUT_FILENO,
};
const stderr_file = std.fs.File{
    .handle = std.posix.STDERR_FILENO,
};

fn writeStdout(bytes: []const u8) !void {
    std.debug.assert(bytes.len > 0);
    try stdout_file.writeAll(bytes);
}

// ============================================================
// JSON output helpers
// ============================================================

fn writeJsonError(err: anyerror) !void {
    var buf: [256]u8 = undefined;
    const json = std.fmt.bufPrint(
        &buf,
        "{{\"status\": \"error\", " ++
            "\"error\": \"{s}\"}}\n",
        .{@errorName(err)},
    ) catch
        "{\"status\": \"error\", " ++
            "\"error\": \"unknown\"}\n";
    try writeStdout(json);
}

fn jsonEscape(
    arena: Allocator,
    input: []const u8,
) ![]const u8 {
    std.debug.assert(input.len <= MAX_OUTPUT_BYTES);
    var buf: std.ArrayList(u8) = .empty;

    for (input) |c| {
        switch (c) {
            '"' => try buf.appendSlice(arena, "\\\""),
            '\\' => try buf.appendSlice(arena, "\\\\"),
            '\n' => try buf.appendSlice(arena, "\\n"),
            '\r' => try buf.appendSlice(arena, "\\r"),
            '\t' => try buf.appendSlice(arena, "\\t"),
            else => try buf.append(arena, c),
        }
    }

    std.debug.assert(buf.items.len >= input.len);
    return buf.items;
}

// ============================================================
// String helpers
// ============================================================

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn startsWith(
    haystack: []const u8,
    prefix: []const u8,
) bool {
    return std.mem.startsWith(u8, haystack, prefix);
}
