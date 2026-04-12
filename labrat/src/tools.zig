//! labrat tools — shared utilities for AI agent toolboxes.
//!
//! Common infrastructure used by `bonsai_researcher.zig`
//! and `mnist_researcher.zig`: benchmark JSON types, file
//! I/O, path resolution, JSON/string helpers, and benchmark
//! comparison logic.
//!
//! Design: functions accept already-resolved filesystem
//! paths.  Callers use `resolveToFs` when they need to
//! translate monorepo-relative paths to zap/-relative
//! filesystem paths.

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

pub const MAX_FILE_SIZE: u32 = 2 * 1024 * 1024;
pub const MAX_OUTPUT_BYTES: u32 = 4 * 1024 * 1024;
pub const BENCH_DIR = "benchmarks";
pub const NN_DIR = "nn";

// Compile-time validation (Rule 14).
comptime {
    std.debug.assert(MAX_FILE_SIZE > 0);
    std.debug.assert(MAX_OUTPUT_BYTES >= MAX_FILE_SIZE);
}

// ============================================================
// Benchmark JSON types — mirrors benchmark.zig output
// ============================================================

pub const ArchLayer = struct {
    input_size: u64 = 0,
    output_size: u64 = 0,
    activation: []const u8 = "",
};

pub const EpochEntry = struct {
    epoch: u64 = 0,
    train_loss: f64 = 0,
    duration_ms: f64 = 0,
    validation_loss: ?f64 = null,
    validation_accuracy_pct: ?f64 = null,
};

pub const TestResultJson = struct {
    correct: u64 = 0,
    total: u64 = 0,
    accuracy_pct: f64 = 0,
    duration_ms: f64 = 0,
};

pub const BenchConfig = struct {
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

pub const BenchResult = struct {
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
// String helpers
// ============================================================

pub fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn startsWith(
    haystack: []const u8,
    prefix: []const u8,
) bool {
    return std.mem.startsWith(u8, haystack, prefix);
}

pub fn indexOf(
    haystack: []const u8,
    needle: []const u8,
) ?usize {
    return std.mem.indexOf(u8, haystack, needle);
}

pub fn truncate(s: []const u8, max: usize) []const u8 {
    std.debug.assert(max > 0);
    return if (s.len <= max) s else s[0..max];
}

pub fn trimCR(line: []const u8) []const u8 {
    if (line.len > 0 and line[line.len - 1] == '\r') {
        return line[0 .. line.len - 1];
    }
    return line;
}

pub fn leadingSpaces(line: []const u8) usize {
    for (line, 0..) |c, i| {
        if (c != ' ') return i;
    }
    return line.len;
}

/// Split "key=value" into [key, value].  Returns null if
/// no separator found.
pub fn splitOnce(
    s: []const u8,
    sep: u8,
) ?[2][]const u8 {
    const idx = std.mem.indexOfScalar(u8, s, sep) orelse
        return null;
    return .{ s[0..idx], s[idx + 1 ..] };
}

/// Extract the value from a `const name: type = VALUE;`
/// line.  Returns the trimmed VALUE string between '='
/// and ';', or null if the line does not start with
/// `prefix`.
pub fn extractAfterEq(
    trimmed: []const u8,
    prefix: []const u8,
) ?[]const u8 {
    if (!startsWith(trimmed, prefix)) return null;
    const eq_pos = std.mem.indexOfScalar(
        u8,
        trimmed,
        '=',
    );
    const semi_pos = std.mem.indexOfScalar(
        u8,
        trimmed,
        ';',
    );
    if (eq_pos == null or semi_pos == null) return null;
    if (semi_pos.? <= eq_pos.?) return null;
    return std.mem.trim(
        u8,
        trimmed[eq_pos.? + 1 .. semi_pos.?],
        " ",
    );
}

/// Extract value from `.field = VALUE` patterns inside
/// struct literals.  VALUE ends at ',' or '}' or end of
/// line.
pub fn extractField(
    trimmed: []const u8,
    needle: []const u8,
) ?[]const u8 {
    const idx = std.mem.indexOf(
        u8,
        trimmed,
        needle,
    ) orelse return null;
    const start = idx + needle.len;
    const rest = trimmed[start..];
    var end: usize = rest.len;
    for (rest, 0..) |c, i| {
        if (c == ',' or c == ' ' or c == '}') {
            end = i;
            break;
        }
    }
    if (end == 0) return null;
    return rest[0..end];
}

/// Extract a dot-prefixed enum value: `.act = .relu` →
/// "relu".
pub fn extractDotField(
    trimmed: []const u8,
    needle: []const u8,
) ?[]const u8 {
    const idx = std.mem.indexOf(
        u8,
        trimmed,
        needle,
    ) orelse return null;
    const start = idx + needle.len;
    const rest = trimmed[start..];
    var end: usize = rest.len;
    for (rest, 0..) |c, i| {
        if (c == ',' or c == ' ' or c == '}') {
            end = i;
            break;
        }
    }
    if (end == 0) return null;
    return rest[0..end];
}

// ============================================================
// JSON output helpers
// ============================================================

/// JSON-escape a string for embedding inside a quoted
/// JSON value.  Uses std.json.Stringify.encodeJsonString
/// for correct handling of all control characters
/// (U+0000–U+001F) and Unicode escaping.
///
/// Returns the escaped content WITHOUT surrounding
/// quotes — callers embed the result inside their own
/// `\"{s}\"` format strings.
pub fn jsonEscape(
    arena: Allocator,
    input: []const u8,
) ![]const u8 {
    std.debug.assert(input.len <= MAX_OUTPUT_BYTES);

    var tmp: std.io.Writer.Allocating = .init(arena);
    try std.json.Stringify.encodeJsonString(
        input,
        .{},
        &tmp.writer,
    );
    const encoded = tmp.written();

    // encodeJsonString wraps the result in quotes.
    // Strip them — callers provide their own quoting.
    std.debug.assert(encoded.len >= 2);
    std.debug.assert(encoded[0] == '"');
    std.debug.assert(encoded[encoded.len - 1] == '"');
    return encoded[1 .. encoded.len - 1];
}

/// Truncate stderr output from the end (keeping the most
/// recent lines) and JSON-escape it for embedding in a
/// JSON response.
pub fn truncateAndEscape(
    arena: Allocator,
    stderr_output: []const u8,
    max_error_bytes: u32,
) ![]const u8 {
    std.debug.assert(max_error_bytes > 0);
    const truncated = if (stderr_output.len > max_error_bytes)
        stderr_output[stderr_output.len - max_error_bytes ..]
    else
        stderr_output;
    return jsonEscape(arena, truncated);
}

/// Emit a build-failure JSON response to stdout.  Used
/// by research toolboxes when a `zig build` subprocess
/// fails.
pub fn writeBuildError(
    arena: Allocator,
    stderr_output: []const u8,
    max_error_bytes: u32,
) !void {
    const escaped = try truncateAndEscape(
        arena,
        stderr_output,
        max_error_bytes,
    );
    const result = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"error\", " ++
            "\"error\": \"build_failed\", " ++
            "\"output\": \"{s}\"}}\n",
        .{escaped},
    );
    std.debug.assert(result.len > 0);
    try writeStdout(result);
}

pub fn writeJsonError(err: anyerror) !void {
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

/// Write an indented, formatted line into a buffer.
pub fn writeIndented(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    indent: usize,
    comptime fmt: []const u8,
    args: anytype,
) void {
    buf.appendNTimes(arena, ' ', indent) catch return;
    const text = std.fmt.allocPrint(
        arena,
        fmt,
        args,
    ) catch return;
    buf.appendSlice(arena, text) catch return;
}

// ============================================================
// Path resolution
// ============================================================

/// Convert a monorepo-root-relative path to a filesystem
/// path by prepending the fs_root prefix.  All config
/// paths (read_scope, write_scope, read_files, bench_dir,
/// history_dir, etc.) are monorepo-relative.  This
/// function makes them resolvable from CWD.
pub fn resolveToFs(
    arena: Allocator,
    root: []const u8,
    path: []const u8,
) ![]const u8 {
    std.debug.assert(path.len > 0);
    std.debug.assert(root.len > 0);

    return try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ root, path },
    );
}

/// Return the absolute path of the current working
/// directory.  Agents call this to orient themselves
/// before issuing shell commands or path-relative I/O.
pub fn cwdAbsolute(
    arena: Allocator,
) ![]const u8 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const abs = try std.fs.cwd().realpath(
        ".",
        &path_buf,
    );
    return try arena.dupe(u8, abs);
}

// ============================================================
// File I/O
//
// All paths are raw filesystem paths — callers must call
// resolveToFs() first when working with monorepo-relative
// paths.
// ============================================================

pub fn readFile(
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

pub fn writeFile(
    path: []const u8,
    content: []const u8,
) !void {
    std.debug.assert(path.len > 0);
    std.debug.assert(content.len > 0);
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(content);
}

pub fn copyFile(
    source: []const u8,
    dest: []const u8,
) !void {
    std.debug.assert(source.len > 0);
    std.debug.assert(dest.len > 0);
    try std.fs.cwd().copyFile(
        source,
        std.fs.cwd(),
        dest,
        .{},
    );
}

pub fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

// ============================================================
// Stdout / stderr I/O
// ============================================================

pub const stdout_file = std.fs.File{
    .handle = std.posix.STDOUT_FILENO,
};

pub const stderr_file = std.fs.File{
    .handle = std.posix.STDERR_FILENO,
};

pub fn writeStdout(bytes: []const u8) !void {
    std.debug.assert(bytes.len > 0);
    try stdout_file.writeAll(bytes);
}

// ============================================================
// Sorting
// ============================================================

/// Selection sort for small string arrays.
pub fn sortStrings(items: [][]const u8) void {
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
// Benchmark file helpers
//
// All functions take a resolved filesystem path for the
// benchmark directory, so callers control resolution.
// ============================================================

/// Check whether a filename matches any of the given
/// benchmark prefixes and ends with `.json`.
pub fn isBenchmarkFile(
    name: []const u8,
    prefixes: []const []const u8,
) bool {
    std.debug.assert(prefixes.len > 0);
    if (!std.mem.endsWith(u8, name, ".json")) {
        return false;
    }
    for (prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            return true;
        }
    }
    return false;
}

/// List all benchmark JSON filenames, sorted ascending.
pub fn listBenchmarkFiles(
    arena: Allocator,
    bench_dir_fs: []const u8,
    prefixes: []const []const u8,
) ![]const []const u8 {
    std.debug.assert(bench_dir_fs.len > 0);
    var dir = std.fs.cwd().openDir(
        bench_dir_fs,
        .{ .iterate = true },
    ) catch return &.{};
    defer dir.close();

    var names: std.ArrayList([]const u8) = .empty;
    var iter = dir.iterate();

    while (try iter.next()) |entry| {
        if (isBenchmarkFile(entry.name, prefixes)) {
            const dupe = try arena.dupe(u8, entry.name);
            try names.append(arena, dupe);
        }
    }

    sortStrings(names.items);
    return names.items;
}

/// Find any single benchmark file.  Returns null when
/// none exist.
pub fn findOneBenchmark(
    arena: Allocator,
    bench_dir_fs: []const u8,
    prefixes: []const []const u8,
) ?[]const u8 {
    std.debug.assert(bench_dir_fs.len > 0);
    var dir = std.fs.cwd().openDir(
        bench_dir_fs,
        .{ .iterate = true },
    ) catch return null;
    defer dir.close();

    var iter = dir.iterate();
    while (iter.next() catch null) |entry| {
        if (isBenchmarkFile(entry.name, prefixes)) {
            return arena.dupe(u8, entry.name) catch null;
        }
    }
    return null;
}

/// Delete all benchmark JSON files.  Returns the count
/// of files removed.
pub fn cleanBenchmarkFiles(
    bench_dir_fs: []const u8,
    prefixes: []const []const u8,
) u32 {
    std.debug.assert(bench_dir_fs.len > 0);
    var dir = std.fs.cwd().openDir(
        bench_dir_fs,
        .{ .iterate = true },
    ) catch return 0;
    defer dir.close();

    var count: u32 = 0;
    var iter = dir.iterate();
    while (iter.next() catch null) |entry| {
        if (isBenchmarkFile(entry.name, prefixes)) {
            dir.deleteFile(entry.name) catch {};
            count += 1;
        }
    }
    return count;
}

// ============================================================
// Benchmark comparison
// ============================================================

/// Build a human-readable architecture string like
/// "784->128->10" from the benchmark config layers.
pub fn buildArchString(
    arena: Allocator,
    layers: []const ArchLayer,
) []const u8 {
    if (layers.len == 0) return "(empty)";
    std.debug.assert(layers.len < 64);
    var buf: std.ArrayList(u8) = .empty;

    // Prepend the input size of the first layer so that
    // a 784->128->10 network reads "784->128->10", not
    // "128->10".
    var first_buf: [16]u8 = undefined;
    const first_s = std.fmt.bufPrint(
        &first_buf,
        "{d}",
        .{layers[0].input_size},
    ) catch return "?";
    buf.appendSlice(arena, first_s) catch return "?";

    for (layers) |layer| {
        buf.appendSlice(arena, "->") catch return "?";
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

/// Parse one benchmark JSON and format a comparison entry.
/// `bench_dir_fs` is the resolved filesystem path to the
/// benchmarks directory.
pub fn formatCompareEntry(
    arena: Allocator,
    bench_dir_fs: []const u8,
    name: []const u8,
) ![]const u8 {
    std.debug.assert(name.len > 0);
    std.debug.assert(bench_dir_fs.len > 0);

    const path = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ bench_dir_fs, name },
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

    const arch_str = buildArchString(
        arena,
        c.architecture,
    );
    const test_acc =
        r.final_test_accuracy_pct orelse 0.0;
    const val_acc =
        r.final_validation_accuracy_pct orelse 0.0;

    // Escape string fields to prevent malformed JSON
    // when values contain quotes or backslashes.
    const name_escaped = try jsonEscape(arena, name);
    const optimizer_escaped = try jsonEscape(
        arena,
        c.optimizer,
    );
    const arch_escaped = try jsonEscape(
        arena,
        arch_str,
    );

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
            name_escaped,
            test_acc,
            val_acc,
            r.throughput_images_per_sec,
            r.total_training_ms,
            optimizer_escaped,
            c.learning_rate,
            c.batch_size,
            c.num_epochs,
            c.param_count,
            arch_escaped,
        },
    );
}

/// Write a JSON array comparing all benchmark runs to
/// stdout.  `bench_dir_fs` is the resolved filesystem
/// path to the benchmarks directory.
pub fn toolBenchCompare(
    arena: Allocator,
    bench_dir_fs: []const u8,
    prefixes: []const []const u8,
) !void {
    std.debug.assert(bench_dir_fs.len > 0);
    const names = try listBenchmarkFiles(
        arena,
        bench_dir_fs,
        prefixes,
    );
    var buf: std.ArrayList(u8) = .empty;

    try buf.appendSlice(arena, "[\n");
    for (names, 0..) |name, i| {
        if (i > 0) try buf.appendSlice(arena, ",\n");
        const entry = formatCompareEntry(
            arena,
            bench_dir_fs,
            name,
        );
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
