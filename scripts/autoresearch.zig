//! nnzap autoresearch — AI agent toolbox.
//!
//! A CLI binary that outputs JSON on stdout, designed for
//! AI coding agents to invoke via terminal.  Each subcommand
//! is a "tool" the agent calls.
//!
//! Tools:
//!   help               Show available tools
//!   benchmark-list     List benchmark JSON filenames
//!   benchmark-latest   Output latest benchmark result
//!   benchmark-compare  Compare all benchmark runs
//!   config-show        Current hyperparameters from main.zig
//!   config-set K=V ... Modify hyperparameters in main.zig
//!   config-backup      Backup main.zig before editing
//!   config-restore     Restore main.zig from backup
//!   train              Build + run training, output result
//!   clean              Delete old benchmark files
//!
//! Output contract:
//!   stdout  → JSON (parsed by agent)
//!   stderr  → human-readable diagnostics
//!   exit 0  → success
//!   exit 1  → failure (stdout still valid JSON with error)
//!
//! Build:
//!   zig build
//!
//! Run:
//!   ./zig-out/bin/autoresearch <tool> [args...]

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_FILE_SIZE: usize = 2 * 1024 * 1024;
const MAX_BENCHMARKS: usize = 512;
const MAX_LAYERS: usize = 16;
const MAX_ARGS: usize = 64;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

const MAIN_PATH = "src/main.zig";
const BACKUP_PATH = "src/main.zig.bak";
const BENCH_DIR = "benchmarks";

// ============================================================
// JSON parse types — mirrors benchmark.zig output
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
// Config types — parsed from main.zig source
// ============================================================

const Layer = struct {
    in: u32,
    out: u32,
    act: []const u8,
};

const MainConfig = struct {
    layers: [MAX_LAYERS]Layer = undefined,
    layer_count: usize = 0,
    max_batch: []const u8 = "64",
    learning_rate: []const u8 = "0.1",
    num_epochs: []const u8 = "10",
    seed: []const u8 = "42",
    optimizer: []const u8 = "sgd",
    beta1: []const u8 = "0.9",
    beta2: []const u8 = "0.999",
    epsilon: []const u8 = "1e-8",
};

const ConfigChanges = struct {
    lr: ?[]const u8 = null,
    batch: ?[]const u8 = null,
    epochs: ?[]const u8 = null,
    seed: ?[]const u8 = null,
    optimizer: ?[]const u8 = null,
    arch: ?[]const u8 = null,
    beta1: ?[]const u8 = null,
    beta2: ?[]const u8 = null,
    epsilon: ?[]const u8 = null,
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
    if (eql(cmd, "help")) return toolHelp();
    if (eql(cmd, "benchmark-list")) return toolBenchList(arena);
    if (eql(cmd, "benchmark-latest")) return toolBenchLatest(arena);
    if (eql(cmd, "benchmark-compare")) return toolBenchCompare(arena);
    if (eql(cmd, "config-show")) return toolConfigShow(arena);
    if (eql(cmd, "config-set")) return toolConfigSet(arena, args);
    if (eql(cmd, "config-backup")) return toolConfigBackup();
    if (eql(cmd, "config-restore")) return toolConfigRestore();
    if (eql(cmd, "train")) return toolTrain(arena);
    if (eql(cmd, "clean")) return toolClean();

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
        \\      "name": "benchmark-list",
        \\      "description": "List all benchmark JSON files",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "benchmark-latest",
        \\      "description": "Output the latest benchmark",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "benchmark-compare",
        \\      "description": "Compare all benchmark runs",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "config-show",
        \\      "description": "Show current hyperparameters",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "config-set",
        \\      "description": "Modify hyperparameters",
        \\      "args": [
        \\        "lr=<float>",
        \\        "batch=<int>",
        \\        "epochs=<int>",
        \\        "seed=<int>",
        \\        "optimizer=sgd|adam",
        \\        "arch=in:out:act,in:out:act,...",
        \\        "beta1=<float>",
        \\        "beta2=<float>",
        \\        "epsilon=<float>"
        \\      ]
        \\    },
        \\    {
        \\      "name": "config-backup",
        \\      "description": "Backup src/main.zig",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "config-restore",
        \\      "description": "Restore src/main.zig from backup",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "train",
        \\      "description": "Build and run MNIST training",
        \\      "args": []
        \\    },
        \\    {
        \\      "name": "clean",
        \\      "description": "Delete old benchmark files",
        \\      "args": []
        \\    }
        \\  ]
        \\}
        \\
    );
}

// ============================================================
// Tool: benchmark-list
// ============================================================

fn toolBenchList(arena: Allocator) !void {
    const names = try listBenchmarkFiles(arena);
    var buf: std.ArrayList(u8) = .empty;

    try buf.appendSlice(arena, "{\"files\": [");
    for (names, 0..) |name, i| {
        if (i > 0) try buf.appendSlice(arena, ", ");
        try buf.appendSlice(arena, "\"");
        try buf.appendSlice(arena, name);
        try buf.appendSlice(arena, "\"");
    }
    try buf.appendSlice(arena, "]}\n");
    try writeStdout(buf.items);
}

// ============================================================
// Tool: benchmark-latest
// ============================================================

fn toolBenchLatest(arena: Allocator) !void {
    const names = try listBenchmarkFiles(arena);
    if (names.len == 0) return error.NoBenchmarkFiles;

    const latest = names[names.len - 1];
    const path = try std.fmt.allocPrint(
        arena,
        "{s}/{s}",
        .{ BENCH_DIR, latest },
    );
    const content = try readFile(arena, path);
    try writeStdout(content);
}

// ============================================================
// Tool: benchmark-compare
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
    try writeStdout(buf.items);
}

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

// ============================================================
// Tool: config-show
// ============================================================

fn toolConfigShow(arena: Allocator) !void {
    const source = try readFile(arena, MAIN_PATH);
    const config = parseMainConfig(source);
    const json = try formatConfigJson(arena, &config);
    try writeStdout(json);
}

fn parseMainConfig(source: []const u8) MainConfig {
    std.debug.assert(source.len > 0);

    var config: MainConfig = .{};
    var in_layout = false;

    var iter = std.mem.splitScalar(u8, source, '\n');
    while (iter.next()) |raw_line| {
        const line = trimCR(raw_line);
        const trimmed = std.mem.trimLeft(u8, line, " ");

        if (in_layout) {
            if (std.mem.startsWith(u8, trimmed, "});")) {
                in_layout = false;
            } else {
                parseLayoutLine(&config, trimmed);
            }
            continue;
        }

        parseScalarLine(&config, trimmed);

        if (std.mem.startsWith(
            u8,
            trimmed,
            "const MnistLayout",
        )) {
            in_layout = true;
        }
    }
    return config;
}

fn parseScalarLine(
    config: *MainConfig,
    trimmed: []const u8,
) void {
    if (extractAfterEq(trimmed, "const max_batch:")) |v| {
        config.max_batch = v;
    }
    if (extractAfterEq(trimmed, "const learning_rate:")) |v| {
        config.learning_rate = v;
    }
    if (extractAfterEq(trimmed, "const num_epochs:")) |v| {
        config.num_epochs = v;
    }
    if (extractAfterEq(trimmed, "const seed:")) |v| {
        config.seed = v;
    }
    if (std.mem.startsWith(u8, trimmed, "net.updateAdam(")) {
        config.optimizer = "adam";
        parseAdamArgs(config, trimmed);
    } else if (std.mem.startsWith(
        u8,
        trimmed,
        "net.update(",
    )) {
        config.optimizer = "sgd";
    }
}

fn parseLayoutLine(
    config: *MainConfig,
    trimmed: []const u8,
) void {
    if (config.layer_count >= MAX_LAYERS) return;
    if (!std.mem.startsWith(u8, trimmed, ".{")) return;

    const in_val = extractField(trimmed, ".in = ");
    const out_val = extractField(trimmed, ".out = ");
    const act_val = extractDotField(trimmed, ".act = .");

    if (in_val != null and out_val != null) {
        config.layers[config.layer_count] = .{
            .in = std.fmt.parseInt(
                u32,
                in_val.?,
                10,
            ) catch 0,
            .out = std.fmt.parseInt(
                u32,
                out_val.?,
                10,
            ) catch 0,
            .act = act_val orelse "relu",
        };
        config.layer_count += 1;
    }
}

fn parseAdamArgs(
    config: *MainConfig,
    trimmed: []const u8,
) void {
    // net.updateAdam(device, enc, learning_rate, 0.9, 0.999, 1e-8);
    // Extract the last three args: beta1, beta2, epsilon.
    const open = std.mem.indexOf(u8, trimmed, "(");
    const close = std.mem.lastIndexOf(u8, trimmed, ")");
    if (open == null or close == null) return;

    const inner = trimmed[open.? + 1 .. close.?];
    // Split by comma, take last three.
    var parts: [8][]const u8 = undefined;
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, inner, ',');
    while (it.next()) |part| {
        if (count < 8) {
            parts[count] = std.mem.trim(u8, part, " ");
            count += 1;
        }
    }
    // Args: device, enc, lr, beta1, beta2, epsilon
    if (count >= 6) {
        config.beta1 = parts[3];
        config.beta2 = parts[4];
        config.epsilon = parts[5];
    }
}

fn formatConfigJson(
    arena: Allocator,
    config: *const MainConfig,
) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\n  \"architecture\": [\n");

    for (config.layers[0..config.layer_count], 0..) |layer, i| {
        if (i > 0) try buf.appendSlice(arena, ",\n");
        const entry = try std.fmt.allocPrint(
            arena,
            "    {{\"in\": {d}, \"out\": {d}, " ++
                "\"act\": \"{s}\"}}",
            .{ layer.in, layer.out, layer.act },
        );
        try buf.appendSlice(arena, entry);
    }

    try buf.appendSlice(arena, "\n  ],\n");

    const rest = try std.fmt.allocPrint(
        arena,
        "  \"max_batch\": {s},\n" ++
            "  \"learning_rate\": {s},\n" ++
            "  \"num_epochs\": {s},\n" ++
            "  \"seed\": {s},\n" ++
            "  \"optimizer\": \"{s}\"",
        .{
            config.max_batch,
            config.learning_rate,
            config.num_epochs,
            config.seed,
            config.optimizer,
        },
    );
    try buf.appendSlice(arena, rest);

    if (eql(config.optimizer, "adam")) {
        const adam = try std.fmt.allocPrint(
            arena,
            ",\n  \"beta1\": {s},\n" ++
                "  \"beta2\": {s},\n" ++
                "  \"epsilon\": {s}",
            .{
                config.beta1,
                config.beta2,
                config.epsilon,
            },
        );
        try buf.appendSlice(arena, adam);
    }

    try buf.appendSlice(arena, "\n}\n");
    return buf.items;
}

// ============================================================
// Tool: config-set
// ============================================================

fn toolConfigSet(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len == 0) return error.NoConfigArgs;

    const changes = parseConfigArgs(args);
    const source = try readFile(arena, MAIN_PATH);
    const modified = try applyChanges(
        arena,
        source,
        &changes,
    );
    try writeFile(MAIN_PATH, modified);

    var count: u32 = 0;
    if (changes.lr != null) count += 1;
    if (changes.batch != null) count += 1;
    if (changes.epochs != null) count += 1;
    if (changes.seed != null) count += 1;
    if (changes.optimizer != null) count += 1;
    if (changes.arch != null) count += 1;

    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"ok\", " ++
            "\"changes_applied\": {d}}}\n",
        .{count},
    );
    try writeStdout(json);
}

fn parseConfigArgs(
    args: []const []const u8,
) ConfigChanges {
    var changes: ConfigChanges = .{};
    for (args) |arg| {
        const kv = splitOnce(arg, '=') orelse continue;
        const key = kv[0];
        const val = kv[1];
        if (eql(key, "lr")) changes.lr = val;
        if (eql(key, "batch")) changes.batch = val;
        if (eql(key, "epochs")) changes.epochs = val;
        if (eql(key, "seed")) changes.seed = val;
        if (eql(key, "optimizer")) changes.optimizer = val;
        if (eql(key, "arch")) changes.arch = val;
        if (eql(key, "beta1")) changes.beta1 = val;
        if (eql(key, "beta2")) changes.beta2 = val;
        if (eql(key, "epsilon")) changes.epsilon = val;
    }
    return changes;
}

fn applyChanges(
    arena: Allocator,
    source: []const u8,
    changes: *const ConfigChanges,
) ![]const u8 {
    std.debug.assert(source.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    var skip_layout = false;
    var first_net_param = true;
    var iter = std.mem.splitScalar(u8, source, '\n');
    // Track whether this is the last segment (no trailing \n).
    var first_line = true;

    while (iter.next()) |raw_line| {
        if (!first_line) try buf.append(arena, '\n');
        first_line = false;

        const line = trimCR(raw_line);
        const trimmed = std.mem.trimLeft(u8, line, " ");
        const indent = leadingSpaces(line);

        // Skip lines inside old architecture block.
        if (skip_layout) {
            if (std.mem.startsWith(u8, trimmed, "});")) {
                skip_layout = false;
            }
            continue;
        }

        // Architecture block replacement.
        if (changes.arch != null and
            std.mem.startsWith(
                u8,
                trimmed,
                "const MnistLayout",
            ))
        {
            skip_layout = true;
            const block = try buildLayoutBlock(
                arena,
                changes.arch.?,
            );
            try buf.appendSlice(arena, block);
            continue;
        }

        // Scalar replacements.
        if (tryReplaceLine(
            arena,
            &buf,
            trimmed,
            indent,
            changes,
            &first_net_param,
        )) continue;

        // No replacement — keep original line.
        try buf.appendSlice(arena, line);
    }

    return buf.items;
}

/// Check if this line should be replaced.  If so, write
/// the replacement to `buf` and return true.  Otherwise
/// return false and the caller keeps the original.
fn tryReplaceLine(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    trimmed: []const u8,
    indent: usize,
    changes: *const ConfigChanges,
    first_net_param: *bool,
) bool {
    if (changes.lr) |lr| {
        if (startsWith(trimmed, "const learning_rate:")) {
            writeIndented(
                arena,
                buf,
                indent,
                "const learning_rate: f32 = {s};",
                .{lr},
            );
            return true;
        }
    }
    if (changes.batch) |batch| {
        if (startsWith(trimmed, "const max_batch:")) {
            writeIndented(
                arena,
                buf,
                indent,
                "const max_batch: u32 = {s};",
                .{batch},
            );
            return true;
        }
    }
    if (changes.epochs) |epochs| {
        if (startsWith(trimmed, "const num_epochs:")) {
            writeIndented(
                arena,
                buf,
                indent,
                "const num_epochs: u32 = {s};",
                .{epochs},
            );
            return true;
        }
    }
    if (changes.seed) |seed_val| {
        if (startsWith(trimmed, "const seed:")) {
            writeIndented(
                arena,
                buf,
                indent,
                "const seed: u64 = {s};",
                .{seed_val},
            );
            return true;
        }
    }
    return tryReplaceOptimizer(
        arena,
        buf,
        trimmed,
        indent,
        changes,
        first_net_param,
    );
}

fn tryReplaceOptimizer(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    trimmed: []const u8,
    indent: usize,
    changes: *const ConfigChanges,
    first_net_param: *bool,
) bool {
    const opt = changes.optimizer orelse return false;

    // Update call: net.update(...) or net.updateAdam(...)
    if (startsWith(trimmed, "net.update")) {
        if (eql(opt, "adam")) {
            const b1 = changes.beta1 orelse "0.9";
            const b2 = changes.beta2 orelse "0.999";
            const eps = changes.epsilon orelse "1e-8";
            writeIndented(arena, buf, indent, "net.updateAdam(" ++
                "device, enc, " ++
                "learning_rate, " ++
                "{s}, {s}, {s});", .{ b1, b2, eps });
        } else {
            writeIndented(arena, buf, indent, "net.update(" ++
                "device, enc, learning_rate);", .{});
        }
        return true;
    }

    // Benchmark config .optimizer field.
    if (startsWith(trimmed, ".optimizer = .")) {
        writeIndented(
            arena,
            buf,
            indent,
            ".optimizer = .{s},",
            .{opt},
        );
        return true;
    }

    // trainEpoch net parameter type (first occurrence).
    if (first_net_param.* and
        std.mem.indexOf(u8, trimmed, "MnistNet,") != null and
        startsWith(trimmed, "net:"))
    {
        first_net_param.* = false;
        if (eql(opt, "adam")) {
            writeIndented(
                arena,
                buf,
                indent,
                "net: *MnistNet,",
                .{},
            );
        } else {
            writeIndented(
                arena,
                buf,
                indent,
                "net: *const MnistNet,",
                .{},
            );
        }
        return true;
    }

    return false;
}

fn buildLayoutBlock(
    arena: Allocator,
    arch_str: []const u8,
) ![]const u8 {
    std.debug.assert(arch_str.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(
        arena,
        "const MnistLayout = nnzap.NetworkLayout(&.{",
    );

    var layer_iter = std.mem.splitScalar(
        u8,
        arch_str,
        ',',
    );
    var first = true;
    while (layer_iter.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (trimmed.len == 0) continue;

        const layer = try parseLayerSpec(trimmed);
        if (!first) {
            try buf.appendSlice(arena, ",");
        }
        first = false;

        const line = try std.fmt.allocPrint(
            arena,
            "\n    .{{ .in = {s}, .out = {s}, " ++
                ".act = .{s} }}",
            .{ layer[0], layer[1], layer[2] },
        );
        try buf.appendSlice(arena, line);
    }

    try buf.appendSlice(arena, ",\n})");
    return buf.items;
}

/// Parse "784:128:relu" into [in, out, act] strings.
fn parseLayerSpec(
    spec: []const u8,
) ![3][]const u8 {
    var parts: [3][]const u8 = undefined;
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, spec, ':');
    while (it.next()) |p| {
        if (count >= 3) return error.InvalidLayerSpec;
        parts[count] = std.mem.trim(u8, p, " ");
        count += 1;
    }
    if (count != 3) return error.InvalidLayerSpec;
    return parts;
}

// ============================================================
// Tool: config-backup / config-restore
// ============================================================

fn toolConfigBackup() !void {
    try copyFile(MAIN_PATH, BACKUP_PATH);
    try writeStdout(
        "{\"status\": \"ok\", " ++
            "\"backup\": \"src/main.zig.bak\"}\n",
    );
}

fn toolConfigRestore() !void {
    if (!fileExists(BACKUP_PATH)) {
        return error.NoBackupFile;
    }
    try copyFile(BACKUP_PATH, MAIN_PATH);
    try writeStdout(
        "{\"status\": \"ok\", " ++
            "\"restored\": \"src/main.zig\"}\n",
    );
}

// ============================================================
// Tool: train
// ============================================================

fn toolTrain(arena: Allocator) !void {
    // Clean old benchmarks so we find exactly the new one.
    _ = cleanBenchmarkFiles();

    std.debug.print(
        "autoresearch: running zig build run...\n",
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
                "\"error\": \"spawn_failed: {s}\"}}\n",
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
        try writeTrainError(arena, result.stderr);
        return;
    }

    // Find and output the benchmark JSON.
    const bench_name = findOneBenchmark(arena);
    if (bench_name) |name| {
        const path = try std.fmt.allocPrint(
            arena,
            "{s}/{s}",
            .{ BENCH_DIR, name },
        );
        const content = try readFile(arena, path);
        try writeStdout(content);
    } else {
        try writeStdout(
            "{\"status\": \"error\", " ++
                "\"error\": \"no benchmark produced\"}\n",
        );
    }
}

fn writeTrainError(
    arena: Allocator,
    stderr_output: []const u8,
) !void {
    // Truncate stderr for JSON embedding.
    const max_err = 2000;
    const truncated = if (stderr_output.len > max_err)
        stderr_output[stderr_output.len - max_err ..]
    else
        stderr_output;

    // Escape for JSON string (minimal: replace " and \).
    const escaped = try jsonEscape(arena, truncated);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"status\": \"error\", " ++
            "\"error\": \"build_failed\", " ++
            "\"output\": \"{s}\"}}\n",
        .{escaped},
    );
    try writeStdout(json);
}

// ============================================================
// Tool: clean
// ============================================================

fn toolClean() !void {
    const count = cleanBenchmarkFiles();
    var buf: [128]u8 = undefined;
    const json = std.fmt.bufPrint(
        &buf,
        "{{\"status\": \"ok\", " ++
            "\"files_deleted\": {d}}}\n",
        .{count},
    ) catch unreachable;
    try writeStdout(json);
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
    return count;
}

fn isBenchmarkFile(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "mnist_") and
        std.mem.endsWith(u8, name, ".json");
}

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

fn buildArchString(
    arena: Allocator,
    layers: []const ArchLayer,
) []const u8 {
    if (layers.len == 0) return "(empty)";
    var buf: std.ArrayList(u8) = .empty;
    for (layers) |layer| {
        if (buf.items.len > 0) {
            buf.appendSlice(arena, "->") catch return "?";
        }
        var num_buf: [16]u8 = undefined;
        const s = std.fmt.bufPrint(
            &num_buf,
            "{d}",
            .{layer.output_size},
        ) catch return "?";
        buf.appendSlice(arena, s) catch return "?";
    }
    return buf.items;
}

// ============================================================
// File I/O
// ============================================================

fn readFile(
    arena: Allocator,
    path: []const u8,
) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try file.readToEndAlloc(
        arena,
        MAX_FILE_SIZE,
    );
}

fn writeFile(
    path: []const u8,
    content: []const u8,
) !void {
    std.debug.assert(path.len > 0);
    std.debug.assert(content.len > 0);
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(content);
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

fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

const stdout_file = std.fs.File{
    .handle = std.posix.STDOUT_FILENO,
};
const stderr_file = std.fs.File{
    .handle = std.posix.STDERR_FILENO,
};

fn writeStdout(bytes: []const u8) !void {
    try stdout_file.writeAll(bytes);
}

// ============================================================
// JSON output helpers
// ============================================================

fn writeJsonError(err: anyerror) !void {
    var buf: [256]u8 = undefined;
    const json = std.fmt.bufPrint(
        &buf,
        "{{\"error\": \"{s}\"}}\n",
        .{@errorName(err)},
    ) catch
        "{\"error\": \"unknown\"}\n";
    try writeStdout(json);
}

fn jsonEscape(
    arena: Allocator,
    input: []const u8,
) ![]const u8 {
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

fn trimCR(line: []const u8) []const u8 {
    if (line.len > 0 and line[line.len - 1] == '\r') {
        return line[0 .. line.len - 1];
    }
    return line;
}

fn leadingSpaces(line: []const u8) usize {
    for (line, 0..) |c, i| {
        if (c != ' ') return i;
    }
    return line.len;
}

/// Split "key=value" into [key, value].  Returns null if
/// no '=' found.
fn splitOnce(
    s: []const u8,
    sep: u8,
) ?[2][]const u8 {
    const idx = std.mem.indexOfScalar(u8, s, sep) orelse
        return null;
    return .{ s[0..idx], s[idx + 1 ..] };
}

/// Extract the value from a `const name: type = VALUE;`
/// line.  Returns the trimmed VALUE string between '=' and
/// ';', or null if the line does not start with `prefix`.
fn extractAfterEq(
    trimmed: []const u8,
    prefix: []const u8,
) ?[]const u8 {
    if (!startsWith(trimmed, prefix)) return null;
    const eq = std.mem.indexOfScalar(u8, trimmed, '=');
    const semi = std.mem.indexOfScalar(u8, trimmed, ';');
    if (eq == null or semi == null) return null;
    if (semi.? <= eq.?) return null;
    return std.mem.trim(
        u8,
        trimmed[eq.? + 1 .. semi.?],
        " ",
    );
}

/// Extract value from `.field = VALUE` patterns inside
/// struct literals.  VALUE ends at ',' or '}' or end of
/// line.
fn extractField(
    trimmed: []const u8,
    needle: []const u8,
) ?[]const u8 {
    const idx = std.mem.indexOf(u8, trimmed, needle);
    if (idx == null) return null;
    const start = idx.? + needle.len;
    const rest = trimmed[start..];
    // End at comma, space, or closing brace.
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
fn extractDotField(
    trimmed: []const u8,
    needle: []const u8,
) ?[]const u8 {
    const idx = std.mem.indexOf(u8, trimmed, needle);
    if (idx == null) return null;
    const start = idx.? + needle.len;
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

/// Write an indented, formatted line into a buffer.
fn writeIndented(
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
