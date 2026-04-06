//! MNIST researcher — configures the generic
//! toolbox for MNIST hyperparameter optimisation.
//!
//! Custom tools (config-show, config-set, config-backup,
//! config-restore) handle MNIST-specific source code
//! parsing and modification.  All other tools are
//! provided by the generic toolbox.
//!
//! Usage:
//!   zig build
//!   ./zig-out/bin/mnist_researcher <tool> [args...]

const std = @import("std");
const Allocator = std.mem.Allocator;
const tools = @import("tools.zig");
const toolbox = @import("toolbox.zig");

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_LAYERS: u32 = 16;
const MAX_ARGS: u32 = 64;

const MAIN_PATH = "nn/examples/mnist.zig";
const BACKUP_PATH = "nn/examples/mnist.zig.bak";

// ============================================================
// Config types — MNIST-specific
// ============================================================

const Layer = struct {
    in: u32 = 0,
    out: u32 = 0,
    act: []const u8 = "relu",
};

const MainConfig = struct {
    layers: [MAX_LAYERS]Layer = undefined,
    layer_count: u32 = 0,
    max_batch: []const u8 = "0",
    learning_rate: []const u8 = "0",
    num_epochs: []const u8 = "0",
    seed: []const u8 = "0",
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
// Toolbox configuration
// ============================================================

const config = toolbox.ToolboxConfig{
    .name = "mnist",
    .project_root = "../nn",
    .write_scope = &.{
        "nn/examples/mnist.zig",
    },
    .read_scope = &.{
        "nn/src/",
        "nn/examples/",
        "benchmarks/",
        ".mnist_history/",
    },
    .read_files = &.{
        "nn/build.zig",
        "nn/build.zig.zon",
    },
    .engine_files = &.{
        "nn/examples/mnist.zig",
    },
    .check_command = &.{ "zig", "build" },
    .test_command = &.{ "zig", "build", "test" },
    .bench_command = &.{ "zig", "build", "run" },
    .bench_output_file = true,
    .bench_dir = "benchmarks",
    .bench_prefixes = &.{ "mnist_", "inference_" },
    .history_dir = ".mnist_history",
    .snapshot_dir = ".mnist_snapshots",
    .custom_dispatch = &mnistDispatch,
};

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    toolbox.run(&config);
}

// ============================================================
// Custom dispatch — MNIST-specific tools
// ============================================================

fn mnistDispatch(
    arena: Allocator,
    cmd: []const u8,
    args: []const []const u8,
) !bool {
    if (tools.eql(cmd, "config-show")) {
        try toolConfigShow(arena);
        return true;
    }
    if (tools.eql(cmd, "config-set")) {
        try toolConfigSet(arena, args);
        return true;
    }
    if (tools.eql(cmd, "config-backup")) {
        try toolConfigBackup(arena);
        return true;
    }
    if (tools.eql(cmd, "config-restore")) {
        try toolConfigRestore(arena);
        return true;
    }
    return false;
}

// ============================================================
// config-show — parse current hyperparameters from source
// ============================================================

fn toolConfigShow(arena: Allocator) !void {
    const main_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        MAIN_PATH,
    );
    const source = try tools.readFile(arena, main_fs);
    const cfg = parseMainConfig(source);
    const json = try formatConfigJson(arena, &cfg);
    try tools.writeStdout(json);
}

fn parseMainConfig(source: []const u8) MainConfig {
    std.debug.assert(source.len > 0);

    var cfg: MainConfig = .{};
    var in_layout = false;

    var iter = std.mem.splitScalar(u8, source, '\n');
    while (iter.next()) |raw_line| {
        const line = tools.trimCR(raw_line);
        const trimmed = std.mem.trimLeft(
            u8,
            line,
            " ",
        );

        if (in_layout) {
            if (std.mem.startsWith(
                u8,
                trimmed,
                "});",
            )) {
                in_layout = false;
            } else {
                parseLayoutLine(&cfg, trimmed);
            }
            continue;
        }

        parseScalarLine(&cfg, trimmed);

        if (std.mem.startsWith(
            u8,
            trimmed,
            "const MnistLayout",
        )) {
            in_layout = true;
        }
    }
    return cfg;
}

fn parseScalarLine(
    cfg: *MainConfig,
    trimmed: []const u8,
) void {
    if (tools.extractAfterEq(
        trimmed,
        "const max_batch:",
    )) |v| {
        cfg.max_batch = v;
    }
    if (tools.extractAfterEq(
        trimmed,
        "const learning_rate:",
    )) |v| {
        cfg.learning_rate = v;
    }
    if (tools.extractAfterEq(
        trimmed,
        "const num_epochs:",
    )) |v| {
        cfg.num_epochs = v;
    }
    if (tools.extractAfterEq(
        trimmed,
        "const seed:",
    )) |v| {
        cfg.seed = v;
    }
    if (std.mem.startsWith(
        u8,
        trimmed,
        "net.updateAdam(",
    )) {
        cfg.optimizer = "adam";
        parseAdamArgs(cfg, trimmed);
    } else if (std.mem.startsWith(
        u8,
        trimmed,
        "net.update(",
    )) {
        cfg.optimizer = "sgd";
    }
}

fn parseLayoutLine(
    cfg: *MainConfig,
    trimmed: []const u8,
) void {
    if (cfg.layer_count >= MAX_LAYERS) return;
    if (!std.mem.startsWith(u8, trimmed, ".{")) return;

    const in_val = tools.extractField(
        trimmed,
        ".in = ",
    );
    const out_val = tools.extractField(
        trimmed,
        ".out = ",
    );
    const act_val = tools.extractDotField(
        trimmed,
        ".act = .",
    );

    if (in_val != null and out_val != null) {
        cfg.layers[cfg.layer_count] = .{
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
        cfg.layer_count += 1;
    }
}

fn parseAdamArgs(
    cfg: *MainConfig,
    trimmed: []const u8,
) void {
    // net.updateAdam(device, enc, lr, 0.9, 0.999, 1e-8);
    // Extract the last three args: beta1, beta2, epsilon.
    const open = std.mem.indexOf(u8, trimmed, "(");
    const close = std.mem.lastIndexOf(
        u8,
        trimmed,
        ")",
    );
    if (open == null or close == null) return;

    const inner = trimmed[open.? + 1 .. close.?];
    // Split by comma, take last three.
    var parts: [8][]const u8 = undefined;
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, inner, ',');
    while (it.next()) |part| {
        if (count < 8) {
            parts[count] = std.mem.trim(
                u8,
                part,
                " ",
            );
            count += 1;
        }
    }
    // Args: device, enc, lr, beta1, beta2, epsilon.
    if (count >= 6) {
        cfg.beta1 = parts[3];
        cfg.beta2 = parts[4];
        cfg.epsilon = parts[5];
    }
}

fn formatConfigJson(
    arena: Allocator,
    cfg: *const MainConfig,
) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(
        arena,
        "{\n  \"architecture\": [\n",
    );

    const layer_slice =
        cfg.layers[0..cfg.layer_count];
    for (layer_slice, 0..) |layer, i| {
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
            cfg.max_batch,
            cfg.learning_rate,
            cfg.num_epochs,
            cfg.seed,
            cfg.optimizer,
        },
    );
    try buf.appendSlice(arena, rest);

    if (tools.eql(cfg.optimizer, "adam")) {
        const adam = try std.fmt.allocPrint(
            arena,
            ",\n  \"beta1\": {s},\n" ++
                "  \"beta2\": {s},\n" ++
                "  \"epsilon\": {s}",
            .{
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
            },
        );
        try buf.appendSlice(arena, adam);
    }

    try buf.appendSlice(arena, "\n}\n");
    return buf.items;
}

// ============================================================
// config-set — modify hyperparameters in source file
// ============================================================

fn resolveConfigArgs(
    arena: Allocator,
    args: []const []const u8,
) []const []const u8 {
    std.debug.assert(args.len > 0);

    // Check for -f flag.
    if (args.len >= 2 and tools.eql(args[0], "-f")) {
        const path = args[1];
        const content = tools.readFile(
            arena,
            path,
        ) catch return args;

        // Clean up temp file.
        std.fs.cwd().deleteFile(path) catch {};

        const parsed = std.json.parseFromSliceLeaky(
            std.json.Value,
            arena,
            content,
            .{},
        ) catch return args;

        const obj = switch (parsed) {
            .object => |o| o,
            else => return args,
        };

        const arr_val = obj.get("settings") orelse
            return args;
        const arr = switch (arr_val) {
            .array => |a| a.items,
            else => return args,
        };
        if (arr.len == 0) return args;

        const result = arena.alloc(
            []const u8,
            arr.len,
        ) catch return args;
        var count: usize = 0;
        for (arr) |item| {
            switch (item) {
                .string => |s| {
                    result[count] = s;
                    count += 1;
                },
                else => {},
            }
        }
        if (count == 0) return args;
        return result[0..count];
    }

    // No -f flag — use positional args as-is.
    return args;
}

fn toolConfigSet(
    arena: Allocator,
    args: []const []const u8,
) !void {
    if (args.len == 0) return error.NoConfigArgs;

    const resolved = resolveConfigArgs(arena, args);
    const changes = parseConfigArgs(resolved);
    const main_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        MAIN_PATH,
    );
    const source = try tools.readFile(arena, main_fs);
    const modified = try applyChanges(
        arena,
        source,
        &changes,
    );
    try tools.writeFile(main_fs, modified);

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
    try tools.writeStdout(json);
}

fn parseConfigArgs(
    args: []const []const u8,
) ConfigChanges {
    var changes: ConfigChanges = .{};
    for (args) |arg| {
        const kv = tools.splitOnce(arg, '=') orelse
            continue;
        const key = kv[0];
        const val = kv[1];
        if (tools.eql(key, "lr")) changes.lr = val;
        if (tools.eql(key, "batch")) {
            changes.batch = val;
        }
        if (tools.eql(key, "epochs")) {
            changes.epochs = val;
        }
        if (tools.eql(key, "seed")) changes.seed = val;
        if (tools.eql(key, "optimizer")) {
            changes.optimizer = val;
        }
        if (tools.eql(key, "arch")) changes.arch = val;
        if (tools.eql(key, "beta1")) {
            changes.beta1 = val;
        }
        if (tools.eql(key, "beta2")) {
            changes.beta2 = val;
        }
        if (tools.eql(key, "epsilon")) {
            changes.epsilon = val;
        }
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
    // Track whether this is the last segment.
    var first_line = true;

    while (iter.next()) |raw_line| {
        if (!first_line) try buf.append(arena, '\n');
        first_line = false;

        const line = tools.trimCR(raw_line);
        const trimmed = std.mem.trimLeft(
            u8,
            line,
            " ",
        );
        const indent = tools.leadingSpaces(line);

        // Skip lines inside old architecture block.
        if (skip_layout) {
            if (std.mem.startsWith(
                u8,
                trimmed,
                "});",
            )) {
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

/// A scalar config line replacement: when the change
/// value is present and the trimmed line starts with
/// the prefix, write the declaration with the new value.
const ScalarRule = struct {
    value: ?[]const u8,
    prefix: []const u8,
    declaration: []const u8,
};

/// Build the table of scalar replacement rules from
/// the current config changes.
fn scalarRules(
    changes: *const ConfigChanges,
) [4]ScalarRule {
    return .{
        .{
            .value = changes.lr,
            .prefix = "const learning_rate:",
            .declaration = "const learning_rate: f32 = ",
        },
        .{
            .value = changes.batch,
            .prefix = "const max_batch:",
            .declaration = "const max_batch: u32 = ",
        },
        .{
            .value = changes.epochs,
            .prefix = "const num_epochs:",
            .declaration = "const num_epochs: u32 = ",
        },
        .{
            .value = changes.seed,
            .prefix = "const seed:",
            .declaration = "const seed: u64 = ",
        },
    };
}

fn writeScalarReplacement(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    indent: usize,
    declaration: []const u8,
    value: []const u8,
) void {
    buf.appendNTimes(arena, ' ', indent) catch return;
    buf.appendSlice(arena, declaration) catch return;
    buf.appendSlice(arena, value) catch return;
    buf.appendSlice(arena, ";") catch return;
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
    for (scalarRules(changes)) |rule| {
        if (rule.value) |val| {
            if (tools.startsWith(trimmed, rule.prefix)) {
                writeScalarReplacement(
                    arena,
                    buf,
                    indent,
                    rule.declaration,
                    val,
                );
                return true;
            }
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

/// Replace `net.update(...)` or `net.updateAdam(...)`
/// with the correct call for the chosen optimizer.
fn tryReplaceUpdateCall(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    trimmed: []const u8,
    indent: usize,
    optimizer: []const u8,
    changes: *const ConfigChanges,
) bool {
    if (!tools.startsWith(trimmed, "net.update")) {
        return false;
    }
    if (tools.eql(optimizer, "adam")) {
        const b1 = changes.beta1 orelse "0.9";
        const b2 = changes.beta2 orelse "0.999";
        const eps = changes.epsilon orelse "1e-8";
        tools.writeIndented(
            arena,
            buf,
            indent,
            "net.updateAdam(" ++
                "device, enc, " ++
                "learning_rate, " ++
                "{s}, {s}, {s});",
            .{ b1, b2, eps },
        );
    } else {
        tools.writeIndented(
            arena,
            buf,
            indent,
            "net.update(" ++
                "device, enc, learning_rate);",
            .{},
        );
    }
    return true;
}

/// Replace the net parameter type in trainEpoch
/// based on the optimizer (first occurrence only).
fn tryReplaceNetParam(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    trimmed: []const u8,
    indent: usize,
    optimizer: []const u8,
    first_net_param: *bool,
) bool {
    if (!first_net_param.*) return false;
    if (std.mem.indexOf(
        u8,
        trimmed,
        "MnistNet,",
    ) == null) return false;
    if (!tools.startsWith(trimmed, "net:")) {
        return false;
    }

    first_net_param.* = false;
    if (tools.eql(optimizer, "adam")) {
        tools.writeIndented(
            arena,
            buf,
            indent,
            "net: *MnistNet,",
            .{},
        );
    } else {
        tools.writeIndented(
            arena,
            buf,
            indent,
            "net: *const MnistNet,",
            .{},
        );
    }
    return true;
}

/// Dispatch optimizer-related line replacements:
/// update calls, .optimizer field, net parameter type.
fn tryReplaceOptimizer(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    trimmed: []const u8,
    indent: usize,
    changes: *const ConfigChanges,
    first_net_param: *bool,
) bool {
    const opt = changes.optimizer orelse return false;

    if (tryReplaceUpdateCall(
        arena,
        buf,
        trimmed,
        indent,
        opt,
        changes,
    )) return true;

    // Benchmark config .optimizer field.
    if (tools.startsWith(trimmed, ".optimizer = .")) {
        tools.writeIndented(
            arena,
            buf,
            indent,
            ".optimizer = .{s},",
            .{opt},
        );
        return true;
    }

    return tryReplaceNetParam(
        arena,
        buf,
        trimmed,
        indent,
        opt,
        first_net_param,
    );
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
// config-backup / config-restore
// ============================================================

fn toolConfigBackup(arena: Allocator) !void {
    const src_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        MAIN_PATH,
    );
    const dst_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        BACKUP_PATH,
    );
    try tools.copyFile(src_fs, dst_fs);
    try tools.writeStdout(
        "{\"status\": \"ok\", " ++
            "\"backup\": \"src/main.zig.bak\"}\n",
    );
}

fn toolConfigRestore(arena: Allocator) !void {
    const bak_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        BACKUP_PATH,
    );
    if (!tools.fileExists(bak_fs)) {
        return error.NoBackupFile;
    }
    const main_fs = try tools.resolveToFs(
        arena,
        config.fs_root,
        MAIN_PATH,
    );
    try tools.copyFile(bak_fs, main_fs);
    try tools.writeStdout(
        "{\"status\": \"ok\", " ++
            "\"restored\": \"src/main.zig\"}\n",
    );
}
