//! Bonsai 1.7B inference benchmark — structured throughput measurement.
//!
//! Measures prefill and decode throughput, per-token latency percentiles,
//! and writes structured JSON results to nn/benchmarks/.
//!
//! Uses the library's `generate` API so the benchmark automatically
//! benefits from all engine optimisations (spin-wait, fused kernels,
//! unretained command buffers, etc.) without reimplementing dispatch.
//!
//! Usage:
//!   zig build run-bonsai-bench -- [model_dir]
//!
//! The model directory defaults to ~/models/bonsai-1.7b.

const std = @import("std");
const nn = @import("nn");

// ── Config aliases ───────────────────────────────────
const Config = nn.Bonsai1_7B;
const BonsaiModel = nn.Model(Config);
const transformer = nn.transformer;

// ── Limits ───────────────────────────────────────────
const MAX_SHARDS: u32 = 16;
const MAX_PATH_LEN: u32 = 4096;
const MAX_PROMPT_TOKENS: u32 = 8192;
const WARMUP_TOKENS: u32 = 8;
const MEASURE_TOKENS: u32 = 64;
const TOTAL_DECODE_TOKENS: u32 = WARMUP_TOKENS + MEASURE_TOKENS;

const BENCH_PROMPT =
    "Explain the theory of general relativity in detail.";

// ============================================================
// Entry point
// ============================================================

pub fn main() !void {
    const model_dir = resolveModelDir();

    std.debug.print(
        "\n\x1b[1m[nnmetal] Bonsai 1.7B Benchmark\x1b[0m\n" ++
            "model:          {s}\n" ++
            "warmup tokens:  {d}\n" ++
            "measure tokens: {d}\n\n",
        .{ model_dir, WARMUP_TOKENS, MEASURE_TOKENS },
    );

    // ── Metal device and pipelines ───────────────────
    std.debug.print("Initializing Metal...\n", .{});

    var device: nn.Device = undefined;
    try device.init();

    var pipelines: nn.TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    // Compile specialized QMV kernels with model dimensions
    // baked in as constexpr for full loop unrolling.
    try nn.specialized_qmv.initOnDevice(
        &device,
        Config.hidden_size,
        Config.intermediate_size,
        Config.group_size,
    );

    // ── Model allocation and weight loading ──────────
    std.debug.print("Allocating model buffers...\n", .{});

    var model: BonsaiModel = undefined;
    try model.init(device.obj);
    defer model.deinit();

    std.debug.print("Loading weights...\n", .{});
    const t_load_start = std.time.nanoTimestamp();

    var shard_storage: [MAX_SHARDS][MAX_PATH_LEN]u8 =
        undefined;
    var shard_slices: [MAX_SHARDS][]const u8 = undefined;
    const shard_count = try findSafetensorShards(
        model_dir,
        &shard_storage,
        &shard_slices,
    );
    try model.loadWeights(
        shard_slices[0..shard_count],
    );

    const t_load_end = std.time.nanoTimestamp();
    const load_ns: u64 =
        @intCast(t_load_end - t_load_start);

    // ── Tokenizer ────────────────────────────────────
    std.debug.print("Loading tokenizer...\n", .{});

    var tokenizer_path_buf: [MAX_PATH_LEN]u8 = undefined;
    const tokenizer_path = try std.fmt.bufPrint(
        &tokenizer_path_buf,
        "{s}/tokenizer.json",
        .{@as([]const u8, model_dir)},
    );

    var tokenizer: nn.Tokenizer = undefined;
    try tokenizer.init(
        std.heap.page_allocator,
        tokenizer_path,
    );
    defer tokenizer.deinit();

    // ── Encode prompt ────────────────────────────────
    var prompt_ids: [MAX_PROMPT_TOKENS]u32 = undefined;
    const prompt_len = try tokenizer.applyChatTemplate(
        BENCH_PROMPT,
        &prompt_ids,
    );
    std.debug.assert(prompt_len > 0);
    std.debug.assert(prompt_len <= MAX_PROMPT_TOKENS);

    std.debug.print(
        "Prompt: {d} tokens\n\n",
        .{prompt_len},
    );

    // ── EOS tokens (Qwen3 uses two stop tokens) ─────
    const eos_ids = [_]u32{
        tokenizer.eos_token_id,
        tokenizer.im_end_token_id,
    };

    // ── Sampling scratch (heap, init-time only) ──────
    const page_alloc = std.heap.page_allocator;

    const scratch = try page_alloc.alloc(
        f32,
        Config.vocab_size,
    );
    defer page_alloc.free(scratch);

    const indices = try page_alloc.alloc(
        u32,
        Config.vocab_size,
    );
    defer page_alloc.free(indices);

    // Greedy decoding: temperature=0 for determinism.
    const sampling = transformer.SamplingParams{
        .temperature = 0.0,
        .top_k = 0,
        .top_p = 1.0,
        .seed = 42,
    };

    // ── Generate (prefill + warmup + measurement) ────
    // A single `generate` call handles chunked prefill,
    // warmup decode, and measured decode.  Per-token
    // timings are collected by the library itself, so
    // we get latency data without reimplementing GPU
    // dispatch.
    var output_tokens: [TOTAL_DECODE_TOKENS]u32 =
        undefined;
    var per_token_ns: [TOTAL_DECODE_TOKENS]u64 =
        undefined;
    var args = model.forwardDecodeArgs(0, 0);

    std.debug.print(
        "Running generate ({d} warmup + {d} measure)...\n",
        .{ WARMUP_TOKENS, MEASURE_TOKENS },
    );

    const result = transformer.generate(
        Config,
        &device,
        &pipelines,
        &args,
        .{
            .prompt_ids = prompt_ids[0..prompt_len],
            .params = sampling,
            .eos_ids = &eos_ids,
            .output_tokens = &output_tokens,
            .scratch = scratch,
            .indices = indices,
            .per_token_ns = &per_token_ns,
        },
    );

    // ── Compute statistics ───────────────────────────
    // Skip the first WARMUP_TOKENS timings; use the
    // remaining entries for latency percentiles.
    const generated = result.tokens_generated;
    const warmup_count = @min(WARMUP_TOKENS, generated);
    const measure_count: u32 =
        if (generated > warmup_count)
            generated - warmup_count
        else
            0;

    if (measure_count == 0) {
        std.debug.print(
            "error: not enough tokens generated " ++
                "({d}) for measurement\n",
            .{generated},
        );
        return;
    }

    const results = computeResults(
        load_ns,
        result.prefill_ns,
        prompt_len,
        &per_token_ns,
        warmup_count,
        measure_count,
    );

    // ── Print summary to stderr ──────────────────────
    printSummary(&results, prompt_len);

    // ── Write JSON to benchmarks/ ────────────────────
    try writeJsonResults(&results, prompt_len);
}

// ============================================================
// Statistics
// ============================================================

const BenchResults = struct {
    load_ms: f64,
    prefill_tok_per_sec: f64,
    decode_tok_per_sec: f64,
    decode_p50_us: u64,
    decode_p99_us: u64,
    measured_tokens: u32,
    timestamp_ns: i128,
    power_source: PowerSource,
};

const PowerSource = enum {
    ac,
    battery,
    unknown,

    fn label(self: PowerSource) []const u8 {
        return switch (self) {
            .ac => "ac",
            .battery => "battery",
            .unknown => "unknown",
        };
    }
};

/// Compute aggregate statistics from raw per-token
/// nanosecond timings.  The timing array contains both
/// warmup and measurement entries; `warmup_count` entries
/// are skipped before extracting percentiles.
fn computeResults(
    load_ns: u64,
    prefill_ns: u64,
    prompt_len: u32,
    all_timings: *[TOTAL_DECODE_TOKENS]u64,
    warmup_count: u32,
    measure_count: u32,
) BenchResults {
    std.debug.assert(prompt_len > 0);
    std.debug.assert(measure_count > 0);
    std.debug.assert(
        warmup_count + measure_count <= TOTAL_DECODE_TOKENS,
    );

    const load_ms = nanosToMs(load_ns);
    const prefill_ms = nanosToMs(prefill_ns);

    const prefill_tps: f64 = if (prefill_ms > 0.0)
        @as(f64, @floatFromInt(prompt_len)) /
            (prefill_ms / 1000.0)
    else
        0.0;

    // Extract measurement slice (skip warmup entries).
    const measure_slice =
        all_timings[warmup_count..][0..measure_count];

    // Sort measurement timings for percentile extraction.
    std.sort.pdq(
        u64,
        measure_slice,
        {},
        std.sort.asc(u64),
    );

    // Total decode time is the sum of all measured token
    // timings (not wall clock across the phase).
    var total_decode_ns: u64 = 0;
    for (measure_slice) |t| {
        total_decode_ns += t;
    }
    const decode_ms = nanosToMs(total_decode_ns);

    const decode_tps: f64 = if (decode_ms > 0.0)
        @as(f64, @floatFromInt(measure_count)) /
            (decode_ms / 1000.0)
    else
        0.0;

    // p50 = sorted[len/2], p99 = sorted[len*99/100].
    const p50_idx = measure_count / 2;
    const p99_idx = (measure_count * 99) / 100;
    const p50_us = measure_slice[p50_idx] / 1000;
    const p99_us = measure_slice[p99_idx] / 1000;

    return BenchResults{
        .load_ms = load_ms,
        .prefill_tok_per_sec = prefill_tps,
        .decode_tok_per_sec = decode_tps,
        .decode_p50_us = p50_us,
        .decode_p99_us = p99_us,
        .measured_tokens = measure_count,
        .timestamp_ns = std.time.nanoTimestamp(),
        .power_source = detectPowerSource(),
    };
}

// ============================================================
// Output — console summary
// ============================================================

fn printSummary(
    r: *const BenchResults,
    prompt_len: u32,
) void {
    std.debug.assert(r.measured_tokens > 0);
    std.debug.assert(prompt_len > 0);

    std.debug.print(
        "\n\x1b[1m--- benchmark results " ++
            "----------------------------\x1b[0m\n" ++
            "load:    {d:.1} ms\n" ++
            "prefill: {d} tokens  ({d:.1} tok/s)\n" ++
            "decode:  {d} tokens  ({d:.1} tok/s)\n" ++
            "latency: p50={d} us  p99={d} us\n" ++
            "power:   {s}\n\n",
        .{
            r.load_ms,
            prompt_len,
            r.prefill_tok_per_sec,
            r.measured_tokens,
            r.decode_tok_per_sec,
            r.decode_p50_us,
            r.decode_p99_us,
            r.power_source.label(),
        },
    );
}

// ============================================================
// Output — JSON file
// ============================================================

/// Write structured JSON results to
/// nn/benchmarks/bonsai_bench_<epoch_ns>.json.
fn writeJsonResults(
    r: *const BenchResults,
    prompt_len: u32,
) !void {
    std.debug.assert(r.measured_tokens > 0);
    std.debug.assert(prompt_len > 0);

    // Ensure the benchmarks directory exists.
    std.fs.cwd().makePath("benchmarks") catch |err| {
        std.debug.print(
            "error: cannot create benchmarks/: {}\n",
            .{err},
        );
        return err;
    };

    // Build the timestamp string from epoch nanos.
    // We use the raw nanosecond epoch to avoid needing
    // a full calendar library.
    const epoch_ns = r.timestamp_ns;
    const epoch_secs: u64 = @intCast(
        @divFloor(epoch_ns, 1_000_000_000),
    );
    var ts_buf: [64]u8 = undefined;
    const ts_str = formatEpochUtc(epoch_secs, &ts_buf);

    // Build the filename.
    var fname_buf: [MAX_PATH_LEN]u8 = undefined;
    const fname = std.fmt.bufPrint(
        &fname_buf,
        "benchmarks/bonsai_bench_{d}.json",
        .{epoch_secs},
    ) catch unreachable;

    // Format JSON into a stack buffer.
    var json_buf: [2048]u8 = undefined;
    const json = std.fmt.bufPrint(
        &json_buf,
        "{{\n" ++
            "  \"timestamp_utc\": \"{s}\",\n" ++
            "  \"model\": \"Bonsai-1.7B\",\n" ++
            "  \"prompt_tokens\": {d},\n" ++
            "  \"warmup_tokens\": {d},\n" ++
            "  \"measured_tokens\": {d},\n" ++
            "  \"load_ms\": {d:.1},\n" ++
            "  \"prefill_tok_per_sec\": {d:.1},\n" ++
            "  \"decode_tok_per_sec\": {d:.1},\n" ++
            "  \"decode_p50_us\": {d},\n" ++
            "  \"decode_p99_us\": {d},\n" ++
            "  \"power_source\": \"{s}\"\n" ++
            "}}\n",
        .{
            ts_str,
            prompt_len,
            WARMUP_TOKENS,
            r.measured_tokens,
            r.load_ms,
            r.prefill_tok_per_sec,
            r.decode_tok_per_sec,
            r.decode_p50_us,
            r.decode_p99_us,
            r.power_source.label(),
        },
    ) catch unreachable;

    // Write the file.
    const file = std.fs.cwd().createFile(
        fname,
        .{},
    ) catch |err| {
        std.debug.print(
            "error: cannot create {s}: {}\n",
            .{ fname, err },
        );
        return err;
    };
    defer file.close();

    file.writeAll(json) catch |err| {
        std.debug.print(
            "error: write failed for {s}: {}\n",
            .{ fname, err },
        );
        return err;
    };

    // Also write JSON to stdout so engine_research can
    // capture it without needing to find the file.
    _ = std.posix.write(
        std.posix.STDOUT_FILENO,
        json,
    ) catch {};

    std.debug.print(
        "Results written to {s}\n",
        .{fname},
    );
}

// ============================================================
// Power source detection
// ============================================================

/// Detect whether the machine is on AC or battery power
/// by running `pmset -g batt` and parsing the first line.
/// Returns .unknown if detection fails (e.g. on a desktop
/// Mac with no battery, or if pmset is unavailable).
fn detectPowerSource() PowerSource {
    const result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &.{ "/usr/bin/pmset", "-g", "batt" },
        .max_output_bytes = 4096,
    }) catch return .unknown;

    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (!ok) return .unknown;

    // First line: "Now drawing from 'AC Power'" or
    //             "Now drawing from 'Battery Power'".
    const stdout = result.stdout;
    if (std.mem.indexOf(u8, stdout, "'AC Power'")) |_| {
        return .ac;
    }
    if (std.mem.indexOf(
        u8,
        stdout,
        "'Battery Power'",
    )) |_| {
        return .battery;
    }
    return .unknown;
}

// ============================================================
// UTC timestamp formatting
// ============================================================

/// Format a Unix epoch (seconds) as an ISO 8601 UTC string.
/// Uses a simple arithmetic calendar decomposition —
/// no allocator, no dependency.
fn formatEpochUtc(
    epoch_secs: u64,
    buf: *[64]u8,
) []const u8 {
    std.debug.assert(epoch_secs < 32_503_680_000);
    // Seconds-in-a-day is exact; no leap-second fuzz
    // needed for a benchmark timestamp.
    const secs_per_day: u64 = 86400;
    const day_count = epoch_secs / secs_per_day;
    const day_secs = epoch_secs % secs_per_day;

    const hour: u64 = day_secs / 3600;
    const minute: u64 = (day_secs % 3600) / 60;
    const second: u64 = day_secs % 60;

    // Civil date from day count (algorithm from
    // Howard Hinnant's chrono paper, public domain).
    const z = day_count + 719468;
    const era = z / 146097;
    const doe = z - era * 146097;
    const yoe = (doe - doe / 1460 + doe / 36524 -
        doe / 146096) / 365;
    const y = yoe + era * 400;
    const doy = doe -
        (365 * yoe + yoe / 4 - yoe / 100);
    const mp = (5 * doy + 2) / 153;
    const d = doy - (153 * mp + 2) / 5 + 1;
    const m = if (mp < 10) mp + 3 else mp - 9;
    const year = if (m <= 2) y + 1 else y;

    std.debug.assert(m >= 1 and m <= 12);
    std.debug.assert(d >= 1 and d <= 31);

    const result = std.fmt.bufPrint(
        buf,
        "{d:0>4}-{d:0>2}-{d:0>2}T" ++
            "{d:0>2}:{d:0>2}:{d:0>2}Z",
        .{ year, m, d, hour, minute, second },
    ) catch unreachable;

    return result;
}

// ============================================================
// CLI — model directory resolution
// ============================================================

/// Resolve the model directory from CLI args or the
/// default ~/models/bonsai-1.7b path.
fn resolveModelDir() [:0]const u8 {
    var iter = std.process.args();
    _ = iter.next(); // Skip argv[0].

    // First positional argument overrides the default.
    if (iter.next()) |arg| {
        std.debug.assert(arg.len > 0);
        return arg;
    }

    // Fall back to ~/models/bonsai-1.7b.
    const home = std.posix.getenv("HOME") orelse {
        std.debug.print(
            "error: HOME not set and no model dir " ++
                "provided\n",
            .{},
        );
        std.process.exit(1);
    };
    std.debug.assert(home.len > 0);

    // Build the default path into a static buffer so
    // the returned slice lives for the program lifetime.
    const S = struct {
        var buf: [MAX_PATH_LEN]u8 = undefined;
        var sentinel_buf: [MAX_PATH_LEN + 1]u8 =
            undefined;
    };
    const path = std.fmt.bufPrint(
        &S.buf,
        "{s}/models/bonsai-1.7b",
        .{home},
    ) catch {
        std.debug.print(
            "error: HOME path too long\n",
            .{},
        );
        std.process.exit(1);
    };

    // Copy into sentinel-terminated buffer.
    @memcpy(S.sentinel_buf[0..path.len], path);
    S.sentinel_buf[path.len] = 0;
    return S.sentinel_buf[0..path.len :0];
}

// ============================================================
// Safetensors shard discovery
// ============================================================

/// Scan `dir_path` for *.safetensors files and populate
/// `path_storage` and `path_slices`.  Returns the number
/// of shards found.  Shards are sorted alphabetically for
/// deterministic loading order.
fn findSafetensorShards(
    dir_path: []const u8,
    path_storage: *[MAX_SHARDS][MAX_PATH_LEN]u8,
    path_slices: *[MAX_SHARDS][]const u8,
) !u32 {
    std.debug.assert(dir_path.len > 0);
    std.debug.assert(dir_path.len < MAX_PATH_LEN);

    var dir = std.fs.cwd().openDir(
        dir_path,
        .{ .iterate = true },
    ) catch |err| {
        std.debug.print(
            "error: cannot open model directory: " ++
                "{s} ({})\n",
            .{ dir_path, err },
        );
        return err;
    };
    defer dir.close();

    var count: u32 = 0;
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        const name = entry.name;
        if (!std.mem.endsWith(
            u8,
            name,
            ".safetensors",
        )) {
            continue;
        }
        if (count >= MAX_SHARDS) {
            std.debug.print(
                "error: too many shards (max {d})\n",
                .{MAX_SHARDS},
            );
            return error.TooManyShards;
        }
        const path = std.fmt.bufPrint(
            &path_storage[count],
            "{s}/{s}",
            .{ dir_path, name },
        ) catch {
            return error.PathTooLong;
        };
        path_slices[count] = path;
        count += 1;
    }

    if (count == 0) {
        std.debug.print(
            "error: no .safetensors files in {s}\n",
            .{dir_path},
        );
        return error.NoSafetensorFiles;
    }

    // Sort for deterministic shard loading order.
    std.sort.pdq(
        []const u8,
        path_slices[0..count],
        {},
        pathLessThan,
    );

    std.debug.print(
        "Found {d} safetensor shard(s)\n",
        .{count},
    );
    return count;
}

fn pathLessThan(
    _: void,
    a: []const u8,
    b: []const u8,
) bool {
    return std.mem.order(u8, a, b) == .lt;
}

// ============================================================
// Utility
// ============================================================

fn nanosToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}
