//! Bonsai 1.7B inference benchmark — structured throughput measurement.
//!
//! Measures prefill and decode throughput, per-token latency percentiles,
//! and writes structured JSON results to nn/benchmarks/.
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
const MAX_TIMINGS: u32 = WARMUP_TOKENS + MEASURE_TOKENS + 256;

const BENCH_PROMPT =
    "Explain the theory of general relativity in detail.";

// ============================================================
// Entry point
// ============================================================

pub fn main() !void {
    const model_dir = resolveModelDir();

    std.debug.print(
        "\n\x1b[1m[nnzap] Bonsai 1.7B Benchmark\x1b[0m\n" ++
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
    var prng = std.Random.DefaultPrng.init(
        sampling.seed,
    );
    const rng = prng.random();

    // ── Prefill ──────────────────────────────────────
    std.debug.print("Running prefill...\n", .{});
    const t_prefill_start = std.time.nanoTimestamp();
    for (prompt_ids[0..prompt_len], 0..) |tok, pos| {
        dispatchOneDecode(
            &device,
            &pipelines,
            &model,
            tok,
            @intCast(pos),
        );
    }
    const t_prefill_end = std.time.nanoTimestamp();
    const prefill_ns: u64 =
        @intCast(t_prefill_end - t_prefill_start);

    // Sample first token from prefill logits.
    const logits_slice = model.logits.asSlice();
    var next_token = transformer.sampleToken(
        logits_slice,
        sampling,
        scratch,
        indices,
        rng,
    );

    // ── Decode: warmup + measurement ─────────────────
    std.debug.print(
        "Warming up ({d} tokens)...\n",
        .{WARMUP_TOKENS},
    );
    var position: u32 = @intCast(prompt_len);
    next_token = runDecodePhase(
        &device,
        &pipelines,
        &model,
        next_token,
        &position,
        WARMUP_TOKENS,
        &eos_ids,
        sampling,
        scratch,
        indices,
        rng,
        null, // No timing collection during warmup.
    );

    std.debug.print(
        "Measuring ({d} tokens)...\n",
        .{MEASURE_TOKENS},
    );
    var timings_ns: [MAX_TIMINGS]u64 = undefined;
    var timing_count: u32 = 0;
    _ = runDecodePhase(
        &device,
        &pipelines,
        &model,
        next_token,
        &position,
        MEASURE_TOKENS,
        &eos_ids,
        sampling,
        scratch,
        indices,
        rng,
        &.{
            .buf = &timings_ns,
            .count = &timing_count,
        },
    );

    // ── Compute statistics ───────────────────────────
    const results = computeResults(
        load_ns,
        prefill_ns,
        prompt_len,
        &timings_ns,
        timing_count,
    );

    // ── Print summary to stderr ──────────────────────
    printSummary(&results, prompt_len);

    // ── Write JSON to benchmarks/ ────────────────────
    try writeJsonResults(&results, prompt_len);
}

// ============================================================
// Decode phase runner
// ============================================================

/// Timing collection context passed into the decode loop.
const TimingCtx = struct {
    buf: *[MAX_TIMINGS]u64,
    count: *u32,
};

/// Run `token_count` decode steps, optionally collecting
/// per-token nanosecond timings.  Returns the last sampled
/// token so the caller can continue generation.
fn runDecodePhase(
    device: *const nn.Device,
    pipelines: *const nn.TransformerPipelines,
    model: *const BonsaiModel,
    first_token: u32,
    position: *u32,
    token_count: u32,
    eos_ids: []const u32,
    sampling: transformer.SamplingParams,
    scratch: []f32,
    sort_indices: []u32,
    rng: std.Random,
    timing: ?*const TimingCtx,
) u32 {
    std.debug.assert(token_count > 0);
    std.debug.assert(position.* < Config.max_context_length);

    var current_token = first_token;
    var generated: u32 = 0;

    while (generated < token_count) {
        // Stop on EOS or context exhaustion.
        if (isEos(current_token, eos_ids)) break;
        if (position.* >= Config.max_context_length) break;

        const t_start = std.time.nanoTimestamp();
        dispatchOneDecode(
            device,
            pipelines,
            model,
            current_token,
            position.*,
        );
        const t_end = std.time.nanoTimestamp();

        position.* += 1;
        generated += 1;

        // Record timing if collecting.
        if (timing) |ctx| {
            if (ctx.count.* < MAX_TIMINGS) {
                ctx.buf[ctx.count.*] =
                    @intCast(t_end - t_start);
                ctx.count.* += 1;
            }
        }

        // Sample next token.
        const logits_slice = model.logits.asSlice();
        current_token = transformer.sampleToken(
            logits_slice,
            sampling,
            scratch,
            sort_indices,
            rng,
        );
    }

    return current_token;
}

fn isEos(token: u32, eos_ids: []const u32) bool {
    std.debug.assert(eos_ids.len > 0);
    for (eos_ids) |eid| {
        if (token == eid) return true;
    }
    return false;
}

// ============================================================
// GPU dispatch helper
// ============================================================

/// Run a single-token forward decode pass (embedding →
/// N blocks → final norm → LM head) and wait for the
/// GPU to finish.  After this returns, `model.logits`
/// contains valid [vocab_size] f32 values.
fn dispatchOneDecode(
    device: *const nn.Device,
    pipelines: *const nn.TransformerPipelines,
    model_ptr: *const BonsaiModel,
    token_id: u32,
    position: u32,
) void {
    std.debug.assert(token_id < Config.vocab_size);
    std.debug.assert(
        position < Config.max_context_length,
    );

    const args = model_ptr.forwardDecodeArgs(
        token_id,
        position,
    );

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    nn.transformer.forwardDecode(
        Config,
        device,
        enc,
        pipelines,
        args,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);
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
};

/// Compute aggregate statistics from raw per-token
/// nanosecond timings.  Sorts the timing array in place
/// for percentile extraction.
fn computeResults(
    load_ns: u64,
    prefill_ns: u64,
    prompt_len: u32,
    timings_ns: *[MAX_TIMINGS]u64,
    timing_count: u32,
) BenchResults {
    std.debug.assert(prompt_len > 0);
    std.debug.assert(timing_count > 0);

    const load_ms = nanosToMs(load_ns);
    const prefill_ms = nanosToMs(prefill_ns);

    const prefill_tps: f64 = if (prefill_ms > 0.0)
        @as(f64, @floatFromInt(prompt_len)) /
            (prefill_ms / 1000.0)
    else
        0.0;

    // Sort timings for percentile calculation.
    const slice = timings_ns[0..timing_count];
    std.sort.pdq(u64, slice, {}, std.sort.asc(u64));

    // Total decode time is the sum of all individual
    // token timings (not wall clock across the phase).
    var total_decode_ns: u64 = 0;
    for (slice) |t| {
        total_decode_ns += t;
    }
    const decode_ms = nanosToMs(total_decode_ns);

    const decode_tps: f64 = if (decode_ms > 0.0)
        @as(f64, @floatFromInt(timing_count)) /
            (decode_ms / 1000.0)
    else
        0.0;

    // p50 = sorted[len/2], p99 = sorted[len*99/100].
    const p50_idx = timing_count / 2;
    const p99_idx = (timing_count * 99) / 100;
    const p50_us = slice[p50_idx] / 1000;
    const p99_us = slice[p99_idx] / 1000;

    return BenchResults{
        .load_ms = load_ms,
        .prefill_tok_per_sec = prefill_tps,
        .decode_tok_per_sec = decode_tps,
        .decode_p50_us = p50_us,
        .decode_p99_us = p99_us,
        .measured_tokens = timing_count,
        .timestamp_ns = std.time.nanoTimestamp(),
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
            "latency: p50={d} us  p99={d} us\n\n",
        .{
            r.load_ms,
            prompt_len,
            r.prefill_tok_per_sec,
            r.measured_tokens,
            r.decode_tok_per_sec,
            r.decode_p50_us,
            r.decode_p99_us,
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
            "  \"decode_p99_us\": {d}\n" ++
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
