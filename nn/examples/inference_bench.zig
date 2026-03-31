//! nnzap inference benchmark
//!
//! Measures three inference paths and writes results to JSON:
//!
//!   Phase 1 — GPU batched throughput
//!     Forward-pass 10,240 samples in batches of 64, packed
//!     into multi-batch command buffers.  Measures images/sec.
//!
//!   Phase 2 — GPU single-sample latency
//!     Forward-pass 1 sample with forwardInfer, commitAndWait,
//!     repeat 1,000 times.  Reports p50/p99 in microseconds.
//!
//!   Phase 3 — CPU single-sample latency
//!     Forward-pass 1 sample with forwardCPU (pure Zig, no
//!     Metal).  Repeat 1,000 times.  Reports p50/p99 in us.
//!
//! Run:  zig build run-infer

const std = @import("std");
const nn = @import("nn");

const Buffer = nn.Buffer;
const Device = nn.Device;
const nanosToMs = nn.benchmark.nanosToMs;

// -- Network architecture (same as MNIST) -------------

const Layout = nn.NetworkLayout(&.{
    .{ .in = 784, .out = 128, .act = .relu },
    .{ .in = 128, .out = 64, .act = .relu },
    .{ .in = 64, .out = 10, .act = .none },
});

const BATCH_SIZE: u32 = 64;
const Net = nn.Network(Layout, BATCH_SIZE);

// -- Benchmark parameters (Rule 4 — hard limits) -----

const WARMUP_ITERS: u32 = 100;
const GPU_THROUGHPUT_BATCHES: u32 = 160;
const GPU_LATENCY_ITERS: u32 = 1_000;
const CPU_LATENCY_ITERS: u32 = 1_000;
const BATCHES_PER_CMD: u32 = 64;
const MAX_LATENCY_SAMPLES: u32 = 10_000;

const seed: u64 = 42;

comptime {
    std.debug.assert(GPU_THROUGHPUT_BATCHES > 0);
    std.debug.assert(GPU_LATENCY_ITERS > 0);
    std.debug.assert(CPU_LATENCY_ITERS > 0);
    std.debug.assert(BATCHES_PER_CMD > 0);
    std.debug.assert(
        GPU_LATENCY_ITERS <= MAX_LATENCY_SAMPLES,
    );
    std.debug.assert(
        CPU_LATENCY_ITERS <= MAX_LATENCY_SAMPLES,
    );
}

// -- Result types ------------------------------------

const GpuBatchedResult = struct {
    total_samples: u32,
    batch_size: u32,
    total_ms: f64,
    images_per_sec: f64,
};

const LatencyResult = struct {
    iterations: u32,
    mean_us: f64,
    p50_us: f64,
    p99_us: f64,
    min_us: f64,
    max_us: f64,
};

// -- Architecture for JSON output --------------------

const ArchLayer = struct {
    input_size: u32,
    output_size: u32,
    activation: []const u8,
};

/// Comptime-extracted architecture description for JSON.
const architecture = blk: {
    var specs: [Layout.num_layers]ArchLayer = undefined;
    for (0..Layout.num_layers) |i| {
        specs[i] = .{
            .input_size = Layout.layers[i].in,
            .output_size = Layout.layers[i].out,
            .activation = @tagName(
                Layout.layers[i].act,
            ),
        };
    }
    break :blk specs;
};

// ====================================================
// Main
// ====================================================

pub fn main() !void {
    std.debug.print(
        "\nnnzap inference benchmark\n" ++
            "=========================\n\n",
        .{},
    );

    var device: Device = undefined;
    try device.init();
    std.debug.print(
        "Metal: unified_memory={}\n",
        .{device.unified_memory},
    );
    Layout.printSummary();

    // In-place init (Rule 13).
    var net: Net = undefined;
    try net.init(&device);
    defer net.deinit();

    var prng = std.Random.DefaultPrng.init(seed);
    heInit(
        Layout,
        net.paramSlice(),
        prng.random(),
    );

    // Single input buffer, reused for all phases.
    // Sized for the largest batch (BATCH_SIZE); single-
    // sample phases just read the first input_size floats.

    var input_buf = try device.createBuffer(
        BATCH_SIZE * Layout.input_size,
    );
    defer input_buf.deinit();

    fillRandom(input_buf.asSlice(), prng.random());

    // Warm the GPU pipeline before measuring.
    std.debug.print(
        "Warming up ({d} iterations)... ",
        .{WARMUP_ITERS},
    );
    warmup(&device, &net, input_buf);
    std.debug.print("done.\n\n", .{});

    const gpu_batched = benchGpuBatched(
        &device,
        &net,
        input_buf,
    );
    printGpuBatched(gpu_batched);

    const gpu_single = benchGpuSingle(
        &device,
        &net,
        input_buf,
    );
    printLatency(
        "Phase 2 \xe2\x80\x94 GPU single-sample",
        gpu_single,
    );

    const cpu_single = benchCpuSingle(
        &net,
        input_buf.asSlice(),
    );
    printLatency(
        "Phase 3 \xe2\x80\x94 CPU single-sample",
        cpu_single,
    );

    try saveResults(
        gpu_batched,
        gpu_single,
        cpu_single,
    );
}

// ====================================================
// Warmup
// ====================================================

/// Run warmup forward passes to prime the GPU pipeline.
/// Metal JITs shader compilation on first dispatch, so
/// cold starts would skew latency measurements.
/// Warms both the batched path (forwardInfer) and the
/// single-sample fused path (forwardInferFused).
fn warmup(
    device: *const Device,
    net: *const Net,
    input: Buffer,
) void {
    std.debug.assert(input.len >= Layout.input_size);
    std.debug.assert(WARMUP_ITERS > 0);

    // Warm fused single-sample path (unretained to match
    // the measurement path).
    var i: u32 = 0;
    while (i < WARMUP_ITERS) : (i += 1) {
        const cmd = device.beginCommandBufferUnretained();
        const enc = device.beginCompute(cmd);
        net.forwardInferFused(device, enc, input);
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }

    // Warm batched path (separate dispatches).
    i = 0;
    while (i < WARMUP_ITERS / 10) : (i += 1) {
        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);
        net.forwardInfer(device, enc, input, 1);
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }
}

// ====================================================
// Phase 1 — GPU batched throughput
// ====================================================

/// Forward-pass many samples in batches, packed into
/// multi-batch command buffers.  Measures peak throughput
/// with amortised Metal dispatch overhead.
fn benchGpuBatched(
    device: *const Device,
    net: *const Net,
    input: Buffer,
) GpuBatchedResult {
    const min_len = BATCH_SIZE * Layout.input_size;
    std.debug.assert(input.len >= min_len);
    std.debug.assert(GPU_THROUGHPUT_BATCHES > 0);

    const total_samples: u32 =
        GPU_THROUGHPUT_BATCHES * BATCH_SIZE;
    const start = std.time.nanoTimestamp();

    var batch: u32 = 0;
    while (batch < GPU_THROUGHPUT_BATCHES) {
        const remaining =
            GPU_THROUGHPUT_BATCHES - batch;
        const group = @min(BATCHES_PER_CMD, remaining);

        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);

        var g: u32 = 0;
        while (g < group) : (g += 1) {
            net.forwardInfer(
                device,
                enc,
                input,
                BATCH_SIZE,
            );
        }

        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
        batch += group;
    }

    const elapsed_ms = nanosToMs(
        std.time.nanoTimestamp() - start,
    );

    return .{
        .total_samples = total_samples,
        .batch_size = BATCH_SIZE,
        .total_ms = elapsed_ms,
        .images_per_sec = if (elapsed_ms > 0.0)
            @as(f64, @floatFromInt(total_samples)) /
                (elapsed_ms / 1000.0)
        else
            0.0,
    };
}

// ====================================================
// Phase 2 — GPU single-sample latency
// ====================================================

/// One forward pass per command buffer, commitAndWait
/// each time.  Measures full Metal round-trip for
/// batch=1 — includes command buffer creation, encoder
/// setup, dispatch, and GPU-to-CPU synchronisation.
///
/// Uses the fused 3-layer kernel to minimise dispatch
/// overhead: 1 dispatch instead of 3 per forward pass.
fn benchGpuSingle(
    device: *const Device,
    net: *const Net,
    input: Buffer,
) LatencyResult {
    std.debug.assert(input.len >= Layout.input_size);
    std.debug.assert(GPU_LATENCY_ITERS > 0);

    var latencies: [GPU_LATENCY_ITERS]f64 = undefined;

    var i: u32 = 0;
    while (i < GPU_LATENCY_ITERS) : (i += 1) {
        const start = std.time.nanoTimestamp();

        // Use unretained references to skip retain/release
        // on bound buffers — safe because all network
        // buffers outlive every command buffer.
        const cmd = device.beginCommandBufferUnretained();
        const enc = device.beginCompute(cmd);
        net.forwardInferFused(device, enc, input);
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);

        latencies[i] = nanosToUs(
            std.time.nanoTimestamp() - start,
        );
    }

    return computeStats(&latencies, GPU_LATENCY_ITERS);
}

// ====================================================
// Phase 3 — CPU single-sample latency
// ====================================================

/// Pure Zig forward pass — no Metal dispatch at all.
/// Reads weights directly from the unified-memory params
/// buffer (zero copy) and computes matmul + bias + act
/// in tight loops.
fn benchCpuSingle(
    net: *const Net,
    input_data: []const f32,
) LatencyResult {
    std.debug.assert(input_data.len >= Layout.input_size);
    std.debug.assert(CPU_LATENCY_ITERS > 0);

    var latencies: [CPU_LATENCY_ITERS]f64 = undefined;

    var i: u32 = 0;
    while (i < CPU_LATENCY_ITERS) : (i += 1) {
        const start = std.time.nanoTimestamp();
        net.forwardCPU(input_data, 1);
        latencies[i] = nanosToUs(
            std.time.nanoTimestamp() - start,
        );
    }

    return computeStats(&latencies, CPU_LATENCY_ITERS);
}

// ====================================================
// Statistics
// ====================================================

/// Compute latency statistics from raw measurements.
/// Sorts the array in-place to extract percentiles.
fn computeStats(
    latencies: []f64,
    count: u32,
) LatencyResult {
    std.debug.assert(count > 0);
    std.debug.assert(latencies.len >= count);

    const slice = latencies[0..count];
    sortAscending(slice);

    var sum: f64 = 0.0;
    for (slice) |v| sum += v;

    // Nearest-rank percentile on 0-based indices.
    const n: usize = @as(usize, count) - 1;
    const p50_idx: usize = n * 50 / 100;
    const p99_idx: usize = n * 99 / 100;

    return .{
        .iterations = count,
        .mean_us = sum / @as(f64, @floatFromInt(count)),
        .p50_us = slice[p50_idx],
        .p99_us = slice[p99_idx],
        .min_us = slice[0],
        .max_us = slice[count - 1],
    };
}

/// Insertion sort — O(n^2) but n <= 1,000 so the ~1M
/// comparisons are negligible (<1 ms).  Avoids stdlib
/// sort API version dependencies.
fn sortAscending(slice: []f64) void {
    std.debug.assert(slice.len > 0);
    std.debug.assert(slice.len <= MAX_LATENCY_SAMPLES);

    var i: usize = 1;
    while (i < slice.len) : (i += 1) {
        const key = slice[i];
        var j: usize = i;
        while (j > 0) {
            if (slice[j - 1] <= key) break;
            slice[j] = slice[j - 1];
            j -= 1;
        }
        slice[j] = key;
    }
}

// ====================================================
// Printing
// ====================================================

fn printGpuBatched(r: GpuBatchedResult) void {
    std.debug.assert(r.total_samples > 0);
    std.debug.assert(r.batch_size > 0);

    const batches = r.total_samples / r.batch_size;
    std.debug.print(
        "Phase 1 \xe2\x80\x94 GPU batched throughput\n" ++
            "  {d} samples, batch={d}," ++
            " {d} batches\n" ++
            "  Time: {d:.1} ms" ++
            "  ({d:.0} images/sec)\n\n",
        .{
            r.total_samples,
            r.batch_size,
            batches,
            r.total_ms,
            r.images_per_sec,
        },
    );
}

fn printLatency(
    label: []const u8,
    r: LatencyResult,
) void {
    std.debug.assert(label.len > 0);
    std.debug.assert(r.iterations > 0);

    std.debug.print(
        "{s} ({d} iterations)\n" ++
            "  mean={d:.1} us  p50={d:.1} us" ++
            "  p99={d:.1} us\n" ++
            "  min={d:.1} us  max={d:.1} us\n\n",
        .{
            label,
            r.iterations,
            r.mean_us,
            r.p50_us,
            r.p99_us,
            r.min_us,
            r.max_us,
        },
    );
}

// ====================================================
// JSON output
// ====================================================

const BenchmarkView = struct {
    timestamp_utc: []const u8,
    architecture: []const ArchLayer,
    param_count: u32,
    gpu_batched: GpuBatchedResult,
    gpu_single_sample: LatencyResult,
    cpu_single_sample: LatencyResult,
};

/// Write benchmark results to benchmarks/inference_*.json.
fn saveResults(
    gpu_batched: GpuBatchedResult,
    gpu_single: LatencyResult,
    cpu_single: LatencyResult,
) !void {
    std.debug.assert(gpu_batched.total_samples > 0);
    std.debug.assert(gpu_single.iterations > 0);

    try ensureDir("benchmarks");

    var ts_buf: [32]u8 = undefined;
    const timestamp = formatTimestamp(&ts_buf);

    var fs_buf: [32]u8 = undefined;
    const file_ts = sanitiseForPath(
        &fs_buf,
        timestamp,
    );

    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(
        &path_buf,
        "benchmarks/inference_{s}.json",
        .{file_ts},
    ) catch unreachable;

    const file = try std.fs.cwd().createFile(
        path,
        .{},
    );
    defer file.close();

    const view = BenchmarkView{
        .timestamp_utc = timestamp,
        .architecture = &architecture,
        .param_count = Layout.param_count,
        .gpu_batched = gpu_batched,
        .gpu_single_sample = gpu_single,
        .cpu_single_sample = cpu_single,
    };

    var write_buf: [8192]u8 = undefined;
    var fw = file.writer(&write_buf);
    try std.json.Stringify.value(
        view,
        .{ .whitespace = .indent_4 },
        &fw.interface,
    );
    try fw.interface.flush();

    std.debug.print(
        "Benchmark saved: {s}\n",
        .{path},
    );
}

// ====================================================
// Helpers
// ====================================================

/// He uniform initialisation: w ~ U(-limit, limit),
/// limit = sqrt(6 / fan_in).  Biases start at zero.
fn heInit(
    comptime L: type,
    params: []f32,
    random: std.Random,
) void {
    std.debug.assert(params.len == L.param_count);
    std.debug.assert(L.num_layers > 0);

    inline for (0..L.num_layers) |li| {
        const fan_in: f32 = @floatFromInt(
            L.layers[li].in,
        );
        const limit = @sqrt(6.0 / fan_in);
        for (L.getWeightSlice(params, li)) |*w| {
            w.* = random.float(f32) * 2.0 * limit -
                limit;
        }
        @memset(L.getBiasSlice(params, li), 0.0);
    }
}

/// Fill a buffer with random values in [0, 1).
fn fillRandom(slice: []f32, random: std.Random) void {
    std.debug.assert(slice.len > 0);
    for (slice) |*v| v.* = random.float(f32);
}

/// Convert a nanosecond delta to microseconds.
fn nanosToUs(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000.0;
}

/// Format current UTC time as ISO-8601.
fn formatTimestamp(buf: *[32]u8) []const u8 {
    const secs: u64 = @intCast(std.time.timestamp());
    std.debug.assert(secs > 0);

    const es = std.time.epoch.EpochSeconds{
        .secs = secs,
    };
    const day_sec = es.getDaySeconds();
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();

    return std.fmt.bufPrint(
        buf,
        "{d:0>4}-{d:0>2}-{d:0>2}" ++
            "T{d:0>2}:{d:0>2}:{d:0>2}Z",
        .{
            yd.year,
            @intFromEnum(md.month),
            @as(u32, md.day_index) + 1,
            day_sec.getHoursIntoDay(),
            day_sec.getMinutesIntoHour(),
            day_sec.getSecondsIntoMinute(),
        },
    ) catch unreachable;
}

/// Replace ':' with '-' for filesystem-safe timestamps.
fn sanitiseForPath(
    buf: *[32]u8,
    src: []const u8,
) []const u8 {
    std.debug.assert(src.len > 0);
    std.debug.assert(src.len <= 32);

    @memcpy(buf[0..src.len], src);
    for (buf[0..src.len]) |*c| {
        if (c.* == ':') c.* = '-';
    }
    return buf[0..src.len];
}

/// Create a directory if it does not already exist.
fn ensureDir(path: []const u8) !void {
    std.debug.assert(path.len > 0);

    std.fs.cwd().makeDir(path) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
}
