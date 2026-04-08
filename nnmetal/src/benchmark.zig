//! Benchmark result collection and JSON serialization.
//!
//! Records training configuration, per-epoch metrics, and test
//! results into a fixed-capacity struct.  Writes results to JSON
//! files in the benchmarks/ directory for tracking optimisation
//! progress and hyperparameter research.
//!
//! Usage:
//!     var bench: Benchmark = undefined;
//!     bench.init(MnistLayout, .{
//!         .batch_size = 64,
//!         .learning_rate = 0.1,
//!         .num_epochs = 10,
//!         .train_samples = 50_000,
//!         .validation_samples = 10_000,
//!         .test_samples = 10_000,
//!     });
//!
//!     // In the training loop:
//!     bench.recordEpoch(.{
//!         .epoch = 1,
//!         .train_loss = loss,
//!         .duration_ms = epoch_ms,
//!         .validation_loss = val_loss,
//!         .validation_accuracy_pct = val_acc,
//!     });
//!
//!     // After training:
//!     bench.setTrainingTime(total_ms);
//!     bench.recordTest(9786, 10000, 77.0);
//!     try bench.save("mnist");

const std = @import("std");

// -- Hard limits (Rule 4) --

const MAX_EPOCHS: u32 = 1000;
const MAX_LAYERS: u32 = 64;

// ============================================================
// Public types
// ============================================================

pub const Optimizer = enum {
    sgd,
    adam,
};

pub const LossFunction = enum {
    cross_entropy,
    mse,
};

/// Describes one layer in a benchmark result.  Mirrors
/// layout.LayerDesc but uses a string activation name so the
/// struct is self-describing in JSON output.
pub const LayerSpec = struct {
    input_size: u32,
    output_size: u32,
    activation: []const u8,
};

/// Training configuration — the knobs for a benchmark run.
pub const Config = struct {
    batch_size: u32,
    learning_rate: f32,
    learning_rate_decay: f32 = 0.0,
    optimizer: Optimizer = .sgd,
    loss_function: LossFunction = .cross_entropy,
    num_epochs: u32,
    seed: u64 = 42,
    train_samples: u32,
    validation_samples: u32 = 0,
    test_samples: u32,
};

/// Metrics captured for a single training epoch.
///
/// Pass this struct to Benchmark.recordEpoch().  Validation
/// fields are optional — omit them (null) when no validation
/// set is configured.
pub const EpochResult = struct {
    epoch: u32,
    train_loss: f32,
    duration_ms: f64,
    validation_loss: ?f32 = null,
    validation_accuracy_pct: ?f64 = null,
};

/// Metrics captured for a test-set evaluation.
pub const TestResult = struct {
    correct: u32,
    total: u32,
    accuracy_pct: f64,
    duration_ms: f64,
};

// ============================================================
// Benchmark collector
// ============================================================

/// Collects configuration, per-epoch training metrics, and
/// test results into fixed-capacity storage.  Call save() at
/// the end to write a JSON file to benchmarks/.
///
/// All storage is inline with hard caps (Rule 2).  Use
/// in-place init (Rule 13) since the struct is ~20 KB.
pub const Benchmark = struct {
    // -- Architecture (comptime-extracted) --
    layers: [MAX_LAYERS]LayerSpec,
    layer_count: u32,
    param_count: u32,

    // -- Hyperparameters --
    config: Config,

    // -- Per-epoch results (filled during training) --
    epochs: [MAX_EPOCHS]EpochResult,
    epoch_count: u32,

    // -- Test result (filled after training) --
    test_result: TestResult,
    has_test_result: bool,

    // -- Aggregate timing --
    total_training_ms: f64,

    /// Initialise in-place from a comptime Layout type.
    /// Extracts the architecture description and stores
    /// the training configuration.
    pub fn init(
        self: *Benchmark,
        comptime Layout: type,
        config: Config,
    ) void {
        comptime {
            std.debug.assert(Layout.num_layers > 0);
            std.debug.assert(
                Layout.num_layers <= MAX_LAYERS,
            );
        }
        std.debug.assert(config.num_epochs <= MAX_EPOCHS);
        std.debug.assert(config.batch_size > 0);

        const arch = comptime extractArchitecture(Layout);

        self.* = .{
            .layers = undefined,
            .layer_count = Layout.num_layers,
            .param_count = Layout.param_count,
            .config = config,
            .epochs = undefined,
            .epoch_count = 0,
            .test_result = undefined,
            .has_test_result = false,
            .total_training_ms = 0.0,
        };

        @memcpy(
            self.layers[0..Layout.num_layers],
            &arch,
        );
    }

    /// Record one epoch's metrics.  Pass an EpochResult
    /// struct — validation fields default to null when
    /// omitted.  Call once per epoch in ascending order.
    pub fn recordEpoch(
        self: *Benchmark,
        result: EpochResult,
    ) void {
        std.debug.assert(result.epoch > 0);
        std.debug.assert(self.epoch_count < MAX_EPOCHS);

        self.epochs[self.epoch_count] = result;
        self.epoch_count += 1;
    }

    /// Record test-set evaluation results.
    pub fn recordTest(
        self: *Benchmark,
        correct: u32,
        total: u32,
        duration_ms: f64,
    ) void {
        std.debug.assert(total > 0);
        std.debug.assert(correct <= total);

        const pct =
            @as(f64, @floatFromInt(correct)) /
            @as(f64, @floatFromInt(total)) * 100.0;

        self.test_result = .{
            .correct = correct,
            .total = total,
            .accuracy_pct = pct,
            .duration_ms = duration_ms,
        };
        self.has_test_result = true;
    }

    /// Set total wall-clock training time (milliseconds).
    pub fn setTrainingTime(
        self: *Benchmark,
        total_ms: f64,
    ) void {
        std.debug.assert(total_ms >= 0.0);
        std.debug.assert(self.epoch_count > 0);

        self.total_training_ms = total_ms;
    }

    /// Write the benchmark to benchmarks/{name}_{ts}.json.
    pub fn save(
        self: *const Benchmark,
        name: []const u8,
    ) !void {
        std.debug.assert(name.len > 0);
        std.debug.assert(self.epoch_count > 0);

        try ensureDir("benchmarks");

        // Timestamp for the JSON body (ISO-8601).
        var ts_buf: [32]u8 = undefined;
        const timestamp = formatTimestamp(&ts_buf);

        // Filesystem-safe copy (colons → dashes).
        var fs_buf: [32]u8 = undefined;
        const file_ts = sanitiseForPath(
            &fs_buf,
            timestamp,
        );

        var path_buf: [256]u8 = undefined;
        const path = std.fmt.bufPrint(
            &path_buf,
            "benchmarks/{s}_{s}.json",
            .{ name, file_ts },
        ) catch unreachable;

        const file = try std.fs.cwd().createFile(
            path,
            .{},
        );
        defer file.close();

        var write_buf: [8192]u8 = undefined;
        var fw = file.writer(&write_buf);
        try writeResult(self, timestamp, &fw.interface);
        try fw.interface.flush();

        std.debug.print(
            "Benchmark saved: {s}\n",
            .{path},
        );
    }
};

// ============================================================
// JSON serialization (view structs for std.json)
// ============================================================

// View structs hold slices into the Benchmark's fixed arrays.
// No copies, no allocation — just a reshape for the
// serialiser.

const ResultView = struct {
    timestamp_utc: []const u8,
    config: ConfigView,
    final_train_loss: f32,
    final_validation_loss: ?f32,
    final_validation_accuracy_pct: ?f64,
    final_test_accuracy_pct: ?f64,
    epochs: []const EpochResult,
    test_result: ?TestResult,
    total_training_ms: f64,
    throughput_images_per_sec: f64,
};

const ConfigView = struct {
    architecture: []const LayerSpec,
    param_count: u32,
    batch_size: u32,
    learning_rate: f32,
    learning_rate_decay: f32,
    optimizer: Optimizer,
    loss_function: LossFunction,
    num_epochs: u32,
    seed: u64,
    train_samples: u32,
    validation_samples: u32,
    test_samples: u32,
};

/// Build a ResultView from a Benchmark and write it as
/// pretty-printed JSON.
fn writeResult(
    bench: *const Benchmark,
    timestamp: []const u8,
    w: *std.io.Writer,
) !void {
    std.debug.assert(bench.epoch_count > 0);
    std.debug.assert(timestamp.len > 0);

    const cfg = &bench.config;
    const last = bench.epochs[bench.epoch_count - 1];

    const view = ResultView{
        .timestamp_utc = timestamp,
        .final_train_loss = last.train_loss,
        .final_validation_loss = last.validation_loss,
        .final_validation_accuracy_pct = last.validation_accuracy_pct,
        .final_test_accuracy_pct = if (bench.has_test_result)
            bench.test_result.accuracy_pct
        else
            null,
        .config = .{
            .architecture = bench.layers[0..bench.layer_count],
            .param_count = bench.param_count,
            .batch_size = cfg.batch_size,
            .learning_rate = cfg.learning_rate,
            .learning_rate_decay = cfg.learning_rate_decay,
            .optimizer = cfg.optimizer,
            .loss_function = cfg.loss_function,
            .num_epochs = cfg.num_epochs,
            .seed = cfg.seed,
            .train_samples = cfg.train_samples,
            .validation_samples = cfg.validation_samples,
            .test_samples = cfg.test_samples,
        },
        .epochs = bench.epochs[0..bench.epoch_count],
        .test_result = if (bench.has_test_result)
            bench.test_result
        else
            null,
        .total_training_ms = bench.total_training_ms,
        .throughput_images_per_sec = computeThroughput(
            cfg.train_samples,
            bench.epoch_count,
            bench.total_training_ms,
        ),
    };

    try std.json.Stringify.value(
        view,
        .{ .whitespace = .indent_4 },
        w,
    );
}

// ============================================================
// Comptime helpers
// ============================================================

/// Extract architecture from a comptime NetworkLayout type.
/// Activation names come from @tagName — they are string
/// literals with static lifetime.
pub fn extractArchitecture(
    comptime Layout: type,
) [Layout.num_layers]LayerSpec {
    comptime {
        std.debug.assert(Layout.num_layers > 0);
        std.debug.assert(Layout.num_layers <= MAX_LAYERS);
    }

    var specs: [Layout.num_layers]LayerSpec = undefined;
    inline for (0..Layout.num_layers) |i| {
        specs[i] = .{
            .input_size = Layout.layers[i].in,
            .output_size = Layout.layers[i].out,
            .activation = @tagName(
                Layout.layers[i].act,
            ),
        };
    }
    return specs;
}

// ============================================================
// Timestamp formatting
// ============================================================

/// Format current UTC time as ISO-8601:
/// "YYYY-MM-DDTHH:MM:SSZ".
pub fn formatTimestamp(buf: *[32]u8) []const u8 {
    const secs: u64 = @intCast(std.time.timestamp());
    std.debug.assert(secs > 0);

    const es = std.time.epoch.EpochSeconds{
        .secs = secs,
    };
    const day_sec = es.getDaySeconds();
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();

    const fmt = "{d:0>4}-{d:0>2}-{d:0>2}" ++
        "T{d:0>2}:{d:0>2}:{d:0>2}Z";

    return std.fmt.bufPrint(buf, fmt, .{
        yd.year,
        @intFromEnum(md.month),
        @as(u32, md.day_index) + 1,
        day_sec.getHoursIntoDay(),
        day_sec.getMinutesIntoHour(),
        day_sec.getSecondsIntoMinute(),
    }) catch unreachable;
}

/// Copy `src` into `buf`, replacing ':' with '-' for
/// filesystem-safe timestamps.
pub fn sanitiseForPath(
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

// ============================================================
// Filesystem helpers
// ============================================================

/// Create a directory if it does not already exist.
fn ensureDir(path: []const u8) !void {
    std.debug.assert(path.len > 0);

    std.fs.cwd().makeDir(path) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
}

// ============================================================
// Arithmetic helpers
// ============================================================

/// Images processed per second across the full training run.
pub fn computeThroughput(
    train_samples: u32,
    epoch_count: u32,
    total_ms: f64,
) f64 {
    std.debug.assert(epoch_count > 0);

    if (total_ms <= 0.0) return 0.0;

    const images =
        @as(f64, @floatFromInt(train_samples)) *
        @as(f64, @floatFromInt(epoch_count));
    return images / (total_ms / 1000.0);
}

/// Convert a nanosecond delta (i128) to milliseconds (f64).
/// Shared between the benchmark module and callers that time
/// training/evaluation phases with std.time.nanoTimestamp().
pub fn nanosToMs(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

test {
    _ = @import("benchmark_test.zig");
}
