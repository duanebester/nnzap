const std = @import("std");
const benchmark = @import("benchmark.zig");
const Benchmark = benchmark.Benchmark;
const extractArchitecture = benchmark.extractArchitecture;
const computeThroughput = benchmark.computeThroughput;
const sanitiseForPath = benchmark.sanitiseForPath;
const formatTimestamp = benchmark.formatTimestamp;
const nanosToMs = benchmark.nanosToMs;

test "extractArchitecture" {
    const l = @import("layout.zig");
    const Layout = l.NetworkLayout(&.{
        .{ .in = 784, .out = 128, .act = .relu },
        .{ .in = 128, .out = 64, .act = .relu },
        .{ .in = 64, .out = 10, .act = .none },
    });
    const arch = extractArchitecture(Layout);

    try std.testing.expectEqual(
        @as(u32, 784),
        arch[0].input_size,
    );
    try std.testing.expectEqual(
        @as(u32, 128),
        arch[0].output_size,
    );
    try std.testing.expectEqualSlices(
        u8,
        "relu",
        arch[0].activation,
    );

    try std.testing.expectEqual(
        @as(u32, 64),
        arch[1].output_size,
    );

    try std.testing.expectEqual(
        @as(u32, 10),
        arch[2].output_size,
    );
    try std.testing.expectEqualSlices(
        u8,
        "none",
        arch[2].activation,
    );
}

test "computeThroughput" {
    // 50k images x 10 epochs in 5000 ms = 100k img/s.
    const t = computeThroughput(50_000, 10, 5000.0);
    try std.testing.expectApproxEqAbs(
        @as(f64, 100_000.0),
        t,
        0.1,
    );

    // Zero time → zero throughput (avoid division by zero).
    const z = computeThroughput(50_000, 10, 0.0);
    try std.testing.expectEqual(@as(f64, 0.0), z);
}

test "recordEpoch and recordTest" {
    const l = @import("layout.zig");
    const Layout = l.NetworkLayout(&.{
        .{ .in = 2, .out = 4, .act = .relu },
        .{ .in = 4, .out = 1, .act = .none },
    });

    var bench: Benchmark = undefined;
    bench.init(Layout, .{
        .batch_size = 32,
        .learning_rate = 0.01,
        .num_epochs = 5,
        .train_samples = 800,
        .validation_samples = 200,
        .test_samples = 200,
    });

    // No epochs recorded yet.
    try std.testing.expectEqual(
        @as(u32, 0),
        bench.epoch_count,
    );
    try std.testing.expect(!bench.has_test_result);

    // Record three epochs with validation metrics.
    bench.recordEpoch(.{
        .epoch = 1,
        .train_loss = 2.3,
        .duration_ms = 100.0,
        .validation_loss = 2.1,
        .validation_accuracy_pct = 40.0,
    });
    bench.recordEpoch(.{
        .epoch = 2,
        .train_loss = 1.1,
        .duration_ms = 95.0,
        .validation_loss = 1.0,
        .validation_accuracy_pct = 65.0,
    });
    bench.recordEpoch(.{
        .epoch = 3,
        .train_loss = 0.5,
        .duration_ms = 90.0,
        .validation_loss = 0.6,
        .validation_accuracy_pct = 82.0,
    });

    try std.testing.expectEqual(
        @as(u32, 3),
        bench.epoch_count,
    );
    try std.testing.expectEqual(
        @as(u32, 2),
        bench.epochs[1].epoch,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.1),
        bench.epochs[1].train_loss,
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f32, 1.0),
        bench.epochs[1].validation_loss.?,
        1e-6,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 65.0),
        bench.epochs[1].validation_accuracy_pct.?,
        0.01,
    );

    // Record epoch without validation (defaults to null).
    bench.recordEpoch(.{
        .epoch = 4,
        .train_loss = 0.3,
        .duration_ms = 88.0,
    });
    try std.testing.expectEqual(
        @as(?f32, null),
        bench.epochs[3].validation_loss,
    );

    // Record test result.
    bench.recordTest(180, 200, 50.0);
    try std.testing.expect(bench.has_test_result);
    try std.testing.expectEqual(
        @as(u32, 180),
        bench.test_result.correct,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 90.0),
        bench.test_result.accuracy_pct,
        0.01,
    );

    // Training time.
    bench.setTrainingTime(373.0);
    try std.testing.expectApproxEqAbs(
        @as(f64, 373.0),
        bench.total_training_ms,
        0.01,
    );
}

test "sanitiseForPath" {
    var buf: [32]u8 = undefined;
    const result = sanitiseForPath(
        &buf,
        "2025-07-15T14:30:00Z",
    );
    try std.testing.expectEqualSlices(
        u8,
        "2025-07-15T14-30-00Z",
        result,
    );
}

test "formatTimestamp shape" {
    var buf: [32]u8 = undefined;
    const ts = formatTimestamp(&buf);

    // 20 chars: "YYYY-MM-DDTHH:MM:SSZ".
    try std.testing.expect(ts.len == 20);
    try std.testing.expectEqual(@as(u8, 'T'), ts[10]);
    try std.testing.expectEqual(@as(u8, 'Z'), ts[19]);
    try std.testing.expectEqual(@as(u8, '2'), ts[0]);
    try std.testing.expectEqual(@as(u8, '0'), ts[1]);
}

test "init extracts architecture" {
    const l = @import("layout.zig");
    const Layout = l.NetworkLayout(&.{
        .{ .in = 784, .out = 128, .act = .relu },
        .{ .in = 128, .out = 64, .act = .relu },
        .{ .in = 64, .out = 10, .act = .none },
    });

    var bench: Benchmark = undefined;
    bench.init(Layout, .{
        .batch_size = 64,
        .learning_rate = 0.1,
        .num_epochs = 10,
        .train_samples = 50_000,
        .validation_samples = 10_000,
        .test_samples = 10_000,
    });

    try std.testing.expectEqual(
        @as(u32, 3),
        bench.layer_count,
    );
    try std.testing.expectEqual(
        @as(u32, Layout.param_count),
        bench.param_count,
    );
    try std.testing.expectEqual(
        @as(u32, 784),
        bench.layers[0].input_size,
    );
    try std.testing.expectEqualSlices(
        u8,
        "relu",
        bench.layers[0].activation,
    );
    try std.testing.expectEqualSlices(
        u8,
        "none",
        bench.layers[2].activation,
    );
    try std.testing.expectEqual(
        @as(u32, 10_000),
        bench.config.validation_samples,
    );
}

test "nanosToMs" {
    const ms = nanosToMs(1_500_000);
    try std.testing.expectApproxEqAbs(
        @as(f64, 1.5),
        ms,
        1e-9,
    );
}
