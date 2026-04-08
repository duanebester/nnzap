const std = @import("std");
const mnist = @import("mnist.zig");
const Mnist = mnist.Mnist;
const oneHot = mnist.oneHot;
const readU32BE = mnist.readU32BE;
const train_count = mnist.train_count;
const test_count = mnist.test_count;
const image_size = mnist.image_size;
const num_classes = mnist.num_classes;

test "oneHot" {
    const allocator = std.testing.allocator;

    const raw = [_]u8{ 0, 3, 9, 1 };
    const oh = try oneHot(allocator, &raw);
    defer allocator.free(oh);

    // Label 0 → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    try std.testing.expectEqual(@as(f32, 1.0), oh[0 * 10 + 0]);
    try std.testing.expectEqual(@as(f32, 0.0), oh[0 * 10 + 1]);

    // Label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    try std.testing.expectEqual(@as(f32, 1.0), oh[1 * 10 + 3]);
    try std.testing.expectEqual(@as(f32, 0.0), oh[1 * 10 + 0]);

    // Label 9 → [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    try std.testing.expectEqual(@as(f32, 1.0), oh[2 * 10 + 9]);

    // Label 1 → [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    try std.testing.expectEqual(@as(f32, 1.0), oh[3 * 10 + 1]);

    // Total ones should equal number of labels.
    var ones: u32 = 0;
    for (oh) |v| {
        if (v == 1.0) ones += 1;
    }
    try std.testing.expectEqual(@as(u32, 4), ones);
}

test "readU32BE" {
    const bytes = [_]u8{ 0x00, 0x00, 0x08, 0x03 };
    try std.testing.expectEqual(@as(u32, 0x0803), readU32BE(&bytes));

    const count_bytes = [_]u8{ 0x00, 0x00, 0xEA, 0x60 };
    try std.testing.expectEqual(@as(u32, 60_000), readU32BE(&count_bytes));
}

test "load MNIST from disk" {
    const allocator = std.testing.allocator;

    // Skip gracefully if the data files haven't been downloaded.
    var data = Mnist.load(allocator, "../data/mnist") catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print(
                "  (skipped — ../data/mnist not found, " ++
                    "run: curl + gunzip first)\n",
                .{},
            );
            return;
        }
        return err;
    };
    defer data.deinit(allocator);

    // -- Shape checks --
    try std.testing.expectEqual(
        @as(usize, train_count * image_size),
        data.train_images.len,
    );
    try std.testing.expectEqual(
        @as(usize, train_count * num_classes),
        data.train_labels.len,
    );
    try std.testing.expectEqual(
        @as(usize, train_count),
        data.train_labels_raw.len,
    );
    try std.testing.expectEqual(
        @as(usize, test_count * image_size),
        data.test_images.len,
    );
    try std.testing.expectEqual(
        @as(usize, test_count * num_classes),
        data.test_labels.len,
    );
    try std.testing.expectEqual(
        @as(usize, test_count),
        data.test_labels_raw.len,
    );

    // -- Pixel range: every value in [0, 1] --
    for (data.train_images) |v| {
        try std.testing.expect(v >= 0.0 and v <= 1.0);
    }
    for (data.test_images) |v| {
        try std.testing.expect(v >= 0.0 and v <= 1.0);
    }

    // -- Raw labels in [0, 9] --
    for (data.train_labels_raw) |l| {
        try std.testing.expect(l < num_classes);
    }
    for (data.test_labels_raw) |l| {
        try std.testing.expect(l < num_classes);
    }

    // -- One-hot consistency: each row sums to 1.0 and the
    //    hot index matches the raw label --
    for (0..train_count) |i| {
        const row = data.train_labels[i * num_classes ..][0..num_classes];
        var sum: f32 = 0.0;
        var hot: u8 = 0;
        for (row, 0..) |v, c| {
            sum += v;
            if (v == 1.0) hot = @intCast(c);
        }
        try std.testing.expectApproxEqAbs(
            @as(f32, 1.0),
            sum,
            1e-6,
        );
        try std.testing.expectEqual(
            data.train_labels_raw[i],
            hot,
        );
    }

    // -- Label distribution: every digit appears at least once
    //    in the training set (MNIST has ~6k per digit) --
    var digit_counts = [_]u32{0} ** num_classes;
    for (data.train_labels_raw) |l| {
        digit_counts[l] += 1;
    }
    for (digit_counts, 0..) |count, d| {
        if (count == 0) {
            std.debug.print(
                "  digit {d} has zero samples!\n",
                .{d},
            );
            return error.TestUnexpectedResult;
        }
        // Each digit should have roughly 5,000-7,000 samples.
        try std.testing.expect(count > 4000);
        try std.testing.expect(count < 8000);
    }
}
