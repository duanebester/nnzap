//! MNIST IDX file loader
//!
//! Reads the classic MNIST handwritten digit dataset from IDX binary
//! files, normalises pixel values to f32 [0, 1], and converts labels
//! to one-hot f32 vectors.
//!
//! IDX format (big-endian):
//!   Images — 16-byte header: magic(0x0803) count rows cols + raw u8
//!   Labels —  8-byte header: magic(0x0801) count             + raw u8
//!
//! Usage:
//!   var mnist = try Mnist.load(allocator, "../data/mnist");
//!   defer mnist.deinit(allocator);
//!
//!   const pixel = mnist.train_images[img_idx * 784 + px];  // f32 in [0,1]
//!   const label = mnist.train_labels[img_idx * 10 + class]; // one-hot f32

const std = @import("std");

pub const image_rows: u32 = 28;
pub const image_cols: u32 = 28;
pub const image_size: u32 = image_rows * image_cols; // 784
pub const num_classes: u32 = 10;
pub const train_count: u32 = 60_000;
pub const test_count: u32 = 10_000;

pub const Mnist = struct {
    /// Normalised pixel data [count × 784], row-major, f32 in [0, 1].
    train_images: []f32,
    /// One-hot label vectors [count × 10], f32.
    train_labels: []f32,
    /// Raw integer labels [count], u8 in [0, 9].
    train_labels_raw: []u8,

    /// Test set (same layout as training).
    test_images: []f32,
    test_labels: []f32,
    test_labels_raw: []u8,

    /// Load all four MNIST files from `base_dir`.
    /// Expects the standard filenames:
    ///   train-images-idx3-ubyte
    ///   train-labels-idx1-ubyte
    ///   t10k-images-idx3-ubyte
    ///   t10k-labels-idx1-ubyte
    pub fn load(
        allocator: std.mem.Allocator,
        base_dir: []const u8,
    ) !Mnist {
        // -- Training set --
        const train_imgs = try loadImages(
            allocator,
            base_dir,
            "train-images-idx3-ubyte",
            train_count,
        );
        errdefer allocator.free(train_imgs);

        const train_raw = try loadLabelsRaw(
            allocator,
            base_dir,
            "train-labels-idx1-ubyte",
            train_count,
        );
        errdefer allocator.free(train_raw);

        const train_oh = try oneHot(allocator, train_raw);
        errdefer allocator.free(train_oh);

        // -- Test set --
        const test_imgs = try loadImages(
            allocator,
            base_dir,
            "t10k-images-idx3-ubyte",
            test_count,
        );
        errdefer allocator.free(test_imgs);

        const test_raw = try loadLabelsRaw(
            allocator,
            base_dir,
            "t10k-labels-idx1-ubyte",
            test_count,
        );
        errdefer allocator.free(test_raw);

        const test_oh = try oneHot(allocator, test_raw);
        errdefer allocator.free(test_oh);

        return .{
            .train_images = train_imgs,
            .train_labels = train_oh,
            .train_labels_raw = train_raw,
            .test_images = test_imgs,
            .test_labels = test_oh,
            .test_labels_raw = test_raw,
        };
    }

    pub fn deinit(self: *Mnist, allocator: std.mem.Allocator) void {
        allocator.free(self.test_labels_raw);
        allocator.free(self.test_labels);
        allocator.free(self.test_images);
        allocator.free(self.train_labels_raw);
        allocator.free(self.train_labels);
        allocator.free(self.train_images);
        self.* = undefined;
    }

    /// Copy a batch of images into a pre-allocated f32 slice.
    /// `dst` must hold at least `batch_size * image_size` floats.
    /// `indices` contains the sample indices to copy.
    pub fn fillImageBatch(
        images: []const f32,
        indices: []const u32,
        dst: []f32,
    ) void {
        std.debug.assert(dst.len >= indices.len * image_size);
        for (indices, 0..) |idx, b| {
            const src_off = @as(usize, idx) * image_size;
            const dst_off = b * image_size;
            @memcpy(
                dst[dst_off..][0..image_size],
                images[src_off..][0..image_size],
            );
        }
    }

    /// Copy a batch of one-hot labels into a pre-allocated f32 slice.
    /// `dst` must hold at least `batch_size * num_classes` floats.
    pub fn fillLabelBatch(
        labels: []const f32,
        indices: []const u32,
        dst: []f32,
    ) void {
        std.debug.assert(dst.len >= indices.len * num_classes);
        for (indices, 0..) |idx, b| {
            const src_off = @as(usize, idx) * num_classes;
            const dst_off = b * num_classes;
            @memcpy(
                dst[dst_off..][0..num_classes],
                labels[src_off..][0..num_classes],
            );
        }
    }
};

// ============================================================================
// Internal helpers
// ============================================================================

/// Read a big-endian u32 from a byte slice.
fn readU32BE(bytes: []const u8) u32 {
    return std.mem.readInt(u32, bytes[0..4], .big);
}

/// Load and normalise an IDX3 image file (magic 0x0803).
fn loadImages(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    filename: []const u8,
    expected_count: u32,
) ![]f32 {
    const raw = try readFile(allocator, base_dir, filename);
    defer allocator.free(raw);

    // Validate header.
    if (raw.len < 16) return error.InvalidHeader;
    const magic = readU32BE(raw[0..4]);
    if (magic != 0x0803) return error.BadMagicNumber;

    const count = readU32BE(raw[4..8]);
    if (count != expected_count) return error.UnexpectedCount;

    const rows = readU32BE(raw[8..12]);
    const cols = readU32BE(raw[12..16]);
    if (rows != image_rows or cols != image_cols)
        return error.UnexpectedDimensions;

    const pixel_count: usize = @as(usize, count) * image_size;
    const expected_len = 16 + pixel_count;
    if (raw.len < expected_len) return error.TruncatedFile;

    // Normalise u8 → f32 in [0, 1].
    const images = try allocator.alloc(f32, pixel_count);
    errdefer allocator.free(images);

    const pixels = raw[16..][0..pixel_count];
    for (pixels, 0..) |byte, i| {
        images[i] = @as(f32, @floatFromInt(byte)) / 255.0;
    }

    return images;
}

/// Load raw u8 labels from an IDX1 label file (magic 0x0801).
fn loadLabelsRaw(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    filename: []const u8,
    expected_count: u32,
) ![]u8 {
    const raw = try readFile(allocator, base_dir, filename);
    defer allocator.free(raw);

    // Validate header.
    if (raw.len < 8) return error.InvalidHeader;
    const magic = readU32BE(raw[0..4]);
    if (magic != 0x0801) return error.BadMagicNumber;

    const count = readU32BE(raw[4..8]);
    if (count != expected_count) return error.UnexpectedCount;

    const expected_len = 8 + @as(usize, count);
    if (raw.len < expected_len) return error.TruncatedFile;

    // Copy label bytes into their own allocation so we can
    // free the raw file buffer.
    const labels = try allocator.alloc(u8, count);
    errdefer allocator.free(labels);
    @memcpy(labels, raw[8..][0..count]);

    // Validate range.
    for (labels) |l| {
        if (l >= num_classes) return error.LabelOutOfRange;
    }

    return labels;
}

/// Convert raw u8 labels to one-hot f32 vectors [count × 10].
fn oneHot(
    allocator: std.mem.Allocator,
    raw: []const u8,
) ![]f32 {
    const total = raw.len * num_classes;
    const out = try allocator.alloc(f32, total);
    errdefer allocator.free(out);
    @memset(out, 0.0);

    for (raw, 0..) |label, i| {
        out[i * num_classes + label] = 1.0;
    }

    return out;
}

/// Read an entire file into memory.
fn readFile(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    filename: []const u8,
) ![]u8 {
    // Build path: base_dir ++ "/" ++ filename.
    const path = try std.fs.path.join(allocator, &.{
        base_dir, filename,
    });
    defer allocator.free(path);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // MNIST files are at most ~47 MB; 64 MB cap is generous.
    const max_size = 64 * 1024 * 1024;
    return file.readToEndAlloc(allocator, max_size);
}

// ============================================================================
// Tests
// ============================================================================

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
