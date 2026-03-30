//! nnzap MNIST benchmark (#13)
//!
//! Architecture: 784 → 128 (relu) → 64 (relu) → 10 (none)
//! Loss:         softmax + cross-entropy (fused backward)
//! Optimiser:    mini-batch SGD
//!
//! The dataset is split into three partitions:
//!   Train:      50,000 samples (SGD weight updates)
//!   Validation: 10,000 samples (per-epoch monitoring)
//!   Test:       10,000 samples (final held-out evaluation)
//!
//! The last layer uses act=.none so the network outputs raw
//! logits.  Softmax is fused into the CE backward kernel for
//! training, and applied separately during evaluation for
//! loss computation.
//!
//! Results are saved to benchmarks/{name}_{timestamp}.json
//! for tracking optimisation progress over time.
//!
//! Run:  zig build run

const std = @import("std");
const nnzap = @import("nnzap");

const Mnist = nnzap.Mnist;
const Buffer = nnzap.Buffer;
const Device = nnzap.Device;
const Benchmark = nnzap.Benchmark;
const EpochResult = nnzap.benchmark.EpochResult;
const nanosToMs = nnzap.benchmark.nanosToMs;

// -- Network architecture (comptime-resolved) ---------

const MnistLayout = nnzap.NetworkLayout(&.{
    .{ .in = 784, .out = 128, .act = .relu },
    .{ .in = 128, .out = 64, .act = .relu },
    .{ .in = 64, .out = 10, .act = .none },
});

const max_batch: u32 = 64;
const num_classes: u32 = MnistLayout.output_size;
const MnistNet = nnzap.Network(MnistLayout, max_batch);

// -- Hyperparameters ----------------------------------

const seed: u64 = 42;
const num_epochs: u32 = 20;
const learning_rate: f32 = 0.2;

// -- Dataset split ------------------------------------

const total_train: u32 = 60_000;
const val_count: u32 = 10_000;
const train_count: u32 = total_train - val_count; // 50,000
const test_count: u32 = 10_000;
const batches_per_epoch: u32 = train_count / max_batch;

// -- Evaluation result --------------------------------

const EvalResult = struct {
    correct: u32,
    total: u32,
    mean_loss: f32,
    accuracy_pct: f64,
};

// -- GPU buffer set for training + evaluation ---------

const Buffers = struct {
    input: Buffer,
    target: Buffer,
    loss_grad: Buffer,
    probs: Buffer,
    loss: Buffer,

    fn init(device: *const Device) !Buffers {
        var self: Buffers = undefined;

        self.input = try device.createBuffer(
            max_batch * MnistLayout.input_size,
        );
        errdefer self.input.deinit();

        self.target = try device.createBuffer(
            max_batch * num_classes,
        );
        errdefer self.target.deinit();

        self.loss_grad = try device.createBuffer(
            max_batch * num_classes,
        );
        errdefer self.loss_grad.deinit();

        self.probs = try device.createBuffer(
            max_batch * num_classes,
        );
        errdefer self.probs.deinit();

        self.loss = try device.createBuffer(max_batch);
        errdefer self.loss.deinit();

        return self;
    }

    fn deinit(self: *Buffers) void {
        self.loss.deinit();
        self.probs.deinit();
        self.loss_grad.deinit();
        self.target.deinit();
        self.input.deinit();
        self.* = undefined;
    }
};

// ====================================================
// Main
// ====================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print(
        "\nnnzap MNIST benchmark\n" ++
            "=====================\n\n",
        .{},
    );

    // -- Metal device --

    var device: Device = undefined;
    try device.init();
    std.debug.print(
        "Metal: unified_memory={}\n",
        .{device.unified_memory},
    );
    MnistLayout.printSummary();

    // -- Dataset --

    std.debug.print("Loading MNIST... ", .{});
    var mnist = try Mnist.load(allocator, "data/mnist");
    defer mnist.deinit(allocator);
    std.debug.print("done.\n", .{});

    // -- Network --

    var net: MnistNet = undefined;
    try net.init(&device);
    defer net.deinit();

    var prng = std.Random.DefaultPrng.init(seed);
    heInit(MnistLayout, net.paramSlice(), prng.random());

    // -- Buffers --

    var bufs = try Buffers.init(&device);
    defer bufs.deinit();

    // -- Train / validation split --
    //
    // Shuffle all 60k indices once with a deterministic seed,
    // then freeze the partition.  First 50k → training (re-
    // shuffled each epoch), last 10k → validation (fixed).

    const all_indices = try allocator.alloc(
        u32,
        total_train,
    );
    defer allocator.free(all_indices);
    fillSequential(all_indices);
    prng.random().shuffle(u32, all_indices);

    const train_indices = all_indices[0..train_count];
    const val_indices = all_indices[train_count..];

    std.debug.assert(val_indices.len == val_count);

    std.debug.print(
        "Split: train={d}  val={d}  test={d}\n\n",
        .{ train_count, val_count, test_count },
    );

    // -- Benchmark recorder --

    var bench: Benchmark = undefined;
    bench.init(MnistLayout, .{
        .batch_size = max_batch,
        .learning_rate = learning_rate,
        .optimizer = .sgd,
        .loss_function = .cross_entropy,
        .num_epochs = num_epochs,
        .seed = seed,
        .train_samples = train_count,
        .validation_samples = val_count,
        .test_samples = test_count,
    });

    // -- Training loop --

    std.debug.print(
        "Training: {d} epochs x {d} batches" ++
            "  (batch={d}, lr={d:.2})\n",
        .{
            num_epochs,
            batches_per_epoch,
            max_batch,
            learning_rate,
        },
    );
    printSeparator();

    const train_start = std.time.nanoTimestamp();

    var epoch: u32 = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        const epoch_start = std.time.nanoTimestamp();

        const train_loss = trainEpoch(
            &device,
            &net,
            &mnist,
            &bufs,
            train_indices,
            prng.random(),
        );

        // Validation evaluation (reuses the same GPU buffers).
        const val = evaluate(
            &device,
            &net,
            &bufs,
            mnist.train_images,
            mnist.train_labels,
            mnist.train_labels_raw,
            val_indices,
        );

        const epoch_ms = nanosToMs(
            std.time.nanoTimestamp() - epoch_start,
        );

        bench.recordEpoch(.{
            .epoch = epoch + 1,
            .train_loss = train_loss,
            .duration_ms = epoch_ms,
            .validation_loss = val.mean_loss,
            .validation_accuracy_pct = val.accuracy_pct,
        });

        std.debug.print(
            "  Epoch {d:>2}: loss={d:.4}" ++
                "  val_loss={d:.4}" ++
                "  val_acc={d:.2}%" ++
                "  ({d:.0} ms)\n",
            .{
                epoch + 1,
                train_loss,
                val.mean_loss,
                val.accuracy_pct,
                epoch_ms,
            },
        );
    }

    const train_ms = nanosToMs(
        std.time.nanoTimestamp() - train_start,
    );
    bench.setTrainingTime(train_ms);

    printSeparator();
    std.debug.print(
        "Total: {d:.0} ms  ({d:.0} ms/epoch," ++
            " {d:.0} img/s)\n\n",
        .{
            train_ms,
            train_ms / @as(f64, num_epochs),
            @as(f64, train_count) * num_epochs /
                (train_ms / 1000.0),
        },
    );

    // -- Test-set evaluation --

    var test_indices: [test_count]u32 = undefined;
    fillSequential(&test_indices);

    std.debug.print("Evaluating test set... ", .{});
    const eval_start = std.time.nanoTimestamp();

    const test_eval = evaluate(
        &device,
        &net,
        &bufs,
        mnist.test_images,
        mnist.test_labels,
        mnist.test_labels_raw,
        &test_indices,
    );

    const eval_ms = nanosToMs(
        std.time.nanoTimestamp() - eval_start,
    );

    bench.recordTest(test_eval.correct, test_eval.total, eval_ms);

    std.debug.print(
        "{d}/{d} correct ({d:.2}%)  [{d:.0} ms]\n\n",
        .{ test_eval.correct, test_eval.total, test_eval.accuracy_pct, eval_ms },
    );

    // -- Save benchmark results --

    try bench.save("mnist");
}

// ====================================================
// Training
// ====================================================

/// Run one full epoch of mini-batch SGD over the training
/// set.  Shuffles `indices` in-place (only the training
/// portion — validation indices are untouched).  Returns
/// the mean CE loss computed on the final training batch.
fn trainEpoch(
    device: *const Device,
    net: *const MnistNet,
    mnist: *const Mnist,
    bufs: *const Buffers,
    indices: []u32,
    random: std.Random,
) f32 {
    std.debug.assert(indices.len == train_count);
    std.debug.assert(batches_per_epoch > 0);

    // Fisher-Yates shuffle for this epoch.
    random.shuffle(u32, indices);

    var batch: u32 = 0;
    while (batch < batches_per_epoch) : (batch += 1) {
        const off: usize = @as(usize, batch) * max_batch;
        const idx = indices[off..][0..max_batch];

        Mnist.fillImageBatch(
            mnist.train_images,
            idx,
            bufs.input.asSlice(),
        );
        Mnist.fillLabelBatch(
            mnist.train_labels,
            idx,
            bufs.target.asSlice(),
        );

        const is_last = (batch == batches_per_epoch - 1);
        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);

        net.forward(device, enc, bufs.input, max_batch);

        // Loss reporting on final batch only.
        if (is_last) {
            encodeLoss(device, net, bufs, max_batch, enc);
        }

        MnistNet.encodeSoftmaxCEGrad(
            device,
            enc,
            net.getOutput().*,
            bufs.target,
            bufs.loss_grad,
            num_classes,
            max_batch,
        );

        net.backward(
            device,
            enc,
            bufs.input,
            bufs.loss_grad,
            max_batch,
        );

        net.update(device, enc, learning_rate);

        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }

    return sumSlice(bufs.loss.asSlice()) /
        @as(f32, max_batch);
}

/// Encode softmax → CE loss into an active encoder.
fn encodeLoss(
    device: *const Device,
    net: *const MnistNet,
    bufs: *const Buffers,
    batch_size: u32,
    enc: anytype,
) void {
    std.debug.assert(batch_size > 0);
    std.debug.assert(batch_size <= max_batch);

    MnistNet.encodeSoftmax(
        device,
        enc,
        net.getOutput().*,
        bufs.probs,
        num_classes,
        batch_size,
    );
    MnistNet.encodeCELoss(
        device,
        enc,
        bufs.probs,
        bufs.target,
        bufs.loss,
        num_classes,
        batch_size,
    );
}

// ====================================================
// Evaluation
// ====================================================

/// Evaluate accuracy and mean CE loss over an arbitrary
/// subset of samples specified by `indices`.  Handles
/// partial final batches.  Used for both validation and
/// test evaluation.
fn evaluate(
    device: *const Device,
    net: *const MnistNet,
    bufs: *const Buffers,
    images: []const f32,
    labels_onehot: []const f32,
    labels_raw: []const u8,
    indices: []const u32,
) EvalResult {
    const count: u32 = @intCast(indices.len);
    std.debug.assert(count > 0);
    std.debug.assert(images.len > 0);
    std.debug.assert(labels_raw.len > 0);

    var correct: u32 = 0;
    var loss_sum: f64 = 0.0;
    var offset: u32 = 0;

    while (offset < count) {
        const remaining = count - offset;
        const current: u32 = @min(max_batch, remaining);
        const idx = indices[offset..][0..current];

        Mnist.fillImageBatch(
            images,
            idx,
            bufs.input.asSlice(),
        );
        Mnist.fillLabelBatch(
            labels_onehot,
            idx,
            bufs.target.asSlice(),
        );

        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);
        net.forward(device, enc, bufs.input, current);
        encodeLoss(device, net, bufs, current, enc);
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);

        // Accumulate per-sample CE loss.
        for (bufs.loss.asSlice()[0..current]) |v| {
            loss_sum += @as(f64, v);
        }

        // CPU argmax over raw logits.
        correct += countCorrect(
            net.getOutput().asSlice(),
            labels_raw,
            idx,
            current,
        );

        offset += current;
    }

    const n: f64 = @floatFromInt(count);
    const c: f64 = @floatFromInt(correct);

    return .{
        .correct = correct,
        .total = count,
        .mean_loss = @floatCast(loss_sum / n),
        .accuracy_pct = c / n * 100.0,
    };
}

/// Count how many logit-argmax predictions match the raw
/// labels for the given batch indices.
fn countCorrect(
    logits: []const f32,
    labels_raw: []const u8,
    indices: []const u32,
    count: u32,
) u32 {
    std.debug.assert(count > 0);
    std.debug.assert(logits.len >= count * num_classes);

    var correct: u32 = 0;
    var s: u32 = 0;
    while (s < count) : (s += 1) {
        const base: usize = @as(usize, s) * num_classes;
        const row = logits[base..][0..num_classes];
        const pred = argmax(row);
        if (pred == labels_raw[indices[s]]) {
            correct += 1;
        }
    }
    return correct;
}

// ====================================================
// Helpers
// ====================================================

/// He uniform initialisation: w ~ U(-limit, limit),
/// limit = sqrt(6 / fan_in).  Biases start at zero.
fn heInit(
    comptime Layout: type,
    params: []f32,
    random: std.Random,
) void {
    std.debug.assert(params.len == Layout.param_count);
    std.debug.assert(Layout.num_layers > 0);

    inline for (0..Layout.num_layers) |li| {
        const fan_in: f32 = @floatFromInt(
            Layout.layers[li].in,
        );
        const limit = @sqrt(6.0 / fan_in);

        for (Layout.getWeightSlice(params, li)) |*w| {
            w.* = random.float(f32) * 2.0 * limit - limit;
        }
        @memset(Layout.getBiasSlice(params, li), 0.0);
    }
}

/// Index of the largest element in `row`.
fn argmax(row: []const f32) u8 {
    std.debug.assert(row.len > 0);
    std.debug.assert(row.len <= 255);

    var best: u8 = 0;
    var best_val: f32 = row[0];
    for (row[1..], 1..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best = @intCast(i);
        }
    }
    return best;
}

/// Sum every element of an f32 slice.
fn sumSlice(s: []const f32) f32 {
    std.debug.assert(s.len > 0);

    var total: f32 = 0.0;
    for (s) |v| {
        total += v;
    }
    return total;
}

/// Fill a slice with 0, 1, 2, …, n-1.
fn fillSequential(buf: []u32) void {
    std.debug.assert(buf.len > 0);

    for (buf, 0..) |*v, i| {
        v.* = @intCast(i);
    }
}

fn printSeparator() void {
    std.debug.print(
        "-------------------------------------------" ++
            "-----------\n",
        .{},
    );
}
