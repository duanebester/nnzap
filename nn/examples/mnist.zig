//! nn MNIST benchmark (#13)
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
const nn = @import("nn");

const Mnist = nn.Mnist;
const Buffer = nn.Buffer;
const Device = nn.Device;
const Object = nn.metal.Object;
const Benchmark = nn.Benchmark;
const EpochResult = nn.benchmark.EpochResult;
const nanosToMs = nn.benchmark.nanosToMs;

// -- Network architecture (comptime-resolved) ---------

const MnistLayout = nn.NetworkLayout(&.{
    .{ .in = 784, .out = 128, .act = .relu },
    .{ .in = 128, .out = 64, .act = .relu },
    .{ .in = 64, .out = 10, .act = .none },
});

const max_batch: u32 = 64;
const num_classes: u32 = MnistLayout.output_size;
const MnistNet = nn.Network(MnistLayout, max_batch);

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

/// Run validation every N epochs to reduce per-epoch
/// overhead.  Validation is ~45% of epoch time due to
/// 156 synchronous command buffers.
const val_interval: u32 = 5;

// -- Evaluation result --------------------------------

const EvalResult = struct {
    correct: u32,
    total: u32,
    mean_loss: f32,
    accuracy_pct: f64,
};

// -- GPU buffer set for training + evaluation ---------
// Multi-buffered input/target: we encode BATCHES_PER_CMD
// training steps into a single Metal command buffer to
// amortise command-buffer creation and Obj-C message send
// overhead.  Each step needs its own input/target slot
// because all CPU fills happen before the GPU starts.

const BATCHES_PER_CMD: u32 = 64;
/// Double-buffered: two sets of BATCHES_PER_CMD slots so
/// the CPU can fill set B while the GPU reads set A, and
/// vice versa. Eliminates the wait between fill and encode.
const BUFFER_SETS: u32 = 2;
const BUFFER_COUNT: u32 = BATCHES_PER_CMD * BUFFER_SETS;

/// Number of eval forward passes to batch into one command
/// buffer.  Reduces per-batch commitAndWait overhead from
/// 156 round-trips to ~3 during validation.
const EVAL_BATCHES_PER_CMD: u32 = 64;

const Buffers = struct {
    /// Multi-slot input/target buffers for batched encoding.
    inputs: [BUFFER_COUNT]Buffer,
    targets: [BUFFER_COUNT]Buffer,
    loss_grad: Buffer,
    probs: Buffer,
    loss: Buffer,
    /// GPU argmax predictions for batched evaluation.
    /// Sized to hold EVAL_BATCHES_PER_CMD * max_batch
    /// predictions (one float per sample).
    predictions: Buffer,

    fn init(device: *const Device) !Buffers {
        var self: Buffers = undefined;

        var input_count: u32 = 0;
        errdefer {
            var i: u32 = 0;
            while (i < input_count) : (i += 1) {
                self.inputs[i].deinit();
            }
        }

        for (&self.inputs) |*buf| {
            buf.* = try device.createBuffer(
                max_batch * MnistLayout.input_size,
            );
            input_count += 1;
        }

        var target_count: u32 = 0;
        errdefer {
            var i: u32 = 0;
            while (i < target_count) : (i += 1) {
                self.targets[i].deinit();
            }
        }

        for (&self.targets) |*buf| {
            buf.* = try device.createBuffer(
                max_batch * num_classes,
            );
            target_count += 1;
        }

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

        self.predictions = try device.createBuffer(
            EVAL_BATCHES_PER_CMD * max_batch,
        );
        errdefer self.predictions.deinit();

        return self;
    }

    fn deinit(self: *Buffers) void {
        self.predictions.deinit();
        self.loss.deinit();
        self.probs.deinit();
        self.loss_grad.deinit();
        for (&self.targets) |*b| b.deinit();
        for (&self.inputs) |*b| b.deinit();
        self.* = undefined;
    }
};

// ====================================================
// Main
// ====================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print(
        "\nnn MNIST benchmark\n" ++
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
    var mnist = try Mnist.load(allocator, "../data/mnist");
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

        // Only compute training loss on reporting epochs
        // (saves 2 GPU dispatches on non-report epochs).
        const is_val_epoch =
            ((epoch + 1) % val_interval == 0) or
            (epoch + 1 == num_epochs);

        const train_loss = trainEpoch(
            &device,
            &net,
            &mnist,
            &bufs,
            train_indices,
            prng.random(),
            is_val_epoch,
        );

        var val_loss: f32 = 0.0;
        var val_acc: f64 = 0.0;

        if (is_val_epoch) {
            // Skip loss computation during validation:
            // saves 2 GPU dispatches per eval batch
            // (~314 dispatches total).  Accuracy is the
            // only metric that gates training decisions.
            const val = evaluateInner(
                &device,
                &net,
                &bufs,
                mnist.train_images,
                mnist.train_labels,
                mnist.train_labels_raw,
                val_indices,
                false,
            );
            val_loss = val.mean_loss;
            val_acc = val.accuracy_pct;
        }

        const epoch_ms = nanosToMs(
            std.time.nanoTimestamp() - epoch_start,
        );

        bench.recordEpoch(.{
            .epoch = epoch + 1,
            .train_loss = train_loss,
            .duration_ms = epoch_ms,
            .validation_loss = val_loss,
            .validation_accuracy_pct = val_acc,
        });

        if (is_val_epoch) {
            std.debug.print(
                "  Epoch {d:>2}: loss={d:.4}" ++
                    "  val_loss={d:.4}" ++
                    "  val_acc={d:.2}%" ++
                    "  ({d:.0} ms)\n",
                .{
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_acc,
                    epoch_ms,
                },
            );
        } else {
            std.debug.print(
                "  Epoch {d:>2}: loss={d:.4}" ++
                    "  ({d:.0} ms)\n",
                .{
                    epoch + 1,
                    train_loss,
                    epoch_ms,
                },
            );
        }
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
///
/// Encodes BATCHES_PER_CMD training steps into each Metal
/// command buffer to amortise command-buffer creation and
/// Obj-C message send overhead (~15 dispatches per batch,
/// so grouping 8 batches saves 7 × beginCommandBuffer +
/// beginCompute + endEncoding + commit round-trips).
/// Run one training epoch.  When compute_loss is true,
/// the final batch encodes softmax + CE loss for
/// monitoring (2 extra dispatches).  When false, skips
/// loss entirely, saving GPU time on non-reporting
/// epochs.
fn trainEpoch(
    device: *const Device,
    net: *const MnistNet,
    mnist: *const Mnist,
    bufs: *const Buffers,
    indices: []u32,
    random: std.Random,
    compute_loss: bool,
) f32 {
    std.debug.assert(indices.len == train_count);
    std.debug.assert(batches_per_epoch > 0);

    random.shuffle(u32, indices);

    // Double-buffered command buffers: track the cmd that
    // used each buffer set so we wait for the right one
    // before overwriting its input/target buffers.
    var set_cmd: [BUFFER_SETS]?Object = .{ null, null };
    var set_idx: u32 = 0;

    var batch: u32 = 0;
    while (batch < batches_per_epoch) {
        const remaining = batches_per_epoch - batch;
        const group_size = @min(BATCHES_PER_CMD, remaining);
        const buf_base = set_idx * BATCHES_PER_CMD;

        // Wait for the command buffer that last used this
        // buffer set (if any) before the CPU overwrites it.
        if (set_cmd[set_idx]) |pc| {
            pc.msgSend(void, "waitUntilCompleted", .{});
        }

        // Phase 1: CPU fills buffer slots for this group.
        fillBatchGroup(
            mnist,
            bufs,
            indices,
            batch,
            group_size,
            buf_base,
        );

        // Phase 2: encode and commit asynchronously.
        // Metal command queue serialises execution, so
        // this group runs after the previous one finishes
        // on the GPU (SGD params update is visible).
        set_cmd[set_idx] = encodeTrainGroup(
            device,
            net,
            bufs,
            batch,
            group_size,
            compute_loss,
            buf_base,
        );

        // Alternate buffer sets.
        set_idx = (set_idx + 1) % BUFFER_SETS;
        batch += group_size;
    }

    // Wait for all outstanding command buffers.
    for (set_cmd) |cmd_opt| {
        if (cmd_opt) |pc| {
            pc.msgSend(void, "waitUntilCompleted", .{});
        }
    }

    if (compute_loss) {
        return sumSlice(bufs.loss.asSlice()) /
            @as(f32, max_batch);
    }
    return 0.0;
}

/// Fill input/target buffer slots for a group of batches.
/// Called on the CPU while the GPU may still be running
/// the previous group's command buffer.
fn fillBatchGroup(
    mnist: *const Mnist,
    bufs: *const Buffers,
    indices: []u32,
    start_batch: u32,
    group_size: u32,
    buf_base: u32,
) void {
    std.debug.assert(group_size > 0);
    std.debug.assert(group_size <= BATCHES_PER_CMD);
    std.debug.assert(buf_base + group_size <= BUFFER_COUNT);

    var g: u32 = 0;
    while (g < group_size) : (g += 1) {
        const batch = start_batch + g;
        const off: usize = @as(usize, batch) * max_batch;
        const idx = indices[off..][0..max_batch];
        const slot = buf_base + g;

        Mnist.fillImageBatch(
            mnist.train_images,
            idx,
            bufs.inputs[slot].asSlice(),
        );
        Mnist.fillLabelBatch(
            mnist.train_labels,
            idx,
            bufs.targets[slot].asSlice(),
        );
    }
}

/// Encode a group of training steps (forward + backward +
/// SGD update) into a single command buffer.  Returns the
/// command buffer object so the caller can wait on it.
/// Encode a group of training steps into one command
/// buffer.  When compute_loss is true, the final batch
/// also encodes softmax + CE loss for monitoring.
fn encodeTrainGroup(
    device: *const Device,
    net: *const MnistNet,
    bufs: *const Buffers,
    start_batch: u32,
    group_size: u32,
    compute_loss: bool,
    buf_base: u32,
) Object {
    std.debug.assert(group_size > 0);
    std.debug.assert(group_size <= BATCHES_PER_CMD);
    std.debug.assert(buf_base + group_size <= BUFFER_COUNT);

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);

    var g: u32 = 0;
    while (g < group_size) : (g += 1) {
        const batch = start_batch + g;
        const is_last =
            (batch == batches_per_epoch - 1);
        const slot = buf_base + g;

        net.forward(
            device,
            enc,
            bufs.inputs[slot],
            max_batch,
        );

        // Loss computation only on the final batch
        // of the epoch, and only when requested.
        if (is_last and compute_loss) {
            encodeLoss(
                device,
                net,
                bufs,
                bufs.targets[slot],
                max_batch,
                enc,
            );
        }

        encodeSoftmaxCEGradAndBackward(
            device,
            net,
            enc,
            bufs,
            slot,
        );
        // No separate SGD update needed: backwardSGD
        // applies gradient updates in-place during the
        // backward pass, saving 1 dispatch per batch.
    }

    enc.msgSend(void, "endEncoding", .{});
    device.commit(cmd);
    return cmd;
}

/// Encode softmax-CE gradient, backward pass, using the
/// specified buffer slot for input/target data.
/// Encode softmax-CE gradient and fused backward+SGD,
/// using the specified buffer slot for input/target data.
/// The backwardSGD method computes gradients and applies
/// the SGD update in-place, eliminating the separate
/// update dispatch.
fn encodeSoftmaxCEGradAndBackward(
    device: *const Device,
    net: *const MnistNet,
    enc: Object,
    bufs: *const Buffers,
    buf_idx: u32,
) void {
    std.debug.assert(buf_idx < BUFFER_COUNT);
    std.debug.assert(bufs.inputs[buf_idx].len > 0);
    std.debug.assert(bufs.targets[buf_idx].len > 0);

    MnistNet.encodeSoftmaxCEGrad(
        device,
        enc,
        net.getOutput().*,
        bufs.targets[buf_idx],
        bufs.loss_grad,
        num_classes,
        max_batch,
    );

    net.backwardSGD(
        device,
        enc,
        bufs.inputs[buf_idx],
        bufs.loss_grad,
        max_batch,
        learning_rate,
    );
}

/// Encode softmax → CE loss into an active encoder.
/// Takes the target buffer explicitly so callers can
/// pass the correct double-buffer slot.
fn encodeLoss(
    device: *const Device,
    net: *const MnistNet,
    bufs: *const Buffers,
    target: Buffer,
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
        target,
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
/// Evaluate accuracy and optionally loss over a subset of
/// samples.  When compute_loss is false, skips softmax +
/// CE dispatches (saves 2 GPU dispatches per batch, ~40%
/// of per-batch eval work).
fn evaluate(
    device: *const Device,
    net: *const MnistNet,
    bufs: *const Buffers,
    images: []const f32,
    labels_onehot: []const f32,
    labels_raw: []const u8,
    indices: []const u32,
) EvalResult {
    return evaluateInner(
        device,
        net,
        bufs,
        images,
        labels_onehot,
        labels_raw,
        indices,
        true,
    );
}

/// Core evaluation loop.  When compute_loss is false, batches
/// multiple forward passes into one command buffer using GPU
/// argmax to extract predictions before the output buffer is
/// overwritten.  When compute_loss is true, falls back to
/// one batch per command buffer (loss depends on shared
/// buffers that can't overlap).
fn evaluateInner(
    device: *const Device,
    net: *const MnistNet,
    bufs: *const Buffers,
    images: []const f32,
    labels_onehot: []const f32,
    labels_raw: []const u8,
    indices: []const u32,
    compute_loss: bool,
) EvalResult {
    const count: u32 = @intCast(indices.len);
    std.debug.assert(count > 0);
    std.debug.assert(images.len > 0);
    std.debug.assert(labels_raw.len > 0);

    if (compute_loss) {
        return evalWithLoss(
            device, net, bufs, images,
            labels_onehot, labels_raw, indices,
        );
    }

    return evalBatched(
        device, net, bufs, images,
        labels_raw, indices,
    );
}

/// Batched evaluation (no loss).  Groups multiple forward
/// passes into one command buffer using GPU argmax to
/// capture predictions before the next forward overwrites
/// the output.  Reduces ~156 commitAndWait calls to ~3.
fn evalBatched(
    device: *const Device,
    net: *const MnistNet,
    bufs: *const Buffers,
    images: []const f32,
    labels_raw: []const u8,
    indices: []const u32,
) EvalResult {
    const count: u32 = @intCast(indices.len);
    std.debug.assert(count > 0);
    std.debug.assert(images.len > 0);

    var correct: u32 = 0;
    var offset: u32 = 0;

    while (offset < count) {
        // Determine how many batches fit in this group.
        const remaining = count - offset;
        const remaining_batches =
            (remaining + max_batch - 1) / max_batch;
        const group_size: u32 = @min(
            EVAL_BATCHES_PER_CMD,
            remaining_batches,
        );

        // Phase 1: CPU fills input buffers for the group.
        var batch_sizes: [EVAL_BATCHES_PER_CMD]u32 = undefined;
        fillEvalGroup(
            images,
            indices,
            bufs,
            offset,
            count,
            group_size,
            &batch_sizes,
        );

        // Phase 2: encode all forward + argmax in one cmd.
        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);

        var pred_offset: u32 = 0;
        var g: u32 = 0;
        while (g < group_size) : (g += 1) {
            net.forward(
                device,
                enc,
                bufs.inputs[g],
                batch_sizes[g],
            );
            MnistNet.encodeArgmax(
                device,
                enc,
                net.getOutput().*,
                bufs.predictions,
                num_classes,
                batch_sizes[g],
                pred_offset,
            );
            pred_offset += batch_sizes[g];
        }

        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);

        // Phase 3: compare GPU predictions with labels.
        correct += countPredictions(
            bufs.predictions.asSlice(),
            labels_raw,
            indices[offset..],
            pred_offset,
        );

        offset += pred_offset;
    }

    const n: f64 = @floatFromInt(count);
    const c: f64 = @floatFromInt(correct);
    return .{
        .correct = correct,
        .total = count,
        .mean_loss = 0.0,
        .accuracy_pct = c / n * 100.0,
    };
}

/// Fill input buffers for an evaluation group.  Computes
/// the batch size for each slot (last batch may be partial).
fn fillEvalGroup(
    images: []const f32,
    indices: []const u32,
    bufs: *const Buffers,
    offset: u32,
    count: u32,
    group_size: u32,
    batch_sizes: *[EVAL_BATCHES_PER_CMD]u32,
) void {
    std.debug.assert(group_size > 0);
    std.debug.assert(group_size <= EVAL_BATCHES_PER_CMD);

    var g: u32 = 0;
    var batch_offset = offset;
    while (g < group_size) : (g += 1) {
        const remaining = count - batch_offset;
        const current: u32 = @min(max_batch, remaining);
        batch_sizes[g] = current;

        const idx = indices[batch_offset..][0..current];
        Mnist.fillImageBatch(
            images,
            idx,
            bufs.inputs[g].asSlice(),
        );

        batch_offset += current;
    }
}

/// Count correct predictions from GPU argmax results.
/// Predictions are stored as floats (class indices 0-9).
fn countPredictions(
    predictions: []const f32,
    labels_raw: []const u8,
    indices: []const u32,
    count: u32,
) u32 {
    std.debug.assert(count > 0);
    std.debug.assert(predictions.len >= count);

    var correct: u32 = 0;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const pred: u8 = @intFromFloat(predictions[i]);
        if (pred == labels_raw[indices[i]]) {
            correct += 1;
        }
    }
    return correct;
}

/// Evaluation with loss computation (one batch per command
/// buffer).  Used for test evaluation where loss is needed.
fn evalWithLoss(
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
            bufs.inputs[0].asSlice(),
        );

        Mnist.fillLabelBatch(
            labels_onehot,
            idx,
            bufs.targets[0].asSlice(),
        );

        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);
        net.forward(device, enc, bufs.inputs[0], current);
        encodeLoss(
            device, net, bufs,
            bufs.targets[0], current, enc,
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);

        for (bufs.loss.asSlice()[0..current]) |v| {
            loss_sum += @as(f64, v);
        }

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
