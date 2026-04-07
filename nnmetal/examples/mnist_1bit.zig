//! 1-bit MNIST integration test
//!
//! Validates the Q1_0_g128 quantization and 1-bit inference
//! pipeline end-to-end:
//!
//!   1. Train a 784→128→64→10 MLP for 3 epochs (f32 SGD).
//!   2. Evaluate f32 inference accuracy on 10k test images.
//!   3. Quantize weights to 1-bit via the f32_to_1bit kernel.
//!   4. Evaluate 1-bit inference accuracy on 10k test images.
//!   5. CPU reference: dequantize + matmul on one sample,
//!      compare with GPU 1-bit output (kernel correctness).
//!   6. Print side-by-side comparison.
//!
//! Expected: f32 ≈ 95%+, 1-bit lower (not trained for 1-bit).
//! CPU reference match confirms kernel correctness regardless
//! of accuracy.

const std = @import("std");
const nn = @import("nn");

const Mnist = nn.Mnist;
const Buffer = nn.Buffer;
const Device = nn.Device;
const PackedBuffer = nn.PackedBuffer;

// -- Network architecture (same as mnist.zig) ---------

const MnistLayout = nn.NetworkLayout(&.{
    .{ .in = 784, .out = 128, .act = .relu },
    .{ .in = 128, .out = 64, .act = .relu },
    .{ .in = 64, .out = 10, .act = .none },
});

const max_batch: u32 = 64;
const num_classes: u32 = MnistLayout.output_size;
const MnistNet = nn.Network(MnistLayout, max_batch);

// -- Training config ----------------------------------

const num_epochs: u32 = 3;
const learning_rate: f32 = 0.2;
const seed: u64 = 42;

// -- Dataset ------------------------------------------

const train_count: u32 = 50_000;
const test_count: u32 = 10_000;
const batches_per_epoch: u32 = train_count / max_batch;

// ============================================================
// Buffer wrappers
// ============================================================

/// Training-phase GPU buffers.  Input and target are filled
/// on the CPU before each forward pass; probs holds the
/// softmax output used by the CE gradient kernel.
const TrainBuffers = struct {
    input: Buffer,
    target: Buffer,
    loss_grad: Buffer,

    fn init(device: *const Device) !TrainBuffers {
        std.debug.assert(max_batch > 0);
        std.debug.assert(num_classes > 0);

        var input = try device.createBuffer(
            max_batch * MnistLayout.input_size,
        );
        errdefer input.deinit();

        var target = try device.createBuffer(
            max_batch * num_classes,
        );
        errdefer target.deinit();

        const loss_grad = try device.createBuffer(
            max_batch * num_classes,
        );

        return .{
            .input = input,
            .target = target,
            .loss_grad = loss_grad,
        };
    }

    fn deinit(self: *TrainBuffers) void {
        self.loss_grad.deinit();
        self.target.deinit();
        self.input.deinit();
        self.* = undefined;
    }
};

/// 1-bit packed weight set: one PackedBuffer per layer
/// plus a flat f32 bias buffer (biases are too small to
/// warrant quantisation — 202 floats total).
const PackedWeights = struct {
    layers: [MnistLayout.num_layers]PackedBuffer,
    bias: Buffer,

    fn init(device: *const Device) !PackedWeights {
        std.debug.assert(MnistLayout.num_layers > 0);
        std.debug.assert(MnistNet.total_bias_count > 0);

        var result: PackedWeights = undefined;
        var layer_count: u32 = 0;
        errdefer {
            var i: u32 = 0;
            while (i < layer_count) : (i += 1) {
                result.layers[i].deinit();
            }
        }

        inline for (0..MnistLayout.num_layers) |i| {
            result.layers[i] = try PackedBuffer.init(
                device.obj,
                MnistLayout.weight_counts[i],
                MnistLayout.PACK_GROUP_SIZE,
            );
            layer_count += 1;
        }

        result.bias = try device.createBuffer(
            MnistNet.total_bias_count,
        );

        return result;
    }

    fn deinit(self: *PackedWeights) void {
        self.bias.deinit();
        for (&self.layers) |*l| l.deinit();
        self.* = undefined;
    }
};

// ============================================================
// Main
// ============================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print(
        "\n1-bit MNIST integration test\n" ++
            "============================\n\n",
        .{},
    );

    var device: Device = undefined;
    try device.init();

    var mnist = try Mnist.load(allocator, "../data/mnist");
    defer mnist.deinit(allocator);

    var net: MnistNet = undefined;
    try net.init(&device);
    defer net.deinit();

    var prng = std.Random.DefaultPrng.init(seed);
    heInit(MnistLayout, net.paramSlice(), prng.random());

    var bufs = try TrainBuffers.init(&device);
    defer bufs.deinit();

    // Phase 1: train in f32.
    trainAll(&device, &net, &mnist, &bufs, prng.random());

    // Phase 2: f32 baseline.
    const f32_correct = evalF32(
        &device,
        &net,
        bufs.input,
        mnist.test_images,
        mnist.test_labels_raw,
    );

    // Phase 3: quantize to 1-bit.
    var packed_wts = try PackedWeights.init(&device);
    defer packed_wts.deinit();
    quantize(&device, &net, &packed_wts);

    // Phase 4: 1-bit evaluation.
    const q1_correct = eval1Bit(
        &device,
        &net,
        bufs.input,
        &packed_wts,
        mnist.test_images,
        mnist.test_labels_raw,
    );

    // Phase 5: CPU reference verification.
    try verifyCpuReference(
        allocator,
        &device,
        &net,
        &packed_wts,
        bufs.input,
        mnist.test_images,
    );

    // Phase 6: summary.
    printSummary(f32_correct, q1_correct);
}

// ============================================================
// Training
// ============================================================

/// Run all training epochs, printing progress.
fn trainAll(
    device: *const Device,
    net: *const MnistNet,
    mnist: *const Mnist,
    bufs: *const TrainBuffers,
    random: std.Random,
) void {
    std.debug.assert(batches_per_epoch > 0);
    std.debug.assert(num_epochs > 0);

    std.debug.print(
        "Training: {d} epochs x {d} batches " ++
            "(batch={d}, lr={d:.1})\n",
        .{
            num_epochs, batches_per_epoch,
            max_batch,  learning_rate,
        },
    );

    var indices: [train_count]u32 = undefined;
    fillSequential(&indices);

    const start = std.time.nanoTimestamp();
    for (0..num_epochs) |epoch| {
        random.shuffle(u32, &indices);
        trainEpoch(device, net, mnist, bufs, &indices);
        std.debug.print("  Epoch {d}  ({d:.0} ms)\n", .{
            epoch + 1,
            msFromNanos(
                std.time.nanoTimestamp() - start,
            ),
        });
    }
    std.debug.print("\n", .{});
}

/// One pass over the training set.  Each batch is a
/// separate command buffer (simple, not throughput-
/// optimal — this test cares about correctness).
fn trainEpoch(
    device: *const Device,
    net: *const MnistNet,
    mnist: *const Mnist,
    bufs: *const TrainBuffers,
    indices: []const u32,
) void {
    std.debug.assert(indices.len >= train_count);
    std.debug.assert(batches_per_epoch > 0);

    var batch: u32 = 0;
    while (batch < batches_per_epoch) : (batch += 1) {
        const off: usize = @as(usize, batch) * max_batch;
        fillBatch(mnist, indices[off..][0..max_batch], bufs);

        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);

        net.forward(device, enc, bufs.input, max_batch);

        // Fused softmax + cross-entropy gradient.
        // Outputs the loss gradient into loss_grad, which
        // backwardSGD reads to propagate through layers.
        MnistNet.encodeSoftmaxCEGrad(
            device,
            enc,
            net.getOutput().*,
            bufs.target,
            bufs.loss_grad,
            num_classes,
            max_batch,
        );

        net.backwardSGD(
            device,
            enc,
            bufs.input,
            bufs.loss_grad,
            max_batch,
            learning_rate,
        );

        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }
}

// ============================================================
// Evaluation
// ============================================================

/// Evaluate f32 inference accuracy on the full test set.
fn evalF32(
    device: *const Device,
    net: *const MnistNet,
    input_buf: Buffer,
    test_images: []const f32,
    labels_raw: []const u8,
) u32 {
    std.debug.assert(
        test_images.len >= @as(usize, test_count) * 784,
    );
    std.debug.assert(labels_raw.len >= test_count);

    std.debug.print("f32 inference...  ", .{});
    var correct: u32 = 0;
    var offset: u32 = 0;

    while (offset < test_count) {
        const bs: u32 = @min(
            max_batch,
            test_count - offset,
        );
        fillImages(test_images, offset, bs, input_buf);

        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);
        net.forwardInfer(device, enc, input_buf, bs);
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);

        correct += countCorrect(
            net.getOutput().asSlice(),
            labels_raw,
            offset,
            bs,
        );
        offset += bs;
    }

    printAccuracy(correct, test_count);
    return correct;
}

/// Evaluate 1-bit quantized inference on the full test set.
fn eval1Bit(
    device: *const Device,
    net: *const MnistNet,
    input_buf: Buffer,
    packed_wts: *const PackedWeights,
    test_images: []const f32,
    labels_raw: []const u8,
) u32 {
    std.debug.assert(
        test_images.len >= @as(usize, test_count) * 784,
    );
    std.debug.assert(labels_raw.len >= test_count);

    std.debug.print("1-bit inference... ", .{});
    var correct: u32 = 0;
    var offset: u32 = 0;

    while (offset < test_count) {
        const bs: u32 = @min(
            max_batch,
            test_count - offset,
        );
        fillImages(test_images, offset, bs, input_buf);

        const cmd = device.beginCommandBuffer();
        const enc = device.beginCompute(cmd);
        net.forwardInfer1Bit(
            device,
            enc,
            input_buf,
            &packed_wts.layers,
            packed_wts.bias,
            bs,
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);

        correct += countCorrect(
            net.getOutput().asSlice(),
            labels_raw,
            offset,
            bs,
        );
        offset += bs;
    }

    printAccuracy(correct, test_count);
    return correct;
}

// ============================================================
// Quantization
// ============================================================

/// Dispatch f32_to_1bit for every layer and copy biases.
fn quantize(
    device: *const Device,
    net: *const MnistNet,
    packed_wts: *PackedWeights,
) void {
    std.debug.assert(
        net.params.len == MnistLayout.param_count,
    );
    std.debug.assert(
        packed_wts.bias.len >= MnistNet.total_bias_count,
    );

    std.debug.print("Quantizing to 1-bit... ", .{});

    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    net.quantizeWeightsTo1Bit(
        device,
        enc,
        &packed_wts.layers,
        packed_wts.bias,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    std.debug.print("done.\n", .{});
}

// ============================================================
// CPU reference verification
// ============================================================

/// Run one test image through GPU 1-bit inference and a CPU
/// dequantize-then-matmul reference.  Compare the 10 output
/// logits.  Any mismatch beyond tolerance indicates a kernel
/// bug (independent of model accuracy).
fn verifyCpuReference(
    allocator: std.mem.Allocator,
    device: *const Device,
    net: *const MnistNet,
    packed_wts: *const PackedWeights,
    input_buf: Buffer,
    test_images: []const f32,
) !void {
    std.debug.assert(test_images.len >= 784);
    std.debug.assert(
        packed_wts.bias.len >= MnistNet.total_bias_count,
    );

    std.debug.print("CPU reference check... ", .{});

    // Copy first test image into the Metal input buffer.
    @memcpy(
        input_buf.asSlice()[0..784],
        test_images[0..784],
    );

    // GPU 1-bit inference on a single sample.
    const cmd = device.beginCommandBuffer();
    const enc = device.beginCompute(cmd);
    net.forwardInfer1Bit(
        device,
        enc,
        input_buf,
        &packed_wts.layers,
        packed_wts.bias,
        1,
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    const gpu_out = net.getOutput().asSlice();

    // CPU: dequantize + matmul for all three layers.
    const cpu_out = try cpuForward1Bit(
        allocator,
        packed_wts,
        test_images[0..784],
    );
    defer allocator.free(cpu_out);

    // The GPU tiled matmul accumulates in a different order
    // than the CPU sequential loop.  FP32 non-associativity
    // causes small differences — 0.05 is conservative.
    const tolerance: f32 = 0.05;
    for (0..num_classes) |i| {
        const diff = @abs(gpu_out[i] - cpu_out[i]);
        if (diff > tolerance) {
            std.debug.print(
                "\nMISMATCH [{d}]: GPU={d:.4}" ++
                    " CPU={d:.4} diff={d:.6}\n",
                .{ i, gpu_out[i], cpu_out[i], diff },
            );
            return error.CpuReferenceMismatch;
        }
    }
    std.debug.print("PASS\n", .{});
}

/// Full 3-layer CPU forward pass using dequantized 1-bit
/// weights.  Returns a heap-allocated 10-element slice
/// (caller frees).
fn cpuForward1Bit(
    allocator: std.mem.Allocator,
    packed_wts: *const PackedWeights,
    input: []const f32,
) ![]f32 {
    std.debug.assert(input.len >= 784);
    std.debug.assert(
        packed_wts.bias.len >= MnistNet.total_bias_count,
    );

    const bias = packed_wts.bias.asSlice();

    // Layer 0: 784 → 128 (relu).
    const w0 = try allocator.alloc(f32, 784 * 128);
    defer allocator.free(w0);
    dequantizeWeights(packed_wts.layers[0], w0);

    var a0: [128]f32 = undefined;
    cpuMatmulBiasAct(
        input[0..784],
        w0,
        bias[0..128],
        &a0,
        784,
        128,
        .relu,
    );

    // Layer 1: 128 → 64 (relu).
    const w1 = try allocator.alloc(f32, 128 * 64);
    defer allocator.free(w1);
    dequantizeWeights(packed_wts.layers[1], w1);

    var a1: [64]f32 = undefined;
    cpuMatmulBiasAct(
        &a0,
        w1,
        bias[128..192],
        &a1,
        128,
        64,
        .relu,
    );

    // Layer 2: 64 → 10 (none).
    const w2 = try allocator.alloc(f32, 64 * 10);
    defer allocator.free(w2);
    dequantizeWeights(packed_wts.layers[2], w2);

    const out = try allocator.alloc(f32, num_classes);
    errdefer allocator.free(out);
    cpuMatmulBiasAct(
        &a1,
        w2,
        bias[192..202],
        out,
        64,
        10,
        .none,
    );

    return out;
}

/// Read packed Q1_0_g128 bits and f16 scales from a Metal
/// shared buffer, reconstruct f32 weights on the CPU.
/// bit=1 → +scale, bit=0 → −scale.
fn dequantizeWeights(
    packed_buf: PackedBuffer,
    out: []f32,
) void {
    std.debug.assert(out.len >= packed_buf.packed_count);
    std.debug.assert(packed_buf.group_size > 0);

    // Raw byte pointer into the Metal shared buffer.
    const raw: [*]const u8 = @ptrCast(
        packed_buf.obj.msgSend(
            *anyopaque,
            "contents",
            .{},
        ),
    );

    const bit_bytes = packed_buf.packedBytes();
    const group_size = packed_buf.group_size;

    // f16 scales start right after the packed bits.
    // All MNIST layers produce even byte counts, so
    // 2-byte f16 alignment is satisfied.
    std.debug.assert(bit_bytes % 2 == 0);
    const scale_ptr: [*]const f16 = @ptrCast(
        @alignCast(raw + bit_bytes),
    );

    for (0..packed_buf.packed_count) |i| {
        const group: u32 = @intCast(i / group_size);
        const scale: f32 = @floatCast(scale_ptr[group]);
        const byte_idx = i / 8;
        const bit_pos: u3 = @intCast(i % 8);
        const bit = (raw[byte_idx] >> bit_pos) & 1;
        out[i] = if (bit == 1) scale else -scale;
    }
}

/// Single-sample CPU matmul + bias + activation.
/// W is [in_size × out_size] row-major (matching the
/// existing matmul convention where weights are stored
/// as [K × N]).
fn cpuMatmulBiasAct(
    input: []const f32,
    weights: []const f32,
    bias: []const f32,
    output: []f32,
    in_size: u32,
    out_size: u32,
    act: nn.Activation,
) void {
    std.debug.assert(input.len >= in_size);
    std.debug.assert(
        weights.len >= @as(usize, in_size) * out_size,
    );

    for (0..out_size) |o| {
        var sum: f32 = 0.0;
        for (0..in_size) |k| {
            // W is [K × N]: element (k, o) at index k * out_size + o.
            sum += weights[k * @as(usize, out_size) + o] * input[k];
        }
        sum += bias[o];
        output[o] = switch (act) {
            .relu => @max(0.0, sum),
            .none => sum,
            .tanh_act => 1.0 - 2.0 /
                (@exp(2.0 * sum) + 1.0),
            .sigmoid => 1.0 / (1.0 + @exp(-sum)),
        };
    }
}

// ============================================================
// Data helpers
// ============================================================

/// Copy a batch of training images and one-hot labels into
/// the Metal shared buffers used by the forward pass.
fn fillBatch(
    mnist: *const Mnist,
    indices: []const u32,
    bufs: *const TrainBuffers,
) void {
    std.debug.assert(indices.len > 0);
    std.debug.assert(indices.len <= max_batch);

    const inp = bufs.input.asSlice();
    const tgt = bufs.target.asSlice();
    for (indices, 0..) |idx, b| {
        const img_off: usize = @as(usize, idx) * 784;
        @memcpy(
            inp[b * 784 ..][0..784],
            mnist.train_images[img_off..][0..784],
        );

        const lbl_off: usize = @as(usize, idx) * 10;
        @memcpy(
            tgt[b * 10 ..][0..10],
            mnist.train_labels[lbl_off..][0..10],
        );
    }
}

/// Copy a contiguous range of test images into a Metal
/// shared buffer for inference.
fn fillImages(
    images: []const f32,
    start: u32,
    count: u32,
    buf: Buffer,
) void {
    std.debug.assert(count > 0);
    std.debug.assert(
        images.len >=
            (@as(usize, start) + count) * 784,
    );

    const dst = buf.asSlice();
    for (0..count) |b| {
        const idx: usize = @as(usize, start) + b;
        @memcpy(
            dst[b * 784 ..][0..784],
            images[idx * 784 ..][0..784],
        );
    }
}

/// Compare GPU output logits against ground-truth labels.
/// Returns the number of correct predictions in the batch.
fn countCorrect(
    logits: []const f32,
    labels_raw: []const u8,
    start: u32,
    count: u32,
) u32 {
    std.debug.assert(count > 0);
    std.debug.assert(
        logits.len >= @as(usize, count) * num_classes,
    );

    var correct: u32 = 0;
    for (0..count) |b| {
        const row = logits[b * 10 ..][0..10];
        const label_idx: usize = @as(usize, start) + b;
        if (argmax(row) == labels_raw[label_idx]) {
            correct += 1;
        }
    }
    return correct;
}

// ============================================================
// General helpers
// ============================================================

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

/// Index of the largest element in a logit row.
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

/// Fill a slice with 0, 1, 2, …, n-1.
fn fillSequential(buf: []u32) void {
    std.debug.assert(buf.len > 0);
    for (buf, 0..) |*v, i| {
        v.* = @intCast(i);
    }
}

fn msFromNanos(nanos: i128) f64 {
    return @as(f64, @floatFromInt(nanos)) / 1_000_000.0;
}

fn printAccuracy(correct: u32, total: u32) void {
    std.debug.assert(total > 0);

    const pct: f64 =
        @as(f64, @floatFromInt(correct)) /
        @as(f64, @floatFromInt(total)) * 100.0;
    std.debug.print(
        "{d}/{d} ({d:.2}%)\n",
        .{ correct, total, pct },
    );
}

fn printSummary(f32_correct: u32, q1_correct: u32) void {
    std.debug.assert(test_count > 0);

    const n: f64 = @floatFromInt(test_count);
    const f32_pct: f64 =
        @as(f64, @floatFromInt(f32_correct)) / n * 100.0;
    const q1_pct: f64 =
        @as(f64, @floatFromInt(q1_correct)) / n * 100.0;
    std.debug.print(
        "\n============================\n" ++
            "f32  accuracy: {d:.2}%\n" ++
            "1-bit accuracy: {d:.2}%\n" ++
            "delta:          {d:.2}pp\n" ++
            "============================\n\n",
        .{ f32_pct, q1_pct, f32_pct - q1_pct },
    );
}
