//! Forward pass for comptime-defined neural networks.
//!
//! Chains matmul → bias_add → activation for each layer, encoding all
//! operations into a single compute command encoder.  All buffer sizes
//! are comptime-known from the NetworkLayout; only batch_size varies
//! at runtime (bounded by a comptime maximum).

const std = @import("std");
const objc = @import("objc");
const metal = @import("metal.zig");
const layout = @import("layout.zig");

const Buffer = metal.Buffer;
const Device = metal.Device;

/// Absolute upper bound on batch size across all networks.
const MAX_BATCH_SIZE: u32 = 4096;

/// Generic forward-pass network parameterised by a comptime
/// NetworkLayout and a comptime maximum batch size.  All Metal
/// buffers are pre-allocated at init time; the forward pass
/// encodes GPU work without any allocation.
pub fn Network(
    comptime Layout: type,
    comptime max_batch_size: u32,
) type {
    comptime {
        std.debug.assert(Layout.num_layers > 0);
        std.debug.assert(max_batch_size > 0);
        std.debug.assert(max_batch_size <= MAX_BATCH_SIZE);
        // Scratch buffer size must not overflow u32.  Since
        // activation_sizes[i] <= max_activation_size for all i,
        // this single check covers every per-layer buffer too.
        std.debug.assert(
            max_batch_size <=
                std.math.maxInt(u32) /
                    Layout.max_activation_size,
        );
    }

    return struct {
        const Self = @This();

        /// Flat parameter buffer (weights + biases, all layers).
        params: Buffer,

        /// Per-layer pre-activation values (matmul + bias_add
        /// output, before activation).  Needed by the backward
        /// pass for activation gradients (e.g. relu_backward
        /// reads the pre-activation).  Unused when act == .none.
        pre_activations: [Layout.num_layers]Buffer,

        /// Per-layer post-activation values (after activation).
        /// post_activations[i] is the input to layer i+1.
        /// post_activations[num_layers - 1] is the network
        /// output.
        post_activations: [Layout.num_layers]Buffer,

        /// Scratch buffer for matmul output before bias_add.
        /// Reused every layer; sized to the largest layer output
        /// times max_batch_size.
        scratch: Buffer,

        /// Flat gradient buffer (same layout as params).
        /// Weight and bias gradients are written here by
        /// the backward pass — indexed with the same
        /// weight_offsets and bias_offsets as params.
        grads: Buffer,

        /// Two scratch buffers for backward-pass gradient
        /// propagation.  They ping-pong: one holds the
        /// upstream gradient while the other receives the
        /// input gradient for the next layer.
        backward_scratch: [2]Buffer,

        // Adam optimiser state (first and second moment buffers).
        // Allocated only when updateAdam is called; otherwise undefined.
        adam_m: Buffer,
        adam_v: Buffer,
        adam_initialised: bool,
        adam_timestep: u32,

        // ====================================================
        // Lifecycle
        // ====================================================

        /// Allocate all Metal buffers for the network.  Sizes
        /// are comptime-known; only the device handle is needed
        /// at runtime.
        pub fn init(
            self: *Self,
            device: *const Device,
        ) !void {
            std.debug.assert(Layout.param_count > 0);
            std.debug.assert(Layout.max_activation_size > 0);

            self.params = try device.createBuffer(
                Layout.param_count,
            );
            errdefer self.params.deinit();

            const scratch_size: u32 =
                max_batch_size * Layout.max_activation_size;

            self.scratch = try device.createBuffer(
                scratch_size,
            );
            errdefer self.scratch.deinit();

            self.grads = try device.createBuffer(
                Layout.param_count,
            );
            errdefer self.grads.deinit();

            self.backward_scratch[0] = try device.createBuffer(
                scratch_size,
            );
            errdefer self.backward_scratch[0].deinit();

            self.backward_scratch[1] = try device.createBuffer(
                scratch_size,
            );
            errdefer self.backward_scratch[1].deinit();

            self.adam_m = try device.createBuffer(
                Layout.param_count,
            );
            errdefer self.adam_m.deinit();
            @memset(self.adam_m.asSlice(), 0.0);

            self.adam_v = try device.createBuffer(
                Layout.param_count,
            );
            errdefer self.adam_v.deinit();
            @memset(self.adam_v.asSlice(), 0.0);

            self.adam_initialised = true;
            self.adam_timestep = 0;

            // Track how many activation buffers have been
            // allocated so errdefer can clean up the right
            // count on partial failure.

            var pre_act_count: u32 = 0;
            errdefer {
                var idx: u32 = 0;
                while (idx < pre_act_count) : (idx += 1) {
                    self.pre_activations[idx].deinit();
                }
            }

            var post_act_count: u32 = 0;
            errdefer {
                var idx: u32 = 0;
                while (idx < post_act_count) : (idx += 1) {
                    self.post_activations[idx].deinit();
                }
            }

            inline for (0..Layout.num_layers) |i| {
                const size: u32 = max_batch_size *
                    Layout.activation_sizes[i];

                self.pre_activations[i] =
                    try device.createBuffer(size);
                pre_act_count += 1;

                self.post_activations[i] =
                    try device.createBuffer(size);
                post_act_count += 1;
            }
        }

        /// Release all Metal buffers.
        pub fn deinit(self: *Self) void {
            // Struct must not already be undefined.
            std.debug.assert(self.params.len > 0);
            std.debug.assert(self.scratch.len > 0);

            if (self.adam_initialised) {
                self.adam_v.deinit();
                self.adam_m.deinit();
            }

            for (&self.post_activations) |*b| {
                b.deinit();
            }
            for (&self.pre_activations) |*b| {
                b.deinit();
            }
            for (&self.backward_scratch) |*b| {
                b.deinit();
            }
            self.grads.deinit();
            self.scratch.deinit();
            self.params.deinit();

            self.* = undefined;
        }

        // ====================================================
        // Accessors
        // ====================================================

        /// Return the flat parameter slice so the caller can
        /// fill weights and biases via Layout.getWeightSlice /
        /// Layout.getBiasSlice.
        pub fn paramSlice(self: *const Self) []f32 {
            std.debug.assert(self.params.len > 0);
            std.debug.assert(
                self.params.len == Layout.param_count,
            );
            return self.params.asSlice();
        }

        /// Return a pointer to the final output buffer
        /// (post-activation of the last layer).
        pub fn getOutput(
            self: *const Self,
        ) *const Buffer {
            const last = Layout.num_layers - 1;
            std.debug.assert(
                self.post_activations[last].len > 0,
            );
            std.debug.assert(
                self.post_activations[last].len >=
                    Layout.output_size,
            );
            return &self.post_activations[last];
        }

        /// Return the flat gradient slice so the caller can
        /// read gradients and apply parameter updates.
        pub fn gradSlice(self: *const Self) []f32 {
            std.debug.assert(self.grads.len > 0);
            std.debug.assert(
                self.grads.len == Layout.param_count,
            );
            return self.grads.asSlice();
        }

        // ====================================================
        // Forward pass
        // ====================================================

        /// Encode the full forward pass into the given compute
        /// encoder.  The caller owns the command buffer and
        /// must call endEncoding / commitAndWait afterwards.
        ///
        /// Uses fused matmul+bias+relu kernels to eliminate
        /// separate bias_add and activation dispatches —
        /// 1 dispatch per layer instead of 3.
        pub fn forward(
            self: *const Self,
            device: *const Device,
            encoder: objc.Object,
            input: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(batch_size <= max_batch_size);
            std.debug.assert(
                input.len >=
                    batch_size * Layout.input_size,
            );

            inline for (0..Layout.num_layers) |i| {
                // First layer reads the external input;
                // subsequent layers chain from the previous
                // post-activation.
                const layer_input: Buffer =
                    if (i == 0)
                        input
                    else
                        self.post_activations[i - 1];

                if (Layout.layers[i].act == .relu) {
                    // Fused matmul + bias + ReLU: one
                    // dispatch replaces three, saving two
                    // memory barriers and two kernel launches.
                    encodeMatmulBiasRelu(
                        Layout.layers[i].in,
                        Layout.layers[i].out,
                        Layout.weight_offsets[i],
                        Layout.bias_offsets[i],
                        device,
                        encoder,
                        layer_input,
                        self.params,
                        self.post_activations[i],
                        self.pre_activations[i],
                        batch_size,
                    );
                } else if (Layout.layers[i].act == .none) {
                    // Fused matmul + bias: one dispatch
                    // replaces two, saving one barrier.
                    encodeMatmulBias(
                        Layout.layers[i].in,
                        Layout.layers[i].out,
                        Layout.weight_offsets[i],
                        Layout.bias_offsets[i],
                        device,
                        encoder,
                        layer_input,
                        self.params,
                        self.post_activations[i],
                        batch_size,
                    );
                } else {
                    // Tanh/sigmoid: no fused kernel yet,
                    // fall back to separate dispatches.
                    encodeMatmulBias(
                        Layout.layers[i].in,
                        Layout.layers[i].out,
                        Layout.weight_offsets[i],
                        Layout.bias_offsets[i],
                        device,
                        encoder,
                        layer_input,
                        self.params,
                        self.pre_activations[i],
                        batch_size,
                    );
                    encodeActivation(
                        Layout.layers[i].act,
                        Layout.layers[i].out,
                        device,
                        encoder,
                        self.pre_activations[i],
                        self.post_activations[i],
                        batch_size,
                    );
                }
            }
        }

        // ====================================================
        // Backward pass
        // ====================================================

        /// Encode the full backward pass into the given
        /// compute encoder.  The caller owns the command
        /// buffer and must call endEncoding / commitAndWait
        /// afterwards.
        ///
        /// Writes weight and bias gradients into self.grads
        /// (same flat layout as self.params — use
        /// Layout.getWeightSlice / getBiasSlice to index).
        ///
        /// Requires a prior forward() call so that
        /// pre_activations and post_activations are
        /// populated.
        pub fn backward(
            self: *const Self,
            device: *const Device,
            encoder: objc.Object,
            input: Buffer,
            loss_grad: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(batch_size <= max_batch_size);
            std.debug.assert(
                loss_grad.len >=
                    batch_size * Layout.output_size,
            );
            std.debug.assert(
                input.len >=
                    batch_size * Layout.input_size,
            );

            // Track which backward_scratch buffer holds the
            // upstream gradient (from the layer above).  The
            // other buffer is free for writing.  The very
            // first iteration reads from loss_grad instead.
            comptime var active: u1 = 0;

            inline for (0..Layout.num_layers) |fwd_i| {
                const i = Layout.num_layers - 1 - fwd_i;
                const is_last =
                    (i == Layout.num_layers - 1);
                const act = Layout.layers[i].act;

                // Layer input (needed for weight gradient).
                const layer_input: Buffer =
                    if (i == 0)
                        input
                    else
                        self.post_activations[i - 1];

                if (act != .none) {
                    const target: u1 =
                        comptime if (is_last)
                            0
                        else
                            1 - active;

                    const grad_output: Buffer =
                        if (is_last)
                            loss_grad
                        else
                            self.backward_scratch[active];

                    encodeActivationBackward(
                        act,
                        Layout.layers[i].out,
                        device,
                        encoder,
                        self.pre_activations[i],
                        self.post_activations[i],
                        grad_output,
                        self.backward_scratch[target],
                        batch_size,
                    );

                    encodeBiasGrad(
                        Layout.layers[i].out,
                        Layout.bias_offsets[i],
                        device,
                        encoder,
                        self.backward_scratch[target],
                        self.grads,
                        batch_size,
                    );

                    encodeWeightGrad(
                        Layout.layers[i].in,
                        Layout.layers[i].out,
                        Layout.weight_offsets[i],
                        device,
                        encoder,
                        layer_input,
                        self.backward_scratch[target],
                        self.grads,
                        batch_size,
                    );

                    if (i > 0) {
                        const ig_target: u1 =
                            comptime if (is_last)
                                1
                            else
                                active;

                        encodeInputGrad(
                            Layout.layers[i].in,
                            Layout.layers[i].out,
                            Layout.weight_offsets[i],
                            device,
                            encoder,
                            self.backward_scratch[target],
                            self.params,
                            self.backward_scratch[
                                ig_target
                            ],
                            batch_size,
                        );

                        active = ig_target;
                    }
                } else {
                    // No activation: grad_pre_act IS the
                    // upstream grad_output (passthrough).
                    const grad_pre_act: Buffer =
                        if (is_last)
                            loss_grad
                        else
                            self.backward_scratch[active];

                    // -- Bias gradient --
                    encodeBiasGrad(
                        Layout.layers[i].out,
                        Layout.bias_offsets[i],
                        device,
                        encoder,
                        grad_pre_act,
                        self.grads,
                        batch_size,
                    );

                    // -- Weight gradient --
                    encodeWeightGrad(
                        Layout.layers[i].in,
                        Layout.layers[i].out,
                        Layout.weight_offsets[i],
                        device,
                        encoder,
                        layer_input,
                        grad_pre_act,
                        self.grads,
                        batch_size,
                    );

                    // -- Input gradient (skip for layer 0) --
                    if (i > 0) {
                        const ig_target: u1 = comptime if (is_last)
                            0
                        else
                            1 - active;

                        encodeInputGrad(
                            Layout.layers[i].in,
                            Layout.layers[i].out,
                            Layout.weight_offsets[i],
                            device,
                            encoder,
                            grad_pre_act,
                            self.params,
                            self.backward_scratch[
                                ig_target
                            ],
                            batch_size,
                        );

                        active = ig_target;
                    }
                }
            }

            // No barrier needed between backward and the
            // subsequent SGD update: Metal serialises
            // dispatches within a single compute encoder
            // on Apple Silicon. The forward pass relies on
            // this same guarantee (no barriers between
            // chained layer dispatches that share buffers).
        }

        // ====================================================
        // SGD parameter update
        // ====================================================

        /// Encode an SGD update step: params -= lr * grads.
        /// Dispatches the sgd_update kernel over the full
        /// parameter buffer.  Requires a prior backward()
        /// call so that grads are populated.
        pub fn update(
            self: *const Self,
            device: *const Device,
            encoder: objc.Object,
            learning_rate: f32,
        ) void {
            std.debug.assert(learning_rate > 0.0);
            std.debug.assert(
                self.params.len == Layout.param_count,
            );

            metal.setBuffer(encoder, self.params, 0);
            metal.setBuffer(encoder, self.grads, 1);
            metal.setBytes(
                encoder,
                f32,
                &learning_rate,
                2,
            );

            device.dispatch1D(
                encoder,
                device.sgd_update,
                Layout.param_count,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder. The next
            // forward pass reads params via the same encoder.
        }

        // ====================================================
        // Adam parameter update
        // ====================================================

        /// Adam optimiser step.
        ///
        /// Maintains per-parameter first and second moment
        /// estimates in `adam_m` and `adam_v` buffers.  Bias
        /// correction is precomputed on the CPU and passed
        /// to the kernel to avoid per-thread pow().
        pub fn updateAdam(
            self: *Self,
            device: *const Device,
            encoder: objc.Object,
            learning_rate: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
        ) void {
            std.debug.assert(learning_rate > 0.0);
            std.debug.assert(self.adam_initialised);
            std.debug.assert(beta1 > 0.0 and beta1 < 1.0);
            std.debug.assert(beta2 > 0.0 and beta2 < 1.0);
            std.debug.assert(epsilon > 0.0);

            self.adam_timestep += 1;
            const t: f32 =
                @floatFromInt(self.adam_timestep);

            // Precompute bias correction on CPU (avoids
            // pow() in every GPU thread).
            const correction1 = 1.0 - std.math.pow(
                f32,
                beta1,
                t,
            );
            const correction2 = 1.0 - std.math.pow(
                f32,
                beta2,
                t,
            );

            metal.setBuffer(encoder, self.params, 0);
            metal.setBuffer(encoder, self.grads, 1);
            metal.setBuffer(encoder, self.adam_m, 2);
            metal.setBuffer(encoder, self.adam_v, 3);
            metal.setBytes(encoder, f32, &learning_rate, 4);
            metal.setBytes(encoder, f32, &beta1, 5);
            metal.setBytes(encoder, f32, &beta2, 6);
            metal.setBytes(encoder, f32, &epsilon, 7);
            metal.setBytes(encoder, f32, &correction1, 8);
            metal.setBytes(encoder, f32, &correction2, 9);

            device.dispatch1D(
                encoder,
                device.adam_update,
                Layout.param_count,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder on Apple Silicon.
        }

        // ====================================================
        // Loss functions
        // ====================================================

        /// Encode per-element MSE loss:
        ///   out[i] = 0.5 * (pred[i] - target[i])².
        /// Dispatches over `count` elements.  The caller
        /// must sum the output buffer on the CPU (or with a
        /// reduction kernel) to get the scalar loss value.
        pub fn encodeMSELoss(
            device: *const Device,
            encoder: objc.Object,
            pred: Buffer,
            target: Buffer,
            output: Buffer,
            count: u32,
        ) void {
            std.debug.assert(count > 0);
            std.debug.assert(pred.len >= count);
            std.debug.assert(target.len >= count);
            std.debug.assert(output.len >= count);

            metal.setBuffer(encoder, pred, 0);
            metal.setBuffer(encoder, target, 1);
            metal.setBuffer(encoder, output, 2);

            device.dispatch1D(
                encoder,
                device.mse_forward,
                count,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder on Apple Silicon.
        }

        /// Encode per-element MSE gradient:
        ///   grad[i] = (pred[i] - target[i]) / batch_size.
        /// The division by batch_size produces the mean
        /// gradient, which is standard for mini-batch SGD.
        pub fn encodeMSEGrad(
            device: *const Device,
            encoder: objc.Object,
            pred: Buffer,
            target: Buffer,
            grad: Buffer,
            count: u32,
            batch_size: u32,
        ) void {
            std.debug.assert(count > 0);
            std.debug.assert(batch_size > 0);
            std.debug.assert(pred.len >= count);
            std.debug.assert(target.len >= count);
            std.debug.assert(grad.len >= count);

            metal.setBuffer(encoder, pred, 0);
            metal.setBuffer(encoder, target, 1);
            metal.setBuffer(encoder, grad, 2);
            metal.setBytes(
                encoder,
                u32,
                &batch_size,
                3,
            );

            device.dispatch1D(
                encoder,
                device.mse_backward,
                count,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder on Apple Silicon.
        }

        // ====================================================
        // Softmax + Cross-entropy
        // ====================================================

        /// Encode row-wise softmax: logits → probabilities.
        /// Dispatches one thread per sample.  Use this for
        /// inference when you need class probabilities.
        /// During training, prefer encodeSoftmaxCEGrad which
        /// fuses softmax into the backward pass.
        pub fn encodeSoftmax(
            device: *const Device,
            encoder: objc.Object,
            logits: Buffer,
            probs: Buffer,
            num_classes: u32,
            batch_size: u32,
        ) void {
            std.debug.assert(num_classes > 0);
            std.debug.assert(batch_size > 0);
            std.debug.assert(
                logits.len >= batch_size * num_classes,
            );
            std.debug.assert(
                probs.len >= batch_size * num_classes,
            );

            metal.setBuffer(encoder, logits, 0);
            metal.setBuffer(encoder, probs, 1);
            metal.setBytes(
                encoder,
                u32,
                &num_classes,
                2,
            );

            device.dispatch1D(
                encoder,
                device.softmax_forward,
                batch_size,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder on Apple Silicon.
        }

        /// Encode per-sample cross-entropy loss:
        ///   loss[s] = -sum_c( target[s*C+c] *
        ///              log(probs[s*C+c] + eps) )
        /// probs and target are [batch_size × num_classes].
        /// output is [batch_size] (one scalar per sample).
        /// The caller sums / averages on the CPU for
        /// logging.
        pub fn encodeCELoss(
            device: *const Device,
            encoder: objc.Object,
            probs: Buffer,
            target: Buffer,
            output: Buffer,
            num_classes: u32,
            batch_size: u32,
        ) void {
            std.debug.assert(num_classes > 0);
            std.debug.assert(batch_size > 0);
            std.debug.assert(
                probs.len >= batch_size * num_classes,
            );
            std.debug.assert(
                target.len >= batch_size * num_classes,
            );
            std.debug.assert(output.len >= batch_size);

            metal.setBuffer(encoder, probs, 0);
            metal.setBuffer(encoder, target, 1);
            metal.setBuffer(encoder, output, 2);
            metal.setBytes(
                encoder,
                u32,
                &num_classes,
                3,
            );

            device.dispatch1D(
                encoder,
                device.ce_forward,
                batch_size,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder on Apple Silicon.
        }

        /// Encode fused softmax + cross-entropy backward.
        /// Takes raw logits + one-hot targets, computes
        /// softmax internally, and produces the mean
        /// gradient w.r.t. logits:
        ///   grad[i] = (softmax[i] - target[i]) / batch_size
        ///
        /// The last layer should use act = .none so that
        /// backward() passes this gradient straight through
        /// to the weight/bias gradient computations.
        pub fn encodeSoftmaxCEGrad(
            device: *const Device,
            encoder: objc.Object,
            logits: Buffer,
            target: Buffer,
            grad: Buffer,
            num_classes: u32,
            batch_size: u32,
        ) void {
            std.debug.assert(num_classes > 0);
            std.debug.assert(batch_size > 0);
            std.debug.assert(
                logits.len >= batch_size * num_classes,
            );
            std.debug.assert(
                target.len >= batch_size * num_classes,
            );
            std.debug.assert(
                grad.len >= batch_size * num_classes,
            );

            metal.setBuffer(encoder, logits, 0);
            metal.setBuffer(encoder, target, 1);
            metal.setBuffer(encoder, grad, 2);
            metal.setBytes(
                encoder,
                u32,
                &num_classes,
                3,
            );
            metal.setBytes(
                encoder,
                u32,
                &batch_size,
                4,
            );

            device.dispatch1D(
                encoder,
                device.softmax_ce_backward,
                batch_size,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder. The backward
            // pass reads this gradient via the same encoder.
        }

        // ====================================================
        // Private dispatch helpers (Rule 20)
        // ====================================================

        /// Encode matmul: A[M×K] * B[K×N] → out[M×N].
        /// M = batch_size, K = in_size, N = out_size.
        fn encodeMatmul(
            comptime in_size: u32,
            comptime out_size: u32,
            comptime weight_offset: u32,
            device: *const Device,
            encoder: objc.Object,
            layer_input: Buffer,
            params: Buffer,
            scratch: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(params.len > weight_offset);

            metal.setBuffer(encoder, layer_input, 0);
            metal.setBufferWithOffset(
                encoder,
                params,
                weight_offset,
                1,
            );
            metal.setBuffer(encoder, scratch, 2);

            const m_val: u32 = batch_size;
            const k_val: u32 = in_size;
            const n_val: u32 = out_size;
            metal.setBytes(encoder, u32, &m_val, 3);
            metal.setBytes(encoder, u32, &k_val, 4);
            metal.setBytes(encoder, u32, &n_val, 5);

            // Width = N (columns), height = M (rows).
            device.dispatch2D(
                encoder,
                device.matmul_tiled,
                out_size,
                batch_size,
            );
            // Barrier managed by caller.
        }

        /// Encode bias_add: input[row] + bias[col] → output,
        /// broadcast across all rows.
        fn encodeBiasAdd(
            comptime out_size: u32,
            comptime bias_offset: u32,
            device: *const Device,
            encoder: objc.Object,
            scratch: Buffer,
            params: Buffer,
            output: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(params.len > bias_offset);

            metal.setBuffer(encoder, scratch, 0);
            metal.setBufferWithOffset(
                encoder,
                params,
                bias_offset,
                1,
            );
            metal.setBuffer(encoder, output, 2);

            const n_val: u32 = out_size;
            metal.setBytes(encoder, u32, &n_val, 3);

            // Width = N (columns), height = M (rows).
            device.dispatch2D(
                encoder,
                device.bias_add,
                out_size,
                batch_size,
            );
            // Barrier managed by caller.
        }

        /// Encode element-wise activation: pre_act → post_act.
        fn encodeActivation(
            comptime act: layout.Activation,
            comptime out_size: u32,
            device: *const Device,
            encoder: objc.Object,
            pre_act: Buffer,
            post_act: Buffer,
            batch_size: u32,
        ) void {
            comptime {
                std.debug.assert(act != .none);
            }
            std.debug.assert(batch_size > 0);

            const pipeline: metal.ComputePipeline =
                switch (act) {
                    .relu => device.relu_forward,
                    .tanh_act => device.tanh_forward,
                    .sigmoid => device.sigmoid_forward,
                    .none => unreachable,
                };

            metal.setBuffer(encoder, pre_act, 0);
            metal.setBuffer(encoder, post_act, 1);

            const count: u32 = out_size * batch_size;
            device.dispatch1D(encoder, pipeline, count);
            // Barrier managed by caller.
        }

        /// Encode fused matmul + bias + ReLU:
        ///   out = max(0, X * W + bias)
        /// Also stores pre-activation for the backward pass.
        /// Replaces encodeMatmul + encodeBiasAdd +
        /// encodeActivation with a single dispatch.
        fn encodeMatmulBiasRelu(
            comptime in_size: u32,
            comptime out_size: u32,
            comptime weight_offset: u32,
            comptime bias_offset: u32,
            device: *const Device,
            encoder: objc.Object,
            layer_input: Buffer,
            params: Buffer,
            post_act: Buffer,
            pre_act: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(params.len > weight_offset);
            std.debug.assert(params.len > bias_offset);

            // Buffer layout matches matmul_bias_relu kernel:
            //   buffer(0) = W [M×K], buffer(1) = x [K×N]
            // We follow the same convention as encodeMatmul:
            //   layer_input is the left matrix (M×K),
            //   weights are the right matrix (K×N).
            metal.setBuffer(encoder, layer_input, 0);
            metal.setBufferWithOffset(
                encoder,
                params,
                weight_offset,
                1,
            );
            metal.setBuffer(encoder, post_act, 2);

            const m_val: u32 = batch_size;
            const k_val: u32 = in_size;
            const n_val: u32 = out_size;
            metal.setBytes(encoder, u32, &m_val, 3);
            metal.setBytes(encoder, u32, &k_val, 4);
            metal.setBytes(encoder, u32, &n_val, 5);

            metal.setBufferWithOffset(
                encoder,
                params,
                bias_offset,
                6,
            );
            metal.setBuffer(encoder, pre_act, 7);

            device.dispatch2D(
                encoder,
                device.matmul_bias_relu,
                out_size,
                batch_size,
            );
            // Barrier managed by caller (forward/backward)
            // to allow batching independent dispatches.
        }

        /// Encode fused matmul + bias:
        ///   out = X * W + bias
        /// Replaces encodeMatmul + encodeBiasAdd with a
        /// single dispatch, saving one memory barrier.
        fn encodeMatmulBias(
            comptime in_size: u32,
            comptime out_size: u32,
            comptime weight_offset: u32,
            comptime bias_offset: u32,
            device: *const Device,
            encoder: objc.Object,
            layer_input: Buffer,
            params: Buffer,
            output: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(params.len > weight_offset);
            std.debug.assert(params.len > bias_offset);

            metal.setBuffer(encoder, layer_input, 0);
            metal.setBufferWithOffset(
                encoder,
                params,
                weight_offset,
                1,
            );
            metal.setBuffer(encoder, output, 2);

            const m_val: u32 = batch_size;
            const k_val: u32 = in_size;
            const n_val: u32 = out_size;
            metal.setBytes(encoder, u32, &m_val, 3);
            metal.setBytes(encoder, u32, &k_val, 4);
            metal.setBytes(encoder, u32, &n_val, 5);

            metal.setBufferWithOffset(
                encoder,
                params,
                bias_offset,
                6,
            );

            device.dispatch2D(
                encoder,
                device.matmul_bias,
                out_size,
                batch_size,
            );
            // Barrier managed by caller (forward/backward)
            // to allow batching independent dispatches.
        }

        // ====================================================
        // Backward dispatch helpers (Rule 20)
        // ====================================================

        /// Encode activation backward: compute gradient
        /// through the activation function.
        ///
        /// relu_backward reads the pre-activation (forward
        /// input).  tanh/sigmoid backward read the
        /// post-activation (forward output).
        fn encodeActivationBackward(
            comptime act: layout.Activation,
            comptime out_size: u32,
            device: *const Device,
            encoder: objc.Object,
            pre_act: Buffer,
            post_act: Buffer,
            grad_output: Buffer,
            grad_pre_act: Buffer,
            batch_size: u32,
        ) void {
            comptime {
                std.debug.assert(act != .none);
            }
            std.debug.assert(batch_size > 0);

            const pipeline: metal.ComputePipeline =
                switch (act) {
                    .relu => device.relu_backward,
                    .tanh_act => device.tanh_backward,
                    .sigmoid => device.sigmoid_backward,
                    .none => unreachable,
                };

            // relu_backward: buffer(0) = pre-activation.
            // tanh/sigmoid:  buffer(0) = post-activation.
            const act_input: Buffer = switch (act) {
                .relu => pre_act,
                .tanh_act, .sigmoid => post_act,
                .none => unreachable,
            };

            metal.setBuffer(encoder, act_input, 0);
            metal.setBuffer(encoder, grad_output, 1);
            metal.setBuffer(encoder, grad_pre_act, 2);

            const count: u32 = out_size * batch_size;
            device.dispatch1D(encoder, pipeline, count);
            // Barrier managed by caller (backward).
        }

        /// Encode bias gradient: column-wise sum of
        /// grad_pre_act [batch × out] → grads at
        /// bias_offset.  Each thread sums one column.
        fn encodeBiasGrad(
            comptime out_size: u32,
            comptime bias_offset: u32,
            device: *const Device,
            encoder: objc.Object,
            grad_pre_act: Buffer,
            grads: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(grads.len > bias_offset);

            metal.setBuffer(encoder, grad_pre_act, 0);
            metal.setBufferWithOffset(
                encoder,
                grads,
                bias_offset,
                1,
            );

            const m_val: u32 = batch_size;
            const n_val: u32 = out_size;
            metal.setBytes(encoder, u32, &m_val, 2);
            metal.setBytes(encoder, u32, &n_val, 3);

            device.dispatch1D(
                encoder,
                device.bias_grad,
                out_size,
            );
            // No barrier here: bias_grad writes to grads[bias],
            // which is only read later by the SGD update. The
            // caller (backward) adds a single barrier after all
            // layers to ensure grad visibility before the update.
        }

        /// Encode weight gradient: Xᵀ · grad_pre_act.
        /// X is [batch × in], grad is [batch × out],
        /// result is [in × out] written to grads at
        /// weight_offset.
        fn encodeWeightGrad(
            comptime in_size: u32,
            comptime out_size: u32,
            comptime weight_offset: u32,
            device: *const Device,
            encoder: objc.Object,
            layer_input: Buffer,
            grad_pre_act: Buffer,
            grads: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(grads.len > weight_offset);

            // matmul_transA: Aᵀ · B.
            // A = layer_input [M × K], B = grad [M × N].
            // Output = [K × N] = [in × out].
            metal.setBuffer(encoder, layer_input, 0);
            metal.setBuffer(encoder, grad_pre_act, 1);
            metal.setBufferWithOffset(
                encoder,
                grads,
                weight_offset,
                2,
            );

            const m_val: u32 = batch_size;
            const k_val: u32 = in_size;
            const n_val: u32 = out_size;
            metal.setBytes(encoder, u32, &m_val, 3);
            metal.setBytes(encoder, u32, &k_val, 4);
            metal.setBytes(encoder, u32, &n_val, 5);

            // Output is [in × out].
            device.dispatch2D(
                encoder,
                device.matmul_transA,
                out_size,
                in_size,
            );
            // No barrier here: weight_grad writes to
            // grads[weight], which is only read later by the
            // SGD update. The caller (backward) adds a single
            // barrier after all layers to ensure grad
            // visibility before the update.
        }

        /// Encode input gradient: grad_pre_act · Wᵀ.
        /// grad is [batch × out], W is [in × out],
        /// result is [batch × in].
        fn encodeInputGrad(
            comptime in_size: u32,
            comptime out_size: u32,
            comptime weight_offset: u32,
            device: *const Device,
            encoder: objc.Object,
            grad_pre_act: Buffer,
            params: Buffer,
            grad_input: Buffer,
            batch_size: u32,
        ) void {
            std.debug.assert(batch_size > 0);
            std.debug.assert(params.len > weight_offset);

            // matmul_transB: A · Bᵀ.
            // A = grad [M × K], B = W [N × K].
            // Output = [M × N] = [batch × in].
            metal.setBuffer(encoder, grad_pre_act, 0);
            metal.setBufferWithOffset(
                encoder,
                params,
                weight_offset,
                1,
            );
            metal.setBuffer(encoder, grad_input, 2);

            const m_val: u32 = batch_size;
            const k_val: u32 = out_size;
            const n_val: u32 = in_size;
            metal.setBytes(encoder, u32, &m_val, 3);
            metal.setBytes(encoder, u32, &k_val, 4);
            metal.setBytes(encoder, u32, &n_val, 5);

            // Output is [batch × in].
            device.dispatch2D(
                encoder,
                device.matmul_transB,
                in_size,
                batch_size,
            );
            // No barrier needed: Metal serialises dispatches
            // within a single compute encoder on Apple Silicon.
            // The forward pass relies on this same guarantee
            // (no barriers between chained layer dispatches).
        }
    };
}
