//! Model loader — Step 2c of the Bonsai implementation plan.
//!
//! Loads Bonsai 1.7B / 4B / 8B (Qwen3-style) transformer weights
//! from safetensors files into pre-allocated Metal GPU buffers.
//! All buffers are allocated once during `init`; `loadWeights` copies
//! tensor data from memory-mapped safetensors files into the existing
//! buffers without further allocation.
//!
//! MLX affine quantisation stores `(scale, bias)` per group.  At load
//! time we convert to symmetric encoding: `symmetric_scale = scale / 2`.
//! The bias tensor is discarded — the kernel uses symmetric dequant.

const std = @import("std");
const objc = @import("objc");
const metal = @import("metal.zig");
const layout = @import("layout.zig");
const transformer = @import("transformer.zig");
const safetensors = @import("safetensors.zig");

const log = std.log.scoped(.model);
const divCeil = layout.divCeil;

// -- Hard limits (Rule 4) --

const MAX_LAYERS: u32 = 64;
const MAX_SAFETENSORS_FILES: u32 = 16;
const MAX_TENSOR_NAME_LENGTH: u32 = 256;

// ============================================================================
// Model — owns all GPU resources for a transformer model
// ============================================================================

pub fn Model(comptime Config: type) type {
    return struct {
        const Self = @This();

        // Comptime validation — catch invalid configs at compile time
        // rather than discovering mismatched buffers at runtime.
        comptime {
            std.debug.assert(Config.num_layers > 0);
            std.debug.assert(Config.num_layers <= MAX_LAYERS);
            std.debug.assert(Config.hidden_size > 0);
            std.debug.assert(Config.vocab_size > 0);
            std.debug.assert(Config.group_size > 0);
            std.debug.assert(Config.intermediate_size > 0);
            std.debug.assert(Config.query_dim > 0);
            std.debug.assert(Config.kv_dim > 0);
            std.debug.assert(Config.head_dim > 0);
        }

        // -- Per-layer packed weight buffers (1-bit, 7 projections) --
        q_proj: [Config.num_layers]metal.PackedBuffer,
        k_proj: [Config.num_layers]metal.PackedBuffer,
        v_proj: [Config.num_layers]metal.PackedBuffer,
        o_proj: [Config.num_layers]metal.PackedBuffer,
        gate_proj: [Config.num_layers]metal.PackedBuffer,
        up_proj: [Config.num_layers]metal.PackedBuffer,
        down_proj: [Config.num_layers]metal.PackedBuffer,

        // -- Per-layer RMSNorm scales (f16) --
        attn_norm: [Config.num_layers]metal.HalfBuffer,
        ffn_norm: [Config.num_layers]metal.HalfBuffer,
        q_norm: [Config.num_layers]metal.HalfBuffer,
        k_norm: [Config.num_layers]metal.HalfBuffer,

        // -- Embedding (packed 1-bit) --
        embedding: metal.PackedBuffer,

        // -- LM head: separate when embeddings are not tied --
        lm_head: if (!Config.tie_word_embeddings)
            metal.PackedBuffer
        else
            void,

        // -- Final RMSNorm scale (f16) --
        final_norm: metal.HalfBuffer,

        // -- KV caches (f16, per-layer) --
        k_cache: [Config.num_layers]metal.HalfBuffer,
        v_cache: [Config.num_layers]metal.HalfBuffer,

        // -- Activation scratch buffers (shared across layers) --
        // f32: residual (precision-critical running sum),
        //      attn_scratch (softmax scores, device memory).
        // f16: norm_out, q, k, v, attn_out, proj_out, gate,
        //      up, mlp_out — match quantized projection I/O.
        residual: metal.Buffer,
        norm_out: metal.HalfBuffer,
        q: metal.HalfBuffer,
        k: metal.HalfBuffer,
        v: metal.HalfBuffer,
        attn_out: metal.HalfBuffer,
        proj_out: metal.HalfBuffer,
        attn_scratch: metal.Buffer,
        gate: metal.HalfBuffer,
        up: metal.HalfBuffer,
        mlp_out: metal.HalfBuffer,

        // -- Forward pass I/O (shared across layers) --
        logits: metal.Buffer, // [vocab_size] f32 — LM head output.
        token_ids: metal.Buffer, // [max_prefill_length] u32 as f32 slots.

        // GPU completion flag for spin-wait synchronization.
        // Allocated as a 1-element f32 buffer (same size as u32).
        completion_flag: metal.Buffer,

        // -- Device reference for downstream use --
        device_obj: objc.Object,

        // ----------------------------------------------------------------
        // Public API
        // ----------------------------------------------------------------

        /// Allocate all Metal buffers (zeroed).  The caller must call
        /// `deinit` to release them.  No further allocation occurs
        /// after init returns — `loadWeights` copies into existing
        /// buffers.
        pub fn init(self: *Self, device_obj: objc.Object) !void {
            comptime {
                std.debug.assert(Config.vocab_size > 0);
                std.debug.assert(Config.num_layers <= MAX_LAYERS);
            }

            self.device_obj = device_obj;

            // Embedding: packed 1-bit with per-group f16 scales.
            self.embedding = try metal.PackedBuffer.init(
                device_obj,
                Config.vocab_size * Config.hidden_size,
                Config.group_size,
            );
            errdefer self.embedding.deinit();

            // LM head: separate allocation only when not tied.
            if (!Config.tie_word_embeddings) {
                self.lm_head = try metal.PackedBuffer.init(
                    device_obj,
                    Config.hidden_size * Config.vocab_size,
                    Config.group_size,
                );
            }
            errdefer {
                if (!Config.tie_word_embeddings) {
                    self.lm_head.deinit();
                }
            }

            // Final RMSNorm scale (f16, hidden_size elements).
            self.final_norm = try metal.HalfBuffer.init(
                device_obj,
                Config.hidden_size,
            );
            errdefer self.final_norm.deinit();

            // Per-layer weights, norms, and KV caches.
            try self.allocateLayerBuffers(device_obj);
            errdefer self.deinitAllLayerBuffers();

            // Decode activation scratch (f32).
            try self.allocateActivationBuffers(device_obj);
        }

        /// Release all Metal buffers and invalidate the model.
        pub fn deinit(self: *Self) void {
            comptime {
                std.debug.assert(Config.num_layers > 0);
                std.debug.assert(Config.num_layers <= MAX_LAYERS);
            }

            // Activation scratch.
            self.deinitActivationBuffers();

            // Per-layer buffers (forward order is fine — Metal
            // buffers are independent reference-counted objects).
            self.deinitAllLayerBuffers();

            // Model-level buffers.
            self.final_norm.deinit();
            if (!Config.tie_word_embeddings) {
                self.lm_head.deinit();
            }
            self.embedding.deinit();
            self.* = undefined;
        }

        /// Load weights from one or more safetensors files into the
        /// pre-allocated Metal buffers.  Each file is opened, memory-
        /// mapped, and closed within this function — no file handles
        /// leak.  The page allocator is used for the file descriptor
        /// array; this is init-time only (Rule 2).
        pub fn loadWeights(
            self: *Self,
            file_paths: []const []const u8,
        ) !void {
            std.debug.assert(file_paths.len > 0);
            std.debug.assert(file_paths.len <= MAX_SAFETENSORS_FILES);

            // Heap-allocate the SafetensorsFile array — each is ~50 KB
            // due to the [1024]TensorDescriptor array, too large to
            // stack-allocate multiple copies.
            const page_alloc = std.heap.page_allocator;

            const files = try page_alloc.alloc(
                safetensors.SafetensorsFile,
                file_paths.len,
            );
            defer page_alloc.free(files);

            // Open all files before loading so that sharded tensors
            // can be found across any file.
            var count_opened: u32 = 0;
            defer {
                for (files[0..count_opened]) |*f| {
                    f.deinit();
                }
            }

            for (file_paths) |path| {
                try files[count_opened].init(path);
                count_opened += 1;
                log.info("opened safetensors shard: {s}", .{path});
            }

            const opened = files[0..count_opened];
            try self.loadEmbeddingWeights(opened);
            try self.loadFinalNormWeights(opened);
            if (!Config.tie_word_embeddings) {
                try self.loadLmHeadWeights(opened);
            }
            for (0..Config.num_layers) |i| {
                try self.loadLayerWeights(@intCast(i), opened);
            }
            log.info(
                "loaded all weights ({d} layers)",
                .{Config.num_layers},
            );
        }

        /// Construct a `ForwardBlockArgs` for one decoder block by
        /// indexing into the per-layer arrays.  Activation scratch
        /// buffers are shared across all layers (single-token decode).
        pub fn forwardBlockArgs(
            self: *const Self,
            layer_index: u32,
            position: u32,
            seq_len: u32,
        ) transformer.ForwardBlockArgs {
            std.debug.assert(layer_index < Config.num_layers);
            std.debug.assert(
                position < Config.max_context_length,
            );
            std.debug.assert(seq_len > 0);
            std.debug.assert(
                seq_len <= Config.max_context_length,
            );

            const i = layer_index;
            return .{
                .q_proj = self.q_proj[i],
                .k_proj = self.k_proj[i],
                .v_proj = self.v_proj[i],
                .o_proj = self.o_proj[i],
                .gate_proj = self.gate_proj[i],
                .up_proj = self.up_proj[i],
                .down_proj = self.down_proj[i],
                .attn_norm_scale = self.attn_norm[i].obj,
                .ffn_norm_scale = self.ffn_norm[i].obj,
                .q_norm_scale = self.q_norm[i].obj,
                .k_norm_scale = self.k_norm[i].obj,
                .residual = self.residual.obj,
                .norm_out = self.norm_out.obj,
                .q = self.q.obj,
                .k = self.k.obj,
                .v = self.v.obj,
                .attn_out = self.attn_out.obj,
                .proj_out = self.proj_out.obj,
                .attn_scratch = self.attn_scratch.obj,
                .gate = self.gate.obj,
                .up = self.up.obj,
                .mlp_out = self.mlp_out.obj,
                .k_cache = self.k_cache[i].obj,
                .v_cache = self.v_cache[i].obj,
                .position = position,
                .seq_len = seq_len,
            };
        }

        /// Construct a `ForwardDecodeArgs` for a full single-token
        /// decode pass.  Borrows slices of the per-layer arrays and
        /// all activation scratch buffers.  The LM head resolves to
        /// the embedding PackedBuffer when `tie_word_embeddings` is
        /// true, or the separate `lm_head` buffer otherwise.
        pub fn forwardDecodeArgs(
            self: *const Self,
            token_id: u32,
            position: u32,
        ) transformer.ForwardDecodeArgs {
            std.debug.assert(
                position < Config.max_context_length,
            );
            std.debug.assert(token_id < Config.vocab_size);

            const lm_head_buf: metal.PackedBuffer =
                if (Config.tie_word_embeddings)
                    self.embedding
                else
                    self.lm_head;

            return .{
                .embedding = self.embedding,
                .lm_head = lm_head_buf,
                .final_norm_scale = self.final_norm.obj,
                .q_proj = &self.q_proj,
                .k_proj = &self.k_proj,
                .v_proj = &self.v_proj,
                .o_proj = &self.o_proj,
                .gate_proj = &self.gate_proj,
                .up_proj = &self.up_proj,
                .down_proj = &self.down_proj,
                .attn_norm = &self.attn_norm,
                .ffn_norm = &self.ffn_norm,
                .q_norm = &self.q_norm,
                .k_norm = &self.k_norm,
                .k_cache = &self.k_cache,
                .v_cache = &self.v_cache,
                .residual = self.residual.obj,
                .norm_out = self.norm_out.obj,
                .q = self.q.obj,
                .k = self.k.obj,
                .v = self.v.obj,
                .attn_out = self.attn_out.obj,
                .proj_out = self.proj_out.obj,
                .attn_scratch = self.attn_scratch.obj,
                .gate = self.gate.obj,
                .up = self.up.obj,
                .mlp_out = self.mlp_out.obj,
                .token_ids = self.token_ids.obj,
                .logits = self.logits.obj,
                .flag_buf = self.completion_flag.obj,
                .flag_ptr = blk: {
                    const slice = self.completion_flag.asSlice();
                    break :blk @ptrCast(&slice[0]);
                },
                .token_id = token_id,
                .position = position,
            };
        }

        // ----------------------------------------------------------------
        // Private allocation helpers
        // ----------------------------------------------------------------

        /// Allocate all per-layer buffers (weights, norms, KV caches).
        /// Tracks progress so that partially-allocated layers are
        /// cleaned up on failure.
        fn allocateLayerBuffers(
            self: *Self,
            dev: objc.Object,
        ) !void {
            std.debug.assert(Config.num_layers > 0);
            std.debug.assert(Config.num_layers <= MAX_LAYERS);

            var count_allocated: u32 = 0;
            errdefer {
                // Clean up fully-allocated layers on partial failure.
                var j: u32 = count_allocated;
                while (j > 0) {
                    j -= 1;
                    self.deinitOneLayer(j);
                }
            }

            for (0..Config.num_layers) |i| {
                try self.allocateOneLayer(dev, @intCast(i));
                count_allocated += 1;
            }
        }

        /// Allocate all 13 buffers for a single decoder layer.
        /// Internal errdefer chain handles partial failure within
        /// the layer.
        fn allocateOneLayer(
            self: *Self,
            dev: objc.Object,
            idx: u32,
        ) !void {
            std.debug.assert(idx < Config.num_layers);
            // Aliases for readability (matches test convention).
            const H = Config.hidden_size;
            const QD = Config.query_dim;
            const KVD = Config.kv_dim;
            const I = Config.intermediate_size;
            const GS = Config.group_size;
            const HD = Config.head_dim;
            const kv_cache_len: u32 =
                Config.max_context_length * KVD;

            // Attention projections.
            self.q_proj[idx] =
                try metal.PackedBuffer.init(dev, H * QD, GS);
            errdefer self.q_proj[idx].deinit();
            self.k_proj[idx] =
                try metal.PackedBuffer.init(dev, H * KVD, GS);
            errdefer self.k_proj[idx].deinit();
            self.v_proj[idx] =
                try metal.PackedBuffer.init(dev, H * KVD, GS);
            errdefer self.v_proj[idx].deinit();
            self.o_proj[idx] =
                try metal.PackedBuffer.init(dev, QD * H, GS);
            errdefer self.o_proj[idx].deinit();

            // MLP projections.
            self.gate_proj[idx] =
                try metal.PackedBuffer.init(dev, H * I, GS);
            errdefer self.gate_proj[idx].deinit();
            self.up_proj[idx] =
                try metal.PackedBuffer.init(dev, H * I, GS);
            errdefer self.up_proj[idx].deinit();
            self.down_proj[idx] =
                try metal.PackedBuffer.init(dev, I * H, GS);
            errdefer self.down_proj[idx].deinit();

            // RMSNorm scales (f16).
            self.attn_norm[idx] =
                try metal.HalfBuffer.init(dev, H);
            errdefer self.attn_norm[idx].deinit();
            self.ffn_norm[idx] =
                try metal.HalfBuffer.init(dev, H);
            errdefer self.ffn_norm[idx].deinit();
            self.q_norm[idx] =
                try metal.HalfBuffer.init(dev, HD);
            errdefer self.q_norm[idx].deinit();
            self.k_norm[idx] =
                try metal.HalfBuffer.init(dev, HD);
            errdefer self.k_norm[idx].deinit();

            // KV caches (f16, max_context_length × kv_dim).
            self.k_cache[idx] =
                try metal.HalfBuffer.init(dev, kv_cache_len);
            errdefer self.k_cache[idx].deinit();
            self.v_cache[idx] =
                try metal.HalfBuffer.init(dev, kv_cache_len);
        }

        /// Allocate the 11 f32 activation scratch buffers used
        /// during single-token decode.  Each buffer is sized for
        /// one token; prefill will require separate handling.
        fn allocateActivationBuffers(
            self: *Self,
            dev: objc.Object,
        ) !void {
            std.debug.assert(Config.hidden_size > 0);
            std.debug.assert(Config.intermediate_size > 0);
            const H = Config.hidden_size;
            const QD = Config.query_dim;
            const KVD = Config.kv_dim;
            const I = Config.intermediate_size;

            self.residual = try metal.Buffer.init(dev, H);
            errdefer self.residual.deinit();
            self.norm_out = try metal.HalfBuffer.init(dev, H);
            errdefer self.norm_out.deinit();
            self.q = try metal.HalfBuffer.init(dev, QD);
            errdefer self.q.deinit();
            self.k = try metal.HalfBuffer.init(dev, KVD);
            errdefer self.k.deinit();
            self.v = try metal.HalfBuffer.init(dev, KVD);
            errdefer self.v.deinit();
            self.attn_out = try metal.HalfBuffer.init(dev, QD);
            errdefer self.attn_out.deinit();
            self.proj_out = try metal.HalfBuffer.init(dev, H);
            errdefer self.proj_out.deinit();

            const scratch_len: u32 =
                Config.num_query_heads * Config.max_context_length;
            self.attn_scratch =
                try metal.Buffer.init(dev, scratch_len);
            errdefer self.attn_scratch.deinit();

            self.gate = try metal.HalfBuffer.init(dev, I);
            errdefer self.gate.deinit();
            self.up = try metal.HalfBuffer.init(dev, I);
            errdefer self.up.deinit();
            self.mlp_out = try metal.HalfBuffer.init(dev, I);
            errdefer self.mlp_out.deinit();

            // Logits output: [vocab_size] f32.
            self.logits = try metal.Buffer.init(
                dev,
                Config.vocab_size,
            );
            errdefer self.logits.deinit();

            // Token IDs input: reuses f32 buffer for u32 storage
            // (same element size).  Pre-allocated at prefill
            // capacity so no allocation occurs after init.
            self.token_ids = try metal.Buffer.init(
                dev,
                Config.max_prefill_length,
            );
            errdefer self.token_ids.deinit();

            // Completion flag: 1 element for GPU→CPU spin-wait.
            self.completion_flag = try metal.Buffer.init(
                dev,
                1,
            );
            self.completion_flag.asSlice()[0] = 0;
        }

        // ----------------------------------------------------------------
        // Private deallocation helpers
        // ----------------------------------------------------------------

        /// Release all 13 buffers for one decoder layer.
        fn deinitOneLayer(self: *Self, idx: u32) void {
            std.debug.assert(idx < Config.num_layers);
            self.v_cache[idx].deinit();
            self.k_cache[idx].deinit();
            self.k_norm[idx].deinit();
            self.q_norm[idx].deinit();
            self.ffn_norm[idx].deinit();
            self.attn_norm[idx].deinit();
            self.down_proj[idx].deinit();
            self.up_proj[idx].deinit();
            self.gate_proj[idx].deinit();
            self.o_proj[idx].deinit();
            self.v_proj[idx].deinit();
            self.k_proj[idx].deinit();
            self.q_proj[idx].deinit();
        }

        /// Release buffers for all layers.
        fn deinitAllLayerBuffers(self: *Self) void {
            comptime std.debug.assert(Config.num_layers > 0);
            for (0..Config.num_layers) |i| {
                self.deinitOneLayer(@intCast(i));
            }
        }

        /// Release all activation scratch buffers.
        fn deinitActivationBuffers(self: *Self) void {
            std.debug.assert(self.residual.len > 0);
            std.debug.assert(self.mlp_out.len > 0);
            std.debug.assert(self.logits.len > 0);
            self.completion_flag.deinit();
            self.token_ids.deinit();
            self.logits.deinit();
            self.mlp_out.deinit();
            self.up.deinit();
            self.gate.deinit();
            self.attn_scratch.deinit();
            self.proj_out.deinit();
            self.attn_out.deinit();
            self.v.deinit();
            self.k.deinit();
            self.q.deinit();
            self.norm_out.deinit();
            self.residual.deinit();
        }

        // ----------------------------------------------------------------
        // Private weight-loading helpers (methods — need self for buffers)
        // ----------------------------------------------------------------

        /// Load embedding packed bits and converted scales.
        fn loadEmbeddingWeights(
            self: *Self,
            files: []const safetensors.SafetensorsFile,
        ) !void {
            std.debug.assert(files.len > 0);
            std.debug.assert(
                self.embedding.packed_count ==
                    Config.vocab_size * Config.hidden_size,
            );

            const weight = try findTensorInFiles(
                files,
                "model.embed_tokens.weight",
            );
            const scales = try findTensorInFiles(
                files,
                "model.embed_tokens.scales",
            );
            loadPackedProjection(&self.embedding, weight, scales);
            log.info("loaded embedding", .{});
        }

        /// Load final RMSNorm scale (f32 → f16 conversion).
        fn loadFinalNormWeights(
            self: *Self,
            files: []const safetensors.SafetensorsFile,
        ) !void {
            std.debug.assert(files.len > 0);
            std.debug.assert(
                self.final_norm.len == Config.hidden_size,
            );

            const desc = try findTensorInFiles(
                files,
                "model.norm.weight",
            );
            loadNormScale(&self.final_norm, desc);
            log.info("loaded final norm", .{});
        }

        /// Load separate LM head (only when !tie_word_embeddings).
        fn loadLmHeadWeights(
            self: *Self,
            files: []const safetensors.SafetensorsFile,
        ) !void {
            comptime std.debug.assert(
                !Config.tie_word_embeddings,
            );
            std.debug.assert(files.len > 0);

            const weight = try findTensorInFiles(
                files,
                "lm_head.weight",
            );
            const scales = try findTensorInFiles(
                files,
                "lm_head.scales",
            );
            loadPackedProjection(&self.lm_head, weight, scales);
            log.info("loaded lm_head", .{});
        }

        /// Load all 11 tensors for one decoder layer: 7 packed
        /// projections and 4 RMSNorm scales.
        fn loadLayerWeights(
            self: *Self,
            layer_index: u32,
            files: []const safetensors.SafetensorsFile,
        ) !void {
            std.debug.assert(layer_index < Config.num_layers);
            std.debug.assert(files.len > 0);
            var name_buf: [MAX_TENSOR_NAME_LENGTH]u8 = undefined;
            const idx = layer_index;

            // Attention projections.
            try loadPackedFromFiles(
                &self.q_proj[idx],
                files,
                &name_buf,
                idx,
                "self_attn.q_proj",
            );
            try loadPackedFromFiles(
                &self.k_proj[idx],
                files,
                &name_buf,
                idx,
                "self_attn.k_proj",
            );
            try loadPackedFromFiles(
                &self.v_proj[idx],
                files,
                &name_buf,
                idx,
                "self_attn.v_proj",
            );
            try loadPackedFromFiles(
                &self.o_proj[idx],
                files,
                &name_buf,
                idx,
                "self_attn.o_proj",
            );

            // MLP projections.
            try loadPackedFromFiles(
                &self.gate_proj[idx],
                files,
                &name_buf,
                idx,
                "mlp.gate_proj",
            );
            try loadPackedFromFiles(
                &self.up_proj[idx],
                files,
                &name_buf,
                idx,
                "mlp.up_proj",
            );
            try loadPackedFromFiles(
                &self.down_proj[idx],
                files,
                &name_buf,
                idx,
                "mlp.down_proj",
            );

            // Norm scales (f32 → f16).
            try loadNormFromFiles(
                &self.attn_norm[idx],
                files,
                &name_buf,
                idx,
                "input_layernorm",
            );
            try loadNormFromFiles(
                &self.ffn_norm[idx],
                files,
                &name_buf,
                idx,
                "post_attention_layernorm",
            );
            try loadNormFromFiles(
                &self.q_norm[idx],
                files,
                &name_buf,
                idx,
                "self_attn.q_norm",
            );
            try loadNormFromFiles(
                &self.k_norm[idx],
                files,
                &name_buf,
                idx,
                "self_attn.k_norm",
            );
            log.info(
                "loaded layer {d}/{d}",
                .{ layer_index + 1, Config.num_layers },
            );
        }
    };
}

// ============================================================================
// Free-standing helpers — no self, primitive arguments (Rule 20)
// ============================================================================

/// Search all opened safetensors files for a tensor by name.
/// Linear scan is fine — called O(tensors × files) at load time
/// only, never on the hot path.
fn findTensorInFiles(
    files: []const safetensors.SafetensorsFile,
    name: []const u8,
) !*const safetensors.TensorDescriptor {
    std.debug.assert(files.len > 0);
    std.debug.assert(name.len > 0);
    for (files) |*file| {
        if (file.getTensor(name)) |desc| {
            return desc;
        }
    }
    log.err("tensor not found in any shard: {s}", .{name});
    return error.TensorNotFound;
}

/// Format a per-layer tensor name into a reusable stack buffer.
/// Example: layer_index=5, component="self_attn.q_proj",
/// suffix=".weight" → "model.layers.5.self_attn.q_proj.weight".
///
/// The returned slice points into `buffer` and is valid until the
/// next call that reuses the same buffer.
fn formatLayerTensorName(
    buffer: *[MAX_TENSOR_NAME_LENGTH]u8,
    layer_index: u32,
    component: []const u8,
    suffix: []const u8,
) []const u8 {
    std.debug.assert(component.len > 0);
    std.debug.assert(suffix.len > 0);
    return std.fmt.bufPrint(
        buffer,
        "model.layers.{d}.{s}{s}",
        .{ layer_index, component, suffix },
    ) catch {
        // Name exceeds MAX_TENSOR_NAME_LENGTH — programming error.
        unreachable;
    };
}

/// Load a packed projection (bits + scales) from safetensors files.
/// Formats the `.weight` and `.scales` tensor names, finds them
/// across all shards, and copies data into the PackedBuffer.
fn loadPackedFromFiles(
    packed_buf: *metal.PackedBuffer,
    files: []const safetensors.SafetensorsFile,
    name_buf: *[MAX_TENSOR_NAME_LENGTH]u8,
    layer_index: u32,
    component: []const u8,
) !void {
    std.debug.assert(files.len > 0);
    std.debug.assert(packed_buf.packed_count > 0);

    // Look up the packed bits tensor.
    const weight_name = formatLayerTensorName(
        name_buf,
        layer_index,
        component,
        ".weight",
    );
    const weight_desc = try findTensorInFiles(
        files,
        weight_name,
    );

    // Look up the f16 scale tensor (name_buf reuse is safe —
    // findTensorInFiles already returned a stable pointer above).
    const scales_name = formatLayerTensorName(
        name_buf,
        layer_index,
        component,
        ".scales",
    );
    const scales_desc = try findTensorInFiles(
        files,
        scales_name,
    );

    loadPackedProjection(packed_buf, weight_desc, scales_desc);
}

/// Load a norm scale tensor (f32 → f16) from safetensors files.
fn loadNormFromFiles(
    half_buf: *metal.HalfBuffer,
    files: []const safetensors.SafetensorsFile,
    name_buf: *[MAX_TENSOR_NAME_LENGTH]u8,
    layer_index: u32,
    component: []const u8,
) !void {
    std.debug.assert(files.len > 0);
    std.debug.assert(half_buf.len > 0);
    const name = formatLayerTensorName(
        name_buf,
        layer_index,
        component,
        ".weight",
    );
    const desc = try findTensorInFiles(files, name);
    loadNormScale(half_buf, desc);
}

/// Copy packed bits and convert scales from MLX affine to symmetric
/// encoding, writing both into a pre-allocated PackedBuffer's Metal
/// shared memory.
///
/// MLX affine stores `(scale, bias)` per group.  We convert:
///     symmetric_scale = mlx_scale / 2
/// The bias is discarded — our kernel uses symmetric dequant:
///     value = bit ? +scale : -scale
///
/// Weight dtype is U8 or U32 — MLX 1-bit models pack bits into
/// U32 elements.  On little-endian (Apple Silicon) the raw bytes
/// are identical, so we memcpy either way.
fn loadPackedProjection(
    packed_buf: *metal.PackedBuffer,
    weight_desc: *const safetensors.TensorDescriptor,
    scales_desc: *const safetensors.TensorDescriptor,
) void {
    // MLX stores packed bits as U32; raw bytes are identical to
    // U8 on little-endian (Apple Silicon).  Accept either.
    std.debug.assert(
        weight_desc.dtype == .u8 or weight_desc.dtype == .u32,
    );
    std.debug.assert(scales_desc.dtype == .f16);

    const raw: [*]u8 = @ptrCast(
        packed_buf.obj.msgSend(*anyopaque, "contents", .{}),
    );

    // Copy packed bits directly into the buffer's bits region.
    const bit_len = packed_buf.packedBytes();
    std.debug.assert(weight_desc.data.len >= bit_len);
    @memcpy(raw[0..bit_len], weight_desc.data[0..bit_len]);

    // Convert scales: affine → symmetric (divide by 2).
    const num_groups = packed_buf.numGroups();
    const scale_offset = packed_buf.scaleOffset();
    const target: [*]f16 = @ptrCast(
        @alignCast(raw + scale_offset),
    );

    std.debug.assert(scales_desc.data.len >= num_groups * 2);
    const source: [*]const f16 = @ptrCast(
        @alignCast(scales_desc.data.ptr),
    );

    convertScalesAffineToSymmetric(
        source[0..num_groups],
        target[0..num_groups],
    );
}

/// Copy a norm scale tensor into a pre-allocated HalfBuffer.
///
/// MLX Bonsai stores norms as F16; other checkpoints may use
/// F32.  When the source is already F16, a direct memcpy
/// suffices.  When F32, we narrow to F16 (loss is negligible
/// for RMSNorm scales which are typically near 1.0).
fn loadNormScale(
    half_buf: *metal.HalfBuffer,
    desc: *const safetensors.TensorDescriptor,
) void {
    std.debug.assert(
        desc.dtype == .f16 or desc.dtype == .f32,
    );

    const raw = half_buf.obj.msgSend(
        [*]u8,
        "contents",
        .{},
    );
    const target: [*]f16 = @ptrCast(@alignCast(raw));

    if (desc.dtype == .f16) {
        // F16 source: direct byte copy, no conversion needed.
        const byte_len = half_buf.len * 2;
        std.debug.assert(desc.data.len >= byte_len);
        const src: [*]const f16 = @ptrCast(
            @alignCast(desc.data.ptr),
        );
        @memcpy(
            target[0..half_buf.len],
            src[0..half_buf.len],
        );
    } else {
        // F32 source: narrow to f16.
        std.debug.assert(desc.data.len >= half_buf.len * 4);
        const source: [*]const f32 = @ptrCast(
            @alignCast(desc.data.ptr),
        );
        convertF32ToF16(
            source[0..half_buf.len],
            target[0..half_buf.len],
        );
    }
}

/// Convert MLX affine scales to symmetric scales: out = in / 2.
/// Operates element-wise on f16 slices.
pub fn convertScalesAffineToSymmetric(
    source: []const f16,
    target: []f16,
) void {
    std.debug.assert(source.len == target.len);
    std.debug.assert(source.len > 0);
    for (source, target) |s, *t| {
        const val: f32 = @floatCast(s);
        t.* = @floatCast(val / 2.0);
    }
}

/// Convert f32 elements to f16, element-wise.
pub fn convertF32ToF16(source: []const f32, target: []f16) void {
    std.debug.assert(source.len == target.len);
    std.debug.assert(source.len > 0);
    for (source, target) |s, *t| {
        t.* = @floatCast(s);
    }
}

/// Get a mutable f16 slice into a HalfBuffer's Metal shared memory.
/// Valid only while no GPU write is in flight on this buffer.
fn halfBufferSlice(buf: metal.HalfBuffer) []f16 {
    std.debug.assert(buf.len > 0);
    const raw = buf.obj.msgSend([*]u8, "contents", .{});
    const aligned: [*]align(@alignOf(f16)) u8 =
        @alignCast(raw);
    const ptr: [*]f16 = @ptrCast(aligned);
    return ptr[0..buf.len];
}

test {
    _ = @import("model_test.zig");
}
