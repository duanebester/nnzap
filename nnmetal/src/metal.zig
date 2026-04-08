//! Metal compute backend for nnmetal
//!
//! Wraps Apple's Metal API via zig-objc for GPU compute on Apple Silicon.
//! Uses shared buffers (zero-copy unified memory) so the same physical memory
//! is accessible from both CPU (Zig) and GPU (Metal shaders).

const std = @import("std");
const objc = @import("objc");

/// Re-export objc.Object so callers (e.g. main.zig) can
/// name the type for command buffers and encoders without
/// importing the objc module directly.
pub const Object = objc.Object;

const log = std.log.scoped(.metal);

// ============================================================================
// Limits (Rule 4 — put a limit on everything)
// ============================================================================

/// Metal supports buffer indices 0–30 in compute encoders.
const MAX_BUFFER_INDEX: u32 = 31;

/// Upper bound on total threads across all dimensions in a single dispatch.
const MAX_DISPATCH_THREADS: u32 = 65536 * 65536;

// ============================================================================
// Metal API types (subset needed for compute)
// ============================================================================

pub const MTLResourceOptions = packed struct(c_ulong) {
    cpu_cache_mode: CPUCacheMode = .default,
    storage_mode: StorageMode = .shared,
    hazard_tracking_mode: HazardTrackingMode = .default,
    _pad: @Type(.{
        .int = .{
            .signedness = .unsigned,
            .bits = @bitSizeOf(c_ulong) - 10,
        },
    }) = 0,

    pub const CPUCacheMode = enum(u4) {
        default = 0,
        write_combined = 1,
    };

    pub const StorageMode = enum(u4) {
        shared = 0,
        managed = 1,
        private = 2,
        memoryless = 3,
    };

    pub const HazardTrackingMode = enum(u2) {
        default = 0,
        untracked = 1,
        tracked = 2,
    };

    pub const storage_shared: MTLResourceOptions = .{
        .storage_mode = .shared,
    };
    pub const storage_private: MTLResourceOptions = .{
        .storage_mode = .private,
    };
};

pub const MTLSize = extern struct {
    width: c_ulong,
    height: c_ulong,
    depth: c_ulong,
};

pub extern "c" fn MTLCreateSystemDefaultDevice() ?*anyopaque;

// ============================================================================
// Buffer — a Metal shared buffer with a typed Zig slice view
// ============================================================================

pub const Buffer = struct {
    obj: objc.Object,
    len: u32, // number of f32 elements

    /// Allocate a new Metal buffer with shared storage
    /// (zero-copy on Apple Silicon).
    pub fn init(
        device: objc.Object,
        num_elements: u32,
    ) !Buffer {
        std.debug.assert(num_elements > 0);

        const byte_len: usize = @as(usize, num_elements) * @sizeOf(f32);
        // Overflow guard: byte_len must not wrap around zero.
        std.debug.assert(byte_len > 0);

        const raw = device.msgSend(
            ?*anyopaque,
            "newBufferWithLength:options:",
            .{
                @as(c_ulong, @intCast(byte_len)),
                MTLResourceOptions.storage_shared,
            },
        ) orelse return error.MetalBufferAllocFailed;

        const result = Buffer{
            .obj = objc.Object.fromId(raw),
            .len = num_elements,
        };

        // Zero-fill to prevent stale data leaks (buffer bleed prevention).
        const slice = result.asSlice();
        @memset(slice, 0.0);

        return result;
    }

    /// Allocate a new Metal buffer with shared storage and
    /// write-combined CPU cache mode.  Write-combined mode
    /// bypasses the CPU cache on writes, avoiding cache
    /// pollution for buffers the CPU fills but never reads.
    /// Ideal for input buffers written by CPU (fillImageBatch)
    /// and read only by the GPU (forward pass).
    pub fn initWriteCombined(
        device: objc.Object,
        num_elements: u32,
    ) !Buffer {
        std.debug.assert(num_elements > 0);

        const byte_len: usize =
            @as(usize, num_elements) * @sizeOf(f32);
        std.debug.assert(byte_len > 0);

        const opts = MTLResourceOptions{
            .cpu_cache_mode = .write_combined,
            .storage_mode = .shared,
        };

        const raw = device.msgSend(
            ?*anyopaque,
            "newBufferWithLength:options:",
            .{
                @as(c_ulong, @intCast(byte_len)),
                opts,
            },
        ) orelse return error.MetalBufferAllocFailed;

        const result = Buffer{
            .obj = objc.Object.fromId(raw),
            .len = num_elements,
        };

        // Zero-fill to prevent stale data leaks.
        const slice = result.asSlice();
        @memset(slice, 0.0);

        return result;
    }

    /// Get a Zig []f32 slice that points directly into the Metal
    /// shared buffer. On Apple Silicon this IS the GPU memory —
    /// zero copy.
    pub fn asSlice(self: Buffer) []f32 {
        std.debug.assert(self.len > 0);

        const ptr = self.obj.msgSend(*anyopaque, "contents", .{});
        const float_ptr: [*]f32 = @ptrCast(@alignCast(ptr));
        return float_ptr[0..@as(usize, self.len)];
    }

    /// Return the underlying Metal buffer object
    /// (for binding to shaders).
    pub fn metalBuffer(self: Buffer) objc.Object {
        return self.obj;
    }

    pub fn deinit(self: *Buffer) void {
        self.obj.msgSend(void, "release", .{});
        self.* = undefined;
    }
};

/// Half-precision buffer for inference weight storage.
/// Stores float16 values — half the memory bandwidth of
/// float32 buffers.  Created via GPU conversion kernel
/// from an existing float32 buffer.
pub const HalfBuffer = struct {
    obj: objc.Object,
    len: u32, // number of f16 elements

    /// Allocate a half-precision Metal buffer.
    pub fn init(
        device: objc.Object,
        num_elements: u32,
    ) !HalfBuffer {
        std.debug.assert(num_elements > 0);

        // f16 is 2 bytes per element.
        const byte_len: usize =
            @as(usize, num_elements) * 2;
        std.debug.assert(byte_len > 0);

        const raw = device.msgSend(
            ?*anyopaque,
            "newBufferWithLength:options:",
            .{
                @as(c_ulong, @intCast(byte_len)),
                MTLResourceOptions.storage_shared,
            },
        ) orelse return error.MetalBufferAllocFailed;

        return .{
            .obj = objc.Object.fromId(raw),
            .len = num_elements,
        };
    }

    pub fn deinit(self: *HalfBuffer) void {
        self.obj.msgSend(void, "release", .{});
        self.* = undefined;
    }
};

// ============================================================================
// PackedBuffer — 1-bit packed weights with per-group f16 scales
// ============================================================================

/// A Metal shared buffer storing 1-bit packed weights in Q1_0_g128
/// format. The buffer contains two regions in a single allocation:
///
///   [packed_bits (uint8_t)] [scales (f16)]
///
/// The packed bits and scales live in the same MTLBuffer. The
/// encoder binds the same buffer twice with different offsets to
/// present both regions to the kernel without an extra allocation.
pub const PackedBuffer = struct {
    obj: objc.Object,
    packed_count: u32, // number of 1-bit weight elements
    group_size: u32, // weights per scale group (128)
    byte_len: u32, // total allocation in bytes

    /// Number of packed bytes: ceil(packed_count / 8).
    pub fn packedBytes(self: PackedBuffer) u32 {
        std.debug.assert(self.packed_count > 0);
        std.debug.assert(self.group_size > 0);
        return @as(u32, @intCast(
            std.math.divCeil(u32, self.packed_count, 8) catch
                unreachable,
        ));
    }

    /// Number of scale groups: ceil(packed_count / group_size).
    pub fn numGroups(self: PackedBuffer) u32 {
        std.debug.assert(self.packed_count > 0);
        std.debug.assert(self.group_size > 0);
        return @as(u32, @intCast(
            std.math.divCeil(
                u32,
                self.packed_count,
                self.group_size,
            ) catch unreachable,
        ));
    }

    /// Byte offset where the f16 scale array begins
    /// (= packedBytes, since scales follow packed bits).
    pub fn scaleOffset(self: PackedBuffer) u32 {
        const offset = self.packedBytes();
        std.debug.assert(offset > 0);
        std.debug.assert(offset < self.byte_len);
        return offset;
    }

    /// Allocate a new PackedBuffer for `num_weights` 1-bit
    /// elements with the given group size.
    pub fn init(
        device: objc.Object,
        num_weights: u32,
        group_size_val: u32,
    ) !PackedBuffer {
        std.debug.assert(num_weights > 0);
        std.debug.assert(group_size_val > 0);

        const packed_bytes: u32 = @intCast(
            std.math.divCeil(u32, num_weights, 8) catch
                unreachable,
        );
        const num_groups: u32 = @intCast(
            std.math.divCeil(
                u32,
                num_weights,
                group_size_val,
            ) catch unreachable,
        );
        const scale_bytes: u32 = num_groups * 2; // f16 = 2 bytes
        const total_bytes: u32 = packed_bytes + scale_bytes;
        std.debug.assert(total_bytes > 0);

        const raw = device.msgSend(
            ?*anyopaque,
            "newBufferWithLength:options:",
            .{
                @as(c_ulong, total_bytes),
                MTLResourceOptions.storage_shared,
            },
        ) orelse return error.MetalBufferAllocFailed;

        const result = PackedBuffer{
            .obj = objc.Object.fromId(raw),
            .packed_count = num_weights,
            .group_size = group_size_val,
            .byte_len = total_bytes,
        };

        // Zero-fill to prevent stale data leaks (Rule 21).
        const ptr = result.obj.msgSend(
            *anyopaque,
            "contents",
            .{},
        );
        const byte_ptr: [*]u8 = @ptrCast(ptr);
        @memset(byte_ptr[0..total_bytes], 0);

        return result;
    }

    /// Return the underlying Metal buffer object.
    pub fn metalBuffer(self: PackedBuffer) objc.Object {
        return self.obj;
    }

    pub fn deinit(self: *PackedBuffer) void {
        self.obj.msgSend(void, "release", .{});
        self.* = undefined;
    }
};

// ============================================================================
// MultiBuffered — generic N-buffered resource for overlapping CPU/GPU work
// ============================================================================

/// Generic multi-buffered container for overlapping CPU and GPU work.
/// Use N=2 for double-buffering (standard), N=3 for triple-buffering
/// (hides longer CPU prep latency behind GPU execution).
///
/// The key invariant: `getCurrent()` and `swap()` are separate
/// operations. The caller decides when to swap — not the buffer.
/// This lets you bind the same buffer to multiple encoder slots,
/// read results back, etc. without accidentally advancing past it.
pub fn MultiBuffered(comptime T: type, comptime N: u32) type {
    // Triple-buffering is the practical max — beyond that, the
    // latency hiding gains vanish and memory cost dominates.
    comptime {
        std.debug.assert(N >= 2);
        std.debug.assert(N <= 4);
    }

    const IndexType = std.math.IntFittingRange(0, N - 1);

    return struct {
        const Self = @This();

        buffers: [N]T,
        current: IndexType,

        /// Return a pointer to the current buffer without advancing.
        /// Call this as many times as you need — it's idempotent.
        pub fn getCurrent(self: *Self) *T {
            std.debug.assert(self.current < N);
            return &self.buffers[self.current];
        }

        /// Return a const pointer to the current buffer without
        /// advancing.
        pub fn getCurrentConst(self: *const Self) *const T {
            std.debug.assert(self.current < N);
            return &self.buffers[self.current];
        }

        /// Advance to the next buffer. Call this only after the GPU
        /// is done with the current buffer (e.g. after commitAndWait
        /// or a fence).
        pub fn swap(self: *Self) void {
            std.debug.assert(self.current < N);
            self.current = @intCast(
                ((@as(u32, self.current)) + 1) % N,
            );
        }
    };
}

// ============================================================================
// ComputePipeline — a compiled Metal compute kernel
// ============================================================================

pub const ComputePipeline = struct {
    state: objc.Object, // MTLComputePipelineState
    /// Cached from MTLComputePipelineState at init time.
    /// Avoids repeated Obj-C message sends on every
    /// dispatch1D call (~15 dispatches per batch).
    max_threads_per_group: c_ulong,

    pub fn init(
        device: objc.Object,
        library: objc.Object,
        function_name: [:0]const u8,
    ) !ComputePipeline {
        // Get NSString for function name.
        const NSString = objc.getClass("NSString") orelse
            return error.ClassNotFound;
        const name_ns = NSString.msgSend(
            objc.Object,
            "stringWithUTF8String:",
            .{function_name.ptr},
        );

        // Get the function from the library.
        const func_raw = library.msgSend(
            ?*anyopaque,
            "newFunctionWithName:",
            .{name_ns.value},
        ) orelse {
            log.err(
                "Metal function not found: {s}",
                .{function_name},
            );
            return error.MetalFunctionNotFound;
        };
        const func = objc.Object.fromId(func_raw);
        defer func.msgSend(void, "release", .{});

        // Create compute pipeline state.
        var error_ptr: ?*anyopaque = null;
        const state_raw = device.msgSend(
            ?*anyopaque,
            "newComputePipelineStateWithFunction:error:",
            .{ func.value, &error_ptr },
        ) orelse {
            if (error_ptr) |err| {
                const err_obj = objc.Object.fromId(err);
                const desc = err_obj.msgSend(
                    objc.Object,
                    "localizedDescription",
                    .{},
                );
                const c_str = desc.msgSend(
                    [*:0]const u8,
                    "UTF8String",
                    .{},
                );
                log.err(
                    "Pipeline creation failed: {s}",
                    .{c_str},
                );
            }
            return error.MetalPipelineCreationFailed;
        };

        const state = objc.Object.fromId(state_raw);

        // Cache the hardware thread limit once at init
        // to avoid Obj-C message send overhead per dispatch.
        const max_threads = state.msgSend(
            c_ulong,
            "maxTotalThreadsPerThreadgroup",
            .{},
        );
        std.debug.assert(max_threads > 0);

        return .{
            .state = state,
            .max_threads_per_group = max_threads,
        };
    }
};

// ============================================================================
// Pipeline specification table
// ============================================================================

/// Kernel names for all pre-compiled compute pipelines.
/// Each entry maps a Device field name to a Metal shader function name.
const PipelineSpec = struct {
    field_name: []const u8,
    shader_name: [:0]const u8,
};

const pipeline_specs = [_]PipelineSpec{
    .{ .field_name = "vector_add", .shader_name = "vector_add" },
    .{ .field_name = "matmul", .shader_name = "matmul" },
    .{ .field_name = "matmul_tiled", .shader_name = "matmul_tiled" },
    .{
        .field_name = "matmul_bias",
        .shader_name = "matmul_bias",
    },
    .{
        .field_name = "matmul_bias_relu",
        .shader_name = "matmul_bias_relu",
    },
    .{
        .field_name = "matmul_bias_relu_infer",
        .shader_name = "matmul_bias_relu_infer",
    },
    .{
        .field_name = "matmul_bias_v2",
        .shader_name = "matmul_bias_v2",
    },
    .{
        .field_name = "matmul_bias_relu_v2",
        .shader_name = "matmul_bias_relu_v2",
    },
    .{
        .field_name = "matmul_transA",
        .shader_name = "matmul_transA",
    },
    .{
        .field_name = "matmul_transB",
        .shader_name = "matmul_transB",
    },

    .{ .field_name = "relu_forward", .shader_name = "relu_forward" },
    .{ .field_name = "tanh_forward", .shader_name = "tanh_forward" },
    .{
        .field_name = "sigmoid_forward",
        .shader_name = "sigmoid_forward",
    },
    .{
        .field_name = "relu_backward",
        .shader_name = "relu_backward",
    },
    .{
        .field_name = "tanh_backward",
        .shader_name = "tanh_backward",
    },
    .{
        .field_name = "sigmoid_backward",
        .shader_name = "sigmoid_backward",
    },
    .{ .field_name = "bias_add", .shader_name = "bias_add" },
    .{ .field_name = "bias_grad", .shader_name = "bias_grad" },
    .{ .field_name = "sgd_update", .shader_name = "sgd_update" },
    .{ .field_name = "mse_forward", .shader_name = "mse_forward" },
    .{
        .field_name = "mse_backward",
        .shader_name = "mse_backward",
    },
    .{
        .field_name = "softmax_forward",
        .shader_name = "softmax_forward",
    },
    .{ .field_name = "ce_forward", .shader_name = "ce_forward" },
    .{
        .field_name = "softmax_ce_backward",
        .shader_name = "softmax_ce_backward",
    },
    .{
        .field_name = "adam_update",
        .shader_name = "adam_update",
    },
    .{
        .field_name = "argmax_predictions",
        .shader_name = "argmax_predictions",
    },
    .{
        .field_name = "bias_grad_sgd",
        .shader_name = "bias_grad_sgd",
    },
    .{
        .field_name = "weight_grad_sgd",
        .shader_name = "weight_grad_sgd",
    },
    .{
        .field_name = "forward_fused_infer_3layer",
        .shader_name = "forward_fused_infer_3layer",
    },
    .{
        .field_name = "forward_fused_infer_3layer_v2",
        .shader_name = "forward_fused_infer_3layer_v2",
    },
    .{
        .field_name = "forward_fused_infer_3layer_v3",
        .shader_name = "forward_fused_infer_3layer_v3",
    },
    .{
        .field_name = "forward_fused_infer_batched",
        .shader_name = "forward_fused_infer_batched",
    },
    .{
        .field_name = "matmul_bias_packed",
        .shader_name = "matmul_bias_packed",
    },
    .{
        .field_name = "matmul_bias_relu_infer_packed",
        .shader_name = "matmul_bias_relu_infer_packed",
    },
    .{
        .field_name = "f32_to_f16",
        .shader_name = "f32_to_f16",
    },
    .{
        .field_name = "forward_fused_infer_batched_f16",
        .shader_name = "forward_fused_infer_batched_f16",
    },
    .{
        .field_name = "forward_fused_infer_single_f16",
        .shader_name = "forward_fused_infer_single_f16",
    },
    .{
        .field_name = "f32_to_1bit",
        .shader_name = "f32_to_1bit",
    },
    .{
        .field_name = "qmv",
        .shader_name = "qmv",
    },
    .{
        .field_name = "qmv_fast",
        .shader_name = "qmv_fast",
    },
    .{
        .field_name = "qmv_fast_sm",
        .shader_name = "qmv_fast_sm",
    },
    .{
        .field_name = "qmv_fast_sm2",
        .shader_name = "qmv_fast_sm2",
    },
    .{
        .field_name = "qmv_const",
        .shader_name = "qmv_const",
    },
    .{
        .field_name = "qmv_fast_multigroup",
        .shader_name = "qmv_fast_multigroup",
    },
    .{
        .field_name = "qmv_const_multigroup",
        .shader_name = "qmv_const_multigroup",
    },
    .{
        .field_name = "qmv_fused_pair",
        .shader_name = "qmv_fused_pair",
    },
    .{
        .field_name = "qmv_fused_pair_sm",
        .shader_name = "qmv_fused_pair_sm",
    },
    .{
        .field_name = "qmv_fused_pair_const",
        .shader_name = "qmv_fused_pair_const",
    },
    .{
        .field_name = "qmm",
        .shader_name = "qmm",
    },
    .{
        .field_name = "qmv_f16in",
        .shader_name = "qmv_f16in",
    },
    .{
        .field_name = "qmv_const_f16in",
        .shader_name = "qmv_const_f16in",
    },
    .{
        .field_name = "qmv_fused_pair_const_f16in",
        .shader_name = "qmv_fused_pair_const_f16in",
    },
    .{
        .field_name = "qmv_const_multigroup_f16in",
        .shader_name = "qmv_const_multigroup_f16in",
    },
    .{
        .field_name = "qmv_f16io",
        .shader_name = "qmv_f16io",
    },
    .{
        .field_name = "qmv_const_f16io",
        .shader_name = "qmv_const_f16io",
    },
    .{
        .field_name = "qmv_fused_pair_const_f16io",
        .shader_name = "qmv_fused_pair_const_f16io",
    },
    .{
        .field_name = "qmv_const_multigroup_f16io",
        .shader_name = "qmv_const_multigroup_f16io",
    },
    .{
        .field_name = "qmv_f16io_resadd",
        .shader_name = "qmv_f16io_resadd",
    },
    .{
        .field_name = "qmv_const_f16io_resadd",
        .shader_name = "qmv_const_f16io_resadd",
    },
    .{
        .field_name = "qmv_const_multigroup_f16io_resadd",
        .shader_name = "qmv_const_multigroup_f16io_resadd",
    },
};

// ============================================================================
// Device — the main Metal context
// ============================================================================

pub const Device = struct {
    obj: objc.Object,
    command_queue: objc.Object,
    library: objc.Object,
    unified_memory: bool,
    /// Half of MTLDevice.recommendedMaxWorkingSetSize — the
    /// maximum bytes nnmetal will allocate via Metal buffers.
    /// Computed once at init. Keeps nnmetal a well-behaved library
    /// by leaving the other half for the OS, other apps, and the
    /// CPU data pipeline.
    memory_budget_bytes: u64,

    // Pre-compiled compute pipelines for all our kernels.
    vector_add: ComputePipeline,
    matmul: ComputePipeline,
    matmul_tiled: ComputePipeline,
    matmul_bias: ComputePipeline,
    matmul_bias_relu: ComputePipeline,
    matmul_bias_v2: ComputePipeline,
    matmul_bias_relu_v2: ComputePipeline,
    matmul_bias_relu_infer: ComputePipeline,
    matmul_transA: ComputePipeline,
    matmul_transB: ComputePipeline,

    relu_forward: ComputePipeline,
    tanh_forward: ComputePipeline,
    sigmoid_forward: ComputePipeline,
    relu_backward: ComputePipeline,
    tanh_backward: ComputePipeline,
    sigmoid_backward: ComputePipeline,
    bias_add: ComputePipeline,
    bias_grad: ComputePipeline,
    sgd_update: ComputePipeline,
    adam_update: ComputePipeline,
    mse_forward: ComputePipeline,
    mse_backward: ComputePipeline,
    softmax_forward: ComputePipeline,
    ce_forward: ComputePipeline,
    softmax_ce_backward: ComputePipeline,
    argmax_predictions: ComputePipeline,
    bias_grad_sgd: ComputePipeline,
    weight_grad_sgd: ComputePipeline,
    forward_fused_infer_3layer: ComputePipeline,
    forward_fused_infer_3layer_v2: ComputePipeline,
    forward_fused_infer_3layer_v3: ComputePipeline,
    forward_fused_infer_batched: ComputePipeline,
    matmul_bias_packed: ComputePipeline,
    matmul_bias_relu_infer_packed: ComputePipeline,
    f32_to_f16: ComputePipeline,
    forward_fused_infer_batched_f16: ComputePipeline,
    forward_fused_infer_single_f16: ComputePipeline,
    f32_to_1bit: ComputePipeline,
    qmv: ComputePipeline,
    qmv_fast: ComputePipeline,
    qmv_fast_sm: ComputePipeline,
    qmv_fast_sm2: ComputePipeline,
    qmv_const: ComputePipeline,
    qmv_fast_multigroup: ComputePipeline,
    qmv_const_multigroup: ComputePipeline,
    qmv_fused_pair: ComputePipeline,
    qmv_fused_pair_sm: ComputePipeline,
    qmv_fused_pair_const: ComputePipeline,
    qmm: ComputePipeline,
    // f16-input QMV variants for the f16 activation pipeline.
    // These read half* input instead of float*, halving constant-
    // cache pressure and activation memory bandwidth.
    qmv_f16in: ComputePipeline,
    qmv_const_f16in: ComputePipeline,
    qmv_fused_pair_const_f16in: ComputePipeline,
    qmv_const_multigroup_f16in: ComputePipeline,
    // f16-input, f16-output QMV variants.  Used when both the
    // input activation (e.g. norm_out) and the output activation
    // (e.g. Q, K, V, gate, up) are f16 buffers.
    qmv_f16io: ComputePipeline,
    qmv_const_f16io: ComputePipeline,
    qmv_fused_pair_const_f16io: ComputePipeline,
    qmv_const_multigroup_f16io: ComputePipeline,
    // QMV + fused residual accumulate (f16 in, f32 += out).
    qmv_f16io_resadd: ComputePipeline,
    qmv_const_f16io_resadd: ComputePipeline,
    qmv_const_multigroup_f16io_resadd: ComputePipeline,

    // Specialized QMV pipelines — compiled at init for the
    // specific model dimensions. Set by transformer init
    // via initSpecializedQMV(). Null when not initialized.
    spec_qmv_f16io: ?ComputePipeline = null,
    spec_qmv_f16io_resadd: ?ComputePipeline = null,
    spec_qmv_fused_pair_f16io: ?ComputePipeline = null,
    spec_qmv_f16in: ?ComputePipeline = null,
    spec_qmv_mg_f16io_resadd: ?ComputePipeline = null,
    spec_qmv_fused_pair_silu_f16io: ?ComputePipeline = null,
    spec_qmv_fused_norm_pair_silu_f16io: ?ComputePipeline = null,

    // K values the specialized pipelines target.
    // Used by dispatch functions to verify dims.K matches
    // before selecting a specialized pipeline.
    spec_hidden_K: u32 = 0,
    spec_inter_K: u32 = 0,

    pub fn init(self: *Device) !void {
        const device = objc.Object.fromId(
            MTLCreateSystemDefaultDevice() orelse
                return error.MetalNotAvailable,
        );

        const unified_memory = device.msgSend(
            bool,
            "hasUnifiedMemory",
            .{},
        );
        if (unified_memory) {
            log.info(
                "Apple Silicon unified memory detected" ++
                    " — zero-copy enabled",
                .{},
            );
        } else {
            log.warn(
                "Discrete GPU detected — copies may be needed",
                .{},
            );
        }

        const command_queue = device.msgSend(
            objc.Object,
            "newCommandQueue",
            .{},
        );

        // Compile the Metal shader library from embedded source.
        const library = try compileLibrary(device);

        // Half of what Metal itself recommends as the safe working
        // set ceiling. recommendedMaxWorkingSetSize is typically
        // 70-75% of total unified RAM on Apple Silicon — so /2
        // gives nnmetal ~35-37% of total RAM, leaving the rest
        // available for the rest of the system.
        const recommended_max: u64 = @intCast(device.msgSend(
            c_ulong,
            "recommendedMaxWorkingSetSize",
            .{},
        ));
        const memory_budget_bytes: u64 = recommended_max / 2;
        log.info(
            "Memory budget: {d} MB " ++
                "(recommendedMaxWorkingSetSize={d} MB, using half)",
            .{
                memory_budget_bytes / (1024 * 1024),
                recommended_max / (1024 * 1024),
            },
        );

        // Set non-pipeline fields, then compile pipelines below.
        self.obj = device;
        self.command_queue = command_queue;
        self.library = library;
        self.unified_memory = unified_memory;
        self.memory_budget_bytes = memory_budget_bytes;

        try self.compilePipelines(device, library);
    }

    /// Compile all compute pipelines from the shader library.
    /// Uses comptime field iteration to avoid repetition.
    fn compilePipelines(
        self: *Device,
        device_obj: objc.Object,
        library: objc.Object,
    ) !void {
        inline for (pipeline_specs) |spec| {
            @field(self, spec.field_name) =
                try ComputePipeline.init(
                    device_obj,
                    library,
                    spec.shader_name,
                );
        }
    }

    /// Create a shared buffer (zero-copy on Apple Silicon).
    pub fn createBuffer(
        self: *const Device,
        num_elements: u32,
    ) !Buffer {
        std.debug.assert(num_elements > 0);

        // Fail fast before asking Metal for memory we shouldn't take.
        // currentAllocatedSize is queried live so the check is always
        // fresh — no stale snapshot can mask a budget breach.
        const incoming_bytes: u64 =
            @as(u64, num_elements) * @sizeOf(f32);
        const current_bytes: u64 = @intCast(self.obj.msgSend(
            c_ulong,
            "currentAllocatedSize",
            .{},
        ));
        if (current_bytes + incoming_bytes > self.memory_budget_bytes) {
            // Return the error — the caller has the context to log
            // a useful message (which buffer, which network, etc.).
            // Logging here would be noise without that context.
            return error.MetalMemoryBudgetExceeded;
        }

        return Buffer.init(self.obj, num_elements);
    }

    /// Create multi-buffered Metal shared buffers for overlapping
    /// CPU/GPU work. Use buffer_count=2 for double-buffering,
    /// 3 for triple-buffering.
    pub fn createMultiBuffered(
        self: *const Device,
        comptime buffer_count: u32,
        num_elements: u32,
    ) !MultiBuffered(Buffer, buffer_count) {
        comptime {
            std.debug.assert(buffer_count >= 2);
            std.debug.assert(buffer_count <= 4);
        }

        var buffers: [buffer_count]Buffer = undefined;
        var initialized_count: u32 = 0;

        errdefer {
            // Clean up any buffers we successfully allocated
            // before the failure.
            for (buffers[0..initialized_count]) |*b| {
                b.deinit();
            }
        }

        for (&buffers) |*b| {
            b.* = try Buffer.init(self.obj, num_elements);
            initialized_count += 1;
        }

        return .{
            .buffers = buffers,
            .current = 0,
        };
    }

    /// Begin a new command buffer for submitting GPU work.
    pub fn beginCommandBuffer(
        self: *const Device,
    ) objc.Object {
        return self.command_queue.msgSend(
            objc.Object,
            "commandBuffer",
            .{},
        );
    }

    /// Begin a command buffer that skips retain/release on
    /// bound resources.  The caller MUST ensure all buffers
    /// and pipeline states outlive the command buffer.  Saves
    /// ~6 retain/release Obj-C calls per inference dispatch,
    /// which matters for single-sample latency where Metal
    /// API overhead dominates GPU compute time.
    pub fn beginCommandBufferUnretained(
        self: *const Device,
    ) objc.Object {
        std.debug.assert(
            self.command_queue.value != null,
        );
        return self.command_queue.msgSend(
            objc.Object,
            "commandBufferWithUnretainedReferences",
            .{},
        );
    }

    /// Create a compute command encoder on the given command buffer.
    pub fn beginCompute(
        self: *const Device,
        cmd_buf: objc.Object,
    ) objc.Object {
        // Assert the device has a valid command queue.
        std.debug.assert(self.command_queue.value != null);
        std.debug.assert(cmd_buf.value != null);

        // MTLDispatchTypeConcurrent = 1: allows the GPU to
        // overlap independent dispatches between barriers.
        // Q/KV projections and Q-norm/K-norm-rope dispatches
        // run concurrently within each block (~28 overlap
        // opportunities per token).  Barriers already enforce
        // all data dependencies.
        return cmd_buf.msgSend(
            objc.Object,
            "computeCommandEncoderWithDispatchType:",
            .{@as(c_ulong, 1)},
        );
    }

    /// Dispatch a 1D compute kernel.
    pub fn dispatch1D(
        self: *const Device,
        encoder: objc.Object,
        pipeline: ComputePipeline,
        count: u32,
    ) void {
        std.debug.assert(count > 0);
        // Assert the device is in a valid state.
        std.debug.assert(self.command_queue.value != null);

        encoder.msgSend(
            void,
            "setComputePipelineState:",
            .{pipeline.state.value},
        );

        // Use cached hardware limit to avoid Obj-C message
        // send overhead on every dispatch (Rule 7 — amortise).
        const threads_per_group = @min(
            pipeline.max_threads_per_group,
            @as(c_ulong, count),
        );
        std.debug.assert(threads_per_group > 0);

        // Switch to dispatchThreadgroups for better efficiency.
        // Round up thread count to full threadgroups.
        const thread_count = @as(c_ulong, count);
        const groups_needed = (thread_count + threads_per_group - 1) / threads_per_group;

        const grid_size = MTLSize{
            .width = groups_needed,
            .height = 1,
            .depth = 1,
        };
        const group_size = MTLSize{
            .width = threads_per_group,
            .height = 1,
            .depth = 1,
        };

        encoder.msgSend(
            void,
            "dispatchThreadgroups:threadsPerThreadgroup:",
            .{ grid_size, group_size },
        );
    }

    /// Dispatch a 2D compute kernel (e.g. matmul).
    /// Uses dispatchThreadgroups to guarantee full 16×16
    /// threadgroups, which is required for tiled matmul
    /// kernels where every thread cooperatively loads
    /// shared memory tiles. Over-dispatched threads are
    /// bounds-checked inside each kernel.
    pub fn dispatch2D(
        self: *const Device,
        encoder: objc.Object,
        pipeline: ComputePipeline,
        width: u32,
        height: u32,
    ) void {
        std.debug.assert(width > 0);
        std.debug.assert(height > 0);
        // Assert the device is in a valid state.
        std.debug.assert(self.command_queue.value != null);

        encoder.msgSend(
            void,
            "setComputePipelineState:",
            .{pipeline.state.value},
        );

        // 16×16 = 256 threads per threadgroup matches the
        // TS=16 tile size in the Metal shaders.
        const tile: c_ulong = 16;
        const w: c_ulong = @as(c_ulong, width);
        const h: c_ulong = @as(c_ulong, height);

        // Round up to full threadgroups so every thread
        // in a tile participates in shared memory loads.
        const groups_x = (w + tile - 1) / tile;
        const groups_y = (h + tile - 1) / tile;

        const grid_size = MTLSize{
            .width = groups_x,
            .height = groups_y,
            .depth = 1,
        };
        const group_size = MTLSize{
            .width = tile,
            .height = tile,
            .depth = 1,
        };

        encoder.msgSend(
            void,
            "dispatchThreadgroups:threadsPerThreadgroup:",
            .{ grid_size, group_size },
        );
    }

    /// Dispatch a compute kernel with explicit threadgroup and
    /// grid dimensions.  Use this for kernels whose dispatch
    /// geometry doesn't follow the standard 1D or 2D patterns
    /// — e.g. `qmv` (64 threads/group, ceil(M/2) groups).
    pub fn dispatchCustom(
        self: *const Device,
        encoder: objc.Object,
        pipeline: ComputePipeline,
        grid: MTLSize,
        group: MTLSize,
    ) void {
        std.debug.assert(grid.width > 0);
        std.debug.assert(group.width > 0);
        std.debug.assert(self.command_queue.value != null);

        // Threadgroup size must not exceed pipeline's hardware
        // limit.  Width × height × depth is the total thread
        // count per threadgroup.
        const threads_per_group =
            group.width * group.height * group.depth;
        std.debug.assert(
            threads_per_group <= pipeline.max_threads_per_group,
        );

        encoder.msgSend(
            void,
            "setComputePipelineState:",
            .{pipeline.state.value},
        );

        encoder.msgSend(
            void,
            "dispatchThreadgroups:threadsPerThreadgroup:",
            .{ grid, group },
        );
    }

    /// Commit a command buffer and wait for completion
    /// (synchronous).
    pub fn commitAndWait(
        self: *const Device,
        cmd_buf: objc.Object,
    ) void {
        // Assert the device is in a valid state.
        std.debug.assert(self.command_queue.value != null);
        std.debug.assert(cmd_buf.value != null);

        cmd_buf.msgSend(void, "commit", .{});
        cmd_buf.msgSend(void, "waitUntilCompleted", .{});
    }

    /// Commit a command buffer (asynchronous — returns
    /// immediately).
    pub fn commit(
        self: *const Device,
        cmd_buf: objc.Object,
    ) void {
        // Assert the device is in a valid state.
        std.debug.assert(self.command_queue.value != null);
        std.debug.assert(cmd_buf.value != null);

        cmd_buf.msgSend(void, "commit", .{});
    }

    /// Commit a command buffer and spin-wait on a completion
    /// flag in shared memory.  The GPU kernel writes 1 to
    /// flag_ptr[0] when done (via atomic_store_explicit with
    /// memory_order_release).  The CPU spin-reads the same
    /// unified memory location, avoiding the Mach kernel
    /// trap in waitUntilCompleted (~100-150 us overhead).
    /// Resets the flag to 0 after completion for reuse.
    pub fn commitAndSpinOnFlag(
        self: *const Device,
        cmd_buf: objc.Object,
        flag_ptr: *volatile u32,
    ) void {
        std.debug.assert(
            self.command_queue.value != null,
        );
        std.debug.assert(cmd_buf.value != null);
        // Flag must be 0 (reset) before commit.
        std.debug.assert(flag_ptr.* == 0);

        cmd_buf.msgSend(void, "commit", .{});

        // Spin until GPU sets flag to 1.
        // On Apple Silicon unified memory, GPU writes
        // are visible to CPU after cache coherency
        // propagation (~10-50 ns typical).
        while (flag_ptr.* == 0) {
            // Busy-wait — appropriate for sub-10 us
            // GPU work where kernel trap overhead
            // exceeds the wait time.
            std.atomic.spinLoopHint();
        }

        // Reset for next use.
        flag_ptr.* = 0;
    }

    /// Insert a buffer-scope memory barrier on the compute
    /// encoder. Required between dispatches that share buffers
    /// within the same encoder — ensures writes from the
    /// previous dispatch are visible to the next.
    pub fn memoryBarrier(
        self: *const Device,
        encoder: objc.Object,
    ) void {
        // Assert the device is in a valid state.
        std.debug.assert(self.command_queue.value != null);

        // MTLBarrierScopeBuffers = 1.
        encoder.msgSend(
            void,
            "memoryBarrierWithScope:",
            .{@as(c_ulong, 1)},
        );
    }
};

// ============================================================================
// Helpers
// ============================================================================

/// Bind a Metal buffer to a compute encoder at the given index.
pub fn setBuffer(
    encoder: objc.Object,
    buffer: Buffer,
    index: u32,
) void {
    std.debug.assert(index <= MAX_BUFFER_INDEX);
    std.debug.assert(buffer.len > 0);

    encoder.msgSend(void, "setBuffer:offset:atIndex:", .{
        buffer.obj.value,
        @as(c_ulong, 0),
        @as(c_ulong, index),
    });
}

/// Bind a Metal buffer to a compute encoder at the given index,
/// starting from a specific element offset. The offset is in
/// f32 elements (not bytes) — converted internally.
pub fn setBufferWithOffset(
    encoder: objc.Object,
    buffer: Buffer,
    offset_elements: u32,
    index: u32,
) void {
    std.debug.assert(index <= MAX_BUFFER_INDEX);
    std.debug.assert(buffer.len > 0);
    std.debug.assert(offset_elements < buffer.len);

    const offset_bytes: c_ulong =
        @as(c_ulong, offset_elements) * @sizeOf(f32);
    encoder.msgSend(void, "setBuffer:offset:atIndex:", .{
        buffer.obj.value,
        offset_bytes,
        @as(c_ulong, index),
    });
}

/// Bind up to 4 Metal buffers in one Obj-C message send using
/// setBuffers:offsets:withRange:.  Saves N-1 message sends
/// compared to N individual setBuffer calls (~15 us each).
/// All buffers are bound at consecutive indices starting from
/// `start_index` with zero byte offset.
/// Batch-set buffers with per-buffer byte offsets.
/// Uses a single setBuffers:offsets:withRange: Obj-C
/// call instead of N separate setBuffer calls, saving
/// (N-1) message sends at ~10 us each.
pub fn setBuffersBatchOffsets(
    encoder: objc.Object,
    buffers: []const Buffer,
    byte_offsets: []const c_ulong,
    start_index: u32,
) void {
    std.debug.assert(buffers.len > 0);
    std.debug.assert(buffers.len <= 4);
    std.debug.assert(byte_offsets.len == buffers.len);
    std.debug.assert(
        start_index + buffers.len - 1 <= MAX_BUFFER_INDEX,
    );

    var raw_ptrs: [4]?*anyopaque = .{
        null, null, null, null,
    };
    var offsets: [4]c_ulong = .{ 0, 0, 0, 0 };
    for (buffers, 0..) |buf, i| {
        std.debug.assert(buf.len > 0);
        raw_ptrs[i] = buf.obj.value;
        offsets[i] = byte_offsets[i];
    }

    const range: [2]c_ulong = .{
        @as(c_ulong, start_index),
        @as(c_ulong, buffers.len),
    };

    encoder.msgSend(
        void,
        "setBuffers:offsets:withRange:",
        .{
            @as(
                *const anyopaque,
                @ptrCast(&raw_ptrs),
            ),
            @as(
                *const anyopaque,
                @ptrCast(&offsets),
            ),
            @as([2]c_ulong, range),
        },
    );
}

pub fn setBuffersBatch(
    encoder: objc.Object,
    buffers: []const Buffer,
    start_index: u32,
) void {
    std.debug.assert(buffers.len > 0);
    std.debug.assert(buffers.len <= 4);
    std.debug.assert(
        start_index + buffers.len - 1 <= MAX_BUFFER_INDEX,
    );

    // Build C arrays for the batch call.
    var raw_ptrs: [4]?*anyopaque = .{
        null, null, null, null,
    };
    var offsets: [4]c_ulong = .{ 0, 0, 0, 0 };
    for (buffers, 0..) |buf, i| {
        std.debug.assert(buf.len > 0);
        raw_ptrs[i] = buf.obj.value;
        offsets[i] = 0;
    }

    // NSRange { location, length }.
    const range: [2]c_ulong = .{
        @as(c_ulong, start_index),
        @as(c_ulong, buffers.len),
    };

    encoder.msgSend(
        void,
        "setBuffers:offsets:withRange:",
        .{
            @as(*const anyopaque, @ptrCast(&raw_ptrs)),
            @as(*const anyopaque, @ptrCast(&offsets)),
            @as([2]c_ulong, range),
        },
    );
}

/// Bind raw bytes (e.g. a uint or float constant) to a compute
/// encoder.
pub fn setBytes(
    encoder: objc.Object,
    comptime T: type,
    value: *const T,
    index: u32,
) void {
    // Ensure T is not a zero-sized type — Metal needs real bytes.
    comptime {
        std.debug.assert(@sizeOf(T) > 0);
    }
    std.debug.assert(index <= MAX_BUFFER_INDEX);

    encoder.msgSend(void, "setBytes:length:atIndex:", .{
        @as(*const anyopaque, @ptrCast(value)),
        @as(c_ulong, @sizeOf(T)),
        @as(c_ulong, index),
    });
}

/// Bind a PackedBuffer's two regions (packed bits and f16 scales)
/// to a compute encoder at consecutive indices.  Both bindings
/// reference the same underlying MTLBuffer — the scales binding
/// uses a byte offset equal to packedBytes().
///
/// After this call:
///   buffer(bits_index)   → packed uint8_t bits, offset 0
///   buffer(bits_index+1) → f16 scales, offset = packedBytes()
pub fn setPackedBuffer(
    encoder: objc.Object,
    packed_buffer: PackedBuffer,
    bits_index: u32,
) void {
    std.debug.assert(bits_index + 1 <= MAX_BUFFER_INDEX);
    std.debug.assert(packed_buffer.packed_count > 0);

    // Bind packed bits at offset 0.
    encoder.msgSend(void, "setBuffer:offset:atIndex:", .{
        packed_buffer.obj.value,
        @as(c_ulong, 0),
        @as(c_ulong, bits_index),
    });

    // Bind scales at byte offset = packedBytes() (same MTLBuffer).
    const scale_byte_offset: c_ulong =
        @as(c_ulong, packed_buffer.scaleOffset());
    encoder.msgSend(void, "setBuffer:offset:atIndex:", .{
        packed_buffer.obj.value,
        scale_byte_offset,
        @as(c_ulong, bits_index + 1),
    });
}

fn compileLibrary(device: objc.Object) !objc.Object {
    const shader_source = @embedFile("shaders/compute.metal");

    // The embedded shader source must not be empty — a missing or
    // truncated file would produce a silent compilation "success"
    // with no kernels.
    comptime {
        std.debug.assert(shader_source.len > 0);
    }

    const NSString = objc.getClass("NSString") orelse
        return error.ClassNotFound;
    const source_ns = NSString.msgSend(
        objc.Object,
        "stringWithUTF8String:",
        .{@as([*:0]const u8, shader_source.ptr)},
    );

    var error_ptr: ?*anyopaque = null;
    const lib_raw = device.msgSend(
        ?*anyopaque,
        "newLibraryWithSource:options:error:",
        .{
            source_ns.value,
            @as(?*anyopaque, null),
            &error_ptr,
        },
    ) orelse {
        if (error_ptr) |err| {
            const err_obj = objc.Object.fromId(err);
            const desc = err_obj.msgSend(
                objc.Object,
                "localizedDescription",
                .{},
            );
            const c_str = desc.msgSend(
                [*:0]const u8,
                "UTF8String",
                .{},
            );
            log.err(
                "Shader compilation failed: {s}",
                .{c_str},
            );
        }
        return error.MetalShaderCompilationFailed;
    };

    return objc.Object.fromId(lib_raw);
}

/// Compile a Metal shader library from a runtime-provided
/// source string.  Unlike `compileLibrary`, this does not
/// use `@embedFile` — the caller supplies the shader text,
/// which may be generated at init time with concrete model
/// dimensions baked in as constants.
pub fn compileLibraryFromSource(
    device: objc.Object,
    source: [*:0]const u8,
) !objc.Object {
    std.debug.assert(device.value != null);

    const NSString = objc.getClass("NSString") orelse
        return error.ClassNotFound;
    const source_ns = NSString.msgSend(
        objc.Object,
        "stringWithUTF8String:",
        .{source},
    );

    var error_ptr: ?*anyopaque = null;
    const lib_raw = device.msgSend(
        ?*anyopaque,
        "newLibraryWithSource:options:error:",
        .{
            source_ns.value,
            @as(?*anyopaque, null),
            &error_ptr,
        },
    ) orelse {
        if (error_ptr) |err| {
            const err_obj = objc.Object.fromId(err);
            const desc = err_obj.msgSend(
                objc.Object,
                "localizedDescription",
                .{},
            );
            const c_str = desc.msgSend(
                [*:0]const u8,
                "UTF8String",
                .{},
            );
            log.err(
                "Specialized shader compilation " ++ "failed: {s}",
                .{c_str},
            );
        }
        return error.MetalShaderCompilationFailed;
    };

    return objc.Object.fromId(lib_raw);
}

test {
    _ = @import("metal_test.zig");
}
