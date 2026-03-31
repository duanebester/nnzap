//! Metal compute backend for nnzap
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
};

// ============================================================================
// Device — the main Metal context
// ============================================================================

pub const Device = struct {
    obj: objc.Object,
    command_queue: objc.Object,
    library: objc.Object,
    unified_memory: bool,

    // Pre-compiled compute pipelines for all our kernels.
    vector_add: ComputePipeline,
    matmul: ComputePipeline,
    matmul_tiled: ComputePipeline,
    matmul_bias: ComputePipeline,
    matmul_bias_relu: ComputePipeline,
    matmul_bias_v2: ComputePipeline,
    matmul_bias_relu_v2: ComputePipeline,
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

        // Set non-pipeline fields, then compile pipelines below.
        self.obj = device;
        self.command_queue = command_queue;
        self.library = library;
        self.unified_memory = unified_memory;

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

    /// Create a compute command encoder on the given command buffer.
    pub fn beginCompute(
        self: *const Device,
        cmd_buf: objc.Object,
    ) objc.Object {
        // Assert the device has a valid command queue.
        std.debug.assert(self.command_queue.value != null);
        std.debug.assert(cmd_buf.value != null);

        return cmd_buf.msgSend(
            objc.Object,
            "computeCommandEncoder",
            .{},
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
