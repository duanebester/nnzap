# Architecture: nnzap — Zig + Metal ML Engine for Apple Silicon

A GPU-accelerated neural network library that exploits Apple Silicon's unified memory for true zero-copy compute. Written in Zig with Metal compute shaders.

## Why this is a sweet spot

Most ML frameworks were designed around NVIDIA's discrete GPU model, where data must be copied across the PCIe bus:

```
NVIDIA (discrete GPU):

  CPU RAM                           GPU VRAM
  ┌─────────┐    PCIe copy →     ┌─────────┐
  │ input    │ ──────────────────→│ input'   │
  │ weights  │ ──────────────────→│ weights' │
  └─────────┘                     └─────────┘
                                       │
                                   GPU compute
                                       │
  ┌─────────┐    ← PCIe copy     ┌─────────┐
  │ result'  │ ←──────────────────│ result   │
  └─────────┘                     └─────────┘

  Cost: 2× PCIe round-trips per forward pass
```

Apple Silicon has **unified memory** — CPU and GPU share the same physical RAM:

```
Apple Silicon (unified memory):

  Shared Physical RAM
  ┌──────────────────────────────────────────┐
  │                                          │
  │   ┌──────────┐   (same address space)    │
  │   │ input    │◄──── CPU writes here      │
  │   │ weights  │◄──── GPU reads here       │
  │   │ result   │◄──── GPU writes, CPU reads│
  │   └──────────┘                           │
  │                                          │
  └──────────────────────────────────────────┘

  Cost: 0 copies. The []f32 buffer Zig writes to IS the GPU buffer.
```

The `[]f32` slice Zig writes to **IS** the Metal buffer. No copies. No marshaling. No serialization. The pointer Zig gets from `buffer.contents()` is the same physical memory the GPU shader indexes into.

## The architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          COMPILE TIME                               │
│                                                                     │
│  layout.zig (comptime)                                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Network: [784→128 relu, 128→64 relu, 64→10 tanh]            │ │
│  │  → param_count = 109,386                                      │ │
│  │  → weight_offsets = [0, 100_480, 108_736]                     │ │
│  │  → bias_offsets = [100_352, 108_672, 109_376]                 │ │
│  │  → activation sizes, buffer sizes — all known at comptime     │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     METAL SHARED BUFFERS                            │
│                     (unified memory)                                │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────────────────┐   │
│  │ params[] │ │ grads[]  │ │ MultiBuffered(Buffer, N)         │   │
│  │ f32×109k │ │ f32×109k │ │ ┌────────┐┌────────┐┌────────┐ │   │
│  │          │ │          │ │ │  buf 0  ││  buf 1 ││ buf N-1│ │   │
│  └──────────┘ └──────────┘ │ └────────┘└────────┘└────────┘ │   │
│                             │  getCurrent() ──► *Buffer      │   │
│                             │  swap()        ──► advance     │   │
│                             └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
  ┌──────────────┐             ┌────────────────┐
  │  CPU cores   │             │   GPU cores    │
  │  (Zig)       │             │   (Metal)      │
  │              │             │                │
  │  • prep next │             │  • matmul      │
  │    batch     │             │  • relu/tanh   │
  │  • load data │             │  • sgd_update  │
  │  • metrics   │             │  • bias_add    │
  └──────────────┘             └────────────────┘
```

## Project structure

```
nnzap/
  build.zig            -- Build script (links zig-objc, Metal, Foundation, CoreGraphics)
  build.zig.zon        -- Package manifest (zig_objc dependency)
  .gitignore           -- Excludes build artifacts, snapshots, benchmarks
  src/
    root.zig           -- Library entry point, re-exports metal + layout + network
    metal.zig          -- Metal compute backend (Device, Buffer, MultiBuffered, pipelines)
    layout.zig         -- Comptime network layout (sizes, offsets, shapes, slice helpers)
    network.zig        -- Network struct with forward/backward pass (comptime-parameterised)
    main.zig           -- MNIST training loop, evaluation, benchmark recording
    benchmark.zig      -- Benchmark result struct, JSON serialisation, epoch/test recording
    mnist.zig          -- MNIST IDX data loader, normalisation, one-hot encoding
    shaders/
      compute.metal    -- All GPU compute kernels (MSL)
  scripts/
    autoresearch.zig   -- Hyperparameter research CLI (config editing, experiment runner)
    program.md         -- AI agent instructions for hyperparameter experiments
    engine_research.zig -- Engine research CLI (snapshot/rollback, compile/test/bench)
    engine_program.md  -- AI agent instructions for engine optimisation experiments
  docs/
    ARCHITECTURE.md    -- This file
    STRATEGY.md        -- Where nnzap wins (Moreau pattern, target domains)
```

## What's implemented

### 1. Metal shared buffers (`metal.zig`)

The foundation: `MTLBuffer` objects allocated with `.storage_shared` — unified memory that both CPU and GPU can access without copying.

```zig
pub const Buffer = struct {
    obj: objc.Object,
    len: usize, // number of f32 elements

    /// Allocate a new Metal buffer with shared storage (zero-copy on Apple Silicon).
    pub fn init(device: objc.Object, num_elements: usize) !Buffer {
        const byte_len = num_elements * @sizeOf(f32);
        const raw = device.msgSend(
            ?*anyopaque,
            "newBufferWithLength:options:",
            .{ @as(c_ulong, @intCast(byte_len)), MTLResourceOptions.storage_shared },
        ) orelse return error.MetalBufferAllocFailed;
        return .{
            .obj = objc.Object.fromId(raw),
            .len = num_elements,
        };
    }

    /// Get a Zig []f32 slice that points directly into the Metal shared buffer.
    /// On Apple Silicon this IS the GPU memory — zero copy.
    pub fn asSlice(self: Buffer) []f32 {
        const ptr = self.obj.msgSend(*anyopaque, "contents", .{});
        const float_ptr: [*]f32 = @ptrCast(@alignCast(ptr));
        return float_ptr[0..self.len];
    }

    /// Return the underlying Metal buffer object (for binding to shaders).
    pub fn metalBuffer(self: Buffer) objc.Object {
        return self.obj;
    }

    pub fn deinit(self: *Buffer) void {
        self.obj.msgSend(void, "release", .{});
        self.* = undefined;
    }
};
```

Key insight: `asSlice()` returns a `[]f32` backed by GPU-visible memory. When you write `slice[i] = 42.0`, the GPU can see that value immediately — no upload step needed. `init` returns `!Buffer` because the Metal allocation can fail (null pointer check). This is the `zig-objc` `msgSend` pattern from [mitchellh/zig-objc](https://github.com/mitchellh/zig-objc).

### 2. Multi-buffering (`metal.zig`)

Overlap CPU data loading with GPU compute by ping-ponging between N sets of buffers. Generic over any type `T` and configurable buffer count `N` (2 for double, 3 for triple):

```zig
pub fn MultiBuffered(comptime T: type, comptime N: u32) type {
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

        /// Return a const pointer to the current buffer without advancing.
        pub fn getCurrentConst(self: *const Self) *const T {
            std.debug.assert(self.current < N);
            return &self.buffers[self.current];
        }

        /// Advance to the next buffer. Call this only after the GPU is done
        /// with the current buffer (e.g. after commitAndWait or a fence).
        pub fn swap(self: *Self) void {
            std.debug.assert(self.current < N);
            self.current = @intCast(((@as(u32, self.current)) + 1) % N);
        }
    };
}
```

**Design choices:**

- **Generic over `T`**: double-buffer `Buffer`s, activation structs, entire layer state — whatever you need.
- **`getCurrent()` and `swap()` are separate operations**: the caller decides when to swap, not the buffer. You can bind the same buffer to multiple encoder slots, read results back, etc. without accidentally advancing.
- **`IntFittingRange` for the index**: a `u1` for double-buffering, `u2` for triple — structurally impossible to hold an invalid index.
- **`N` is comptime**: change `2` to `3` for triple-buffering when CPU batch prep is the bottleneck.

The `Device` provides a convenience constructor with proper errdefer cleanup:

```zig
pub fn createMultiBuffered(
    self: *const Device,
    comptime buffer_count: u32,
    num_elements: usize,
) !MultiBuffered(Buffer, buffer_count) {
    var buffers: [buffer_count]Buffer = undefined;
    var initialized_count: u32 = 0;

    errdefer {
        // Clean up any buffers we successfully allocated before the failure.
        for (buffers[0..initialized_count]) |*b| {
            b.deinit();
        }
    }

    for (&buffers) |*b| {
        b.* = try Buffer.init(self.obj, num_elements);
        initialized_count += 1;
    }

    return .{ .buffers = buffers, .current = 0 };
}
```

The overlap pattern for a training loop:

```
  ┌────────────────────────────────────────────────────────────────┐
  │ Step N:   GPU reads buf[0]    │  CPU fills buf[1] with batch  │
  │ Step N+1: GPU reads buf[1]    │  CPU fills buf[0] with batch  │
  │ Step N+2: GPU reads buf[0]    │  CPU fills buf[1] with batch  │
  └────────────────────────────────────────────────────────────────┘
  Triple-buffering adds a third buffer so neither side ever stalls:
  │ Step N:   GPU reads buf[0]    │  CPU fills buf[2]  │ buf[1] ready │
```

### 3. Metal compute shaders (`compute.metal`)

All GPU kernels live in a single Metal Shading Language file:

| Kernel             | Purpose                                   |
| ------------------ | ----------------------------------------- |
| `vector_add`       | Element-wise addition (proof of life)     |
| `matmul`           | Matrix multiplication (M×K · K×N)         |
| `relu_forward`     | ReLU activation: max(0, x)                |
| `tanh_forward`     | Tanh activation                           |
| `sigmoid_forward`  | Sigmoid activation: 1/(1+exp(-x))         |
| `relu_backward`    | ReLU gradient: x > 0 ? 1 : 0              |
| `tanh_backward`    | Tanh gradient: 1 - tanh²(x)               |
| `sigmoid_backward` | Sigmoid gradient: σ(x)(1 - σ(x))          |
| `bias_add`         | Add bias vector to each row               |
| `matmul_transA`    | A^T · B (weight gradients: dW = X^T · dY) |
| `matmul_transB`    | A · B^T (input gradients: dX = dY · W^T)  |
| `bias_grad`        | Column-wise sum (bias grads: db = Σ dY)   |
| `sgd_update`       | SGD: param -= lr \* grad                  |

The matmul kernel — the workhorse of neural networks:

```metal
kernel void matmul(
    device const float* W   [[buffer(0)]],  // [M x K]
    device const float* x   [[buffer(1)]],  // [K x N]
    device float*       out [[buffer(2)]],  // [M x N]
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 gid               [[thread_position_in_grid]])
{
    // gid.x = column (n), gid.y = row (m)
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0;
    for (uint i = 0; i < K; i++) {
        sum += W[gid.y * K + i] * x[i * N + gid.x];
    }
    out[gid.y * N + gid.x] = sum;
}
```

Each GPU thread computes one element of the output matrix. Metal dispatches thousands of these threads across the GPU cores automatically. Dimension arguments use `constant uint&` — Metal's read-only optimized memory for small values passed from the CPU.

### 4. ComputePipeline and Device struct (`metal.zig`)

Each kernel is wrapped in a `ComputePipeline` struct that holds the compiled `MTLComputePipelineState`:

```zig
pub const ComputePipeline = struct {
    state: objc.Object, // MTLComputePipelineState

    pub fn init(device: objc.Object, library: objc.Object, function_name: [:0]const u8) !ComputePipeline {
        // ... look up function by name, compile to pipeline state ...
    }
};
```

The `Device` struct owns the Metal device, command queue, compiled shader library, and all pre-compiled pipelines:

```zig
pub const Device = struct {
    obj: objc.Object,
    command_queue: objc.Object,
    library: objc.Object,
    unified_memory: bool,

    // Pre-compiled compute pipelines for all our kernels
    vector_add: ComputePipeline,
    matmul: ComputePipeline,
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

    pub fn init() !Device {
        const device_ptr = MTLCreateSystemDefaultDevice() orelse
            return error.MetalNotAvailable;
        const device = objc.Object.fromId(device_ptr);

        const unified_memory = device.msgSend(bool, "hasUnifiedMemory", .{});
        // ...
        const library = try compileLibrary(device);

        return .{
            .obj = device,
            .command_queue = command_queue,
            .library = library,
            .unified_memory = unified_memory,
            .vector_add = try ComputePipeline.init(device, library, "vector_add"),
            .matmul = try ComputePipeline.init(device, library, "matmul"),
            // ... all 13 pipelines pre-compiled at init ...
        };
    }
};
```

The shader MSL source is embedded into the binary via `@embedFile("shaders/compute.metal")` at compile time, then compiled to GPU machine code at runtime via `newLibraryWithSource:options:error:`. Each kernel function becomes a `ComputePipeline` state object.

`Device` also provides dispatch helpers:

- **`dispatch1D(encoder, pipeline, count)`** — for element-wise ops (activations, SGD update).
- **`dispatch2D(encoder, pipeline, width, height)`** — for 2D ops (matmul, bias_add).

Both use Metal's non-uniform threadgroup API (`dispatchThreads:threadsPerThreadgroup:`) so we don't need to manually pad grid dimensions to tile boundaries.

### 5. Comptime layout (`layout.zig`)

Network architecture is defined as a comptime array of `LayerDesc` structs. Each layer specifies its input size, output size, and activation function. All memory layout math happens at compile time — zero runtime cost:

```zig
pub const Activation = enum { relu, tanh_act, sigmoid, none };

pub const LayerDesc = struct {
    in: u32,
    out: u32,
    act: Activation = .relu,
};

const Layout = NetworkLayout(&.{
    .{ .in = 784, .out = 128, .act = .relu },
    .{ .in = 128, .out = 64, .act = .relu },
    .{ .in = 64, .out = 10, .act = .tanh_act },
});

// All of these are compile-time constants:
// Layout.param_count     = 109386
// Layout.weight_offsets  = [0, 100480, 108736]
// Layout.bias_offsets    = [100352, 108672, 109376]
// Layout.weight_counts   = [100352, 8192, 640]
// Layout.bias_counts     = [128, 64, 10]
// Layout.activation_sizes = [128, 64, 10]
// Layout.max_activation_size = 128
// Layout.input_size      = 784
// Layout.output_size     = 10
```

Slice helpers index into the flat parameter buffer with comptime-known offsets:

```zig
/// Return the weight slice for layer `layer` from a flat parameter buffer.
/// The offset and length are comptime-known — this compiles to pointer arithmetic.
pub fn getWeightSlice(params: []f32, comptime layer: usize) []f32 {
    comptime {
        std.debug.assert(layer < n);
    }
    const offset = weight_offsets[layer];
    const len = weight_counts[layer];
    std.debug.assert(params.len >= offset + len);
    return params[offset..][0..len];
}

/// Return the bias slice for layer `layer` from a flat parameter buffer.
/// The offset and length are comptime-known — this compiles to pointer arithmetic.
pub fn getBiasSlice(params: []f32, comptime layer: usize) []f32 {
    comptime {
        std.debug.assert(layer < n);
    }
    const offset = bias_offsets[layer];
    const len = bias_counts[layer];
    std.debug.assert(params.len >= offset + len);
    return params[offset..][0..len];
}
```

Because the offsets are comptime-known, these compile down to simple pointer arithmetic — no hash maps, no string lookups, no dynamic dispatch. The comptime assert on `layer` catches out-of-bounds layer indices at compile time, and the runtime assert on `params.len` guards against undersized buffers.

### 6. Helper functions (`metal.zig`)

Clean wrappers around Metal's verbose `setBuffer:offset:atIndex:` and `setBytes:length:atIndex:` calls:

```zig
/// Bind a Metal buffer to a compute encoder at the given index.
pub fn setBuffer(encoder: objc.Object, buffer: Buffer, index: u32) void {
    encoder.msgSend(void, "setBuffer:offset:atIndex:", .{
        buffer.obj.value,
        @as(c_ulong, 0),
        @as(c_ulong, index),
    });
}

/// Bind raw bytes (e.g. a uint or float constant) to a compute encoder.
pub fn setBytes(encoder: objc.Object, comptime T: type, value: *const T, index: u32) void {
    encoder.msgSend(void, "setBytes:length:atIndex:", .{
        @as(*const anyopaque, @ptrCast(value)),
        @as(c_ulong, @sizeOf(T)),
        @as(c_ulong, index),
    });
}
```

`setBytes` takes a `comptime T` and a `*const T` pointer — type-safe and explicit about the size being sent to the GPU. These turn a 5-line `msgSend` into a 1-liner, keeping the dispatch code readable.

### 7. Network forward pass (`network.zig`)

The `Network` struct wires all the individual kernels into a complete forward pass. It's parameterised by a comptime `NetworkLayout` and a comptime `max_batch_size`, so all buffer sizes are resolved at compile time:

```zig
const TinyLayout = NetworkLayout(&.{
    .{ .in = 2, .out = 4, .act = .relu },
    .{ .in = 4, .out = 1, .act = .sigmoid },
});
const TinyNet = Network(TinyLayout, 32); // max 32 samples per batch

var net: TinyNet = undefined;
try net.init(&device);
defer net.deinit();
```

**Pre-allocated buffers** (Rule 2 — no allocation after init):

| Buffer                | Size                                   | Purpose                                     |
| --------------------- | -------------------------------------- | ------------------------------------------- |
| `params`              | `Layout.param_count`                   | Flat weights + biases for all layers        |
| `scratch`             | `max_batch_size × max_activation_size` | Reused each layer for matmul output         |
| `pre_activations[i]`  | `max_batch_size × activation_sizes[i]` | Before activation (needed by backward pass) |
| `post_activations[i]` | `max_batch_size × activation_sizes[i]` | After activation (input to next layer)      |

**Forward pass** encodes all layers into a single compute command encoder — one command buffer submission for the entire network:

```
For each layer i (unrolled at comptime via inline for):
  1. matmul:     input[batch × in] × W[in × out] → scratch
  2. bias_add:   scratch + bias → pre_act[i]  (or post_act[i] if act == .none)
  3. activation: pre_act[i] → post_act[i]     (skipped if act == .none)
  4. barrier:    memoryBarrier between each dispatch
```

The matrix layout convention is `[batch × features]` row-major: each row is one sample, each column is one feature/neuron. This aligns naturally with the `bias_add` shader which adds `bias[col]` per-column.

Three private helper functions (`encodeMatmul`, `encodeBiasAdd`, `encodeActivation`) take only primitive/comptime arguments — no `self` — following Rule 20 (hot loop extraction). The activation pipeline is selected at comptime via a switch on `Layout.layers[i].act`.

### 8. Network backward pass (`network.zig`)

The backward pass mirrors the forward pass in reverse, computing
gradients for every weight and bias in the network via
backpropagation. Like the forward pass, it encodes all work into
a **single compute command encoder** — one command buffer
submission for the entire backward pass.

**Additional pre-allocated buffers** (Rule 2 — no allocation
after init):

| Buffer                | Size                                   | Purpose                                                |
| --------------------- | -------------------------------------- | ------------------------------------------------------ |
| `grads`               | `Layout.param_count`                   | Flat weight + bias gradients (same layout as `params`) |
| `backward_scratch[0]` | `max_batch_size × max_activation_size` | Ping-pong buffer for upstream gradient propagation     |
| `backward_scratch[1]` | `max_batch_size × max_activation_size` | Ping-pong buffer for upstream gradient propagation     |

The `grads` buffer shares the same flat layout as `params` —
weight and bias gradients are indexed with the same comptime
`weight_offsets` and `bias_offsets` from `NetworkLayout`. The
two `backward_scratch` buffers alternate roles: one holds the
upstream gradient from the layer above, the other receives the
input gradient for the next layer down.

**Per-layer backward** (reversed via comptime `inline for`):

```
For each layer i in reverse (num_layers-1 → 0):

  If layer has activation (act != .none):
    1. activation_backward:  grad_output ⊙ act'(pre_act[i])
                             → backward_scratch[target]
    2. bias_grad:            col_sum(backward_scratch[target])
                             → grads[bias_offset[i]]
    3. weight_grad:          input[i]ᵀ · backward_scratch[target]
                             → grads[weight_offset[i]]
    4. input_grad (if i>0):  backward_scratch[target] · Wᵀ
                             → backward_scratch[next]

  If layer has no activation (act == .none):
    1. bias_grad:            col_sum(grad_pre_act)
                             → grads[bias_offset[i]]
    2. weight_grad:          input[i]ᵀ · grad_pre_act
                             → grads[weight_offset[i]]
    3. input_grad (if i>0):  grad_pre_act · Wᵀ
                             → backward_scratch[next]
```

**Three new Metal kernels** support the backward pass:

- **`matmul_transA`**: computes Aᵀ · B — used for weight
  gradients (dW = Xᵀ · dY).
- **`matmul_transB`**: computes A · Bᵀ — used for input
  gradients (dX = dY · Wᵀ).
- **`bias_grad`**: column-wise sum over the batch dimension —
  used for bias gradients (db = Σ dY).

Four private helper functions (`encodeActivationBackward`,
`encodeBiasGrad`, `encodeWeightGrad`, `encodeInputGrad`) follow
Rule 20 (hot loop extraction) — they take only primitive/comptime
arguments and the Metal encoder, no `self`. The `gradSlice()`
accessor returns the flat gradient buffer for the caller to read
computed gradients or apply parameter updates.

**Memory barrier discipline**: a `memoryBarrier` is inserted
between each dispatch to ensure gradient writes are visible to
subsequent reads within the same command encoder.

### 9. Adam optimiser (`network.zig` + `compute.metal`)

The Adam optimiser (Kingma & Ba, 2014) maintains per-parameter
first and second moment estimates for adaptive learning rates.
The implementation follows the same pattern as SGD: a Metal
compute kernel handles the math, and the `Network` struct
provides a high-level method.

**Metal kernel** (`compute.metal`):

```metal
kernel void adam_update(
    device float*       params      [[buffer(0)]],
    device const float* grads       [[buffer(1)]],
    device float*       m           [[buffer(2)]],  // first moment
    device float*       v           [[buffer(3)]],  // second moment
    constant float& lr              [[buffer(4)]],
    constant float& beta1           [[buffer(5)]],
    constant float& beta2           [[buffer(6)]],
    constant float& epsilon         [[buffer(7)]],
    constant float& correction1     [[buffer(8)]],  // 1 - beta1^t
    constant float& correction2     [[buffer(9)]],  // 1 - beta2^t
    ...)
```

Bias correction terms (`1 - beta^t`) are precomputed on the CPU
and passed as constants to avoid `pow()` in every GPU thread.

**Pre-allocated buffers** (Rule 2 — no allocation after init):

| Buffer   | Size                 | Purpose             |
| -------- | -------------------- | ------------------- |
| `adam_m` | `Layout.param_count` | First moment (mean) |
| `adam_v` | `Layout.param_count` | Second moment (var) |

Both buffers are zero-filled at init and persist across training
steps. The `updateAdam` method increments an internal timestep
counter for bias correction.

**Usage** — drop-in replacement for SGD:

```zig
// SGD:
net.update(device, enc, learning_rate);

// Adam:
net.updateAdam(device, enc, 0.001, 0.9, 0.999, 1e-8);
//                          lr    beta1 beta2  eps
```

## Verified working

Actual output from `zig build run`:

```
$ zig build run

nnzap -- GPU neural network engine for Apple Silicon

Initialising Metal device...
info(metal): Apple Silicon unified memory detected — zero-copy enabled
Metal device ready (unified memory: true)

Running vector_add on GPU (1024 elements)...
vector_add verified -- all 1024 elements correct!

+--------------------------------------------+
|          nnzap Network Layout              |
+--------------------------------------------+
|  Layer 0:  784 -> 128   (relu    )        |
|  Layer 1:  128 -> 64    (relu    )        |
|  Layer 2:   64 -> 10    (tanh_act)        |
+--------------------------------------------+
|  Total parameters: 109386               |
|  Max activation:   128                  |
+--------------------------------------------+

Weight offsets: 0 100480 108736
Bias offsets:   100352 108672 109376

Allocated 109386 params (437544 bytes) in Metal shared buffer
Layer 0 weights: 100352 floats @ offset 0
Layer 0 biases:  128 floats @ offset 100352

All done! Metal compute pipeline is working.

--- Backward Pass Demo (numerical gradient check) ---

Target:   1.0000
Output:   0.706822
MSE loss: 0.042977

Gradient check: 17 parameters, max relative error = 2.410442e-3
Backward pass VERIFIED!
```

## Why this is genuinely fast

| Aspect                | General frameworks (PyTorch, etc.)     | nnzap                                 |
| --------------------- | -------------------------------------- | ------------------------------------- |
| **Memory copies**     | CPU→GPU→CPU per batch (discrete GPU)   | Zero — unified memory, same pointer   |
| **Python overhead**   | GIL, interpreter, dynamic dispatch     | None — pure Zig, compiles to native   |
| **Buffer management** | Runtime allocation, reference counting | Comptime-sized, pre-allocated         |
| **Graph overhead**    | Dynamic graph trace on every forward   | Static layout, no graph construction  |
| **Binary size**       | ~2 GB (PyTorch install)                | ~200 KB (single static binary)        |
| **Unified memory**    | Bolted on (MPS backend, still copies)  | Native — designed around it           |
| **Shader compile**    | Runtime JIT per kernel invocation      | Once at init, cached pipeline objects |

## Dependencies

- **[zig-objc](https://github.com/mitchellh/zig-objc)** (mitchellh) — Objective-C runtime bindings for Zig. Provides `msgSend`, class lookup, selector handling. Used for all Metal API calls.
- **Apple Metal framework** — GPU compute API. Linked via `build.zig` with `linkFramework("Metal")`.
- **Apple Foundation framework** — Required by Metal for `NSString`, `NSError`, etc. Linked with `linkFramework("Foundation")`.
- **Apple CoreGraphics framework** — Linked with `linkFramework("CoreGraphics")`.

## Agent (`scripts/agent.zig`)

The agent is a Zig binary that talks to Claude via the Anthropic Messages API.
Claude decides which experiments to run; the agent executes them using the
autoresearch toolbox, sends results back, and loops until Claude stops or
`MAX_TURNS` is reached. The agent IS the runtime — like Claude Code, but ours.

### Architecture

```
┌──────────────────────┐
│  Claude (API)         │  The brain. Decides experiments.
└──────────┬───────────┘
           │ HTTP (via curl)
           ▼
┌──────────────────────┐
│  agent.zig            │  The runtime. Executes tools.
│                       │  Manages conversation + history.
└──────────┬───────────┘
           │ spawns
           ▼
┌──────────────────────┐
│  autoresearch.zig     │  The toolbox. config/train/bench.
└──────────┬───────────┘
           │ spawns
           ▼
┌──────────────────────┐
│  nnzap (main.zig)     │  The training binary.
└──────────────────────┘
```

### Context seeding

The agent seeds Claude with everything it needs to operate autonomously:

1. **System prompt** (`SYSTEM_PROMPT` constant) — role description, goal
   (maximize `final_test_accuracy_pct`), experiment protocol (config_show →
   config_backup → config_set → train → evaluate → keep or config_restore),
   decision rules (±0.05 pp threshold), architecture constraints (784 input,
   10 output, matching layers, valid batch sizes), and phased strategy
   (optimizer → learning rate → architecture → batch size → fine-tuning).

2. **Tool schemas** (`TOOL_SCHEMAS` constant) — JSON tool definitions for
   `config_show`, `config_set`, `config_backup`, `config_restore`, `train`,
   `benchmark_compare`, and `benchmark_latest`. Passed to the Anthropic API
   so Claude can call them by name.

3. **Experiment history** — the full content of `experiments.jsonl` is injected
   into the first user message. Claude sees what has been tried in previous
   runs and avoids repeating past work.

### Loop structure

```
startup:
  1. Load API key     (ANTHROPIC_API_KEY env var)
  2. Load history     (.agent_history/experiments.jsonl)
  3. Build toolbox    (zig build)
  4. Send initial message to Claude (system prompt + history)

turn loop (up to MAX_TURNS):
  5. Call Anthropic Messages API
  6. Parse response: extract text blocks + tool_use blocks
  7. Log Claude's reasoning to stderr
  8. If stop_reason == "end_turn" or no tool calls: break
  9. For each tool call:
       Route to autoresearch command
       Capture JSON output
       If tool is "train": append benchmark to experiments.jsonl
 10. Send tool results back to Claude as the next message
 11. Goto 5

finale:
 12. Save full conversation to .agent_history/run_{timestamp}.json
```

Claude drives the experiment strategy. It decides what to try, evaluates
results, and stops when improvements plateau.

### Persistent history

All state lives in `.agent_history/`:

```
.agent_history/
├── experiments.jsonl              # Append-only train results (cross-run memory)
├── _request.json                  # Last API request body (debug)
├── run_2025-03-26T17-45-00Z.json  # Conversation log from run 1
└── run_2025-03-27T09-12-00Z.json  # Conversation log from run 2
```

**`experiments.jsonl`** — each successful `train` call appends its full
benchmark JSON (collapsed to one line) to this file. On the next agent run,
this content is injected into the first user message so Claude sees what has
been tried before. Crash-safe: if interrupted, all completed experiments are
preserved. Capped at `MAX_HISTORY_INJECT` bytes (30 KB) when loaded.

**`run_{timestamp}.json`** — the full conversation (all messages exchanged
between agent and Claude) saved as a JSON array. For debugging and auditing.
Not read by the agent on startup.

### Constants

```zig
MAX_TURNS            = 200       // Agent loop cap
MAX_MESSAGES         = 512       // Conversation length cap
MAX_TOOL_CALLS       = 16        // Tool calls per API response
MAX_TOOL_OUTPUT      = 50_000    // Truncation limit per tool result
MAX_HISTORY_INJECT   = 30_000    // History bytes injected into prompt
```

## Roadmap

### Done

- [x] **1. Metal compute shader** — `vector_add` proof of life, GPU dispatch working
- [x] **2. Matrix multiplication** — `matmul` kernel, non-uniform threadgroup dispatch
- [x] **3. Comptime network layout** — All sizes, offsets, shapes computed at compile time
- [x] **4. Multi-buffering** — `MultiBuffered(T, N)` for overlapping CPU/GPU work (double/triple)
- [x] **5. Activation shaders** — Forward + backward for ReLU, tanh, sigmoid
- [x] **6. Bias add + SGD** — `bias_add` and `sgd_update` kernels
- [x] **7. Slice helpers** — `getWeightSlice` and `getBiasSlice` with comptime bounds checks
- [x] **8. Wire forward pass** — `Network` struct chains matmul → bias_add → activation per layer

- [x] **9. Wire backward pass** — Reverse chain with gradient kernels, numerically verified
- [x] **10. Training loop** — Epochs, MSE loss + gradient on GPU, SGD update, loss tracking
- [x] **10a. MSE loss kernels** — mse_forward and mse_backward GPU shaders, wired into Network
- [x] **12. Softmax + cross-entropy** — Numerically stable softmax, CE loss, fused softmax+CE backward kernel
- [x] **13. Adam optimiser** — `adam_update` GPU kernel, bias-corrected moments, `updateAdam` method
- [x] **14. Autoresearch toolbox** — Zig CLI (`scripts/autoresearch.zig`) for autonomous hyperparameter optimization
- [x] **15. Engine research toolbox** — Zig CLI (`scripts/engine_research.zig`) for autonomous engine optimisation (Metal dispatch, kernels, buffers)
- [x] **16. Agent** — LLM-powered experiment runner (`scripts/agent.zig`) that calls Claude via the Anthropic API, executes autoresearch tools, persists results to JSONL history, and accumulates context across runs

### Next

- [ ] **11. Multi-core data loading** — Zig `std.Thread` pool for batch preparation
- [ ] **17. Benchmarks vs PyTorch MPS** — MNIST training time comparison
- [ ] **18. Tiled matmul kernel** — Shared memory tiling for large matrices
- [ ] **19. Learning rate scheduling** — Cosine annealing, warmup
- [ ] **20. Kernel fusion** — Combine matmul + bias_add + activation into fewer dispatches
- [ ] **21. Async command buffers** — Double-buffered commit for CPU/GPU overlap
