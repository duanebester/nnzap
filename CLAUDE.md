# Engineering Notes

We are using `claude-opus-4-6`!

## Project Map

```
nn/
├── build.zig                    152 lines   Build config
├── build.zig.zon                             Package manifest
├── examples/
│   ├── bonsai.zig               553 lines   Bonsai tree classifier
│   ├── bonsai_bench.zig         636 lines   Bonsai benchmarking
│   ├── inference_bench.zig      681 lines   Inference benchmarking
│   ├── mnist.zig              1,042 lines   MNIST training
│   └── mnist_1bit.zig          803 lines   1-bit MNIST variant
├── src/
│   ├── benchmark.zig            706 lines   Benchmarking infra
│   ├── layout.zig               636 lines   Comptime network layout
│   ├── metal.zig              1,568 lines   Metal GPU bindings
│   ├── mnist.zig                407 lines   MNIST data loading
│   ├── model.zig              1,276 lines   Model (safetensors/loading)
│   ├── network.zig            3,308 lines   Core network (forward/backward/train)
│   ├── root.zig                  49 lines   Public re-exports
│   ├── safetensors.zig          736 lines   Safetensors format parser
│   ├── shaders/
│   │   ├── compute.metal      4,675 lines   GPU compute kernels
│   │   ├── qmv_specialized.metal  902 lines   Quantized matmul kernels
│   │   └── transformer.metal  1,343 lines   Transformer-specific kernels
│   ├── specialized_qmv.zig     132 lines   Specialized quantized matmul
│   ├── tokenizer.zig          1,574 lines   Tokenizer
│   └── transformer.zig       5,982 lines   Transformer implementation
├── benchmarks/                               JSON benchmark results
└── data/mnist/                               MNIST raw dataset
                              ──────
                              27,161 lines total

zap/
├── build.zig                                 Build config
├── build.zig.zon                             Package manifest
├── src/
│   ├── agent_core.zig        2,271 lines   Shared agent framework
│   ├── api_client.zig          786 lines   Anthropic HTTP client
│   ├── bonsai_agent.zig        468 lines   Bonsai agent profile
│   ├── bonsai_research.zig      93 lines   Bonsai toolbox config
│   ├── mnist_agent.zig         529 lines   MNIST agent profile
│   ├── mnist_research.zig      885 lines   MNIST toolbox config + custom tools
│   ├── toolbox.zig            2,399 lines   Generic toolbox (shared)
│   └── tools.zig               699 lines   Shared CLI/file utilities
├── programs/
│   ├── autoresearch_framework.md             Autoresearch framework prompt
│   ├── bonsai_program.md                     Bonsai skill file
│   ├── bonsai_system.md                      Bonsai system prompt
│   ├── mnist_program.md                      MNIST skill file
│   ├── mnist_system.md                       MNIST system prompt
│   └── program.md                            Shared conventions
├── benchmarks/                               JSON benchmark results
└── data/mnist/                               MNIST raw dataset
```

Heavy hitters: `transformer.zig`, `network.zig`, `compute.metal` (~14k lines, half the nn codebase).
Comptime spine: `layout.zig` resolves all buffer shapes at compile time.
Three shader files: `compute.metal` (general NN kernels), `transformer.metal` (attention-specific), `qmv_specialized.metal` (quantized matmul).
Agent spine: `agent_core.zig` provides the shared experiment loop; profiles (`mnist_agent.zig`, `bonsai_agent.zig`) configure it.
Toolbox spine: `toolbox.zig` provides all generic CLI tools; domain binaries (`bonsai_research.zig`, `mnist_research.zig`) are thin config wrappers.

### 0. **Simplicity and Elegance**

Simplicity is not a free pass or a first attempt — it is the hardest revision. The goal is to find the "super idea" that solves safety, performance, and developer experience simultaneously. An hour or day of design is worth weeks or months in production. Spend mental energy upfront, proactively rather than reactively, because when the thinking is done, what is spent on the design will be dwarfed by implementation, testing, and maintenance.

> "Simplicity and elegance are unpopular because they require hard work and discipline to achieve." — Edsger Dijkstra

It's easy to say "let's do something simple", but to do that in practice takes thought, multiple passes, many sketches, and still we may have to throw one away. For nnzap, this means: don't settle on the first kernel dispatch strategy or buffer layout that works. Sketch alternatives. The elegant design will be the one where the Metal buffer layout, the comptime network description, and the kernel dispatch strategy all reinforce each other.

### 1. **Zero Technical Debt Policy**

_Solve problems correctly the first time_. When you encounter a potential latency spike or algorithmic issue, fix it now — don't defer. The second pass may never come.

### 2. **Static Memory Allocation**

This is huge for a GPU compute library:

- **No dynamic allocation after initialization**
- Pre-allocate all Metal shared buffers (params, grads, activations) at startup
- Use fixed-capacity arrays/pools instead of growing `ArrayList`s during training
- Use comptime-known sizes from `NetworkLayout` to determine buffer capacities at compile time

For nnzap, this means: parameter buffers, gradient buffers, and activation buffers should have fixed upper bounds allocated at init time. This eliminates allocation jitter during forward/backward passes and prevents GPU stalls caused by buffer reallocation.

### 3. **Assertion Density**

**Minimum 2 assertions per function**. For nnzap:

- Assert buffer sizes match comptime-known layout dimensions before kernel dispatch
- Assert matrix dimensions are compatible before matmul (`W` is `[M x K]`, `x` is `[K x N]`)
- **Pair assertions**: assert parameter slice bounds when writing from CPU AND when binding to GPU encoder
- Assert compile-time constants (e.g., `comptime { std.debug.assert(Layout.param_count > 0); }`)
- Assert thread grid sizes do not exceed Metal device limits before dispatch
- **Split compound assertions**: prefer `assert(a); assert(b);` over `assert(a and b);` — the former is simpler to read, and provides more precise information if the condition fails
- Use single-line `if` to assert an implication: `if (a) assert(b);`
- On occasion, use a blatantly true assertion instead of a comment as stronger documentation where the assertion condition is critical and surprising
- **Assertions are not a substitute for human understanding.** Build a precise mental model of the code first, encode your understanding in the form of assertions, write the code and comments to explain and justify the mental model to your reviewer, and use testing/fuzzing as the final line of defense. A fuzzer can prove only the presence of bugs, not their absence

### 4. **Put a Limit on Everything**

Every loop, every queue, every buffer needs a hard cap:

```example.zig
const MAX_LAYERS = 64;
const MAX_PARAMS_PER_NETWORK = 16 * 1024 * 1024; // 16M parameters
const MAX_BATCH_SIZE = 4096;
const MAX_THREADS_PER_DISPATCH = 65536;
const MAX_BUFFER_COUNT = 4; // multi-buffering cap
```

This prevents infinite loops and tail latency spikes. If you hit a limit, **fail fast**.

### 5. **70-Line Function Limit**

Hard limit. Split large compute dispatch or training loop functions by:

- Keeping control flow (switches, ifs) in parent functions
- Moving pure computation to helpers
- "Push ifs up, fors down"
- **Good function shape** is the inverse of an hourglass: a few parameters, a simple return type, and a lot of meaty logic between the braces
- **Centralize state manipulation**: let the parent function keep all relevant state in local variables, and use helpers to compute what needs to change, rather than applying the change directly. Keep leaf functions pure

### 6. **Explicit Control Flow**

- No recursion (use explicit iteration over layer indices — network graphs are sequential)
- Minimize abstractions ("abstractions are never zero cost")
- Avoid `async`/suspend patterns that hide control flow — Metal command buffers already handle asynchrony explicitly
- **Split compound conditions**: split complex `else if` chains into nested `else { if {} }` trees to make branches and cases explicit. Consider whether a single `if` also needs a matching `else` branch to ensure positive and negative spaces are handled or asserted
- **Functions must run to completion** without suspending, so that precondition assertions remain true throughout the lifetime of the function
- **Don't do things directly in reaction to external events.** Your program should run at its own pace — not react to external signals or callbacks synchronously. This keeps control flow under your control, improves performance through batching instead of context switching on every event, and makes it easier to maintain bounds on work done per time period. For nnzap, this means the training loop drives the tempo: it pulls batches and dispatches kernels on its schedule, not in reaction to Metal completion handlers or data-loading callbacks

### 7. **Back-of-Envelope Performance Sketches**

Before implementing, sketch resource usage:

- How many floats per forward pass? (GPU memory bandwidth)
- How many kernel dispatches per training step? (command buffer overhead)
- How many bytes transferred between CPU and GPU? (should be zero on unified memory)
- What is the arithmetic intensity of each kernel? (compute-bound vs memory-bound)

Optimize for **network → disk → memory → CPU** (slowest first), adjusted for frequency. On Apple Silicon unified memory, the CPU↔GPU copy cost is zero — but cache coherency and memory bandwidth still matter. A matmul kernel bottlenecked on memory bandwidth won't benefit from more ALUs.

### 8. **Batching as Religion**

We're already doing this with Metal command buffers, but:

- Don't dispatch one kernel per operation — encode multiple kernels into a single command buffer
- Amortize command buffer creation across training steps
- Let the GPU sprint on large batches, not stall on tiny dispatches
- **Distinguish control plane from data plane**: a clear delineation through batching enables a high level of assertion safety without losing performance. Assert heavily on the control plane (layer setup, buffer binding); batch tightly on the data plane (kernel dispatches, gradient accumulation)

### 9. **Naming Discipline**

- Units/qualifiers last: `count_elements`, `offset_bytes`, `size_floats`
- Same-length related names for visual alignment: `source`/`target` not `src`/`dest`; `params`/`grads_` (or use `weight`/`gradient`)
- Callbacks go last in parameter lists
- Use `snake_case` for function, variable, and file names. The underscore is the closest thing we have as programmers to a space, and helps separate words and encourage descriptive names
- Use proper capitalization for acronyms: `MSELoss`, not `MseLoss`; `GPUBuffer`, not `GpuBuffer`
- **Do not abbreviate** variable names (unless the variable is a primitive integer in a sort function or matrix calculation)
- **Infuse names with meaning**: `gpa: Allocator` and `arena: Allocator` are better than `allocator: Allocator` — they inform the reader whether `deinit` should be called
- **Don't overload names** with multiple meanings that are context-dependent — `buffer` could mean a Metal buffer or a Zig slice; be specific (`metal_buffer`, `param_slice`)
- **Think of how names read outside the code** — in docs or conversation. Nouns compose better than adjectives or participles: `pipeline` over `preparing`
- **Named arguments via `options: struct`** pattern when arguments can be mixed up. A function taking two `u64`s must use an options struct
- **Struct field ordering**: fields first, then types, then methods. Important things near the top. `main` goes first in a file
- **Descriptive commit messages** that inform and delight the reader. PR descriptions are not stored in git history and are invisible in `git blame` — they are not a replacement for commit messages

### 10. **Shrink Scope Aggressively**

- Declare variables at smallest possible scope
- Calculate/check values close to use (avoid POCPOU bugs)
- Don't leave variables around after they're needed
- **Don't duplicate variables or take aliases** — this reduces the probability that state gets out of sync. In particular, don't cache a buffer's `.asSlice()` result across a GPU submission boundary — the slice is a view into shared memory and the semantics change after `commitAndWait`
- **Simpler return types reduce dimensionality**: `void` > `bool` > `u64` > `?u64` > `!u64`. Simpler signatures mean fewer branches at call sites, and this dimensionality is **viral** through call chains

### 11. **Handle the Negative Space**

For every valid state you handle, assert the invalid states too:

```example.zig
if (layer_index < Layout.num_layers) {
    // Valid — dispatch the kernel for this layer.
} else {
    unreachable; // Assert we never get here.
}
```

**State invariants positively.** Negations are error-prone. This form is easy to get right:

```example.zig
if (index < length) {
    // The invariant holds.
} else {
    // The invariant doesn't hold.
}
```

This form is harder, and goes against the grain of how `index` is typically compared to `length`:

```example.zig
if (index >= length) {
    // It's not true that the invariant holds.
}
```

Prefer the positive form.

### 12. **Minimal Dependencies**

nnzap depends on:

- **zig-objc** — Objective-C runtime bindings for Metal API calls (necessary; Metal has no C API)
- **Apple Metal framework** — the GPU compute backend (the whole point)

Beyond these, avoid pulling in external Zig packages when you can implement cleanly yourself. Each dependency is a supply chain risk. Linear algebra helpers, activation functions, and data loading should be implemented in-house — they're small and domain-specific.

### 13. **In-Place Initialization**

For large structs (like `Device` with its pre-compiled pipelines), use out-pointers:

```example.zig
pub fn init(self: *Device) !void {
    self.* = .{ ... };  // No stack copy.
}
```

This avoids stack growth and copy-move allocations. In-place initializations are **viral** — if any field is initialized in-place, the entire container struct should be initialized in-place as well.

**Pass large args as `*const`**: if an argument type is more than 16 bytes and you don't intend for it to be copied, pass it as `*const`. This catches bugs where the caller makes an accidental copy on the stack before calling the function.

### 14. **Comptime All the Things**

nnzap's `NetworkLayout` resolves all buffer sizes, weight/bias offsets, and activation sizes at compile time. Lean into this:

- **Never compute at runtime what can be computed at comptime**. Layer dimensions, parameter counts, and buffer offsets are all known statically
- **Use comptime assertions** to validate network architecture (e.g., layer `i`'s output must match layer `i+1`'s input)
- **Comptime slice helpers** like `getWeightSlice` and `getBiasSlice` should compile down to pointer arithmetic with zero branching
- When adding new layout features, ensure they remain `comptime`-evaluable — no allocators, no runtime state

### 15. **Explicitly-Sized Types**

Use explicitly-sized types like `u32` for everything. Avoid architecture-specific `usize` unless interfacing with Zig's standard library or slice indexing that requires it. Explicit sizes make overflow behavior, memory layout, and cross-platform behavior predictable and consistent. This is especially important when passing dimensions to Metal shaders via `setBytes`, where the GPU expects exact sizes.

### 16. **Always Say Why**

Code alone is not documentation.

- **Never forget to say why.** If you explain the rationale for a decision, it increases the reader's understanding and shares criteria with which to evaluate the decision and its importance
- Comments are sentences: space after the slash, capital letter, full stop (or a colon if they relate to something that follows)
- Comments after the end of a line _can_ be phrases, with no punctuation
- When writing a test, describe the goal and methodology at the top — help the reader get up to speed or skip over sections without forcing them to dive in

### 17. **All Errors Must Be Handled**

92% of catastrophic failures in distributed data-intensive systems are the result of incorrect handling of non-fatal errors explicitly signaled in software ([OSDI'14](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-yuan.pdf)). Every error return must be handled — no silent discards, no blind `catch unreachable` without justification.

Metal errors are particularly important: shader compilation failures, pipeline creation failures, and buffer allocation failures must all produce actionable diagnostics (as already done with `log.err` in `compileLibrary` and `ComputePipeline.init`).

### 18. **Off-By-One Discipline**

The usual suspects for off-by-one errors are casual interactions between an `index`, a `count`, and a `size`. Treat them as distinct conceptual types with clear conversion rules:

- `index` → `count`: add one (indexes are 0-based, counts are 1-based)
- `count` → `size`: multiply by the unit (e.g., `param_count * @sizeOf(f32)` for byte size)

Show division intent explicitly: use `@divExact()`, `@divFloor()`, or `div_ceil()` to prove to the reader you've thought through rounding scenarios. This is critical for kernel thread grid calculations where misaligned dispatch sizes cause silent data corruption.

### 19. **Explicit Options at Call Sites**

Explicitly pass options to library functions at the call site instead of relying on defaults:

```example.zig
// Prefer:
@prefetch(a, .{ .cache = .data, .rw = .read, .locality = 3 });

// Over:
@prefetch(a, .{});
```

This improves readability and avoids latent, potentially catastrophic bugs if the library ever changes its defaults. The same applies to Metal resource options — always spell out `.storage_mode = .shared` rather than relying on the default.

### 20. **Hot Loop Extraction**

Extract hot loops into standalone functions with primitive arguments — no `self`. This way the compiler doesn't need to prove it can cache struct fields in registers, and a human reader can spot redundant computations more easily.

```example.zig
// Instead of a method that accesses self.field in a tight loop:
fn fillBufferLinear(slice: []f32, scale: f32, offset: f32) void {
    // Tight loop with only primitives — no pointer chasing through self.
    for (slice, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * scale + offset;
    }
}
```

This is especially relevant for CPU-side data preparation (batch loading, normalization) that runs while the GPU crunches the previous batch.

### 21. **Buffer Bleeds**

The inverse of a buffer overflow: a buffer underflow where a buffer is not fully utilized and padding is not zeroed correctly. This can leak sensitive information and break deterministic guarantees. When writing to Metal shared buffers, zero-fill any padding between used regions — stale data in a gradient buffer from a previous batch can silently corrupt training.

### 22. **Resource Allocation Grouping**

Use newlines to **group resource allocation and deallocation** — place a blank line before the allocation and after the corresponding `defer` statement. This makes leaks visually obvious:

```example.zig
var param_buf = try device.createBuffer(Layout.param_count);
defer param_buf.deinit();

var grad_buf = try device.createBuffer(Layout.param_count);
defer grad_buf.deinit();

var activations = try device.createMultiBuffered(2, Layout.max_activation_size);
defer for (&activations.buffers) |*b| b.deinit();

// ... use buffers ...
```

### 23. **Style By The Numbers**

- Run `zig fmt`.
- Use 4 spaces of indentation, rather than 2 spaces — more obvious to the eye at a distance.
- **Hard limit all line lengths to 100 columns**, without exception. Nothing should be hidden by a horizontal scrollbar. Let your editor help you by setting a column ruler. To wrap a function signature, call, or data structure, add a trailing comma, and let `zig fmt` do the rest. The motivation is physical: just enough to fit two copies of the code side-by-side on a screen.
- Add braces to `if` statements unless the entire statement fits on a single line — for consistency and defense in depth against "goto fail;" bugs.

### 24. **Compiler Warnings**

Appreciate **all compiler warnings at the compiler's strictest setting** from day one. Warnings are free bug reports. Never suppress them — fix the underlying issue.

### 25. **Tooling**

Our primary tool is Zig. It may not be the best for everything, but it's good enough for most things. When you need a script, write it in `zap/src/*.zig` instead of as a shell script — this makes scripts cross-platform, type-safe, and increases the probability they'll work for everyone on the team. Standardizing on Zig for tooling reduces dimensionality as the team and range of personal tastes grows.

> "The right tool for the job is often the tool you are already using — adding new tools has a higher cost than many people appreciate." — John Carmack

### 26. **GPU/CPU Boundary Discipline**

The CPU/GPU boundary is where the hardest bugs hide:

- **Never read from a Metal shared buffer while the GPU may be writing to it.** Always `commitAndWait` (or use a fence/semaphore) before reading results back on the CPU
- **Double-buffering exists for a reason**: the CPU preps batch N+1 while the GPU computes batch N. Violating this overlap invariant causes data races that are invisible to Zig's safety checks
- **Assert buffer binding order**: Metal shader `[[buffer(N)]]` indices must match the `setBuffer` calls. A mismatch silently feeds wrong data to the kernel
- **Validate dispatch dimensions**: a 2D dispatch for matmul must match the output matrix dimensions. An undersized grid silently skips elements; an oversized grid reads garbage
