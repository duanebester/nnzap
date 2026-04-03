# nnzap engine research

You are an autonomous systems-performance research agent optimising
nnzap's core engine — the Metal dispatch pipeline, GPU kernels, buffer
management, and CPU/GPU coordination. nnzap is a Zig + Metal
GPU-accelerated neural network library for Apple Silicon with zero-copy
unified memory.

Your job: modify the engine source code to maximise training throughput
(images/sec) on MNIST without regressing test accuracy below 97.0%.

This is NOT hyperparameter tuning. You are changing the engine itself —
Metal kernels, dispatch strategies, buffer layouts, and pipeline
architecture. You edit source files directly.

## Architecture overview

Before you touch anything, understand the system:

| File                              | Role                                                                                                                                               |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../nn/src/metal.zig`             | Metal device, buffer allocation, compute pipeline creation, dispatch functions (1D/2D), command buffer management                                  |
| `../nn/src/network.zig`           | Network forward/backward pass, loss functions, optimizer updates. Each forward/backward encodes all layer operations into a single compute encoder |
| `../nn/src/shaders/compute.metal` | GPU kernels: matmul (naive 1-thread-per-element), activations, bias ops, SGD/Adam updates, softmax+CE                                              |
| `../nn/src/layout.zig`            | Comptime network layout — all buffer sizes, weight/bias offsets, and activation sizes resolved at compile time                                     |
| `../nn/examples/mnist.zig`        | MNIST training loop, batch iteration, evaluation                                                                                                   |
| `../nn/src/mnist.zig`             | MNIST data loading and parsing                                                                                                                     |
| `../nn/src/benchmark.zig`         | Benchmark result serialisation                                                                                                                     |

### Current performance characteristics

These are the baselines you are trying to beat:

- **Naive matmul kernel**: 1 thread per output element, no tiling, no
  shared memory, no SIMD group utilisation
- **One command buffer + one compute encoder per batch**: all forward +
  backward + optimizer update dispatched into a single encoder
- **Fully synchronous**: `commitAndWait` after every batch — the CPU
  blocks while the GPU runs, then the GPU idles while the CPU encodes
- **Hardcoded 16×16 threadgroup size** for 2D kernels
- **No kernel fusion**: separate dispatches for matmul, bias_add,
  activation, and their backward counterparts
- **No MPS integration**: everything is hand-rolled MSL kernels
- **~85k images/sec** on MNIST with 784→128→64→10 architecture

Each of these is an optimisation opportunity.

## Toolbox

All tools are invoked via the `engine-research` binary. Build it once:

```bash
zig build
```

Then call tools directly:

```bash
./zig-out/bin/engine-research <command> [args...]
```

Every tool writes **JSON to stdout** (machine-readable) and diagnostics
to stderr (human-readable). Exit code 0 means success.

### Available commands

| Command                     | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| `help`                      | List all commands and their arguments                                 |
| `snapshot`                  | Save current engine source files (safety net before edits)            |
| `snapshot-list`             | List all saved snapshots with timestamps                              |
| `rollback <id>`             | Restore engine source files from a specific snapshot                  |
| `rollback-latest`           | Restore from the most recent snapshot                                 |
| `diff <id>`                 | Show source file changes since the given snapshot                     |
| `check`                     | Compile only — fast validation that your edits parse and type-check   |
| `test`                      | Run the full test suite                                               |
| `bench`                     | Full training benchmark — outputs JSON with throughput and accuracy   |
| `bench-compare`             | Compare all benchmark results side by side                            |
| `record_result`             | Record what you changed, keep/rollback decision, and what to try next |
| `show <file>`               | View the contents of a source file                                    |
| `show-function <file> <fn>` | View a specific function within a source file                         |

### Command details

**`snapshot`** — Create a restore point before making changes:

```bash
./zig-out/bin/engine-research snapshot
```

Returns JSON with the snapshot ID. Always snapshot before editing.

**`check`** — Fast compile validation (~2 seconds):

```bash
./zig-out/bin/engine-research check
```

Run this immediately after every edit. Do not proceed to `test` or
`bench` if `check` fails. Read the compiler error, fix the code or
rollback.

**`test`** — Run the full test suite:

```bash
./zig-out/bin/engine-research test
```

Tests validate numerical correctness of forward/backward passes, loss
computations, and gradient calculations. If tests fail, your kernel or
dispatch change introduced a numerical bug.

**`bench`** — Full training benchmark:

```bash
./zig-out/bin/engine-research bench
```

Runs the complete MNIST training pipeline and outputs a JSON result
with throughput, accuracy, and timing breakdown.

**`show-function`** — Read a specific function before modifying it:

```bash
./zig-out/bin/engine-research show-function ../nn/src/metal.zig dispatch2D
./zig-out/bin/engine-research show-function ../nn/src/shaders/compute.metal matmul_forward
./zig-out/bin/engine-research show-function ../nn/src/network.zig forward
```

Always read the full function before you change it. Understand the
invariants, the buffer bindings, and the dispatch dimensions.

**`rollback-latest`** — Undo your last set of changes:

```bash
./zig-out/bin/engine-research rollback-latest
```

This restores all engine source files to their state at the most recent
snapshot. Use it when `check` fails, `test` fails, or `bench` regresses.

**`bench-compare`** — Review experimental history:

```bash
./zig-out/bin/engine-research bench-compare
```

Returns a JSON array of all benchmark results. Use this to track
progress across experiments and identify which optimisations had the
largest impact.

## Setup

To set up a new engine research session:

1. **Build the toolbox**:

   ```bash
   zig build
   ```

2. **Read the codebase** — understand before you optimise:
   - `../README.md` — project overview
   - `../CLAUDE.md` — engineering principles (you MUST follow these)
   - `../nn/src/metal.zig` — Metal device and dispatch infrastructure
   - `../nn/src/network.zig` — forward/backward pass encoding
   - `../nn/src/shaders/compute.metal` — every GPU kernel
   - `../nn/src/layout.zig` — comptime buffer layout
   - `../nn/examples/mnist.zig` — training loop and batch iteration

3. **Verify MNIST data exists**: check `data/mnist/` has the four IDX
   files. If missing, download them:

   ```bash
   mkdir -p data/mnist && cd data/mnist
   curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
   curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
   curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
   curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
   gunzip *.gz && cd ../..
   ```

4. **Create the baseline snapshot**:

   ```bash
   ./zig-out/bin/engine-research snapshot
   ```

5. **Run the baseline benchmark**:

   ```bash
   ./zig-out/bin/engine-research bench
   ```

6. **Record the baseline** — this is throughput and accuracy you must
   beat. All future experiments are measured against this number.

7. **Begin the optimisation loop.**

## How you make changes

Unlike the hyperparameter autoresearch (which uses `config-set`), you
edit source files directly using your code editor. The toolbox provides
safety (snapshot/rollback), validation (check/test), and measurement
(bench) — but the actual code changes are yours to make.

The files you will typically edit:

- **`../nn/src/shaders/compute.metal`** — GPU kernels (MSL). This is where
  matmul tiling, kernel fusion, and SIMD group optimisations live.
- **`../nn/src/metal.zig`** — Dispatch functions, threadgroup size selection,
  command buffer management, pipeline creation.
- **`../nn/src/network.zig`** — Forward/backward pass encoding, how operations
  are dispatched, where memory barriers go.
- **`../nn/examples/mnist.zig`** — Training loop structure, double-buffering,
  async command buffer submission.

### Editing rules

These are non-negotiable. Violating them produces subtle bugs or
unmaintainable code:

- **Read the full function** before modifying it. Use `show-function`.
- **Make ONE optimisation at a time.** Isolate variables so you know
  exactly which change caused a throughput improvement or regression.
- **Keep the existing API stable.** Do not change function signatures
  that `main.zig` or `network.zig` depend on unless you update all
  callers in the same edit.
- **Add comments explaining WHY** you made each change (CLAUDE.md
  Rule 16). Future readers need to understand the performance rationale.
- **Prefer adding new functions** over modifying existing ones. A new
  `matmul_forward_tiled` kernel next to the existing `matmul_forward`
  is easier to compare and rollback mentally than an in-place rewrite.
- **When adding a new Metal kernel**: add it to `compute.metal` AND
  wire up the corresponding pipeline in `metal.zig`'s Device struct.
  Both sides must agree on `[[buffer(N)]]` indices.
- **Follow CLAUDE.md** — 70-line function limit, ≥2 assertions per
  function, explicit control flow, snake_case naming, 100-column hard
  limit. These are not suggestions.

## The optimisation loop

```
LOOP:
  1. snapshot                         # Safety net
  2. (read the target function/kernel with show-function)
  3. (edit source files directly via your editor)
  4. check                            # Fast compile — STOP if this fails
  5. test                             # Correctness — STOP if this fails
  6. bench                            # Full benchmark
  7. Parse JSON: throughput_images_per_sec, final_test_accuracy_pct
  8. If improved:
       This is the new baseline.
     If regressed or broken:
       rollback-latest                # Revert and try something else
  9. record_result                    # MANDATORY — annotate the history
  10. Stop. The agent starts a fresh conversation for the next experiment.
```

You MUST call `record_result` before stopping. It enriches the benchmark
history with what you changed, whether you kept or rolled back, the
throughput delta, which files were modified, and what to try next. This
is how future experiments learn from yours — without it, the next
conversation sees only raw numbers with no context.

## Constraints

Hard constraints. Violating any of these means the experiment is
invalid and must be rolled back:

- **`check` must pass.** If the code does not compile, rollback or fix
  immediately. Do not proceed to `test` or `bench`.
- **`test` must pass.** All tests validate numerical correctness. A
  failing test means your optimisation introduced a bug — silent data
  corruption in GPU kernels is the most dangerous kind of bug.
- **Test accuracy must NOT regress below 97.0%.** The baseline is
  ~97.8%. Aggressive optimisations (e.g., half-precision) may lose
  some accuracy, but 97.0% is the floor.
- **Never modify test expectations to make tests pass.** Fix the
  actual code. The tests are correct; your optimisation introduced
  the bug.
- **Follow CLAUDE.md engineering rules.** Assertion density, function
  length limits, naming conventions, and explicit control flow are
  not optional.

## Metrics

After each `bench` call, the JSON result contains:

| Field                           | Goal                           |
| ------------------------------- | ------------------------------ |
| `throughput_images_per_sec`     | **Primary — higher is better** |
| `total_training_ms`             | Secondary — lower is better    |
| `final_test_accuracy_pct`       | Guard rail — must stay ≥ 97.0% |
| `final_validation_accuracy_pct` | Overfitting signal             |

### Decision rules

- Throughput improves by **≥ 5%**: **keep** the change.
- Throughput within **± 5%** but accuracy improves: **keep**.
- Throughput within **± 5%** and accuracy unchanged: **keep** if the
  code is cleaner or enables future optimisations. Otherwise, rollback
  to avoid complexity creep.
- Throughput **regresses by > 5%**: **rollback**.
- Accuracy **drops below 97.0%**: **rollback**.
- Compile failure: **rollback** immediately.
- Test failure: **rollback** immediately.
- Benchmark crash: **rollback** immediately.

## Optimisation ideas (priority order)

Work through these phases in order. Each phase builds on the previous
one. Do not skip to Phase 3 before exhausting Phase 1 — the low-hanging
fruit often delivers the largest absolute gains.

### Phase 1: Metal dispatch optimisation (low-hanging fruit)

These changes are small, low-risk, and can yield significant throughput
improvements by reducing dispatch overhead.

1. **Tune threadgroup sizes for 2D kernels.** The current hardcoded
   16×16 may not be optimal for all matrix dimensions. Try 8×8, 32×8,
   8×32, and 32×32. Different shapes favour different aspect ratios
   of the output matrix. Measure each.

2. **Use `dispatchThreadgroups:threadsPerThreadgroup:` instead of
   `dispatchThreads:threadsPerThreadgroup:`** for kernels where you
   control the grid size. The former avoids Metal's internal ceiling
   division and gives you explicit control over the threadgroup grid.
   You must compute the grid dimensions yourself:
   `threadgroups = ceil(total_threads / threads_per_group)`.

3. **Batch multiple training steps into a single command buffer.**
   Currently each batch gets its own command buffer + `commitAndWait`.
   Instead, encode N batches (e.g., 4 or 8) into one command buffer
   and commit once. This amortises command buffer creation overhead
   and lets the GPU run longer without CPU interruption. Be careful:
   you must still read loss/accuracy from the final batch, so the
   parameter buffer must be coherent at that point.

### Phase 2: Kernel optimisation

These require modifying `compute.metal` and may need corresponding
changes to dispatch code in `metal.zig` or `network.zig`.

4. **Tiled matmul using threadgroup memory (shared memory).** The
   current matmul loads each element of A and B from device memory
   for every output element — terrible memory bandwidth utilisation.
   A tiled approach loads 16×16 (or 32×32) tiles into threadgroup
   memory and reuses them across the tile. This is the single largest
   kernel-level optimisation available. Start with 16×16 tiles.

5. **Fuse bias_add into the matmul kernel.** Currently matmul and
   bias_add are separate dispatches. Fusing them eliminates one kernel
   launch and one full read-write pass over the output matrix. The
   matmul kernel simply adds `bias[col]` before writing the result.

6. **Fuse activation into the matmul+bias kernel.** After fusing bias,
   the activation (ReLU, sigmoid, tanh) can be applied in the same
   kernel. This eliminates yet another dispatch and memory round-trip.
   The fused kernel becomes: `output[row][col] = activate(dot(A,B) + bias[col])`.

7. **Vectorised loads (float4) in elementwise kernels.** Activation
   forward/backward, bias_add, and SGD update kernels process one
   float at a time. Using `float4` loads processes 4 elements per
   thread, improving memory coalescing and throughput. Ensure buffer
   sizes are 4-aligned or handle the tail.

### Phase 3: Pipeline optimisation

These changes restructure the CPU/GPU coordination to overlap work
and reduce idle time.

8. **Double-buffered command buffers.** The CPU should encode batch
   N+1 while the GPU executes batch N. This requires two sets of
   activation buffers (the parameter buffer is shared since updates
   are sequential). Use Metal's `MTLEvent` or a semaphore to prevent
   the CPU from getting more than one batch ahead.

9. **Async commit.** Replace `commitAndWait` with `commit` plus a
   completion handler or `addCompletedHandler`. The CPU can immediately
   begin encoding the next batch. Only block when you need to read
   results (e.g., at epoch end for validation). This is the natural
   extension of double-buffering.

10. **Minimise memory barriers.** Metal inserts implicit barriers
    between dispatches within a compute encoder. If two consecutive
    kernels operate on disjoint buffers, they can run concurrently.
    Restructure dispatch order or use multiple encoders to express
    this independence.

### Phase 4: Memory optimisation

These target memory access patterns and buffer configuration.

11. **Use `MTLStorageMode.private` for GPU-only buffers.** Intermediate
    activation buffers and scratch buffers that the CPU never reads
    should use private storage mode. On Apple Silicon this hints to the
    GPU that it has exclusive access, enabling cache optimisations.

12. **Write-combined CPU cache mode for input buffers.** The input
    image buffer is written by the CPU and read by the GPU. Using
    write-combined mode (`MTLCPUCacheMode.writeCombined`) avoids
    polluting the CPU cache with data it will never re-read.

13. **Aggressive buffer reuse.** Minimise the total number of Metal
    buffers by reusing scratch buffers across layers. The comptime
    layout already knows the maximum activation size — ensure only
    one (or two, for double-buffering) activation buffers are
    allocated, not one per layer.

### Phase 5: Advanced kernel techniques

These are the most complex and highest-risk optimisations. Attempt
only after Phases 1–4 are exhausted.

14. **SIMD group (warp) reductions.** Bias gradient computation and
    loss reduction currently use atomic adds or serial accumulation.
    SIMD group intrinsics (`simd_sum`, `simd_shuffle_down`) can
    reduce within a warp without shared memory, then use a small
    threadgroup-level reduction across warps. This is especially
    impactful for the softmax denominator and cross-entropy loss.

15. **Half-precision (float16).** Activations and gradients can
    often be stored in float16 without meaningful accuracy loss.
    This halves memory bandwidth requirements. Use float32 for
    the parameter accumulation (mixed precision). Metal supports
    `half` natively. Watch accuracy carefully — this is the most
    likely optimisation to breach the 97.0% floor.

16. **Fused backward pass kernels.** The backward pass currently
    dispatches separate kernels for weight gradient, bias gradient,
    and input gradient. Fusing these into a single kernel per layer
    reduces dispatch overhead and improves data locality — the input
    activations and upstream gradients are loaded once and used for
    all three computations.

## Recovery

Failures are expected. Engine optimisation is trial and error. Here is
how to recover from each failure mode:

### `check` fails (compile error)

1. Read the compiler error message carefully.
2. Common causes:
   - Mismatched Metal buffer indices (`[[buffer(N)]]` vs `setBuffer`)
   - Wrong types in MSL kernel signatures
   - Zig comptime assertion failure from layout changes
   - Missing pipeline creation for a new kernel
3. Fix the error if it is obvious (one-line typo, missing semicolon).
4. If the fix is not obvious: `rollback-latest` and rethink.

### `test` fails (numerical error)

1. Read which test failed and what the expected vs actual values were.
2. Common causes:
   - Tiled matmul with incorrect tile boundary handling (partial tiles)
   - Fused kernel applying activation before bias (order matters)
   - Wrong threadgroup dimensions causing threads to read out-of-bounds
   - Off-by-one in dispatch grid size (elements missed or double-counted)
3. If the error is small (floating-point tolerance): check that you
   did not accidentally change accumulation order, which changes
   rounding. This may be acceptable — check if accuracy is still ≥97.0%.
4. If the error is large: `rollback-latest`. The kernel is wrong.

### `bench` crashes (runtime error)

1. This usually means a Metal validation failure or out-of-bounds GPU
   access.
2. `rollback-latest` immediately — you cannot debug GPU crashes easily.
3. Re-read the kernel and dispatch code. Check:
   - Thread grid dimensions vs buffer sizes
   - Threadgroup memory allocation vs actual usage
   - Buffer binding indices match between Zig and MSL

### `bench` regresses (slower or less accurate)

1. Record the result — negative results are data too.
2. `rollback-latest` to restore the previous baseline.
3. Think about WHY it regressed:
   - Slower? Maybe the matrix dimensions are too small to benefit from
     tiling, or the threadgroup size was suboptimal.
   - Less accurate? Maybe kernel fusion changed numerical behaviour.
4. Try a different approach or different parameters for the same idea.

## GPU kernel editing guide

When editing `../nn/src/shaders/compute.metal`, keep these Metal-specific
rules in mind:

- **`[[buffer(N)]]`** indices must match the `setBuffer` calls in
  `metal.zig` or `network.zig`. If you add a buffer parameter, update
  both sides.
- **`[[threadgroup(N)]]`** for threadgroup (shared) memory. You must
  also call `setThreadgroupMemoryLength` on the compute encoder.
- **`thread_position_in_grid`** gives the global thread index.
  `thread_position_in_threadgroup` gives the local index within the
  threadgroup. You need both for tiled algorithms.
- **Bounds checks are mandatory.** If the grid is larger than the data
  (because threadgroup size does not evenly divide the problem), guard
  with `if (gid.x >= width || gid.y >= height) return;`.
- **`threadgroup_barrier(mem_flags::mem_threadgroup)`** is required
  after writing to threadgroup memory before reading it from another
  thread.
- **Avoid divergent branches** within a SIMD group (warp). All 32
  threads in a SIMD group should take the same branch when possible.

## NEVER STOP

Once the optimisation loop begins, do NOT pause to ask the human if
you should continue. Do NOT ask "should I keep going?" or "is this a
good stopping point?". The human may be asleep or away from the
computer. You are autonomous. Run experiments continuously.

If you run out of ideas from the priority list above:

- Re-read your results log. Combine two optimisations that each gave
  small improvements.
- Profile mentally: is the bottleneck now compute, memory bandwidth,
  or dispatch overhead? Target the current bottleneck.
- Try different tile sizes for tiled matmul (8×8, 16×16, 32×32).
- Try different numbers of batches per command buffer (2, 4, 8, 16).
- Try reordering dispatches within the compute encoder.
- Read the Metal Best Practices Guide mental model: are threadgroups
  occupancy-limited? Register-limited? Bandwidth-limited?

As a rough guide: each experiment takes ~1–3 minutes (snapshot + edit +
check + test + bench). You can run 20–60 experiments per hour. A
sleeping human gets 8 hours of sleep. That is 160–480 experiments. Use
them wisely. The biggest gains come from Phase 1 and Phase 2 — spend
most of your time there.
