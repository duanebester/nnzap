# ⚡ nnzap

GPU-accelerated neural network library for Apple Silicon, written in Zig.

**Zero-copy. Comptime layouts. Metal compute.**

## Why

On Apple Silicon, CPU and GPU share the same physical memory. Most ML
frameworks still copy buffers around as if they're talking to a discrete
GPU over PCIe. nnzap exploits unified memory directly — the `[]f32`
slice your Zig code writes to _is_ the GPU buffer.

```
Unified Memory  ──[compute]──>  same memory
   zero copy                    zero copy
```

## Architecture

```
┌──────────────────────────────────────────┐
│  Comptime: Network layout                │
│  Sizes, shapes, offsets, buffer sizes    │
│  All resolved at compile time.           │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Metal Shared Buffers (unified memory)   │
│  params[]  grads[]  activations[0..1]    │
│  Double-buffered for CPU/GPU overlap     │
└──────────────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 ┌─────────────┐     ┌──────────────┐
 │  CPU (Zig)  │     │  GPU (Metal) │
 │  prep next  │     │  compute     │
 │  batch      │     │  shaders     │
 └─────────────┘     └──────────────┘
```

## Quick start

### 1. Download MNIST

Fetch and decompress the four dataset files into `data/mnist/`:

```bash
mkdir -p data/mnist && cd data/mnist

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
cd ../..
```

You should now have four files (~55 MB total):

| File                      | Samples        | Format       |
| ------------------------- | -------------- | ------------ |
| `train-images-idx3-ubyte` | 60,000 × 28×28 | uint8 pixels |
| `train-labels-idx1-ubyte` | 60,000         | uint8 [0–9]  |
| `t10k-images-idx3-ubyte`  | 10,000 × 28×28 | uint8 pixels |
| `t10k-labels-idx1-ubyte`  | 10,000         | uint8 [0–9]  |

### 2. Train

```bash
zig build run
```

This trains a 784→128→64→10 network on MNIST with mini-batch SGD
and softmax + cross-entropy loss. The 60k training images are split
into 50k for training and 10k for per-epoch validation:

```
nnzap MNIST benchmark
=====================

Loading MNIST... done.
Split: train=50000  val=10000  test=10000

Training: 10 epochs x 781 batches  (batch=64, lr=0.10)
------------------------------------------------------
  Epoch  1: loss=0.2474  val_loss=0.1952  val_acc=94.03%  (677 ms)
  Epoch  2: loss=0.0757  val_loss=0.1370  val_acc=95.88%  (566 ms)
  ...
  Epoch 10: loss=0.0214  val_loss=0.0786  val_acc=97.54%  (581 ms)
------------------------------------------------------
Total: 5846 ms  (585 ms/epoch, 85523 img/s)

Evaluating test set... 9782/10000 correct (97.82%)  [83 ms]

Benchmark saved: benchmarks/mnist_2025-07-15T14-30-00Z.json
```

### 3. Run tests

```bash
zig build test
```

## Dataset split

The 60k MNIST training set is shuffled once with a deterministic seed,
then partitioned into three non-overlapping sets:

| Set        | Samples | Purpose                               |
| ---------- | ------- | ------------------------------------- |
| Train      | 50,000  | SGD weight updates                    |
| Validation | 10,000  | Per-epoch monitoring (detect overfit) |
| Test       | 10,000  | Final held-out evaluation             |

The validation set is fixed for the entire run. Only the training
indices are re-shuffled each epoch. The test set is the standard
MNIST `t10k` split, never seen during training.

## Benchmarks

Every training run automatically saves a JSON result file to
`benchmarks/`. Each file captures the full configuration and
results so you can track optimisation progress, compare
hyperparameters, and reproduce runs.

### Output location

```
benchmarks/{name}_{timestamp}.json
```

For example: `benchmarks/mnist_2025-07-15T14-30-00Z.json`

### JSON schema

```json
{
  "timestamp_utc": "2025-07-15T14:30:00Z",
  "config": {
    "architecture": [
      { "input_size": 784, "output_size": 128, "activation": "relu" },
      { "input_size": 128, "output_size": 64, "activation": "relu" },
      { "input_size": 64, "output_size": 10, "activation": "none" }
    ],
    "param_count": 109386,
    "batch_size": 64,
    "learning_rate": 0.1,
    "learning_rate_decay": 0,
    "optimizer": "sgd",
    "loss_function": "cross_entropy",
    "num_epochs": 10,
    "seed": 42,
    "train_samples": 50000,
    "validation_samples": 10000,
    "test_samples": 10000
  },
  "final_train_loss": 0.0214,
  "final_validation_loss": 0.0786,
  "final_validation_accuracy_pct": 97.54,
  "final_test_accuracy_pct": 97.82,
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 0.247,
      "duration_ms": 677.0,
      "validation_loss": 0.195,
      "validation_accuracy_pct": 94.03
    },
    {
      "epoch": 2,
      "train_loss": 0.076,
      "duration_ms": 566.0,
      "validation_loss": 0.137,
      "validation_accuracy_pct": 95.88
    }
  ],
  "test_result": {
    "correct": 9782,
    "total": 10000,
    "accuracy_pct": 97.82,
    "duration_ms": 83.0
  },
  "total_training_ms": 5846.0,
  "throughput_images_per_sec": 85523.0
}
```

### Using the benchmark API

The `Benchmark` struct records results during training and writes
them to disk. Wire it into any training loop in three steps:

**1. Initialise with your layout and config:**

```zig
const nnzap = @import("nnzap");
const Benchmark = nnzap.Benchmark;
const nanosToMs = nnzap.benchmark.nanosToMs;

var bench: Benchmark = undefined;
bench.init(MyLayout, .{
    .batch_size = 64,
    .learning_rate = 0.1,
    .learning_rate_decay = 0.0,
    .optimizer = .sgd,
    .loss_function = .cross_entropy,
    .num_epochs = 10,
    .seed = 42,
    .train_samples = 50_000,
    .validation_samples = 10_000,
    .test_samples = 10_000,
});
```

**2. Record epochs and test results:**

```zig
// In the training loop:
const epoch_start = std.time.nanoTimestamp();
const train_loss = trainEpoch(...);
const val = evaluate(...);  // validation set
const epoch_ms = nanosToMs(std.time.nanoTimestamp() - epoch_start);

bench.recordEpoch(.{
    .epoch = epoch + 1,
    .train_loss = train_loss,
    .duration_ms = epoch_ms,
    .validation_loss = val.mean_loss,
    .validation_accuracy_pct = val.accuracy_pct,
});

// After training:
bench.setTrainingTime(total_training_ms);

// After test evaluation:
bench.recordTest(correct, total, eval_ms);
```

**3. Save to disk:**

```zig
try bench.save("mnist");
// → benchmarks/mnist_2025-07-15T14-30-00Z.json
```

The first argument to `save()` is the benchmark name — use it to
group runs by experiment (e.g. `"mnist"`, `"mnist_adam"`,
`"mnist_lr_sweep"`).

### Comparing runs

Benchmark JSON files are designed for programmatic consumption.
Compare runs with any tool that reads JSON:

```bash
# Compare all runs with the autoresearch tool:
./zig-out/bin/autoresearch benchmark-compare
```

## Autoresearch

nnzap includes an autonomous research toolbox for optimising
hyperparameters. Inspired by
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch),
it lets an AI coding agent run experiments while you sleep.

### How it works

A Zig CLI binary (`scripts/autoresearch.zig`) provides tools that
output JSON, designed for AI agents to invoke via terminal:

| Tool                | Description                              |
| ------------------- | ---------------------------------------- |
| `config-show`       | Show current hyperparameters             |
| `config-set K=V...` | Modify hyperparameters in main.zig       |
| `config-backup`     | Backup main.zig before editing           |
| `config-restore`    | Restore main.zig from backup             |
| `train`             | Build + run training, output JSON result |
| `benchmark-compare` | Compare all benchmark runs               |
| `benchmark-latest`  | Output latest benchmark                  |
| `clean`             | Delete old benchmark files               |

### Quick start

```bash
# 1. Build the toolbox
zig build

# 2. Check current config
./zig-out/bin/autoresearch config-show

# 3. Run a training experiment
./zig-out/bin/autoresearch train

# 4. Modify and re-run
./zig-out/bin/autoresearch config-backup
./zig-out/bin/autoresearch config-set lr=0.001 optimizer=adam
./zig-out/bin/autoresearch train

# 5. Revert if worse
./zig-out/bin/autoresearch config-restore
```

### Agent mode

Point an AI coding agent (Claude Code, Codex, etc.) at
`scripts/program.md` and let it run. The agent reads the current
config, modifies hyperparameters, trains, evaluates, and loops —
keeping improvements and reverting regressions. Each experiment
takes ~1 minute, yielding ~60 experiments/hour.

See [scripts/program.md](scripts/program.md) for the full agent
instructions.

## Agent

The agent is a Zig binary that talks to Claude via the Anthropic
API. Claude decides which experiments to run; the agent executes
them using the autoresearch toolbox, sends results back, and loops.
The agent IS the runtime — like Claude Code, but ours.

```
┌──────────────────┐
│  Claude (API)     │  decides experiments
└────────┬─────────┘
         │ HTTP (curl)
         ▼
┌──────────────────┐
│  agent.zig        │  executes tools, manages history
└────────┬─────────┘
         │ spawns
         ▼
┌──────────────────┐
│  autoresearch.zig │  config/train/benchmark
└──────────────────┘
```

### Quick start

```bash
# 1. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Build and run
zig build
./zig-out/bin/agent
```

The agent seeds Claude with a system prompt (constraints, strategy,
decision rules) and injects experiment history from previous runs
so Claude avoids repeating past work. Each `train` result is
appended to `.agent_history/experiments.jsonl` immediately —
context accumulates across runs.

### Output

- **stderr** — Claude's reasoning, tool calls, training progress.
- **`.agent_history/`** — JSONL experiment log + conversation logs.

See [scripts/agent_program.md](scripts/agent_program.md) for full
documentation.

## Engine research

Beyond hyperparameter tuning, nnzap includes a second autonomous
research toolbox for optimising the engine itself — Metal dispatch
strategies, GPU kernels, buffer management, and CPU/GPU pipeline
coordination.

### How it works

An AI coding agent edits the core source files directly
(`metal.zig`, `network.zig`, `compute.metal`, `layout.zig`),
using snapshot/rollback for safety and automated benchmarking
to measure the impact of each change.

A Zig CLI binary (`scripts/engine_research.zig`) provides the
safety net and measurement tools:

| Tool                        | Description                                  |
| --------------------------- | -------------------------------------------- |
| `snapshot`                  | Save all engine source files (restore point) |
| `snapshot-list`             | List all saved snapshots                     |
| `rollback <id>`             | Restore engine files from a snapshot         |
| `rollback-latest`           | Restore from most recent snapshot            |
| `diff <id>`                 | Show changes since a snapshot                |
| `check`                     | Compile-only validation (~2s)                |
| `test`                      | Run full test suite                          |
| `bench`                     | Full training benchmark (JSON output)        |
| `bench-compare`             | Compare all benchmark results                |
| `show <file>`               | View a source file                           |
| `show-function <file> <fn>` | Extract a specific function body             |

### Quick start

```bash
# 1. Build the toolbox
zig build

# 2. Create a baseline snapshot
./zig-out/bin/engine_research snapshot

# 3. Run baseline benchmark
./zig-out/bin/engine_research bench

# 4. (Agent edits source files directly)

# 5. Fast compile check
./zig-out/bin/engine_research check

# 6. Run tests
./zig-out/bin/engine_research test

# 7. Benchmark the change
./zig-out/bin/engine_research bench

# 8. If worse, rollback
./zig-out/bin/engine_research rollback-latest
```

### Agent mode

The engine agent is a standalone Zig binary that talks to Claude
via the Anthropic API — just like the hyperparameter agent, but
for engine optimisation. Claude reads source code, edits files
directly, validates with check/test, benchmarks, and keeps or
reverts each experiment autonomously.

```bash
# 1. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Build and run
zig build
./zig-out/bin/engine_agent
```

See [scripts/engine_program.md](scripts/engine_program.md) for
the full agent instructions.

### Architecture

```
┌──────────────────┐
│  Claude (API)     │  reads code, decides edits
└────────┬─────────┘
         │ HTTP
         ▼
┌──────────────────┐
│  engine_agent.zig │  executes tools, manages history
└────────┬─────────┘
         │ spawns / file I/O
    ┌────┴─────────────┐
    ▼                  ▼
┌──────────────┐  ┌───────────────┐
│ engine_      │  │ src/*.zig     │
│ research.zig │  │ compute.metal │
│ (toolbox)    │  │ (direct edit) │
└──────────────┘  └───────────────┘
```

### Output

- **stderr** — Claude's reasoning, tool calls, build/test progress.
- **`.engine_agent_history/`** — JSONL benchmark log + conversation logs.

### What it can optimise

- **Metal kernels** — tiled matmul, SIMD reductions, vectorised loads
- **Dispatch strategy** — threadgroup sizes, 1D vs 2D dispatch
- **Pipeline architecture** — command buffer batching, async commit
- **Buffer management** — storage modes, cache hints, buffer reuse
- **Kernel fusion** — combining matmul+bias+activation into fewer dispatches

## Status

Early development. Current capabilities:

- [x] Metal device initialisation + shader compilation
- [x] Shared buffer allocation (zero-copy on Apple Silicon)
- [x] Double-buffered activations
- [x] Comptime network layout (offsets, sizes, shapes)
- [x] GPU vector addition (pipeline proof)
- [x] GPU matmul kernel
- [x] Activation functions (relu, tanh, sigmoid) — forward + backward
- [x] Bias add + SGD update kernels
- [x] Forward pass wiring
- [x] Backward pass wiring
- [x] MSE loss function (forward + backward kernels)
- [x] Softmax + cross-entropy loss (fused backward kernel)
- [x] Training loop (epochs, SGD update, loss tracking)
- [x] MNIST data loader (IDX format, normalisation, one-hot encoding)
- [x] Train / validation / test split
- [x] MNIST training benchmark (97.8% test accuracy, ~85k img/s)
- [x] Benchmark system (JSON results, config tracking, per-epoch validation)
- [x] Autoresearch toolbox (Zig CLI for autonomous experiment sweeps)
- [x] Engine research toolbox (Zig CLI for autonomous engine optimisation)
- [x] Agent (LLM-powered experiment runner via Anthropic API)
- [x] Engine agent (LLM-powered engine optimisation via Anthropic API)
- [ ] Multi-core data loading (std.Thread.Pool)
- [ ] Learning rate scheduling / decay
- [x] Adam optimiser (GPU kernel + bias-corrected moments)
- [ ] Tiled matmul kernel

## Docs

- [Architecture](docs/ARCHITECTURE.md) — how it works
- [Strategy](docs/STRATEGY.md) — where it wins
- [Autoresearch](scripts/program.md) — autonomous hyperparameter agent
- [Agent](scripts/agent_program.md) — LLM-powered experiment runner
- [Engine research](scripts/engine_program.md) — autonomous engine optimisation agent
- [Engine agent](scripts/engine_agent.zig) — LLM-powered engine optimisation runner

## Dependencies

- [zig-objc](https://github.com/mitchellh/zig-objc) — Objective-C runtime bindings for Metal API calls
- Apple Metal framework (macOS / Apple Silicon)
