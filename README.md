# ⚡ nnzap

GPU-accelerated neural network library + autonomous experiment runner for Apple Silicon, written in Zig.

**Zero-copy. Comptime layouts. Metal compute.**

## Packages

| Package        | Description                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------- |
| [`nn/`](nn/)   | GPU-powered neural network library — Metal compute, comptime layouts, zero-copy unified memory |
| [`zap/`](zap/) | Autonomous experiment runner — LLM-powered research loops à la Karpathy's autoresearch         |

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

## Project structure

```
nnzap/
├── nn/                            # Neural network library
│   ├── build.zig                  # Library build (links Metal, Foundation, zig-objc)
│   ├── build.zig.zon              # Package manifest
│   ├── src/
│   │   ├── root.zig               # Library entry point, re-exports
│   │   ├── metal.zig              # Metal compute backend (Device, Buffer, pipelines)
│   │   ├── layout.zig             # Comptime network layout (sizes, offsets, shapes)
│   │   ├── network.zig            # Network struct with forward/backward pass
│   │   ├── mnist.zig              # MNIST IDX data loader
│   │   ├── benchmark.zig          # Benchmark result struct, JSON serialisation
│   │   └── shaders/
│   │       └── compute.metal      # All GPU compute kernels (MSL)
│   └── examples/
│       └── mnist.zig              # MNIST training loop, evaluation, benchmarking
├── zap/                           # Autonomous experiment runner
│   ├── build.zig                  # CLI tools build
│   ├── build.zig.zon              # Package manifest
│   ├── src/
│   │   ├── agent_core.zig         # Shared agent framework (loop, dispatch, API)
│   │   ├── mnist_agent.zig        # MNIST training agent (profile config)
│   │   ├── mnist_research.zig     # MNIST research CLI (config/train/bench)
│   │   ├── bonsai_agent.zig       # Bonsai inference agent (profile config + two-tier)
│   │   ├── bonsai_research.zig    # Bonsai research CLI (snapshot/rollback/bench/edit)
│   │   ├── api_client.zig         # Anthropic HTTP client
│   │   ├── ollama_client.zig      # Local LLM client (two-tier mode)
│   │   └── tools.zig              # Shared CLI/file utilities
│   ├── programs/
│   │   ├── program.md             # Shared conventions
│   │   ├── mnist_program.md       # MNIST agent skill file
│   │   ├── mnist_system.md        # MNIST system prompt
│   │   ├── mnist_tools.json       # MNIST tool schemas
│   │   ├── bonsai_program.md      # Bonsai agent skill file
│   │   ├── bonsai_system.md       # Bonsai system prompt
│   │   ├── bonsai_tools.json      # Bonsai tool schemas
│   │   ├── bonsai_strategist.md   # Two-tier strategist prompt
│   │   └── bonsai_executor.md     # Two-tier executor prompt
├── reference/
│   ├── mlx_reference.py           # MLX baseline for comparison
│   └── pytorch_reference.py       # PyTorch baseline for comparison
├── data/                          # MNIST dataset (downloaded at runtime)
├── docs/
│   ├── ARCHITECTURE.md            # Detailed architecture documentation
│   └── STRATEGY.md                # Strategic positioning (Moreau pattern)
├── CLAUDE.md                      # Engineering principles
└── README.md                      # This file
```

## Quick start

### 1. Download MNIST

Fetch and decompress the four dataset files into `data/mnist_torch/MNIST/raw/`:

```bash
mkdir -p data/mnist_torch/MNIST/raw && cd data/mnist_torch/MNIST/raw
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ../../../..
```

Or use torchvision (auto-downloads):

```bash
python3 -c "from torchvision.datasets import MNIST; MNIST('data/mnist_torch', download=True)"
```

### 2. Train

```bash
cd nn
zig build run
```

Output:

```
+-----------------------------------+
|       nn Network Layout           |
+-----------------------------------+
|  Layer 0:  784 -> 128   (relu    ) |
|  Layer 1:  128 -> 64    (relu    ) |
|  Layer 2:   64 -> 10    (none    ) |
+-----------------------------------+
|  Total params: 109386             |
|  Max activation: 128              |
+-----------------------------------+

Epoch  1/20 | loss 0.3508 | val loss 0.2461 | val acc 92.68% | 321 ms
Epoch  2/20 | loss 0.1912 | val loss 0.1771 | val acc 94.77% | 270 ms
...
Epoch 20/20 | loss 0.0274 | val loss 0.0718 | val acc 97.85% | 259 ms
```

### 3. Run tests

```bash
cd nn
zig build test
```

## Dataset split

The 60,000 MNIST training images are split:

| Set        | Samples | Usage                            |
| ---------- | ------- | -------------------------------- |
| Training   | 50,000  | Weight updates via backprop      |
| Validation | 10,000  | Per-epoch accuracy & overfitting |
| Test       | 10,000  | Final evaluation (separate file) |

Validation comes from the last 10k of the training file. The test set
is the standard MNIST `t10k-*` files, untouched during training.

## Benchmarks

Every training run writes a JSON benchmark file to `benchmarks/`.

### Output location

```
benchmarks/
  mnist_20250120_143022.json
  mnist_20250120_150415.json
  ...
```

### Using the benchmark API

```zig
const nn = @import("nn");
const Benchmark = nn.Benchmark;
const nanosToMs = nn.benchmark.nanosToMs;

var bench: Benchmark = undefined;
bench.init(MyLayout, .{
    .batch_size = 64,
    .learning_rate = 0.1,
    .learning_rate_decay = 0.0,
    .optimizer = .sgd,
    .loss_function = .cross_entropy,
    .num_epochs = 20,
    .seed = 42,
    .train_samples = 50_000,
    .validation_samples = 10_000,
    .test_samples = 10_000,
});

// In the training loop:
const epoch_start = std.time.nanoTimestamp();
const train_loss = trainEpoch(...);
const val = evaluate(...);
const epoch_ms = nanosToMs(std.time.nanoTimestamp() - epoch_start);

bench.recordEpoch(.{
    .epoch = epoch + 1,
    .train_loss = train_loss,
    .duration_ms = epoch_ms,
    .validation_loss = val.mean_loss,
    .validation_accuracy_pct = val.accuracy_pct,
});

// After training:
bench.setTrainingTime(total_ms);
bench.recordTest(correct, total, test_ms);
try bench.save("mnist");
```

## Autoresearch

The `zap/` package contains tools for autonomous ML experiment loops,
built on a shared agent framework (`agent_core.zig`) with pluggable
research profiles.

### How it works

```
┌────────────────┐     ┌──────────────┐     ┌──────────────┐
│ mnist_research │────>│  nn (build   │────>│  benchmarks/  │
│ CLI toolbox    │     │  + run)      │     │  JSON results │
└────────────────┘     └──────────────┘     └──────────────┘
        │                                         │
        └──────── compare / config-set ───────────┘
```

### MNIST research (hyperparameter optimisation)

```bash
# Build both packages
cd nn && zig build && cd ..
cd zap && zig build && cd ..

# Run from zap directory
cd zap
./zig-out/bin/mnist_research help
./zig-out/bin/mnist_research config-show
./zig-out/bin/mnist_research train
./zig-out/bin/mnist_research benchmark-compare
```

### MNIST agent

The agent wraps mnist_research with an LLM (Claude) that decides what
experiments to run:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd zap
zig build
./zig-out/bin/mnist_agent
```

### Bonsai research (inference optimisation)

Bonsai research targets the core library code — Metal kernels, dispatch
strategies, buffer layouts:

```bash
cd zap
./zig-out/bin/bonsai_research help
./zig-out/bin/bonsai_research snapshot
./zig-out/bin/bonsai_research bench
```

### Bonsai agent

Single-tier (Claude drives everything):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd zap
./zig-out/bin/bonsai_agent
```

Two-tier (Opus strategist + local LLM executor):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LOCAL_LLM_MODEL=default
cd zap
./zig-out/bin/bonsai_agent
```

## Status

### Done

- [x] Metal shared buffers (zero-copy unified memory)
- [x] Compute pipeline creation and dispatch (1D + 2D)
- [x] Double-buffered activations for CPU/GPU overlap
- [x] Comptime network layout with adjacency validation
- [x] Forward pass (matmul → bias_add → activation per layer)
- [x] Backward pass (full backpropagation)
- [x] SGD + Adam optimisers
- [x] MSE + softmax-cross-entropy loss
- [x] MNIST data loader with one-hot encoding
- [x] Benchmark recording + JSON serialisation
- [x] Fused matmul+bias+relu kernel
- [x] Tiled matmul kernel (16×16 tiles, shared memory)
- [x] Multi-batch command buffer encoding
- [x] Autoresearch toolbox
- [x] LLM agent (hyperparameter + engine)

### Next

- [ ] Tiled matmul for backward pass
- [ ] Conv2D layer
- [ ] Attention / transformer block
- [ ] Model serialisation (save/load weights)
- [ ] CoreML export

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — detailed technical architecture
- [`docs/STRATEGY.md`](docs/STRATEGY.md) — strategic positioning and target domains
- [`CLAUDE.md`](CLAUDE.md) — engineering principles and coding rules

## Dependencies

- **Zig ≥ 0.15.2**
- **macOS** with Metal support (Apple Silicon recommended)
- [`zig-objc`](https://github.com/mitchellh/zig-objc) — Zig bindings for Objective-C runtime (for Metal API)
