# ⚡ nnzap

GPU-accelerated neural network library + autonomous experiment runner for Apple Silicon, written in Zig.

**Zero-copy. Comptime layouts. Metal compute.**

## Packages

| Package        | Description                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------- |
| [`nn/`](nn/)   | GPU-powered neural network library — Metal compute, comptime layouts, zero-copy unified memory |
| [`zap/`](zap/) | Autonomous experiment runner — LLM-powered research loops with a generic toolbox framework     |

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
├── nn/                              # Neural network library
│   ├── build.zig                    # Library build (links Metal, Foundation, zig-objc)
│   ├── build.zig.zon                # Package manifest
│   ├── src/
│   │   ├── root.zig                 # Library entry point, re-exports
│   │   ├── metal.zig                # Metal compute backend (Device, Buffer, pipelines)
│   │   ├── layout.zig               # Comptime network layout (sizes, offsets, shapes)
│   │   ├── network.zig              # Network struct with forward/backward pass
│   │   ├── transformer.zig          # Transformer implementation (Bonsai 1.7B)
│   │   ├── model.zig                # Model loading (safetensors)
│   │   ├── safetensors.zig          # Safetensors format parser
│   │   ├── tokenizer.zig            # Tokenizer
│   │   ├── mnist.zig                # MNIST IDX data loader
│   │   ├── benchmark.zig            # Benchmark recording + JSON serialisation
│   │   └── shaders/
│   │       ├── compute.metal        # General NN GPU kernels (MSL)
│   │       └── transformer.metal    # Attention-specific GPU kernels
│   └── examples/
│       ├── mnist.zig                # MNIST training loop + evaluation
│       ├── mnist_1bit.zig           # 1-bit MNIST variant
│       ├── bonsai.zig               # Bonsai tree classifier
│       ├── bonsai_bench.zig         # Bonsai benchmarking
│       └── inference_bench.zig      # Inference benchmarking
├── zap/                             # Autonomous experiment runner
│   ├── build.zig                    # CLI tools build
│   ├── build.zig.zon                # Package manifest
│   ├── src/
│   │   ├── agent_core.zig           # Generic agent framework (loop, dispatch, API)
│   │   ├── toolbox.zig              # Generic toolbox (23 tools, ToolboxConfig)
│   │   ├── tools.zig                # Shared CLI/file utilities
│   │   ├── api_client.zig           # Anthropic HTTP client
│   │   ├── bonsai_agent.zig         # Bonsai agent profile (~100 lines)
│   │   ├── bonsai_research.zig      # Bonsai toolbox config (~90 lines)
│   │   ├── mnist_agent.zig          # MNIST agent profile (~130 lines)
│   │   └── mnist_research.zig       # MNIST toolbox config + custom tools
│   └── programs/
│       ├── program.md               # Shared conventions
│       ├── bonsai_program.md        # Bonsai skill file
│       ├── bonsai_system.md         # Bonsai system prompt
│       ├── mnist_program.md         # MNIST skill file
│       └── mnist_system.md          # MNIST system prompt
├── reference/                       # Baseline implementations for comparison
│   ├── mlx_reference.py             # MLX MNIST baseline
│   ├── mlx_inference.py             # MLX inference baseline
│   ├── mlx_bonsai.py                # MLX Bonsai baseline
│   ├── pytorch_reference.py         # PyTorch MNIST baseline
│   └── pytorch_inference.py         # PyTorch inference baseline
├── docs/
│   ├── ARCHITECTURE.md              # Detailed technical architecture
│   ├── STRATEGY.md                  # Strategic positioning (Moreau pattern)
│   ├── BONSAI.md                    # Bonsai model documentation
│   └── PERF_PLAN.md                 # Performance roadmap
├── data/                            # Datasets (downloaded at runtime)
├── CLAUDE.md                        # Engineering principles
└── README.md                        # This file
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

The `zap/` package runs autonomous ML experiment loops. An LLM
(Claude) drives the research — reading code, editing files, running
benchmarks, and iterating — while a generic toolbox framework handles
the mechanics.

### Framework

Zap is built around two generic engines and thin domain configs:

```
                    ┌───────────────────┐
                    │   agent_core.zig  │  Generic agent loop
                    │   (API, dispatch, │  (turn management,
                    │    history)       │   context building)
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ...
     ┌────────────┐  ┌──────────────┐
     │ bonsai     │  │ mnist        │  Agent profiles
     │ _agent.zig │  │ _agent.zig   │  (~100 lines each)
     └────────────┘  └──────────────┘


                    ┌───────────────────┐
                    │   toolbox.zig     │  Generic toolbox
                    │   (23 tools:      │  (snapshot, bench,
                    │    file I/O,      │   edit, diff,
                    │    build, ...)    │   commit, ...)
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ...
     ┌────────────┐  ┌──────────────┐
     │ bonsai     │  │ mnist        │  Toolbox configs
     │ _research  │  │ _research    │  (~90 lines each)
     └────────────┘  └──────────────┘
```

**Agent profiles** configure the LLM loop: system prompt, tool
schemas (defined as comptime `ToolDef` structs), history fields,
and turn limits.

**Toolbox configs** configure the CLI: write scope, read scope,
build/test/bench commands, snapshot directory, and an optional
`custom_dispatch` callback for domain-specific tools.

Adding a new domain means writing ~200 lines of config (one agent
profile, one toolbox config, one system prompt) — no copy-pasting
thousands of lines of tool logic.

### MNIST research (hyperparameter optimisation)

```bash
cd zap && zig build

./zig-out/bin/mnist_research help
./zig-out/bin/mnist_research config-show
./zig-out/bin/mnist_research bench
./zig-out/bin/mnist_research bench-compare
```

### MNIST agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd zap && zig build
./zig-out/bin/mnist_agent
```

### Bonsai research (inference optimisation)

```bash
cd zap && zig build

./zig-out/bin/bonsai_research help
./zig-out/bin/bonsai_research snapshot
./zig-out/bin/bonsai_research bench
```

### Bonsai agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd zap && zig build
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
- [x] Tiled matmul kernel (16×16 tiles, shared memory, forward + backward)
- [x] Multi-batch command buffer encoding
- [x] Transformer implementation (Bonsai 1.7B inference)
- [x] Safetensors model loading
- [x] Tokenizer
- [x] Generic autoresearch toolbox framework
- [x] Comptime tool definitions (schemas + dispatch)
- [x] LLM agents (hyperparameter + engine optimisation)

### Next

- [ ] Conv2D layer
- [ ] Batched prefill (quantised matrix-matrix multiply for multi-token prefill)
- [ ] Flash attention (tiled attention for long contexts)
- [ ] Transformer training / backward pass
- [ ] Save trained weights to safetensors
- [ ] Training checkpoints (periodic weight snapshots mid-training)
- [ ] CoreML export

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — detailed technical architecture
- [`docs/STRATEGY.md`](docs/STRATEGY.md) — strategic positioning and target domains
- [`docs/BONSAI.md`](docs/BONSAI.md) — Bonsai model documentation
- [`docs/PERF_PLAN.md`](docs/PERF_PLAN.md) — performance roadmap
- [`zap/docs/framework_design.md`](zap/docs/framework_design.md) — toolbox framework design
- [`CLAUDE.md`](CLAUDE.md) — engineering principles and coding rules

## Dependencies

- **Zig ≥ 0.15.2**
- **macOS** with Metal support (Apple Silicon recommended)
- [`zig-objc`](https://github.com/mitchellh/zig-objc) — Zig bindings for Objective-C runtime (for Metal API)
