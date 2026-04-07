# ⚡ nnmetal + labrat

Give an AI agent a real 1.7B inference engine and let it optimise
Metal kernels autonomously overnight.

nnmetal is a GPU-accelerated neural network library + autonomous
experiment runner for Apple Silicon, written in Zig. The neural
network library (`nnmetal/`) exploits unified memory for zero-copy
GPU compute. The experiment runner (`labrat/`) wraps it in an LLM
agent loop that reads code, edits shaders, benchmarks, and
iterates — no human in the loop.

## How it works

Two binaries, one loop:

- **`bonsai_agent`** — the outer loop. Talks to Claude, decides
  what to try next, dispatches tool calls.
- **`bonsai_researcher`** — the toolbox. Executes tool calls in a
  sandboxed environment: snapshot, edit, compile, test, benchmark,
  rollback, commit.

The agent calls the researcher as a subprocess for every tool
invocation. The researcher enforces write scope (only engine files),
read scope (only project files), and timeout limits. The agent
never touches the filesystem directly.

```
  ┌─────────────────────────────────────────────┐
  │  bonsai_agent (outer loop)                  │
  │                                             │
  │  1. Build context (rules, history,          │
  │     summaries of prior experiments)         │
  │  2. Send to Claude                          │
  │  3. Claude picks a tool                     │
  │  4. Spawn: bonsai_researcher <tool> [args]  │
  │  5. Feed output back to Claude              │
  │  6. Repeat until experiment done            │
  │  7. Start next experiment                   │
  └─────────────────────────────────────────────┘
```

Each experiment follows a strict protocol:

1. **Snapshot** — save all engine source files as a restore point.
2. **Read** — study the code and prior experiment summaries.
3. **Edit** — make ONE targeted change (isolate variables).
4. **Check** — compile-only validation (~2s). Stop if it fails.
5. **Test** — run full test suite. Stop if it fails.
6. **Bench** — measure decode tok/s, prefill tok/s, p99 latency.
7. **Keep or rollback** — ≥5% improvement → commit. Otherwise
   rollback and record why it failed.
8. **Summarise** — write a summary so future experiments don't
   repeat the same mistake.

## Quick start: Bonsai autoresearch

### 1. Build

```bash
cd nnmetal && zig build
cd ../labrat && zig build
```

### 2. Run baseline benchmark

```bash
./zig-out/bin/bonsai_researcher bench
```

```
{
  "model": "Bonsai-1.7B",
  "decode_tok_per_sec": 182.8,
  "prefill_tok_per_sec": 106.1,
  "decode_p99_us": 6163
}
```

### 3. Start the agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./zig-out/bin/bonsai_agent
```

The agent runs autonomously. Each experiment takes 2–5 minutes.
You can expect ~12–30 experiments per hour depending on the
complexity of the changes. Let it run overnight and check
results in the morning.

### 4. Check results

```bash
./zig-out/bin/bonsai_researcher bench-compare
```

This prints every benchmark run side by side so you can see
the progression of decode tok/s, prefill tok/s, and p99
latency across experiments.

### 5. Review what was tried

The agent writes a summary after every experiment. These
summaries accumulate in `.bonsai_history/` and are injected
into every future experiment so the agent learns from its
own history.

```bash
cat .bonsai_history/summaries.txt
```

```
Experiment 1: Fused gate/up QMV + SiLU + elementwise multiply
into single specialised kernel. Saves 1 dispatch + 2 barriers
per block × 28 blocks = 28 dispatches + 56 barriers per decode
token. Result: 182.8 → ~190 tok/s (~3-7% improvement).

Experiment 2: Tried concurrent dispatch type. Results noisy,
not clearly better. Rolled back. The QMV kernels dominate each
barrier-to-barrier segment so there's little to overlap.
```

## Quick start: MNIST training

nnmetal also trains small networks from scratch. This is the
pedagogical path — useful for understanding the library before
diving into transformer inference.

### 1. Download MNIST

```bash
mkdir -p data/mnist_torch/MNIST/raw && cd data/mnist_torch/MNIST/raw
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ../../../..
```

Or use torchvision:

```bash
python3 -c "from torchvision.datasets import MNIST; MNIST('data/mnist_torch', download=True)"
```

### 2. Train

```bash
cd nnmetal && zig build run
```

```
+-----------------------------------+
|       nnmetal Network Layout      |
+-----------------------------------+
|  Layer 0:  784 -> 128   (relu    ) |
|  Layer 1:  128 -> 64    (relu    ) |
|  Layer 2:   64 -> 10    (none    ) |
+-----------------------------------+
|  Total params: 109386             |
|  Max activation: 128              |
+-----------------------------------+

Epoch  1/20 | loss 0.3508 | val acc 92.68% | 321 ms
Epoch  2/20 | loss 0.1912 | val acc 94.77% | 270 ms
...
Epoch 20/20 | loss 0.0274 | val acc 97.85% | 259 ms
```

### 3. Run tests

```bash
cd nnmetal && zig build test
```

### 4. MNIST autoresearch (optional)

The MNIST agent optimises hyperparameters (learning rate,
architecture, optimizer, batch size) rather than kernel code:

```bash
cd labrat && zig build
export ANTHROPIC_API_KEY=sk-ant-...
./zig-out/bin/mnist_agent
```

## Why unified memory matters

On Apple Silicon, CPU and GPU share the same physical memory.
Most ML frameworks still copy buffers around as if they're
talking to a discrete GPU over PCIe. nnmetal exploits unified
memory directly — the `[]f32` slice your Zig code writes to
_is_ the GPU buffer.

```
  CPU writes params  ──>  same physical memory  <──  GPU reads params
       zero copy              unified DRAM              zero copy
```

No `memcpy`. No staging buffers. No PCIe transfer. The Metal
shared buffer IS the Zig slice.

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

## Autoresearch framework

Labrat is built around two generic engines and thin domain configs:

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
    │  bonsai    │  │  mnist       │  Agent profiles
    │  _agent    │  │  _agent      │  (~100 lines each)
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
    │  bonsai    │  │  mnist       │  Researcher configs
    │_researcher │  │_researcher   │  (~90 lines each)
    └────────────┘  └──────────────┘
```

**Agent profiles** configure the LLM loop: system prompt, tool
schemas (defined as comptime `ToolDef` structs), history fields,
and turn limits.

**Researcher configs** configure the CLI: write scope, read scope,
build/test/bench commands, snapshot directory, and an optional
`custom_dispatch` callback for domain-specific tools.

Adding a new domain means writing ~200 lines of config (one agent
profile, one researcher config, one system prompt) — no
copy-pasting thousands of lines of tool logic.

## Project structure

```
nnmetal + labrat/
├── nnmetal/                         # Neural network library
│   ├── src/
│   │   ├── metal.zig                # Metal compute backend
│   │   ├── layout.zig               # Comptime network layout
│   │   ├── network.zig              # Forward/backward pass
│   │   ├── transformer.zig          # Transformer (Bonsai 1.7B)
│   │   ├── model.zig                # Safetensors model loading
│   │   ├── safetensors.zig          # Safetensors format parser
│   │   ├── tokenizer.zig            # Tokenizer
│   │   ├── mnist.zig                # MNIST data loader
│   │   ├── benchmark.zig            # Benchmark recording + JSON
│   │   └── shaders/
│   │       ├── compute.metal        # General NN kernels
│   │       ├── transformer.metal    # Attention kernels
│   │       └── qmv_specialized.metal # Quantised matmul kernels
│   └── examples/
│       ├── mnist.zig                # MNIST training
│       ├── bonsai.zig               # Bonsai inference
│       └── bonsai_bench.zig         # Bonsai benchmarking
├── labrat/                          # Autonomous experiment runner
│   ├── src/
│   │   ├── agent_core.zig           # Generic agent framework
│   │   ├── toolbox.zig              # Generic toolbox (23 tools)
│   │   ├── tools.zig                # Shared CLI utilities
│   │   ├── api_client.zig           # Anthropic HTTP client
│   │   ├── bonsai_agent.zig         # Bonsai agent profile
│   │   ├── bonsai_researcher.zig    # Bonsai researcher config
│   │   ├── mnist_agent.zig          # MNIST agent profile
│   │   └── mnist_researcher.zig     # MNIST researcher config
│   └── programs/
│       ├── bonsai_system.md         # Bonsai system prompt
│       └── mnist_system.md          # MNIST system prompt
├── reference/                       # Baseline implementations
│   ├── mlx_bonsai.py                # MLX Bonsai baseline
│   ├── mlx_reference.py             # MLX MNIST baseline
│   └── pytorch_reference.py         # PyTorch MNIST baseline
├── CLAUDE.md                        # Engineering principles
└── README.md
```

## Key files

| File                | Lines | What it does                                        |
| ------------------- | ----: | --------------------------------------------------- |
| `transformer.zig`   | 5,982 | Transformer dispatch, decode loop, all QMV variants |
| `network.zig`       | 3,308 | Core NN forward/backward/train                      |
| `compute.metal`     | 4,675 | GPU kernels: matmul, activations, loss, QMV         |
| `transformer.metal` | 1,343 | Attention kernels: RMSNorm, RoPE, GQA, KV cache     |
| `agent_core.zig`    | 2,271 | Shared agent framework (loop, API, context)         |
| `toolbox.zig`       | 2,399 | Generic toolbox (23 tools)                          |

## Status

### Done

- [x] Metal shared buffers (zero-copy unified memory)
- [x] Comptime network layout with adjacency validation
- [x] Forward + backward pass, SGD + Adam optimisers
- [x] Tiled matmul kernels (16×16, shared memory)
- [x] Double-buffered activations for CPU/GPU overlap
- [x] Transformer implementation (Bonsai 1.7B inference)
- [x] 1-bit quantised matrix-vector multiply (Q1_0_g128)
- [x] Safetensors model loading + tokenizer
- [x] Generic autoresearch framework (agent + toolbox)
- [x] LLM agents (kernel optimisation + hyperparameter search)

### Next

- [ ] Batched prefill (quantised matrix-matrix multiply)
- [ ] Flash attention (tiled attention for long contexts)
- [ ] Transformer training / backward pass
- [ ] Conv2D layer
- [ ] Save/load trained weights (safetensors)
- [ ] CoreML export

## Dependencies

- **Zig ≥ 0.15.2**
- **macOS** with Metal support (Apple Silicon recommended)
- [`zig-objc`](https://github.com/mitchellh/zig-objc) — Zig
  bindings for Objective-C runtime (for Metal API)
