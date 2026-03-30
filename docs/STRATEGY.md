# Strategy: where nnzap wins

## The wrong question

"Can nnzap compete with PyTorch?"

No. PyTorch has thousands of contributors, every layer type imaginable,
an ecosystem of pre-trained models, and a decade of kernel optimisation.
MLX — Apple's own framework — already occupies the "ML framework
designed for Apple Silicon unified memory" niche with a team at Apple
behind it. Chasing general-purpose ML training is chasing a moving
target maintained by teams of dozens to hundreds.

## The right question

"What narrow problem domain can nnzap solve 100× faster than the
general-purpose tools, specifically because of the unified memory +
comptime + zero-allocation design?"

This is the [Moreau](https://www.moreau.so/) playbook.

## The Moreau pattern

Moreau is a differentiable convex solver from the creators of CVXPY.
They did not build a general-purpose ML framework. They built the
fastest differentiable convex solver, period, and they charge commercial
licenses for it.

Their architecture:

- All memory allocated upfront.
- One compile, many solves.
- Batched GPU dispatch.
- Zero per-solve overhead.
- Deterministic, same API on CPU and GPU.

Their results: 162× faster than Clarabel, 370× faster than Mosek on
batched robotics MPC. The biggest speedups are on batched small
problems — exactly where framework overhead dominates.

The parallels to nnzap are not accidental:

| Moreau principle              | nnzap equivalent                    |
| ----------------------------- | ----------------------------------- |
| All memory allocated upfront  | Rule 2: static allocation after init|
| One compile, many solves      | Comptime layout + pre-compiled Metal|
| Batched GPU dispatch          | Rule 8: batching as religion        |
| Zero per-solve overhead       | Unified memory, no graph tracing    |
| Deterministic                 | No dynamic dispatch, no Python GIL  |

## Where nnzap's architecture is the exactly correct tradeoff

### 1. Real-time inference on Apple devices

Small models (dense nets, small transformers) in latency-sensitive
contexts where PyTorch's startup time, memory footprint, and 2 GB
install are non-starters. Robotics, audio, embedded control on
Mac/iPad. A 200 KB static binary with zero-copy inference and
sub-millisecond cold start.

### 2. Differentiable simulation on Apple Silicon

Physics, control, or planning problems where the forward/backward pass
through a small network must be deterministic, allocation-free, and
fast. Same idea as Moreau but for differentiable dynamics. Apple
Silicon Macs are popular in robotics labs.

### 3. On-device training

Fine-tuning small models directly on-device without a cloud roundtrip.
Zero-copy means training on live sensor data with minimal latency.
CoreML does inference, not training. PyTorch on-device is a 2 GB
dependency with Python overhead.

### 4. Batched small-model evaluation

Running thousands of small networks — hyperparameter search, ensemble
methods, evolutionary strategies, population-based training — where
per-model framework overhead dominates. Each model variant is a
different comptime type with zero runtime graph construction. The GPU
sprints on the actual math instead of stalling on Python object
allocation and dynamic dispatch.

## Why the naive matmul doesn't matter (yet)

The naive one-thread-per-element matmul kernel is 10–50× slower than a
tiled implementation for large matrices (1024×1024+). But nnzap's
target domains involve small matrices where:

- Framework overhead dominates kernel time.
- A 100 μs matmul is lost in 10 ms of Python startup.
- Batch count matters more than single-kernel throughput.

Tiled matmul is worth building eventually. But the competitive moat is
not in the kernel — it is in the zero-overhead architecture around it.

## What winning looks like

1. Pick one domain from the list above.
2. Build a benchmark that makes people's jaws drop.
   (e.g., "1000 forward-backward passes in 4 ms on M4, cold start to
   first result in 200 μs, total binary size 180 KB.")
3. Publish the benchmark with reproducible numbers.
4. Offer it as an embeddable library or commercial engine.

The goal is not "better PyTorch." The goal is: for the right problem,
on the right hardware, there is nothing else even close.
