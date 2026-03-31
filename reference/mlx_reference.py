#!/usr/bin/env python3
"""MLX reference implementation for nnzap MNIST benchmark.

MLX is Apple's array framework for Apple Silicon — it runs on
the same Metal GPU with unified memory that nnzap targets.
This makes it the fairest high-level comparison.

Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (raw logits)
Loss:         softmax + cross-entropy
Optimiser:    mini-batch SGD (no momentum, no weight decay)

Hyperparameters match nn/examples/mnist.zig exactly:
  - seed:          42
  - batch_size:    64
  - learning_rate: 0.2
  - epochs:        20
  - val_interval:  5 (validate every 5th epoch + last)
  - train split:   50,000 (first 50k of shuffled 60k)
  - val split:     10,000 (last 10k of shuffled 60k)
  - test split:    10,000

Weight init matches nnzap's heInit():
  w ~ U(-limit, limit),  limit = sqrt(6 / fan_in)
  b = 0

MNIST data is loaded directly from the IDX binary files in
data/mnist/ — the same files nnzap reads. No torchvision
download required.

Usage:
  python scripts/mlx_reference.py

  # Compare against nnzap and/or PyTorch benchmarks:
  python scripts/mlx_reference.py --compare "benchmarks/mnist_*.json"

  # Save results:
  python scripts/mlx_reference.py --save

Requires: mlx
  pip install mlx
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils


# ── Architecture ──────────────────────────────────────────────

class MnistNet(nn.Module):
    """784 → 128 (ReLU) → 64 (ReLU) → 10 (raw logits).

    Matches nnzap's NetworkLayout exactly. The last layer
    has no activation — softmax is fused into the loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.maximum(self.fc1(x), 0.0)    # ReLU
        x = mx.maximum(self.fc2(x), 0.0)    # ReLU
        x = self.fc3(x)                      # Raw logits
        return x

    def param_count(self) -> int:
        return sum(
            v.size for _, v in mlx.utils.tree_flatten(
                self.parameters(),
            )
        )


# ── Initialisation ────────────────────────────────────────────

def he_init(model: MnistNet) -> None:
    """He uniform init matching nnzap's heInit().

    For each linear layer:
      weights ~ U(-limit, limit),  limit = sqrt(6 / fan_in)
      biases  = 0

    Note: MLX Linear stores weight as [out, in] and the
    forward pass computes x @ W.T + b. We initialise the
    full [out x in] weight matrix with He uniform.
    """
    for layer in [model.fc1, model.fc2, model.fc3]:
        fan_in = layer.weight.shape[1]
        limit = math.sqrt(6.0 / fan_in)
        layer.weight = mx.random.uniform(
            low=-limit,
            high=limit,
            shape=layer.weight.shape,
        )
        layer.bias = mx.zeros(layer.bias.shape)


# ── MNIST IDX loader ─────────────────────────────────────────
#
# Reads the same binary files nnzap uses (data/mnist/).
# No torchvision dependency required.

def _read_idx_images(path: str) -> mx.array:
    """Load IDX3 image file → float32 array [N, 784] in [0, 1]."""
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x0803, f"Bad image magic: {magic:#x}"
        assert rows == 28 and cols == 28
        raw = f.read(count * 784)
        assert len(raw) == count * 784

    # Convert bytes → float32 normalised to [0, 1].
    pixels = mx.array(list(raw), dtype=mx.uint8)
    pixels = pixels.astype(mx.float32) / 255.0
    return pixels.reshape(count, 784)


def _read_idx_labels(path: str) -> mx.array:
    """Load IDX1 label file → uint32 array [N]."""
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 0x0801, f"Bad label magic: {magic:#x}"
        raw = f.read(count)
        assert len(raw) == count

    return mx.array(list(raw), dtype=mx.uint32)


def load_mnist(base_dir: str) -> dict:
    """Load all four MNIST IDX files from base_dir.

    Returns dict with train_images, train_labels,
    test_images, test_labels as MLX arrays.
    """
    d = Path(base_dir)

    train_images = _read_idx_images(
        str(d / "train-images-idx3-ubyte"),
    )
    train_labels = _read_idx_labels(
        str(d / "train-labels-idx1-ubyte"),
    )
    test_images = _read_idx_images(
        str(d / "t10k-images-idx3-ubyte"),
    )
    test_labels = _read_idx_labels(
        str(d / "t10k-labels-idx1-ubyte"),
    )

    assert train_images.shape == (60000, 784)
    assert train_labels.shape == (60000,)
    assert test_images.shape == (10000, 784)
    assert test_labels.shape == (10000,)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


# ── Loss ──────────────────────────────────────────────────────

def cross_entropy_loss(
    model: MnistNet,
    images: mx.array,
    labels: mx.array,
) -> mx.array:
    """Mean cross-entropy loss over a batch.

    Uses the numerically-stable logsumexp formulation:
      CE = -logits[label] + log(sum(exp(logits)))
    which is equivalent to -log(softmax[label]).
    """
    logits = model(images)
    return mx.mean(nn.losses.cross_entropy(logits, labels))


# ── Evaluation ────────────────────────────────────────────────

def evaluate(
    model: MnistNet,
    images: mx.array,
    labels: mx.array,
    batch_size: int,
) -> dict:
    """Evaluate accuracy and mean CE loss.

    Iterates in batches to avoid a single giant allocation.
    Matches nnzap's evaluate() semantics.
    """
    count = images.shape[0]
    assert count > 0

    total_loss = 0.0
    correct = 0
    total = 0
    offset = 0

    while offset < count:
        end = min(offset + batch_size, count)
        batch_images = images[offset:end]
        batch_labels = labels[offset:end]

        logits = model(batch_images)

        # Per-sample CE loss, summed (we average manually).
        losses = nn.losses.cross_entropy(logits, batch_labels)
        batch_loss = mx.sum(losses)

        preds = mx.argmax(logits, axis=1)
        batch_correct = mx.sum(preds == batch_labels)

        # Force evaluation so we can accumulate on CPU.
        mx.eval(batch_loss, batch_correct)

        total_loss += batch_loss.item()
        correct += batch_correct.item()
        total += end - offset
        offset = end

    mean_loss = total_loss / total
    accuracy_pct = correct / total * 100.0

    return {
        "correct": int(correct),
        "total": total,
        "mean_loss": mean_loss,
        "accuracy_pct": accuracy_pct,
    }


# ── Training ──────────────────────────────────────────────────

def train_epoch(
    model: MnistNet,
    images: mx.array,
    labels: mx.array,
    indices: mx.array,
    optimizer: optim.SGD,
    batch_size: int,
    loss_and_grad_fn,
) -> float:
    """Train for one epoch, return last-batch mean loss.

    nnzap reports train_loss as the loss of the final batch
    (not the epoch average). We match that here.
    """
    count = indices.shape[0]
    assert count > 0

    last_loss = 0.0
    offset = 0

    while offset + batch_size <= count:
        batch_idx = indices[offset:offset + batch_size]
        batch_images = images[batch_idx]
        batch_labels = labels[batch_idx]

        loss, grads = loss_and_grad_fn(
            model, batch_images, batch_labels,
        )
        optimizer.update(model, grads)

        # Force evaluation to commit GPU work and avoid
        # an unbounded graph of lazy operations.
        mx.eval(model.parameters(), optimizer.state, loss)

        last_loss = loss.item()
        offset += batch_size

    return last_loss


# ── Comparison ────────────────────────────────────────────────

def find_benchmarks() -> dict[str, str]:
    """Find the most recent nnzap and PyTorch benchmarks."""
    found = {}

    nnzap_files = sorted(glob.glob("benchmarks/mnist_*.json"))
    if nnzap_files:
        found["nnzap"] = nnzap_files[-1]

    pytorch_files = sorted(glob.glob("benchmarks/pytorch_mnist_*.json"))
    if pytorch_files:
        found["pytorch"] = pytorch_files[-1]

    return found


def compare_all(
    mlx_results: dict,
    benchmark_paths: dict[str, str],
) -> None:
    """Print comparison table across all available benchmarks."""
    benchmarks = {"MLX": mlx_results}
    for name, path in benchmark_paths.items():
        with open(path) as f:
            benchmarks[name] = json.load(f)

    names = list(benchmarks.keys())
    col_w = 12

    sep = "-" * (30 + col_w * len(names) + 2)
    header = f"{'Metric':<30}" + "".join(
        f"{n:>{col_w}}" for n in names
    )

    print(f"\n{'Cross-framework comparison':^{len(sep)}}")
    print(sep)
    print(header)
    print(sep)

    rows = [
        (
            "Final val accuracy (%)",
            "final_validation_accuracy_pct",
            ".2f",
        ),
        (
            "Final val loss",
            "final_validation_loss",
            ".4f",
        ),
        (
            "Test accuracy (%)",
            "final_test_accuracy_pct",
            ".2f",
        ),
        (
            "Training time (ms)",
            "total_training_ms",
            ".0f",
        ),
        (
            "Throughput (img/s)",
            "throughput_images_per_sec",
            ".0f",
        ),
    ]

    for label, key, fmt in rows:
        cells = ""
        for name in names:
            val = benchmarks[name].get(key, 0)
            cells += f"{val:>{col_w}{fmt}}"
        print(f"{label:<30}{cells}")

    print(sep)

    # Per-epoch table.
    print(f"\n{'Per-epoch validation accuracy':^{len(sep)}}")
    print(sep)

    ep_header = f"{'Epoch':<8}"
    for name in names:
        ep_header += f"{name + ' acc':>{col_w}}"
        ep_header += f"{name + ' ms':>{col_w}}"
    print(ep_header)
    print(sep)

    max_epochs = max(
        len(b.get("epochs", [])) for b in benchmarks.values()
    )
    for i in range(max_epochs):
        row = f"{i + 1:<8}"
        for name in names:
            epochs = benchmarks[name].get("epochs", [])
            if i < len(epochs):
                ep = epochs[i]
                row += f"{ep['validation_accuracy_pct']:>{col_w}.2f}"
                row += f"{ep['duration_ms']:>{col_w}.0f}"
            else:
                row += " " * col_w * 2
        print(row)

    print(sep)


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX reference for nnzap MNIST benchmark",
    )
    parser.add_argument(
        "--data-dir",
        default="data/mnist",
        help=(
            "Directory containing MNIST IDX files "
            "(default: data/mnist — same as nnzap)"
        ),
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Glob pattern for benchmark JSONs to compare",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to benchmarks/ as JSON",
    )
    args = parser.parse_args()

    # ── Hyperparameters (match nnzap exactly) ──

    seed = 42
    num_epochs = 20
    learning_rate = 0.2
    batch_size = 64
    total_train = 60_000
    val_count = 10_000
    train_count = total_train - val_count  # 50,000
    test_count = 10_000
    val_interval = 5

    print()
    print("MLX MNIST reference benchmark")
    print("=============================")
    print()
    print(f"MLX backend: Metal (Apple Silicon unified memory)")
    print()

    # ── Architecture summary ──

    print("+-----------------------------------+")
    print("|        MLX Network Layout         |")
    print("+-----------------------------------+")
    print("|  Layer 0:  784 -> 128   (relu   ) |")
    print("|  Layer 1:  128 -> 64    (relu   ) |")
    print("|  Layer 2:   64 -> 10    (none   ) |")
    print("+-----------------------------------+")

    # ── Seed ──

    mx.random.seed(seed)

    # ── Model ──

    model = MnistNet()
    he_init(model)
    mx.eval(model.parameters())

    param_count = model.param_count()
    print(f"|  Total params: {param_count:<16} |")
    print("+-----------------------------------+")
    print()

    assert param_count == 109386, (
        f"Expected 109,386 params, got {param_count}"
    )

    # ── Dataset ──

    print("Loading MNIST... ", end="", flush=True)
    data = load_mnist(args.data_dir)
    print("done.")

    # Shuffle all 60k indices with a fixed seed, then
    # freeze the partition (same strategy as nnzap).
    mx.random.seed(seed)
    all_indices = mx.random.permutation(total_train)
    mx.eval(all_indices)

    train_indices = all_indices[:train_count]
    val_indices = all_indices[train_count:]

    assert train_indices.shape[0] == train_count
    assert val_indices.shape[0] == val_count

    # Pre-slice validation and test sets for evaluation.
    val_images = data["train_images"][val_indices]
    val_labels = data["train_labels"][val_indices]
    test_images = data["test_images"]
    test_labels = data["test_labels"]
    mx.eval(val_images, val_labels)

    print(
        f"Split: train={train_count}  "
        f"val={val_count}  "
        f"test={test_count}"
    )
    print()

    # ── Optimizer ──
    # Plain SGD: no momentum, no weight decay — matches nnzap.

    optimizer = optim.SGD(learning_rate=learning_rate)

    # ── Compiled loss + grad function ──
    # mx.compile JIT-compiles the loss+grad computation into
    # a single fused Metal kernel graph — this is where MLX
    # gets its speed advantage over PyTorch MPS.

    loss_and_grad_fn = nn.value_and_grad(
        model, cross_entropy_loss,
    )

    # ── Training ──

    batches_per_epoch = train_count // batch_size

    print(
        f"Training: {num_epochs} epochs x "
        f"{batches_per_epoch} batches  "
        f"(batch={batch_size}, lr={learning_rate:.2f})"
    )

    sep = "-" * 54
    print(sep)

    epoch_results = []
    train_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()

        # Reshuffle training indices each epoch (nnzap does
        # this too — random.shuffle in trainEpoch).
        shuffled = mx.random.permutation(train_count)
        epoch_indices = train_indices[shuffled]
        mx.eval(epoch_indices)

        train_loss = train_epoch(
            model,
            data["train_images"],
            data["train_labels"],
            epoch_indices,
            optimizer,
            batch_size,
            loss_and_grad_fn,
        )

        # Validate every val_interval epochs and on the
        # last epoch — matches nnzap's val_interval=5.
        is_val_epoch = (
            epoch % val_interval == 0 or epoch == num_epochs
        )

        val_loss = 0.0
        val_acc = 0.0
        if is_val_epoch:
            val = evaluate(
                model, val_images, val_labels, batch_size,
            )
            val_loss = val["mean_loss"]
            val_acc = val["accuracy_pct"]

        epoch_ms = (time.perf_counter() - epoch_start) * 1000.0

        epoch_results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "duration_ms": epoch_ms,
            "validation_loss": val_loss,
            "validation_accuracy_pct": val_acc,
        })

        if is_val_epoch:
            print(
                f"  Epoch {epoch:>2}: "
                f"loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.2f}%  "
                f"({epoch_ms:.0f} ms)"
            )
        else:
            print(
                f"  Epoch {epoch:>2}: "
                f"loss={train_loss:.4f}  "
                f"({epoch_ms:.0f} ms)"
            )

    total_training_ms = (
        (time.perf_counter() - train_start) * 1000.0
    )

    print(sep)
    ms_per_epoch = total_training_ms / num_epochs
    throughput = (
        train_count * num_epochs
        / (total_training_ms / 1000.0)
    )
    print(
        f"Total: {total_training_ms:.0f} ms  "
        f"({ms_per_epoch:.0f} ms/epoch, "
        f"{throughput:.0f} img/s)"
    )
    print()

    # ── Test evaluation ──

    print("Evaluating test set... ", end="", flush=True)
    eval_start = time.perf_counter()
    test_eval = evaluate(
        model, test_images, test_labels, batch_size,
    )
    eval_ms = (time.perf_counter() - eval_start) * 1000.0

    print(
        f"{test_eval['correct']}/{test_eval['total']} "
        f"correct ({test_eval['accuracy_pct']:.2f}%)  "
        f"[{eval_ms:.0f} ms]"
    )
    print()

    # ── Assemble results ──

    results = {
        "timestamp_utc": (
            datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "framework": "mlx",
        "mlx_version": mx.__version__,
        "device": "apple_silicon_gpu",
        "config": {
            "architecture": [
                {
                    "input_size": 784,
                    "output_size": 128,
                    "activation": "relu",
                },
                {
                    "input_size": 128,
                    "output_size": 64,
                    "activation": "relu",
                },
                {
                    "input_size": 64,
                    "output_size": 10,
                    "activation": "none",
                },
            ],
            "param_count": param_count,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_decay": 0,
            "optimizer": "sgd",
            "loss_function": "cross_entropy",
            "num_epochs": num_epochs,
            "seed": seed,
            "train_samples": train_count,
            "validation_samples": val_count,
            "test_samples": test_count,
        },
        "final_train_loss": epoch_results[-1]["train_loss"],
        "final_validation_loss": (
            epoch_results[-1]["validation_loss"]
        ),
        "final_validation_accuracy_pct": (
            epoch_results[-1]["validation_accuracy_pct"]
        ),
        "final_test_accuracy_pct": (
            test_eval["accuracy_pct"]
        ),
        "epochs": epoch_results,
        "test_result": {
            "correct": test_eval["correct"],
            "total": test_eval["total"],
            "accuracy_pct": test_eval["accuracy_pct"],
            "duration_ms": eval_ms,
        },
        "total_training_ms": total_training_ms,
        "throughput_images_per_sec": throughput,
    }

    # ── Save results ──

    if args.save:
        benchmarks_dir = Path("benchmarks")
        benchmarks_dir.mkdir(exist_ok=True)

        timestamp = (
            datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H-%M-%SZ")
        )
        out_path = (
            benchmarks_dir / f"mlx_mnist_{timestamp}.json"
        )

        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {out_path}")
        print()

    # ── Compare ──

    if args.compare is not None:
        paths = glob.glob(args.compare)
        if not paths:
            print(
                f"Warning: no files matched '{args.compare}'",
                file=sys.stderr,
            )
        benchmark_paths = {}
        for p in sorted(paths):
            if "pytorch" in p:
                benchmark_paths["pytorch"] = p
            elif "mlx" not in p:
                benchmark_paths["nnzap"] = p
        if benchmark_paths:
            compare_all(results, benchmark_paths)
    else:
        # Auto-detect benchmarks for comparison.
        benchmark_paths = find_benchmarks()
        if benchmark_paths:
            compare_all(results, benchmark_paths)


if __name__ == "__main__":
    main()
