#!/usr/bin/env python3
"""PyTorch reference implementation for nnzap MNIST benchmark.

Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (none / raw logits)
Loss:         softmax + cross-entropy (PyTorch fuses these)
Optimiser:    mini-batch SGD (no momentum, no weight decay)

Hyperparameters and dataset split match nn/examples/mnist.zig exactly:
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

Usage:
  python scripts/pytorch_reference.py

  # With a specific data directory:
  python scripts/pytorch_reference.py --data-dir ./data/mnist_torch

  # Compare against a specific nnzap benchmark:
  python scripts/pytorch_reference.py --compare benchmarks/mnist_*.json

Requires: torch, torchvision
  pip install torch torchvision
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ── Architecture ──────────────────────────────────────────────

class MnistNet(nn.Module):
    """784 → 128 (ReLU) → 64 (ReLU) → 10 (raw logits).

    Matches nnzap's NetworkLayout exactly. The last layer
    has no activation — softmax is fused into the loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   # Layer 0: relu
        self.fc2 = nn.Linear(128, 64)    # Layer 1: relu
        self.fc3 = nn.Linear(64, 10)     # Layer 2: none (raw logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Initialisation ────────────────────────────────────────────

def he_init(model: MnistNet) -> None:
    """He uniform init matching nnzap's heInit().

    For each linear layer:
      weights ~ U(-limit, limit),  limit = sqrt(6 / fan_in)
      biases  = 0

    This is equivalent to Kaiming uniform with gain=sqrt(3),
    which differs from PyTorch's default (gain adjusted for
    leaky_relu slope).  We set it manually to match nnzap.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fan_in = module.in_features
            limit = math.sqrt(6.0 / fan_in)
            nn.init.uniform_(module.weight, -limit, limit)
            nn.init.zeros_(module.bias)


# ── Dataset ───────────────────────────────────────────────────

def load_datasets(
    data_dir: str,
    seed: int,
    train_count: int,
    val_count: int,
) -> tuple[Subset, Subset, datasets.MNIST]:
    """Load MNIST and split into train/val/test.

    Mirrors nnzap's split strategy:
      1. Shuffle all 60k training indices with a fixed seed.
      2. First 50k → training set (reshuffled each epoch).
      3. Last 10k  → validation set (fixed).
      4. Standard 10k test set (untouched).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),          # u8 [0,255] → f32 [0,1]
        transforms.Lambda(
            lambda t: t.view(-1),       # [1, 28, 28] → [784]
        ),
    ])

    full_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Deterministic shuffle of all 60k indices, then freeze
    # the partition (same strategy as nnzap).
    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(
        len(full_train), generator=generator,
    ).tolist()

    train_indices = all_indices[:train_count]
    val_indices = all_indices[train_count:train_count + val_count]

    assert len(train_indices) == train_count
    assert len(val_indices) == val_count

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train, val_indices)

    return train_set, val_set, test_set


# ── Evaluation ────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: MnistNet,
    dataset: Subset | datasets.MNIST,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate accuracy and mean CE loss over a dataset.

    Matches nnzap's evaluate(): iterates in batches, sums
    per-sample CE loss, counts argmax-correct predictions,
    then computes mean loss and accuracy percentage.
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        # Per-sample CE loss (sum, not mean — we average
        # manually to match nnzap's accumulation).
        loss = F.cross_entropy(
            logits, labels, reduction="sum",
        )
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    mean_loss = total_loss / total
    accuracy_pct = correct / total * 100.0

    return {
        "correct": correct,
        "total": total,
        "mean_loss": mean_loss,
        "accuracy_pct": accuracy_pct,
    }


# ── Training ──────────────────────────────────────────────────

def train_epoch(
    model: MnistNet,
    loader: DataLoader,
    optimizer: torch.optim.SGD,
    device: torch.device,
) -> float:
    """Train for one epoch, return last-batch loss.

    nnzap reports train_loss as the loss of the final batch
    in each epoch (not the epoch average).  We match that
    here for apples-to-apples comparison.
    """
    model.train()

    last_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        loss.backward()
        optimizer.step()

        last_loss = loss.item()

    return last_loss


# ── Comparison helper ─────────────────────────────────────────

def compare_with_nnzap(
    pytorch_results: dict,
    nnzap_path: str,
) -> None:
    """Print side-by-side comparison with an nnzap benchmark."""
    with open(nnzap_path) as f:
        nnzap = json.load(f)

    sep = "-" * 54
    print(f"\n{'Comparison with nnzap':^54}")
    print(sep)
    print(f"{'Metric':<30} {'nnzap':>10} {'PyTorch':>10}")
    print(sep)

    nz_val_acc = nnzap["final_validation_accuracy_pct"]
    pt_val_acc = pytorch_results["final_validation_accuracy_pct"]
    print(
        f"{'Final val accuracy (%)':<30} "
        f"{nz_val_acc:>10.2f} {pt_val_acc:>10.2f}"
    )

    nz_val_loss = nnzap["final_validation_loss"]
    pt_val_loss = pytorch_results["final_validation_loss"]
    print(
        f"{'Final val loss':<30} "
        f"{nz_val_loss:>10.4f} {pt_val_loss:>10.4f}"
    )

    nz_test_acc = nnzap["final_test_accuracy_pct"]
    pt_test_acc = pytorch_results["final_test_accuracy_pct"]
    print(
        f"{'Test accuracy (%)':<30} "
        f"{nz_test_acc:>10.2f} {pt_test_acc:>10.2f}"
    )

    nz_time = nnzap["total_training_ms"]
    pt_time = pytorch_results["total_training_ms"]
    print(
        f"{'Training time (ms)':<30} "
        f"{nz_time:>10.0f} {pt_time:>10.0f}"
    )

    nz_tput = nnzap["throughput_images_per_sec"]
    pt_tput = pytorch_results["throughput_images_per_sec"]
    print(
        f"{'Throughput (img/s)':<30} "
        f"{nz_tput:>10.0f} {pt_tput:>10.0f}"
    )

    print(sep)

    # Per-epoch comparison.
    print(f"\n{'Per-epoch validation accuracy':^54}")
    print(sep)
    print(
        f"{'Epoch':<8} {'nnzap acc':>10} {'PT acc':>10}"
        f" {'nnzap ms':>10} {'PT ms':>10}"
    )
    print(sep)

    nz_epochs = nnzap["epochs"]
    pt_epochs = pytorch_results["epochs"]

    for nz_ep, pt_ep in zip(nz_epochs, pt_epochs):
        print(
            f"{nz_ep['epoch']:<8} "
            f"{nz_ep['validation_accuracy_pct']:>10.2f} "
            f"{pt_ep['validation_accuracy_pct']:>10.2f} "
            f"{nz_ep['duration_ms']:>10.0f} "
            f"{pt_ep['duration_ms']:>10.0f}"
        )

    print(sep)


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch reference for nnzap MNIST benchmark",
    )
    parser.add_argument(
        "--data-dir",
        default="data/mnist_torch",
        help="Directory for torchvision MNIST download",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to nnzap benchmark JSON for comparison",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to benchmarks/ as JSON",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (auto-detected if omitted)",
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
    val_interval = 5  # Validate every N epochs (matches nnzap).

    # ── Device ──

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("PyTorch MNIST reference benchmark")
    print("=================================")
    print()
    print(f"Device: {device}")
    print()

    # ── Architecture summary ──

    print("+-----------------------------------+")
    print("|     PyTorch Network Layout        |")
    print("+-----------------------------------+")
    print("|  Layer 0:  784 -> 128   (relu   ) |")
    print("|  Layer 1:  128 -> 64    (relu   ) |")
    print("|  Layer 2:   64 -> 10    (none   ) |")
    print("+-----------------------------------+")

    # ── Seed everything ──

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Model ──

    model = MnistNet().to(device)
    he_init(model)

    param_count = model.param_count()
    print(f"|  Total params: {param_count:<16} |")
    print("+-----------------------------------+")
    print()

    assert param_count == 109386, (
        f"Expected 109,386 params, got {param_count}"
    )

    # ── Dataset ──

    print("Loading MNIST... ", end="", flush=True)
    train_set, val_set, test_set = load_datasets(
        args.data_dir, seed, train_count, val_count,
    )
    print("done.")

    assert len(train_set) == train_count
    assert len(val_set) == val_count
    assert len(test_set) == test_count

    print(
        f"Split: train={train_count}  "
        f"val={val_count}  "
        f"test={test_count}"
    )
    print()

    # ── Optimizer ──
    # Plain SGD: no momentum, no weight decay — matches nnzap.

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.0,
        weight_decay=0.0,
    )

    # ── Training ──

    batches_per_epoch = train_count // batch_size

    # Use a separate generator for the DataLoader's shuffle
    # so that the epoch-level reshuffling is reproducible.
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )

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

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
        )

        # Validate every val_interval epochs and on the
        # last epoch — matches nnzap's val_interval logic.
        is_val_epoch = (
            epoch % val_interval == 0 or epoch == num_epochs
        )

        val_loss = 0.0
        val_acc = 0.0

        if is_val_epoch:
            val = evaluate(model, val_set, batch_size, device)
            val_loss = val["mean_loss"]
            val_acc = val["accuracy_pct"]

        epoch_ms = (
            (time.perf_counter() - epoch_start) * 1000.0
        )

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
    test_eval = evaluate(model, test_set, batch_size, device)
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
        "framework": "pytorch",
        "pytorch_version": torch.__version__,
        "device": str(device),
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
        "final_test_accuracy_pct": test_eval["accuracy_pct"],
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
            benchmarks_dir / f"pytorch_mnist_{timestamp}.json"
        )

        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {out_path}")
        print()

    # ── Compare ──

    if args.compare is not None:
        import glob

        paths = glob.glob(args.compare)
        if not paths:
            print(
                f"Warning: no files matched '{args.compare}'",
                file=sys.stderr,
            )
        else:
            # Use the most recent nnzap benchmark.
            nnzap_path = sorted(paths)[-1]
            print(f"Comparing with: {nnzap_path}")
            compare_with_nnzap(results, nnzap_path)
    else:
        # Auto-detect nnzap benchmarks for comparison.
        import glob

        nnzap_files = sorted(
            glob.glob("benchmarks/mnist_*.json"),
        )
        if nnzap_files:
            nnzap_path = nnzap_files[-1]
            print(f"Auto-comparing with: {nnzap_path}")
            compare_with_nnzap(results, nnzap_path)


if __name__ == "__main__":
    main()
