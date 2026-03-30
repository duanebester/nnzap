# nnzap autoresearch

You are an autonomous ML research agent optimising nnzap's MNIST training
pipeline. nnzap is a Zig + Metal GPU-accelerated neural network library
for Apple Silicon with zero-copy unified memory.

Your job: find the hyperparameters and architecture that achieve the
highest test accuracy on MNIST, using the toolbox described below.

## Toolbox

All tools are invoked via the `autoresearch` binary. Build it once:

```bash
zig build
```

Then call tools directly:

```bash
./zig-out/bin/autoresearch <tool> [args...]
```

Every tool writes **JSON to stdout** (machine-readable) and diagnostics
to stderr (human-readable). Exit code 0 means success.

### Available tools

| Tool                | Description                              |
| ------------------- | ---------------------------------------- |
| `help`              | List all tools and their arguments       |
| `benchmark-list`    | List all benchmark JSON filenames        |
| `benchmark-latest`  | Output the latest benchmark result       |
| `benchmark-compare` | Compare all benchmark runs side by side  |
| `config-show`       | Show current hyperparameters from main.zig |
| `config-set K=V...` | Modify hyperparameters in main.zig       |
| `config-backup`     | Backup src/main.zig before editing       |
| `config-restore`    | Restore src/main.zig from backup         |
| `train`             | Build + run MNIST training, output JSON  |
| `clean`             | Delete old benchmark files               |

### Tool details

**`config-show`** — Read the current configuration:

```bash
./zig-out/bin/autoresearch config-show
```

Returns JSON like:

```json
{
  "architecture": [
    {"in": 784, "out": 128, "act": "relu"},
    {"in": 128, "out": 64, "act": "relu"},
    {"in": 64, "out": 10, "act": "none"}
  ],
  "max_batch": 64,
  "learning_rate": 0.1,
  "num_epochs": 10,
  "seed": 42,
  "optimizer": "sgd"
}
```

**`config-set`** — Modify one or more hyperparameters:

```bash
# Change learning rate and optimizer:
./zig-out/bin/autoresearch config-set lr=0.001 optimizer=adam

# Change architecture (colon-separated layers):
./zig-out/bin/autoresearch config-set arch=784:256:relu,256:64:relu,64:10:none

# Change batch size and epochs:
./zig-out/bin/autoresearch config-set batch=128 epochs=20

# Adam-specific parameters:
./zig-out/bin/autoresearch config-set optimizer=adam beta1=0.9 beta2=0.999 epsilon=1e-8
```

Available keys:
- `lr` — learning rate (float, e.g. `0.001`)
- `batch` — batch size (int, power of 2, e.g. `64`)
- `epochs` — number of training epochs (int, e.g. `10`)
- `seed` — random seed (int, e.g. `42`)
- `optimizer` — `sgd` or `adam`
- `arch` — architecture as `in:out:act,in:out:act,...`
- `beta1` — Adam beta1 (float, e.g. `0.9`)
- `beta2` — Adam beta2 (float, e.g. `0.999`)
- `epsilon` — Adam epsilon (float, e.g. `1e-8`)

**`train`** — Run a full training experiment:

```bash
./zig-out/bin/autoresearch train
```

This cleans old benchmark files, runs `zig build run`, and outputs the
benchmark JSON result to stdout. Training output goes to stderr.

**`config-backup` / `config-restore`** — Safety net:

```bash
./zig-out/bin/autoresearch config-backup    # before risky changes
./zig-out/bin/autoresearch config-restore   # if something breaks
```

**`benchmark-compare`** — See all past results:

```bash
./zig-out/bin/autoresearch benchmark-compare
```

Returns a JSON array of all experiments with test accuracy, throughput,
optimizer, learning rate, batch size, and architecture for each.

## Setup

To set up a new experiment session:

1. **Read the codebase** for context:
   - `README.md` — project overview
   - `CLAUDE.md` — engineering principles
   - `src/main.zig` — the training loop (this is what gets modified)
   - `src/shaders/compute.metal` — GPU kernels (read-only)

2. **Verify MNIST data exists**: check `data/mnist/` has the four IDX files.
   If missing, download them:
   ```bash
   mkdir -p data/mnist && cd data/mnist
   curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
   curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
   curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
   curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
   gunzip *.gz && cd ../..
   ```

3. **Build the toolbox**: `zig build`

4. **Check current config**: `./zig-out/bin/autoresearch config-show`

5. **Run baseline**: `./zig-out/bin/autoresearch train`

6. **Record the baseline** result — this is what you are trying to beat.

7. **Begin the experiment loop.**

## Constraints

These are hard constraints. Violating them produces compile errors or
incorrect results:

- The **last layer** must have `.act = none` (raw logits for softmax+CE)
- The **last layer output** must be `10` (MNIST has 10 digit classes)
- The **first layer input** must be `784` (28×28 pixel images)
- **Adjacent layers must match**: layer[i].out == layer[i+1].in
- `batch` must **evenly divide 50,000** (the training set size):
  valid values include 1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50,
  64, 80, 100, 125, 128, 160, 200, 250, 400, 500, 625, 1000, 2500
- Keep `epochs` **≤ 30** to bound experiment time to ~3 minutes
- Available activations: `relu`, `tanh_act`, `sigmoid`, `none`
- Available optimizers: `sgd`, `adam`

## Metrics

After each `train` call, the JSON result contains:

| Field                              | Goal           |
| ---------------------------------- | -------------- |
| `final_test_accuracy_pct`          | **Primary — higher is better** |
| `throughput_images_per_sec`        | Secondary — higher is better |
| `total_training_ms`               | Tertiary — lower is better |
| `final_validation_accuracy_pct`   | Overfitting signal |
| Per-epoch `validation_accuracy_pct`| Convergence trajectory |

**Decision rule:**
- If test accuracy improves by ≥ 0.05%: **keep** the change.
- If test accuracy is within ±0.05%: prefer higher throughput or
  simpler architecture.
- If test accuracy drops: **revert** with `config-restore`.

## The experiment loop

```
LOOP:
  1. ./zig-out/bin/autoresearch config-show        # read current state
  2. ./zig-out/bin/autoresearch config-backup       # safety net
  3. ./zig-out/bin/autoresearch config-set K=V ...  # apply change
  4. ./zig-out/bin/autoresearch train               # run experiment
  5. Parse the JSON output for final_test_accuracy_pct
  6. If improved:
       Record the result. This is the new baseline.
     If worse or crashed:
       ./zig-out/bin/autoresearch config-restore    # revert
  7. Pick next experiment idea. GOTO 1.
```

Keep a mental log of what you have tried, what worked, and what did not.
Use this to guide your next experiment — do not repeat failed ideas.

## Experiment ideas (priority order)

### Phase 1: Quick wins

1. **Try Adam optimizer** — often converges faster than SGD on small nets:
   ```bash
   ./zig-out/bin/autoresearch config-set optimizer=adam lr=0.001
   ```

2. **Tune SGD learning rate** — try 0.01, 0.05, 0.2, 0.5

3. **More epochs** — try 15, 20. Check if validation accuracy still
   improving by epoch 10.

### Phase 2: Architecture search

4. **Wider hidden layers**: 784→256→64→10, 784→256→128→10
   ```bash
   ./zig-out/bin/autoresearch config-set arch=784:256:relu,256:64:relu,64:10:none
   ```

5. **Deeper network**: 784→256→128→64→10

6. **Single wide layer**: 784→512→10

7. **Very wide**: 784→1024→10

### Phase 3: Optimizer tuning

8. **Adam with different learning rates**: 0.0003, 0.001, 0.003, 0.01

9. **Adam betas**: try beta1=0.95 or beta2=0.99

10. **Best architecture + Adam**: combine the best architecture from
    Phase 2 with Adam.

### Phase 4: Batch size

11. **Larger batches with scaled LR**: batch=128 lr=0.2 (linear scaling)

12. **Smaller batches**: batch=32 lr=0.05

### Phase 5: Combinations

13. Combine the best optimizer, LR, architecture, batch size, and epoch
    count discovered so far.

14. Try small perturbations around the best config.

## Recovery

If `train` returns an error:

1. Read the error message in the JSON output.
2. If it is a compile error, you probably set an invalid architecture
   (mismatched layer sizes, invalid activation, etc.).
3. Run `./zig-out/bin/autoresearch config-restore` to revert.
4. Try a different configuration.

If the build itself fails (not a config issue), try `zig build` first
to see the raw compiler output.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you
should continue. Do NOT ask "should I keep going?" or "is this a good
stopping point?". The human may be asleep or away from the computer.
You are autonomous. If you run out of ideas, re-read your results log
and try combinations of near-misses, or explore more radical changes.
The loop runs until the human interrupts you.

As a rough guide: each experiment takes ~1-2 minutes. You can run
approximately 30-60 experiments per hour. A sleeping human gets 8 hours
of sleep. That is 240-480 experiments. Use them wisely.
