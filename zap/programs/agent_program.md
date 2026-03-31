# nnzap engine agent — LLM-powered experiment runner

The engine agent is a Zig binary that talks to Claude via the
Anthropic API. Claude decides which experiments to run; the agent
executes them using the engine_research toolbox, sends results back,
and loops until the experiment is complete. A two-loop architecture
gives each experiment a fresh conversation while preserving
cross-experiment learning through `experiments.jsonl`.

The agent IS the runtime — like Claude Code, but ours.

## How it works

```
┌──────────────────────────────────────┐
│  1. Load API key (ANTHROPIC_API_KEY)  │
│  2. Load engineering rules (CLAUDE.md)│
│  3. Build toolbox (zig build)         │
└───────────────────┬──────────────────┘
                    │
                    ▼
         ╔══════════════════════╗
         ║  OUTER LOOP          ║
         ║  (per experiment)    ║◄─────────────────────────────┐
         ╚══════════╤═══════════╝                              │
                    │                                          │
                    ▼                                          │
┌──────────────────────────────────────┐                       │
│  Load fresh history from JSONL       │                       │
│  Build fresh initial message         │                       │
│  (history + rules + context)         │                       │
└───────────────────┬──────────────────┘                       │
                    │                                          │
                    ▼                                          │
         ┌─────────────────────┐                               │
         │  INNER LOOP         │                               │
         │  (per turn)         │◄──┐                           │
         └─────────┬───────────┘   │                           │
                   │               │                           │
                   ▼               │                           │
┌──────────────────────────────────────┐                       │
│  Send messages to Claude             │                       │
│  Claude responds with tool calls     │                       │
│    e.g. snapshot, edit_file, check,  │                       │
│         bench, rollback_latest       │                       │
└───────────────────┬──────────────────┘                       │
                    │                                          │
                    ▼                                          │
┌──────────────────────────────────────┐                       │
│  Agent executes each tool call       │                       │
│    → calls engine_research CLI       │                       │
│    → captures JSON output            │                       │
│    → persists bench results to JSONL │                       │
│    → sends results back to Claude    │                       │
└───────────────────┬──────────────────┘                       │
                    │                                          │
              ┌─────┴─────┐                                    │
              │           │                                    │
         tool_use     end_turn                                 │
              │       or limit                                 │
              │           │                                    │
              └──►next    ▼                                    │
                  turn  ┌──────────────────────────┐           │
                        │  Save experiment run log │           │
                        │  run_{timestamp}.json    │           │
                        └────────────┬─────────────┘           │
                                     │ next experiment         │
                                     └─────────────────────────┘
                                     │ MAX_EXPERIMENTS reached
                                     ▼
                        ┌──────────────────────────┐
                        │  Print run summary        │
                        └──────────────────────────┘
```

### The key difference from the old single-loop agent

The previous version used a single flat `while` loop: one long
conversation for all experiments, with `MIN_TURNS` nudging to
keep Claude going and compaction logic to manage context growth.
This caused problems — context bloat, an infinite-read-loop bug
in compaction, and Claude losing track of earlier experiments.

The two-loop design solves all of these:

- **Outer loop** iterates over experiments (up to
  `MAX_EXPERIMENTS = 50`).
- **Inner loop** iterates over conversation turns within a single
  experiment (up to `MAX_TURNS_PER_EXPERIMENT = 25`).
- Each experiment starts a **fresh conversation** with Claude.
  History is injected via `experiments.jsonl`, not carried in the
  conversation. No compaction needed because conversations are
  short.
- When Claude's `stop_reason` is `end_turn` (not `tool_use`),
  the inner loop breaks and the outer loop starts the next
  experiment.

## Quick start

```bash
# 1. Set your API key.
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Build everything.
zig build

# 3. Run the engine agent.
./zig-out/bin/engine_agent
```

The agent takes no arguments. Configuration is via environment
variables.

### Environment variables

| Variable            | Required | Default                    | Description               |
| ------------------- | -------- | -------------------------- | ------------------------- |
| `ANTHROPIC_API_KEY` | Yes      | —                          | Your Anthropic API key    |
| `ANTHROPIC_MODEL`   | No       | `claude-sonnet-4-20250514` | Model to use for planning |

## Context seeding — the system prompt

The system prompt is compiled into the binary as a Zig multiline
string constant (`SYSTEM_PROMPT`). It contains everything Claude
needs to operate autonomously:

### What Claude is told

1. **Role**: autonomous systems-performance research agent
   optimising the nnzap neural network engine.
2. **Goal**: improve throughput (`throughput_images_per_sec`)
   while maintaining or improving accuracy
   (`final_test_accuracy_pct`).
3. **Protocol**: the exact sequence for each experiment —
   `show` the relevant code → `snapshot` → `edit_file` →
   `check` → `test` → `bench` → evaluate → keep or
   `rollback_latest`.
4. **Safety**: always snapshot before editing, always check and
   test before benching, rollback immediately on failure.
5. **History**: the first user message injects the full JSONL
   experiment log so Claude sees what has been tried before.
6. **Engineering rules**: the contents of `CLAUDE.md` are
   injected into the initial message so Claude follows project
   conventions when editing source files.

### Modifying the prompt

Edit the `SYSTEM_PROMPT` constant in `src/engine_agent.zig`.
Rebuild with `zig build`. The prompt is a Zig multiline string
(`\\` prefix per line).

## Tools

The agent exposes sixteen tools to Claude, each mapping to an
engine_research CLI command or a direct file operation:

| Tool              | Backend           | Description                                             |
| ----------------- | ----------------- | ------------------------------------------------------- |
| `snapshot`        | `engine_research` | Save engine source files as a restore point             |
| `snapshot_list`   | `engine_research` | List all saved snapshots with timestamps                |
| `rollback`        | `engine_research` | Restore engine files from a specific snapshot           |
| `rollback_latest` | `engine_research` | Restore from the most recent snapshot                   |
| `diff`            | `engine_research` | Show source changes since a snapshot                    |
| `check`           | `engine_research` | Compile-only validation (~2s)                           |
| `test`            | `engine_research` | Run full test suite for correctness                     |
| `bench`           | `engine_research` | Full MNIST training benchmark (~10s), returns JSON      |
| `bench_compare`   | `engine_research` | Compare all benchmark results side by side              |
| `show`            | `engine_research` | View a source file as structured JSON with line numbers |
| `show_function`   | `engine_research` | Extract a specific function with line numbers           |
| `read_file`       | direct file I/O   | Read raw contents of a project file                     |
| `write_file`      | direct file I/O   | Replace entire contents of an engine source file        |
| `edit_file`       | direct file I/O   | Targeted find-and-replace in a source file              |
| `list_directory`  | direct file I/O   | List files and subdirectories                           |
| `run_command`     | shell subprocess  | Execute a shell command (120s timeout)                  |

### Tool schemas

Tool definitions are passed to the Anthropic API as a JSON `tools`
array. They are compiled into the binary as the `TOOL_SCHEMAS`
constant. Each tool has a name, description, and `input_schema`.

Most tools take no input. Tools with inputs include `rollback`
(snapshot ID), `diff` (snapshot ID), `show` (file path),
`show_function` (file + function name), `read_file` (path),
`write_file` (path + content), `edit_file` (path + old/new
content), `list_directory` (path), and `run_command` (command
string).

### Adding new tools

1. Add the tool schema to `TOOL_SCHEMAS` in `engine_agent.zig`.
2. Add a dispatch branch in `dispatchTool`.
3. Implement the tool execution (typically calling an
   engine_research command or doing direct file I/O).
4. Rebuild with `zig build`.

## Design decisions

### Fresh context per experiment

Each experiment starts a new conversation with Claude. The full
experiment history is loaded from `experiments.jsonl` and injected
into the initial message. This means Claude always sees the
complete record of what worked and what failed, without carrying
stale conversation context from previous experiments.

Conversations are short (max 25 turns), so context bloat is not
an issue. The old compaction logic — which caused an infinite read
loop bug — has been removed entirely.

### `end_turn` signals experiment completion

The inner loop watches Claude's `stop_reason`. When it is
`end_turn` (rather than `tool_use`), Claude has finished the
current experiment. The inner loop breaks, the experiment's run
log is saved, and the outer loop starts the next experiment with
a fresh conversation.

### Cross-experiment learning

Bench results are appended to `experiments.jsonl` immediately
after each successful benchmark. When the next experiment starts,
the outer loop loads the updated history file. Claude sees the
full record — including results from the experiment that just
finished — and can make informed decisions about what to try next.

This is the mechanism by which the agent improves across
experiments: the JSONL file is the memory, not the conversation.

### Limits

| Constant                   | Value | Purpose                             |
| -------------------------- | ----- | ----------------------------------- |
| `MAX_EXPERIMENTS`          | 50    | Hard cap on outer loop iterations   |
| `MAX_TURNS_PER_EXPERIMENT` | 25    | Hard cap on inner loop iterations   |
| `MAX_MESSAGES`             | 512   | Message array capacity              |
| `MAX_TOOL_CALLS`           | 16    | Tool calls per API response         |
| `MAX_TOOL_OUTPUT`          | 50000 | Bytes per tool result               |
| `MAX_HISTORY_INJECT`       | 30000 | Bytes of JSONL injected into prompt |

The inner loop also breaks on message limit or unrecoverable API
failure.

## Storage

All persistent state lives in `.engine_agent_history/`:

```
.engine_agent_history/
├── experiments.jsonl                  # Append-only bench results
├── _request.json                      # Last API request (debug)
├── run_2025-06-10T14-20-00Z.json      # Experiment 1 conversation
├── run_2025-06-10T14-23-45Z.json      # Experiment 2 conversation
└── run_2025-06-10T14-27-12Z.json      # Experiment 3 conversation
```

### `experiments.jsonl` — cross-experiment memory

Each time `bench` is called successfully, the full benchmark JSON
is collapsed to a single line and appended to this file. On the
next experiment, this content is injected into the first user
message so Claude sees what has been tried before.

The file is capped at `MAX_HISTORY_INJECT` bytes (30 KB) when
loaded into the prompt. This keeps the context window manageable
while preserving the most recent experiments.

### `run_{timestamp}.json` — conversation logs

Each experiment saves its own conversation log. The full message
array (all messages exchanged between the agent and Claude for
that experiment) is saved as a JSON array. These files are for
debugging and auditing — the agent does not read them on startup.

Because each experiment gets its own log, you can review
individual experiments without digging through a single monolithic
conversation file.

### `_request.json` — debug artifact

The last API request body sent to Claude. Useful for debugging
prompt construction issues. Overwritten on each API call.

## Output

### stderr — human-readable progress

```
nnzap engine agent — LLM-powered experiment runner
====================================================

Model: claude-sonnet-4-20250514
Building toolbox...
  done.

═══ Experiment 1 / 50 ═══

Loaded 2649 bytes of experiment history.

--- Turn 1 ---

  Claude: I'll examine the current engine source to understand
          the hot path.
  Tool:   show {"file":"../nn/src/metal.zig"}

--- Turn 2 ---

  Claude: The dispatch loop can be tightened. Let me snapshot first.
  Tool:   snapshot
  Tool:   edit_file {"path":"../nn/src/metal.zig", ...}

--- Turn 3 ---

  Claude: Check the edit compiles.
  Tool:   check

--- Turn 4 ---

  Claude: Passed. Now bench.
  Tool:   bench

--- Turn 5 ---

  Claude: 14,200 img/s vs 12,800 baseline — 10.9% improvement.
          Keeping the change.

Experiment 1 finished (5 turns).
Run log saved: .engine_agent_history/run_2025-06-10T14-20-00Z.json

═══ Experiment 2 / 50 ═══

Loaded 3104 bytes of experiment history.

--- Turn 1 ---

  Claude: Previous experiment improved dispatch. Now I'll look at
          the matmul kernel.
  Tool:   show_function {"file":"../nn/src/shaders/forward.metal", ...}

...

==============================================
  Run summary
==============================================
  Experiments: 3
  Turns:       14
  Tool calls:  28
  Benchmarks:  3
  API errors:  0
  Total time:  4m 12s
==============================================
```

## Architecture

The agent follows nnzap's engineering principles:

- **Rule 4 (limits)**: `MAX_EXPERIMENTS = 50`,
  `MAX_TURNS_PER_EXPERIMENT = 25`, `MAX_MESSAGES = 512`,
  `MAX_TOOL_CALLS = 16` per response, `MAX_TOOL_OUTPUT = 50000`.
- **Rule 5 (no compaction)**: conversations are short enough that
  context bloat is not an issue. Cross-experiment memory lives in
  the JSONL file, not in the conversation.
- **Rule 6 (explicit control flow)**: two nested `while` loops.
  No callbacks, no async. The outer loop drives experiments; the
  inner loop drives turns.
- **Rule 8 (batching)**: each experiment batches all its turns
  into one conversation. Cross-experiment learning is batched
  through the JSONL file.
- **Rule 10 (shrink scope)**: tool execution is isolated in
  `executeSingleTool`. API calling in `callApi`. Parsing in
  `parseApiResponse`. Each experiment's message array is scoped
  to the inner loop.
- **Rule 12 (minimal dependencies)**: uses `std.http` for
  HTTPS. No external Zig packages.
- **Rule 17 (handle all errors)**: API failures break the inner
  loop gracefully. History writes are fire-and-forget. Tool
  failures are reported back to Claude as tool errors.
- **Rule 25 (tooling in Zig)**: the entire agent is one Zig file.

### Relationship to other tools

```
┌──────────────────────┐
│  Claude (API)         │  The brain. Decides experiments.
└──────────┬───────────┘
           │ HTTPS
           ▼
┌──────────────────────┐
│  engine_agent.zig     │  The runtime. Two-loop architecture.
│  (zap/src/)           │  Manages conversations + history.
└──────────┬───────────┘
           │ spawns
           ▼
┌──────────────────────┐
│  engine_research.zig  │  The toolbox. snapshot/check/test/bench.
│  (zap/src/)           │
└──────────┬───────────┘
           │ spawns
           ▼
┌──────────────────────┐
│  mnist.zig            │  The training binary.
│  (nn/examples/)       │
└──────────────────────┘
```

The agent never touches `../nn/src/` files directly except through the
`read_file`, `write_file`, and `edit_file` tools (which enforce
path allowlists). It calls engine_research for snapshot, build,
test, and bench operations. The agent's only responsibilities are:

1. Managing the two-loop conversation lifecycle with Claude.
2. Routing tool calls to engine_research commands or file I/O.
3. Persisting bench results to the JSONL log.
4. Injecting history into fresh conversations for cross-experiment
   learning.

## Timing and cost

- Each `bench` call takes ~10–15 seconds (build + train +
  evaluate).
- Each `check` call takes ~2 seconds (compile only).
- API calls take 1–5 seconds depending on response length.
- A single experiment typically takes 3–8 turns and completes
  in 1–3 minutes.
- A full agent run of 5–10 experiments completes in 10–30
  minutes.
- Claude decides when each experiment is done (via `end_turn`)
  and the outer loop starts the next one.
- API cost is modest: short tool-calling conversations with
  Sonnet cost roughly $0.01–0.03 per experiment.

## Recovery

If the agent is interrupted (Ctrl-C):

```bash
# The last experiment's source edits may still be applied.
# Roll back to the most recent snapshot.
./zig-out/bin/engine_research rollback-latest
```

All completed bench results are already persisted in
`experiments.jsonl` — no data is lost. The conversation log for
the interrupted experiment will not be saved (it is written at
the end of each experiment), but the JSONL history survives.

### Clearing history

To start fresh (new agent, no memory of past experiments):

```bash
rm .engine_agent_history/experiments.jsonl
```

To wipe everything:

```bash
rm -rf .engine_agent_history/
```

The agent recreates the directory on next run.

### Changing models

To use a different Claude model (e.g., for cost or capability):

```bash
export ANTHROPIC_MODEL=claude-sonnet-4-20250514
./zig-out/bin/engine_agent
```

Sonnet is the default and recommended for the best balance of
tool-calling reliability and cost. Opus is more capable but
slower and more expensive.
