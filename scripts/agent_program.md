# nnzap agent — LLM-powered experiment runner

The agent is a Zig binary that talks to Claude via the Anthropic API.
Claude decides which experiments to run; the agent executes them using
the autoresearch toolbox, sends results back, and loops until Claude
stops or `MAX_TURNS` is reached.

The agent IS the runtime — like Claude Code, but ours.

## How it works

```
┌──────────────────────────────────────┐
│  1. Load API key (ANTHROPIC_API_KEY)  │
│  2. Load experiment history (JSONL)   │
│  3. Build toolbox (zig build)         │
│  4. Build system prompt + context     │
│  5. Send initial message to Claude    │
└───────────────────┬──────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Claude responds with tool calls     │◄──┐
│    e.g. config_show, config_set,     │   │
│         train, config_restore        │   │
└───────────────────┬──────────────────┘   │
                    │                      │
                    ▼                      │
┌──────────────────────────────────────┐   │
│  Agent executes each tool call       │   │
│    → calls autoresearch CLI          │   │
│    → captures JSON output            │   │
│    → persists train results to JSONL │   │
│    → sends results back to Claude    │   │
└───────────────────┬──────────────────┘   │
                    │ next turn             │
                    └──────────────────────►┘
                    │ Claude says "end_turn"
                    │ or MAX_TURNS reached
                    ▼
┌──────────────────────────────────────┐
│  Save conversation to run log        │
│  Print summary                       │
└──────────────────────────────────────┘
```

### The key difference from the old static agent

The previous version had a hardcoded `PLAN` array — a fixed list of
experiments compiled into the binary. Claude was not involved; the
agent was a deterministic sweep runner.

Now Claude IS the planner. The system prompt seeds it with domain
knowledge (constraints, strategy, decision rules), and the experiment
history from previous runs gives it context to avoid repeating past
work. Claude decides what to try, in what order, and when to stop.

## Quick start

```bash
# 1. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Build everything
zig build

# 3. Run the agent
./zig-out/bin/agent

# Or use the build step
zig build agent
```

The agent takes no arguments. Configuration is via environment
variables.

### Environment variables

| Variable            | Required | Default           | Description               |
| ------------------- | -------- | ----------------- | ------------------------- |
| `ANTHROPIC_API_KEY` | Yes      | —                 | Your Anthropic API key    |
| `ANTHROPIC_MODEL`   | No       | `claude-opus-4-6` | Model to use for planning |

## Context seeding — the system prompt

The system prompt is compiled into the binary as a Zig multiline
string constant (`SYSTEM_PROMPT`). It contains everything Claude
needs to operate autonomously:

### What Claude is told

1. **Role**: autonomous ML research agent optimizing MNIST training.
2. **Goal**: maximize `final_test_accuracy_pct`.
3. **Protocol**: the exact sequence for each experiment —
   `config_show` → `config_backup` → `config_set` → `train` →
   evaluate → keep or `config_restore`.
4. **Decision rules**:
   - Accuracy ≥ baseline + 0.05 pp → keep.
   - Accuracy within ± 0.05 pp → keep if throughput improved.
   - Accuracy dropped → revert.
5. **Constraints**: architecture rules (784 input, 10 output,
   matching layers), valid batch sizes, epoch limits, available
   activations and optimizers.
6. **Strategy**: phased approach — optimizer first, then learning
   rate, architecture, batch size, fine-tuning.
7. **History**: the first user message injects the full JSONL
   experiment log so Claude knows what has been tried before.

### Modifying the prompt

Edit the `SYSTEM_PROMPT` constant in `scripts/agent.zig`. Rebuild
with `zig build`. The prompt is a Zig multiline string (`\\` prefix
per line).

## Tools

The agent exposes seven tools to Claude, each mapping to an
autoresearch CLI command:

| Tool                | Autoresearch command | Description                                  |
| ------------------- | -------------------- | -------------------------------------------- |
| `config_show`       | `config-show`        | Show current hyperparameters                 |
| `config_set`        | `config-set K=V ...` | Modify hyperparameters                       |
| `config_backup`     | `config-backup`      | Backup main.zig before changes               |
| `config_restore`    | `config-restore`     | Restore main.zig from backup                 |
| `train`             | `train`              | Build + run training, returns benchmark JSON |
| `benchmark_compare` | `benchmark-compare`  | Compare all past benchmark runs              |
| `benchmark_latest`  | `benchmark-latest`   | Output the most recent benchmark             |

### Tool schemas

Tool definitions are passed to the Anthropic API as a JSON `tools`
array. They are compiled into the binary as the `TOOL_SCHEMAS`
constant. Each tool has a name, description, and `input_schema`.

Most tools take no input. The exception is `config_set`, which takes
a `settings` array of `key=value` strings:

```json
{
  "name": "config_set",
  "input_schema": {
    "type": "object",
    "properties": {
      "settings": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Key=value pairs, e.g. [\"optimizer=adam\", \"lr=0.001\"]"
      }
    },
    "required": ["settings"]
  }
}
```

### Adding new tools

1. Add the tool schema to `TOOL_SCHEMAS` in `agent.zig`.
2. Add a dispatch branch in `dispatchTool`.
3. Implement the tool execution (typically calling an autoresearch
   or engine_research command).
4. Rebuild with `zig build`.

## Storage

All persistent state lives in `.agent_history/`:

```
.agent_history/
├── experiments.jsonl              # Append-only train results
├── _request.json                  # Last API request (debug)
├── run_2025-03-26T17-45-00Z.json  # Conversation log from run 1
└── run_2025-03-27T09-12-00Z.json  # Conversation log from run 2
```

### `experiments.jsonl` — cross-run memory

Each time `train` is called successfully, the full benchmark JSON
is collapsed to a single line and appended to this file. On the
next agent run, this content is injected into the first user
message so Claude sees what has been tried before.

This is the mechanism by which the agent improves across runs:
Claude reads the history, sees which experiments worked and which
failed, and makes informed decisions about what to try next.

The file is capped at `MAX_HISTORY_INJECT` bytes (30 KB) when
loaded into the prompt. This keeps the context window manageable
while preserving the most recent experiments.

### `run_{timestamp}.json` — conversation logs

The full conversation (all messages exchanged between the agent
and Claude) is saved as a JSON array at the end of each run.
These files are for debugging and auditing — the agent does not
read them on startup.

### `_request.json` — debug artifact

The last API request body sent to Claude. Useful for debugging
prompt construction issues. Overwritten on each API call.

## Output

### stderr — human-readable progress

```
nnzap agent — LLM-powered experiment runner
============================================

Model: claude-opus-4-6
Building toolbox...
  done.
Loaded 2649 bytes of experiment history.

--- Turn 1 ---

  Claude: I'll start by examining the current configuration.
  Tool:   config_show

--- Turn 2 ---

  Claude: The baseline is SGD with lr=0.1. Let me try Adam.
  Tool:   config_backup
  Tool:   config_set {"settings":["optimizer=adam","lr=0.001"]}

--- Turn 3 ---

  Claude: Now let me train with Adam.
  Tool:   train

--- Turn 4 ---

  Claude: 97.94% with Adam vs 97.82% baseline — keeping it.
          Let me try a wider architecture next.
  Tool:   config_backup
  Tool:   config_set {"settings":["arch=784:256:relu,256:128:relu,128:10:none"]}

...

--- Turn 14 ---

  Claude: Improvements have plateaued. Best result: 98.12%
          with Adam lr=0.001, 256->128->10 architecture.
          Summary: ...

Agent finished (14 turns).
Run log saved: .agent_history/run_2025-03-26T17-45-00Z.json
```

## Architecture

The agent follows nnzap's engineering principles:

- **Rule 4 (limits)**: `MAX_TURNS = 200`, `MAX_MESSAGES = 512`,
  `MAX_TOOL_CALLS = 16` per response, `MAX_TOOL_OUTPUT = 50000`.
- **Rule 6 (explicit control flow)**: a simple `while` loop over
  turns. No callbacks, no async.
- **Rule 10 (shrink scope)**: tool execution is isolated in
  `executeSingleTool`. API calling in `callApi`. Parsing in
  `parseApiResponse`.
- **Rule 12 (minimal dependencies)**: uses `curl` as a subprocess
  for HTTP — no TLS/HTTP library dependencies. curl ships with
  every Mac.
- **Rule 17 (handle all errors)**: API failures break the loop
  gracefully. History writes are fire-and-forget. Train failures
  are reported back to Claude as tool errors.
- **Rule 25 (tooling in Zig)**: the entire agent is one Zig file.

### Relationship to other tools

```
┌──────────────────────┐
│  Claude (API)         │  The brain. Decides experiments.
└──────────┬───────────┘
           │ HTTP (via curl)
           ▼
┌──────────────────────┐
│  agent.zig            │  The runtime. Executes tools.
│  (this binary)        │  Manages conversation + history.
└──────────┬───────────┘
           │ spawns
           ▼
┌──────────────────────┐
│  autoresearch.zig     │  The toolbox. config/train/bench.
└──────────┬───────────┘
           │ spawns
           ▼
┌──────────────────────┐
│  nnzap (main.zig)     │  The training binary.
└──────────────────────┘
```

The agent never touches `src/main.zig` directly. It calls
autoresearch, which handles all config parsing and file I/O.
The agent's only responsibilities are:

1. Managing the conversation with Claude.
2. Routing tool calls to autoresearch commands.
3. Persisting experiment results to the JSONL log.
4. Injecting history into the prompt for cross-run learning.

## Timing and cost

- Each `train` call takes ~10–15 seconds (build + train + evaluate).
- API calls take 1–5 seconds depending on response length.
- A typical run of 10–15 experiments completes in 3–5 minutes.
- Claude decides how many experiments to run — it will naturally
  stop when improvements plateau.
- API cost is modest: tool-calling conversations with Sonnet cost
  roughly $0.01–0.05 per agent run.

## Recovery

If the agent is interrupted (Ctrl-C):

```bash
# The last experiment's config change may still be applied.
./zig-out/bin/autoresearch config-restore
```

All completed train results are already persisted in
`experiments.jsonl` — no data is lost. The conversation log
for the interrupted run will not be saved (it is written at
the end), but the JSONL history survives.

### Clearing history

To start fresh (new agent, no memory of past experiments):

```bash
rm .agent_history/experiments.jsonl
```

To wipe everything:

```bash
rm -rf .agent_history/
```

The agent recreates the directory on next run.

### Changing models

To use a different Claude model (e.g., for cost or capability):

```bash
export ANTHROPIC_MODEL=claude-3-5-haiku-20241022
./zig-out/bin/agent
```

Sonnet is recommended for the best balance of tool-calling
reliability and cost. Haiku is cheaper but may make less
sophisticated experiment choices.
