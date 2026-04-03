# Autoresearch Framework Plan

Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
project — where the agent runtime is external (Claude Code) and the human
iterates on a `program.md` skill file. We built our own agent runtime in Zig
instead, but the design principles transfer: thin agent, fat toolbox,
externalised prompts.

This document describes the refactor from two duplicated agent
implementations into a shared framework with pluggable research
profiles.

## Current state

We have two independent agent systems that share the same structural
spine but duplicate ~1,500 lines of identical logic:

| File                  | Lines | Role                             |
| --------------------- | ----- | -------------------------------- |
| `agent.zig`           | 768   | MNIST training agent             |
| `engine_agent.zig`    | 3,929 | Bonsai inference agent           |
| `autoresearch.zig`    | 1,038 | MNIST toolbox binary             |
| `engine_research.zig` | 1,339 | Bonsai toolbox binary            |
| `tools.zig`           | 630   | Shared CLI/file utilities        |
| `api_client.zig`      | 802   | Anthropic HTTP client            |
| `ollama_client.zig`   | 1,065 | Local LLM client (two-tier mode) |
| **Total**             | 9,571 |                                  |

## What the two agents share

Both agents follow the same loop:

```
load config → load prompts → build context → call API
  → parse tool_use → spawn toolbox subprocess → feed result back → repeat
```

Duplicated across `agent.zig` and `engine_agent.zig`:

- **API calling with retry** — `callApi` / `callApiWithRetry` (curl-based)
- **Tool dispatch** — an if-chain mapping tool names to subprocess calls
- **Subprocess spawning** — `callAutoresearch` / `callEngineResearch`
  (identical except for the binary path constant)
- **History/experiment persistence** — JSONL append, history loading
- **Message building** — assistant/tool-result message assembly
- **Setup** — API key loading, model selection, `zig build`
- **Hardcoded prompts and tool schemas** — string literals in Zig source

## What differs between them

| Concern              | MNIST (`agent.zig`) | Bonsai (`engine_agent.zig`)       |
| -------------------- | ------------------- | --------------------------------- |
| Toolbox binary       | `autoresearch`      | `engine_research`                 |
| Tool count           | 7 tools             | 20 tools                          |
| Loop shape           | Single-tier only    | Single-tier OR two-tier           |
| Two-tier mode        | —                   | Opus strategist + Ollama executor |
| Inline tool handlers | 1 (`config_set`)    | 9 (file I/O, shell, git, etc.)    |
| History format       | Same JSONL          | Same JSONL + summaries JSONL      |
| Prompt iteration     | Requires recompile  | Requires recompile                |
| Engineering rules    | —                   | Loads `CLAUDE.md` at runtime      |

## Problems with the current architecture

### `engine_agent.zig` (3,929 lines, 121 symbols)

The big one. It does everything in a single file:

- Hardcoded prompts and tool schemas (~400 lines)
- 9 tool handlers implemented inline (~900 lines)
- History/summary persistence (~350 lines)
- Path validation and file I/O helpers (~150 lines, duplicating `tools.zig`)
- Hand-rolled JSON field extraction and unescaping (~150 lines)
- Two code paths for tool execution (some shell out, some run inline)

The hand-rolled JSON parser (`extractRequiredField`) searches for
`"field":"value"` with string matching. It breaks on nested objects,
numeric values, and fields with similar name prefixes. This should
be `std.json.parseFromSliceLeaky`.

### `agent.zig` (768 lines)

Simpler, but still has hardcoded prompts and its own copy of the
API loop, dispatch, and history logic.

### Duplication map

These function pairs are structurally identical:

| `agent.zig`         | `engine_agent.zig`   | What it does                     |
| ------------------- | -------------------- | -------------------------------- |
| `callAutoresearch`  | `callEngineResearch` | Spawn toolbox subprocess         |
| `dispatchTool`      | `dispatchTool`       | Tool name → CLI args             |
| `callApi`           | `callApi`            | HTTP request via curl            |
| —                   | `callApiWithRetry`   | Retry wrapper (missing in agent) |
| `buildToolbox`      | `buildToolbox`       | `zig build`                      |
| `loadApiKey`        | `loadApiKey`         | Read env var                     |
| `loadModel`         | `loadModel`          | Read env var or default          |
| `ensureHistoryDir`  | `ensureHistoryDir`   | Mkdir for JSONL files            |
| `appendExperiment`  | `appendExperiment`   | Append to JSONL                  |
| `printHeader`       | `printHeader`        | Banner                           |
| `executeTools`      | `executeTools`       | Map over tool calls              |
| `executeSingleTool` | `executeSingleTool`  | Dispatch + log one tool          |

## Target architecture

### Concept: profiles

A **profile** is a research domain. Each profile is named after its
optimisation target, not the implementation layer:

- **mnist** — optimise MNIST training (hyperparams, architecture)
- **bonsai** — optimise Bonsai 1.7B inference (Metal kernels, dispatch)

Future profiles (tokenizer tuning, quantisation search, etc.) follow
the same pattern.

A profile consists of:

1. A **toolbox binary** — standalone CLI that implements domain tools
2. A **tool mapping table** — maps LLM tool names to CLI subcommands
3. **Prompt files** — externalised `.md` and `.json` in `programs/`
4. An optional **custom loop** — e.g. two-tier mode for bonsai

### File layout after refactor

```
src/
  agent_core.zig        ~1,000 lines  Generic agent loop + dispatch  (NEW)
  api_client.zig           802 lines  Anthropic HTTP client           (unchanged)
  ollama_client.zig      1,065 lines  Local LLM client               (unchanged)
  tools.zig                630 lines  Shared CLI/file utilities       (unchanged)

  mnist_agent.zig         ~100 lines  MNIST profile config            (renamed from agent.zig)
  mnist_research.zig     1,038 lines  MNIST toolbox binary            (renamed from autoresearch.zig)

  bonsai_agent.zig        ~600 lines  Bonsai profile config + 2-tier  (renamed from engine_agent.zig)
  bonsai_research.zig   ~2,000 lines  Bonsai toolbox binary           (renamed from engine_research.zig)

programs/
  mnist_system.md                     MNIST system prompt
  mnist_tools.json                    MNIST tool schemas
  mnist_program.md                    MNIST skill file (renamed from agent_program.md)

  bonsai_system.md                    Bonsai system prompt
  bonsai_strategist.md                Bonsai two-tier strategist prompt
  bonsai_executor.md                  Bonsai two-tier executor prompt
  bonsai_tools.json                   Bonsai tool schemas
  bonsai_program.md                   Bonsai skill file (renamed from engine_program.md)

  program.md                          Shared conventions (unchanged)
```

History directories follow the same naming:

- `.mnist_history/experiments.jsonl`
- `.bonsai_history/experiments.jsonl`
- `.bonsai_history/summaries.jsonl`

### Rename map

| Before                       | After                        |
| ---------------------------- | ---------------------------- |
| `src/agent.zig`              | `src/mnist_agent.zig`        |
| `src/autoresearch.zig`       | `src/mnist_research.zig`     |
| `src/engine_agent.zig`       | `src/bonsai_agent.zig`       |
| `src/engine_research.zig`    | `src/bonsai_research.zig`    |
| `programs/agent_program.md`  | `programs/mnist_program.md`  |
| `programs/engine_program.md` | `programs/bonsai_program.md` |
| `.agent_history/`            | `.mnist_history/`            |
| `.engine_agent_history/`     | `.bonsai_history/`           |

### Architecture diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        agent_core.zig                        │
│                        (~1,000 lines)                        │
│                                                              │
│  Provides:                                                   │
│    AgentConfig          — profile definition struct           │
│    ToolMapping          — tool name → CLI subcommand + shape  │
│    run()                — generic experiment loop             │
│    callToolbox()        — spawn subprocess, capture output    │
│    dispatchTool()       — route via ToolMapping table         │
│    loadPromptFile()     — read .md/.json from programs/       │
│    callApiWithRetry()   — HTTP with exponential backoff       │
│    appendExperiment()   — JSONL history persistence           │
│    buildHistorySummary()— load + format past experiments      │
│                                                              │
│  Does NOT contain:                                           │
│    Tool implementations (live in toolbox binaries)           │
│    Prompts (live in programs/*.md)                            │
│    Domain-specific context building                          │
│    Two-tier orchestration (lives in bonsai_agent.zig)        │
└──────────────┬───────────────────────┬───────────────────────┘
               │                       │
      ┌────────▼────────┐    ┌─────────▼─────────┐
      │  mnist_agent.zig │    │  bonsai_agent.zig  │
      │  (~100 lines)    │    │  (~600 lines)      │
      │                  │    │                    │
      │  config:         │    │  config:           │
      │   toolbox=mnist  │    │   toolbox=bonsai   │
      │   prompts=mnist  │    │   prompts=bonsai   │
      │   tool_map=[7]   │    │   tool_map=[20]    │
      │                  │    │                    │
      │  main:           │    │  main:             │
      │   core.run(cfg)  │    │   if two-tier:     │
      │                  │    │     custom loop    │
      │                  │    │   else:            │
      │                  │    │     core.run(cfg)  │
      └────────┬─────────┘    └─────────┬──────────┘
               │                        │
               │  ALL tools go through  │
               │  one path: spawn proc  │
               │                        │
      ┌────────▼─────────┐    ┌─────────▼──────────┐
      │ mnist_research    │    │ bonsai_research     │
      │ (toolbox binary)  │    │ (toolbox binary)    │
      │                   │    │                     │
      │ config-show       │    │ snapshot            │
      │ config-set K=V    │    │ rollback <id>       │
      │ config-backup     │    │ diff <id>           │
      │ config-restore    │    │ check / test        │
      │ train             │    │ bench / bench-infer  │
      │ benchmark-latest  │    │ read-file <path>    │
      │ benchmark-compare │    │ write-file -f       │
      │                   │    │ edit-file -f         │
      │                   │    │ run-cmd -f           │
      │                   │    │ commit -f            │
      │                   │    │ add-summary -f       │
      │                   │    │ history [count]      │
      │                   │    │ show / show-function │
      │                   │    │ cwd / list-dir       │
      └───────────────────┘    └─────────────────────┘
```

## The key design: ToolMapping table

Every tool dispatch follows one of three shapes:

1. **no_input** — tool has no parameters, just pass the subcommand name.
   Examples: `snapshot`, `check`, `test`, `bench`, `cwd`.

2. **string_arg** — extract one JSON field, pass as a CLI positional arg.
   Examples: `rollback <snapshot_id>`, `diff <snapshot_id>`,
   `show <path>`, `history <count>`, `read_file <path>`.

3. **json_payload** — write the full JSON input to `_tool_input.json`,
   pass `-f _tool_input.json` on the CLI. The toolbox binary reads,
   parses with `std.json`, and deletes it.
   Examples: `write_file`, `edit_file`, `run_command`, `commit`.

This replaces the 130-line if-chain in `engine_agent.zig` with a
data-driven dispatcher:

```zig
pub const ToolShape = enum {
    no_input,
    string_arg,
    json_payload,
};

pub const ToolMapping = struct {
    tool_name: []const u8,      // LLM-facing name: "read_file"
    subcommand: []const u8,     // CLI-facing name: "read-file"
    shape: ToolShape,
    field: ?[]const u8 = null,  // For string_arg: which JSON field to extract.
};
```

The bonsai profile declares its mapping as:

```zig
const bonsai_tools = [_]core.ToolMapping{
    .{ .tool_name = "snapshot",       .subcommand = "snapshot",       .shape = .no_input },
    .{ .tool_name = "snapshot_list",  .subcommand = "snapshot-list",  .shape = .no_input },
    .{ .tool_name = "rollback",       .subcommand = "rollback",       .shape = .string_arg, .field = "snapshot_id" },
    .{ .tool_name = "rollback_latest",.subcommand = "rollback-latest",.shape = .no_input },
    .{ .tool_name = "diff",           .subcommand = "diff",           .shape = .string_arg, .field = "snapshot_id" },
    .{ .tool_name = "check",          .subcommand = "check",          .shape = .no_input },
    .{ .tool_name = "test",           .subcommand = "test",           .shape = .no_input },
    .{ .tool_name = "bench",          .subcommand = "bench",          .shape = .no_input },
    .{ .tool_name = "bench_infer",    .subcommand = "bench-infer",    .shape = .no_input },
    .{ .tool_name = "bench_compare",  .subcommand = "bench-compare",  .shape = .no_input },
    .{ .tool_name = "read_file",      .subcommand = "read-file",      .shape = .string_arg, .field = "path" },
    .{ .tool_name = "write_file",     .subcommand = "write-file",     .shape = .json_payload },
    .{ .tool_name = "edit_file",      .subcommand = "edit-file",      .shape = .json_payload },
    .{ .tool_name = "list_directory", .subcommand = "list-dir",       .shape = .string_arg, .field = "path" },
    .{ .tool_name = "run_command",    .subcommand = "run-cmd",        .shape = .json_payload },
    .{ .tool_name = "cwd",           .subcommand = "cwd",            .shape = .no_input },
    .{ .tool_name = "commit",         .subcommand = "commit",         .shape = .json_payload },
    .{ .tool_name = "add_summary",    .subcommand = "add-summary",    .shape = .json_payload },
    .{ .tool_name = "history",        .subcommand = "history",        .shape = .string_arg, .field = "count" },
    .{ .tool_name = "show",           .subcommand = "show",           .shape = .string_arg, .field = "path" },
    .{ .tool_name = "show_function",  .subcommand = "show-function",  .shape = .json_payload },
};
```

The generic dispatcher in `agent_core.zig` walks this table:

```zig
fn dispatchTool(
    config: *const AgentConfig,
    arena: Allocator,
    call: ToolCall,
) ToolOutput {
    std.debug.assert(call.name.len > 0);

    for (config.tool_map) |mapping| {
        if (api.eql(call.name, mapping.tool_name)) {
            return switch (mapping.shape) {
                .no_input => callToolbox(
                    config, arena, &.{mapping.subcommand},
                ),
                .string_arg => dispatchStringArg(
                    config, arena, mapping, call.input_json,
                ),
                .json_payload => dispatchJsonPayload(
                    config, arena, mapping.subcommand, call.input_json,
                ),
            };
        }
    }

    return .{ .stdout = "Error: unknown tool", .success = false };
}
```

This is ~15 lines replacing ~130 lines. Every profile gets the same
dispatcher for free.

## The key simplification: all tools behind subprocess boundary

Before (engine_agent.zig, two code paths):

```
tool_use("read_file", json_input)
  │
  ▼
engine_agent.zig:
  extractRequiredField(json, "path")    ← hand-rolled JSON parser
  unescapeJsonString(raw_path)          ← hand-rolled unescaper
  isAllowedReadPath(path)              ← path validation
  resolveToFs(path)                    ← duplicates tools.zig
  readFileContent(path)                ← duplicates tools.zig
  return result

tool_use("snapshot", json_input)
  │
  ▼
engine_agent.zig:
  callEngineResearch(&.{"snapshot"})   ← different code path
  return stdout
```

After (agent_core.zig, one code path for all tools):

```
tool_use("read_file", {"path": "nn/src/metal.zig"})
  │
  ▼
agent_core.zig:
  find "read_file" in tool_map → shape=string_arg, field="path"
  extract "path" from JSON → "nn/src/metal.zig"
  spawn: bonsai_research read-file nn/src/metal.zig
  return stdout

tool_use("snapshot", {})
  │
  ▼
agent_core.zig:
  find "snapshot" in tool_map → shape=no_input
  spawn: bonsai_research snapshot
  return stdout
```

The agent does not parse tool inputs, validate paths, or know what
any tool does. All smarts live in the toolbox binary where they can
be tested independently without an API key:

```sh
./zig-out/bin/bonsai_research read-file nn/src/metal.zig
./zig-out/bin/bonsai_research check
./zig-out/bin/bonsai_research bench
./zig-out/bin/mnist_research config-show
./zig-out/bin/mnist_research train
```

## AgentConfig struct

```zig
pub const AgentConfig = struct {
    // Identity.
    name: []const u8,                  // "mnist" or "bonsai"
    toolbox_path: []const u8,          // "./zig-out/bin/bonsai_research"
    history_dir: []const u8,           // ".bonsai_history"

    // Prompts (loaded from files at startup).
    system_prompt_path: []const u8,    // "programs/bonsai_system.md"
    tool_schemas_path: []const u8,     // "programs/bonsai_tools.json"

    // Tool dispatch table.
    tool_map: []const ToolMapping,

    // Limits (Rule 4 — hard caps on everything).
    max_experiments: u32 = 50,
    max_turns_per_experiment: u32 = 80,
    turn_warning_threshold: u32 = 70,
    max_messages: u32 = 512,
    max_tool_calls: u32 = 16,
    max_tool_output: u32 = 50_000,

    // API defaults.
    model: ?[]const u8 = null,         // Override via env or fall back to DEFAULT_MODEL.
    max_tokens: u32 = 32_768,
    thinking_budget: u32 = 16_384,

    // Callbacks for domain-specific behaviour.
    // These let profiles customise context building
    // without the core knowing about code files or
    // engineering rules.
    build_context_fn: ?*const fn (Allocator) []const u8 = null,
    on_experiment_end_fn: ?*const fn (Allocator, ExperimentResult) void = null,
};
```

## What `agent_core.zig` contains (~1,000 lines)

Only the generic control plane:

- **AgentConfig, ToolMapping, ToolShape** — profile definition types
- **run()** — generic experiment loop (load prompts, build context,
  call API, dispatch tools, manage turns, repeat)
- **dispatchTool()** — table-driven, routes via ToolMapping
- **callToolbox()** — spawn subprocess, capture stdout/stderr
- **dispatchStringArg()** — extract one JSON field, pass as CLI arg
- **dispatchJsonPayload()** — write `_tool_input.json`, pass `-f`
- **callApiWithRetry()** — HTTP with exponential backoff
- **callApi()** — single API call via curl
- **loadPromptFile()** — read `.md` or `.json` from `programs/`
- **appendExperiment()** — append to `{history_dir}/experiments.jsonl`
- **buildHistorySummary()** — load and format past experiments
- **buildToolbox()** — run `zig build` to compile toolbox binary
- **loadApiKey(), loadModel()** — env var helpers
- **executeTools(), executeSingleTool()** — map over tool call array
- **contextSizeBytes(), timestampMs(), nanosToMsStr()** — timing

Does NOT contain: tool implementations, prompts, path validation,
file I/O, domain-specific context, two-tier orchestration.

## What `mnist_agent.zig` contains (~100 lines)

Just configuration:

```zig
const core = @import("agent_core.zig");

const config = core.AgentConfig{
    .name = "mnist",
    .toolbox_path = "./zig-out/bin/mnist_research",
    .history_dir = ".mnist_history",
    .system_prompt_path = "programs/mnist_system.md",
    .tool_schemas_path = "programs/mnist_tools.json",
    .tool_map = &mnist_tools,
};

const mnist_tools = [_]core.ToolMapping{
    .{ .tool_name = "config_show",       .subcommand = "config-show",       .shape = .no_input },
    .{ .tool_name = "config_set",        .subcommand = "config-set",        .shape = .json_payload },
    .{ .tool_name = "config_backup",     .subcommand = "config-backup",     .shape = .no_input },
    .{ .tool_name = "config_restore",    .subcommand = "config-restore",    .shape = .no_input },
    .{ .tool_name = "train",             .subcommand = "train",             .shape = .no_input },
    .{ .tool_name = "benchmark_latest",  .subcommand = "benchmark-latest",  .shape = .no_input },
    .{ .tool_name = "benchmark_compare", .subcommand = "benchmark-compare", .shape = .no_input },
};

pub fn main() void {
    core.run(&config) catch |err| {
        // ...
    };
}
```

## What `bonsai_agent.zig` contains (~600 lines)

Configuration plus the two-tier orchestration loop:

- **config** — AgentConfig with bonsai tool map (20 entries)
- **main()** — if `LOCAL_LLM_MODEL` env is set, run two-tier; else `core.run()`
- **runTwoTierLoop()** — Opus strategist + Ollama executor loop (~300 lines)
- **loadCodeContext()** — load hot-path source files for strategist context
- **buildStrategyContext()** — format strategy prompt with history + code
- **printTwoTierSummary()** — run summary for two-tier mode

The two-tier loop calls `core.dispatchTool()` internally for tool
execution, so it still benefits from the generic dispatch table.
It only exists in bonsai because MNIST doesn't need it — MNIST
experiments are cheap enough to run entirely on Claude.

## What the toolbox binaries contain

### `mnist_research.zig` (~1,038 lines, renamed from `autoresearch.zig`)

No changes to tool implementations. Same subcommands:
`config-show`, `config-set`, `config-backup`, `config-restore`,
`train`, `benchmark-latest`, `benchmark-compare`, `help`, `clean`.

One change: `config-set` currently receives pre-parsed `K=V` args
from the agent's inline `executeConfigSet` handler. After the
refactor, it receives `-f input.json` and parses the settings
array itself using `std.json`.

### `bonsai_research.zig` (~2,000 lines, renamed from `engine_research.zig`)

Absorbs the 9 tool handlers currently inline in `engine_agent.zig`:

**Existing tools** (~1,000 lines, unchanged):

- `snapshot`, `snapshot-list`, `rollback`, `rollback-latest`
- `diff`, `check`, `test`
- `bench`, `bench-infer`, `bench-compare`
- `show`, `show-function`

**New subcommands** (moved from `engine_agent.zig`):

File I/O:

- `read-file <path>` — validate path against allow-list, read, print
- `write-file -f <input.json>` — parse JSON for path + content, validate, write
- `edit-file -f <input.json>` — parse JSON for path + old/new, find-and-replace
- `list-dir <path>` — validate path, list contents

Environment:

- `cwd` — print working directory
- `run-cmd -f <input.json>` — parse JSON for command string, execute via `/bin/sh`

Record-keeping:

- `commit -f <input.json>` — parse JSON for message, `git add` + `git commit`
- `add-summary -f <input.json>` — parse JSON for summary, append to summaries JSONL
- `history [count]` — return last N experiment records as JSON array

**What moves from `engine_agent.zig` to `bonsai_research.zig`:**

| Function               | Lines | Category    |
| ---------------------- | ----- | ----------- |
| `executeReadFile`      | 65    | File I/O    |
| `executeWriteFile`     | 85    | File I/O    |
| `executeEditFile`      | 180   | File I/O    |
| `executeListDirectory` | 75    | File I/O    |
| `isAllowedReadPath`    | 25    | File I/O    |
| `isAllowedWritePath`   | 15    | File I/O    |
| `executeRunCommand`    | 35    | Environment |
| `runShellCommand`      | 70    | Environment |
| `executeCwd`           | 10    | Environment |
| `executeCommit`        | 70    | Record      |
| `executeAddSummary`    | 50    | Record      |
| `executeHistory`       | 90    | Record      |
| `countSummaryLines`    | 30    | Record      |
| `appendSummary`        | 80    | Record      |
| **Total**              | 880   | —           |

After deduplication with `tools.zig` helpers and replacing hand-rolled
JSON parsing with `std.json`, these ~880 lines become ~630 lines in
`bonsai_research.zig`.

**Path validation moves here too.** The allow-lists
(`ALLOWED_READ_PREFIXES`, `ALLOWED_READ_FILES`, `ALLOWED_WRITE_FILES`)
belong in the toolbox, not the agent. The agent should not know or
care what paths are valid — that's the tool's job.

### Input passing convention: `-f input.json`

For tools that receive structured input from the LLM (content
payloads, multiline strings), the agent writes the raw JSON tool
input to `_tool_input.json` and passes `-f _tool_input.json` on
the command line. The toolbox binary reads, parses with
`std.json.parseFromSliceLeaky`, and deletes the temp file.

This avoids shell quoting nightmares. It is already the pattern used
by `executeCommit` with `_commit_msg.txt`.

For tools with a single string argument (`read-file`, `diff`,
`rollback`, etc.), the agent extracts the field value and passes it
as a positional CLI arg. No temp file needed.

## Implementation plan

Seven steps. Each step is independently shippable and leaves
all agents fully functional.

### Step 0: Prove the framework with mnist (the simple profile)

**Branch:** `framework-mnist`

Extract `agent_core.zig` from `agent.zig`. This is the least risky
starting point because `agent.zig` is simpler (768 lines, single-tier
only, 7 tools, 1 inline handler).

1. Create `agent_core.zig` with `AgentConfig`, `ToolMapping`,
   `run()`, `callToolbox()`, `dispatchTool()`, API calling, and
   history management — extracted from `agent.zig`.

2. Rename `agent.zig` → `mnist_agent.zig`. Reduce it to a config
   struct + `main()` that calls `core.run()`.

3. Rename `autoresearch.zig` → `mnist_research.zig`. Add a
   `config-set -f input.json` subcommand so the inline
   `executeConfigSet` handler can move out of the agent.

4. Update `build.zig` with new binary names.

5. Verify: `zig build && ./zig-out/bin/mnist_agent` runs correctly.

**Lines moved:** ~600 from `agent.zig` into `agent_core.zig`.
**Risk:** Low. `agent.zig` is well-understood and single-tier.
**Validation:** The mnist agent runs identically to before.

### Step 1: Externalise prompts to `programs/`

**Both profiles.** Move hardcoded string literals to files:

| Source constant             | Destination file                |
| --------------------------- | ------------------------------- |
| `agent.zig` SYSTEM_PROMPT   | `programs/mnist_system.md`      |
| `agent.zig` TOOL_SCHEMAS    | `programs/mnist_tools.json`     |
| `engine_agent.zig` SYSTEM   | `programs/bonsai_system.md`     |
| `engine_agent.zig` STRAT    | `programs/bonsai_strategist.md` |
| `engine_agent.zig` EXECUTOR | `programs/bonsai_executor.md`   |
| `engine_agent.zig` SCHEMAS  | `programs/bonsai_tools.json`    |

`agent_core.zig` gets `loadPromptFile(arena, path) []const u8`
(same pattern as the existing `loadEngineeringRules`). Short
fallback strings if files are missing.

**Lines removed:** ~400 from engine_agent, ~100 from agent.
**Risk:** Low. Read a file, use the string. Already proven pattern.
**Payoff:** Iterate on agent behaviour without recompiling.

### Step 2: Move file I/O tools to `bonsai_research`

New subcommands in `bonsai_research.zig`:

- `read-file <path>`
- `write-file -f <input.json>`
- `edit-file -f <input.json>`
- `list-dir <path>`

Move `isAllowedReadPath`, `isAllowedWritePath`, and the allow-list
constants. Use `tools.zig` helpers (`resolveToFs`, `readFile`,
`writeFile`) instead of the duplicates in `engine_agent.zig`.

Use `std.json.parseFromSliceLeaky` instead of `extractRequiredField`
and `unescapeJsonString`.

**Lines removed from engine_agent:** ~450
**Lines added to bonsai_research:** ~300

### Step 3: Move environment and shell tools to `bonsai_research`

New subcommands:

- `cwd`
- `run-cmd -f <input.json>`

**Lines removed from engine_agent:** ~115
**Lines added to bonsai_research:** ~80

### Step 4: Move record-keeping tools to `bonsai_research`

New subcommands:

- `commit -f <input.json>`
- `add-summary -f <input.json>`
- `history [count]`

Includes `appendSummary`, `countSummaryLines`, and the JSONL
file path constants.

**Lines removed from engine_agent:** ~345
**Lines added to bonsai_research:** ~250

### Step 5: Migrate `bonsai_agent.zig` onto `agent_core`

Rename `engine_agent.zig` → `bonsai_agent.zig`.

Replace the 130-line if-chain `dispatchTool` with the `bonsai_tools`
mapping table. Replace `callEngineResearch` with `core.callToolbox`.
Extract the generic loop into `core.run()`, keeping two-tier mode
as bonsai-specific code.

Delete dead helpers:

- `extractRequiredField`, `unescapeJsonString` — replaced by `std.json`
- `readFileContent`, `writeFileContent` — moved to toolbox
- `resolveToFs` — already in `tools.zig`
- `isAllowedReadPath`, `isAllowedWritePath` — moved to toolbox
- `ALLOWED_*` constants — moved to toolbox
- `HISTORY_PATH`, `SUMMARIES_PATH` — used only by record tools
- `formatHistoryLine`, `buildHistorySummary`, `buildSummariesSection`
  — moved to `agent_core` (generic) or toolbox (`history` subcommand)
- `getStrOr`, `getNumStr`, `parseTokenUsage`, `truncateAtSentence`

**Lines removed:** ~1,900 (engine_agent 3,929 → bonsai_agent ~600)
**Validation:** Both single-tier and two-tier modes work.

### Step 6: Rename files, update build.zig, clean up

Final renames, update `build.zig` binary names and build steps,
rename history directories, update any references in `CLAUDE.md`
or program files.

| build.zig step name | Binary name       |
| ------------------- | ----------------- |
| `mnist-agent`       | `mnist_agent`     |
| `mnist-research`    | `mnist_research`  |
| `bonsai-agent`      | `bonsai_agent`    |
| `bonsai-research`   | `bonsai_research` |

## Final line counts (estimated)

| File                    | Before | After  | Delta  |
| ----------------------- | ------ | ------ | ------ |
| `agent_core.zig`        | —      | ~1,000 | new    |
| `mnist_agent.zig`       | 768    | ~100   | -668   |
| `mnist_research.zig`    | 1,038  | ~1,100 | +62    |
| `bonsai_agent.zig`      | 3,929  | ~600   | -3,329 |
| `bonsai_research.zig`   | 1,339  | ~2,000 | +661   |
| `api_client.zig`        | 802    | 802    | 0      |
| `ollama_client.zig`     | 1,065  | 1,065  | 0      |
| `tools.zig`             | 630    | 630    | 0      |
| **Total Zig**           | 9,571  | ~7,297 | -2,274 |
| `programs/*.md + .json` | 1,286  | ~1,550 | +264   |

Net: ~2,000 lines of Zig deleted (deduplication + replacing hand-rolled
JSON parsing with `std.json`). ~264 lines of markdown/JSON added
(prompts that existed as Zig string literals, now externalised).

## Special cases and notes

### `appendExperiment` after bench

Currently `engine_agent.zig` appends bench results to
`experiments.jsonl` inline after a bench tool call succeeds.
After the refactor, two options:

1. `bonsai_research bench` appends its own results to JSONL (it
   already writes to `benchmarks/`).
2. `agent_core` has an `on_experiment_end_fn` callback that the
   bonsai profile uses to record results.

Option 1 is simpler. The toolbox already knows the result — why
send it back to the agent just to write it to disk?

### `buildHistorySummary` and `buildSummariesSection`

Currently called by the engine agent to build initial context.
After the refactor, the agent calls `bonsai_research history 10`
and `bonsai_research summaries` to get pre-formatted text. This
keeps JSONL parsing logic out of `agent_core`.

### Two-tier `loadCodeContext`

The strategist needs source file contents loaded at startup. This
stays in `bonsai_agent.zig` (it's context building, not tool
execution) or becomes `bonsai_research code-context` that dumps
all hot-path files.

### `std.json` over hand-rolled parsing

When `bonsai_research` reads `-f input.json`, use
`std.json.parseFromSliceLeaky` — not the `extractRequiredField`
pattern. The hand-rolled JSON field extractor breaks on nested
objects, numeric values, and fields with similar name prefixes.

### Tool schema validation at startup

`agent_core.run()` should validate at startup that every tool
name in the loaded `tools.json` schema has a corresponding entry
in the profile's `tool_map`. This catches wiring mistakes
immediately instead of at runtime when the LLM calls a tool
that doesn't dispatch anywhere.

### Adding a new profile

To add a new research profile (e.g. tokenizer optimisation):

1. Write `tokenizer_research.zig` — toolbox with domain tools.
2. Write `programs/tokenizer_system.md` — system prompt.
3. Write `programs/tokenizer_tools.json` — tool schemas.
4. Write `tokenizer_agent.zig` — ~50 lines of config + `core.run()`.
5. Add build targets in `build.zig`.

No changes to `agent_core.zig`. No changes to existing profiles.
