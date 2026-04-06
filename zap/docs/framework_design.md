# Zap Framework Design

> Making zap a general-purpose LLM-driven optimisation
> framework for any Zig library.

## Status quo

After removing the two-tier Ollama code, zap has a clean
two-process architecture:

```
Claude (API)
    ↕ HTTPS
Agent binary (brain)        ← agent_core.zig + profile
    ↕ child process spawn
Research binary (hands)     ← toolbox CLI
    ↕ child process spawn
Benchmark binary (ruler)    ← domain-specific measurement
```

Source layout:

```
zap/src/
├── agent_core.zig       2,271 lines   Generic experiment loop
├── api_client.zig         786 lines   Anthropic HTTP client
├── toolbox.zig          2,375 lines   Generic toolbox (shared)
├── tools.zig              695 lines   Shared CLI/file utilities
├── bonsai_agent.zig       468 lines   Bonsai agent profile
├── bonsai_research.zig     93 lines   Bonsai toolbox config
├── mnist_agent.zig        173 lines   MNIST agent profile
└── mnist_research.zig     879 lines   MNIST toolbox config + custom tools
```

The agent core is ~95% generic. The experiment loop,
table-driven tool dispatch, API retry logic, history
persistence, and prompt loading know nothing about neural
networks. Two profiles (mnist, bonsai) configure the core
via `AgentConfig` — a struct literal with tool mappings,
prompt paths, and limits.

The toolbox (`toolbox.zig`) provides all generic CLI
tools: sandboxed file I/O, snapshot/rollback,
build/test/bench dispatch, benchmark management, git
commit, and experiment history. Domain binaries are thin
config wrappers (~50–100 lines) with an optional
`custom_dispatch` callback for domain-specific tools.

A new domain plugs in by writing an agent profile
(~100 lines) and a toolbox config (~50 lines). The
remaining coupling points are minor and tracked below.

## Coupling points

### 1. `formatHistoryLine` in agent_core.zig — ✅ resolved

The history formatter hardcodes three MNIST field names:

```zig
const throughput = getNumStr(arena, obj, "throughput_images_per_sec");
const accuracy   = getNumStr(arena, obj, "final_test_accuracy_pct");
const train_ms   = getNumStr(arena, obj, "total_training_ms");
```

A bonsai benchmark produces `decode_tok_per_sec` and
`prefill_tok_per_sec`. A compression benchmark would
produce `compress_mb_per_sec` and `compression_ratio`.
The formatter cannot represent any of these.

### 2. `isBenchmarkFile` in tools.zig — ✅ resolved

Hardcodes `mnist_` and `inference_` as filename prefixes:

```zig
pub fn isBenchmarkFile(name: []const u8) bool {
    return (startsWith(name, "mnist_") or
        startsWith(name, "inference_")) and
        endsWith(name, ".json");
}
```

Bonsai benchmarks are named `bonsai_bench_*.json` and
silently fail this check.

### 3. `BenchResult` / `BenchConfig` in tools.zig — deferred

These structs mirror the MNIST training benchmark JSON
schema exactly. Bonsai produces a completely different
schema (tok/s, prefill, decode latency). There is no
shared type that covers both.

### 4. Toolbox binaries are 1000–2000 lines each — ✅ resolved

`bonsai_research.zig` (2,181 lines) and
`mnist_research.zig` (1,100 lines) are standalone CLI
programs with extensive domain logic. Most of the code
in `bonsai_research.zig` is generic: sandboxed file I/O,
snapshot/rollback, build/test/bench dispatch, git commit,
history append. A new domain would have to copy-paste
and adapt ~1,500 lines of boilerplate.

### 5. Path resolution assumes monorepo layout — ✅ resolved

`tools.resolveToFs` prepends `../` to reach the monorepo
root from `zap/`. This works for nnzap but breaks if zap
is used as a standalone tool against an external project.

## Design: three interfaces

The framework needs three standardised interfaces. Each
is independent — they can be implemented incrementally.

### Interface 1: History fields

**Problem.** The framework needs to show the LLM a
compact summary of prior experiments. Today this summary
hardcodes MNIST metrics.

**Solution.** Add a `history_fields` slice to
`AgentConfig`. Each entry names a JSON key to extract
from the benchmark JSONL and a display label:

```zig
pub const HistoryField = struct {
    /// JSON key in the benchmark output.
    json_key: []const u8,
    /// Short label for the history summary.
    label: []const u8,
};

pub const AgentConfig = struct {
    // ... existing fields ...

    /// Metrics to extract from benchmark JSON for the
    /// compact history. Walked by formatHistoryLine
    /// instead of hardcoded field names.
    history_fields: []const HistoryField = &.{},
};
```

The `formatHistoryLine` function becomes a loop:

```zig
fn formatHistoryLine(
    config: *const AgentConfig,
    arena: Allocator,
    index: u32,
    line: []const u8,
) ?[]const u8 {
    const obj = parseJsonObject(arena, line) orelse
        return null;
    const ts = getStrOr(obj, "timestamp_utc", "?");

    var buf: std.ArrayList(u8) = .empty;
    const prefix = std.fmt.allocPrint(
        arena,
        "  {d}. {s}",
        .{ index, ts },
    ) catch return null;
    buf.appendSlice(arena, prefix) catch return null;

    for (config.history_fields) |field| {
        const val = getNumStr(arena, obj, field.json_key);
        const part = std.fmt.allocPrint(
            arena,
            "  {s}={s}",
            .{ field.label, val },
        ) catch continue;
        buf.appendSlice(arena, part) catch continue;
    }
    buf.appendSlice(arena, "\n") catch return null;
    return buf.items;
}
```

MNIST profile:

```zig
.history_fields = &.{
    .{ .json_key = "throughput_images_per_sec", .label = "throughput" },
    .{ .json_key = "final_test_accuracy_pct", .label = "acc%" },
    .{ .json_key = "total_training_ms", .label = "time_ms" },
},
```

Bonsai profile:

```zig
.history_fields = &.{
    .{ .json_key = "decode_tok_per_sec", .label = "tok/s" },
    .{ .json_key = "prefill_tok_per_sec", .label = "prefill" },
    .{ .json_key = "decode_p99_us", .label = "p99_us" },
},
```

Compression library profile:

```zig
.history_fields = &.{
    .{ .json_key = "compress_mb_per_sec", .label = "MB/s" },
    .{ .json_key = "compression_ratio", .label = "ratio" },
},
```

**Effort.** Small. ~30 lines changed in `agent_core.zig`,
~3 lines added per profile. No breaking changes — the
default `&.{}` falls back to raw JSON dump.

### Interface 2: Toolbox config

**Problem.** Each domain needs a CLI toolbox binary, and
90% of the code is the same across domains: sandboxed
file I/O, snapshot/rollback, build/test dispatch,
benchmark comparison, git, history.

**Solution.** Extract a `toolbox.zig` module from the
common parts of `bonsai_research.zig`. Domain-specific
toolbox binaries become thin wrappers: a config struct
plus optional custom tool handlers.

```zig
pub const ToolboxConfig = struct {
    /// Display name for log messages.
    name: []const u8,

    /// Project root relative to the zap/ working
    /// directory. Used for build/test commands.
    project_root: []const u8,

    /// Files the agent may write (monorepo-relative).
    write_scope: []const []const u8,

    /// Directory prefixes the agent may read.
    read_scope: []const []const u8,

    /// Individual files the agent may read
    /// outside the prefix scope.
    read_files: []const []const u8,

    /// Files to include in snapshots.
    engine_files: []const []const u8,

    /// Build check command (compile-only validation).
    check_command: []const []const u8,

    /// Test command.
    test_command: []const []const u8,

    /// Primary benchmark command. Stdout is captured
    /// as JSON and persisted to experiments.jsonl.
    bench_command: []const []const u8,

    /// Additional benchmark commands (keyed by tool
    /// name, e.g. "bench-infer").
    extra_bench: []const ExtraBench,

    /// Benchmark directory (for compare/list/clean).
    bench_dir: []const u8,

    /// Benchmark filename prefix(es). Used by
    /// isBenchmarkFile to filter directory listings.
    bench_prefixes: []const []const u8,

    /// History directory (JSONL storage).
    history_dir: []const u8,

    /// Snapshot directory.
    snapshot_dir: []const u8,

    /// Optional extension: domain-specific tools
    /// that the generic toolbox doesn't handle.
    /// Called before returning "unknown tool".
    custom_dispatch: ?*const fn (
        Allocator,
        []const u8,
        []const []const u8,
    ) bool = null,
};

pub const ExtraBench = struct {
    tool_name: []const u8,
    command: []const []const u8,
};
```

A bonsai toolbox becomes:

```zig
const toolbox = @import("toolbox.zig");

const config = toolbox.ToolboxConfig{
    .name = "bonsai",
    .project_root = "../nn",
    .write_scope = &.{
        "nn/src/transformer.zig",
        "nn/src/network.zig",
        "nn/src/shaders/transformer.metal",
        "nn/src/shaders/compute.metal",
        // ...
    },
    .read_scope = &.{
        "nn/src/", "nn/examples/", "src/",
        "programs/", "benchmarks/",
    },
    .read_files = &.{
        "CLAUDE.md", "nn/build.zig",
    },
    .engine_files = &.{
        "nn/src/transformer.zig",
        "nn/src/network.zig",
        // ...
    },
    .check_command = &.{ "zig", "build" },
    .test_command = &.{ "zig", "build", "test" },
    .bench_command = &.{
        "zig", "build", "run-bonsai-bench",
    },
    .extra_bench = &.{.{
        .tool_name = "bench-infer",
        .command = &.{ "zig", "build", "run-infer" },
    }},
    .bench_dir = "benchmarks",
    .bench_prefixes = &.{ "bonsai_bench_" },
    .history_dir = ".bonsai_history",
    .snapshot_dir = ".engine_snapshots",
};

pub fn main() !void {
    toolbox.run(&config);
}
```

That is roughly 40 lines instead of 2,181.

**What the generic toolbox provides:**

| Tool              | Description                          |
| ----------------- | ------------------------------------ |
| `snapshot`        | Copy engine_files to snapshot dir.   |
| `snapshot-list`   | List available snapshots.            |
| `rollback`        | Restore engine_files from snapshot.  |
| `rollback-latest` | Restore most recent snapshot.        |
| `diff`            | Diff current state against snapshot. |
| `check`           | Run check_command in project_root.   |
| `test`            | Run test_command in project_root.    |
| `bench`           | Run bench_command, capture JSON.     |
| `bench-compare`   | Compare latest two benchmarks.       |
| `read-file`       | Sandboxed file read.                 |
| `write-file`      | Sandboxed file write.                |
| `edit-file`       | Sandboxed search-and-replace.        |
| `list-dir`        | Sandboxed directory listing.         |
| `show`            | Show source file with line numbers.  |
| `show-function`   | Extract a function body.             |
| `cwd`             | Print working directory.             |
| `run-cmd`         | Sandboxed shell command.             |
| `commit`          | Git add + commit.                    |
| `add-summary`     | Append to summaries.jsonl.           |
| `history`         | Read from experiments.jsonl.         |

The `custom_dispatch` callback handles domain-specific
tools. MNIST would use this for `config-show` and
`config-set` (the source-rewriting hyperparameter tools).

**Note on tool schemas.** Tool definitions are now
comptime `ToolDef` structs in the agent profile (see
open question 4, resolved). The toolbox config does
_not_ need a separate tool-schemas field — tools are
compiled in via the agent profile, and `toolSchemas()`
derives the JSON at compile time. The toolbox only
needs `custom_dispatch` to handle domain-specific
tool _execution_.

**Effort.** Medium. The refactor is mechanical — extract
functions from `bonsai_research.zig`, parameterise the
constants, expose a `run` entry point. Roughly 800 lines
of new shared code, then each domain shrinks to ~50–100
lines.

### Interface 3: Benchmark contract

**Problem.** The framework needs to know which metric to
optimise and whether a change helped. Today this
knowledge lives in prose prompts and the LLM's reasoning.
There is no programmatic way for the framework to make
keep/rollback decisions.

**Current state.** Three benchmark binaries produce three
different JSON schemas:

| Binary            | Key metrics                             |
| ----------------- | --------------------------------------- |
| `mnist.zig`       | throughput_images_per_sec, accuracy     |
| `inference_bench` | images_per_sec, p50/p99 latency         |
| `bonsai_bench`    | decode_tok_per_sec, prefill_tok_per_sec |

**Option A: Convention over configuration (recommended
now).** The framework does not parse benchmark JSON
structurally. It relies on:

1. `history_fields` to extract display values.
2. `persist_tools` to know which tool outputs are
   benchmark results.
3. The LLM to reason about whether results improved.

This is where we are today (modulo the hardcoded field
names). It works because the LLM is good at comparing
numbers and making keep/rollback decisions. The framework
just needs to show it the right numbers.

**Option B: Minimal envelope (recommended later).** When
we want automated keep/rollback without LLM reasoning,
benchmarks emit a standard header:

```json
{
  "timestamp_utc": "2025-07-15T14:30:00Z",
  "zap_envelope": {
    "primary": {
      "name": "decode_tok_per_sec",
      "value": 224.5,
      "direction": "higher_is_better"
    },
    "guards": [
      {
        "name": "test_accuracy_pct",
        "value": 97.8,
        "direction": "higher_is_better",
        "threshold": 97.0
      }
    ]
  },
  "decode_tok_per_sec": 224.5,
  "prefill_tok_per_sec": 1832.0,
  "decode_p99_us": 4500
}
```

The `zap_envelope.primary` field lets the toolbox do
automated regression detection: if `primary.value` is
worse than the previous run (respecting `direction`),
print a warning. The `guards` array lets it flag
regressions in secondary metrics (e.g. accuracy dropped
below 97%).

This requires changing every benchmark binary to emit
the envelope. Not hard, but not necessary until we have
a concrete need for automated decisions.

**Recommendation.** Option A now. The `history_fields`
change (Interface 1) is sufficient for multi-domain
support. Option B is a future enhancement for when the
agent loop wants to make autonomous keep/rollback
decisions without burning an API call.

## What a new domain looks like

Say someone wants zap to optimise their Zig HTTP server's
request throughput. They create four files:

### 1. Agent profile (~100 lines)

```zig
// zap/src/httpd_agent.zig

const std = @import("std");
const core = @import("agent_core.zig");
const api = core.api;

const httpd_tools = [_]core.ToolMapping{
    .{ .tool_name = "snapshot",       .subcommand = "snapshot",       .shape = .no_input },
    .{ .tool_name = "rollback_latest",.subcommand = "rollback-latest",.shape = .no_input },
    .{ .tool_name = "check",          .subcommand = "check",          .shape = .no_input },
    .{ .tool_name = "test",           .subcommand = "test",           .shape = .no_input },
    .{ .tool_name = "bench",          .subcommand = "bench",          .shape = .no_input },
    .{ .tool_name = "bench_compare",  .subcommand = "bench-compare",  .shape = .no_input },
    .{ .tool_name = "read_file",      .subcommand = "read-file",      .shape = .string_arg, .field = "path" },
    .{ .tool_name = "edit_file",      .subcommand = "edit-file",      .shape = .json_payload },
    .{ .tool_name = "add_summary",    .subcommand = "add-summary",    .shape = .json_payload },
};

const config = core.AgentConfig{
    .name = "httpd",
    .toolbox_path = "./zig-out/bin/httpd_research",
    .history_dir = ".httpd_history",
    .system_prompt_path = "programs/httpd_system.md",
    .tool_schemas_path = "programs/httpd_tools.json",
    .tool_map = &httpd_tools,
    .persist_tools = &.{"bench"},
    .history_fields = &.{
        .{ .json_key = "requests_per_sec", .label = "req/s" },
        .{ .json_key = "p99_latency_ms", .label = "p99_ms" },
        .{ .json_key = "error_rate_pct", .label = "err%" },
    },
};

pub fn main() void {
    core.run(&config) catch |err| {
        api.log("FATAL: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
}
```

### 2. Toolbox config (~50 lines)

```zig
// zap/src/httpd_research.zig

const toolbox = @import("toolbox.zig");

const config = toolbox.ToolboxConfig{
    .name = "httpd",
    .project_root = "../httpd",
    .write_scope = &.{
        "httpd/src/server.zig",
        "httpd/src/router.zig",
        "httpd/src/io_uring.zig",
    },
    .read_scope = &.{
        "httpd/src/", "httpd/tests/",
        "httpd/benchmarks/",
    },
    .read_files = &.{ "httpd/build.zig" },
    .engine_files = &.{
        "httpd/src/server.zig",
        "httpd/src/router.zig",
        "httpd/src/io_uring.zig",
    },
    .check_command = &.{ "zig", "build" },
    .test_command = &.{ "zig", "build", "test" },
    .bench_command = &.{ "zig", "build", "run-bench" },
    .extra_bench = &.{},
    .bench_dir = "benchmarks",
    .bench_prefixes = &.{ "httpd_" },
    .history_dir = ".httpd_history",
    .snapshot_dir = ".httpd_snapshots",
};

pub fn main() !void {
    toolbox.run(&config);
}
```

### 3. System prompt (~40 lines)

```markdown
<!-- zap/programs/httpd_system.md -->

You are an autonomous performance research agent
optimising a Zig HTTP server for maximum throughput
on Linux with io_uring.

## Goal

Maximise requests/sec on the wrk2 benchmark while
keeping p99 latency under 10ms and error rate at 0%.

## Protocol

1. snapshot — save current source.
2. Read source to understand the baseline.
3. Edit source to apply your optimisation.
4. check — must compile.
5. test — must pass.
6. bench — measure throughput.
7. Compare to previous. Keep if improved, rollback
   if regressed.
8. add_summary with your findings.
9. Repeat.
```

### 4. Tool schemas (~80 lines)

A JSON array in Anthropic tool format listing the
tools from the agent profile's tool mapping table.

### 5. Build step (~15 lines in build.zig)

```zig
const httpd_research = b.addExecutable(.{
    .name = "httpd_research",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/httpd_research.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "toolbox.zig", .module = toolbox_module },
        },
    }),
});
```

Total: ~300 lines of domain-specific code to enable
LLM-driven optimisation of an entirely new project.
Compare to the current bonsai setup which required
~3,400 lines (2,181 toolbox + 226 agent + 960 lines
of two-tier code that we just deleted).

## Implementation plan

### Step 1: history_fields — ✅ done

Added `HistoryField` struct and `history_fields` slice
to `AgentConfig`. Rewrote `formatHistoryLine` to walk
the slice instead of hardcoding field names. Both
profiles set their fields; empty slice falls back to
raw JSON.

Files changed:

- `agent_core.zig` — add type, modify formatter,
  thread config through `buildHistorySummary` →
  `formatHistoryLines` → `formatHistoryLine`.
- `bonsai_agent.zig` — add `.history_fields`.
- `mnist_agent.zig` — add `.history_fields`.

Actual size: ~40 lines changed.

### Step 2: bench_prefixes — ✅ done

Replaced `isBenchmarkFile` in `tools.zig` with a
function that accepts a `prefixes: []const []const u8`
parameter. Threaded through `listBenchmarkFiles`,
`findOneBenchmark`, `cleanBenchmarkFiles`, and
`toolBenchCompare`. Each toolbox binary defines its
own `bench_prefixes` constant.

Files changed:

- `tools.zig` — parameterise 5 functions.
- `bonsai_research.zig` — define + pass prefixes.
- `mnist_research.zig` — define + pass prefixes.

Actual size: ~30 lines changed.

### Step 3: toolbox.zig — ✅ done

Extracted generic toolbox from `bonsai_research.zig`
into `toolbox.zig`. Domain binaries are now thin
config wrappers.

`ToolboxConfig` struct with 15 fields: name,
project_root, write/read scopes, engine_files,
check/test/bench commands, extra_bench, bench_output_file
(stdout vs file mode), bench_dir, bench_prefixes,
history_dir, snapshot_dir, custom_dispatch callback.

23 standard tools + extra_bench loop + custom_dispatch
fallback. Consolidated `toolCheck`/`toolTest` into
shared `runAndReport` helper. Added `bench-list`,
`bench-latest`, `bench-clean` from MNIST. Unified bench
with `bench_output_file` flag for both stdout-capture
(bonsai) and file-based (MNIST) output modes.

MNIST subcommands standardised: `train` → `bench`,
`benchmark-compare` → `bench-compare`,
`benchmark-latest` → `bench-latest`.

Files changed:

- New `toolbox.zig` (2,375 lines).
- `bonsai_research.zig` — 2,140 → 93 lines (−96%).
- `mnist_research.zig` — 1,149 → 879 lines (−23%,
  keeps ~750 lines of config parsing).
- `mnist_agent.zig` — 3 subcommand renames.
- `build.zig` — add `toolbox_module`.

### Step 4: path resolution — ✅ done

`resolveToFs` now takes a `root` parameter instead of
hardcoding `"../"`. `ToolboxConfig` gained an `fs_root`
field (defaults to `".."`) so standalone deployments can
override the monorepo prefix.

Files changed:

- `tools.zig` — added `root: []const u8` parameter to
  `resolveToFs`; uses `"{s}/{s}"` format with `root`
  instead of hardcoded `"../"`.
- `toolbox.zig` — added `fs_root` field to
  `ToolboxConfig`; threaded `config.fs_root` through
  all 23 call sites; added `config` parameter to
  `applyEdit` helper.
- `agent_core.zig` — wrapper passes `".."` as root.
- `mnist_research.zig` — 6 call sites pass
  `config.fs_root`.

### Step 5 (future): benchmark envelope

When we have a concrete need for automated keep/rollback,
add the `zap_envelope` field to benchmark JSON output.
This is an additive change — benchmarks that don't emit
the envelope still work via the LLM's reasoning.

Files changed:

- `nn/src/benchmark.zig` — emit envelope in `save`.
- `nn/examples/bonsai_bench.zig` — emit envelope.
- `toolbox.zig` — parse envelope for regression warnings.

Not needed until we have a third domain or want to
reduce API calls by automating keep/rollback.

## Non-goals

- **Plugin system.** No dynamic loading, no runtime
  registration. Domains are compiled in. This matches
  the project's static-allocation philosophy.

- **Universal benchmark schema.** The framework does not
  need to understand every metric. It needs to display
  numbers and let the LLM reason about them.

- **Multi-language support.** Zap optimises Zig projects.
  The toolbox runs `zig build`. Supporting other build
  systems is out of scope.

- **Remote execution.** The agent runs locally, spawns
  local processes, reads local files. No SSH, no
  containers, no cloud.

## Open questions

1. **Should `toolbox.zig` be a Zig module or a standalone
   library? — resolved.** Module, imported by each
   toolbox binary via `build.zig`. The `run()` entry
   point handles arena setup and arg parsing; domain
   binaries call `toolbox.run(&config)` from a 3-line
   `main`. `custom_dispatch` provides the escape hatch
   for domain-specific tools without sacrificing the
   single-entry-point simplicity.

2. **How should `custom_dispatch` work? — resolved.**
   Function pointer `?*const fn (Allocator, []const u8,
[]const []const u8) anyerror!bool`. Returns true if
   handled. The toolbox dispatch tries standard tools
   first, then `extra_bench`, then `custom_dispatch`,
   then falls through to "unknown tool". MNIST uses
   this for `config-show`, `config-set`,
   `config-backup`, `config-restore`.

3. **Should the agent profile include the system prompt
   inline or keep it as a file path?** File paths are
   more flexible (edit without recompile) but add a
   runtime dependency. Current design: file paths.

4. **Tool schema / dispatch table sync — resolved.**
   Eliminated the separate JSON files entirely. Tools
   are now defined once as comptime `ToolDef` structs
   that carry both dispatch info (name, subcommand,
   shape) and API schema info (description, properties).
   `toolMappings()` and `toolSchemas()` derive the
   dispatch table and JSON string at compile time.
   Mismatch is structurally impossible — single source
   of truth, validated by the compiler.
