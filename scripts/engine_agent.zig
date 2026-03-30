//! nnzap engine agent — LLM-powered autonomous engine
//! optimisation runner.
//!
//! A tool-calling loop that talks to Claude via the
//! Anthropic API.  Claude reads source code, makes
//! targeted edits to Metal kernels and dispatch logic,
//! validates correctness, benchmarks throughput, and
//! keeps or reverts each change.
//!
//! The agent IS the runtime — like Claude Code, but ours.
//! It drives the engine_research toolbox and edits source
//! files directly.
//!
//! Setup:
//!   export ANTHROPIC_API_KEY=sk-ant-...
//!   zig build
//!   ./zig-out/bin/engine_agent
//!
//! Persists benchmark results to .engine_agent_history/
//! so context accumulates across runs.
//!
//! Output contract:
//!   stderr -> human-readable progress log
//!   .engine_agent_history/experiments.jsonl
//!     -> bench results (JSONL, one per line)
//!   .engine_agent_history/run_{ts}.json
//!     -> per-run conversation log

const std = @import("std");
const http = std.http;
const Uri = std.Uri;
const Allocator = std.mem.Allocator;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_TURNS: u32 = 200;
const MIN_TURNS: u32 = 10;
const MAX_MESSAGES: u32 = 512;
const MAX_TOOL_CALLS: u32 = 16;
const MAX_TOOL_OUTPUT: usize = 50_000;
const MAX_TOOL_ARGS: u32 = 16;
const MAX_API_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_HISTORY_SIZE: usize = 2 * 1024 * 1024;
const MAX_HISTORY_INJECT: usize = 30_000;
const MAX_FILE_SIZE: usize = 2 * 1024 * 1024;
const MAX_COMMAND_OUTPUT: usize = 1 * 1024 * 1024;
const COMMAND_TIMEOUT_NS: u64 = 120 * std.time.ns_per_s;
const TIMESTAMP_LEN: u32 = 20;

const MAX_RETRY_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY_MS: u64 = 2_000;
const RETRY_MAX_DELAY_MS: u64 = 30_000;

const API_URL: []const u8 =
    "https://api.anthropic.com/v1/messages";
const API_VERSION: []const u8 = "2023-06-01";
const DEFAULT_MODEL: []const u8 =
    "claude-sonnet-4-20250514";
const MAX_TOKENS_STR: []const u8 = "16384";

const TOOL_PATH: []const u8 =
    "./zig-out/bin/engine_research";
const HISTORY_DIR: []const u8 =
    ".engine_agent_history";
const HISTORY_PATH: []const u8 =
    ".engine_agent_history/experiments.jsonl";

// ============================================================
// Allowed paths for write_file / edit_file (safety guard)
// ============================================================

const ALLOWED_WRITE_FILES = [_][]const u8{
    "src/metal.zig",
    "src/network.zig",
    "src/shaders/compute.metal",
    "src/layout.zig",
    "src/main.zig",
};

// ============================================================
// Allowed path prefixes for read_file / list_directory
// ============================================================

const ALLOWED_READ_PREFIXES = [_][]const u8{
    "src/",
    "scripts/",
    "docs/",
    "data/",
    "benchmarks/",
    ".engine_agent_history/",
    ".engine_snapshots/",
};

const ALLOWED_READ_FILES = [_][]const u8{
    "README.md",
    "CLAUDE.md",
    "build.zig",
    "build.zig.zon",
};

// ============================================================
// Types
// ============================================================

const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    input_json: []const u8,
};

const ToolOutput = struct {
    stdout: []const u8,
    success: bool,
};

const ToolResult = struct {
    tool_use_id: []const u8,
    content: []const u8,
    is_error: bool,
};

const ApiResponse = struct {
    success: bool,
    stop_reason: []const u8,
    content_json: []const u8,
    text: []const u8,
    tool_calls: []const ToolCall,
    error_message: []const u8,
    retryable: bool,
};

const ParsedContent = struct {
    text: []const u8,
    tool_calls: []const ToolCall,
};

// ============================================================
// System prompt
//
// Seeds Claude with role, goal, protocol, constraints,
// and optimisation strategy.  Multiline string literal.
// ============================================================

const SYSTEM_PROMPT =
    \\You are an autonomous systems-performance research
    \\agent optimising nnzap's core engine. nnzap is a
    \\Zig + Metal GPU-accelerated neural network library
    \\for Apple Silicon with zero-copy unified memory.
    \\
    \\Goal: maximise throughput_images_per_sec on MNIST
    \\without regressing test accuracy below 97.0%.
    \\
    \\You edit source files directly. This is NOT
    \\hyperparameter tuning -- you change Metal kernels,
    \\dispatch strategies, buffer layouts, and pipeline
    \\architecture.
    \\
    \\## Files
    \\
    \\- src/metal.zig -- Metal device, buffers, pipelines,
    \\  dispatch (1D/2D), command buffer management.
    \\- src/network.zig -- Forward/backward pass encoding,
    \\  loss functions, optimizer updates.
    \\- src/shaders/compute.metal -- GPU kernels (MSL):
    \\  matmul, activations, bias, SGD/Adam, softmax+CE.
    \\- src/layout.zig -- Comptime network layout: buffer
    \\  sizes, offsets, activation sizes.
    \\- src/main.zig -- Training loop, batch iteration,
    \\  evaluation, benchmark recording.
    \\
    \\## Tools
    \\
    \\You have these tools available:
    \\
    \\Engine research (safety + measurement):
    \\  snapshot, snapshot_list, rollback, rollback_latest,
    \\  diff, check, test, bench, bench_compare,
    \\  show, show_function
    \\
    \\File I/O (locked to project directory):
    \\  read_file, write_file, edit_file, list_directory
    \\
    \\Shell (locked to project directory, 120s timeout):
    \\  run_command -- execute any shell command, e.g.
    \\  grep, zig build, wc, head, etc.
    \\
    \\Prefer edit_file over write_file when making small
    \\changes. edit_file does targeted find-and-replace,
    \\which is faster and uses less context than rewriting
    \\the entire file.
    \\
    \\Use run_command to grep for patterns, check file
    \\sizes, run ad-hoc build commands, etc. All commands
    \\execute in the project root with a 120s timeout.
    \\
    \\## Protocol
    \\
    \\For each experiment:
    \\1. snapshot -- safety net before any edits.
    \\2. Read the target code with show_function,
    \\   read_file, or run_command (e.g. grep).
    \\3. Edit the source with edit_file (targeted
    \\   find-and-replace) or write_file (full rewrite
    \\   for large changes). Read before you write.
    \\4. check -- compile validation (~2s). If it fails,
    \\   read the error, fix or rollback_latest.
    \\5. test -- numerical correctness. If it fails,
    \\   rollback_latest.
    \\6. bench -- full MNIST training benchmark.
    \\7. Evaluate: see decision rules below.
    \\8. Keep (new baseline) or rollback_latest.
    \\9. Pick next optimisation. Repeat from step 1.
    \\
    \\## Decision rules
    \\
    \\- Throughput up >= 5%: KEEP.
    \\- Throughput +/- 5%, accuracy up: KEEP.
    \\- Throughput +/- 5%, accuracy same: KEEP if the
    \\  code is cleaner or enables future work.
    \\- Throughput down > 5%: ROLLBACK.
    \\- Accuracy below 97.0%: ROLLBACK.
    \\- Compile or test failure: ROLLBACK immediately.
    \\
    \\## Constraints
    \\
    \\- check must pass before test or bench.
    \\- test must pass. NEVER modify test expectations.
    \\- Accuracy must stay >= 97.0% (baseline ~97.8%).
    \\- ONE optimisation per experiment. Isolate variables.
    \\- Read the full function before modifying it.
    \\- When adding Metal kernels, wire the pipeline in
    \\  metal.zig too. [[buffer(N)]] must match setBuffer.
    \\- Engineering rules: 70-line functions, >= 2
    \\  assertions per function, 100-column limit,
    \\  snake_case naming, comments explain WHY.
    \\
    \\## Current baseline
    \\
    \\- Naive matmul: 1 thread per output element,
    \\  no tiling, no shared memory.
    \\- One command buffer + one encoder per batch.
    \\- Synchronous commitAndWait after every batch.
    \\- Hardcoded 16x16 threadgroup for 2D kernels.
    \\- No kernel fusion (separate matmul, bias,
    \\  activation dispatches).
    \\- ~85k images/sec throughput.
    \\
    \\## Optimisation phases (priority order)
    \\
    \\Phase 1 -- Dispatch tuning (low-risk, high-reward):
    \\  Tune threadgroup sizes (8x8, 32x8, 32x32).
    \\  Use dispatchThreadgroups over dispatchThreads.
    \\  Batch multiple steps per command buffer.
    \\
    \\Phase 2 -- Kernel optimisation:
    \\  Tiled matmul with threadgroup memory (16x16).
    \\  Fuse bias_add into matmul kernel.
    \\  Fuse activation into matmul+bias.
    \\  Vectorised float4 loads in elementwise kernels.
    \\
    \\Phase 3 -- Pipeline:
    \\  Double-buffered command buffers.
    \\  Async commit instead of commitAndWait.
    \\  Minimise memory barriers.
    \\
    \\Phase 4 -- Memory:
    \\  Private storage for GPU-only buffers.
    \\  Write-combined cache for input buffers.
    \\  Buffer reuse across layers.
    \\
    \\Phase 5 -- Advanced:
    \\  SIMD group reductions.
    \\  Half-precision activations/gradients.
    \\  Fused backward pass kernels.
    \\
    \\## Metal kernel editing rules
    \\
    \\- [[buffer(N)]] must match setBuffer calls.
    \\- Bounds check: if (gid.x >= width) return;
    \\- threadgroup_barrier after threadgroup writes.
    \\- Avoid divergent branches in a SIMD group.
    \\- Prefer adding new kernels (e.g. matmul_tiled)
    \\  alongside existing ones for safe comparison.
    \\
    \\## Recovery
    \\
    \\- check fails: fix or rollback_latest.
    \\- test fails: rollback_latest.
    \\- bench crashes: rollback_latest.
    \\- bench regresses: record result, rollback_latest.
    \\
    \\## NEVER STOP
    \\
    \\Do NOT ask the human if you should continue.
    \\The human may be asleep. You are autonomous.
    \\Run experiments continuously. When ideas from
    \\the phase list are exhausted, combine successful
    \\optimisations or try different parameters.
    \\
    \\Each experiment takes ~1-3 minutes. Target 20-60
    \\experiments per hour. A sleeping human gets 8
    \\hours of sleep -- that is 160-480 experiments.
    \\Use them wisely. Phase 1 and 2 deliver the
    \\largest gains -- spend most time there.
    \\
    \\The first user message includes benchmark history
    \\from previous runs. Use it to avoid repeating work.
;

// ============================================================
// Tool schemas
//
// JSON array of tool definitions for the Anthropic API.
// Each tool maps to an engine_research CLI command,
// a direct file I/O operation, or a shell command.
// ============================================================

const TOOL_SCHEMAS =
    \\[
    \\  {"name":"snapshot",
    \\   "description":"Save engine source files as a restore point. Always call before editing.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"snapshot_list",
    \\   "description":"List all saved snapshots with timestamps.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"rollback",
    \\   "description":"Restore engine files from a specific snapshot.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"id":{"type":"string",
    \\       "description":"Snapshot ID from snapshot or snapshot_list"}},
    \\     "required":["id"]}},
    \\  {"name":"rollback_latest",
    \\   "description":"Restore from the most recent snapshot. Use when check/test/bench fails.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"diff",
    \\   "description":"Show source file changes since a snapshot.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"id":{"type":"string",
    \\       "description":"Snapshot ID to diff against"}},
    \\     "required":["id"]}},
    \\  {"name":"check",
    \\   "description":"Compile-only validation (~2s). Run after every edit. STOP if this fails.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"test",
    \\   "description":"Run full test suite for numerical correctness. STOP if this fails.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"bench",
    \\   "description":"Full MNIST training benchmark (~10s). Returns JSON with throughput_images_per_sec and final_test_accuracy_pct.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"bench_compare",
    \\   "description":"Compare all benchmark results side by side.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"show",
    \\   "description":"View an engine source file as structured JSON with line numbers.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"file":{"type":"string",
    \\       "description":"Source file path, e.g. src/metal.zig"}},
    \\     "required":["file"]}},
    \\  {"name":"show_function",
    \\   "description":"Extract a specific function from a source file with line numbers.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{
    \\       "file":{"type":"string",
    \\         "description":"Source file path"},
    \\       "function_name":{"type":"string",
    \\         "description":"Function name to extract"}},
    \\     "required":["file","function_name"]}},
    \\  {"name":"read_file",
    \\   "description":"Read raw contents of a project file. Locked to project directory.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"path":{"type":"string",
    \\       "description":"File path relative to project root, e.g. src/metal.zig"}},
    \\     "required":["path"]}},
    \\  {"name":"write_file",
    \\   "description":"Replace entire contents of an engine source file. Use edit_file for small changes instead.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{
    \\       "path":{"type":"string",
    \\         "description":"Engine file path, e.g. src/metal.zig"},
    \\       "content":{"type":"string",
    \\         "description":"Complete new file contents"}},
    \\     "required":["path","content"]}},
    \\  {"name":"edit_file",
    \\   "description":"Targeted find-and-replace in an engine source file. More efficient than write_file for small edits. The old_content must match exactly.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{
    \\       "path":{"type":"string",
    \\         "description":"Engine file path, e.g. src/metal.zig"},
    \\       "old_content":{"type":"string",
    \\         "description":"Exact text to find (must match exactly)"},
    \\       "new_content":{"type":"string",
    \\         "description":"Replacement text"}},
    \\     "required":["path","old_content","new_content"]}},
    \\  {"name":"list_directory",
    \\   "description":"List files and subdirectories in a project directory.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"path":{"type":"string",
    \\       "description":"Directory path relative to project root, e.g. src/ or src/shaders"}},
    \\     "required":["path"]}},
    \\  {"name":"run_command",
    \\   "description":"Execute a shell command in the project root directory. 120s timeout. Use for grep, wc, head, tail, find, cat, etc. Do NOT run long-lived processes (servers, watchers).",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"command":{"type":"string",
    \\       "description":"Shell command to execute, e.g. grep -rn 'matmul' src/"}},
    \\     "required":["command"]}}
    \\]
;

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    mainInner() catch |err| {
        log(
            "\nFATAL: agent crashed: {s}\n",
            .{@errorName(err)},
        );
        log(
            "Hint: if the agent edited source files " ++
                "before crashing, run:\n" ++
                "  ./zig-out/bin/engine_research " ++
                "rollback-latest\n",
            .{},
        );
        std.process.exit(1);
    };
}

fn mainInner() !void {
    var arena_state = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const run_start = timestampMs();

    printHeader();
    ensureHistoryDir();

    const api_key = loadApiKey() orelse {
        fatal(
            "Set ANTHROPIC_API_KEY env var.\n" ++
                "  export ANTHROPIC_API_KEY=sk-ant-...\n",
        );
        unreachable;
    };
    const model = loadModel();
    log("Model: {s}\n", .{model});
    log(
        "Limits: min={d} max={d} turns\n",
        .{ MIN_TURNS, MAX_TURNS },
    );

    if (!buildToolbox(arena)) {
        fatal("zig build failed.\n");
        unreachable;
    }

    const history = loadHistoryContent(arena);
    const first_msg = buildInitialMessage(
        arena,
        history,
    );

    var messages: [MAX_MESSAGES][]const u8 = undefined;
    var count: u32 = 0;
    messages[0] = first_msg;
    count = 1;

    // Counters for the run summary.
    var bench_count: u32 = 0;
    var tool_call_count: u32 = 0;
    var api_error_count: u32 = 0;
    var total_api_ms: i64 = 0;
    var total_tool_ms: i64 = 0;

    var turn: u32 = 0;
    while (turn < MAX_TURNS) : (turn += 1) {
        log("\n--- Turn {d} ---\n", .{turn + 1});

        // Log context size before API call.
        const ctx_bytes = contextSizeBytes(
            messages[0..count],
        );
        log(
            "  Context: {d} messages, {d} KB\n",
            .{ count, ctx_bytes / 1024 },
        );

        const api_start = timestampMs();
        const resp = callApiWithRetry(
            arena,
            api_key,
            model,
            messages[0..count],
        );
        const api_elapsed = timestampMs() - api_start;
        total_api_ms += api_elapsed;

        if (!resp.success) {
            api_error_count += 1;
            log(
                "  API error (unrecoverable) " ++
                    "after {s}: {s}\n",
                .{
                    nanosToMsStr(
                        arena,
                        api_elapsed * 1_000_000,
                    ),
                    resp.error_message,
                },
            );
            break;
        }

        log(
            "  API response: {s} " ++
                "({d} tool calls, {d} KB)\n",
            .{
                nanosToMsStr(
                    arena,
                    api_elapsed * 1_000_000,
                ),
                resp.tool_calls.len,
                resp.content_json.len / 1024,
            },
        );

        if (resp.text.len > 0) logClaudeText(resp.text);

        messages[count] = buildAssistantMsg(
            arena,
            resp.content_json,
        );
        count += 1;

        // Check if Claude wants to stop.
        const is_tool_use = eql(
            resp.stop_reason,
            "tool_use",
        );
        if (!is_tool_use or resp.tool_calls.len == 0) {
            if (turn + 1 < MIN_TURNS) {
                // Force Claude to keep going.
                log(
                    "  (turn {d}/{d} minimum " ++
                        "-- nudging to continue)\n",
                    .{ turn + 1, MIN_TURNS },
                );
                messages[count] = wrapUserTextMessage(
                    arena,
                    "You have not reached the " ++
                        "minimum experiment count " ++
                        "yet. Keep optimising. " ++
                        "Try the next phase or " ++
                        "a different approach.",
                );
                count += 1;
            } else {
                log("  Claude chose to stop.\n", .{});
                break;
            }
        } else {
            // Execute tool calls and send results.
            const tools_start = timestampMs();
            const results = executeTools(
                arena,
                resp.tool_calls,
            );
            const tools_elapsed = timestampMs() -
                tools_start;
            total_tool_ms += tools_elapsed;
            tool_call_count += @intCast(
                resp.tool_calls.len,
            );

            // Count bench calls for the summary.
            for (resp.tool_calls) |call| {
                if (eql(call.name, "bench")) {
                    bench_count += 1;
                }
            }

            // Log tool results summary.
            var err_count: u32 = 0;
            var result_bytes: usize = 0;
            for (results) |r| {
                result_bytes += r.content.len;
                if (r.is_error) err_count += 1;
            }
            log(
                "  Tools done: {d} calls, " ++
                    "{d} errors, {d} KB result, " ++
                    "{s}\n",
                .{
                    resp.tool_calls.len,
                    err_count,
                    result_bytes / 1024,
                    nanosToMsStr(
                        arena,
                        tools_elapsed * 1_000_000,
                    ),
                },
            );

            messages[count] = buildToolResultsMsg(
                arena,
                results,
            );
            count += 1;
        }

        if (count + 2 >= MAX_MESSAGES) {
            log("Message limit reached.\n", .{});
            break;
        }
    }

    // ── Run summary ─────────────────────────────────────
    const run_elapsed = timestampMs() - run_start;
    log(
        "\n" ++
            "==============================================\n" ++
            "  Run summary\n" ++
            "==============================================\n" ++
            "  Turns:       {d}\n" ++
            "  Tool calls:  {d}\n" ++
            "  Benchmarks:  {d}\n" ++
            "  API errors:  {d}\n" ++
            "  API time:    {s}\n" ++
            "  Tool time:   {s}\n" ++
            "  Total time:  {s}\n" ++
            "  Context:     {d} messages, {d} KB\n" ++
            "==============================================\n",
        .{
            turn + 1,
            tool_call_count,
            bench_count,
            api_error_count,
            nanosToMsStr(arena, total_api_ms * 1_000_000),
            nanosToMsStr(arena, total_tool_ms * 1_000_000),
            nanosToMsStr(arena, run_elapsed * 1_000_000),
            count,
            contextSizeBytes(messages[0..count]) / 1024,
        },
    );

    saveRunLog(arena, messages[0..count]);
}

/// Sum the byte lengths of all messages in the array.
fn contextSizeBytes(
    messages: []const []const u8,
) usize {
    var total: usize = 0;
    for (messages) |msg| {
        total += msg.len;
    }
    return total;
}

// ============================================================
// Setup
// ============================================================

fn loadApiKey() ?[]const u8 {
    const val = std.posix.getenv("ANTHROPIC_API_KEY");
    if (val) |key| {
        std.debug.assert(key.len > 0);
        return key;
    }
    return null;
}

fn loadModel() []const u8 {
    const val = std.posix.getenv("ANTHROPIC_MODEL");
    if (val) |model| {
        std.debug.assert(model.len > 0);
        return model;
    }
    return DEFAULT_MODEL;
}

/// Build the toolbox with `zig build`.
fn buildToolbox(arena: Allocator) bool {
    const build_start = timestampMs();
    log("Building toolbox...\n", .{});

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "zig", "build" },
        .max_output_bytes = MAX_OUTPUT_BYTES,
    }) catch {
        log("  spawn failed.\n", .{});
        return false;
    };

    const build_ms = timestampMs() - build_start;
    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (!ok) {
        log("Build failed:\n{s}\n", .{result.stderr});
    } else {
        log(
            "  done ({s}).\n",
            .{nanosToMsStr(arena, build_ms * 1_000_000)},
        );
    }
    return ok;
}

// ============================================================
// API calling with retry
// ============================================================

/// Call the API with exponential backoff on transient errors.
fn callApiWithRetry(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    messages: []const []const u8,
) ApiResponse {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(messages.len > 0);

    var attempt: u32 = 0;
    while (attempt < MAX_RETRY_ATTEMPTS) : (attempt += 1) {
        if (attempt > 0) {
            const shift: u6 = @intCast(attempt - 1);
            const base: u64 = RETRY_BASE_DELAY_MS;
            const cap: u64 = RETRY_MAX_DELAY_MS;
            const delay_ms: u64 = @min(
                base << shift,
                cap,
            );
            log(
                "  Retrying in {d}ms " ++
                    "(attempt {d}/{d})...\n",
                .{
                    delay_ms,
                    attempt + 1,
                    MAX_RETRY_ATTEMPTS,
                },
            );
            const ns = delay_ms * 1_000_000;
            std.Thread.sleep(ns);
        }

        const resp = callApi(
            arena,
            api_key,
            model,
            messages,
        );

        if (resp.success) return resp;

        if (!resp.retryable) {
            log(
                "  API error (not retryable): {s}\n",
                .{resp.error_message},
            );
            return resp;
        }

        log(
            "  API error (retryable): {s}\n",
            .{resp.error_message},
        );
    }

    return errResp(
        "max retries exhausted",
        false,
    );
}

/// Call the Anthropic Messages API using std.http.
fn callApi(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    messages: []const []const u8,
) ApiResponse {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(messages.len > 0);

    const body = buildRequestJson(
        arena,
        model,
        messages,
    ) catch return errResp(
        "request build failed",
        false,
    );

    // Save last request for debugging.
    writeFile(
        HISTORY_DIR ++ "/_request.json",
        body,
    ) catch {};
    log(
        "  API request: {d} KB\n",
        .{body.len / 1024},
    );

    const uri = Uri.parse(API_URL) catch {
        return errResp(
            "failed to parse API URL",
            false,
        );
    };

    var client = http.Client{ .allocator = arena };
    defer client.deinit();

    var req = client.request(.POST, uri, .{
        .headers = .{
            .accept_encoding = .{
                .override = "identity",
            },
        },
        .extra_headers = &.{
            .{
                .name = "Content-Type",
                .value = "application/json",
            },
            .{
                .name = "x-api-key",
                .value = api_key,
            },
            .{
                .name = "anthropic-version",
                .value = API_VERSION,
            },
        },
    }) catch {
        return errResp(
            "failed to open HTTP request",
            true,
        );
    };
    defer req.deinit();

    req.transfer_encoding = .{
        .content_length = body.len,
    };
    req.sendBodyComplete(body) catch {
        return errResp(
            "failed to send request body",
            true,
        );
    };

    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(
        &redirect_buf,
    ) catch {
        return errResp(
            "failed to receive response",
            true,
        );
    };

    const status = response.head.status;

    // Read the body regardless of status — it may
    // contain a useful error message from the API.
    var transfer_buf: [64]u8 = undefined;
    var reader = response.reader(&transfer_buf);
    const response_data = reader.allocRemaining(
        arena,
        std.Io.Limit.limited(MAX_API_RESPONSE),
    ) catch {
        return errResp(
            "failed to read response body",
            true,
        );
    };

    if (status != .ok) {
        return handleNonOkStatus(
            arena,
            status,
            response_data,
        );
    }

    if (response_data.len == 0) {
        return errResp("empty API response", true);
    }

    log(
        "  API response body: {d} KB\n",
        .{response_data.len / 1024},
    );

    return parseApiResponse(arena, response_data);
}

/// Produce a descriptive error for non-200 API responses.
fn handleNonOkStatus(
    arena: Allocator,
    status: std.http.Status,
    body: []const u8,
) ApiResponse {
    const code: u16 = @intFromEnum(status);

    // Extract the API error message if present.
    const api_msg = extractJsonString(
        body,
        "\"message\":\"",
    ) orelse extractJsonString(
        body,
        "\"message\": \"",
    ) orelse "";

    const detail = if (api_msg.len > 0)
        api_msg
    else
        truncate(body, 500);

    const msg = std.fmt.allocPrint(
        arena,
        "HTTP {d}: {s}",
        .{ code, detail },
    ) catch "HTTP error (unknown)";

    log("  API HTTP {d}: {s}\n", .{ code, detail });

    // 429 = rate limited, 529 = overloaded,
    // 500/502/503 = server errors — all retryable.
    const retryable = (code == 429 or
        code == 529 or
        code >= 500);

    return errResp(msg, retryable);
}

/// Build the full JSON request body.
fn buildRequestJson(
    arena: Allocator,
    model: []const u8,
    messages: []const []const u8,
) ![]u8 {
    std.debug.assert(model.len > 0);
    std.debug.assert(messages.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\"model\":\"");
    try buf.appendSlice(arena, model);
    try buf.appendSlice(arena, "\",\"max_tokens\":");
    try buf.appendSlice(arena, MAX_TOKENS_STR);
    try buf.appendSlice(arena, ",\"system\":");
    try appendJsonString(arena, &buf, SYSTEM_PROMPT);
    try buf.appendSlice(arena, ",\"tools\":");
    try buf.appendSlice(arena, TOOL_SCHEMAS);
    try buf.appendSlice(arena, ",\"messages\":[");

    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(arena, ',');
        try buf.appendSlice(arena, msg);
    }

    try buf.appendSlice(arena, "]}");
    return buf.items;
}

// ============================================================
// Response parsing
// ============================================================

/// Parse the raw API response JSON.
fn parseApiResponse(
    arena: Allocator,
    raw: []const u8,
) ApiResponse {
    std.debug.assert(raw.len > 0);

    // Detect API error responses.
    if (indexOf(raw, "\"type\":\"error\"") != null or
        indexOf(raw, "\"type\": \"error\"") != null)
    {
        const msg = extractJsonString(
            raw,
            "\"message\":\"",
        ) orelse "unknown API error";
        return errResp(msg, false);
    }

    const stop = extractJsonString(
        raw,
        "\"stop_reason\":\"",
    ) orelse extractJsonString(
        raw,
        "\"stop_reason\": \"",
    ) orelse "unknown";

    const content = extractRawArray(
        raw,
        "\"content\"",
    ) orelse "[]";

    const parsed = parseContentBlocks(
        arena,
        content,
    );

    return .{
        .success = true,
        .stop_reason = stop,
        .content_json = content,
        .text = parsed.text,
        .tool_calls = parsed.tool_calls,
        .error_message = "",
        .retryable = false,
    };
}

/// Split the content array into text and tool_use blocks.
fn parseContentBlocks(
    arena: Allocator,
    content_json: []const u8,
) ParsedContent {
    const empty: ParsedContent = .{
        .text = "",
        .tool_calls = &.{},
    };

    const blocks = splitTopLevelObjects(
        arena,
        content_json,
    ) catch return empty;

    var text_buf: std.ArrayList(u8) = .empty;
    var calls: [MAX_TOOL_CALLS]ToolCall = undefined;
    var call_count: u32 = 0;

    for (blocks) |block| {
        if (containsField(block, "\"type\"", "text")) {
            const t = extractJsonString(
                block,
                "\"text\":\"",
            ) orelse extractJsonString(
                block,
                "\"text\": \"",
            ) orelse continue;
            if (text_buf.items.len > 0) {
                text_buf.append(arena, '\n') catch {};
            }
            text_buf.appendSlice(arena, t) catch {};
        }

        if (containsField(
            block,
            "\"type\"",
            "tool_use",
        )) {
            if (call_count >= MAX_TOOL_CALLS) continue;
            const id = extractJsonString(
                block,
                "\"id\":\"",
            ) orelse extractJsonString(
                block,
                "\"id\": \"",
            ) orelse continue;
            const name = extractJsonString(
                block,
                "\"name\":\"",
            ) orelse extractJsonString(
                block,
                "\"name\": \"",
            ) orelse continue;
            const input = extractRawObject(
                block,
                "\"input\"",
            ) orelse "{}";

            calls[call_count] = .{
                .id = id,
                .name = name,
                .input_json = input,
            };
            call_count += 1;
        }
    }

    const result_calls = arena.alloc(
        ToolCall,
        call_count,
    ) catch return .{
        .text = text_buf.items,
        .tool_calls = &.{},
    };
    @memcpy(result_calls, calls[0..call_count]);

    return .{
        .text = text_buf.items,
        .tool_calls = result_calls,
    };
}

// ============================================================
// Tool execution
// ============================================================

/// Execute all tool calls from one API response.
fn executeTools(
    arena: Allocator,
    calls: []const ToolCall,
) []const ToolResult {
    std.debug.assert(calls.len > 0);
    std.debug.assert(calls.len <= MAX_TOOL_CALLS);

    const results = arena.alloc(
        ToolResult,
        calls.len,
    ) catch return &.{};

    for (calls, 0..) |call, i| {
        results[i] = executeSingleTool(arena, call);
    }
    return results;
}

/// Execute one tool call and return the result.
fn executeSingleTool(
    arena: Allocator,
    call: ToolCall,
) ToolResult {
    std.debug.assert(call.id.len > 0);
    std.debug.assert(call.name.len > 0);

    logToolCall(call.name, call.input_json);

    const tool_start = timestampMs();
    const output = dispatchTool(arena, call);
    const tool_elapsed = timestampMs() - tool_start;

    const status = if (output.success) "ok" else "ERR";
    log(
        "    -> {s} ({s}, {d} bytes, {s})\n",
        .{
            call.name,
            status,
            output.stdout.len,
            nanosToMsStr(
                arena,
                tool_elapsed * 1_000_000,
            ),
        },
    );

    // Persist bench results to the history log.
    if (eql(call.name, "bench") and output.success) {
        appendExperiment(arena, output.stdout);
        log("    -> bench result persisted\n", .{});
    }

    return .{
        .tool_use_id = call.id,
        .content = truncate(
            output.stdout,
            MAX_TOOL_OUTPUT,
        ),
        .is_error = !output.success,
    };
}

/// Route a tool call to the correct handler.
fn dispatchTool(
    arena: Allocator,
    call: ToolCall,
) ToolOutput {
    std.debug.assert(call.name.len > 0);

    // Engine research toolbox commands.
    if (eql(call.name, "snapshot")) {
        return callEngineResearch(
            arena,
            &.{"snapshot"},
        );
    }
    if (eql(call.name, "snapshot_list")) {
        return callEngineResearch(
            arena,
            &.{"snapshot-list"},
        );
    }
    if (eql(call.name, "rollback")) {
        return executeRollback(arena, call.input_json);
    }
    if (eql(call.name, "rollback_latest")) {
        return callEngineResearch(
            arena,
            &.{"rollback-latest"},
        );
    }
    if (eql(call.name, "diff")) {
        return executeDiff(arena, call.input_json);
    }
    if (eql(call.name, "check")) {
        return callEngineResearch(
            arena,
            &.{"check"},
        );
    }
    if (eql(call.name, "test")) {
        return callEngineResearch(
            arena,
            &.{"test"},
        );
    }
    if (eql(call.name, "bench")) {
        return callEngineResearch(
            arena,
            &.{"bench"},
        );
    }
    if (eql(call.name, "bench_compare")) {
        return callEngineResearch(
            arena,
            &.{"bench-compare"},
        );
    }
    if (eql(call.name, "show")) {
        return executeShow(arena, call.input_json);
    }
    if (eql(call.name, "show_function")) {
        return executeShowFunction(
            arena,
            call.input_json,
        );
    }

    // File I/O tools.
    if (eql(call.name, "read_file")) {
        return executeReadFile(
            arena,
            call.input_json,
        );
    }
    if (eql(call.name, "write_file")) {
        return executeWriteFile(
            arena,
            call.input_json,
        );
    }
    if (eql(call.name, "edit_file")) {
        return executeEditFile(
            arena,
            call.input_json,
        );
    }
    if (eql(call.name, "list_directory")) {
        return executeListDirectory(
            arena,
            call.input_json,
        );
    }

    // Shell tool.
    if (eql(call.name, "run_command")) {
        return executeRunCommand(
            arena,
            call.input_json,
        );
    }

    return .{
        .stdout = "Error: unknown tool name",
        .success = false,
    };
}

// ============================================================
// Engine research tool dispatchers
// ============================================================

/// Rollback to a specific snapshot by ID.
fn executeRollback(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const id = extractRequiredField(
        input,
        "id",
    ) orelse {
        return .{
            .stdout = "Error: missing 'id' field",
            .success = false,
        };
    };
    std.debug.assert(id.len > 0);
    return callEngineResearch(
        arena,
        &.{ "rollback", id },
    );
}

/// Show diff against a specific snapshot.
fn executeDiff(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const id = extractRequiredField(
        input,
        "id",
    ) orelse {
        return .{
            .stdout = "Error: missing 'id' field",
            .success = false,
        };
    };
    std.debug.assert(id.len > 0);
    return callEngineResearch(
        arena,
        &.{ "diff", id },
    );
}

/// Show a source file via the engine_research toolbox.
fn executeShow(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const file = extractRequiredField(
        input,
        "file",
    ) orelse {
        return .{
            .stdout = "Error: missing 'file' field",
            .success = false,
        };
    };
    std.debug.assert(file.len > 0);
    return callEngineResearch(
        arena,
        &.{ "show", file },
    );
}

/// Extract a specific function from a source file.
fn executeShowFunction(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const file = extractRequiredField(
        input,
        "file",
    ) orelse {
        return .{
            .stdout = "Error: missing 'file' field",
            .success = false,
        };
    };
    const func = extractRequiredField(
        input,
        "function_name",
    ) orelse {
        return .{
            .stdout = "Error: missing 'function_name'",
            .success = false,
        };
    };
    std.debug.assert(file.len > 0);
    return callEngineResearch(
        arena,
        &.{ "show-function", file, func },
    );
}

/// Call `./zig-out/bin/engine_research <args...>` and
/// capture stdout.
fn callEngineResearch(
    arena: Allocator,
    args: []const []const u8,
) ToolOutput {
    std.debug.assert(args.len > 0);
    std.debug.assert(args.len + 1 <= MAX_TOOL_ARGS);

    var argv: [MAX_TOOL_ARGS][]const u8 = undefined;
    argv[0] = TOOL_PATH;
    for (args, 0..) |arg, i| {
        argv[1 + i] = arg;
    }

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = argv[0 .. 1 + args.len],
        .max_output_bytes = MAX_OUTPUT_BYTES,
    }) catch |err| {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: spawn failed: {s}",
            .{@errorName(err)},
        ) catch "Error: spawn failed";
        return .{ .stdout = msg, .success = false };
    };

    // Forward progress to the human on stderr.
    if (result.stderr.len > 0) {
        stderr_file.writeAll(result.stderr) catch {};
    }

    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    const output = if (result.stdout.len > 0)
        result.stdout
    else if (result.stderr.len > 0)
        result.stderr
    else
        "(no output)";

    return .{ .stdout = output, .success = ok };
}

// ============================================================
// File I/O tool dispatchers
// ============================================================

/// Read raw contents of a project file.
fn executeReadFile(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };
    std.debug.assert(path.len > 0);

    if (!isAllowedReadPath(path)) {
        log(
            "    read DENIED: {s}\n",
            .{path},
        );
        const msg = std.fmt.allocPrint(
            arena,
            "Error: read not allowed for '{s}'. " ++
                "Only project files are accessible.",
            .{path},
        ) catch "Error: path not allowed";
        return .{ .stdout = msg, .success = false };
    }

    const content = readFileContent(
        arena,
        path,
    ) orelse {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: cannot read '{s}'",
            .{path},
        ) catch "Error: cannot read file";
        return .{ .stdout = msg, .success = false };
    };

    log(
        "    read {s}: {d} bytes\n",
        .{ path, content.len },
    );
    return .{ .stdout = content, .success = true };
}

/// Write new contents to an engine source file.
/// Restricted to ALLOWED_WRITE_FILES for safety.
fn executeWriteFile(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };
    std.debug.assert(path.len > 0);

    if (!isAllowedWritePath(path)) {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: write not allowed to '{s}'. " ++
                "Allowed: src/metal.zig, " ++
                "src/network.zig, " ++
                "src/shaders/compute.metal, " ++
                "src/layout.zig, src/main.zig",
            .{path},
        ) catch "Error: path not allowed";
        return .{ .stdout = msg, .success = false };
    }

    const raw = extractRequiredField(
        input,
        "content",
    ) orelse {
        return .{
            .stdout = "Error: missing 'content' field",
            .success = false,
        };
    };
    const content = unescapeJsonString(
        arena,
        raw,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape content",
            .success = false,
        };
    };

    writeFileContent(path, content) catch {
        return .{
            .stdout = "Error: failed to write file",
            .success = false,
        };
    };

    log(
        "    wrote {s}: {d} bytes\n",
        .{ path, content.len },
    );

    const msg = std.fmt.allocPrint(
        arena,
        "OK: wrote {d} bytes to {s}",
        .{ content.len, path },
    ) catch "OK: file written";
    return .{ .stdout = msg, .success = true };
}

/// Targeted find-and-replace in an engine source file.
/// More efficient than write_file for small edits.
fn executeEditFile(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };
    std.debug.assert(path.len > 0);

    if (!isAllowedWritePath(path)) {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: edit not allowed for '{s}'",
            .{path},
        ) catch "Error: path not allowed";
        return .{ .stdout = msg, .success = false };
    }

    const old_raw = extractRequiredField(
        input,
        "old_content",
    ) orelse {
        return .{
            .stdout = "Error: missing 'old_content'",
            .success = false,
        };
    };
    const new_raw = extractRequiredField(
        input,
        "new_content",
    ) orelse {
        return .{
            .stdout = "Error: missing 'new_content'",
            .success = false,
        };
    };

    const old_text = unescapeJsonString(
        arena,
        old_raw,
    ) catch {
        return .{
            .stdout = "Error: unescape old_content",
            .success = false,
        };
    };
    const new_text = unescapeJsonString(
        arena,
        new_raw,
    ) catch {
        return .{
            .stdout = "Error: unescape new_content",
            .success = false,
        };
    };

    if (old_text.len == 0) {
        return .{
            .stdout = "Error: old_content is empty",
            .success = false,
        };
    }

    // Read the current file.
    const current = readFileContent(
        arena,
        path,
    ) orelse {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: cannot read '{s}'",
            .{path},
        ) catch "Error: cannot read file";
        return .{ .stdout = msg, .success = false };
    };

    // Find the old text.
    const pos = indexOf(
        current,
        old_text,
    ) orelse {
        // Provide context to help Claude fix the match.
        const preview = truncate(old_text, 100);
        const msg = std.fmt.allocPrint(
            arena,
            "Error: old_content not found in {s}. " ++
                "Searched for: \"{s}...\" " ++
                "({d} chars). Use read_file or " ++
                "show_function to see current " ++
                "contents.",
            .{ path, preview, old_text.len },
        ) catch "Error: old_content not found";
        return .{ .stdout = msg, .success = false };
    };

    // Check for ambiguous matches.
    const after_first = pos + old_text.len;
    if (after_first < current.len) {
        if (indexOf(
            current[after_first..],
            old_text,
        ) != null) {
            return .{
                .stdout = "Error: old_content matches " ++
                    "multiple locations. Make the " ++
                    "search string more specific.",
                .success = false,
            };
        }
    }

    // Build the new file content.
    const new_file = std.fmt.allocPrint(
        arena,
        "{s}{s}{s}",
        .{
            current[0..pos],
            new_text,
            current[after_first..],
        },
    ) catch {
        return .{
            .stdout = "Error: allocating new content",
            .success = false,
        };
    };

    writeFileContent(path, new_file) catch {
        return .{
            .stdout = "Error: failed to write file",
            .success = false,
        };
    };

    const removed = old_text.len;
    const added = new_text.len;

    log(
        "    edit {s}: -{d} +{d} chars " ++
            "({d} bytes total)\n",
        .{ path, removed, added, new_file.len },
    );

    const msg = std.fmt.allocPrint(
        arena,
        "OK: edited {s} " ++
            "(-{d} +{d} chars, " ++
            "{d} bytes total)",
        .{ path, removed, added, new_file.len },
    ) catch "OK: file edited";
    return .{ .stdout = msg, .success = true };
}

/// List files and subdirectories in a project directory.
fn executeListDirectory(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };
    std.debug.assert(path.len > 0);

    if (!isAllowedReadPath(path)) {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: listing not allowed for '{s}'",
            .{path},
        ) catch "Error: path not allowed";
        return .{ .stdout = msg, .success = false };
    }

    // Use run_command internally for simplicity.
    const cmd = std.fmt.allocPrint(
        arena,
        "ls -la {s}",
        .{path},
    ) catch {
        return .{
            .stdout = "Error: formatting command",
            .success = false,
        };
    };
    return runShellCommand(arena, cmd);
}

// ============================================================
// Shell tool dispatcher
// ============================================================

/// Execute a shell command in the project root directory.
fn executeRunCommand(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const command = extractRequiredField(
        input,
        "command",
    ) orelse {
        return .{
            .stdout = "Error: missing 'command' field",
            .success = false,
        };
    };

    if (command.len == 0) {
        return .{
            .stdout = "Error: empty command",
            .success = false,
        };
    }

    log("    cmd: {s}\n", .{truncate(command, 200)});
    return runShellCommand(arena, command);
}

/// Actually run a shell command and capture output.
fn runShellCommand(
    arena: Allocator,
    command: []const u8,
) ToolOutput {
    std.debug.assert(command.len > 0);

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "/bin/sh", "-c", command },
        .max_output_bytes = MAX_COMMAND_OUTPUT,
    }) catch |err| {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: failed to spawn: {s}",
            .{@errorName(err)},
        ) catch "Error: spawn failed";
        return .{ .stdout = msg, .success = false };
    };

    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    // Combine stdout and stderr for full context.
    var buf: std.ArrayList(u8) = .empty;
    if (result.stdout.len > 0) {
        buf.appendSlice(
            arena,
            result.stdout,
        ) catch {};
    }
    if (result.stderr.len > 0) {
        if (buf.items.len > 0) {
            buf.appendSlice(
                arena,
                "\n--- stderr ---\n",
            ) catch {};
        }
        buf.appendSlice(
            arena,
            result.stderr,
        ) catch {};
    }

    const output = if (buf.items.len > 0)
        buf.items
    else
        "(no output)";

    // Log exit code on failure for debugging.
    if (!ok) {
        switch (result.term) {
            .Exited => |code| log(
                "    exit code: {d}\n",
                .{code},
            ),
            else => log(
                "    terminated abnormally\n",
                .{},
            ),
        }
    }

    return .{ .stdout = output, .success = ok };
}

// ============================================================
// Path validation
// ============================================================

/// Check if a path is safe for reading.
/// Rejects path traversal and files outside the project.
fn isAllowedReadPath(path: []const u8) bool {
    std.debug.assert(path.len > 0);

    // Reject path traversal.
    if (indexOf(path, "..") != null) return false;

    // Reject absolute paths.
    if (path.len > 0 and path[0] == '/') return false;

    // Allow exact root-level files.
    for (&ALLOWED_READ_FILES) |file| {
        if (eql(path, file)) return true;
    }

    // Allow paths under allowed prefixes.
    for (&ALLOWED_READ_PREFIXES) |prefix| {
        if (startsWith(path, prefix)) return true;
    }

    // Also allow any write-allowed file to be read.
    for (&ALLOWED_WRITE_FILES) |file| {
        if (eql(path, file)) return true;
    }

    return false;
}

/// Check if a path is in the allowed write list.
fn isAllowedWritePath(path: []const u8) bool {
    std.debug.assert(path.len > 0);

    // Reject path traversal.
    if (indexOf(path, "..") != null) return false;

    // Reject absolute paths.
    if (path.len > 0 and path[0] == '/') return false;

    for (&ALLOWED_WRITE_FILES) |allowed| {
        if (eql(path, allowed)) return true;
    }
    return false;
}

// ============================================================
// Field extraction and file I/O helpers
// ============================================================

/// Extract a JSON string field by key name.
/// Tries both compact and spaced forms.
fn extractRequiredField(
    input_json: []const u8,
    comptime field: []const u8,
) ?[]const u8 {
    comptime {
        std.debug.assert(field.len > 0);
    }
    if (input_json.len == 0) return null;

    return extractJsonString(
        input_json,
        "\"" ++ field ++ "\":\"",
    ) orelse extractJsonString(
        input_json,
        "\"" ++ field ++ "\": \"",
    );
}

/// Read a file's contents into an arena-allocated slice.
fn readFileContent(
    arena: Allocator,
    path: []const u8,
) ?[]const u8 {
    std.debug.assert(path.len > 0);

    const file = std.fs.cwd().openFile(
        path,
        .{},
    ) catch return null;
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_FILE_SIZE,
    ) catch return null;

    std.debug.assert(content.len <= MAX_FILE_SIZE);
    return content;
}

/// Write contents to a file, creating or truncating it.
fn writeFileContent(
    path: []const u8,
    content: []const u8,
) !void {
    std.debug.assert(path.len > 0);
    std.debug.assert(content.len <= MAX_FILE_SIZE);

    const file = try std.fs.cwd().createFile(
        path,
        .{},
    );
    defer file.close();
    try file.writeAll(content);
}

/// Unescape a JSON string value (\\n -> newline, etc.).
fn unescapeJsonString(
    arena: Allocator,
    s: []const u8,
) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '\\' and i + 1 < s.len) {
            const next = s[i + 1];
            const c: ?u8 = switch (next) {
                '"' => '"',
                '\\' => '\\',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '/' => '/',
                else => null,
            };
            if (c) |ch| {
                try buf.append(arena, ch);
                i += 2;
            } else if (next == 'u' and
                i + 5 < s.len)
            {
                // \\uXXXX — decode basic ASCII range.
                const hex = s[i + 2 .. i + 6];
                const cp = std.fmt.parseInt(
                    u16,
                    hex,
                    16,
                ) catch {
                    try buf.append(arena, s[i]);
                    i += 1;
                    continue;
                };
                if (cp < 128) {
                    try buf.append(
                        arena,
                        @intCast(cp),
                    );
                } else {
                    // Non-ASCII: pass through raw.
                    try buf.appendSlice(
                        arena,
                        s[i .. i + 6],
                    );
                }
                i += 6;
            } else {
                try buf.append(arena, s[i]);
                i += 1;
            }
        } else {
            try buf.append(arena, s[i]);
            i += 1;
        }
    }

    std.debug.assert(buf.items.len <= s.len + 256);
    return buf.items;
}

// ============================================================
// Message building
// ============================================================

/// Build the initial user message with history context.
fn buildInitialMessage(
    arena: Allocator,
    history: []const u8,
) []const u8 {
    const has_history = history.len > 0;

    const intro = if (has_history)
        "Benchmark history from previous runs " ++
            "(JSONL -- one result per line):\n\n"
    else
        "No previous benchmark history. " ++
            "This is the first run.\n\n";

    const suffix =
        "\nBegin optimising engine throughput. " ++
        "Start by calling snapshot to create a " ++
        "restore point, then read the source " ++
        "code to understand the baseline.";

    const text = if (has_history)
        std.fmt.allocPrint(
            arena,
            "{s}{s}{s}",
            .{ intro, history, suffix },
        ) catch "Begin optimising."
    else
        std.fmt.allocPrint(
            arena,
            "{s}{s}",
            .{ intro, suffix },
        ) catch "Begin optimising.";

    std.debug.assert(text.len > 0);
    const result = wrapUserTextMessage(arena, text);
    std.debug.assert(result.len > 0);
    return result;
}

/// Wrap Claude's raw content JSON as an assistant message.
fn buildAssistantMsg(
    arena: Allocator,
    content_json: []const u8,
) []const u8 {
    std.debug.assert(content_json.len >= 2);

    return std.fmt.allocPrint(
        arena,
        "{{\"role\":\"assistant\",\"content\":{s}}}",
        .{content_json},
    ) catch "{}";
}

/// Build a user message containing tool results.
fn buildToolResultsMsg(
    arena: Allocator,
    results: []const ToolResult,
) []const u8 {
    std.debug.assert(results.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"role\":\"user\",\"content\":[",
    ) catch return "{}";

    for (results, 0..) |r, i| {
        if (i > 0) buf.append(arena, ',') catch {};
        appendToolResultBlock(arena, &buf, r);
    }

    buf.appendSlice(arena, "]}") catch {};
    return buf.items;
}

/// Append one tool_result block to a buffer.
fn appendToolResultBlock(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    r: ToolResult,
) void {
    std.debug.assert(r.tool_use_id.len > 0);

    buf.appendSlice(
        arena,
        "{\"type\":\"tool_result\"," ++
            "\"tool_use_id\":\"",
    ) catch return;
    buf.appendSlice(
        arena,
        r.tool_use_id,
    ) catch return;
    buf.appendSlice(arena, "\",") catch return;

    if (r.is_error) {
        buf.appendSlice(
            arena,
            "\"is_error\":true,",
        ) catch return;
    }

    buf.appendSlice(
        arena,
        "\"content\":",
    ) catch return;
    appendJsonString(
        arena,
        buf,
        r.content,
    ) catch return;
    buf.append(arena, '}') catch return;
}

/// Wrap a plain text string as a user message.
fn wrapUserTextMessage(
    arena: Allocator,
    text: []const u8,
) []const u8 {
    std.debug.assert(text.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"role\":\"user\",\"content\":",
    ) catch return "{}";
    appendJsonString(
        arena,
        &buf,
        text,
    ) catch return "{}";
    buf.append(arena, '}') catch return "{}";

    std.debug.assert(buf.items.len > 0);
    return buf.items;
}

// ============================================================
// History persistence
//
// Append-only JSONL at .engine_agent_history/experiments.jsonl.
// Each line is a compacted benchmark JSON from a bench call.
// Loaded on startup and injected into the first user message.
// ============================================================

fn ensureHistoryDir() void {
    std.fs.cwd().makeDir(HISTORY_DIR) catch |err| {
        if (err != error.PathAlreadyExists) {
            log(
                "WARNING: mkdir {s}: {s}\n",
                .{ HISTORY_DIR, @errorName(err) },
            );
        }
    };
}

/// Load the raw history file content, truncated to
/// MAX_HISTORY_INJECT to avoid blowing up the context.
fn loadHistoryContent(arena: Allocator) []const u8 {
    const file = std.fs.cwd().openFile(
        HISTORY_PATH,
        .{},
    ) catch return "";
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch return "";

    if (content.len == 0) return "";

    log(
        "Loaded {d} bytes of experiment history.\n",
        .{content.len},
    );

    return truncate(content, MAX_HISTORY_INJECT);
}

/// Append a bench result to the history JSONL file.
fn appendExperiment(
    arena: Allocator,
    benchmark_json: []const u8,
) void {
    std.debug.assert(benchmark_json.len > 0);

    const line = collapseToLine(
        arena,
        benchmark_json,
    ) catch return;

    const file = std.fs.cwd().createFile(
        HISTORY_PATH,
        .{ .truncate = false },
    ) catch return;
    defer file.close();

    file.seekFromEnd(0) catch return;
    file.writeAll(line) catch return;
    file.writeAll("\n") catch {};
}

/// Remove newlines to produce a single-line JSON string.
fn collapseToLine(
    arena: Allocator,
    s: []const u8,
) ![]const u8 {
    std.debug.assert(s.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    for (s) |c| {
        if (c != '\n' and c != '\r') {
            try buf.append(arena, c);
        }
    }

    std.debug.assert(buf.items.len <= s.len);
    return buf.items;
}

/// Save the full conversation as a JSON array.
fn saveRunLog(
    arena: Allocator,
    messages: []const []const u8,
) void {
    std.debug.assert(messages.len > 0);

    var ts_buf: [TIMESTAMP_LEN]u8 = undefined;
    const ts = formatTimestamp(&ts_buf, '-');

    const path = std.fmt.allocPrint(
        arena,
        "{s}/run_{s}.json",
        .{ HISTORY_DIR, ts },
    ) catch return;

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(arena, "[\n") catch return;
    for (messages, 0..) |msg, i| {
        if (i > 0) {
            buf.appendSlice(
                arena,
                ",\n",
            ) catch return;
        }
        buf.appendSlice(arena, "  ") catch return;
        buf.appendSlice(arena, msg) catch return;
    }
    buf.appendSlice(arena, "\n]\n") catch return;

    writeFile(path, buf.items) catch {};
    log("Run log saved: {s}\n", .{path});
}

// ============================================================
// Logging (all output goes to stderr)
// ============================================================

const stdout_file = std.fs.File{
    .handle = std.posix.STDOUT_FILENO,
};
const stderr_file = std.fs.File{
    .handle = std.posix.STDERR_FILENO,
};

fn log(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
}

fn fatal(comptime msg: []const u8) void {
    log("FATAL: " ++ msg, .{});
    std.process.exit(1);
}

fn printHeader() void {
    log(
        "\nnnzap engine agent" ++
            " - LLM-powered engine optimiser\n" ++
            "==================================" ++
            "================\n\n",
        .{},
    );
}

fn logClaudeText(text: []const u8) void {
    std.debug.assert(text.len > 0);
    log(
        "\n  Claude: {s}\n",
        .{truncate(text, 2000)},
    );
}

fn logToolCall(
    name: []const u8,
    input: []const u8,
) void {
    std.debug.assert(name.len > 0);
    if (input.len <= 2) {
        // Empty input like "{}" -- omit it.
        log("  Tool:   {s}\n", .{name});
    } else {
        log(
            "  Tool:   {s} {s}\n",
            .{ name, truncate(input, 200) },
        );
    }
}

// ============================================================
// JSON helpers
// ============================================================

/// Append a JSON-escaped, quoted string to a buffer.
fn appendJsonString(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    s: []const u8,
) !void {
    try buf.append(arena, '"');
    for (s) |c| {
        switch (c) {
            '"' => {
                try buf.appendSlice(arena, "\\\"");
            },
            '\\' => {
                try buf.appendSlice(arena, "\\\\");
            },
            '\n' => {
                try buf.appendSlice(arena, "\\n");
            },
            '\r' => {
                try buf.appendSlice(arena, "\\r");
            },
            '\t' => {
                try buf.appendSlice(arena, "\\t");
            },
            else => {
                if (c < 0x20) {
                    const hex = try std.fmt.allocPrint(
                        arena,
                        "\\u{x:0>4}",
                        .{c},
                    );
                    try buf.appendSlice(arena, hex);
                } else {
                    try buf.append(arena, c);
                }
            },
        }
    }
    try buf.append(arena, '"');
}

/// Extract a JSON string value after a needle ending
/// with `:"`.
fn extractJsonString(
    json: []const u8,
    needle: []const u8,
) ?[]const u8 {
    std.debug.assert(needle.len > 0);

    const idx = indexOf(json, needle) orelse {
        return null;
    };
    const start = idx + needle.len;
    if (start >= json.len) return null;
    return findStringEnd(json[start..]);
}

/// Scan forward past a JSON string body, stopping at
/// the first unescaped `"`.
fn findStringEnd(s: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '\\') {
            i += 2;
            continue;
        }
        if (s[i] == '"') return s[0..i];
        i += 1;
    }
    return null;
}

/// Extract a raw JSON array value for a given field.
fn extractRawArray(
    json: []const u8,
    field: []const u8,
) ?[]const u8 {
    std.debug.assert(field.len > 0);

    const idx = indexOf(json, field) orelse {
        return null;
    };
    var pos = idx + field.len;
    pos = skipColonAndWhitespace(json, pos);
    if (pos >= json.len) return null;
    if (json[pos] != '[') return null;
    return findMatchingBracket(
        json[pos..],
        '[',
        ']',
    );
}

/// Extract a raw JSON object value for a given field.
fn extractRawObject(
    json: []const u8,
    field: []const u8,
) ?[]const u8 {
    std.debug.assert(field.len > 0);

    const idx = indexOf(json, field) orelse {
        return null;
    };
    var pos = idx + field.len;
    pos = skipColonAndWhitespace(json, pos);
    if (pos >= json.len) return null;
    if (json[pos] != '{') return null;
    return findMatchingBracket(
        json[pos..],
        '{',
        '}',
    );
}

/// Skip past `:`, spaces, tabs, and newlines.
fn skipColonAndWhitespace(
    json: []const u8,
    start: usize,
) usize {
    var pos = start;
    while (pos < json.len) {
        const c = json[pos];
        if (c == ':' or c == ' ' or c == '\n' or
            c == '\r' or c == '\t')
        {
            pos += 1;
        } else {
            break;
        }
    }
    return pos;
}

/// Find the substring from `open` to its matching
/// `close`, respecting nesting and JSON strings.
fn findMatchingBracket(
    s: []const u8,
    open: u8,
    close: u8,
) ?[]const u8 {
    std.debug.assert(s.len > 0);
    std.debug.assert(s[0] == open);

    var depth: u32 = 0;
    var in_string = false;
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c == '\\' and in_string) {
            i += 2;
            continue;
        }
        if (c == '"') in_string = !in_string;
        if (!in_string) {
            if (c == open) depth += 1;
            if (c == close) {
                depth -= 1;
                if (depth == 0) {
                    return s[0 .. i + 1];
                }
            }
        }
        i += 1;
    }
    return null;
}

/// Split a JSON array into its top-level objects.
fn splitTopLevelObjects(
    arena: Allocator,
    array_json: []const u8,
) ![]const []const u8 {
    std.debug.assert(array_json.len >= 2);

    var items: std.ArrayList([]const u8) = .empty;
    var pos: usize = 0;

    // Find opening bracket.
    while (pos < array_json.len and
        array_json[pos] != '[') : (pos += 1)
    {}
    pos += 1; // Skip '['.

    while (pos < array_json.len) {
        // Skip whitespace and commas.
        while (pos < array_json.len) {
            const c = array_json[pos];
            if (c == ' ' or c == '\n' or
                c == '\r' or c == '\t' or c == ',')
            {
                pos += 1;
            } else {
                break;
            }
        }
        if (pos >= array_json.len) break;
        if (array_json[pos] == ']') break;
        if (array_json[pos] != '{') {
            pos += 1;
            continue;
        }

        const obj = findMatchingBracket(
            array_json[pos..],
            '{',
            '}',
        ) orelse break;
        try items.append(arena, obj);
        pos += obj.len;
    }

    return items.items;
}

/// Check if a JSON block has "type":"<value>".
fn containsField(
    block: []const u8,
    key: []const u8,
    value: []const u8,
) bool {
    std.debug.assert(key.len > 0);
    std.debug.assert(value.len > 0);

    const idx = indexOf(block, key) orelse {
        return false;
    };
    var pos = idx + key.len;

    pos = skipColonAndWhitespace(block, pos);
    if (pos >= block.len) return false;
    if (block[pos] != '"') return false;
    pos += 1;

    if (pos + value.len > block.len) return false;
    if (!std.mem.eql(
        u8,
        block[pos..][0..value.len],
        value,
    )) {
        return false;
    }
    pos += value.len;

    if (pos >= block.len) return false;
    return block[pos] == '"';
}

// ============================================================
// String helpers
// ============================================================

fn indexOf(
    haystack: []const u8,
    needle: []const u8,
) ?usize {
    return std.mem.indexOf(u8, haystack, needle);
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn startsWith(s: []const u8, prefix: []const u8) bool {
    if (s.len < prefix.len) return false;
    return std.mem.eql(u8, s[0..prefix.len], prefix);
}

fn truncate(s: []const u8, max: usize) []const u8 {
    std.debug.assert(max > 0);
    return if (s.len <= max) s else s[0..max];
}

// ============================================================
// File I/O
// ============================================================

fn writeFile(
    path: []const u8,
    content: []const u8,
) !void {
    std.debug.assert(path.len > 0);
    std.debug.assert(content.len > 0);

    const file = try std.fs.cwd().createFile(
        path,
        .{},
    );
    defer file.close();
    try file.writeAll(content);
}

// ============================================================
// Timestamp formatting
// ============================================================

fn formatTimestamp(
    buf: *[TIMESTAMP_LEN]u8,
    separator: u8,
) []const u8 {
    const secs: u64 = @intCast(std.time.timestamp());
    const es = std.time.epoch.EpochSeconds{
        .secs = secs,
    };
    const epoch_day = es.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = es.getDaySeconds();

    return std.fmt.bufPrint(
        buf,
        "{d:0>4}-{d:0>2}-{d:0>2}" ++
            "T{d:0>2}{c}{d:0>2}{c}{d:0>2}Z",
        .{
            year_day.year,
            @intFromEnum(month_day.month),
            month_day.day_index + 1,
            day_secs.getHoursIntoDay(),
            separator,
            day_secs.getMinutesIntoHour(),
            separator,
            day_secs.getSecondsIntoMinute(),
        },
    ) catch unreachable;
}

// ============================================================
// Timing helpers
// ============================================================

/// Current wall-clock time in milliseconds.
fn timestampMs() i64 {
    const nanos: i128 = std.time.nanoTimestamp();
    return @intCast(@divFloor(nanos, 1_000_000));
}

/// Format a nanosecond duration as a human-readable string.
/// Returns "123ms" for sub-second, "1.234s" for seconds,
/// "2m05s" for minutes.
fn nanosToMsStr(
    arena: Allocator,
    nanos: i64,
) []const u8 {
    std.debug.assert(nanos >= 0);
    const ms = @divFloor(nanos, 1_000_000);
    if (ms < 1_000) {
        return std.fmt.allocPrint(
            arena,
            "{d}ms",
            .{ms},
        ) catch "?ms";
    }
    if (ms < 60_000) {
        const secs = @divFloor(ms, 1_000);
        const rem = @mod(ms, 1_000);
        return std.fmt.allocPrint(
            arena,
            "{d}.{d:0>3}s",
            .{ secs, rem },
        ) catch "?s";
    }
    const total_secs = @divFloor(ms, 1_000);
    const mins = @divFloor(total_secs, 60);
    const secs = @mod(total_secs, 60);
    return std.fmt.allocPrint(
        arena,
        "{d}m{d:0>2}s",
        .{ mins, secs },
    ) catch "?m";
}

// ============================================================
// Helpers
// ============================================================

fn errResp(
    msg: []const u8,
    retryable: bool,
) ApiResponse {
    return .{
        .success = false,
        .stop_reason = "",
        .content_json = "[]",
        .text = "",
        .tool_calls = &.{},
        .error_message = msg,
        .retryable = retryable,
    };
}
