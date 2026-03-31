//! nnzap agent — LLM-powered autonomous experiment runner.
//!
//! A tool-calling loop that talks to Claude via the Anthropic
//! API.  Claude decides which experiments to run; the agent
//! executes them using the autoresearch toolbox, sends results
//! back, and loops until Claude stops or MAX_TURNS is reached.
//!
//! The agent IS the runtime — like Claude Code, but ours.
//!
//! Setup:
//!   export ANTHROPIC_API_KEY=sk-ant-...
//!   zig build
//!   ./zig-out/bin/agent
//!
//! Persists results to .agent_history/ so context accumulates
//! across runs.
//!
//! Output contract:
//!   stderr → human-readable progress log
//!   .agent_history/experiments.jsonl → train results log
//!   .agent_history/run_{ts}.json    → per-run summary

const std = @import("std");
const api = @import("api_client.zig");
const http = std.http;
const Uri = std.Uri;
const Allocator = std.mem.Allocator;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_TURNS: u32 = 200;
const MAX_MESSAGES: u32 = 512;
const MAX_TOOL_CALLS: u32 = 16;
const MAX_TOOL_OUTPUT: usize = 50_000;
const MAX_TOOL_ARGS: u32 = 16;
const MAX_API_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_HISTORY_SIZE: usize = 2 * 1024 * 1024;
const MAX_HISTORY_INJECT: usize = 30_000;

const API_URL: []const u8 =
    "https://api.anthropic.com/v1/messages";
const API_VERSION: []const u8 = "2023-06-01";
const DEFAULT_MODEL: []const u8 = "claude-opus-4-6";
const MAX_TOKENS_STR: []const u8 = "16384";

const TOOL_PATH: []const u8 = "./zig-out/bin/autoresearch";
const HISTORY_DIR: []const u8 = ".agent_history";
const HISTORY_PATH: []const u8 =
    ".agent_history/experiments.jsonl";

// Filesystem paths (relative to zap/ working directory).
const FS_HISTORY_DIR: []const u8 = "../" ++ HISTORY_DIR;
const FS_HISTORY_PATH: []const u8 = "../" ++ HISTORY_PATH;

// Type aliases for the shared API client types.
const ToolCall = api.ToolCall;
const ToolOutput = api.ToolOutput;
const ToolResult = api.ToolResult;
const ApiResponse = api.ApiResponse;

// ============================================================
// System prompt
//
// This is the context seed.  It tells Claude what it is,
// what tools are available, what constraints apply, and
// what strategy to follow.  Multiline string literal —
// each \\-prefixed line is literal content.
// ============================================================

const SYSTEM_PROMPT =
    \\You are an autonomous ML research agent optimizing
    \\nnzap's MNIST training pipeline. nnzap is a Zig +
    \\Metal GPU-accelerated neural network library for
    \\Apple Silicon with zero-copy unified memory.
    \\
    \\Goal: maximize final_test_accuracy_pct on MNIST.
    \\
    \\## Protocol
    \\
    \\For each experiment:
    \\1. config_show — check current state.
    \\2. config_backup — safety net before changes.
    \\3. config_set — apply your experiment.
    \\4. train — run training (returns benchmark JSON with
    \\   final_test_accuracy_pct, throughput_images_per_sec,
    \\   total_training_ms, and per-epoch details).
    \\5. Evaluate the benchmark result:
    \\   - Accuracy up >= 0.05 pp: KEEP the change.
    \\   - Accuracy within +/- 0.05 pp: KEEP if throughput
    \\     improved.
    \\   - Accuracy dropped: REVERT with config_restore.
    \\6. Pick the next experiment and repeat.
    \\
    \\## Constraints
    \\
    \\- First layer input must be 784 (28x28 pixels).
    \\- Last layer output must be 10, activation must be
    \\  none (raw logits for softmax + cross-entropy).
    \\- Adjacent layers: layer[i].out == layer[i+1].in.
    \\- Batch size must evenly divide 50000 (training set).
    \\  Valid: 1,2,4,5,8,10,16,20,25,32,40,50,64,80,100,
    \\  125,128,160,200,250,400,500,625,1000,2500,5000.
    \\- Epochs <= 30 to bound experiment time.
    \\- Activations: relu, tanh_act, sigmoid, none.
    \\- Optimizers: sgd, adam.
    \\
    \\## Strategy
    \\
    \\Phase 1: Try Adam optimizer (often beats SGD on MNIST).
    \\  Compare adam lr=0.001 vs baseline SGD.
    \\Phase 2: Tune learning rate for the winning optimizer.
    \\  Try 0.0003, 0.003, 0.01 for Adam; 0.05, 0.2 for SGD.
    \\Phase 3: Architecture search.
    \\  Try wider (256, 512), deeper (3-4 hidden layers), or
    \\  different width combos.
    \\Phase 4: Batch size (32 vs 64 vs 128).
    \\Phase 5: Fine-tune (Adam betas, more epochs, combos).
    \\
    \\Winners accumulate — keep improvements across phases.
    \\Do not repeat configurations that already failed.
    \\When improvements plateau, summarize and stop.
    \\
    \\The first user message includes experiment history from
    \\previous runs. Use it to avoid repeating past work.
;

// ============================================================
// Tool schemas
//
// JSON array of tool definitions for the Anthropic API.
// Each tool maps to an autoresearch CLI command.
// ============================================================

const TOOL_SCHEMAS =
    \\[
    \\  {
    \\    "name": "config_show",
    \\    "description": "Show current hyperparameters from main.zig. Returns JSON with architecture, learning_rate, optimizer, batch_size, epochs, seed.",
    \\    "input_schema": {
    \\      "type": "object",
    \\      "properties": {},
    \\      "required": []
    \\    }
    \\  },
    \\  {
    \\    "name": "config_set",
    \\    "description": "Modify hyperparameters in main.zig. Pass key=value pairs as the settings array. Keys: lr (float), batch (int, must divide 50000), epochs (int, max 30), seed (int), optimizer (sgd or adam), arch (layer spec like 784:256:relu,256:10:none), beta1 (float), beta2 (float), epsilon (float).",
    \\    "input_schema": {
    \\      "type": "object",
    \\      "properties": {
    \\        "settings": {
    \\          "type": "array",
    \\          "items": {"type": "string"},
    \\          "description": "Key=value pairs, e.g. [\"optimizer=adam\", \"lr=0.001\"]"
    \\        }
    \\      },
    \\      "required": ["settings"]
    \\    }
    \\  },
    \\  {
    \\    "name": "config_backup",
    \\    "description": "Backup main.zig before making changes. Always call before config_set so you can revert.",
    \\    "input_schema": {
    \\      "type": "object",
    \\      "properties": {},
    \\      "required": []
    \\    }
    \\  },
    \\  {
    \\    "name": "config_restore",
    \\    "description": "Restore main.zig from backup. Use after an experiment made things worse.",
    \\    "input_schema": {
    \\      "type": "object",
    \\      "properties": {},
    \\      "required": []
    \\    }
    \\  },
    \\  {
    \\    "name": "train",
    \\    "description": "Build and run MNIST training. Returns full benchmark JSON including final_test_accuracy_pct, throughput_images_per_sec, total_training_ms, per-epoch validation metrics, and test results. Takes about 10 seconds.",
    \\    "input_schema": {
    \\      "type": "object",
    \\      "properties": {},
    \\      "required": []
    \\    }
    \\  },
    \\  {
    \\    "name": "benchmark_compare",
    \\    "description": "Compare all saved benchmark runs side by side. Shows test accuracy, throughput, optimizer, learning rate, and architecture for each past run.",
    \\    "input_schema": {
    \\      "type": "object",
    \\      "properties": {},
    \\      "required": []
    \\    }
    \\  },
    \\  {
    \\    "name": "benchmark_latest",
    \\    "description": "Output the most recent benchmark result as JSON.",
    \\    "input_schema": {
    \\      "type": "object",
    \\      "properties": {},
    \\      "required": []
    \\    }
    \\  }
    \\]
;

// ============================================================
// Main
// ============================================================

pub fn main() !void {
    var arena_state = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    printHeader();
    ensureHistoryDir();

    const api_key = loadApiKey() orelse {
        api.fatal(
            "Set ANTHROPIC_API_KEY env var.\n" ++
                "  export ANTHROPIC_API_KEY=sk-ant-...\n",
        );
    };
    const model = loadModel();
    api.log("Model: {s}\n", .{model});

    if (!buildToolbox(arena)) {
        api.fatal("zig build failed.\n");
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

    var turn: u32 = 0;
    while (turn < MAX_TURNS) : (turn += 1) {
        api.log("\n--- Turn {d} ---\n", .{turn + 1});

        const resp = callApi(
            arena,
            api_key,
            model,
            messages[0..count],
        );
        if (!resp.success) {
            api.log("API error: {s}\n", .{resp.error_message});
            break;
        }
        if (resp.text.len > 0) api.logClaudeText(resp.text);

        messages[count] = api.buildAssistantMsg(
            arena,
            resp.content_json,
        );
        count += 1;

        // Stop when Claude finishes or has no tool calls.
        const is_tool_use = api.eql(
            resp.stop_reason,
            "tool_use",
        );
        if (!is_tool_use or resp.tool_calls.len == 0) break;

        const results = executeTools(
            arena,
            resp.tool_calls,
        );
        messages[count] = api.buildToolResultsMsg(
            arena,
            results,
        );
        count += 1;

        if (count + 2 >= MAX_MESSAGES) {
            api.log("Message limit reached.\n", .{});
            break;
        }
    }

    api.log("\nAgent finished ({d} turns).\n", .{turn + 1});
    api.saveRunLog(arena, messages[0..count], FS_HISTORY_DIR);
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
    api.log("Building toolbox...\n", .{});

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "zig", "build" },
        .max_output_bytes = MAX_OUTPUT_BYTES,
    }) catch {
        return false;
    };

    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (!ok) {
        api.log("Build failed:\n{s}\n", .{result.stderr});
    } else {
        api.log("  done.\n", .{});
    }
    return ok;
}

// ============================================================
// API calling
// ============================================================

/// Call the Anthropic Messages API using std.http.
/// No subprocess, no temp files — direct TLS connection.
fn callApi(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    messages: []const []const u8,
) ApiResponse {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(messages.len > 0);

    const body = api.buildRequestJson(
        arena,
        model,
        messages,
        SYSTEM_PROMPT,
        TOOL_SCHEMAS,
        MAX_TOKENS_STR,
    ) catch return api.errResp("request build failed", false);

    const uri = Uri.parse(API_URL) catch {
        return api.errResp("failed to parse API URL", false);
    };

    var client = http.Client{ .allocator = arena };
    defer client.deinit();

    var req = client.request(.POST, uri, .{
        .headers = .{
            // Disable compression — we don't decode gzip.
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
        return api.errResp("failed to open HTTP request", false);
    };
    defer req.deinit();

    // Content-Length lets the server know the full body
    // size up front — required for the Messages API.
    req.transfer_encoding = .{
        .content_length = body.len,
    };
    req.sendBodyComplete(body) catch {
        return api.errResp("failed to send request body", false);
    };

    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(
        &redirect_buf,
    ) catch {
        return api.errResp("failed to receive response", false);
    };

    if (response.head.status != .ok) {
        return api.errResp("API returned non-200 status", false);
    }

    var transfer_buf: [64]u8 = undefined;
    var reader = response.reader(&transfer_buf);

    const response_data = reader.allocRemaining(
        arena,
        std.Io.Limit.limited(MAX_API_RESPONSE),
    ) catch {
        return api.errResp("failed to read response body", false);
    };

    if (response_data.len == 0) {
        return api.errResp("empty API response", false);
    }

    return api.parseApiResponse(arena, response_data);
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

    api.logToolCall(call.name, call.input_json);
    const output = dispatchTool(arena, call);

    // Persist train results to the history log.
    if (api.eql(call.name, "train") and output.success) {
        appendExperiment(arena, output.stdout);
    }

    return .{
        .tool_use_id = call.id,
        .content = api.truncate(output.stdout, MAX_TOOL_OUTPUT),
        .is_error = !output.success,
    };
}

/// Route a tool call to the correct autoresearch command.
fn dispatchTool(
    arena: Allocator,
    call: ToolCall,
) ToolOutput {
    if (api.eql(call.name, "config_show")) {
        return callAutoresearch(arena, &.{"config-show"});
    }
    if (api.eql(call.name, "config_set")) {
        return executeConfigSet(arena, call.input_json);
    }
    if (api.eql(call.name, "config_backup")) {
        return callAutoresearch(
            arena,
            &.{"config-backup"},
        );
    }
    if (api.eql(call.name, "config_restore")) {
        return callAutoresearch(
            arena,
            &.{"config-restore"},
        );
    }
    if (api.eql(call.name, "train")) {
        return callAutoresearch(arena, &.{"train"});
    }
    if (api.eql(call.name, "benchmark_compare")) {
        return callAutoresearch(
            arena,
            &.{"benchmark-compare"},
        );
    }
    if (api.eql(call.name, "benchmark_latest")) {
        return callAutoresearch(
            arena,
            &.{"benchmark-latest"},
        );
    }
    return .{
        .stdout = "Error: unknown tool name",
        .success = false,
    };
}

/// Parse the settings array from config_set input and
/// call autoresearch config-set with the extracted args.
fn executeConfigSet(
    arena: Allocator,
    input_json: []const u8,
) ToolOutput {
    std.debug.assert(input_json.len > 0);

    const settings = extractSettings(
        arena,
        input_json,
    ) orelse {
        return .{
            .stdout = "Error: missing 'settings' array" ++ " in input. Expected: " ++ "{\"settings\":[\"key=val\"]}",
            .success = false,
        };
    };
    if (settings.len == 0) {
        return .{
            .stdout = "Error: empty settings array",
            .success = false,
        };
    }

    var argv: [MAX_TOOL_ARGS][]const u8 = undefined;
    argv[0] = "config-set";
    const n = @min(settings.len, MAX_TOOL_ARGS - 1);
    for (settings[0..n], 0..) |s, i| {
        argv[1 + i] = s;
    }
    return callAutoresearch(arena, argv[0 .. 1 + n]);
}

/// Call `./zig-out/bin/autoresearch <args...>` and capture
/// stdout.  Returns stdout on success, stderr on failure.
fn callAutoresearch(
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
        return .{
            .stdout = @errorName(err),
            .success = false,
        };
    };

    // Forward training progress to the user.
    if (result.stderr.len > 0) {
        api.stderr_file.writeAll(result.stderr) catch {};
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

/// Extract the settings string array from config_set
/// tool input JSON.
fn extractSettings(
    arena: Allocator,
    input_json: []const u8,
) ?[]const []const u8 {
    const arr = api.extractRawArray(
        input_json,
        "\"settings\"",
    ) orelse return null;
    return api.parseJsonStringArray(arena, arr) catch null;
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
        "Here is the experiment history from previous " ++
            "agent runs (JSONL — one result per line):\n\n"
    else
        "No previous experiment history. " ++
            "This is the first run.\n\n";

    const suffix =
        "\nBegin optimizing MNIST test accuracy. " ++
        "Start by calling config_show to see " ++
        "the current configuration.";

    const text = if (has_history)
        std.fmt.allocPrint(
            arena,
            "{s}{s}{s}",
            .{ intro, history, suffix },
        ) catch "Begin optimizing."
    else
        std.fmt.allocPrint(
            arena,
            "{s}{s}",
            .{ intro, suffix },
        ) catch "Begin optimizing.";

    return api.wrapUserTextMessage(arena, text);
}

// ============================================================
// History persistence
//
// Append-only JSONL log at .agent_history/experiments.jsonl.
// Each line is a compacted benchmark JSON from a train call.
// Loaded on startup and injected into the first user message.
// ============================================================

fn ensureHistoryDir() void {
    std.fs.cwd().makeDir(FS_HISTORY_DIR) catch |err| {
        if (err != error.PathAlreadyExists) {
            api.log(
                "WARNING: mkdir {s}: {s}\n",
                .{ FS_HISTORY_DIR, @errorName(err) },
            );
        }
    };
}

/// Load the raw history file content.  Returns empty string
/// if the file does not exist.  Truncates to
/// MAX_HISTORY_INJECT to avoid blowing up the context.
fn loadHistoryContent(arena: Allocator) []const u8 {
    const file = std.fs.cwd().openFile(
        FS_HISTORY_PATH,
        .{},
    ) catch return "";
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch return "";

    if (content.len == 0) return "";

    api.log(
        "Loaded {d} bytes of experiment history.\n",
        .{content.len},
    );

    return api.truncate(content, MAX_HISTORY_INJECT);
}

/// Append a train benchmark result to the history file.
/// Compacts the JSON to a single line for JSONL format.
fn appendExperiment(
    arena: Allocator,
    benchmark_json: []const u8,
) void {
    std.debug.assert(benchmark_json.len > 0);

    const line = api.collapseToLine(
        arena,
        benchmark_json,
    ) catch return;

    const file = std.fs.cwd().createFile(
        FS_HISTORY_PATH,
        .{ .truncate = false },
    ) catch return;
    defer file.close();

    file.seekFromEnd(0) catch return;
    file.writeAll(line) catch return;
    file.writeAll("\n") catch {};
}

// ============================================================
// Logging
// ============================================================

fn printHeader() void {
    api.log(
        "\nnnzap agent — LLM-powered experiment runner\n" ++
            "============================================" ++
            "\n\n",
        .{},
    );
}
