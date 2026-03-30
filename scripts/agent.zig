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
const TIMESTAMP_LEN: u32 = 20;

const API_URL: []const u8 =
    "https://api.anthropic.com/v1/messages";
const API_VERSION: []const u8 = "2023-06-01";
const DEFAULT_MODEL: []const u8 = "claude-opus-4-6";
const MAX_TOKENS_STR: []const u8 = "16384";

const TOOL_PATH: []const u8 = "./zig-out/bin/autoresearch";
const HISTORY_DIR: []const u8 = ".agent_history";
const HISTORY_PATH: []const u8 =
    ".agent_history/experiments.jsonl";

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
};

const ParsedContent = struct {
    text: []const u8,
    tool_calls: []const ToolCall,
};

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
        fatal(
            "Set ANTHROPIC_API_KEY env var.\n" ++
                "  export ANTHROPIC_API_KEY=sk-ant-...\n",
        );
        unreachable;
    };
    const model = loadModel();
    log("Model: {s}\n", .{model});

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

    var turn: u32 = 0;
    while (turn < MAX_TURNS) : (turn += 1) {
        log("\n--- Turn {d} ---\n", .{turn + 1});

        const resp = callApi(
            arena,
            api_key,
            model,
            messages[0..count],
        );
        if (!resp.success) {
            log("API error: {s}\n", .{resp.error_message});
            break;
        }
        if (resp.text.len > 0) logClaudeText(resp.text);

        messages[count] = buildAssistantMsg(
            arena,
            resp.content_json,
        );
        count += 1;

        // Stop when Claude finishes or has no tool calls.
        const is_tool_use = eql(
            resp.stop_reason,
            "tool_use",
        );
        if (!is_tool_use or resp.tool_calls.len == 0) break;

        const results = executeTools(
            arena,
            resp.tool_calls,
        );
        messages[count] = buildToolResultsMsg(
            arena,
            results,
        );
        count += 1;

        if (count + 2 >= MAX_MESSAGES) {
            log("Message limit reached.\n", .{});
            break;
        }
    }

    log("\nAgent finished ({d} turns).\n", .{turn + 1});
    saveRunLog(arena, messages[0..count]);
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
    log("Building toolbox...\n", .{});

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
        log("Build failed:\n{s}\n", .{result.stderr});
    } else {
        log("  done.\n", .{});
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

    const body = buildRequestJson(
        arena,
        model,
        messages,
    ) catch return errResp("request build failed");

    const uri = Uri.parse(API_URL) catch {
        return errResp("failed to parse API URL");
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
        return errResp("failed to open HTTP request");
    };
    defer req.deinit();

    // Content-Length lets the server know the full body
    // size up front — required for the Messages API.
    req.transfer_encoding = .{
        .content_length = body.len,
    };
    req.sendBodyComplete(body) catch {
        return errResp("failed to send request body");
    };

    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(
        &redirect_buf,
    ) catch {
        return errResp("failed to receive response");
    };

    if (response.head.status != .ok) {
        return errResp("API returned non-200 status");
    }

    var transfer_buf: [64]u8 = undefined;
    var reader = response.reader(&transfer_buf);

    const response_data = reader.allocRemaining(
        arena,
        std.Io.Limit.limited(MAX_API_RESPONSE),
    ) catch {
        return errResp("failed to read response body");
    };

    if (response_data.len == 0) {
        return errResp("empty API response");
    }

    return parseApiResponse(arena, response_data);
}

/// Build the full JSON request body for the Messages API.
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

/// Parse the raw API response JSON into an ApiResponse.
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
        return errResp(msg);
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

    const parsed = parseContentBlocks(arena, content);

    return .{
        .success = true,
        .stop_reason = stop,
        .content_json = content,
        .text = parsed.text,
        .tool_calls = parsed.tool_calls,
        .error_message = "",
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
        // Text block.
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

        // Tool use block.
        if (containsField(block, "\"type\"", "tool_use")) {
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
    const output = dispatchTool(arena, call);

    // Persist train results to the history log.
    if (eql(call.name, "train") and output.success) {
        appendExperiment(arena, output.stdout);
    }

    return .{
        .tool_use_id = call.id,
        .content = truncate(output.stdout, MAX_TOOL_OUTPUT),
        .is_error = !output.success,
    };
}

/// Route a tool call to the correct autoresearch command.
fn dispatchTool(
    arena: Allocator,
    call: ToolCall,
) ToolOutput {
    if (eql(call.name, "config_show")) {
        return callAutoresearch(arena, &.{"config-show"});
    }
    if (eql(call.name, "config_set")) {
        return executeConfigSet(arena, call.input_json);
    }
    if (eql(call.name, "config_backup")) {
        return callAutoresearch(
            arena,
            &.{"config-backup"},
        );
    }
    if (eql(call.name, "config_restore")) {
        return callAutoresearch(
            arena,
            &.{"config-restore"},
        );
    }
    if (eql(call.name, "train")) {
        return callAutoresearch(arena, &.{"train"});
    }
    if (eql(call.name, "benchmark_compare")) {
        return callAutoresearch(
            arena,
            &.{"benchmark-compare"},
        );
    }
    if (eql(call.name, "benchmark_latest")) {
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

/// Extract the settings string array from config_set
/// tool input JSON.
fn extractSettings(
    arena: Allocator,
    input_json: []const u8,
) ?[]const []const u8 {
    const arr = extractRawArray(
        input_json,
        "\"settings\"",
    ) orelse return null;
    return parseJsonStringArray(arena, arr) catch null;
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

    return wrapUserTextMessage(arena, text);
}

/// Wrap Claude's raw content JSON into an assistant message.
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
        "{\"type\":\"tool_result\",\"tool_use_id\":\"",
    ) catch return;
    buf.appendSlice(arena, r.tool_use_id) catch return;
    buf.appendSlice(arena, "\",") catch return;

    if (r.is_error) {
        buf.appendSlice(
            arena,
            "\"is_error\":true,",
        ) catch return;
    }

    buf.appendSlice(arena, "\"content\":") catch return;
    appendJsonString(arena, buf, r.content) catch return;
    buf.append(arena, '}') catch return;
}

/// Wrap a plain text string as a user message JSON object.
fn wrapUserTextMessage(
    arena: Allocator,
    text: []const u8,
) []const u8 {
    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"role\":\"user\",\"content\":",
    ) catch return "{}";
    appendJsonString(arena, &buf, text) catch return "{}";
    buf.append(arena, '}') catch return "{}";
    return buf.items;
}

// ============================================================
// History persistence
//
// Append-only JSONL log at .agent_history/experiments.jsonl.
// Each line is a compacted benchmark JSON from a train call.
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

/// Load the raw history file content.  Returns empty string
/// if the file does not exist.  Truncates to
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

/// Append a train benchmark result to the history file.
/// Compacts the JSON to a single line for JSONL format.
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
    return buf.items;
}

/// Save the full conversation as a JSON array of messages
/// to a timestamped run log file.
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
            buf.appendSlice(arena, ",\n") catch return;
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
        "\nnnzap agent — LLM-powered experiment runner\n" ++
            "============================================" ++
            "\n\n",
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

fn logToolCall(name: []const u8, input: []const u8) void {
    std.debug.assert(name.len > 0);

    if (input.len <= 2) {
        // Empty input like "{}" — omit it.
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
            '"' => try buf.appendSlice(arena, "\\\""),
            '\\' => try buf.appendSlice(arena, "\\\\"),
            '\n' => try buf.appendSlice(arena, "\\n"),
            '\r' => try buf.appendSlice(arena, "\\r"),
            '\t' => try buf.appendSlice(arena, "\\t"),
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

/// Extract a JSON string value after a needle that ends
/// with `:"`.  Handles escaped quotes inside the value.
fn extractJsonString(
    json: []const u8,
    needle: []const u8,
) ?[]const u8 {
    std.debug.assert(needle.len > 0);

    const idx = indexOf(json, needle) orelse return null;
    const start = idx + needle.len;
    if (start >= json.len) return null;
    return findStringEnd(json[start..]);
}

/// Scan forward past a JSON string body, stopping at the
/// first unescaped `"`.  Returns the string content.
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

/// Extract a raw JSON array value for a given field name.
/// The field is the quoted key (e.g., `"content"`).  Skips
/// the `:` and whitespace, then finds matching `[...]`.
fn extractRawArray(
    json: []const u8,
    field: []const u8,
) ?[]const u8 {
    std.debug.assert(field.len > 0);

    const idx = indexOf(json, field) orelse return null;
    var pos = idx + field.len;
    pos = skipColonAndWhitespace(json, pos);
    if (pos >= json.len) return null;
    if (json[pos] != '[') return null;
    return findMatchingBracket(json[pos..], '[', ']');
}

/// Extract a raw JSON object value for a given field name.
fn extractRawObject(
    json: []const u8,
    field: []const u8,
) ?[]const u8 {
    std.debug.assert(field.len > 0);

    const idx = indexOf(json, field) orelse return null;
    var pos = idx + field.len;
    pos = skipColonAndWhitespace(json, pos);
    if (pos >= json.len) return null;
    if (json[pos] != '{') return null;
    return findMatchingBracket(json[pos..], '{', '}');
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

/// Find the substring from `open` to its matching `close`,
/// respecting nesting and JSON string literals.
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
                if (depth == 0) return s[0 .. i + 1];
            }
        }
        i += 1;
    }
    return null;
}

/// Split a JSON array into its top-level object strings.
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
            if (c == ' ' or c == '\n' or c == '\r' or
                c == '\t' or c == ',')
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

/// Parse a JSON array of strings into a Zig slice.
fn parseJsonStringArray(
    arena: Allocator,
    json: []const u8,
) ![]const []const u8 {
    std.debug.assert(json.len >= 2);

    var items: std.ArrayList([]const u8) = .empty;
    var pos: usize = 0;
    while (pos < json.len) {
        // Find opening quote.
        const q1 = std.mem.indexOfScalarPos(
            u8,
            json,
            pos,
            '"',
        ) orelse break;
        const val_start = q1 + 1;
        // Find closing quote handling escapes.
        var q2 = val_start;
        while (q2 < json.len) {
            if (json[q2] == '\\') {
                q2 += 2;
                continue;
            }
            if (json[q2] == '"') break;
            q2 += 1;
        }
        if (q2 >= json.len) break;
        try items.append(arena, json[val_start..q2]);
        pos = q2 + 1;
    }
    return items.items;
}

/// Check if a JSON block has `"type":"<value>"` or
/// `"type": "<value>"`.
fn containsField(
    block: []const u8,
    key: []const u8,
    value: []const u8,
) bool {
    std.debug.assert(key.len > 0);
    std.debug.assert(value.len > 0);

    const idx = indexOf(block, key) orelse return false;
    var pos = idx + key.len;

    // Skip colon and whitespace.
    pos = skipColonAndWhitespace(block, pos);
    if (pos >= block.len) return false;

    // Expect opening quote.
    if (block[pos] != '"') return false;
    pos += 1;

    // Compare value.
    if (pos + value.len > block.len) return false;
    if (!std.mem.eql(u8, block[pos..][0..value.len], value)) {
        return false;
    }
    pos += value.len;

    // Expect closing quote.
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

fn truncate(s: []const u8, max: usize) []const u8 {
    std.debug.assert(max > 0);
    return if (s.len <= max) s else s[0..max];
}

fn trimTrailingNewline(s: []const u8) []const u8 {
    if (s.len == 0) return s;
    var end = s.len;
    if (end > 0 and s[end - 1] == '\n') end -= 1;
    if (end > 0 and s[end - 1] == '\r') end -= 1;
    return s[0..end];
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

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(content);
}

fn writeStdout(bytes: []const u8) !void {
    try stdout_file.writeAll(bytes);
}

// ============================================================
// Timestamp formatting
// ============================================================

fn formatTimestamp(
    buf: *[TIMESTAMP_LEN]u8,
    separator: u8,
) []const u8 {
    const secs: u64 = @intCast(std.time.timestamp());
    const es = std.time.epoch.EpochSeconds{ .secs = secs };
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
// Helpers
// ============================================================

fn errResp(msg: []const u8) ApiResponse {
    return .{
        .success = false,
        .stop_reason = "",
        .content_json = "[]",
        .text = "",
        .tool_calls = &.{},
        .error_message = msg,
    };
}
