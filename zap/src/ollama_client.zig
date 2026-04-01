// ============================================================
// ollama_client.zig — OpenAI-compatible API client for Ollama
//
// Translates between the shared ApiResponse/ToolCall types
// (defined in api_client.zig) and the OpenAI chat completions
// format that Ollama exposes.  Used by the inner experiment
// loop where a local Qwen 27B model executes plans designed
// by Opus.
// ============================================================

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("api_client.zig");

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const DEFAULT_URL: []const u8 =
    "http://localhost:1234/v1/chat/completions";
const OLLAMA_TIMEOUT_SECS_STR: []const u8 = "600";

/// Resolve the local LLM API URL.  Checks LOCAL_LLM_URL
/// env var first, falls back to LM Studio default.
fn resolveApiUrl() []const u8 {
    const val = std.posix.getenv("LOCAL_LLM_URL");
    if (val) |url| {
        std.debug.assert(url.len > 0);
        return url;
    }
    return DEFAULT_URL;
}
const MAX_API_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_TOOL_CALLS: u32 = 16;
const HISTORY_DIR: []const u8 =
    ".engine_agent_history";

comptime {
    std.debug.assert(MAX_API_RESPONSE > 0);
    std.debug.assert(MAX_TOOL_CALLS > 0);
}

// ============================================================
// Tool schema translation
// ============================================================

/// Convert Anthropic tool schemas to OpenAI function tool
/// format.  Anthropic uses `input_schema`; OpenAI wraps
/// each tool in `{"type":"function","function":{...}}` and
/// renames `input_schema` to `parameters`.
pub fn convertAnthropicTools(
    arena: Allocator,
    anthropic_tools: []const u8,
) ![]u8 {
    std.debug.assert(anthropic_tools.len > 2);

    const tool_objects = try api.splitTopLevelObjects(
        arena,
        anthropic_tools,
    );
    std.debug.assert(tool_objects.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    try buf.append(arena, '[');

    for (tool_objects, 0..) |tool, i| {
        if (i > 0) try buf.append(arena, ',');

        // Extract fields from the Anthropic tool object.
        const name = api.extractJsonString(
            tool,
            "\"name\":\"",
        ) orelse api.extractJsonString(
            tool,
            "\"name\": \"",
        ) orelse return error.MissingToolName;

        const description = api.extractJsonString(
            tool,
            "\"description\":\"",
        ) orelse api.extractJsonString(
            tool,
            "\"description\": \"",
        ) orelse "";

        const parameters = api.extractRawObject(
            tool,
            "\"input_schema\"",
        ) orelse "{}";

        // Build OpenAI format:
        // {"type":"function","function":{"name":"...",
        //   "description":"...","parameters":{...}}}
        try buf.appendSlice(
            arena,
            "{\"type\":\"function\",\"function\":{",
        );
        try buf.appendSlice(arena, "\"name\":\"");
        try buf.appendSlice(arena, name);
        try buf.appendSlice(arena, "\",\"description\":");
        try api.appendJsonString(arena, &buf, description);
        try buf.appendSlice(arena, ",\"parameters\":");
        try buf.appendSlice(arena, parameters);
        try buf.appendSlice(arena, "}}");
    }

    try buf.append(arena, ']');
    return buf.items;
}

// ============================================================
// Request building
// ============================================================

/// Build an OpenAI chat completions request body.
/// The system prompt is prepended as the first message.
/// `messages` is a slice of pre-built OpenAI-format message
/// JSON strings (without the system message).
pub fn buildRequestJson(
    arena: Allocator,
    model: []const u8,
    system_prompt: []const u8,
    messages: []const []const u8,
    tools_json: []const u8,
    max_tokens_str: []const u8,
) ![]u8 {
    std.debug.assert(model.len > 0);
    std.debug.assert(messages.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\"model\":\"");
    try buf.appendSlice(arena, model);
    try buf.appendSlice(arena, "\",\"max_tokens\":");
    try buf.appendSlice(arena, max_tokens_str);
    try buf.appendSlice(arena, ",\"temperature\":0.6");
    try buf.appendSlice(arena, ",\"stream\":false");
    try buf.appendSlice(arena, ",\"messages\":[");

    // System message first.
    try buf.appendSlice(
        arena,
        "{\"role\":\"system\",\"content\":",
    );
    try api.appendJsonString(arena, &buf, system_prompt);
    try buf.append(arena, '}');

    // Append conversation messages.
    for (messages) |msg| {
        try buf.append(arena, ',');
        try buf.appendSlice(arena, msg);
    }

    try buf.append(arena, ']');

    // Append tools if provided.
    if (tools_json.len > 2) {
        try buf.appendSlice(arena, ",\"tools\":");
        try buf.appendSlice(arena, tools_json);
    }

    try buf.appendSlice(arena, "}");
    return buf.items;
}

// ============================================================
// API calling
// ============================================================

/// Call the Ollama API via curl.  No API key needed —
/// Ollama runs locally.
pub fn callApi(
    arena: Allocator,
    model: []const u8,
    system_prompt: []const u8,
    messages: []const []const u8,
    tools_json: []const u8,
    max_tokens_str: []const u8,
    history_dir: []const u8,
) api.ApiResponse {
    std.debug.assert(model.len > 0);
    std.debug.assert(messages.len > 0);

    const body = buildRequestJson(
        arena,
        model,
        system_prompt,
        messages,
        tools_json,
        max_tokens_str,
    ) catch return api.errResp(
        "ollama request build failed",
        false,
    );

    // Save request for debugging.
    const request_path = std.fmt.allocPrint(
        arena,
        "{s}/_ollama_request.json",
        .{history_dir},
    ) catch return api.errResp(
        "failed to format request path",
        false,
    );
    api.writeFile(request_path, body) catch {};
    api.log(
        "  Ollama request: {d} KB\n",
        .{body.len / 1024},
    );

    // Build curl data argument referencing the file.
    const data_arg = std.fmt.allocPrint(
        arena,
        "@{s}",
        .{request_path},
    ) catch return api.errResp(
        "failed to format curl arg",
        false,
    );

    api.log("  Waiting for Ollama response...\n", .{});

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{
            "curl",
            "-s",
            "--connect-timeout",
            "30",
            "--max-time",
            OLLAMA_TIMEOUT_SECS_STR,
            "-X",
            "POST",
            resolveApiUrl(),
            "-H",
            "Content-Type: application/json",
            "-d",
            data_arg,
            "-w",
            "\n%{http_code}",
        },
        .max_output_bytes = MAX_API_RESPONSE,
    }) catch |err| {
        const msg = std.fmt.allocPrint(
            arena,
            "curl spawn failed: {s}",
            .{@errorName(err)},
        ) catch "curl spawn failed";
        return api.errResp(msg, true);
    };

    // Check curl process exit.
    const curl_ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };

    if (!curl_ok) {
        const detail = if (result.stderr.len > 0)
            api.truncate(result.stderr, 500)
        else
            "unknown curl error";
        const msg = std.fmt.allocPrint(
            arena,
            "ollama curl failed: {s}",
            .{detail},
        ) catch "ollama curl failed";
        return api.errResp(msg, true);
    }

    const output = result.stdout;
    if (output.len < 4) {
        return api.errResp(
            "empty ollama output",
            true,
        );
    }

    // Split body from HTTP status code (curl -w).
    const last_nl = std.mem.lastIndexOfScalar(
        u8,
        output,
        '\n',
    ) orelse {
        return api.errResp(
            "malformed ollama output (no status line)",
            true,
        );
    };

    const response_data = output[0..last_nl];
    const status_str = std.mem.trim(
        u8,
        output[last_nl + 1 ..],
        " \r\n",
    );
    const status_code = std.fmt.parseInt(
        u16,
        status_str,
        10,
    ) catch 0;

    if (status_code == 0) {
        return api.errResp(
            "failed to parse HTTP status from ollama",
            true,
        );
    }

    api.log(
        "  Ollama response: {d} KB (HTTP {d})\n",
        .{ response_data.len / 1024, status_code },
    );

    if (status_code != 200) {
        // Extract error message if present.
        const err_msg = api.extractJsonString(
            response_data,
            "\"message\":\"",
        ) orelse api.truncate(response_data, 500);
        const msg = std.fmt.allocPrint(
            arena,
            "ollama HTTP {d}: {s}",
            .{ status_code, err_msg },
        ) catch "ollama HTTP error";
        const retryable = (status_code >= 500);
        return api.errResp(msg, retryable);
    }

    if (response_data.len == 0) {
        return api.errResp(
            "empty ollama response body",
            true,
        );
    }

    // Persist raw response for debugging.
    const response_path = std.fmt.allocPrint(
        arena,
        "{s}/_ollama_response.json",
        .{history_dir},
    ) catch "";
    if (response_path.len > 0) {
        api.writeFile(response_path, response_data) catch {};
    }

    return parseApiResponse(arena, response_data);
}

// ============================================================
// Response parsing
// ============================================================

/// Parse an OpenAI chat completions response into our
/// shared ApiResponse type.
///
/// OpenAI format:
/// {
///   "choices": [{
///     "message": {
///       "role": "assistant",
///       "content": "...",
///       "tool_calls": [{
///         "id": "call_abc",
///         "type": "function",
///         "function": {"name":"...", "arguments":"{}"}
///       }]
///     },
///     "finish_reason": "stop"
///   }],
///   "usage": {"prompt_tokens":100,"completion_tokens":50}
/// }
pub fn parseApiResponse(
    arena: Allocator,
    raw: []const u8,
) api.ApiResponse {
    std.debug.assert(raw.len > 0);

    // Check for error responses.
    if (api.indexOf(raw, "\"error\"") != null) {
        const msg = api.extractJsonString(
            raw,
            "\"message\":\"",
        ) orelse "unknown ollama error";
        return api.errResp(msg, false);
    }

    // Extract the choices array, then the first choice.
    const choices_arr = api.extractRawArray(
        raw,
        "\"choices\"",
    ) orelse {
        return api.errResp(
            "no choices in ollama response",
            false,
        );
    };

    const choice_objects = api.splitTopLevelObjects(
        arena,
        choices_arr,
    ) catch {
        return api.errResp(
            "failed to parse choices array",
            false,
        );
    };

    if (choice_objects.len == 0) {
        return api.errResp(
            "empty choices array",
            false,
        );
    }

    const choice = choice_objects[0];

    // Extract finish_reason → stop_reason.
    const raw_reason = api.extractJsonString(
        choice,
        "\"finish_reason\":\"",
    ) orelse api.extractJsonString(
        choice,
        "\"finish_reason\": \"",
    ) orelse "unknown";

    // Map OpenAI reasons to Anthropic equivalents so
    // the engine_agent loop logic works unchanged.
    const stop_reason = mapFinishReason(raw_reason);

    // Extract the message object.
    const message_obj = api.extractRawObject(
        choice,
        "\"message\"",
    ) orelse {
        return api.errResp(
            "no message in first choice",
            false,
        );
    };

    // Extract text content.  In OpenAI format, content
    // is a plain string (or null when tool_calls present).
    // Reasoning models (e.g. Qwen reasoning-distilled)
    // may put all output in reasoning_content and leave
    // content empty — fall back to reasoning_content so
    // the executor's thinking is not silently lost.
    const raw_text = api.extractJsonString(
        message_obj,
        "\"content\":\"",
    ) orelse api.extractJsonString(
        message_obj,
        "\"content\": \"",
    ) orelse "";

    const text = if (raw_text.len == 0)
        api.extractJsonString(
            message_obj,
            "\"reasoning_content\":\"",
        ) orelse api.extractJsonString(
            message_obj,
            "\"reasoning_content\": \"",
        ) orelse ""
    else
        raw_text;

    // Extract tool_calls if present.
    const tool_calls = parseToolCalls(
        arena,
        message_obj,
    );

    // Extract token usage.
    const input_tokens = parseTokenField(
        raw,
        "\"prompt_tokens\":",
    );
    const output_tokens = parseTokenField(
        raw,
        "\"completion_tokens\":",
    );

    // Build content_json as an Anthropic-style content
    // array for compatibility with saveRunLog.  This is
    // a best-effort reconstruction — the run log is for
    // debugging, not round-tripping.
    const content_json = buildContentJson(
        arena,
        text,
        tool_calls,
    );

    return .{
        .success = true,
        .stop_reason = stop_reason,
        .content_json = content_json,
        .text = text,
        .tool_calls = tool_calls,
        .error_message = "",
        .retryable = false,
        .input_tokens = input_tokens,
        .output_tokens = output_tokens,
    };
}

/// Map OpenAI finish reasons to the Anthropic equivalents
/// that engine_agent checks.
fn mapFinishReason(reason: []const u8) []const u8 {
    // "stop" in OpenAI means the model chose to stop.
    if (api.eql(reason, "stop")) return "end_turn";
    // "tool_calls" means the model wants to call tools.
    if (api.eql(reason, "tool_calls")) return "tool_use";
    // "length" means max tokens hit.
    if (api.eql(reason, "length")) return "max_tokens";
    return reason;
}

/// Parse the tool_calls array from an OpenAI message.
/// Format: [{"id":"call_abc","type":"function",
///           "function":{"name":"...","arguments":"{}"}}]
fn parseToolCalls(
    arena: Allocator,
    message_json: []const u8,
) []const api.ToolCall {
    std.debug.assert(message_json.len > 0);

    const tc_array = api.extractRawArray(
        message_json,
        "\"tool_calls\"",
    ) orelse return &.{};

    const tc_objects = api.splitTopLevelObjects(
        arena,
        tc_array,
    ) catch return &.{};

    if (tc_objects.len == 0) return &.{};

    var calls: [MAX_TOOL_CALLS]api.ToolCall = undefined;
    var count: u32 = 0;

    for (tc_objects) |tc| {
        if (count >= MAX_TOOL_CALLS) break;

        // Extract id.
        const id = api.extractJsonString(
            tc,
            "\"id\":\"",
        ) orelse api.extractJsonString(
            tc,
            "\"id\": \"",
        ) orelse continue;

        // Extract function object.  Search for
        // "function": (with colon) to avoid matching
        // the value in "type": "function".
        const func_obj = api.extractRawObject(
            tc,
            "\"function\":",
        ) orelse api.extractRawObject(
            tc,
            "\"function\" :",
        ) orelse continue;

        // Extract name and arguments from function.
        const name = api.extractJsonString(
            func_obj,
            "\"name\":\"",
        ) orelse api.extractJsonString(
            func_obj,
            "\"name\": \"",
        ) orelse continue;

        // Arguments is a JSON string containing the
        // tool input.  In OpenAI format it's a string
        // value, but we need to extract the raw JSON.
        const args = extractArgumentsJson(
            func_obj,
        ) orelse "{}";

        calls[count] = .{
            .id = id,
            .name = name,
            .input_json = args,
        };
        count += 1;
    }

    const result = arena.alloc(
        api.ToolCall,
        count,
    ) catch return &.{};
    @memcpy(result, calls[0..count]);
    return result;
}

/// Extract the "arguments" field from a function object.
/// OpenAI serialises arguments as a JSON string (escaped),
/// e.g. `"arguments":"{\"path\":\"nn/src/metal.zig\"}"`.
/// We need to extract and unescape it to get raw JSON.
/// Some Ollama models also emit arguments as a raw object,
/// so we try both.
fn extractArgumentsJson(
    func_json: []const u8,
) ?[]const u8 {
    std.debug.assert(func_json.len > 0);

    // First try: raw object (some local models do this).
    if (api.extractRawObject(
        func_json,
        "\"arguments\"",
    )) |obj| {
        return obj;
    }

    // Second try: JSON string value.
    return api.extractJsonString(
        func_json,
        "\"arguments\":\"",
    ) orelse api.extractJsonString(
        func_json,
        "\"arguments\": \"",
    );
}

/// Parse a token count field from the usage object.
fn parseTokenField(
    raw: []const u8,
    needle: []const u8,
) u32 {
    const str = api.extractJsonNumber(
        raw,
        needle,
    ) orelse return 0;
    return std.fmt.parseInt(u32, str, 10) catch 0;
}

/// Build an Anthropic-style content JSON array from text
/// and tool calls.  Used for run log compatibility.
fn buildContentJson(
    arena: Allocator,
    text: []const u8,
    tool_calls: []const api.ToolCall,
) []const u8 {
    var buf: std.ArrayList(u8) = .empty;
    buf.append(arena, '[') catch return "[]";

    var has_item = false;

    if (text.len > 0) {
        buf.appendSlice(
            arena,
            "{\"type\":\"text\",\"text\":",
        ) catch return "[]";
        api.appendJsonString(
            arena,
            &buf,
            text,
        ) catch return "[]";
        buf.append(arena, '}') catch return "[]";
        has_item = true;
    }

    for (tool_calls) |tc| {
        if (has_item) {
            buf.append(arena, ',') catch continue;
        }
        buf.appendSlice(
            arena,
            "{\"type\":\"tool_use\",\"id\":\"",
        ) catch continue;
        buf.appendSlice(arena, tc.id) catch continue;
        buf.appendSlice(
            arena,
            "\",\"name\":\"",
        ) catch continue;
        buf.appendSlice(arena, tc.name) catch continue;
        buf.appendSlice(
            arena,
            "\",\"input\":",
        ) catch continue;
        buf.appendSlice(
            arena,
            tc.input_json,
        ) catch continue;
        buf.append(arena, '}') catch continue;
        has_item = true;
    }

    buf.append(arena, ']') catch return "[]";
    return buf.items;
}

// ============================================================
// Message builders (OpenAI format)
// ============================================================

/// Build an OpenAI-format assistant message from an
/// ApiResponse.  Includes tool_calls if present.
pub fn buildAssistantMsg(
    arena: Allocator,
    resp: api.ApiResponse,
) []const u8 {
    std.debug.assert(resp.success);

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"role\":\"assistant\"",
    ) catch return "{}";

    // Content field — null when only tool calls.
    if (resp.text.len > 0) {
        buf.appendSlice(arena, ",\"content\":") catch
            return "{}";
        api.appendJsonString(
            arena,
            &buf,
            resp.text,
        ) catch return "{}";
    } else {
        buf.appendSlice(
            arena,
            ",\"content\":null",
        ) catch return "{}";
    }

    // Tool calls array if present.
    if (resp.tool_calls.len > 0) {
        buf.appendSlice(
            arena,
            ",\"tool_calls\":[",
        ) catch return "{}";

        for (resp.tool_calls, 0..) |tc, i| {
            if (i > 0) buf.append(arena, ',') catch {};
            buf.appendSlice(
                arena,
                "{\"id\":\"",
            ) catch continue;
            buf.appendSlice(arena, tc.id) catch continue;
            buf.appendSlice(
                arena,
                "\",\"type\":\"function\"," ++
                    "\"function\":{\"name\":\"",
            ) catch continue;
            buf.appendSlice(arena, tc.name) catch continue;
            buf.appendSlice(
                arena,
                "\",\"arguments\":",
            ) catch continue;
            api.appendJsonString(
                arena,
                &buf,
                tc.input_json,
            ) catch continue;
            buf.appendSlice(arena, "}}") catch continue;
        }

        buf.append(arena, ']') catch {};
    }

    buf.append(arena, '}') catch return "{}";
    return buf.items;
}

/// Build an OpenAI-format tool result message.
/// One message per tool result (unlike Anthropic which
/// groups them).
pub fn buildToolResultMsg(
    arena: Allocator,
    result: api.ToolResult,
) []const u8 {
    std.debug.assert(result.tool_use_id.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"role\":\"tool\",\"tool_call_id\":\"",
    ) catch return "{}";
    buf.appendSlice(
        arena,
        result.tool_use_id,
    ) catch return "{}";
    buf.appendSlice(arena, "\",\"content\":") catch
        return "{}";
    api.appendJsonString(
        arena,
        &buf,
        result.content,
    ) catch return "{}";
    buf.append(arena, '}') catch return "{}";
    return buf.items;
}

/// Build a simple user text message.
pub fn buildUserTextMsg(
    arena: Allocator,
    text: []const u8,
) []const u8 {
    std.debug.assert(text.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"role\":\"user\",\"content\":",
    ) catch return "{}";
    api.appendJsonString(
        arena,
        &buf,
        text,
    ) catch return "{}";
    buf.append(arena, '}') catch return "{}";
    return buf.items;
}
