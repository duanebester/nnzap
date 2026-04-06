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
const json = std.json;
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

comptime {
    std.debug.assert(MAX_API_RESPONSE > 0);
    std.debug.assert(MAX_TOOL_CALLS > 0);
}

// ============================================================
// JSON Value navigation helpers — imported from api_client.
// ============================================================

const getStr = api.getStr;
const getInt = api.getInt;
const getObj = api.getObj;
const getArr = api.getArr;

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

    const parsed = try json.parseFromSliceLeaky(
        json.Value,
        arena,
        anthropic_tools,
        .{},
    );

    const tools = switch (parsed) {
        .array => |a| a.items,
        else => return error.InvalidToolsFormat,
    };
    std.debug.assert(tools.len > 0);

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginArray();
    for (tools) |tool_val| {
        const tool = switch (tool_val) {
            .object => |o| o,
            else => continue,
        };
        try writeOpenAITool(&w, &out, arena, tool);
    }
    try w.endArray();

    return out.written();
}

/// Write a single tool in OpenAI function-calling format.
fn writeOpenAITool(
    w: *json.Stringify,
    out: *std.io.Writer.Allocating,
    arena: Allocator,
    tool: json.ObjectMap,
) !void {
    const name = getStr(tool, "name") orelse
        return error.MissingToolName;
    const description = getStr(
        tool,
        "description",
    ) orelse "";
    std.debug.assert(name.len > 0);

    try w.beginObject();
    try w.objectField("type");
    try w.write("function");
    try w.objectField("function");
    try w.beginObject();
    try w.objectField("name");
    try w.write(name);
    try w.objectField("description");
    try w.write(description);
    try w.objectField("parameters");

    // Write input_schema as raw JSON if present.
    if (tool.get("input_schema")) |schema| {
        const raw: []const u8 =
            json.Stringify.valueAlloc(
                arena,
                schema,
                .{},
            ) catch "{}";
        try w.beginWriteRaw();
        try out.writer.writeAll(raw);
        w.endWriteRaw();
    } else {
        try w.beginWriteRaw();
        try out.writer.writeAll("{}");
        w.endWriteRaw();
    }

    try w.endObject(); // Close function.
    try w.endObject(); // Close tool wrapper.
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

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginObject();
    try w.objectField("model");
    try w.write(model);
    try w.objectField("max_tokens");
    try w.beginWriteRaw();
    try out.writer.writeAll(max_tokens_str);
    w.endWriteRaw();
    try w.objectField("temperature");
    try w.beginWriteRaw();
    try out.writer.writeAll("0.6");
    w.endWriteRaw();
    try w.objectField("stream");
    try w.write(false);

    try writeRequestMessages(
        &w,
        &out,
        system_prompt,
        messages,
        tools_json,
    );

    return out.written();
}

/// Write messages, tools, and close the request object.
fn writeRequestMessages(
    w: *json.Stringify,
    out: *std.io.Writer.Allocating,
    system_prompt: []const u8,
    messages: []const []const u8,
    tools_json: []const u8,
) !void {
    std.debug.assert(system_prompt.len > 0);
    std.debug.assert(messages.len > 0);

    try w.objectField("messages");
    try w.beginArray();

    // System message first.
    try w.beginObject();
    try w.objectField("role");
    try w.write("system");
    try w.objectField("content");
    try w.write(system_prompt);
    try w.endObject();

    // Append pre-built conversation messages as raw JSON.
    for (messages) |msg| {
        try w.beginWriteRaw();
        try out.writer.writeAll(msg);
        w.endWriteRaw();
    }

    try w.endArray();

    // Append tools if provided.
    if (tools_json.len > 2) {
        try w.objectField("tools");
        try w.beginWriteRaw();
        try out.writer.writeAll(tools_json);
        w.endWriteRaw();
    }

    try w.endObject();
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

    return executeCurlAndProcess(
        arena,
        body,
        history_dir,
    );
}

/// Check curl exit status. Returns an error response
/// if curl failed, or null on success.
fn checkCurlExit(
    arena: Allocator,
    result: std.process.Child.RunResult,
) ?api.ApiResponse {
    const curl_ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (curl_ok) return null;

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

/// Save the request, execute curl, and hand off to
/// response processing.
fn executeCurlAndProcess(
    arena: Allocator,
    body: []const u8,
    history_dir: []const u8,
) api.ApiResponse {
    std.debug.assert(body.len > 0);
    std.debug.assert(history_dir.len > 0);
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

    if (checkCurlExit(arena, result)) |err| {
        return err;
    }
    return processRawOutput(
        arena,
        result.stdout,
        history_dir,
    );
}

/// Split the HTTP status code from the curl output, handle
/// non-200 responses, and parse the successful body.
fn processRawOutput(
    arena: Allocator,
    output: []const u8,
    history_dir: []const u8,
) api.ApiResponse {
    std.debug.assert(history_dir.len > 0);

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
        return handleNon200(
            arena,
            response_data,
            status_code,
        );
    }

    if (response_data.len == 0) {
        return api.errResp(
            "empty ollama response body",
            true,
        );
    }

    // Persist raw response for debugging.
    saveDebugResponse(arena, history_dir, response_data);

    return parseApiResponse(arena, response_data);
}

/// Build an error ApiResponse for a non-200 HTTP status.
fn handleNon200(
    arena: Allocator,
    body: []const u8,
    status: u16,
) api.ApiResponse {
    std.debug.assert(status != 200);
    std.debug.assert(status > 0);

    const err_msg = if (body.len > 0)
        extractErrorMsg(arena, body)
    else
        "empty error response";

    const msg = std.fmt.allocPrint(
        arena,
        "ollama HTTP {d}: {s}",
        .{ status, err_msg },
    ) catch "ollama HTTP error";
    return api.errResp(msg, status >= 500);
}

/// Try to extract an error message from a JSON error body.
/// Falls back to truncated raw text if parsing fails.
fn extractErrorMsg(
    arena: Allocator,
    data: []const u8,
) []const u8 {
    std.debug.assert(data.len > 0);

    const parsed = json.parseFromSliceLeaky(
        json.Value,
        arena,
        data,
        .{},
    ) catch return api.truncate(data, 500);

    const obj = switch (parsed) {
        .object => |o| o,
        else => return api.truncate(data, 500),
    };

    // Try top-level "message", then nested "error.message".
    if (getStr(obj, "message")) |msg| {
        return msg;
    }
    if (getObj(obj, "error")) |err| {
        if (getStr(err, "message")) |msg| return msg;
    }
    return api.truncate(data, 500);
}

/// Persist the raw response JSON for debugging.
fn saveDebugResponse(
    arena: Allocator,
    history_dir: []const u8,
    body: []const u8,
) void {
    std.debug.assert(history_dir.len > 0);
    std.debug.assert(body.len > 0);

    const path = std.fmt.allocPrint(
        arena,
        "{s}/_ollama_response.json",
        .{history_dir},
    ) catch return;
    api.writeFile(path, body) catch {};
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
///   "usage": {
///     "prompt_tokens": 100,
///     "completion_tokens": 50
///   }
/// }
pub fn parseApiResponse(
    arena: Allocator,
    raw: []const u8,
) api.ApiResponse {
    std.debug.assert(raw.len > 0);
    std.debug.assert(raw.len <= MAX_API_RESPONSE);

    const root = json.parseFromSliceLeaky(
        json.Value,
        arena,
        raw,
        .{},
    ) catch return api.errResp(
        "invalid JSON from ollama",
        false,
    );

    const obj = switch (root) {
        .object => |o| o,
        else => return api.errResp(
            "expected JSON object from ollama",
            false,
        ),
    };

    // Check for error field in the response.
    if (obj.get("error")) |err_val| {
        switch (err_val) {
            .string => |s| {
                if (s.len > 0) {
                    return api.errResp(s, false);
                }
            },
            .object => |e| {
                const msg = getStr(
                    e,
                    "message",
                ) orelse "unknown ollama error";
                return api.errResp(msg, false);
            },
            else => {},
        }
    }

    return buildOllamaSuccess(arena, obj);
}

/// Extract the first choice from the choices array.
/// Returns the choice object, or null if missing.
fn extractFirstChoice(
    obj: json.ObjectMap,
) ?json.ObjectMap {
    const choices = getArr(obj, "choices") orelse {
        return null;
    };
    if (choices.len == 0) return null;

    return switch (choices[0]) {
        .object => |o| o,
        else => null,
    };
}

/// Build a successful ApiResponse from the parsed choices,
/// message content, tool calls, and token usage.
fn buildOllamaSuccess(
    arena: Allocator,
    obj: json.ObjectMap,
) api.ApiResponse {
    const choice = extractFirstChoice(obj) orelse {
        return api.errResp(
            "no valid choice in ollama response",
            false,
        );
    };

    // finish_reason → stop_reason.
    const raw_reason = getStr(
        choice,
        "finish_reason",
    ) orelse "unknown";
    const stop_reason = mapFinishReason(raw_reason);

    // Message object.
    const message = getObj(choice, "message") orelse
        return api.errResp(
            "no message in first choice",
            false,
        );

    const text = extractMessageText(message);
    const tool_calls = parseToolCalls(arena, message);

    // Token usage.
    const usage = getObj(obj, "usage");
    const input_tokens = parseTokenField(
        usage,
        "prompt_tokens",
    );
    const output_tokens = parseTokenField(
        usage,
        "completion_tokens",
    );

    // Build Anthropic-style content_json for run log.
    const content_json = buildContentJson(
        arena,
        text,
        tool_calls,
    );
    std.debug.assert(stop_reason.len > 0);

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

/// Extract text content from an OpenAI message object.
/// Falls back to reasoning_content for reasoning models
/// like Qwen that put all output there and leave content
/// empty.
fn extractMessageText(
    message: json.ObjectMap,
) []const u8 {
    const content = getStr(message, "content");
    if (content) |text| {
        if (text.len > 0) return text;
    }

    // Reasoning models may use reasoning_content instead.
    return getStr(
        message,
        "reasoning_content",
    ) orelse "";
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
    message: json.ObjectMap,
) []const api.ToolCall {
    const tc_items = getArr(
        message,
        "tool_calls",
    ) orelse return &.{};
    if (tc_items.len == 0) return &.{};

    var calls: [MAX_TOOL_CALLS]api.ToolCall = undefined;
    var count: u32 = 0;

    for (tc_items) |tc_val| {
        if (count >= MAX_TOOL_CALLS) break;
        const tc = switch (tc_val) {
            .object => |o| o,
            else => continue,
        };

        const id = getStr(tc, "id") orelse continue;
        const func = getObj(
            tc,
            "function",
        ) orelse continue;
        const name = getStr(
            func,
            "name",
        ) orelse continue;
        const args = extractArgumentsJson(
            arena,
            func,
        ) orelse "{}";

        calls[count] = .{
            .id = id,
            .name = name,
            .input_json = args,
        };
        count += 1;
    }

    std.debug.assert(count <= MAX_TOOL_CALLS);

    const result = arena.alloc(
        api.ToolCall,
        count,
    ) catch return &.{};
    @memcpy(result, calls[0..count]);
    std.debug.assert(result.len == count);
    return result;
}

/// Extract the "arguments" field from a function object.
/// OpenAI serialises arguments as a JSON string (escaped),
/// e.g. `"arguments":"{\"path\":\"nn/src/metal.zig\"}"`.
/// Some Ollama models also emit arguments as a raw object,
/// so we handle both.
fn extractArgumentsJson(
    arena: Allocator,
    func: json.ObjectMap,
) ?[]const u8 {
    const val = func.get("arguments") orelse
        return null;
    return switch (val) {
        // String value: the unescaped content IS the
        // raw JSON arguments.
        .string => |s| s,
        // Object value: re-serialize to a JSON string.
        .object => json.Stringify.valueAlloc(
            arena,
            val,
            .{},
        ) catch null,
        else => null,
    };
}

/// Parse a token count from the usage object.
fn parseTokenField(
    usage: ?json.ObjectMap,
    key: []const u8,
) u32 {
    std.debug.assert(key.len > 0);
    const obj = usage orelse return 0;
    const val = getInt(obj, key);
    return if (val >= 0) @intCast(val) else 0;
}

// ============================================================
// Content JSON building
// ============================================================

/// Build an Anthropic-style content JSON array from text
/// and tool calls.  Used for run log compatibility.
fn buildContentJson(
    arena: Allocator,
    text: []const u8,
    tool_calls: []const api.ToolCall,
) []const u8 {
    return buildContentJsonInner(
        arena,
        text,
        tool_calls,
    ) catch "[]";
}

/// Inner builder that can propagate errors.
fn buildContentJsonInner(
    arena: Allocator,
    text: []const u8,
    tool_calls: []const api.ToolCall,
) ![]const u8 {
    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginArray();

    if (text.len > 0) {
        try w.beginObject();
        try w.objectField("type");
        try w.write("text");
        try w.objectField("text");
        try w.write(text);
        try w.endObject();
    }

    for (tool_calls) |tc| {
        try writeToolUseBlock(&w, &out, tc);
    }

    try w.endArray();

    const result = out.written();
    std.debug.assert(result.len >= 2); // At least "[]".
    std.debug.assert(result[0] == '[');
    return result;
}

/// Write a single tool_use content block.
fn writeToolUseBlock(
    w: *json.Stringify,
    out: *std.io.Writer.Allocating,
    tc: api.ToolCall,
) !void {
    std.debug.assert(tc.id.len > 0);
    std.debug.assert(tc.name.len > 0);

    try w.beginObject();
    try w.objectField("type");
    try w.write("tool_use");
    try w.objectField("id");
    try w.write(tc.id);
    try w.objectField("name");
    try w.write(tc.name);
    try w.objectField("input");
    // input_json is already valid JSON — write raw.
    try w.beginWriteRaw();
    try out.writer.writeAll(tc.input_json);
    w.endWriteRaw();
    try w.endObject();
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
    return buildAssistantMsgInner(
        arena,
        resp,
    ) catch "{}";
}

/// Inner builder for buildAssistantMsg.
fn buildAssistantMsgInner(
    arena: Allocator,
    resp: api.ApiResponse,
) ![]const u8 {
    std.debug.assert(resp.success);

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginObject();
    try w.objectField("role");
    try w.write("assistant");

    // Content field — null when only tool calls.
    try w.objectField("content");
    if (resp.text.len > 0) {
        try w.write(resp.text);
    } else {
        try w.beginWriteRaw();
        try out.writer.writeAll("null");
        w.endWriteRaw();
    }

    // Tool calls array if present.
    if (resp.tool_calls.len > 0) {
        try w.objectField("tool_calls");
        try w.beginArray();
        for (resp.tool_calls) |tc| {
            try writeAssistantToolCall(&w, tc);
        }
        try w.endArray();
    }

    try w.endObject();

    const result = out.written();
    std.debug.assert(result.len > 2);
    return result;
}

/// Write a single tool call in an assistant message.
/// In OpenAI format, arguments is a JSON string (the
/// raw JSON is escaped inside a quoted string value).
fn writeAssistantToolCall(
    w: *json.Stringify,
    tc: api.ToolCall,
) !void {
    std.debug.assert(tc.id.len > 0);
    std.debug.assert(tc.name.len > 0);

    try w.beginObject();
    try w.objectField("id");
    try w.write(tc.id);
    try w.objectField("type");
    try w.write("function");
    try w.objectField("function");
    try w.beginObject();
    try w.objectField("name");
    try w.write(tc.name);
    try w.objectField("arguments");
    // Stringify quotes and escapes the inner JSON string.
    try w.write(tc.input_json);
    try w.endObject(); // Close function.
    try w.endObject(); // Close tool call.
}

/// Build an OpenAI-format tool result message.
/// One message per tool result (unlike Anthropic which
/// groups them).
pub fn buildToolResultMsg(
    arena: Allocator,
    result: api.ToolResult,
) []const u8 {
    std.debug.assert(result.tool_use_id.len > 0);
    return buildToolResultMsgInner(
        arena,
        result,
    ) catch "{}";
}

/// Inner builder for buildToolResultMsg.
fn buildToolResultMsgInner(
    arena: Allocator,
    result: api.ToolResult,
) ![]const u8 {
    std.debug.assert(result.tool_use_id.len > 0);

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginObject();
    try w.objectField("role");
    try w.write("tool");
    try w.objectField("tool_call_id");
    try w.write(result.tool_use_id);
    try w.objectField("content");
    try w.write(result.content);
    try w.endObject();

    const written = out.written();
    std.debug.assert(written.len > 0);
    return written;
}

/// Build a simple user text message.
pub fn buildUserTextMsg(
    arena: Allocator,
    text: []const u8,
) []const u8 {
    std.debug.assert(text.len > 0);
    return buildUserTextMsgInner(
        arena,
        text,
    ) catch "{}";
}

/// Inner builder for buildUserTextMsg.
fn buildUserTextMsgInner(
    arena: Allocator,
    text: []const u8,
) ![]const u8 {
    std.debug.assert(text.len > 0);

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginObject();
    try w.objectField("role");
    try w.write("user");
    try w.objectField("content");
    try w.write(text);
    try w.endObject();

    const result = out.written();
    std.debug.assert(result.len > text.len);
    return result;
}
