//! api_client.zig — shared Anthropic API client infrastructure.
//!
//! Common types, JSON building, response parsing, logging,
//! and timestamp formatting used by both `agent.zig` and
//! `engine_agent.zig`.  Each agent imports this module and
//! provides its own system prompt, tool schemas, tool
//! dispatch, and API transport (std.http vs curl).
//!
//! JSON parsing uses `std.json.parseFromSliceLeaky` to
//! navigate Value trees.  JSON writing uses
//! `std.json.Stringify` for structured output.
//!
//! Design: all functions are pure or take explicit
//! parameters — no module-level mutable state.  Agent-
//! specific constants (system prompts, tool schemas, paths)
//! are passed in by the caller.

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

pub const TIMESTAMP_LEN: u32 = 20;
pub const MAX_TOOL_CALLS: u32 = 16;

// Compile-time validation (Rule 14).
comptime {
    std.debug.assert(TIMESTAMP_LEN > 0);
    std.debug.assert(MAX_TOOL_CALLS > 0);
}

// ============================================================
// Types
// ============================================================

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    input_json: []const u8,
};

pub const ToolOutput = struct {
    stdout: []const u8,
    success: bool,
};

pub const ToolResult = struct {
    tool_use_id: []const u8,
    content: []const u8,
    is_error: bool,
};

pub const ApiResponse = struct {
    success: bool,
    stop_reason: []const u8,
    content_json: []const u8,
    text: []const u8,
    tool_calls: []const ToolCall,
    error_message: []const u8,
    retryable: bool,
    input_tokens: u32,
    output_tokens: u32,
};

pub const ParsedContent = struct {
    text: []const u8,
    tool_calls: []const ToolCall,
};

// ============================================================
// JSON Value navigation helpers
//
// Short accessors for pulling typed fields out of a parsed
// std.json.ObjectMap.  Each returns a sensible default
// when the field is missing or has the wrong type.
// ============================================================

/// Get a string field from a JSON object, or null.
fn getStr(
    obj: json.ObjectMap,
    key: []const u8,
) ?[]const u8 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .string => |s| s,
        else => null,
    };
}

/// Get an integer field, or 0.
fn getInt(
    obj: json.ObjectMap,
    key: []const u8,
) i64 {
    const val = obj.get(key) orelse return 0;
    return switch (val) {
        .integer => |i| i,
        else => 0,
    };
}

/// Get an object field.
fn getObj(
    obj: json.ObjectMap,
    key: []const u8,
) ?json.ObjectMap {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .object => |o| o,
        else => null,
    };
}

/// Get an array field as a slice of Values.
fn getArr(
    obj: json.ObjectMap,
    key: []const u8,
) ?[]const json.Value {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .array => |a| a.items,
        else => null,
    };
}

// ============================================================
// JSON writing helpers
// ============================================================

/// Append a JSON-encoded, quoted string to a buffer.
/// Uses `std.json.Stringify.encodeJsonString` for correct
/// escaping of all control characters and Unicode.
pub fn appendJsonString(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    s: []const u8,
) !void {
    // Encode into a temporary arena-backed writer,
    // then append the result to the caller's buffer.
    var tmp: std.io.Writer.Allocating = .init(arena);
    try json.Stringify.encodeJsonString(
        s,
        .{},
        &tmp.writer,
    );
    const encoded = tmp.written();
    std.debug.assert(encoded.len >= 2); // At least "".
    try buf.appendSlice(arena, encoded);
}

// ============================================================
// String helpers
// ============================================================

pub fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn truncate(
    s: []const u8,
    max: usize,
) []const u8 {
    std.debug.assert(max > 0);
    return if (s.len <= max) s else s[0..max];
}

// ============================================================
// API request building
//
// The system prompt and tool schemas are agent-specific,
// so the caller passes them in.
// ============================================================

/// Write the extended thinking block if a budget is set.
fn writeThinkingBlock(
    w: *json.Stringify,
    thinking_budget_str: []const u8,
) !void {
    if (thinking_budget_str.len == 0) return;

    const budget = std.fmt.parseInt(
        u32,
        thinking_budget_str,
        10,
    ) catch return;
    if (budget == 0) return;

    try w.objectField("thinking");
    try w.beginObject();
    try w.objectField("type");
    try w.write("enabled");
    try w.objectField("budget_tokens");
    try w.print("{d}", .{budget});
    try w.endObject();
}

/// Build the JSON request body for the Anthropic messages
/// API.  `system_prompt`, `tool_schemas`, and
/// `max_tokens_str` are agent-specific constants passed
/// by the caller.
pub fn buildRequestJson(
    arena: Allocator,
    model: []const u8,
    messages: []const []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
    max_tokens_str: []const u8,
    thinking_budget_str: []const u8,
) ![]u8 {
    std.debug.assert(model.len > 0);
    std.debug.assert(messages.len > 0);
    std.debug.assert(system_prompt.len > 0);
    std.debug.assert(tool_schemas.len > 0);
    std.debug.assert(max_tokens_str.len > 0);

    const max_tokens = std.fmt.parseInt(
        u32,
        max_tokens_str,
        10,
    ) catch 16384;

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginObject();
    try w.objectField("model");
    try w.write(model);
    try w.objectField("max_tokens");
    try w.print("{d}", .{max_tokens});

    // Enable extended thinking when a budget is provided.
    try writeThinkingBlock(&w, thinking_budget_str);

    try writeRequestTail(
        &w,
        &out,
        system_prompt,
        tool_schemas,
        messages,
    );

    return out.written();
}

/// Write cache_control, system, tools, messages, and
/// close the top-level object.  Split from
/// `buildRequestJson` for the 70-line limit.
fn writeRequestTail(
    w: *json.Stringify,
    out: *std.io.Writer.Allocating,
    system_prompt: []const u8,
    tool_schemas: []const u8,
    messages: []const []const u8,
) !void {
    std.debug.assert(system_prompt.len > 0);
    std.debug.assert(tool_schemas.len > 0);

    try w.objectField("cache_control");
    try w.beginObject();
    try w.objectField("type");
    try w.write("ephemeral");
    try w.endObject();

    try w.objectField("system");
    try w.write(system_prompt);

    // Tool schemas arrive as a pre-formed JSON array.
    try w.objectField("tools");
    try w.beginWriteRaw();
    try out.writer.writeAll(tool_schemas);
    w.endWriteRaw();

    // Messages are pre-formed JSON objects.
    try w.objectField("messages");
    try w.beginArray();
    for (messages) |msg| {
        try w.beginWriteRaw();
        try out.writer.writeAll(msg);
        w.endWriteRaw();
    }
    try w.endArray();

    try w.endObject();
}

// ============================================================
// Response parsing
// ============================================================

/// Parse the raw API response JSON into an ApiResponse.
pub fn parseApiResponse(
    arena: Allocator,
    raw: []const u8,
) ApiResponse {
    std.debug.assert(raw.len > 0);

    const root = json.parseFromSliceLeaky(
        json.Value,
        arena,
        raw,
        .{},
    ) catch return errResp(
        "invalid JSON response",
        false,
    );

    const obj = switch (root) {
        .object => |o| o,
        else => return errResp(
            "expected JSON object",
            false,
        ),
    };

    // Detect API error responses.
    if (getStr(obj, "type")) |t| {
        if (eql(t, "error")) {
            return parseErrorObject(obj);
        }
    }

    return buildSuccessResponse(arena, obj);
}

/// Extract the error message from an API error response.
fn parseErrorObject(obj: json.ObjectMap) ApiResponse {
    const err_obj = getObj(obj, "error");
    const msg = if (err_obj) |e|
        getStr(e, "message") orelse "unknown API error"
    else
        "unknown API error";
    return errResp(msg, false);
}

/// Build a successful ApiResponse from a parsed object.
fn buildSuccessResponse(
    arena: Allocator,
    obj: json.ObjectMap,
) ApiResponse {
    const stop = getStr(obj, "stop_reason") orelse
        "unknown";

    // Re-serialize the content array for content_json.
    const content_val = obj.get("content");
    const content_json: []const u8 =
        if (content_val) |cv|
            json.Stringify.valueAlloc(
                arena,
                cv,
                .{},
            ) catch "[]"
        else
            "[]";

    const content_items: []const json.Value =
        getArr(obj, "content") orelse
        &[_]json.Value{};

    const parsed = parseContentBlocks(
        arena,
        content_items,
    );

    // Token usage.
    const usage = getObj(obj, "usage");
    const input_tokens: u32 = if (usage) |u|
        @intCast(@max(0, getInt(u, "input_tokens")))
    else
        0;
    const output_tokens: u32 = if (usage) |u|
        @intCast(@max(0, getInt(u, "output_tokens")))
    else
        0;

    return .{
        .success = true,
        .stop_reason = stop,
        .content_json = content_json,
        .text = parsed.text,
        .tool_calls = parsed.tool_calls,
        .error_message = "",
        .retryable = false,
        .input_tokens = input_tokens,
        .output_tokens = output_tokens,
    };
}

/// Parse content blocks (text and tool_use) from the
/// API response content array.
pub fn parseContentBlocks(
    arena: Allocator,
    blocks: []const json.Value,
) ParsedContent {
    const empty: ParsedContent = .{
        .text = "",
        .tool_calls = &.{},
    };
    if (blocks.len == 0) return empty;

    var text_buf: std.ArrayList(u8) = .empty;
    var calls: [MAX_TOOL_CALLS]ToolCall = undefined;
    var call_count: u32 = 0;

    for (blocks) |block_val| {
        const block = switch (block_val) {
            .object => |o| o,
            else => continue,
        };

        const block_type = getStr(block, "type") orelse
            continue;

        if (eql(block_type, "text")) {
            appendTextBlock(arena, &text_buf, block);
        }

        if (eql(block_type, "tool_use")) {
            if (call_count >= MAX_TOOL_CALLS) continue;
            if (parseToolUseBlock(arena, block)) |call| {
                calls[call_count] = call;
                call_count += 1;
            }
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

/// Append a text block's content to the text buffer.
fn appendTextBlock(
    arena: Allocator,
    text_buf: *std.ArrayList(u8),
    block: json.ObjectMap,
) void {
    const t = getStr(block, "text") orelse return;
    if (text_buf.items.len > 0) {
        text_buf.append(arena, '\n') catch {};
    }
    text_buf.appendSlice(arena, t) catch {};
}

/// Parse a single tool_use block into a ToolCall.
fn parseToolUseBlock(
    arena: Allocator,
    block: json.ObjectMap,
) ?ToolCall {
    const id = getStr(block, "id") orelse return null;
    const name = getStr(block, "name") orelse
        return null;

    // Re-serialize the input object.
    const input_val = block.get("input");
    const input_json: []const u8 =
        if (input_val) |iv|
            json.Stringify.valueAlloc(
                arena,
                iv,
                .{},
            ) catch "{}"
        else
            "{}";

    return .{
        .id = id,
        .name = name,
        .input_json = input_json,
    };
}

// ============================================================
// Message builders
//
// These produce JSON fragments for the Anthropic messages
// API conversation format.
// ============================================================

/// Wrap Claude's raw content JSON into an assistant message.
pub fn buildAssistantMsg(
    arena: Allocator,
    content_json: []const u8,
) []const u8 {
    std.debug.assert(content_json.len >= 2);
    return buildAssistantMsgJson(
        arena,
        content_json,
    ) catch "{}";
}

/// Inner builder for buildAssistantMsg.
fn buildAssistantMsgJson(
    arena: Allocator,
    content_json: []const u8,
) ![]const u8 {
    std.debug.assert(content_json.len >= 2);

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginObject();
    try w.objectField("role");
    try w.write("assistant");
    try w.objectField("content");
    // Content is already valid JSON — write it raw.
    try w.beginWriteRaw();
    try out.writer.writeAll(content_json);
    w.endWriteRaw();
    try w.endObject();

    const result = out.written();
    std.debug.assert(result.len > content_json.len);
    return result;
}

/// Build a user message containing tool results.
pub fn buildToolResultsMsg(
    arena: Allocator,
    results: []const ToolResult,
) []const u8 {
    std.debug.assert(results.len > 0);
    std.debug.assert(results.len <= MAX_TOOL_CALLS);
    return buildToolResultsMsgJson(
        arena,
        results,
    ) catch "{}";
}

/// Inner builder for buildToolResultsMsg.
fn buildToolResultsMsgJson(
    arena: Allocator,
    results: []const ToolResult,
) ![]const u8 {
    std.debug.assert(results.len > 0);

    var out: std.io.Writer.Allocating = .init(arena);
    var w: json.Stringify = .{
        .writer = &out.writer,
    };

    try w.beginObject();
    try w.objectField("role");
    try w.write("user");
    try w.objectField("content");
    try w.beginArray();
    for (results) |r| {
        try writeToolResultBlock(&w, r);
    }
    try w.endArray();
    try w.endObject();

    const result = out.written();
    std.debug.assert(result.len > 0);
    return result;
}

/// Write a single tool_result block via the Stringify.
fn writeToolResultBlock(
    w: *json.Stringify,
    r: ToolResult,
) !void {
    std.debug.assert(r.tool_use_id.len > 0);

    try w.beginObject();
    try w.objectField("type");
    try w.write("tool_result");
    try w.objectField("tool_use_id");
    try w.write(r.tool_use_id);
    if (r.is_error) {
        try w.objectField("is_error");
        try w.write(true);
    }
    try w.objectField("content");
    try w.write(r.content);
    try w.endObject();
}

/// Wrap a plain text string as a user message JSON object.
pub fn wrapUserTextMessage(
    arena: Allocator,
    text: []const u8,
) []const u8 {
    std.debug.assert(text.len > 0);
    return wrapUserTextMessageJson(
        arena,
        text,
    ) catch "{}";
}

/// Inner builder for wrapUserTextMessage.
fn wrapUserTextMessageJson(
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

// ============================================================
// History persistence helpers
// ============================================================

/// Remove newlines to produce a single-line JSON string
/// suitable for JSONL format.
pub fn collapseToLine(
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

/// Save the full conversation as a JSON array of messages
/// to a timestamped run log file.  `history_dir_fs` is
/// the resolved filesystem path to the history directory.
pub fn saveRunLog(
    arena: Allocator,
    messages: []const []const u8,
    history_dir_fs: []const u8,
) void {
    std.debug.assert(messages.len > 0);
    std.debug.assert(history_dir_fs.len > 0);

    var ts_buf: [TIMESTAMP_LEN]u8 = undefined;
    const ts = formatTimestamp(&ts_buf, '-');

    const path = std.fmt.allocPrint(
        arena,
        "{s}/run_{s}.json",
        .{ history_dir_fs, ts },
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

pub const stderr_file = std.fs.File{
    .handle = std.posix.STDERR_FILENO,
};

pub fn log(
    comptime fmt: []const u8,
    args: anytype,
) void {
    std.debug.print(fmt, args);
}

pub fn fatal(comptime msg: []const u8) noreturn {
    log("FATAL: " ++ msg, .{});
    std.process.exit(1);
}

pub fn logClaudeText(text: []const u8) void {
    std.debug.assert(text.len > 0);
    log(
        "\n  Claude: {s}\n",
        .{truncate(text, 2000)},
    );
}

pub fn logToolCall(
    name: []const u8,
    input: []const u8,
) void {
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
// File I/O
// ============================================================

pub fn writeFile(
    path: []const u8,
    content: []const u8,
) !void {
    std.debug.assert(path.len > 0);
    std.debug.assert(content.len > 0);

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(content);
}

// ============================================================
// Timestamp formatting
// ============================================================

pub fn formatTimestamp(
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
// Helpers
// ============================================================

pub fn errResp(
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
        .input_tokens = 0,
        .output_tokens = 0,
    };
}
