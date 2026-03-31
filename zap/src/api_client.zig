//! api_client.zig — shared Anthropic API client infrastructure.
//!
//! Common types, JSON parsing, message building, logging,
//! and timestamp formatting used by both `agent.zig` and
//! `engine_agent.zig`.  Each agent imports this module and
//! provides its own system prompt, tool schemas, tool
//! dispatch, and API transport (std.http vs curl).
//!
//! Design: all functions are pure or take explicit
//! parameters — no module-level mutable state.  Agent-
//! specific constants (system prompts, tool schemas, paths)
//! are passed in by the caller.

const std = @import("std");
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
};

pub const ParsedContent = struct {
    text: []const u8,
    tool_calls: []const ToolCall,
};

// ============================================================
// JSON helpers
//
// Hand-rolled JSON parsing for the Anthropic API response
// format.  We avoid pulling in a full JSON writer because
// the request/response shapes are fixed and small.
// ============================================================

/// Append a JSON-escaped, quoted string to a buffer.
pub fn appendJsonString(
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
pub fn extractJsonString(
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
pub fn findStringEnd(s: []const u8) ?[]const u8 {
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

/// Extract a raw JSON numeric value after a field name.
/// The needle should end with `:` — everything up to
/// the first non-numeric character is returned.
pub fn extractJsonNumber(
    json: []const u8,
    needle: []const u8,
) ?[]const u8 {
    std.debug.assert(needle.len > 0);

    const idx = indexOf(json, needle) orelse return null;
    var pos = idx + needle.len;

    // Skip optional whitespace after the colon.
    while (pos < json.len and
        (json[pos] == ' ' or json[pos] == '\t'))
    {
        pos += 1;
    }
    if (pos >= json.len) return null;

    const start = pos;
    while (pos < json.len) {
        const c = json[pos];
        const is_numeric = (c >= '0' and c <= '9') or
            c == '.' or c == '-' or c == '+' or
            c == 'e' or c == 'E';
        if (!is_numeric) break;
        pos += 1;
    }

    if (pos == start) return null;
    return json[start..pos];
}

/// Extract a raw JSON array value for a given field name.
/// The field is the quoted key (e.g., `"content"`).  Skips
/// the `:` and whitespace, then finds matching `[...]`.
pub fn extractRawArray(
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
pub fn extractRawObject(
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
pub fn skipColonAndWhitespace(
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
pub fn findMatchingBracket(
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
pub fn splitTopLevelObjects(
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
pub fn parseJsonStringArray(
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
pub fn containsField(
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
    if (!std.mem.eql(
        u8,
        block[pos..][0..value.len],
        value,
    )) {
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

pub fn indexOf(
    haystack: []const u8,
    needle: []const u8,
) ?usize {
    return std.mem.indexOf(u8, haystack, needle);
}

pub fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn startsWith(
    s: []const u8,
    prefix: []const u8,
) bool {
    return std.mem.startsWith(u8, s, prefix);
}

pub fn truncate(s: []const u8, max: usize) []const u8 {
    std.debug.assert(max > 0);
    return if (s.len <= max) s else s[0..max];
}

// ============================================================
// API request building
//
// The system prompt and tool schemas are agent-specific,
// so the caller passes them in.
// ============================================================

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
) ![]u8 {
    std.debug.assert(model.len > 0);
    std.debug.assert(messages.len > 0);
    std.debug.assert(system_prompt.len > 0);
    std.debug.assert(tool_schemas.len > 0);
    std.debug.assert(max_tokens_str.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\"model\":\"");
    try buf.appendSlice(arena, model);
    try buf.appendSlice(arena, "\",\"max_tokens\":");
    try buf.appendSlice(arena, max_tokens_str);
    try buf.appendSlice(arena, ",\"system\":");
    try appendJsonString(arena, &buf, system_prompt);
    try buf.appendSlice(arena, ",\"tools\":");
    try buf.appendSlice(arena, tool_schemas);
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
pub fn parseApiResponse(
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

    const parsed = parseContentBlocks(arena, content);

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
pub fn parseContentBlocks(
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

    return std.fmt.allocPrint(
        arena,
        "{{\"role\":\"assistant\",\"content\":{s}}}",
        .{content_json},
    ) catch "{}";
}

/// Build a user message containing tool results.
pub fn buildToolResultsMsg(
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
pub fn appendToolResultBlock(
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
    buf.appendSlice(arena, r.tool_use_id) catch return;
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
    appendJsonString(arena, buf, r.content) catch return;
    buf.append(arena, '}') catch return;
}

/// Wrap a plain text string as a user message JSON object.
pub fn wrapUserTextMessage(
    arena: Allocator,
    text: []const u8,
) []const u8 {
    std.debug.assert(text.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"role\":\"user\",\"content\":",
    ) catch return "{}";
    appendJsonString(arena, &buf, text) catch return "{}";
    buf.append(arena, '}') catch return "{}";

    std.debug.assert(buf.items.len > 0);
    return buf.items;
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
    };
}
