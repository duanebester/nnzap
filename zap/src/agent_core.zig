//! Agent core — shared framework for LLM-powered
//! research agents.
//!
//! Provides the generic experiment loop, table-driven
//! tool dispatch, API calling with retry, prompt
//! loading, and history persistence.  Each research
//! profile (mnist, bonsai, etc.) provides a thin config
//! layer on top of this core.
//!
//! Usage:
//!   const core = @import("agent_core.zig");
//!   const config = core.AgentConfig{ ... };
//!   core.run(&config) catch |err| { ... };

const std = @import("std");
const Allocator = std.mem.Allocator;
pub const api = @import("api_client.zig");
const tools = @import("tools.zig");

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_TOOL_ARGS: u32 = 16;
const MAX_API_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_BYTES = tools.MAX_OUTPUT_BYTES;
const MAX_HISTORY_SIZE: usize = 2 * 1024 * 1024;
const MAX_FILE_SIZE = tools.MAX_FILE_SIZE;
const TOOL_INPUT_PATH: []const u8 = "_tool_input.json";

const MAX_RETRY_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY_MS: u64 = 2_000;
const RETRY_MAX_DELAY_MS: u64 = 30_000;

const API_TIMEOUT_SECS_STR: []const u8 = "600";
const API_URL: []const u8 =
    "https://api.anthropic.com/v1/messages";
const API_VERSION: []const u8 = "2023-06-01";
pub const DEFAULT_MODEL: []const u8 = "claude-opus-4-6";

// ============================================================
// Public type aliases
// ============================================================

pub const ToolCall = api.ToolCall;
pub const ToolOutput = api.ToolOutput;
pub const ToolResult = api.ToolResult;
pub const ApiResponse = api.ApiResponse;

// ============================================================
// Types
// ============================================================

/// How the agent passes input to the toolbox binary.
pub const ToolShape = enum {
    /// Tool takes no parameters.
    no_input,
    /// Extract one JSON field, pass as positional arg.
    string_arg,
    /// Write full JSON to temp file, pass -f.
    json_payload,
};

/// Maps an LLM tool name to a CLI subcommand.
pub const ToolMapping = struct {
    /// LLM-facing name: "read_file".
    tool_name: []const u8,
    /// CLI-facing name: "read-file".
    subcommand: []const u8,
    /// How the agent passes input to the toolbox.
    shape: ToolShape,
    /// For string_arg: which JSON field to extract.
    field: ?[]const u8 = null,
};

// ============================================================
// Comptime tool definitions
//
// Define tools once as ToolDef structs.  At compile time,
// generate both the ToolMapping dispatch table and the
// JSON schema string for the Anthropic API.  Single
// source of truth — no separate JSON files (Rule 14).
// ============================================================

/// JSON Schema property type for tool parameters.
pub const PropertyType = enum {
    string,
    integer,
    string_array,
};

/// One property in a tool's input schema.
pub const Property = struct {
    name: []const u8,
    description: []const u8,
    type: PropertyType = .string,
    required: bool = true,
};

/// Complete tool definition: dispatch info + API schema.
/// Define once, derive both ToolMapping and JSON at
/// comptime.
pub const ToolDef = struct {
    /// LLM-facing name: "read_file".
    name: []const u8,
    /// CLI-facing subcommand: "read-file".
    subcommand: []const u8,
    /// LLM-facing description for the API schema.
    description: []const u8,
    /// Input properties (empty = no parameters).
    properties: []const Property = &.{},
    /// Override auto-derived dispatch shape.  null =
    /// derive: 0 props → no_input, 1 → string_arg,
    /// 2+ → json_payload.
    shape_override: ?ToolShape = null,

    /// Resolve the dispatch shape for this tool.
    fn resolveShape(
        comptime self: ToolDef,
    ) ToolShape {
        if (self.shape_override) |s| return s;
        return switch (self.properties.len) {
            0 => .no_input,
            1 => .string_arg,
            else => .json_payload,
        };
    }

    /// Derive the field name for string_arg dispatch.
    fn resolveField(
        comptime self: ToolDef,
    ) ?[]const u8 {
        if (self.resolveShape() != .string_arg) {
            return null;
        }
        comptime std.debug.assert(
            self.properties.len == 1,
        );
        return self.properties[0].name;
    }
};

/// Generate a ToolMapping slice from ToolDef array.
pub fn toolMappings(
    comptime defs: []const ToolDef,
) []const ToolMapping {
    comptime {
        var maps: [defs.len]ToolMapping = undefined;
        for (defs, 0..) |def, i| {
            maps[i] = .{
                .tool_name = def.name,
                .subcommand = def.subcommand,
                .shape = def.resolveShape(),
                .field = def.resolveField(),
            };
        }
        const final = maps;
        return &final;
    }
}

/// Generate the Anthropic tool schemas JSON array
/// from a ToolDef array.
pub fn toolSchemas(
    comptime defs: []const ToolDef,
) []const u8 {
    comptime {
        @setEvalBranchQuota(100_000);
        var json: []const u8 = "[";
        for (defs, 0..) |def, i| {
            if (i > 0) json = json ++ ",";
            json = json ++ toolJson(def);
        }
        return json ++ "]";
    }
}

// ── Comptime JSON helpers (private) ──────────────────

/// Generate JSON for one tool definition.
fn toolJson(comptime def: ToolDef) []const u8 {
    return "{\"name\":\"" ++ def.name ++ "\"," ++
        "\"description\":\"" ++
        comptimeEscape(def.description) ++
        "\",\"input_schema\":" ++
        schemaJson(def.properties) ++ "}";
}

/// Generate the input_schema JSON object.
fn schemaJson(
    comptime props: []const Property,
) []const u8 {
    comptime {
        var pj: []const u8 = "";
        var rj: []const u8 = "";
        for (props, 0..) |p, i| {
            if (i > 0) pj = pj ++ ",";
            pj = pj ++ propJson(p);
            if (p.required) {
                if (rj.len > 0) rj = rj ++ ",";
                rj = rj ++ "\"" ++ p.name ++ "\"";
            }
        }
        return "{\"type\":\"object\"," ++
            "\"properties\":{" ++ pj ++ "}," ++
            "\"required\":[" ++ rj ++ "]}";
    }
}

/// Generate JSON for one property.
fn propJson(comptime p: Property) []const u8 {
    const type_str = switch (p.type) {
        .string => "\"type\":\"string\"",
        .integer => "\"type\":\"integer\"",
        .string_array => "\"type\":\"array\"," ++
            "\"items\":{\"type\":\"string\"}",
    };
    return "\"" ++ p.name ++ "\":{" ++ type_str ++
        ",\"description\":\"" ++
        comptimeEscape(p.description) ++ "\"}";
}

/// Escape a string for use in JSON at compile time.
fn comptimeEscape(
    comptime s: []const u8,
) []const u8 {
    comptime {
        var r: []const u8 = "";
        for (s) |c| {
            r = r ++ escapeChar(c);
        }
        return r;
    }
}

/// Return the JSON escape sequence for one byte.
fn escapeChar(comptime c: u8) []const u8 {
    return switch (c) {
        '"' => "\\\"",
        '\\' => "\\\\",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
        else => &[_]u8{c},
    };
}

/// One metric to extract from benchmark JSONL for the
/// compact history summary.
pub const HistoryField = struct {
    /// JSON key in the benchmark output.
    json_key: []const u8,
    /// Short label for the history summary line.
    label: []const u8,
};

/// Profile definition.  Each research domain provides
/// one of these to configure the generic agent loop.
pub const AgentConfig = struct {
    // Identity.
    name: []const u8,
    toolbox_path: []const u8,
    history_dir: []const u8,

    // Prompts.
    system_prompt_path: []const u8,
    /// Anthropic tool schemas JSON array.  Generate at
    /// comptime from ToolDef array via toolSchemas().
    tool_schemas: []const u8,

    // Tool dispatch table.
    tool_map: []const ToolMapping,

    // Tool names whose output gets persisted to
    // experiments.jsonl on success.
    persist_tools: []const []const u8 = &.{},

    /// Metrics to extract from benchmark JSON for the
    /// compact history.  Walked by formatHistoryLine
    /// instead of hardcoded field names.
    history_fields: []const HistoryField = &.{},

    // Limits (Rule 4 — hard caps).
    max_experiments: u32 = 50,
    max_turns_per_experiment: u32 = 80,
    turn_warning_threshold: u32 = 70,
    max_messages: u32 = 512,
    max_tool_calls: u32 = 16,
    max_tool_output: usize = 50_000,

    // API configuration.
    max_tokens_str: []const u8 = "32768",
    thinking_budget_str: []const u8 = "16384",

    // Context building callback.  If null, a default
    // context builder is used (history + summaries).
    build_context_fn: ?*const fn (
        Allocator,
        *const AgentConfig,
    ) []const u8 = null,
};

/// Accumulated statistics across experiments.
const RunStats = struct {
    experiments: u32 = 0,
    turns: u32 = 0,
    tool_calls: u32 = 0,
    api_errors: u32 = 0,
    api_ms: i64 = 0,
    tool_ms: i64 = 0,
    input_tokens: u64 = 0,
    output_tokens: u64 = 0,
};

/// Result of one experiment's inner turn loop.
const ExperimentResult = struct {
    turns: u32,
    tool_calls: u32,
    bench_ran: bool,
    api_failed: bool,
    api_ms: i64,
    tool_ms: i64,
    input_tokens: u64,
    output_tokens: u64,
};

/// Bundles per-experiment state threaded through
/// every turn of the conversation loop.
const TurnContext = struct {
    config: *const AgentConfig,
    api_key: []const u8,
    model: []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
    messages: *[512][]const u8,
    count: *u32,
};

// ============================================================
// Generic experiment loop
// ============================================================

/// Run the agent: setup, experiment loop, summary.
/// This is the main entry point for single-tier mode.
pub fn run(config: *const AgentConfig) !void {
    var arena_state = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const run_start = timestampMs();

    printHeader(config.name);
    ensureHistoryDir(arena, config.history_dir);

    const api_key = loadApiKey() orelse {
        api.fatal(
            "Set ANTHROPIC_API_KEY env var.\n" ++
                "  export ANTHROPIC_API_KEY=sk-ant-...\n",
        );
        unreachable;
    };
    const model = loadModel();
    api.log("Model: {s}\n", .{model});
    api.log(
        "Limits: max={d} experiments, " ++
            "max={d} turns/experiment\n",
        .{
            config.max_experiments,
            config.max_turns_per_experiment,
        },
    );

    if (!buildToolbox(arena)) {
        api.fatal("zig build failed.\n");
        unreachable;
    }

    const system_prompt = loadPromptFile(
        arena,
        config.system_prompt_path,
    );
    const tool_schemas = config.tool_schemas;

    var stats = RunStats{};

    stats = runExperiments(
        config,
        arena,
        api_key,
        model,
        system_prompt,
        tool_schemas,
        stats,
    );

    printRunSummary(
        config,
        arena,
        run_start,
        stats,
    );
}

/// Outer experiment loop.  Each iteration is one
/// experiment with a fresh context window.
fn runExperiments(
    config: *const AgentConfig,
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
    initial_stats: RunStats,
) RunStats {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(system_prompt.len > 0);

    var stats = initial_stats;
    var experiment: u32 = 0;

    while (experiment < config.max_experiments) : (experiment += 1) {
        api.log(
            "\n" ++
                "========================================" ++
                "======\n" ++
                "  Experiment {d}/{d}\n" ++
                "========================================" ++
                "======\n",
            .{ experiment + 1, config.max_experiments },
        );

        const result = runSingleExperiment(
            config,
            arena,
            api_key,
            model,
            system_prompt,
            tool_schemas,
        );

        stats.turns += result.turns;
        stats.tool_calls += result.tool_calls;
        stats.api_ms += result.api_ms;
        stats.tool_ms += result.tool_ms;
        stats.input_tokens += result.input_tokens;
        stats.output_tokens += result.output_tokens;

        if (result.api_failed) {
            stats.api_errors += 1;
            api.log("  Stopping: API failure.\n", .{});
            break;
        }

        stats.experiments += 1;

        api.log(
            "  Experiment {d} done ({d} turns).\n",
            .{ experiment + 1, result.turns },
        );
    }

    return stats;
}

/// Inner turn loop for one experiment.
fn runSingleExperiment(
    config: *const AgentConfig,
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
) ExperimentResult {
    const context_text = if (config.build_context_fn) |f|
        f(arena, config)
    else
        buildDefaultContext(arena, config);

    const first_msg = api.wrapUserTextMessage(
        arena,
        context_text,
    );

    var messages: [512][]const u8 = undefined;
    var count: u32 = 0;
    messages[0] = first_msg;
    count = 1;

    var result = ExperimentResult{
        .turns = 0,
        .tool_calls = 0,
        .bench_ran = false,
        .api_failed = false,
        .api_ms = 0,
        .tool_ms = 0,
        .input_tokens = 0,
        .output_tokens = 0,
    };

    const ctx = TurnContext{
        .config = config,
        .api_key = api_key,
        .model = model,
        .system_prompt = system_prompt,
        .tool_schemas = tool_schemas,
        .messages = &messages,
        .count = &count,
    };

    var turn: u32 = 0;
    while (turn < config.max_turns_per_experiment) : (turn += 1) {
        const should_break = runOneTurn(
            &ctx,
            arena,
            turn,
            &result,
        );
        if (should_break) break;
    }

    result.turns = turn + 1;

    // Save this experiment's conversation log.
    const fs_dir = resolveToFs(
        arena,
        config.history_dir,
    ) orelse config.history_dir;
    api.saveRunLog(arena, messages[0..count], fs_dir);

    return result;
}

/// Call the API and process the initial response.
/// Returns null if the API call failed (result.api_failed
/// is set).
fn callApiAndProcess(
    ctx: *const TurnContext,
    arena: Allocator,
    result: *ExperimentResult,
) ?ApiResponse {
    const api_start = timestampMs();
    const resp = callApiWithRetry(
        arena,
        ctx.api_key,
        ctx.model,
        ctx.messages[0..ctx.count.*],
        ctx.system_prompt,
        ctx.tool_schemas,
        ctx.config.history_dir,
        ctx.config.max_tokens_str,
        ctx.config.thinking_budget_str,
    );
    const api_elapsed = timestampMs() - api_start;
    result.api_ms += api_elapsed;

    if (!resp.success) {
        api.log(
            "  API error: {s}\n",
            .{resp.error_message},
        );
        result.api_failed = true;
        return null;
    }

    logApiResponse(arena, resp, api_elapsed);
    extractTokenUsage(
        arena,
        ctx.config.history_dir,
        result,
    );
    return resp;
}

/// Execute one turn: call API, process response,
/// execute tools.  Returns true if the loop should
/// break.
fn runOneTurn(
    ctx: *const TurnContext,
    arena: Allocator,
    turn: u32,
    result: *ExperimentResult,
) bool {
    api.log(
        "\n--- Turn {d} ---\n",
        .{turn + 1},
    );

    const ctx_bytes = contextSizeBytes(
        ctx.messages[0..ctx.count.*],
    );
    api.log(
        "  Context: {d} messages, {d} KB\n",
        .{ ctx.count.*, ctx_bytes / 1024 },
    );

    const resp = callApiAndProcess(
        ctx,
        arena,
        result,
    ) orelse return true;

    if (resp.text.len > 0) api.logClaudeText(resp.text);

    ctx.messages[ctx.count.*] = api.buildAssistantMsg(
        arena,
        resp.content_json,
    );
    ctx.count.* += 1;

    const is_tool_use = api.eql(
        resp.stop_reason,
        "tool_use",
    );
    if (!is_tool_use or resp.tool_calls.len == 0) {
        api.log(
            "  Claude signalled end of experiment.\n",
            .{},
        );
        return true;
    }

    return executeAndAppendTools(
        ctx.config,
        arena,
        resp,
        ctx.messages,
        ctx.count,
        turn,
        result,
    );
}

/// Execute tool calls and append results to messages.
/// Returns true if the loop should break.
fn executeAndAppendTools(
    config: *const AgentConfig,
    arena: Allocator,
    resp: ApiResponse,
    messages: *[512][]const u8,
    count: *u32,
    turn: u32,
    result: *ExperimentResult,
) bool {
    const tools_start = timestampMs();
    const results = executeTools(
        config,
        arena,
        resp.tool_calls,
    );
    const tools_elapsed = timestampMs() - tools_start;
    result.tool_ms += tools_elapsed;
    result.tool_calls += @intCast(resp.tool_calls.len);

    logToolResults(arena, resp, results, tools_elapsed);

    messages[count.*] = api.buildToolResultsMsg(
        arena,
        results,
    );

    // Inject turn-limit warning near the end.
    if (turn + 1 >= config.turn_warning_threshold) {
        injectTurnWarning(
            config,
            arena,
            messages,
            count.*,
            turn,
        );
    }

    count.* += 1;

    if (count.* + 2 >= config.max_messages) {
        api.log("  Message limit reached.\n", .{});
        return true;
    }
    return false;
}

/// Log API response metadata.
fn logApiResponse(
    arena: Allocator,
    resp: ApiResponse,
    api_elapsed: i64,
) void {
    api.log(
        "  API response: {s} ({d} tool calls, " ++
            "{d} KB)\n",
        .{
            nanosToMsStr(arena, api_elapsed * 1_000_000),
            resp.tool_calls.len,
            resp.content_json.len / 1024,
        },
    );
}

/// Extract and accumulate token usage from the saved
/// API response file.
fn extractTokenUsage(
    arena: Allocator,
    history_dir: []const u8,
    result: *ExperimentResult,
) void {
    const fs_dir = resolveToFs(
        arena,
        history_dir,
    ) orelse return;
    const usage_path = std.fmt.allocPrint(
        arena,
        "{s}/_response.json",
        .{fs_dir},
    ) catch return;

    const raw = std.fs.cwd().readFileAlloc(
        arena,
        usage_path,
        MAX_API_RESPONSE,
    ) catch return;
    if (raw.len == 0) return;

    const in_tok, const out_tok = parseTokenUsage(
        arena,
        raw,
    );
    result.input_tokens += in_tok;
    result.output_tokens += out_tok;
    api.log(
        "  Tokens: {d} in, {d} out\n",
        .{ in_tok, out_tok },
    );
}

/// Log tool execution results summary.
fn logToolResults(
    arena: Allocator,
    resp: ApiResponse,
    results: []const ToolResult,
    tools_elapsed: i64,
) void {
    var err_count: u32 = 0;
    var result_bytes: usize = 0;
    for (results) |r| {
        result_bytes += r.content.len;
        if (r.is_error) err_count += 1;
    }
    api.log(
        "  Tools done: {d} calls, {d} errors, " ++
            "{d} KB result, {s}\n",
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
}

/// Inject a turn-limit warning into the tool results
/// message so the agent knows to wrap up.
fn injectTurnWarning(
    config: *const AgentConfig,
    arena: Allocator,
    messages: *[512][]const u8,
    idx: u32,
    turn: u32,
) void {
    const remaining =
        config.max_turns_per_experiment - turn - 1;
    const warn = std.fmt.allocPrint(
        arena,
        ",{{\"type\":\"text\",\"text\":" ++
            "\"WARNING: {d} turns remaining " ++
            "out of {d}. Wrap up NOW. " ++
            "If your current approach is not " ++
            "working, rollback and add_summary " ++
            "describing what you tried, then " ++
            "STOP.\"}}]}}",
        .{ remaining, config.max_turns_per_experiment },
    ) catch return;

    const prev = messages[idx];
    if (prev.len < 2) return;

    // Replace closing "]}" with warning + "]}".
    const trimmed = prev[0 .. prev.len - 2];
    messages[idx] = std.fmt.allocPrint(
        arena,
        "{s}{s}",
        .{ trimmed, warn },
    ) catch prev;
}

/// Print the end-of-run summary.
fn printRunSummary(
    config: *const AgentConfig,
    arena: Allocator,
    run_start: i64,
    stats: RunStats,
) void {
    _ = config;
    const run_elapsed = timestampMs() - run_start;

    // Opus 4.6 pricing: $5/MTok input, $25/MTok output.
    const cost_cents =
        (stats.input_tokens * 5 +
            stats.output_tokens * 25) / 10_000;
    const cost_str = std.fmt.allocPrint(
        arena,
        "${d}.{d:0>2}",
        .{ cost_cents / 100, cost_cents % 100 },
    ) catch "$?.??";

    api.log(
        "\n" ++
            "========================================" ++
            "======\n" ++
            "  Run summary\n" ++
            "========================================" ++
            "======\n" ++
            "  Experiments: {d}\n" ++
            "  Total turns: {d}\n" ++
            "  Tool calls:  {d}\n" ++
            "  API errors:  {d}\n" ++
            "  In tokens:   {d}\n" ++
            "  Out tokens:  {d}\n" ++
            "  Est. cost:   {s}\n" ++
            "  API time:    {s}\n" ++
            "  Tool time:   {s}\n" ++
            "  Total time:  {s}\n" ++
            "========================================" ++
            "======\n",
        .{
            stats.experiments,
            stats.turns,
            stats.tool_calls,
            stats.api_errors,
            stats.input_tokens,
            stats.output_tokens,
            cost_str,
            nanosToMsStr(
                arena,
                stats.api_ms * 1_000_000,
            ),
            nanosToMsStr(
                arena,
                stats.tool_ms * 1_000_000,
            ),
            nanosToMsStr(
                arena,
                run_elapsed * 1_000_000,
            ),
        },
    );
}

// ============================================================
// Tool dispatch
//
// The dispatcher walks the profile's ToolMapping table
// and routes each tool call to the toolbox binary.
// No tool logic lives here — all smarts are in the
// toolbox binary where they can be tested independently.
// ============================================================

/// Route a tool call via the ToolMapping table.
pub fn dispatchTool(
    config: *const AgentConfig,
    arena: Allocator,
    call: ToolCall,
) ToolOutput {
    std.debug.assert(call.name.len > 0);

    for (config.tool_map) |mapping| {
        if (api.eql(call.name, mapping.tool_name)) {
            return switch (mapping.shape) {
                .no_input => callToolbox(
                    config,
                    arena,
                    &.{mapping.subcommand},
                ),
                .string_arg => dispatchStringArg(
                    config,
                    arena,
                    mapping,
                    call.input_json,
                ),
                .json_payload => dispatchJsonPayload(
                    config,
                    arena,
                    mapping.subcommand,
                    call.input_json,
                ),
            };
        }
    }

    return .{
        .stdout = "Error: unknown tool name",
        .success = false,
    };
}

/// Extract one JSON field, pass as positional arg.
fn dispatchStringArg(
    config: *const AgentConfig,
    arena: Allocator,
    mapping: ToolMapping,
    input_json: []const u8,
) ToolOutput {
    const field_name = mapping.field orelse {
        return .{
            .stdout = "Error: no field for string_arg",
            .success = false,
        };
    };

    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        input_json,
        .{},
    ) catch {
        return .{
            .stdout = "Error: invalid JSON input",
            .success = false,
        };
    };
    const obj = switch (parsed) {
        .object => |o| o,
        else => return .{
            .stdout = "Error: expected JSON object",
            .success = false,
        },
    };
    const val = obj.get(field_name) orelse {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: missing '{s}' field",
            .{field_name},
        ) catch "Error: missing field";
        return .{ .stdout = msg, .success = false };
    };
    const str_val = switch (val) {
        .string => |s| s,
        .integer => |i| std.fmt.allocPrint(
            arena,
            "{d}",
            .{i},
        ) catch return .{
            .stdout = "Error: format failed",
            .success = false,
        },
        else => return .{
            .stdout = "Error: field not a string",
            .success = false,
        },
    };

    return callToolbox(
        config,
        arena,
        &.{ mapping.subcommand, str_val },
    );
}

/// Write full JSON input to temp file, pass -f.
fn dispatchJsonPayload(
    config: *const AgentConfig,
    arena: Allocator,
    subcommand: []const u8,
    input_json: []const u8,
) ToolOutput {
    // Write raw JSON input to a temp file.
    const file = std.fs.cwd().createFile(
        TOOL_INPUT_PATH,
        .{},
    ) catch {
        return .{
            .stdout = "Error: cannot create temp file",
            .success = false,
        };
    };
    file.writeAll(input_json) catch {
        file.close();
        return .{
            .stdout = "Error: write temp file failed",
            .success = false,
        };
    };
    file.close();

    const result = callToolbox(
        config,
        arena,
        &.{ subcommand, "-f", TOOL_INPUT_PATH },
    );

    // Clean up temp file (defense in depth — toolbox
    // should also delete it).
    std.fs.cwd().deleteFile(TOOL_INPUT_PATH) catch {};

    return result;
}

/// Spawn the toolbox binary with the given arguments
/// and capture stdout.
pub fn callToolbox(
    config: *const AgentConfig,
    arena: Allocator,
    args: []const []const u8,
) ToolOutput {
    std.debug.assert(args.len > 0);
    std.debug.assert(args.len + 1 <= MAX_TOOL_ARGS);

    var argv: [MAX_TOOL_ARGS][]const u8 = undefined;
    argv[0] = config.toolbox_path;
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

// ============================================================
// API calling with retry
// ============================================================

/// Call the API with exponential backoff on transient
/// errors.
pub fn callApiWithRetry(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    messages: []const []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
    history_dir: []const u8,
    max_tokens_str: []const u8,
    thinking_budget_str: []const u8,
) ApiResponse {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(messages.len > 0);

    var attempt: u32 = 0;
    while (attempt < MAX_RETRY_ATTEMPTS) : (attempt += 1) {
        if (attempt > 0) {
            const shift: u6 = @intCast(attempt - 1);
            const base: u64 = RETRY_BASE_DELAY_MS;
            const cap: u64 = RETRY_MAX_DELAY_MS;
            const delay: u64 = @min(base << shift, cap);
            api.log(
                "  Retrying in {d}ms " ++
                    "(attempt {d}/{d})...\n",
                .{
                    delay,
                    attempt + 1,
                    MAX_RETRY_ATTEMPTS,
                },
            );
            const ns: u64 = delay * 1_000_000;
            std.Thread.sleep(ns);
        }

        const resp = callApi(
            arena,
            api_key,
            model,
            messages,
            system_prompt,
            tool_schemas,
            history_dir,
            max_tokens_str,
            thinking_budget_str,
        );

        if (resp.success) return resp;
        if (!resp.retryable) {
            api.log(
                "  API error (not retryable): {s}\n",
                .{resp.error_message},
            );
            return resp;
        }

        api.log(
            "  API error (retryable): {s}\n",
            .{resp.error_message},
        );
    }

    return api.errResp("max retries exhausted", false);
}

/// Call the Anthropic Messages API via curl.
///
/// Using curl instead of std.http because:
///   1. Proper --connect-timeout and --max-time.
///   2. TLS in a separate process — connection reset
///      cannot kill the agent.
///   3. Battle-tested on macOS.
fn callApi(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    messages: []const []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
    history_dir: []const u8,
    max_tokens_str: []const u8,
    thinking_budget_str: []const u8,
) ApiResponse {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(messages.len > 0);

    const body = api.buildRequestJson(
        arena,
        model,
        messages,
        system_prompt,
        tool_schemas,
        max_tokens_str,
        thinking_budget_str,
    ) catch return api.errResp(
        "request build failed",
        false,
    );

    // Save request for debugging.
    const fs_dir = resolveToFs(
        arena,
        history_dir,
    ) orelse history_dir;
    const request_path = std.fmt.allocPrint(
        arena,
        "{s}/_request.json",
        .{fs_dir},
    ) catch return api.errResp(
        "path format failed",
        false,
    );
    api.writeFile(request_path, body) catch {
        return api.errResp(
            "failed to write request file",
            false,
        );
    };
    api.log(
        "  API request: {d} KB\n",
        .{body.len / 1024},
    );

    return executeCurl(
        arena,
        api_key,
        request_path,
        fs_dir,
    );
}

/// Run curl and parse the result.
fn executeCurl(
    arena: Allocator,
    api_key: []const u8,
    request_path: []const u8,
    fs_dir: []const u8,
) ApiResponse {
    const data_arg = std.fmt.allocPrint(
        arena,
        "@{s}",
        .{request_path},
    ) catch return api.errResp(
        "curl arg format failed",
        false,
    );
    const key_header = std.fmt.allocPrint(
        arena,
        "x-api-key: {s}",
        .{api_key},
    ) catch return api.errResp(
        "key header format failed",
        false,
    );

    api.log("  Waiting for API response...\n", .{});

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{
            "curl",
            "-s",
            "--connect-timeout",
            "30",
            "--max-time",
            API_TIMEOUT_SECS_STR,
            "-X",
            "POST",
            API_URL,
            "-H",
            "Content-Type: application/json",
            "-H",
            key_header,
            "-H",
            "anthropic-version: " ++ API_VERSION,
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

    return parseCurlResult(arena, result, fs_dir);
}

/// Parse curl output: split body from HTTP status code,
/// handle errors, return parsed API response.
/// Check curl exit status.  Returns an error response
/// if curl failed, or null on success.
fn checkCurlSuccess(
    arena: Allocator,
    result: std.process.Child.RunResult,
) ?ApiResponse {
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
        "curl failed: {s}",
        .{detail},
    ) catch "curl failed";
    return api.errResp(msg, true);
}

/// Persist the raw API response for token-usage
/// extraction, then parse it.
fn persistAndParseResponse(
    arena: Allocator,
    response_data: []const u8,
    fs_dir: []const u8,
) ApiResponse {
    std.debug.assert(response_data.len > 0);

    const resp_path = std.fmt.allocPrint(
        arena,
        "{s}/_response.json",
        .{fs_dir},
    ) catch "";
    if (resp_path.len > 0) {
        api.writeFile(
            resp_path,
            response_data,
        ) catch {};
    }

    return api.parseApiResponse(
        arena,
        response_data,
    );
}

fn parseCurlResult(
    arena: Allocator,
    result: std.process.Child.RunResult,
    fs_dir: []const u8,
) ApiResponse {
    // Early exit if curl process failed.
    if (checkCurlSuccess(arena, result)) |err| {
        return err;
    }

    const output = result.stdout;
    if (output.len < 4) {
        return api.errResp("empty curl output", true);
    }

    // Split body from HTTP status (appended by -w).
    const last_nl = std.mem.lastIndexOfScalar(
        u8,
        output,
        '\n',
    ) orelse return api.errResp(
        "malformed curl output",
        true,
    );

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
            "failed to parse HTTP status",
            true,
        );
    }

    api.log(
        "  API response body: {d} KB (HTTP {d})\n",
        .{ response_data.len / 1024, status_code },
    );

    if (status_code != 200) {
        return handleNonOkStatus(
            arena,
            status_code,
            response_data,
        );
    }

    if (response_data.len == 0) {
        return api.errResp(
            "empty API response",
            true,
        );
    }

    return persistAndParseResponse(
        arena,
        response_data,
        fs_dir,
    );
}

/// Descriptive error for non-200 API responses.
fn handleNonOkStatus(
    arena: Allocator,
    code: u16,
    body: []const u8,
) ApiResponse {
    const api_msg = extractApiErrorMsg(arena, body);
    const detail = if (api_msg.len > 0)
        api_msg
    else
        api.truncate(body, 500);

    const msg = std.fmt.allocPrint(
        arena,
        "HTTP {d}: {s}",
        .{ code, detail },
    ) catch "HTTP error (unknown)";

    api.log("  API HTTP {d}: {s}\n", .{ code, detail });

    // 429 = rate limited, 529 = overloaded,
    // 500+ = server errors — all retryable.
    const retryable = (code == 429 or
        code == 529 or code >= 500);
    return api.errResp(msg, retryable);
}

/// Extract error message from API error response JSON.
fn extractApiErrorMsg(
    arena: Allocator,
    body: []const u8,
) []const u8 {
    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        body,
        .{},
    ) catch return "";
    const obj = switch (parsed) {
        .object => |o| o,
        else => return "",
    };
    // Try error.message, then top-level message.
    if (obj.get("error")) |err_val| {
        if (err_val == .object) {
            if (err_val.object.get("message")) |m| {
                if (m == .string) return m.string;
            }
        }
    }
    if (obj.get("message")) |m| {
        if (m == .string) return m.string;
    }
    return "";
}

// ============================================================
// Tool execution
// ============================================================

/// Execute all tool calls from one API response.
pub fn executeTools(
    config: *const AgentConfig,
    arena: Allocator,
    calls: []const ToolCall,
) []const ToolResult {
    std.debug.assert(calls.len > 0);
    std.debug.assert(calls.len <= config.max_tool_calls);

    const results = arena.alloc(
        ToolResult,
        calls.len,
    ) catch return &.{};

    for (calls, 0..) |call, i| {
        results[i] = executeSingleTool(
            config,
            arena,
            call,
        );
    }
    return results;
}

/// Execute one tool call and return the result.
pub fn executeSingleTool(
    config: *const AgentConfig,
    arena: Allocator,
    call: ToolCall,
) ToolResult {
    std.debug.assert(call.id.len > 0);
    std.debug.assert(call.name.len > 0);

    api.logToolCall(call.name, call.input_json);

    const tool_start = timestampMs();
    const output = dispatchTool(config, arena, call);
    const tool_elapsed = timestampMs() - tool_start;

    const status = if (output.success) "ok" else "ERR";
    api.log(
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

    // Persist output for configured tool names.
    if (output.success) {
        for (config.persist_tools) |pt| {
            if (api.eql(call.name, pt)) {
                appendExperiment(
                    arena,
                    config.history_dir,
                    output.stdout,
                );
                api.log(
                    "    -> {s} result persisted\n",
                    .{call.name},
                );
                break;
            }
        }
    }

    return .{
        .tool_use_id = call.id,
        .content = api.truncate(
            output.stdout,
            config.max_tool_output,
        ),
        .is_error = !output.success,
    };
}

// ============================================================
// Prompt loading
// ============================================================

/// Load a prompt file from disk.  Returns the file
/// contents or a short fallback string if missing.
pub fn loadPromptFile(
    arena: Allocator,
    path: []const u8,
) []const u8 {
    std.debug.assert(path.len > 0);

    const file = std.fs.cwd().openFile(
        path,
        .{},
    ) catch |err| {
        api.log(
            "WARNING: cannot open {s}: {s}\n",
            .{ path, @errorName(err) },
        );
        return "(prompt file not found)";
    };
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_FILE_SIZE,
    ) catch {
        api.log(
            "WARNING: cannot read {s}\n",
            .{path},
        );
        return "(prompt file unreadable)";
    };

    if (content.len == 0) {
        api.log(
            "WARNING: {s} is empty.\n",
            .{path},
        );
        return "(empty prompt file)";
    }

    api.log(
        "Loaded prompt: {s} ({d} KB)\n",
        .{ path, content.len / 1024 },
    );
    return content;
}

// ============================================================
// Setup helpers
// ============================================================

/// Build the toolbox with `zig build`.
pub fn buildToolbox(arena: Allocator) bool {
    const build_start = timestampMs();
    api.log("Building toolbox...\n", .{});

    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "zig", "build" },
        .max_output_bytes = MAX_OUTPUT_BYTES,
    }) catch {
        api.log("  spawn failed.\n", .{});
        return false;
    };

    const build_ms = timestampMs() - build_start;
    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (!ok) {
        api.log("Build failed:\n{s}\n", .{result.stderr});
    } else {
        api.log(
            "  done ({s}).\n",
            .{nanosToMsStr(arena, build_ms * 1_000_000)},
        );
    }
    return ok;
}

/// Load API key from ANTHROPIC_API_KEY env var.
pub fn loadApiKey() ?[]const u8 {
    const val = std.posix.getenv("ANTHROPIC_API_KEY");
    if (val) |key| {
        std.debug.assert(key.len > 0);
        return key;
    }
    return null;
}

/// Load model from ANTHROPIC_MODEL env var, or default.
pub fn loadModel() []const u8 {
    const val = std.posix.getenv("ANTHROPIC_MODEL");
    if (val) |model| {
        std.debug.assert(model.len > 0);
        return model;
    }
    return DEFAULT_MODEL;
}

// ============================================================
// Message building
// ============================================================

/// Default context builder when no callback is set.
/// Includes history, summaries, and orientation.
fn buildDefaultContext(
    arena: Allocator,
    config: *const AgentConfig,
) []const u8 {
    const history = buildHistorySummary(
        arena,
        config,
    );
    const summaries = buildSummariesSection(
        arena,
        config.history_dir,
    );
    const orientation = buildOrientation(arena);
    const has_history = history.len > 0;

    const hist_section = if (has_history)
        "## Benchmark history\n\n"
    else
        "No previous experiment history. " ++
            "This is the first run.\n\n";

    const suffix =
        "\n\nBegin optimizing. Start by exploring " ++
        "the current state.";

    return std.fmt.allocPrint(
        arena,
        "{s}\n{s}{s}{s}{s}",
        .{
            orientation,
            hist_section,
            if (has_history) history else "",
            summaries,
            suffix,
        },
    ) catch "Begin optimizing.";
}

/// Build a filesystem orientation section.
pub fn buildOrientation(arena: Allocator) []const u8 {
    const cwd = tools.cwdAbsolute(arena) catch
        return "(cwd unavailable)\n";

    const root_ls = runQuietLs(arena, "..") orelse
        "(cannot list)";
    const nn_ls = runQuietLs(arena, "../nn") orelse
        "(cannot list)";

    return std.fmt.allocPrint(
        arena,
        "## Filesystem orientation\n\n" ++
            "run_command cwd: {s}\n" ++
            "Monorepo root is \"../\" from cwd.\n\n" ++
            "Monorepo root contents:\n{s}\n" ++
            "nn/ contents:\n{s}\n",
        .{ cwd, root_ls, nn_ls },
    ) catch "(orientation unavailable)\n";
}

/// Run `ls -1` on a directory and return stdout.
fn runQuietLs(
    arena: Allocator,
    dir: []const u8,
) ?[]const u8 {
    std.debug.assert(dir.len > 0);
    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "/bin/ls", "-1", dir },
        .max_output_bytes = 4096,
    }) catch return null;

    const ok = switch (result.term) {
        .Exited => |code| code == 0,
        else => false,
    };
    if (!ok or result.stdout.len == 0) return null;
    return result.stdout;
}

// ============================================================
// History persistence
//
// Append-only JSONL at {history_dir}/experiments.jsonl.
// Each line is a compacted benchmark JSON from a tool
// call.  Loaded on startup and injected into the first
// user message as context.
// ============================================================

/// Ensure the history directory exists.
pub fn ensureHistoryDir(
    arena: Allocator,
    history_dir: []const u8,
) void {
    const fs_path = resolveToFs(
        arena,
        history_dir,
    ) orelse return;
    std.fs.cwd().makeDir(fs_path) catch |err| {
        if (err != error.PathAlreadyExists) {
            api.log(
                "WARNING: mkdir {s}: {s}\n",
                .{ fs_path, @errorName(err) },
            );
        }
    };
}

/// Append a benchmark result to experiments.jsonl.
pub fn appendExperiment(
    arena: Allocator,
    history_dir: []const u8,
    benchmark_json: []const u8,
) void {
    std.debug.assert(benchmark_json.len > 0);

    const line = api.collapseToLine(
        arena,
        benchmark_json,
    ) catch return;

    const hist_path = std.fmt.allocPrint(
        arena,
        "{s}/experiments.jsonl",
        .{history_dir},
    ) catch return;
    const fs_path = resolveToFs(
        arena,
        hist_path,
    ) orelse return;

    const file = std.fs.cwd().createFile(
        fs_path,
        .{ .truncate = false },
    ) catch return;
    defer file.close();

    file.seekFromEnd(0) catch return;
    file.writeAll(line) catch return;
    file.writeAll("\n") catch {};
}

/// Build a compact summary table from experiments.jsonl.
pub fn buildHistorySummary(
    arena: Allocator,
    config: *const AgentConfig,
) []const u8 {
    const max_visible: u32 = 10;

    const hist_path = std.fmt.allocPrint(
        arena,
        "{s}/experiments.jsonl",
        .{config.history_dir},
    ) catch return "";
    const fs_path = resolveToFs(
        arena,
        hist_path,
    ) orelse return "";

    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch return "";
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch return "";
    if (content.len == 0) return "";

    var lines: [512][]const u8 = undefined;
    var line_count: u32 = 0;
    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        if (line.len < 2) continue;
        if (line_count < 512) {
            lines[line_count] = line;
            line_count += 1;
        }
    }
    if (line_count == 0) return "";

    const start = if (line_count > max_visible)
        line_count - max_visible
    else
        0;
    const visible = line_count - start;

    api.log(
        "History: {d} experiments " ++
            "(showing last {d}).\n",
        .{ line_count, visible },
    );

    return formatHistoryLines(
        config,
        arena,
        lines[0..line_count],
        start,
        line_count,
        visible,
    );
}

/// Format the visible history lines into a summary.
fn formatHistoryLines(
    config: *const AgentConfig,
    arena: Allocator,
    lines: []const []const u8,
    start: u32,
    total: u32,
    visible: u32,
) []const u8 {
    var buf: std.ArrayList(u8) = .empty;
    const header = std.fmt.allocPrint(
        arena,
        "{d} experiments total " ++
            "(last {d} shown):\n\n",
        .{ total, visible },
    ) catch return "";
    buf.appendSlice(arena, header) catch return "";

    var idx: u32 = start;
    while (idx < total) : (idx += 1) {
        const row = formatHistoryLine(
            config,
            arena,
            idx + 1,
            lines[idx],
        ) orelse continue;
        buf.appendSlice(arena, row) catch continue;
    }

    std.debug.assert(buf.items.len > 0);
    return buf.items;
}

/// Extract key metrics from one JSONL line into a
/// compact human-readable row.  Walks config.history_fields
/// so each domain controls which metrics appear.
fn formatHistoryLine(
    config: *const AgentConfig,
    arena: Allocator,
    index: u32,
    line: []const u8,
) ?[]const u8 {
    std.debug.assert(line.len > 0);
    std.debug.assert(index > 0);

    const root = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        line,
        .{},
    ) catch return null;
    const obj = switch (root) {
        .object => |o| o,
        else => return null,
    };

    const ts = getStrOr(obj, "timestamp_utc", "?");

    // Fallback: no fields configured — show raw line.
    if (config.history_fields.len == 0) {
        return std.fmt.allocPrint(
            arena,
            "  {d}. {s}  (raw) {s}\n",
            .{ index, ts, line },
        ) catch null;
    }

    var buf: std.ArrayList(u8) = .empty;
    const prefix = std.fmt.allocPrint(
        arena,
        "  {d}. {s}",
        .{ index, ts },
    ) catch return null;
    buf.appendSlice(arena, prefix) catch return null;

    for (config.history_fields) |field| {
        const val = getNumStr(
            arena,
            obj,
            field.json_key,
        );
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

/// Load experiment summaries from summaries.jsonl.
pub fn buildSummariesSection(
    arena: Allocator,
    history_dir: []const u8,
) []const u8 {
    const sum_path = std.fmt.allocPrint(
        arena,
        "{s}/summaries.jsonl",
        .{history_dir},
    ) catch return "";
    const fs_path = resolveToFs(
        arena,
        sum_path,
    ) orelse return "";

    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch return "";
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch return "";
    if (content.len == 0) return "";

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "\n## Previous experiment summaries\n\n" ++
            "These describe what was tried and " ++
            "why it succeeded or failed. Do NOT " ++
            "repeat failed approaches.\n\n",
    ) catch return "";

    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        if (line.len < 2) continue;
        formatSummaryLine(arena, &buf, line);
    }

    // Header was added but no summaries parsed.
    if (buf.items.len < 80) return "";

    std.debug.assert(buf.items.len > 0);
    return buf.items;
}

/// Format one summary JSONL line and append to buffer.
fn formatSummaryLine(
    arena: Allocator,
    buf: *std.ArrayList(u8),
    line: []const u8,
) void {
    const root = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        line,
        .{},
    ) catch return;
    const obj = switch (root) {
        .object => |o| o,
        else => return,
    };
    const exp_num = getNumStr(arena, obj, "experiment");
    const summary = switch (obj.get("summary") orelse return) {
        .string => |s| s,
        else => return,
    };

    const row = std.fmt.allocPrint(
        arena,
        "  Experiment {s}: {s}\n",
        .{ exp_num, summary },
    ) catch return;
    buf.appendSlice(arena, row) catch {};
}

// ============================================================
// JSON helpers
// ============================================================

/// Parse token counts from a saved API response.
pub fn parseTokenUsage(
    arena: Allocator,
    raw: []const u8,
) struct { u64, u64 } {
    std.debug.assert(raw.len > 0);
    const root = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        raw,
        .{},
    ) catch return .{ 0, 0 };
    const obj = switch (root) {
        .object => |o| o,
        else => return .{ 0, 0 },
    };
    const usage = switch (obj.get("usage") orelse .null) {
        .object => |u| u,
        else => obj,
    };
    const in_val: u64 = switch (usage.get("input_tokens") orelse .null) {
        .integer => |i| @intCast(@max(0, i)),
        else => 0,
    };
    const out_val: u64 = switch (usage.get("output_tokens") orelse .null) {
        .integer => |i| @intCast(@max(0, i)),
        else => 0,
    };
    return .{ in_val, out_val };
}

/// Extract a string field, returning default if missing.
pub fn getStrOr(
    obj: std.json.ObjectMap,
    key: []const u8,
    default: []const u8,
) []const u8 {
    const val = obj.get(key) orelse return default;
    return switch (val) {
        .string => |s| s,
        else => default,
    };
}

/// Extract a numeric field as a printable string.
pub fn getNumStr(
    arena: Allocator,
    obj: std.json.ObjectMap,
    key: []const u8,
) []const u8 {
    const val = obj.get(key) orelse return "?";
    return switch (val) {
        .integer => |i| std.fmt.allocPrint(
            arena,
            "{d}",
            .{i},
        ) catch "?",
        .float => |f| std.fmt.allocPrint(
            arena,
            "{d:.2}",
            .{f},
        ) catch "?",
        .number_string => |s| s,
        else => "?",
    };
}

// ============================================================
// Path resolution
// ============================================================

/// Convert a monorepo-root-relative path to a filesystem
/// path relative to the zap/ working directory.
/// Wraps tools.resolveToFs with optional semantics.
pub fn resolveToFs(
    arena: Allocator,
    path: []const u8,
) ?[]const u8 {
    if (path.len == 0) return null;
    return tools.resolveToFs(arena, "..", path) catch null;
}

// ============================================================
// Timing helpers
// ============================================================

/// Current wall-clock time in milliseconds.
pub fn timestampMs() i64 {
    const nanos: i128 = std.time.nanoTimestamp();
    return @intCast(@divFloor(nanos, 1_000_000));
}

/// Format a duration as a human-readable string.
/// "123ms" for sub-second, "1.234s" for seconds,
/// "2m05s" for minutes.
pub fn nanosToMsStr(
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

/// Sum the byte lengths of all messages.
pub fn contextSizeBytes(
    messages: []const []const u8,
) usize {
    var total: usize = 0;
    for (messages) |msg| {
        total += msg.len;
    }
    return total;
}

/// Truncate text at the last sentence boundary within
/// max_len bytes.
pub fn truncateAtSentence(
    text: []const u8,
    max_len: usize,
) []const u8 {
    std.debug.assert(max_len > 0);
    if (text.len <= max_len) return text;

    var pos: usize = max_len;
    while (pos > 0) {
        pos -= 1;
        const c = text[pos];
        const is_ender = c == '.' or c == '!' or
            c == '?' or c == ')';
        if (!is_ender) continue;

        if (pos + 1 >= max_len) {
            return text[0 .. pos + 1];
        }
        const next = text[pos + 1];
        if (next == ' ' or next == '\n' or
            next == '\r')
        {
            return text[0 .. pos + 1];
        }
    }

    // No sentence boundary — hard cut.
    return text[0..max_len];
}

// ============================================================
// Logging
// ============================================================

pub fn printHeader(name: []const u8) void {
    api.log(
        "\nnnzap {s} agent" ++
            " — LLM-powered experiment runner\n" ++
            "========================================" ++
            "======\n\n",
        .{name},
    );
}

test "comptime tool schema generation" {
    const defs = [_]ToolDef{
        .{
            .name = "list_items",
            .subcommand = "list-items",
            .description = "List all items.",
        },
        .{
            .name = "read_file",
            .subcommand = "read-file",
            .description = "Read a file.",
            .properties = &.{.{
                .name = "path",
                .description = "File path to read.",
            }},
        },
        .{
            .name = "search",
            .subcommand = "search",
            .description = "Search with options.",
            .properties = &.{
                .{
                    .name = "query",
                    .description = "Search query.",
                },
                .{
                    .name = "limit",
                    .description = "Max results.",
                    .type = .integer,
                    .required = false,
                },
            },
        },
    };

    const maps = comptime toolMappings(&defs);
    const json = comptime toolSchemas(&defs);

    // Mapping count matches def count.
    comptime std.debug.assert(maps.len == 3);

    // Shapes derived correctly from property count.
    comptime std.debug.assert(maps[0].shape == .no_input);
    comptime std.debug.assert(maps[1].shape == .string_arg);
    comptime std.debug.assert(maps[2].shape == .json_payload);

    // Field derived for the string_arg tool.
    comptime std.debug.assert(
        std.mem.eql(u8, maps[1].field.?, "path"),
    );
    comptime std.debug.assert(maps[0].field == null);
    comptime std.debug.assert(maps[2].field == null);

    // JSON is a valid array envelope.
    comptime std.debug.assert(json[0] == '[');
    comptime std.debug.assert(json[json.len - 1] == ']');

    // JSON contains each tool name.
    comptime std.debug.assert(
        std.mem.indexOf(u8, json, "list_items") != null,
    );
    comptime std.debug.assert(
        std.mem.indexOf(u8, json, "read_file") != null,
    );
    comptime std.debug.assert(
        std.mem.indexOf(u8, json, "search") != null,
    );
}
