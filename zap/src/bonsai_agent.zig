//! Bonsai research agent — optimises Bonsai 1.7B
//! inference on Apple Silicon.
//!
//! Single-tier mode: Claude drives everything.
//! Two-tier mode: Opus strategist + local LLM executor.

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("agent_core.zig");
const api = core.api;
const ollama = @import("ollama_client.zig");

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const DEFAULT_LOCAL_MODEL: []const u8 =
    "mlx-qwen3.5-27b-claude-4.6-opus-" ++
    "reasoning-distilled-v2";
const MAX_TURNS_PER_EXECUTION: u32 = 60;
const LOCAL_LLM_MAX_TOKENS_STR: []const u8 = "16384";
const MAX_MESSAGES: u32 = 512;
const MAX_API_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_CODE_CONTEXT: usize = 4 * 1024 * 1024;
const MAX_FILE_SIZE: usize = 2 * 1024 * 1024;
const MAX_EXPERIMENTS: u32 = 50;
const HISTORY_DIR: []const u8 = ".bonsai_history";

/// Source files included as code context for the Opus
/// strategist.  These are the hot-path files the
/// strategist needs to design concrete experiments.
/// Read once at startup (Rule 8 — amortise upfront).
const CODE_CONTEXT_FILES = [_][]const u8{
    "nn/src/shaders/transformer.metal",
    "nn/src/shaders/compute.metal",
    "nn/src/transformer.zig",
    "nn/src/metal.zig",
    "nn/src/model.zig",
};

// ============================================================
// Tool mapping table
//
// Maps each LLM tool name to a CLI subcommand on the
// bonsai_research toolbox binary.  21 entries matching
// bonsai_tools.json.
// ============================================================

const bonsai_tools = [_]core.ToolMapping{
    .{ .tool_name = "snapshot", .subcommand = "snapshot", .shape = .no_input },
    .{ .tool_name = "snapshot_list", .subcommand = "snapshot-list", .shape = .no_input },
    .{ .tool_name = "rollback", .subcommand = "rollback", .shape = .string_arg, .field = "id" },
    .{ .tool_name = "rollback_latest", .subcommand = "rollback-latest", .shape = .no_input },
    .{ .tool_name = "diff", .subcommand = "diff", .shape = .string_arg, .field = "id" },
    .{ .tool_name = "check", .subcommand = "check", .shape = .no_input },
    .{ .tool_name = "test", .subcommand = "test", .shape = .no_input },
    .{ .tool_name = "bench", .subcommand = "bench", .shape = .no_input },
    .{ .tool_name = "bench_infer", .subcommand = "bench-infer", .shape = .no_input },
    .{ .tool_name = "bench_compare", .subcommand = "bench-compare", .shape = .no_input },
    .{ .tool_name = "history", .subcommand = "history", .shape = .string_arg, .field = "count" },
    .{ .tool_name = "show", .subcommand = "show", .shape = .string_arg, .field = "file" },
    .{ .tool_name = "show_function", .subcommand = "show-function", .shape = .json_payload },
    .{ .tool_name = "read_file", .subcommand = "read-file", .shape = .string_arg, .field = "path" },
    .{ .tool_name = "write_file", .subcommand = "write-file", .shape = .json_payload },
    .{ .tool_name = "edit_file", .subcommand = "edit-file", .shape = .json_payload },
    .{ .tool_name = "list_directory", .subcommand = "list-dir", .shape = .string_arg, .field = "path" },
    .{ .tool_name = "cwd", .subcommand = "cwd", .shape = .no_input },
    .{ .tool_name = "run_command", .subcommand = "run-cmd", .shape = .json_payload },
    .{ .tool_name = "commit", .subcommand = "commit", .shape = .json_payload },
    .{ .tool_name = "add_summary", .subcommand = "add-summary", .shape = .json_payload },
};

// ============================================================
// Profile configuration
// ============================================================

const config = core.AgentConfig{
    .name = "bonsai",
    .toolbox_path = "./zig-out/bin/bonsai_research",
    .history_dir = HISTORY_DIR,
    .system_prompt_path = "programs/bonsai_system.md",
    .tool_schemas_path = "programs/bonsai_tools.json",
    .tool_map = &bonsai_tools,
    .persist_tools = &.{ "bench", "bench_infer" },
    .build_context_fn = &buildBonsaiContext,
};

// ============================================================
// Entry point
// ============================================================

pub fn main() void {
    mainInner() catch |err| {
        api.log(
            "\nFATAL: agent crashed: {s}\n",
            .{@errorName(err)},
        );
        api.log(
            "Hint: if the agent edited source files " ++
                "before crashing, run:\n" ++
                "  ./zig-out/bin/bonsai_research " ++
                "rollback-latest\n",
            .{},
        );
        std.process.exit(1);
    };
}

fn mainInner() !void {
    // Check for two-tier mode first.
    const local_model = loadLocalModel();
    if (local_model) |executor_model| {
        return runTwoTierMode(executor_model);
    }
    // Single-tier: use the generic loop.
    return core.run(&config);
}

// ============================================================
// Context builder callback
//
// Builds the initial user message with orientation,
// engineering rules, benchmark history, and summaries.
// ============================================================

fn buildBonsaiContext(
    arena: Allocator,
    cfg: *const core.AgentConfig,
) []const u8 {
    const orientation = core.buildOrientation(arena);
    const rules = loadEngineeringRules(arena);
    const history = core.buildHistorySummary(
        arena,
        cfg.history_dir,
    );
    const summaries = core.buildSummariesSection(
        arena,
        cfg.history_dir,
    );
    const has_history = history.len > 0;

    const rules_section =
        "## Engineering rules (CLAUDE.md)\n\n" ++
        "You MUST follow these rules when editing " ++
        "source code. They are non-negotiable " ++
        "— assertion density, function length " ++
        "limits, naming, explicit control flow, " ++
        "and all other rules apply to every line " ++
        "you write.\n\n";

    const hist_section = if (has_history)
        "## Benchmark history (compact)\n\n" ++
            "Use the history tool for full " ++
            "experiment details.\n\n"
    else
        "No previous benchmark history. " ++
            "This is the first run.\n\n";

    const suffix =
        "\n\n## Begin\n\n" ++
        "Optimise engine throughput. " ++
        "Start by calling snapshot to create a " ++
        "restore point, then read the source " ++
        "code to understand the baseline.";

    return std.fmt.allocPrint(
        arena,
        "{s}\n{s}{s}\n\n{s}{s}{s}{s}",
        .{
            orientation,
            rules_section,
            rules,
            hist_section,
            if (has_history) history else "",
            summaries,
            suffix,
        },
    ) catch "Begin optimising.";
}

// ============================================================
// Engineering rules loader
// ============================================================

/// Load CLAUDE.md engineering rules from disk.
/// Returns the file contents or a fallback message.
fn loadEngineeringRules(arena: Allocator) []const u8 {
    const path = "CLAUDE.md";
    const fs_path = core.resolveToFs(
        arena,
        path,
    ) orelse {
        api.log(
            "WARNING: cannot resolve {s}\n",
            .{path},
        );
        return "(CLAUDE.md not found — " ++
            "follow standard engineering practices)";
    };
    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch |err| {
        api.log(
            "WARNING: cannot open {s}: {s}\n",
            .{ path, @errorName(err) },
        );
        return "(CLAUDE.md not found — " ++
            "follow standard engineering practices)";
    };
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_FILE_SIZE,
    ) catch |err| {
        api.log(
            "WARNING: cannot read {s}: {s}\n",
            .{ path, @errorName(err) },
        );
        return "(CLAUDE.md too large or unreadable)";
    };

    if (content.len == 0) {
        api.log(
            "WARNING: {s} is empty.\n",
            .{path},
        );
        return "(CLAUDE.md is empty)";
    }

    api.log(
        "Engineering rules: {d} KB ({s})\n",
        .{ content.len / 1024, path },
    );
    return content;
}

// ============================================================
// Local model loader
// ============================================================

/// Load the local LLM executor model from
/// LOCAL_LLM_MODEL.  Returns null when not set
/// (single-tier mode).  Falls back to
/// DEFAULT_LOCAL_MODEL if set to "default".
fn loadLocalModel() ?[]const u8 {
    const val = std.posix.getenv("LOCAL_LLM_MODEL");
    if (val) |model| {
        std.debug.assert(model.len > 0);
        if (api.eql(model, "default")) {
            return DEFAULT_LOCAL_MODEL;
        }
        return model;
    }
    return null;
}

// ============================================================
// Two-tier mode
//
// Opus strategist designs experiments (one API call
// per experiment, high-quality reasoning).  A local
// Ollama model executes the plan by driving the tool
// loop (free, fast, many tool calls).
// ============================================================

/// Two-tier execution statistics.
const ExecutionResult = struct {
    turns: u32,
    tool_calls: u32,
    api_ms: i64,
    tool_ms: i64,
    input_tokens: u64,
    output_tokens: u64,
    completed: bool,
};

/// Entry point for two-tier mode.  Sets up arena,
/// keys, and toolbox, then delegates to the outer
/// experiment loop.
fn runTwoTierMode(ollama_model: []const u8) !void {
    std.debug.assert(ollama_model.len > 0);

    var arena_state = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const run_start = core.timestampMs();

    core.printHeader("bonsai (two-tier)");
    core.ensureHistoryDir(arena, HISTORY_DIR);

    const api_key = core.loadApiKey() orelse {
        api.fatal(
            "Set ANTHROPIC_API_KEY env var.\n" ++
                "  export ANTHROPIC_API_KEY=sk-ant-...\n",
        );
        unreachable;
    };
    const opus_model = core.loadModel();
    api.log("Opus model:  {s}\n", .{opus_model});
    api.log("Local model: {s}\n", .{ollama_model});

    if (!core.buildToolbox(arena)) {
        api.fatal("zig build failed.\n");
        unreachable;
    }

    // Load tool schemas and convert to OpenAI format
    // once at startup (Rule 8 — amortise upfront).
    const tool_schemas = core.loadPromptFile(
        arena,
        config.tool_schemas_path,
    );
    const openai_tools = ollama.convertAnthropicTools(
        arena,
        tool_schemas,
    ) catch {
        api.fatal("Failed to convert tool schemas.\n");
        unreachable;
    };

    // Load engineering rules once.
    const rules = loadEngineeringRules(arena);

    // Load hot-path source files once so the strategist
    // can design concrete, line-level experiments.
    const code_context = loadCodeContext(arena);
    api.log(
        "Code context: {d} KB from {d} files\n",
        .{
            code_context.len / 1024,
            CODE_CONTEXT_FILES.len,
        },
    );

    runTwoTierLoop(
        arena,
        run_start,
        api_key,
        opus_model,
        ollama_model,
        rules,
        code_context,
        openai_tools,
    );
}

/// Outer loop for two-tier mode.  Alternates between
/// Opus strategy calls and Ollama execution loops.
fn runTwoTierLoop(
    arena: Allocator,
    run_start: i64,
    api_key: []const u8,
    opus_model: []const u8,
    ollama_model: []const u8,
    rules: []const u8,
    code_context: []const u8,
    openai_tools: []const u8,
) void {
    std.debug.assert(opus_model.len > 0);
    std.debug.assert(ollama_model.len > 0);

    var total_experiments: u32 = 0;
    var total_turns: u32 = 0;
    var total_tool_calls: u32 = 0;
    var total_api_errors: u32 = 0;
    var total_api_ms: i64 = 0;
    var total_tool_ms: i64 = 0;
    var opus_in_tokens: u64 = 0;
    var opus_out_tokens: u64 = 0;

    var experiment: u32 = 0;
    while (experiment < MAX_EXPERIMENTS) : (experiment += 1) {
        api.log(
            "\n" ++
                "======================================" ++
                "========\n" ++
                "  Experiment {d}/{d} (two-tier)\n" ++
                "======================================" ++
                "========\n",
            .{ experiment + 1, MAX_EXPERIMENTS },
        );

        const should_stop = runOneStrategyExperiment(
            arena,
            api_key,
            opus_model,
            ollama_model,
            rules,
            code_context,
            openai_tools,
            experiment,
            &total_turns,
            &total_tool_calls,
            &total_api_errors,
            &total_api_ms,
            &total_tool_ms,
            &opus_in_tokens,
            &opus_out_tokens,
        );
        total_experiments += 1;
        if (should_stop) break;
    }

    printTwoTierSummary(
        arena,
        run_start,
        total_experiments,
        total_turns,
        total_tool_calls,
        total_api_errors,
        total_api_ms,
        total_tool_ms,
        opus_in_tokens,
        opus_out_tokens,
    );
}

/// Run one strategy→execution cycle.  Returns true if
/// the outer loop should stop (strategist failure).
fn runOneStrategyExperiment(
    arena: Allocator,
    api_key: []const u8,
    opus_model: []const u8,
    ollama_model: []const u8,
    rules: []const u8,
    code_context: []const u8,
    openai_tools: []const u8,
    experiment: u32,
    total_turns: *u32,
    total_tool_calls: *u32,
    total_api_errors: *u32,
    total_api_ms: *i64,
    total_tool_ms: *i64,
    opus_in: *u64,
    opus_out: *u64,
) bool {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(ollama_model.len > 0);

    // Phase 1: Opus designs the experiment.
    const history = core.buildHistorySummary(
        arena,
        HISTORY_DIR,
    );
    const summaries = core.buildSummariesSection(
        arena,
        HISTORY_DIR,
    );

    api.log("  Phase 1: Opus strategy...\n", .{});
    const strat_start = core.timestampMs();
    const plan_resp = callOpusStrategist(
        arena,
        api_key,
        opus_model,
        history,
        summaries,
        rules,
        code_context,
    );
    const strat_ms = core.timestampMs() - strat_start;
    total_api_ms.* += strat_ms;
    opus_in.* += plan_resp.input_tokens;
    opus_out.* += plan_resp.output_tokens;

    if (!plan_resp.success or
        plan_resp.text.len == 0)
    {
        api.log(
            "  Strategist failed ({s}): {s}\n",
            .{
                core.nanosToMsStr(
                    arena,
                    strat_ms * 1_000_000,
                ),
                plan_resp.error_message,
            },
        );
        total_api_errors.* += 1;
        return true; // Stop outer loop.
    }

    api.log(
        "  Plan received ({s}):\n",
        .{
            core.nanosToMsStr(
                arena,
                strat_ms * 1_000_000,
            ),
        },
    );
    api.logClaudeText(plan_resp.text);

    // Phase 2: Local LLM executes the plan.
    api.log(
        "\n  Phase 2: Local LLM executor...\n",
        .{},
    );
    const exec = runLocalExecution(
        arena,
        ollama_model,
        plan_resp.text,
        openai_tools,
    );

    total_turns.* += exec.turns;
    total_tool_calls.* += exec.tool_calls;
    total_api_ms.* += exec.api_ms;
    total_tool_ms.* += exec.tool_ms;
    if (!exec.completed) total_api_errors.* += 1;

    api.log(
        "  Experiment {d} done " ++
            "({d} turns, {d} tool calls).\n",
        .{
            experiment + 1,
            exec.turns,
            exec.tool_calls,
        },
    );
    return false; // Continue outer loop.
}

/// Call Opus to design a single experiment plan.
/// Returns the full ApiResponse with the plan in
/// .text.  Uses the strategist prompt (no tools).
fn callOpusStrategist(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    history: []const u8,
    summaries: []const u8,
    rules: []const u8,
    code_context: []const u8,
) core.ApiResponse {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(model.len > 0);

    const context = buildStrategyContext(
        arena,
        history,
        summaries,
        rules,
        code_context,
    );
    const user_msg = api.wrapUserTextMessage(
        arena,
        context,
    );
    const messages = [_][]const u8{user_msg};

    const strategist_prompt = core.loadPromptFile(
        arena,
        "programs/bonsai_strategist.md",
    );

    // Strategist: custom prompt, no tools (text only).
    return core.callApiWithRetry(
        arena,
        api_key,
        model,
        &messages,
        strategist_prompt,
        "[]",
        HISTORY_DIR,
        config.max_tokens_str,
        config.thinking_budget_str,
    );
}

/// Build the user message for the strategist.
/// Includes benchmark history, summaries, engineering
/// rules, and source code context.
fn buildStrategyContext(
    arena: Allocator,
    history: []const u8,
    summaries: []const u8,
    rules: []const u8,
    code_context: []const u8,
) []const u8 {
    std.debug.assert(rules.len > 0);

    const hist_header = if (history.len > 0)
        "## Benchmark history\n\n"
    else
        "No benchmark history yet.\n\n";

    const code_section = if (code_context.len > 0)
        code_context
    else
        "(code context unavailable)\n";

    return std.fmt.allocPrint(
        arena,
        "{s}{s}{s}" ++
            "## Engineering rules (CLAUDE.md)\n\n" ++
            "{s}\n\n" ++
            "## Source code (hot path)\n\n" ++
            "Below are the current source files for " ++
            "the engine hot path. Use these to design " ++
            "specific, line-level changes. Reference " ++
            "exact function names, buffer indices, " ++
            "thread counts, and kernel parameters." ++
            "\n\n{s}\n\n" ++
            "Design the next experiment.",
        .{
            hist_header,
            if (history.len > 0) history else "",
            if (summaries.len > 0) summaries else "",
            rules,
            code_section,
        },
    ) catch "Design an experiment to improve throughput.";
}

// ============================================================
// Code context loader
// ============================================================

/// Read all hot-path source files into a single code
/// context string with file headers.  Each file is
/// wrapped with a delimiter so the strategist can
/// reference specific locations.
fn loadCodeContext(arena: Allocator) []const u8 {
    var buf: std.ArrayList(u8) = .empty;

    for (&CODE_CONTEXT_FILES) |path| {
        const content = readFileContent(
            arena,
            path,
        ) orelse continue;

        // File header with path and line count.
        var line_count: u32 = 1;
        for (content) |c| {
            if (c == '\n') line_count += 1;
        }

        const header = std.fmt.allocPrint(
            arena,
            "### {s} ({d} lines)\n```\n",
            .{ path, line_count },
        ) catch continue;
        buf.appendSlice(arena, header) catch continue;
        buf.appendSlice(arena, content) catch continue;
        buf.appendSlice(
            arena,
            "\n```\n\n",
        ) catch continue;

        // Guard against runaway context size.
        if (buf.items.len > MAX_CODE_CONTEXT) {
            api.log(
                "  Code context truncated at " ++
                    "{d} KB (limit {d} KB)\n",
                .{
                    buf.items.len / 1024,
                    MAX_CODE_CONTEXT / 1024,
                },
            );
            break;
        }
    }

    std.debug.assert(
        buf.items.len <= MAX_CODE_CONTEXT + MAX_FILE_SIZE,
    );
    return buf.items;
}

/// Read a file from a monorepo-relative path.
fn readFileContent(
    arena: Allocator,
    path: []const u8,
) ?[]const u8 {
    std.debug.assert(path.len > 0);

    const fs_path = core.resolveToFs(
        arena,
        path,
    ) orelse return null;

    const file = std.fs.cwd().openFile(
        fs_path,
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

// ============================================================
// Local LLM execution loop
// ============================================================

/// Execute an experiment plan using the local LLM.
/// Drives the tool-calling loop: snapshot, edit, check,
/// test, bench, report.
fn runLocalExecution(
    arena: Allocator,
    model: []const u8,
    plan: []const u8,
    openai_tools: []const u8,
) ExecutionResult {
    std.debug.assert(model.len > 0);
    std.debug.assert(plan.len > 0);

    const executor_prompt = core.loadPromptFile(
        arena,
        "programs/bonsai_executor.md",
    );
    const first_msg = buildExecutorInitialMsg(
        arena,
        plan,
    );
    const history_dir = "../" ++ HISTORY_DIR;

    var msgs: [MAX_MESSAGES][]const u8 = undefined;
    var count: u32 = 0;
    msgs[0] = first_msg;
    count = 1;

    var r = ExecutionResult{
        .turns = 0,
        .tool_calls = 0,
        .api_ms = 0,
        .tool_ms = 0,
        .input_tokens = 0,
        .output_tokens = 0,
        .completed = false,
    };

    var turn: u32 = 0;
    while (turn < MAX_TURNS_PER_EXECUTION) : (turn += 1) {
        const should_stop = runOneExecutorTurn(
            arena,
            model,
            executor_prompt,
            openai_tools,
            history_dir,
            &msgs,
            &count,
            turn,
            &r,
        );
        if (should_stop) break;
    }

    r.turns = turn + 1;
    return r;
}

/// Run one turn of the local executor loop.
/// Returns true if the loop should break.
fn runOneExecutorTurn(
    arena: Allocator,
    model: []const u8,
    executor_prompt: []const u8,
    openai_tools: []const u8,
    history_dir: []const u8,
    msgs: *[MAX_MESSAGES][]const u8,
    count: *u32,
    turn: u32,
    r: *ExecutionResult,
) bool {
    api.log(
        "\n  --- Executor turn {d} ---\n",
        .{turn + 1},
    );
    const t0 = core.timestampMs();
    const resp = ollama.callApi(
        arena,
        model,
        executor_prompt,
        msgs[0..count.*],
        openai_tools,
        LOCAL_LLM_MAX_TOKENS_STR,
        history_dir,
    );
    r.api_ms += core.timestampMs() - t0;

    if (!resp.success) {
        api.log(
            "  Local LLM error: {s}\n",
            .{resp.error_message},
        );
        return true;
    }

    r.input_tokens += resp.input_tokens;
    r.output_tokens += resp.output_tokens;
    if (resp.text.len > 0) api.logClaudeText(resp.text);

    // Append assistant message (OpenAI format).
    msgs[count.*] = ollama.buildAssistantMsg(
        arena,
        resp,
    );
    count.* += 1;

    // Model chose to stop — no tool calls.
    if (!api.eql(resp.stop_reason, "tool_use") or
        resp.tool_calls.len == 0)
    {
        r.completed = true;
        return true;
    }

    // Execute tools and append results.
    const t1 = core.timestampMs();
    const results = core.executeTools(
        &config,
        arena,
        resp.tool_calls,
    );
    r.tool_ms += core.timestampMs() - t1;
    r.tool_calls += @intCast(resp.tool_calls.len);

    // OpenAI format: one message per tool result.
    for (results) |tr| {
        if (count.* >= MAX_MESSAGES - 1) break;
        msgs[count.*] = ollama.buildToolResultMsg(
            arena,
            tr,
        );
        count.* += 1;
    }
    if (count.* >= MAX_MESSAGES - 2) return true;
    return false;
}

/// Build the initial user message for the executor,
/// containing filesystem orientation and the plan
/// from the Opus strategist.
fn buildExecutorInitialMsg(
    arena: Allocator,
    plan: []const u8,
) []const u8 {
    std.debug.assert(plan.len > 0);

    const orientation = core.buildOrientation(arena);
    const text = std.fmt.allocPrint(
        arena,
        "{s}\n\n## Experiment plan\n\n" ++
            "A senior strategist designed this " ++
            "experiment for you. Follow it " ++
            "precisely:\n\n{s}\n\n" ++
            "## Begin\n\n" ++
            "Execute this plan now. Start by " ++
            "calling snapshot to create a restore " ++
            "point, then proceed step by step.",
        .{ orientation, plan },
    ) catch "Execute the experiment plan.";

    return ollama.buildUserTextMsg(arena, text);
}

// ============================================================
// Two-tier summary
// ============================================================

/// Print the run summary for two-tier mode.
fn printTwoTierSummary(
    arena: Allocator,
    run_start: i64,
    total_experiments: u32,
    total_turns: u32,
    total_tool_calls: u32,
    total_api_errors: u32,
    total_api_ms: i64,
    total_tool_ms: i64,
    opus_in_tokens: u64,
    opus_out_tokens: u64,
) void {
    const run_elapsed = core.timestampMs() - run_start;

    // Opus pricing: $5/MTok in, $25/MTok out.
    // Ollama is free (local).
    const cost_cents =
        (opus_in_tokens * 5 +
            opus_out_tokens * 25) / 10_000;
    const cost_str = std.fmt.allocPrint(
        arena,
        "${d}.{d:0>2}",
        .{ cost_cents / 100, cost_cents % 100 },
    ) catch "$?.??";

    api.log(
        "\n" ++
            "======================================" ++
            "========\n" ++
            "  Run summary (two-tier, bonsai)\n" ++
            "======================================" ++
            "========\n" ++
            "  Experiments:    {d}\n" ++
            "  Total turns:    {d} (Ollama)\n" ++
            "  Tool calls:     {d}\n" ++
            "  API errors:     {d}\n" ++
            "  Opus tokens:    {d} in, {d} out\n" ++
            "  Est. cost:      {s} (Opus only)\n" ++
            "  API time:       {s}\n" ++
            "  Tool time:      {s}\n" ++
            "  Total time:     {s}\n" ++
            "======================================" ++
            "========\n",
        .{
            total_experiments,
            total_turns,
            total_tool_calls,
            total_api_errors,
            opus_in_tokens,
            opus_out_tokens,
            cost_str,
            core.nanosToMsStr(
                arena,
                total_api_ms * 1_000_000,
            ),
            core.nanosToMsStr(
                arena,
                total_tool_ms * 1_000_000,
            ),
            core.nanosToMsStr(
                arena,
                run_elapsed * 1_000_000,
            ),
        },
    );
}
