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
const Allocator = std.mem.Allocator;
const tools = @import("tools.zig");
const api = @import("api_client.zig");
const ollama = @import("ollama_client.zig");

// ============================================================
// Constants (Rule 4 — hard limits)
// ============================================================

const MAX_EXPERIMENTS: u32 = 50;
const MAX_TURNS_PER_EXPERIMENT: u32 = 80;
const TURN_WARNING_THRESHOLD: u32 = 70;
const MAX_MESSAGES: u32 = 512;
const MAX_TOOL_CALLS: u32 = 16;
const MAX_TOOL_OUTPUT: usize = 50_000;
const MAX_TOOL_ARGS: u32 = 16;
const MAX_API_RESPONSE: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_HISTORY_SIZE: usize = 2 * 1024 * 1024;
const MAX_FILE_SIZE: usize = 2 * 1024 * 1024;
const MAX_COMMAND_OUTPUT: usize = 1 * 1024 * 1024;
const COMMAND_TIMEOUT_NS: u64 = 120 * std.time.ns_per_s;

const MAX_RETRY_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY_MS: u64 = 2_000;
const RETRY_MAX_DELAY_MS: u64 = 30_000;

// Curl timeout for the full API round-trip. Claude
// Opus on large contexts can think for minutes.
const API_TIMEOUT_SECS_STR: []const u8 = "600";

const API_URL: []const u8 =
    "https://api.anthropic.com/v1/messages";
const API_VERSION: []const u8 = "2023-06-01";
const DEFAULT_MODEL: []const u8 =
    "claude-opus-4-6";
const MAX_TOKENS_STR: []const u8 = "16384";

// Local LLM executor constants for two-tier mode.
// Must match the LM Studio model identifier shown
// after `lms load`.
const DEFAULT_LOCAL_MODEL: []const u8 =
    "mlx-qwen3.5-27b-claude-4.6-opus-" ++
    "reasoning-distilled-v2";
const MAX_TURNS_PER_EXECUTION: u32 = 60;
const LOCAL_LLM_MAX_TOKENS_STR: []const u8 = "16384";

// Source files included as code context for the Opus
// strategist.  These are the hot-path files the
// strategist needs to design concrete experiments.
// Read once at startup (Rule 8 — amortise upfront).
const CODE_CONTEXT_FILES = [_][]const u8{
    "nn/src/shaders/transformer.metal",
    "nn/src/shaders/compute.metal",
    "nn/src/transformer.zig",
    "nn/src/metal.zig",
    "nn/src/model.zig",
};
const MAX_CODE_CONTEXT: usize = 4 * 1024 * 1024;

const TOOL_PATH: []const u8 =
    "./zig-out/bin/engine_research";
const HISTORY_DIR: []const u8 =
    ".engine_agent_history";
const HISTORY_PATH: []const u8 =
    ".engine_agent_history/experiments.jsonl";
const SUMMARIES_PATH: []const u8 =
    ".engine_agent_history/summaries.jsonl";

// ============================================================
// Allowed paths for write_file / edit_file (safety guard)
// ============================================================

const ALLOWED_WRITE_FILES = [_][]const u8{
    "nn/src/transformer.zig",
    "nn/src/network.zig",
    "nn/src/layout.zig",
    "nn/src/model.zig",
    "nn/src/metal.zig",
    "nn/src/safetensors.zig",
    "nn/src/tokenizer.zig",
    "nn/src/shaders/transformer.metal",
    "nn/src/shaders/compute.metal",
    "nn/examples/bonsai.zig",
    "nn/examples/bonsai_bench.zig",
};

// ============================================================
// Engineering rules path (loaded at runtime)
// ============================================================

const CLAUDE_MD_PATH: []const u8 = "CLAUDE.md";

// ============================================================
// Allowed path prefixes for read_file / list_directory
// ============================================================

const ALLOWED_READ_PREFIXES = [_][]const u8{
    "nn/src/",
    "nn/examples/",
    "src/",
    "programs/",
    "docs/",
    "data/",
    "benchmarks/",
    ".engine_agent_history/",
    ".engine_snapshots/",
};

const ALLOWED_READ_FILES = [_][]const u8{
    "README.md",
    "CLAUDE.md",
    "nn/build.zig",
    "nn/build.zig.zon",
    "build.zig",
    "build.zig.zon",
};

// Type aliases for the shared API client types.
const ToolCall = api.ToolCall;
const ToolOutput = api.ToolOutput;
const ToolResult = api.ToolResult;
const ApiResponse = api.ApiResponse;

// ============================================================
// System prompt
//
// Seeds Claude with role, goal, protocol, constraints,
// and optimisation strategy.  Multiline string literal.
// ============================================================

const SYSTEM_PROMPT =
    \\You are an autonomous systems-performance research
    \\agent optimising Bonsai 1.7B inference throughput.
    \\nnzap is a Zig + Metal GPU-accelerated 1-bit
    \\language model inference engine for Apple Silicon
    \\with zero-copy unified memory.
    \\
    \\## Goal
    \\
    \\Maximise decode throughput (tok/s) for Bonsai 1.7B
    \\— a 1-bit Qwen3 architecture with 28 transformer
    \\blocks, 2048 hidden size, 16 query heads (GQA with
    \\8 KV heads), head_dim 128, group_size 128.
    \\
    \\The primary benchmark is bench (or bench_infer),
    \\which runs greedy decode and reports:
    \\
    \\  - decode_tok_per_sec (higher is better)
    \\  - decode_p50_us per token (lower is better)
    \\  - decode_p99_us per token (lower is better)
    \\  - prefill_tok_per_sec (secondary)
    \\
    \\All 71 tests must continue to pass.
    \\
    \\You edit source files directly. This is NOT prompt
    \\tuning — you change Metal kernels, dispatch
    \\strategies, buffer layouts, encoding patterns,
    \\and pipeline architecture.
    \\
    \\## Architecture overview
    \\
    \\The decode hot path for ONE token:
    \\
    \\  1. embedding_lookup — 1-bit packed, dequant to f32
    \\  2. For each of 28 blocks:
    \\     a. rms_norm (attn_norm)
    \\     b. qmv x3 (Q, K, V projections — 1-bit matmul)
    \\     c. rms_norm per-head (Q norm, K norm)
    \\     d. rope (rotary position embeddings)
    \\     e. kv_cache_update (f32 -> f16 write to cache)
    \\     f. gqa_attention (grouped query, decode M=1)
    \\     g. qmv x1 (O projection)
    \\     h. residual_add
    \\     i. rms_norm (ffn_norm)
    \\     j. qmv x2 (gate, up projections)
    \\     k. silu_elementwise_mul (SwiGLU)
    \\     l. qmv x1 (down projection)
    \\     m. residual_add
    \\  3. rms_norm (final norm)
    \\  4. qmv (lm_head — tied to embedding)
    \\
    \\Total per token: 7 qmv dispatches x 28 blocks
    \\+ 1 lm_head = 197 qmv calls. Plus ~6 lightweight
    \\kernels x 28 blocks = ~168 small dispatches.
    \\~365 kernel dispatches per token total.
    \\
    \\## Key kernel: qmv (1-bit matrix-vector multiply)
    \\
    \\The hot path. 197 calls per token. Current design:
    \\  - One simdgroup (32 threads) per output row.
    \\  - Two simdgroups per threadgroup (2 rows).
    \\  - Each lane processes K/32 columns.
    \\  - Inner loop: conditional accumulation via select().
    \\  - Uses identity: scale*(2*set_accum - group_sum).
    \\  - Reduction via simd_sum (hardware intrinsic).
    \\  - Grid: ceil(M/2) threadgroups, 64 threads each.
    \\
    \\## Files
    \\
    \\- nn/src/transformer.zig — TransformerConfig (comptime),
    \\  dispatch functions (dispatchQMV, dispatchRMSNorm,
    \\  etc.), forwardDecode, forwardBlock, sampling, all
    \\  GPU encoding. (~5000 lines, the main file)
    \\- nn/src/model.zig — Model struct, buffer allocation,
    \\  weight loading from safetensors.
    \\- nn/src/metal.zig — Metal device, Buffer, HalfBuffer,
    \\  PackedBuffer, command buffers, dispatch1D/2D/Custom.
    \\- nn/src/shaders/transformer.metal — Metal kernels:
    \\  rms_norm, rope, silu, gqa_attention,
    \\  embedding_lookup, residual_add, kv_cache_update.
    \\- nn/src/shaders/compute.metal — Metal kernels: qmv,
    \\  qmm, f32_to_1bit, matmul variants.
    \\- nn/src/safetensors.zig — Safetensors file parser.
    \\- nn/src/tokenizer.zig — BPE tokenizer.
    \\- nn/examples/bonsai.zig — CLI inference entry point.
    \\- nn/examples/bonsai_bench.zig — Benchmark binary.
    \\
    \\## Tools
    \\
    \\You have these tools available:
    \\
    \\Engine research (safety + measurement):
    \\  snapshot, snapshot_list, rollback, rollback_latest,
    \\  diff, check, test, bench, bench_infer,
    \\  bench_compare, history, show, show_function,
    \\  commit, add_summary
    \\
    \\File I/O (locked to project directory):
    \\  read_file, write_file, edit_file, list_directory
    \\
    \\Environment:
    \\  cwd — returns the absolute working directory for
    \\  run_command, list_directory, etc. Call this first
    \\  if you are unsure where commands execute.
    \\
    \\Shell (locked to project directory, 120s timeout):
    \\  run_command — execute any shell command, e.g.
    \\  grep, zig build, wc, head, etc.
    \\
    \\Prefer edit_file over write_file when making small
    \\changes. edit_file does targeted find-and-replace,
    \\which is faster and uses less context than rewriting
    \\the entire file.
    \\
    \\## Protocol
    \\
    \\This conversation is ONE experiment. You will:
    \\1. snapshot — safety net before any edits.
    \\2. Read the target code with show, show_function,
    \\   read_file, or run_command (e.g. grep).
    \\3. Edit the source with edit_file (targeted
    \\   find-and-replace) or write_file (full rewrite
    \\   for large changes). Read before you write.
    \\4. check — compile validation (~2s). If it fails,
    \\   read the error, fix or rollback_latest.
    \\5. test — numerical correctness (~5s). If it fails,
    \\   rollback_latest.
    \\6. bench — bonsai inference benchmark (~15s). This
    \\   is your primary metric.
    \\7. Evaluate: see decision rules below.
    \\8. Keep (commit with a descriptive message) or
    \\   rollback_latest + add_summary describing what
    \\   was tried and WHY it failed. Then try another
    \\   approach from step 2.
    \\
    \\When you have exhausted ideas or found a KEEP,
    \\STOP. The outer loop will start a new conversation
    \\for the next experiment.
    \\
    \\## Decision rules
    \\
    \\Primary metric: decode_tok_per_sec from bench.
    \\Secondary: decode_p50_us per token.
    \\
    \\- decode_tok_per_sec improves >= 5%: KEEP.
    \\- decode_p50_us improves >= 5%: KEEP.
    \\- Metrics flat but code enables future wins: KEEP.
    \\- Any primary metric regresses > 5%: ROLLBACK.
    \\- Test failure (71 tests): ROLLBACK immediately.
    \\- Compile failure: ROLLBACK immediately.
    \\
    \\## Constraints
    \\
    \\- check must pass before test or bench.
    \\- All 71 tests must pass. NEVER modify test
    \\  expectations or remove tests.
    \\- ONE optimisation per experiment. Isolate variables.
    \\- Read the full function before modifying it.
    \\- When adding Metal kernels, wire the pipeline in
    \\  metal.zig and transformer.zig. [[buffer(N)]] must
    \\  match setBuffer/setPackedBuffer calls.
    \\- Engineering rules: 70-line functions, >= 2
    \\  assertions per function, 100-column limit,
    \\  snake_case naming, comments explain WHY.
    \\
    \\## Current baseline
    \\
    \\Bonsai 1.7B (28 layers, 2048 hidden, Q1_0_g128):
    \\  Prefill: ~29 tok/s (single-token decode path)
    \\  Decode:  ~37 tok/s
    \\  ~365 Metal dispatches per token
    \\
    \\Reference (same hardware):
    \\  MLX Python: ~131 tok/s on M4 Pro.
    \\  The gap is dispatch overhead + kernel tuning.
    \\
    \\## Optimisation phases (priority order)
    \\
    \\Phase 1 — Reduce kernel dispatch overhead:
    \\  ~365 dispatches per token. Each Metal dispatch
    \\  has ~1-2 us overhead from Obj-C message sends.
    \\  Encoding all 28 blocks into a single command
    \\  buffer before committing (already done), but
    \\  fusing adjacent small kernels (rms_norm + qmv,
    \\  residual_add, silu_elementwise_mul) into fewer
    \\  dispatches could cut overhead significantly.
    \\  Another option: reuse compute encoders across
    \\  blocks instead of creating new ones.
    \\
    \\Phase 2 — qmv kernel optimisation:
    \\  197 calls/token, the dominant kernel. Options:
    \\  - Process more rows per threadgroup (4 or 8
    \\    simdgroups instead of 2).
    \\  - Unroll the inner byte loop (8 bits at a time
    \\    already, but could process 4 bytes/uint32 at
    \\    once).
    \\  - Cache input vector in threadgroup shared memory
    \\    so multiple output rows share the same loads.
    \\  - qmv_fast variant: skip bounds checks when K is
    \\    aligned (K%512==0, which Bonsai always is).
    \\  - Pack input reads: load float4 instead of float.
    \\
    \\Phase 3 — Attention kernel:
    \\  gqa_attention does a full KV scan per query head.
    \\  For long contexts this dominates. Flash-attention
    \\  tiling, f16 arithmetic in attention, or splitting
    \\  the softmax into online chunks could help.
    \\
    \\Phase 4 — Memory and precision:
    \\  - f16 activations (currently f32 throughout).
    \\  - Batch multiple qmv calls into one dispatch by
    \\    stacking weight matrices.
    \\  - Command buffer reuse across tokens.
    \\  - Avoid commitAndWait — use MTLEvent signalling
    \\    or completion handlers so CPU can prep the next
    \\    token while GPU finishes the current one.
    \\
    \\Phase 5 — Pipeline architecture:
    \\  - Indirect command buffers for zero-overhead
    \\    dispatch (encode once, replay every token).
    \\  - Double-buffer the command buffer pipeline.
    \\  - Overlap CPU sampling with GPU compute.
    \\
    \\## Metal kernel editing rules
    \\
    \\- [[buffer(N)]] must match setBuffer calls.
    \\- Bounds check: if (gid >= count) return;
    \\- threadgroup_barrier after threadgroup writes.
    \\- Avoid divergent branches in a SIMD group.
    \\- Prefer adding new kernels alongside existing ones
    \\  for safe A/B comparison.
    \\- Test new kernels against the CPU reference in
    \\  transformer.zig's test suite.
    \\
    \\## Recovery
    \\
    \\After every rollback, call add_summary with a
    \\concise description of what was tried, the result,
    \\and why it failed. This prevents future experiments
    \\from repeating the same mistake.
    \\
    \\- check fails: fix or rollback_latest + add_summary.
    \\- test fails: rollback_latest + add_summary.
    \\- bench crashes: rollback_latest + add_summary.
    \\- bench regresses: rollback_latest + add_summary.
    \\
    \\## FOCUS
    \\
    \\This conversation is one experiment. Make it count.
    \\Read only what you need, make ONE targeted change,
    \\validate it, and report the result. Do not try to
    \\run multiple experiments — the outer loop handles
    \\that. The first user message includes a compact
    \\summary of past benchmarks. Use the history tool
    \\for full experiment details. You can pick up
    \\where previous experiments left off.
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
    \\  {"name":"bench_infer",
    \\   "description":"Inference benchmark (~5s). Returns JSON with gpu_batched images/sec, gpu_single_sample p50/p99 latency, and cpu_single_sample p50/p99 latency.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"bench_compare",
    \\   "description":"Compare all benchmark results side by side.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"history",
    \\   "description":"Return the last N full experiment benchmark records as a JSON array. Use this for detailed per-epoch data, config, etc. The initial summary only shows key metrics.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"count":{"type":"integer",
    \\       "description":"Number of recent records to return (default 5, max 20)"}},
    \\     "required":[]}},
    \\  {"name":"show",
    \\   "description":"View an engine source file as structured JSON with line numbers.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"file":{"type":"string",
    \\       "description":"Source file path, e.g. nn/src/metal.zig"}},
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
    \\       "description":"File path relative to project root, e.g. nn/src/metal.zig"}},
    \\     "required":["path"]}},
    \\  {"name":"write_file",
    \\   "description":"Replace entire contents of an engine source file. Use edit_file for small changes instead.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{
    \\       "path":{"type":"string",
    \\         "description":"Engine file path, e.g. nn/src/metal.zig"},
    \\       "content":{"type":"string",
    \\         "description":"Complete new file contents"}},
    \\     "required":["path","content"]}},
    \\  {"name":"edit_file",
    \\   "description":"Targeted find-and-replace in an engine source file. More efficient than write_file for small edits. The old_content must match exactly.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{
    \\       "path":{"type":"string",
    \\         "description":"Engine file path, e.g. nn/src/metal.zig"},
    \\       "old_content":{"type":"string",
    \\         "description":"Exact text to find (must match exactly)"},
    \\       "new_content":{"type":"string",
    \\         "description":"Replacement text"}},
    \\     "required":["path","old_content","new_content"]}},
    \\  {"name":"list_directory",
    \\   "description":"List files and subdirectories in a project directory.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"path":{"type":"string",
    \\       "description":"Directory path relative to project root, e.g. nn/src/ or nn/src/shaders"}},
    \\     "required":["path"]}},
    \\  {"name":"cwd",
    \\   "description":"Return the absolute path of the working directory used by run_command and list_directory. Useful for orienting yourself in the filesystem.",
    \\   "input_schema":{"type":"object","properties":{},"required":[]}},
    \\  {"name":"run_command",
    \\   "description":"Execute a shell command in the project root directory via /bin/sh. 120s timeout. Use for grep, wc, head, tail, find, cat, etc. Do NOT run long-lived processes (servers, watchers). IMPORTANT: A non-zero exit code does NOT mean the tool failed — e.g. grep returns exit code 1 when no matches are found, which is normal. The exit code will be appended to the output when non-zero.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"command":{"type":"string",
    \\       "description":"Shell command to execute. Use single quotes for patterns, e.g. grep -rn 'matmul' src/. For regex alternation in grep use -E flag with pipe: grep -rEn 'foo|bar' src/. Avoid backslash-escaped double quotes inside the command; prefer single quotes."}},
    \\     "required":["command"]}},
    \\  {"name":"commit",
    \\   "description":"Git commit all current changes. Call after a successful KEEP decision to preserve the optimization. The message should summarize what was optimized and the throughput improvement.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"message":{"type":"string",
    \\       "description":"Commit message summarizing the optimization"}},
    \\     "required":["message"]}},
    \\  {"name":"add_summary",
    \\   "description":"Record a concise summary of what was tried and why it succeeded or failed. Call after every rollback so future experiments avoid repeating failed approaches. Also call after a KEEP to record what worked. Summaries are injected into every future experiment's initial context.",
    \\   "input_schema":{"type":"object",
    \\     "properties":{"summary":{"type":"string",
    \\       "description":"Concise summary: what was tried, the throughput result, and why it succeeded or failed."}},
    \\     "required":["summary"]}}
    \\]
;

// ============================================================
// Executor system prompt (local LLM — implements plans)
//
// Lean prompt for the Qwen executor in two-tier mode.
// Contains only: role, tool list, protocol, decision
// rules, and constraints.  No architecture deep-dive,
// no CLAUDE.md rules — Opus already encoded relevant
// details into the experiment plan.  The executor
// pulls additional context via tools as needed.
// ============================================================

const EXECUTOR_SYSTEM_PROMPT =
    \\You are an executor agent for nnzap, a Zig + Metal
    \\GPU inference engine. A senior strategist designed
    \\an experiment plan for you. Implement it precisely
    \\using the tools below.
    \\
    \\## Tools
    \\
    \\Safety: snapshot, snapshot_list, rollback,
    \\  rollback_latest, diff
    \\Validation: check (compile ~2s), test (correctness
    \\  ~5s), bench / bench_infer (perf ~15s),
    \\  bench_compare
    \\Inspection: show, show_function, read_file,
    \\  list_directory, history, cwd
    \\Editing: edit_file (find-and-replace, preferred),
    \\  write_file (full rewrite)
    \\Shell: run_command (120s timeout)
    \\Record: commit, add_summary
    \\
    \\## Protocol
    \\
    \\1. snapshot — before any edits.
    \\2. Read target code with show / show_function /
    \\   read_file / run_command (e.g. grep).
    \\3. Edit with edit_file or write_file.
    \\4. check — must pass before test or bench.
    \\5. test — all 71 tests must pass.
    \\6. bench — primary metric: decode_tok_per_sec.
    \\7. Evaluate (see decision rules).
    \\8. KEEP → commit. ROLLBACK → rollback_latest +
    \\   add_summary explaining what failed and why.
    \\
    \\## Decision rules
    \\
    \\- decode_tok_per_sec improves >= 5%: KEEP.
    \\- decode_p50_us improves >= 5%: KEEP.
    \\- Enables future wins: KEEP.
    \\- Any primary metric regresses > 5%: ROLLBACK.
    \\- Test or compile failure: ROLLBACK immediately.
    \\
    \\## Constraints
    \\
    \\- ONE optimisation per experiment.
    \\- Read a function fully before modifying it.
    \\- Metal [[buffer(N)]] must match setBuffer calls.
    \\- Never modify test expectations or remove tests.
    \\- Use edit_file for small changes (faster, less
    \\  context than write_file).
    \\
    \\When done (KEEP or exhausted ideas), STOP.
;

// ============================================================
// Strategist prompt (Opus — designs experiments)
//
// Used in two-tier mode.  Opus reviews summaries and
// history, then outputs ONE concrete experiment plan
// for the local Ollama executor to implement.
// ============================================================

const STRATEGIST_PROMPT =
    \\You are a systems-performance research strategist
    \\for nnzap, a Zig + Metal GPU-accelerated 1-bit
    \\language model inference engine for Apple Silicon
    \\with zero-copy unified memory.
    \\
    \\## Role
    \\
    \\Review past experiment history and summaries, then
    \\design ONE specific optimisation experiment. An
    \\executor agent (a capable local LLM with full tool
    \\access) will implement your plan — you do NOT
    \\execute anything yourself.
    \\
    \\## Architecture
    \\
    \\Bonsai 1.7B decode hot path per token (28 blocks,
    \\2048 hidden, 16 Q heads / 8 KV heads, head_dim
    \\128, group_size 128):
    \\
    \\  1. embedding_lookup (1-bit packed, dequant)
    \\  2. Per block (x28):
    \\     rms_norm -> qmv x3 (Q,K,V) -> rms_norm ->
    \\     rope -> kv_cache_update -> gqa_attention ->
    \\     qmv (O) -> residual_add -> rms_norm ->
    \\     qmv x2 (gate, up) -> silu_mul -> qmv (down)
    \\     -> residual_add
    \\  3. rms_norm (final) -> qmv (lm_head)
    \\
    \\Total: 197 qmv calls + ~168 lightweight dispatches
    \\= ~365 kernel dispatches per token.
    \\
    \\Key files:
    \\  nn/src/transformer.zig — dispatch, decode loop
    \\  nn/src/model.zig — buffer allocation, weights
    \\  nn/src/metal.zig — Metal device, buffers, dispatch
    \\  nn/src/shaders/compute.metal — qmv, qmm kernels
    \\  nn/src/shaders/transformer.metal — rms, rope, etc
    \\  nn/examples/bonsai_bench.zig — benchmark binary
    \\
    \\## Output format
    \\
    \\Respond with ONLY a detailed experiment plan:
    \\
    \\1. TITLE: One-line optimisation description.
    \\2. HYPOTHESIS: Why this should improve decode tok/s.
    \\3. TARGET FILES: Which files to modify.
    \\4. CHANGES: Exact functions to modify, what to change,
    \\   and concrete implementation guidance. Name
    \\   functions, parameters, buffer indices, thread
    \\   counts. The executor needs enough detail to
    \\   implement without creative leaps.
    \\5. VALIDATION: Expected compile/test/bench results.
    \\6. ROLLBACK IF: When to abandon the approach.
    \\
    \\## Rules
    \\
    \\- Do NOT use any tools. Output ONLY the plan text.
    \\- Do NOT repeat approaches from the summaries —
    \\  check them carefully before proposing.
    \\- Focus on the highest-impact change first.
    \\- ONE change per experiment. Isolate variables.
    \\- The executor has: snapshot, rollback, check, test,
    \\  bench, bench_infer, show, show_function, read_file,
    \\  edit_file, write_file, run_command, commit,
    \\  add_summary. It can read and modify any engine
    \\  source file.
    \\- All 71 tests must keep passing. Never suggest
    \\  removing or weakening tests.
    \\- Engineering rules apply: 70-line functions, >= 2
    \\  assertions per function, 100-column limit,
    \\  snake_case naming, comments explain WHY.
;

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
    ensureHistoryDir(arena);

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
        .{ MAX_EXPERIMENTS, MAX_TURNS_PER_EXPERIMENT },
    );

    if (!buildToolbox(arena)) {
        api.fatal("zig build failed.\n");
        unreachable;
    }

    const rules = loadEngineeringRules(arena);

    // Two-tier mode: Opus strategises, local LLM executes.
    // Requires both ANTHROPIC_API_KEY (for Opus) and
    // LOCAL_LLM_MODEL env var (for the local executor,
    // e.g. LM Studio serving an MLX model).
    const local_model = loadLocalModel();
    if (local_model) |executor_model| {
        api.log(
            "Two-tier mode: {s} (strategist)" ++
                " + {s} (executor via LM Studio)\n",
            .{ model, executor_model },
        );
        return runTwoTierLoop(
            arena,
            run_start,
            api_key,
            model,
            executor_model,
            rules,
        );
    }

    // Global counters for the run summary.
    var total_experiments: u32 = 0;
    var total_turns: u32 = 0;
    var total_bench_count: u32 = 0;
    var total_tool_calls: u32 = 0;
    var total_api_errors: u32 = 0;
    var total_api_ms: i64 = 0;
    var total_tool_ms: i64 = 0;
    var total_input_tokens: u64 = 0;
    var total_output_tokens: u64 = 0;

    // ── Outer loop: one experiment per iteration ─────
    var experiment: u32 = 0;
    while (experiment < MAX_EXPERIMENTS) : (experiment += 1) {
        api.log(
            "\n" ++
                "============================================" ++
                "==\n" ++
                "  Experiment {d}/{d}\n" ++
                "============================================" ++
                "==\n",
            .{ experiment + 1, MAX_EXPERIMENTS },
        );

        // Fresh history and context for each experiment.
        const history = buildHistorySummary(arena);
        const summaries = buildSummariesSection(arena);
        const orientation = buildOrientation(arena);
        const first_msg = buildInitialMessage(
            arena,
            history,
            summaries,
            rules,
            orientation,
        );

        var messages: [MAX_MESSAGES][]const u8 = undefined;
        var count: u32 = 0;
        messages[0] = first_msg;
        count = 1;

        var bench_ran = false;
        var experiment_complete = false;
        var api_failed = false;

        // ── Inner loop: turns within one experiment ──
        var turn: u32 = 0;
        while (turn < MAX_TURNS_PER_EXPERIMENT) : (turn += 1) {
            api.log(
                "\n--- Experiment {d}, Turn {d} ---\n",
                .{ experiment + 1, turn + 1 },
            );

            const ctx_bytes = contextSizeBytes(
                messages[0..count],
            );
            api.log(
                "  Context: {d} messages, {d} KB\n",
                .{ count, ctx_bytes / 1024 },
            );

            const api_start = timestampMs();
            const resp = callApiWithRetry(
                arena,
                api_key,
                model,
                messages[0..count],
                SYSTEM_PROMPT,
                TOOL_SCHEMAS,
            );
            const api_elapsed = timestampMs() -
                api_start;
            total_api_ms += api_elapsed;

            if (!resp.success) {
                total_api_errors += 1;
                api.log(
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
                api_failed = true;
                break;
            }

            api.log(
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

            // Extract and accumulate token usage from
            // the raw response saved by callApi.
            {
                const usage_path = "../" ++
                    HISTORY_DIR ++ "/_response.json";
                const usage_raw =
                    std.fs.cwd().readFileAlloc(
                        arena,
                        usage_path,
                        MAX_API_RESPONSE,
                    ) catch "";
                if (usage_raw.len > 0) {
                    const in_str =
                        api.extractJsonNumber(
                            usage_raw,
                            "\"input_tokens\":",
                        );
                    const out_str =
                        api.extractJsonNumber(
                            usage_raw,
                            "\"output_tokens\":",
                        );
                    const in_tok: u64 =
                        if (in_str) |s|
                            std.fmt.parseInt(
                                u64,
                                s,
                                10,
                            ) catch 0
                        else
                            0;
                    const out_tok: u64 =
                        if (out_str) |s|
                            std.fmt.parseInt(
                                u64,
                                s,
                                10,
                            ) catch 0
                        else
                            0;
                    total_input_tokens += in_tok;
                    total_output_tokens += out_tok;
                    api.log(
                        "  Tokens: {d} in," ++
                            " {d} out\n",
                        .{ in_tok, out_tok },
                    );
                }
            }

            if (resp.text.len > 0) {
                api.logClaudeText(resp.text);
            }

            messages[count] = api.buildAssistantMsg(
                arena,
                resp.content_json,
            );
            count += 1;

            // Check if Claude wants to stop (end_turn).
            const is_tool_use = api.eql(
                resp.stop_reason,
                "tool_use",
            );
            if (!is_tool_use or
                resp.tool_calls.len == 0)
            {
                api.log(
                    "  Claude signalled end of " ++
                        "experiment.\n",
                    .{},
                );
                experiment_complete = true;
                break;
            }

            // Execute tool calls.
            const tools_start = timestampMs();
            const results = executeTools(
                arena,
                resp.tool_calls,
            );
            const tools_elapsed = timestampMs() -
                tools_start;
            total_tool_ms += tools_elapsed;
            total_tool_calls += @intCast(
                resp.tool_calls.len,
            );

            // Track bench calls.
            for (resp.tool_calls) |call| {
                if (api.eql(call.name, "bench") or
                    api.eql(call.name, "bench_infer"))
                {
                    total_bench_count += 1;
                    bench_ran = true;
                }
            }

            // Log tool results summary.
            var err_count: u32 = 0;
            var result_bytes: usize = 0;
            for (results) |r| {
                result_bytes += r.content.len;
                if (r.is_error) err_count += 1;
            }
            api.log(
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

            messages[count] = api.buildToolResultsMsg(
                arena,
                results,
            );

            // Inject a turn-limit warning so the agent
            // knows to wrap up with a summary before the
            // hard cutoff.  Appended as a text block
            // inside the tool-results user message.
            if (turn + 1 >= TURN_WARNING_THRESHOLD) {
                const remaining =
                    MAX_TURNS_PER_EXPERIMENT - turn - 1;
                const warn = std.fmt.allocPrint(
                    arena,
                    ",{{\"type\":\"text\",\"text\":" ++
                        "\"WARNING: {d} turns remaining " ++
                        "out of {d}. Wrap up NOW. " ++
                        "If your current approach is " ++
                        "not working, call " ++
                        "rollback_latest then " ++
                        "add_summary describing what " ++
                        "you tried and why it failed, " ++
                        "then STOP. Do not start new " ++
                        "optimisations.\"}}]}}",
                    .{
                        remaining,
                        MAX_TURNS_PER_EXPERIMENT,
                    },
                ) catch null;

                if (warn) |w| {
                    const prev = messages[count];
                    // Replace closing "]}" with the
                    // warning text block + "]}".
                    if (prev.len >= 2) {
                        const trimmed =
                            prev[0 .. prev.len - 2];
                        messages[count] =
                            std.fmt.allocPrint(
                                arena,
                                "{s}{s}",
                                .{ trimmed, w },
                            ) catch prev;
                    }
                }
            }

            count += 1;

            if (count + 2 >= MAX_MESSAGES) {
                api.log("  Message limit reached.\n", .{});
                break;
            }
        }

        total_turns += turn + 1;

        // Save this experiment's conversation log.
        api.saveRunLog(arena, messages[0..count], "../" ++ HISTORY_DIR);

        if (experiment_complete or bench_ran) {
            total_experiments += 1;
        }

        // If the API failed, stop the outer loop too.
        if (api_failed) {
            api.log(
                "  Stopping: API failure.\n",
                .{},
            );
            break;
        }

        api.log(
            "  Experiment {d} done " ++
                "({d} turns).\n",
            .{ experiment + 1, turn + 1 },
        );
    }

    // ── Run summary ─────────────────────────────────
    const run_elapsed = timestampMs() - run_start;

    // Opus 4.6 pricing: $5/MTok input, $25/MTok output.
    const cost_cents =
        (total_input_tokens * 5 +
            total_output_tokens * 25) / 10_000;
    const cost_str = std.fmt.allocPrint(
        arena,
        "${d}.{d:0>2}",
        .{ cost_cents / 100, cost_cents % 100 },
    ) catch "$?.??";

    api.log(
        "\n" ++
            "============================================" ++
            "==\n" ++
            "  Run summary\n" ++
            "============================================" ++
            "==\n" ++
            "  Experiments: {d}\n" ++
            "  Total turns: {d}\n" ++
            "  Tool calls:  {d}\n" ++
            "  Benchmarks:  {d}\n" ++
            "  API errors:  {d}\n" ++
            "  In tokens:   {d}\n" ++
            "  Out tokens:  {d}\n" ++
            "  Est. cost:   {s}\n" ++
            "  API time:    {s}\n" ++
            "  Tool time:   {s}\n" ++
            "  Total time:  {s}\n" ++
            "============================================" ++
            "==\n",
        .{
            total_experiments,
            total_turns,
            total_tool_calls,
            total_bench_count,
            total_api_errors,
            total_input_tokens,
            total_output_tokens,
            cost_str,
            nanosToMsStr(
                arena,
                total_api_ms * 1_000_000,
            ),
            nanosToMsStr(
                arena,
                total_tool_ms * 1_000_000,
            ),
            nanosToMsStr(
                arena,
                run_elapsed * 1_000_000,
            ),
        },
    );
}

// ============================================================
// Two-tier mode: Opus strategist + Ollama executor
//
// Opus reviews history and designs experiments (one API
// call per experiment, high-quality reasoning).  A local
// Ollama model executes the plan by driving the tool
// loop (free, fast, many tool calls).
// ============================================================

const ExecutionResult = struct {
    turns: u32,
    tool_calls: u32,
    api_ms: i64,
    tool_ms: i64,
    input_tokens: u64,
    output_tokens: u64,
    completed: bool,
};

/// Outer loop for two-tier mode.  Converts tool schemas
/// once, then alternates between Opus strategy calls and
/// Ollama execution loops.
fn runTwoTierLoop(
    arena: Allocator,
    run_start: i64,
    api_key: []const u8,
    opus_model: []const u8,
    ollama_model: []const u8,
    rules: []const u8,
) !void {
    std.debug.assert(opus_model.len > 0);
    std.debug.assert(ollama_model.len > 0);

    // Convert Anthropic tool schemas to OpenAI format
    // once at startup (Rule 8 — amortise upfront work).
    const openai_tools = ollama.convertAnthropicTools(
        arena,
        TOOL_SCHEMAS,
    ) catch {
        api.fatal(
            "Failed to convert tool schemas.\n",
        );
        unreachable;
    };

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

/// Run one strategy→execution cycle.  Returns true if the
/// outer loop should stop (e.g. strategist failure).
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
    const history = buildHistorySummary(arena);
    const summaries = buildSummariesSection(arena);

    api.log("  Phase 1: Opus strategy...\n", .{});
    const strat_start = timestampMs();
    const plan_resp = callOpusStrategist(
        arena,
        api_key,
        opus_model,
        history,
        summaries,
        rules,
        code_context,
    );
    const strat_ms = timestampMs() - strat_start;
    total_api_ms.* += strat_ms;
    opus_in.* += plan_resp.input_tokens;
    opus_out.* += plan_resp.output_tokens;

    if (!plan_resp.success or plan_resp.text.len == 0) {
        api.log(
            "  Strategist failed ({s}): {s}\n",
            .{
                nanosToMsStr(
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
        .{nanosToMsStr(arena, strat_ms * 1_000_000)},
    );
    api.logClaudeText(plan_resp.text);

    // Phase 2: Local LLM executes the plan.
    api.log("\n  Phase 2: Local LLM executor...\n", .{});
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
        .{ experiment + 1, exec.turns, exec.tool_calls },
    );
    return false; // Continue outer loop.
}

/// Call Opus to design a single experiment plan.
/// Returns the full ApiResponse with the plan in .text.
fn callOpusStrategist(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    history: []const u8,
    summaries: []const u8,
    rules: []const u8,
    code_context: []const u8,
) ApiResponse {
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

    // Strategist: custom prompt, no tools (output only).
    return callApiWithRetry(
        arena,
        api_key,
        model,
        &messages,
        STRATEGIST_PROMPT,
        "[]",
    );
}

/// Build the user message context for the strategist.
/// Includes benchmark history, summaries of past
/// experiments, and engineering rules.
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
            "thread counts, and kernel parameters.\n\n" ++
            "{s}\n\n" ++
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

/// Read all hot-path source files into a single code
/// context string with file headers.  Each file is
/// wrapped with a clear delimiter so the strategist
/// can reference specific locations.
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
            "### {s} ({d} lines)\n" ++
                "```\n",
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

    std.debug.assert(buf.items.len <= MAX_CODE_CONTEXT +
        MAX_FILE_SIZE);
    return buf.items;
}

/// Execute an experiment plan using the local LLM
/// (e.g. LM Studio).  Drives the tool-calling loop:
/// snapshot, edit, check, test, bench, report.
fn runLocalExecution(
    arena: Allocator,
    model: []const u8,
    plan: []const u8,
    openai_tools: []const u8,
) ExecutionResult {
    std.debug.assert(model.len > 0);
    std.debug.assert(plan.len > 0);

    const executor_prompt = buildExecutorPrompt();
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
        api.log(
            "\n  --- Executor turn {d} ---\n",
            .{turn + 1},
        );
        const t0 = timestampMs();
        const resp = ollama.callApi(
            arena,
            model,
            executor_prompt,
            msgs[0..count],
            openai_tools,
            LOCAL_LLM_MAX_TOKENS_STR,
            history_dir,
        );
        r.api_ms += timestampMs() - t0;

        if (!resp.success) {
            api.log(
                "  Local LLM error: {s}\n",
                .{resp.error_message},
            );
            break;
        }

        r.input_tokens += resp.input_tokens;
        r.output_tokens += resp.output_tokens;
        if (resp.text.len > 0) {
            api.logClaudeText(resp.text);
        }

        // Append assistant message (OpenAI format).
        msgs[count] = ollama.buildAssistantMsg(
            arena,
            resp,
        );
        count += 1;

        // Model chose to stop — no tool calls.
        if (!api.eql(resp.stop_reason, "tool_use") or
            resp.tool_calls.len == 0)
        {
            r.completed = true;
            break;
        }

        // Execute tools and append results.
        const t1 = timestampMs();
        const results = executeTools(
            arena,
            resp.tool_calls,
        );
        r.tool_ms += timestampMs() - t1;
        r.tool_calls += @intCast(
            resp.tool_calls.len,
        );

        // OpenAI format: one message per tool result.
        for (results) |tr| {
            if (count >= MAX_MESSAGES - 1) break;
            msgs[count] = ollama.buildToolResultMsg(
                arena,
                tr,
            );
            count += 1;
        }
        if (count >= MAX_MESSAGES - 2) break;
    }

    r.turns = turn + 1;
    return r;
}

/// Build the system prompt for the local LLM executor.
/// Returns the lean EXECUTOR_SYSTEM_PROMPT — role,
/// tools, protocol, and constraints only.  The Opus
/// strategist already embedded architecture details
/// and engineering rules into the experiment plan.
fn buildExecutorPrompt() []const u8 {
    comptime std.debug.assert(
        EXECUTOR_SYSTEM_PROMPT.len > 0,
    );
    return EXECUTOR_SYSTEM_PROMPT;
}

/// Build the initial user message for the executor,
/// containing filesystem orientation and the plan
/// from the Opus strategist.
fn buildExecutorInitialMsg(
    arena: Allocator,
    plan: []const u8,
) []const u8 {
    std.debug.assert(plan.len > 0);

    const orientation = buildOrientation(arena);
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
    const run_elapsed = timestampMs() - run_start;

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
            "  Run summary (two-tier)\n" ++
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
            nanosToMsStr(
                arena,
                total_api_ms * 1_000_000,
            ),
            nanosToMsStr(
                arena,
                total_tool_ms * 1_000_000,
            ),
            nanosToMsStr(
                arena,
                run_elapsed * 1_000_000,
            ),
        },
    );
}

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

/// Load the local LLM executor model from LOCAL_LLM_MODEL.
/// Returns null when not set (single-tier mode).
/// Falls back to DEFAULT_LOCAL_MODEL if set to "default".
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

/// Load CLAUDE.md engineering rules from disk.
/// Returns the file contents or a fallback message.
fn loadEngineeringRules(arena: Allocator) []const u8 {
    const fs_path = resolveToFs(arena, CLAUDE_MD_PATH) orelse {
        api.log(
            "WARNING: cannot resolve {s}\n",
            .{CLAUDE_MD_PATH},
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
            .{ CLAUDE_MD_PATH, @errorName(err) },
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
            .{ CLAUDE_MD_PATH, @errorName(err) },
        );
        return "(CLAUDE.md too large or unreadable)";
    };

    if (content.len == 0) {
        api.log("WARNING: {s} is empty.\n", .{CLAUDE_MD_PATH});
        return "(CLAUDE.md is empty)";
    }

    api.log(
        "Engineering rules: {d} KB ({s})\n",
        .{ content.len / 1024, CLAUDE_MD_PATH },
    );
    return content;
}

/// Build the toolbox with `zig build`.
fn buildToolbox(arena: Allocator) bool {
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

// ============================================================
// API calling with retry
// ============================================================

/// Call the API with exponential backoff on transient errors.
fn callApiWithRetry(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    messages: []const []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
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
            api.log(
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
            system_prompt,
            tool_schemas,
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

    return api.errResp(
        "max retries exhausted",
        false,
    );
}

/// Call the Anthropic Messages API via curl.
///
/// Using curl instead of std.http because:
///   1. Proper --connect-timeout and --max-time
///      prevent indefinite hangs.
///   2. curl handles TLS in a separate process —
///      a connection reset cannot kill the agent
///      with a signal.
///   3. Battle-tested on macOS with no silent
///      failure modes.
fn callApi(
    arena: Allocator,
    api_key: []const u8,
    model: []const u8,
    messages: []const []const u8,
    system_prompt: []const u8,
    tool_schemas: []const u8,
) ApiResponse {
    std.debug.assert(api_key.len > 0);
    std.debug.assert(messages.len > 0);

    const body = api.buildRequestJson(
        arena,
        model,
        messages,
        system_prompt,
        tool_schemas,
        MAX_TOKENS_STR,
    ) catch return api.errResp(
        "request build failed",
        false,
    );

    // Save request for debugging and as curl input.
    const request_path = "../" ++ HISTORY_DIR ++
        "/_request.json";
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

    // Build curl arguments. The -w flag appends the
    // HTTP status code on a final line so we can
    // split body from status deterministically.
    const data_arg = std.fmt.allocPrint(
        arena,
        "@{s}",
        .{request_path},
    ) catch return api.errResp(
        "failed to format curl arg",
        false,
    );
    const key_header = std.fmt.allocPrint(
        arena,
        "x-api-key: {s}",
        .{api_key},
    ) catch return api.errResp(
        "failed to format key header",
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
            "curl failed: {s}",
            .{detail},
        ) catch "curl failed";
        return api.errResp(msg, true);
    }

    const output = result.stdout;
    if (output.len < 4) {
        return api.errResp("empty curl output", true);
    }

    // Split response body from the HTTP status code.
    // curl -w '\n%{http_code}' appends it after a
    // newline at the very end.
    const last_nl = std.mem.lastIndexOfScalar(
        u8,
        output,
        '\n',
    ) orelse {
        return api.errResp(
            "malformed curl output (no status line)",
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
            "failed to parse HTTP status from curl",
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
        return api.errResp("empty API response", true);
    }

    // Persist raw response so the main loop can
    // extract token usage without changing the
    // shared ApiResponse type.
    api.writeFile(
        "../" ++ HISTORY_DIR ++ "/_response.json",
        response_data,
    ) catch {};

    return api.parseApiResponse(arena, response_data);
}

/// Produce a descriptive error for non-200 API responses.
fn handleNonOkStatus(
    arena: Allocator,
    code: u16,
    body: []const u8,
) ApiResponse {
    // Extract the API error message if present.
    const api_msg = api.extractJsonString(
        body,
        "\"message\":\"",
    ) orelse api.extractJsonString(
        body,
        "\"message\": \"",
    ) orelse "";

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
    // 500/502/503 = server errors — all retryable.
    const retryable = (code == 429 or
        code == 529 or
        code >= 500);

    return api.errResp(msg, retryable);
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

    const tool_start = timestampMs();
    const output = dispatchTool(arena, call);
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

    // Persist bench/bench_infer results to the history log.
    if ((api.eql(call.name, "bench") or
        api.eql(call.name, "bench_infer")) and
        output.success)
    {
        appendExperiment(arena, output.stdout);
        api.log(
            "    -> {s} result persisted\n",
            .{call.name},
        );
    }

    return .{
        .tool_use_id = call.id,
        .content = api.truncate(
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
    if (api.eql(call.name, "snapshot")) {
        return callEngineResearch(
            arena,
            &.{"snapshot"},
        );
    }
    if (api.eql(call.name, "snapshot_list")) {
        return callEngineResearch(
            arena,
            &.{"snapshot-list"},
        );
    }
    if (api.eql(call.name, "rollback")) {
        return executeRollback(arena, call.input_json);
    }
    if (api.eql(call.name, "rollback_latest")) {
        return callEngineResearch(
            arena,
            &.{"rollback-latest"},
        );
    }
    if (api.eql(call.name, "diff")) {
        return executeDiff(arena, call.input_json);
    }
    if (api.eql(call.name, "check")) {
        return callEngineResearch(
            arena,
            &.{"check"},
        );
    }
    if (api.eql(call.name, "test")) {
        return callEngineResearch(
            arena,
            &.{"test"},
        );
    }
    if (api.eql(call.name, "bench")) {
        return callEngineResearch(
            arena,
            &.{"bench"},
        );
    }
    if (api.eql(call.name, "bench_infer")) {
        return callEngineResearch(
            arena,
            &.{"bench-infer"},
        );
    }
    if (api.eql(call.name, "bench_compare")) {
        return callEngineResearch(
            arena,
            &.{"bench-compare"},
        );
    }
    if (api.eql(call.name, "history")) {
        return executeHistory(
            arena,
            call.input_json,
        );
    }
    if (api.eql(call.name, "show")) {
        return executeShow(arena, call.input_json);
    }
    if (api.eql(call.name, "show_function")) {
        return executeShowFunction(
            arena,
            call.input_json,
        );
    }

    // File I/O tools.
    if (api.eql(call.name, "read_file")) {
        return executeReadFile(
            arena,
            call.input_json,
        );
    }
    if (api.eql(call.name, "write_file")) {
        return executeWriteFile(
            arena,
            call.input_json,
        );
    }
    if (api.eql(call.name, "edit_file")) {
        return executeEditFile(
            arena,
            call.input_json,
        );
    }
    if (api.eql(call.name, "list_directory")) {
        return executeListDirectory(
            arena,
            call.input_json,
        );
    }

    // Environment tool.
    if (api.eql(call.name, "cwd")) {
        return executeCwd(arena);
    }

    // Shell tool.
    if (api.eql(call.name, "run_command")) {
        return executeRunCommand(
            arena,
            call.input_json,
        );
    }

    // Git commit tool.
    if (api.eql(call.name, "commit")) {
        return executeCommit(arena, call.input_json);
    }

    // Summary recording tool.
    if (api.eql(call.name, "add_summary")) {
        return executeAddSummary(arena, call.input_json);
    }

    return .{
        .stdout = "Error: unknown tool name",
        .success = false,
    };
}

// ============================================================
// Engine research tool dispatchers
// ============================================================

/// Return the last N full experiment records from
/// the history JSONL. The agent calls this for
/// detailed data beyond the compact startup summary.
fn executeHistory(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    const count_str = api.extractJsonNumber(
        input,
        "\"count\":",
    ) orelse "5";
    const count = @min(
        std.fmt.parseInt(u32, count_str, 10) catch 5,
        20,
    );
    if (count == 0) {
        return .{ .stdout = "[]", .success = true };
    }

    const fs_path = resolveToFs(arena, HISTORY_PATH) orelse
        return .{
            .stdout = "Error: cannot resolve history path.",
            .success = false,
        };

    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch return .{
        .stdout = "No history file found.",
        .success = true,
    };
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch return .{
        .stdout = "Error reading history.",
        .success = false,
    };
    if (content.len == 0) return .{
        .stdout = "No experiments yet.",
        .success = true,
    };

    // Split into lines and take the last N.
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
    if (line_count == 0) return .{
        .stdout = "No experiments yet.",
        .success = true,
    };

    std.debug.assert(count > 0);
    const start = if (line_count > count)
        line_count - count
    else
        0;

    var buf: std.ArrayList(u8) = .empty;
    buf.append(arena, '[') catch {};
    for (lines[start..line_count], 0..) |line, i| {
        if (i > 0) {
            buf.appendSlice(arena, ",\n") catch {};
        }
        buf.appendSlice(arena, line) catch {};
    }
    buf.append(arena, ']') catch {};

    std.debug.assert(buf.items.len > 0);
    return .{ .stdout = buf.items, .success = true };
}

/// Rollback to a specific snapshot by ID.
fn executeRollback(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const raw_id = extractRequiredField(
        input,
        "id",
    ) orelse {
        return .{
            .stdout = "Error: missing 'id' field",
            .success = false,
        };
    };

    const id = unescapeJsonString(
        arena,
        raw_id,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape id",
            .success = false,
        };
    };

    if (id.len == 0) {
        return .{
            .stdout = "Error: empty id",
            .success = false,
        };
    }

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
    const raw_id = extractRequiredField(
        input,
        "id",
    ) orelse {
        return .{
            .stdout = "Error: missing 'id' field",
            .success = false,
        };
    };

    const id = unescapeJsonString(
        arena,
        raw_id,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape id",
            .success = false,
        };
    };

    if (id.len == 0) {
        return .{
            .stdout = "Error: empty id",
            .success = false,
        };
    }

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
    const raw_file = extractRequiredField(
        input,
        "file",
    ) orelse {
        return .{
            .stdout = "Error: missing 'file' field",
            .success = false,
        };
    };

    const file = unescapeJsonString(
        arena,
        raw_file,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape file",
            .success = false,
        };
    };

    if (file.len == 0) {
        return .{
            .stdout = "Error: empty file path",
            .success = false,
        };
    }

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
    const raw_file = extractRequiredField(
        input,
        "file",
    ) orelse {
        return .{
            .stdout = "Error: missing 'file' field",
            .success = false,
        };
    };
    const raw_func = extractRequiredField(
        input,
        "function_name",
    ) orelse {
        return .{
            .stdout = "Error: missing 'function_name'",
            .success = false,
        };
    };

    const file = unescapeJsonString(
        arena,
        raw_file,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape file",
            .success = false,
        };
    };
    const func = unescapeJsonString(
        arena,
        raw_func,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape function_name",
            .success = false,
        };
    };

    if (file.len == 0) {
        return .{
            .stdout = "Error: empty file path",
            .success = false,
        };
    }
    if (func.len == 0) {
        return .{
            .stdout = "Error: empty function_name",
            .success = false,
        };
    }

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
// File I/O tool dispatchers
// ============================================================

/// Read raw contents of a project file.
fn executeReadFile(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const raw_path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };

    const path = unescapeJsonString(
        arena,
        raw_path,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape path",
            .success = false,
        };
    };

    if (path.len == 0) {
        return .{
            .stdout = "Error: empty path",
            .success = false,
        };
    }

    if (!isAllowedReadPath(path)) {
        api.log(
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

    api.log(
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
    const raw_path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };

    const path = unescapeJsonString(
        arena,
        raw_path,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape path",
            .success = false,
        };
    };

    if (path.len == 0) {
        return .{
            .stdout = "Error: empty path",
            .success = false,
        };
    }

    if (!isAllowedWritePath(path)) {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: write not allowed to '{s}'. " ++
                "Allowed: nn/src/transformer.zig, " ++
                "nn/src/network.zig, nn/src/layout.zig, " ++
                "nn/src/model.zig, nn/src/metal.zig, " ++
                "nn/src/shaders/*.metal, " ++
                "nn/examples/bonsai*.zig",
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

    writeFileContent(arena, path, content) catch {
        return .{
            .stdout = "Error: failed to write file",
            .success = false,
        };
    };

    api.log(
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
    const raw_path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };

    const path = unescapeJsonString(
        arena,
        raw_path,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape path",
            .success = false,
        };
    };

    if (path.len == 0) {
        return .{
            .stdout = "Error: empty path",
            .success = false,
        };
    }

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
    const pos = api.indexOf(
        current,
        old_text,
    ) orelse {
        // Provide context to help Claude fix the match.
        const preview = api.truncate(old_text, 100);
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
        if (api.indexOf(
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

    writeFileContent(arena, path, new_file) catch {
        return .{
            .stdout = "Error: failed to write file",
            .success = false,
        };
    };

    const removed = old_text.len;
    const added = new_text.len;

    api.log(
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
    const raw_path = extractRequiredField(
        input,
        "path",
    ) orelse {
        return .{
            .stdout = "Error: missing 'path' field",
            .success = false,
        };
    };

    const path = unescapeJsonString(
        arena,
        raw_path,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape path",
            .success = false,
        };
    };

    if (path.len == 0) {
        return .{
            .stdout = "Error: empty path",
            .success = false,
        };
    }

    if (!isAllowedReadPath(path)) {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: listing not allowed for '{s}'",
            .{path},
        ) catch "Error: path not allowed";
        return .{ .stdout = msg, .success = false };
    }

    const fs_path = resolveToFs(arena, path) orelse {
        return .{
            .stdout = "Error: failed to resolve path",
            .success = false,
        };
    };

    // Invoke ls directly (no shell) to avoid injection.
    const result = std.process.Child.run(.{
        .allocator = arena,
        .argv = &.{ "/bin/ls", "-la", fs_path },
        .max_output_bytes = MAX_COMMAND_OUTPUT,
    }) catch |err| {
        const msg = std.fmt.allocPrint(
            arena,
            "Error: failed to run ls: {s}",
            .{@errorName(err)},
        ) catch "Error: ls failed";
        return .{ .stdout = msg, .success = false };
    };

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
// Shell tool dispatcher
// ============================================================

/// Execute a shell command in the project root directory.
fn executeRunCommand(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);
    const raw_command = extractRequiredField(
        input,
        "command",
    ) orelse {
        return .{
            .stdout = "Error: missing 'command' field",
            .success = false,
        };
    };

    const command = unescapeJsonString(
        arena,
        raw_command,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape command",
            .success = false,
        };
    };

    if (command.len == 0) {
        return .{
            .stdout = "Error: empty command",
            .success = false,
        };
    }

    api.log("    cmd: {s}\n", .{api.truncate(command, 200)});
    return runShellCommand(arena, command);
}

/// Actually run a shell command and capture output.
/// Return the absolute working directory so the LLM
/// can orient itself before issuing shell commands.
fn executeCwd(arena: Allocator) ToolOutput {
    const abs = tools.cwdAbsolute(arena) catch {
        return .{
            .stdout = "Error: cannot resolve cwd",
            .success = false,
        };
    };
    return .{ .stdout = abs, .success = true };
}

/// Git commit all current changes with a descriptive
/// message. Called after a successful KEEP decision.
/// Record a summary of what was tried and its outcome.
/// Appends to summaries.jsonl so future experiments
/// see what approaches succeeded or failed.
fn executeAddSummary(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);

    const raw_summary = extractRequiredField(
        input,
        "summary",
    ) orelse {
        return .{
            .stdout = "Error: missing 'summary' field",
            .success = false,
        };
    };

    const summary = unescapeJsonString(
        arena,
        raw_summary,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape summary",
            .success = false,
        };
    };

    if (summary.len == 0) {
        return .{
            .stdout = "Error: empty summary",
            .success = false,
        };
    }

    // Count existing summaries to assign the next ID.
    const next_id = countSummaryLines(arena) + 1;

    appendSummary(arena, next_id, summary);

    const msg = std.fmt.allocPrint(
        arena,
        "Summary #{d} recorded.",
        .{next_id},
    ) catch return .{
        .stdout = "Summary recorded.",
        .success = true,
    };

    return .{ .stdout = msg, .success = true };
}

/// Count existing lines in the summaries JSONL file.
fn countSummaryLines(arena: Allocator) u32 {
    const fs_path = resolveToFs(
        arena,
        SUMMARIES_PATH,
    ) orelse return 0;

    const file = std.fs.cwd().openFile(
        fs_path,
        .{},
    ) catch return 0;
    defer file.close();

    const content = file.readToEndAlloc(
        arena,
        MAX_HISTORY_SIZE,
    ) catch return 0;

    var count: u32 = 0;
    var iter = std.mem.splitScalar(
        u8,
        content,
        '\n',
    );
    while (iter.next()) |line| {
        if (line.len >= 2) count += 1;
    }
    return count;
}

fn executeCommit(
    arena: Allocator,
    input: []const u8,
) ToolOutput {
    std.debug.assert(input.len > 0);

    const raw_msg = extractRequiredField(
        input,
        "message",
    ) orelse {
        return .{
            .stdout = "Error: missing 'message' field",
            .success = false,
        };
    };

    const msg = unescapeJsonString(
        arena,
        raw_msg,
    ) catch {
        return .{
            .stdout = "Error: failed to unescape message",
            .success = false,
        };
    };

    if (msg.len == 0) {
        return .{
            .stdout = "Error: empty commit message",
            .success = false,
        };
    }

    // Write the commit message to a temp file to avoid
    // shell quoting issues with newlines and special
    // characters.  Using `git commit -F` reads the
    // message from the file instead of the command line.
    const raw_path = ".engine_agent_history/_commit_msg.txt";
    const msg_path = resolveToFs(
        arena,
        raw_path,
    ) orelse {
        return .{
            .stdout = "Error: cannot resolve temp path",
            .success = false,
        };
    };

    // Pass raw_path (not msg_path) because writeFileContent
    // calls resolveToFs internally.
    writeFileContent(arena, raw_path, msg) catch {
        return .{
            .stdout = "Error: failed to write commit msg",
            .success = false,
        };
    };

    const cmd = std.fmt.allocPrint(
        arena,
        "git add -A && git commit -F '{s}' " ++
            "&& rm -f '{s}'",
        .{ msg_path, msg_path },
    ) catch {
        return .{
            .stdout = "Error: allocation failure",
            .success = false,
        };
    };

    return runShellCommand(arena, cmd);
}

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

    // Determine exit status. Normal exits (even non-zero) are not tool
    // errors — grep returns 1 for "no matches", diff returns 1 for
    // "files differ", etc.  Only signals/abnormal termination count as
    // true tool errors.
    const exit_code: ?u32 = switch (result.term) {
        .Exited => |code| @as(?u32, code),
        else => null,
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

    // Append exit code when non-zero so the LLM knows.
    if (exit_code) |code| {
        if (code != 0) {
            api.log("    exit code: {d}\n", .{code});
            const suffix = std.fmt.allocPrint(
                arena,
                "\n(exit code {d})",
                .{code},
            ) catch "";
            buf.appendSlice(arena, suffix) catch {};
        }
    } else {
        api.log("    terminated abnormally\n", .{});
    }

    const output = if (buf.items.len > 0)
        buf.items
    else
        "(no output)";

    return .{ .stdout = output, .success = exit_code != null };
}

// ============================================================
// Path validation
// ============================================================

/// Convert a monorepo-root-relative path to a filesystem
/// path relative to the zap/ working directory by
/// prepending "../".  Paths that already start with
/// "src/", "programs/", or "build.zig" are local to zap
/// and returned unchanged.
fn resolveToFs(
    arena: Allocator,
    path: []const u8,
) ?[]const u8 {
    std.debug.assert(path.len > 0);

    // Paths local to the zap/ directory need no prefix.
    if (api.startsWith(path, "src/") or
        api.startsWith(path, "programs/") or
        api.eql(path, "build.zig") or
        api.eql(path, "build.zig.zon"))
    {
        return path;
    }

    // Everything else lives in the monorepo root or a
    // sibling directory — prepend "../" to reach it.
    return std.fmt.allocPrint(
        arena,
        "../{s}",
        .{path},
    ) catch null;
}

/// Check if a path is safe for reading.
/// Rejects path traversal and files outside the project.
fn isAllowedReadPath(path: []const u8) bool {
    std.debug.assert(path.len > 0);

    // Reject path traversal.
    if (api.indexOf(path, "..") != null) return false;

    // Reject absolute paths.
    if (path.len > 0 and path[0] == '/') return false;

    // Allow exact root-level files.
    for (&ALLOWED_READ_FILES) |file| {
        if (api.eql(path, file)) return true;
    }

    // Allow paths under allowed prefixes.
    for (&ALLOWED_READ_PREFIXES) |prefix| {
        if (api.startsWith(path, prefix)) return true;
    }

    // Also allow any write-allowed file to be read.
    for (&ALLOWED_WRITE_FILES) |file| {
        if (api.eql(path, file)) return true;
    }

    return false;
}

/// Check if a path is in the allowed write list.
fn isAllowedWritePath(path: []const u8) bool {
    std.debug.assert(path.len > 0);

    // Reject path traversal.
    if (api.indexOf(path, "..") != null) return false;

    // Reject absolute paths.
    if (path.len > 0 and path[0] == '/') return false;

    for (&ALLOWED_WRITE_FILES) |allowed| {
        if (api.eql(path, allowed)) return true;
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

    return api.extractJsonString(
        input_json,
        "\"" ++ field ++ "\":\"",
    ) orelse api.extractJsonString(
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

    const fs_path = resolveToFs(arena, path) orelse
        return null;

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

/// Write contents to a file, creating or truncating it.
fn writeFileContent(
    arena: Allocator,
    path: []const u8,
    content: []const u8,
) !void {
    std.debug.assert(path.len > 0);
    std.debug.assert(content.len <= MAX_FILE_SIZE);

    const fs_path = resolveToFs(arena, path) orelse
        return error.PathResolutionFailed;

    const file = try std.fs.cwd().createFile(
        fs_path,
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

/// Build a filesystem orientation section so the LLM
/// knows where it is without burning tool calls.
fn buildOrientation(arena: Allocator) []const u8 {
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
    if (!ok) return null;
    if (result.stdout.len == 0) return null;
    return result.stdout;
}

/// Build the initial user message with history context.
fn buildInitialMessage(
    arena: Allocator,
    history: []const u8,
    summaries: []const u8,
    rules: []const u8,
    orientation: []const u8,
) []const u8 {
    std.debug.assert(rules.len > 0);
    const has_history = history.len > 0;

    const rules_section =
        "## Engineering rules (CLAUDE.md)\n\n" ++
        "You MUST follow these rules when editing " ++
        "engine source code. They are non-negotiable " ++
        "— assertion density, function length limits, " ++
        "naming, explicit control flow, and all other " ++
        "rules apply to every line you write.\n\n";

    const history_section = if (has_history)
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

    const text = if (has_history)
        std.fmt.allocPrint(
            arena,
            "{s}\n{s}{s}\n\n{s}{s}{s}{s}",
            .{
                orientation,
                rules_section,
                rules,
                history_section,
                history,
                summaries,
                suffix,
            },
        ) catch "Begin optimising."
    else
        std.fmt.allocPrint(
            arena,
            "{s}\n{s}{s}\n\n{s}{s}{s}",
            .{
                orientation,
                rules_section,
                rules,
                history_section,
                summaries,
                suffix,
            },
        ) catch "Begin optimising.";

    std.debug.assert(text.len > 0);
    const result = api.wrapUserTextMessage(arena, text);
    std.debug.assert(result.len > 0);
    return result;
}

// ============================================================
// History persistence
//
// Append-only JSONL at .engine_agent_history/experiments.jsonl.
// Each line is a compacted benchmark JSON from a bench call.
// Loaded on startup and injected into the first user message.
// ============================================================

fn ensureHistoryDir(arena: Allocator) void {
    const fs_path = resolveToFs(arena, HISTORY_DIR) orelse {
        api.log("WARNING: cannot resolve history dir\n", .{});
        return;
    };
    std.fs.cwd().makeDir(fs_path) catch |err| {
        if (err != error.PathAlreadyExists) {
            api.log(
                "WARNING: mkdir {s}: {s}\n",
                .{ fs_path, @errorName(err) },
            );
        }
    };
}

/// Extract key metrics from one JSONL experiment line
/// into a compact human-readable row.
fn formatHistoryLine(
    arena: Allocator,
    index: u32,
    line: []const u8,
) ?[]const u8 {
    std.debug.assert(line.len > 0);
    std.debug.assert(index > 0);

    const ts = api.extractJsonString(
        line,
        "\"timestamp_utc\":\"",
    ) orelse api.extractJsonString(
        line,
        "\"timestamp_utc\": \"",
    ) orelse "?";
    const throughput = api.extractJsonNumber(
        line,
        "\"throughput_images_per_sec\":",
    ) orelse "?";
    const accuracy = api.extractJsonNumber(
        line,
        "\"final_test_accuracy_pct\":",
    ) orelse "?";
    const train_ms = api.extractJsonNumber(
        line,
        "\"total_training_ms\":",
    ) orelse "?";

    return std.fmt.allocPrint(
        arena,
        "  {d}. {s}  " ++
            "throughput={s}  " ++
            "acc={s}%  " ++
            "time={s}ms\n",
        .{
            index,
            ts,
            throughput,
            accuracy,
            train_ms,
        },
    ) catch null;
}

/// Build a compact summary table from the experiment
/// history JSONL. Only key metrics are included —
/// the agent can call the history tool for full
/// details of specific experiments.
fn buildHistorySummary(arena: Allocator) []const u8 {
    const max_visible: u32 = 10;

    const fs_path = resolveToFs(arena, HISTORY_PATH) orelse
        return "";

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

    // Split JSONL into individual lines.
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

    var buf: std.ArrayList(u8) = .empty;
    const header = std.fmt.allocPrint(
        arena,
        "{d} experiments total " ++
            "(last {d} shown):\n\n",
        .{ line_count, visible },
    ) catch return "";
    buf.appendSlice(arena, header) catch return "";

    var idx: u32 = start;
    while (idx < line_count) : (idx += 1) {
        const row = formatHistoryLine(
            arena,
            idx + 1,
            lines[idx],
        ) orelse continue;
        buf.appendSlice(arena, row) catch continue;
    }

    std.debug.assert(buf.items.len > 0);
    return buf.items;
}

/// Load experiment summaries from the summaries JSONL
/// file. Returns a human-readable block describing
/// what previous experiments tried and their outcomes,
/// so the agent avoids repeating failed approaches.
fn buildSummariesSection(
    arena: Allocator,
) []const u8 {
    const fs_path = resolveToFs(
        arena,
        SUMMARIES_PATH,
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

    // Parse each JSONL line and extract the summary.
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

        const exp_num = api.extractJsonNumber(
            line,
            "\"experiment\":",
        ) orelse "?";
        const summary = api.extractJsonString(
            line,
            "\"summary\":\"",
        ) orelse continue;

        const row = std.fmt.allocPrint(
            arena,
            "  Experiment {s}: {s}\n",
            .{ exp_num, summary },
        ) catch continue;
        buf.appendSlice(arena, row) catch continue;
    }

    // Header was added but no summaries parsed.
    if (buf.items.len < 80) return "";

    std.debug.assert(buf.items.len > 0);
    return buf.items;
}

/// Append a bench result to the history JSONL file.
fn appendExperiment(
    arena: Allocator,
    benchmark_json: []const u8,
) void {
    std.debug.assert(benchmark_json.len > 0);

    const line = api.collapseToLine(
        arena,
        benchmark_json,
    ) catch return;

    const fs_path = resolveToFs(arena, HISTORY_PATH) orelse
        return;

    const file = std.fs.cwd().createFile(
        fs_path,
        .{ .truncate = false },
    ) catch return;
    defer file.close();

    file.seekFromEnd(0) catch return;
    file.writeAll(line) catch return;
    file.writeAll("\n") catch {};
}

/// Append an experiment summary to the summaries
/// JSONL file. Each line is a JSON object with the
/// experiment number and Claude's final assessment
/// of what was tried and why it succeeded or failed.
fn appendSummary(
    arena: Allocator,
    experiment_number: u32,
    text: []const u8,
) void {
    std.debug.assert(text.len > 0);
    std.debug.assert(experiment_number > 0);

    var ts_buf: [api.TIMESTAMP_LEN]u8 = undefined;
    const ts = api.formatTimestamp(&ts_buf, '-');

    // Truncate at a sentence boundary so the result
    // tag (KEPT/FAILED/ROLLBACK) and conclusion are
    // never cut mid-sentence.
    const trimmed = truncateAtSentence(text, 500);

    var buf: std.ArrayList(u8) = .empty;
    buf.appendSlice(
        arena,
        "{\"experiment\":",
    ) catch return;

    const num_str = std.fmt.allocPrint(
        arena,
        "{d}",
        .{experiment_number},
    ) catch return;
    buf.appendSlice(arena, num_str) catch return;
    buf.appendSlice(
        arena,
        ",\"timestamp\":\"",
    ) catch return;
    buf.appendSlice(arena, ts) catch return;
    buf.appendSlice(
        arena,
        "\",\"summary\":\"",
    ) catch return;

    // Escape the summary text for JSON.
    for (trimmed) |c| {
        switch (c) {
            '"' => {
                buf.appendSlice(
                    arena,
                    "\\\"",
                ) catch return;
            },
            '\\' => {
                buf.appendSlice(
                    arena,
                    "\\\\",
                ) catch return;
            },
            '\n' => {
                buf.append(arena, ' ') catch return;
            },
            '\r' => {},
            else => {
                buf.append(arena, c) catch return;
            },
        }
    }
    buf.appendSlice(arena, "\"}\n") catch return;

    std.debug.assert(buf.items.len > 0);

    const fs_path = resolveToFs(
        arena,
        SUMMARIES_PATH,
    ) orelse return;

    const file = std.fs.cwd().createFile(
        fs_path,
        .{ .truncate = false },
    ) catch return;
    defer file.close();

    file.seekFromEnd(0) catch return;
    file.writeAll(buf.items) catch return;
}

/// Truncate text at the last sentence boundary within
/// max_len bytes.  A sentence boundary is a period,
/// exclamation mark, question mark, or closing paren
/// followed by whitespace (or sitting at the cut
/// point).  Falls back to a hard cut when no boundary
/// exists — rare for prose but safe.
fn truncateAtSentence(
    text: []const u8,
    max_len: usize,
) []const u8 {
    std.debug.assert(max_len > 0);
    if (text.len <= max_len) return text;

    // Scan backwards from max_len for a sentence end.
    var pos: usize = max_len;
    while (pos > 0) {
        pos -= 1;
        const c = text[pos];
        const is_ender = c == '.' or c == '!' or
            c == '?' or c == ')';
        if (!is_ender) continue;

        // Accept if at the cut boundary or followed
        // by whitespace (start of next sentence).
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

    // No sentence boundary — hard cut as last resort.
    return text[0..max_len];
}

// ============================================================
// Logging
// ============================================================

const stdout_file = std.fs.File{
    .handle = std.posix.STDOUT_FILENO,
};

fn printHeader() void {
    api.log(
        "\nnnzap engine agent" ++
            " - LLM-powered engine optimiser\n" ++
            "==================================" ++
            "================\n\n",
        .{},
    );
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
