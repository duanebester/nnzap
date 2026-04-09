//! Golden output test for Bonsai 1.7B Q4 inference.
//!
//! Loads the real Q4 model, runs a short prompt with
//! greedy decoding, and verifies the output token IDs
//! match a hardcoded golden sequence.  This is the
//! correctness gate for the autonomous Q4 optimisation
//! agent — it tests the ENTIRE pipeline (embedding, all
//! 28 decoder blocks, final norm, LM head, sampling)
//! without constraining how the internals are
//! implemented.
//!
//! Usage:
//!   zig build run-bonsai-q4-golden               # verify
//!   zig build run-bonsai-q4-golden -- --capture   # print golden tokens
//!   zig build run-bonsai-q4-golden -- --diagnose  # dump logits + embeddings

const std = @import("std");
const nn = @import("nn");
const metal = nn.metal;

// ── Config aliases ───────────────────────────────────
const Config = nn.Bonsai1_7B_Q4;
const Q4Model = nn.Model(Config);
const transformer = nn.transformer;

// ── Limits ───────────────────────────────────────────
const MAX_SHARDS: u32 = 16;
const MAX_PATH_LEN: u32 = 4096;
const MAX_PROMPT_TOKENS: u32 = 8192;
const MAX_DECODE_BUF: u32 = 4096;
const MAX_OUTPUT_TOKENS: u32 = 20;

const GOLDEN_PROMPT = "The capital of France is";

/// Number of tokens the golden run is expected to
/// produce (may be less than MAX_OUTPUT_TOKENS if
/// the model hits EOS).
///
/// To regenerate after a deliberate precision change:
///   zig build run-bonsai-q4-golden -- --capture
const GOLDEN_TOKEN_COUNT: u32 = 20;

const GOLDEN_TOKENS = [GOLDEN_TOKEN_COUNT]u32{
    198,  279, 198, 279,  6722, 315, 315, 9625, 374, 198,
    6722, 315, 315, 9625, 374,  198, 279, 6722, 315, 279,
};

// ============================================================
// Generate result — token IDs + count
// ============================================================

const GoldenOutput = struct {
    tokens: [MAX_OUTPUT_TOKENS]u32,
    count: u32,
};

// ============================================================
// Entry point
// ============================================================

const RunMode = enum { verify, capture, diagnose, unfused };

pub fn main() !void {
    const mode = checkRunMode();
    const model_dir = resolveModelDir();

    std.debug.print(
        "\n\x1b[1m[nnmetal] Bonsai 1.7B Q4 Golden Test" ++
            "\x1b[0m\n" ++
            "model:   {s}\n" ++
            "quant:   Q4 MLX (4-bit affine, gs=64)\n" ++
            "mode:    {s}\n" ++
            "prompt:  \"{s}\"\n" ++
            "max tok: {d}\n\n",
        .{
            model_dir,
            if (mode == .unfused) "unfused+capture" else @tagName(mode),
            GOLDEN_PROMPT,
            MAX_OUTPUT_TOKENS,
        },
    );

    var device: nn.Device = undefined;
    try device.init();

    var pipelines: nn.TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    // Compile specialized Q4 QMV kernels with model
    // dimensions baked in as constexpr.
    try nn.specialized_q4mv.initOnDevice(
        &device,
        Config.hidden_size,
        Config.intermediate_size,
        Config.group_size,
    );

    var model_val = try loadModel(model_dir, &device);
    defer model_val.deinit();

    var tokenizer = try loadTokenizer(model_dir);
    defer tokenizer.deinit();

    var prompt_ids: [MAX_PROMPT_TOKENS]u32 = undefined;
    const prompt_len = try tokenizer.applyChatTemplate(
        GOLDEN_PROMPT,
        &prompt_ids,
    );
    std.debug.assert(prompt_len > 0);
    std.debug.assert(prompt_len <= MAX_PROMPT_TOKENS);

    std.debug.print(
        "Prompt encoded: {d} tokens\n",
        .{prompt_len},
    );
    // Disable fused norm kernels if requested — forces the
    // separate RMSNorm + QMV path for all Q4 projections.
    if (mode == .unfused) {
        std.debug.print(
            "\n*** UNFUSED MODE: disabling fused norm" ++
                " Q4 pipelines ***\n\n",
            .{},
        );
        device.spec_q4mv_fused_norm_f16io = null;
        device.spec_q4mv_fused_norm_pair_f16io = null;
        device.spec_q4mv_fused_norm_pair_silu_f16io = null;
    }

    // Print actual prompt token IDs for cross-reference.
    std.debug.print("Prompt IDs:", .{});
    for (prompt_ids[0..prompt_len]) |tid| {
        std.debug.print(" {d}", .{tid});
    }
    std.debug.print("\n", .{});

    if (mode == .diagnose) {
        runDiagnose(
            &model_val,
            &device,
            &pipelines,
            prompt_ids[0..prompt_len],
        );
        return;
    }

    if (mode == .unfused) {
        // Run generate in capture mode with fused kernels
        // disabled — prints tokens for comparison.
        const output = try runGenerate(
            &model_val,
            &device,
            &pipelines,
            prompt_ids[0..prompt_len],
            &tokenizer,
        );
        try printCapture(&output, &tokenizer);
        return;
    }

    const output = try runGenerate(
        &model_val,
        &device,
        &pipelines,
        prompt_ids[0..prompt_len],
        &tokenizer,
    );

    if (mode == .capture) {
        try printCapture(&output, &tokenizer);
    } else {
        verifyGolden(&output);
    }
}

// ============================================================
// Generation
// ============================================================

/// Run greedy generation and return the output tokens
/// with the actual count of tokens produced.
fn runGenerate(
    model: *Q4Model,
    device: *nn.Device,
    pipelines: *nn.TransformerPipelines,
    prompt_ids: []const u32,
    tokenizer: *const nn.Tokenizer,
) !GoldenOutput {
    std.debug.assert(prompt_ids.len > 0);
    std.debug.assert(
        prompt_ids.len < Config.max_context_length,
    );

    // EOS tokens (Qwen3 uses two stop tokens).
    const eos_ids = [_]u32{
        tokenizer.eos_token_id,
        tokenizer.im_end_token_id,
    };

    // Sampling scratch (heap, init-time only).
    const page_alloc = std.heap.page_allocator;

    const scratch = try page_alloc.alloc(
        f32,
        Config.vocab_size,
    );
    defer page_alloc.free(scratch);

    const indices = try page_alloc.alloc(
        u32,
        Config.vocab_size,
    );
    defer page_alloc.free(indices);

    // Greedy decoding: temperature=0 for determinism.
    const sampling = transformer.SamplingParams{
        .temperature = 0.0,
        .top_k = 0,
        .top_p = 1.0,
        .seed = 42,
    };

    // Zero-fill so ungenerated slots are deterministic.
    var output = GoldenOutput{
        .tokens = [_]u32{0} ** MAX_OUTPUT_TOKENS,
        .count = 0,
    };
    var args = model.forwardDecodeArgs(0, 0);

    std.debug.print("Running generate...\n", .{});

    const result = transformer.generate(
        Config,
        device,
        pipelines,
        &args,
        .{
            .prompt_ids = prompt_ids,
            .params = sampling,
            .eos_ids = &eos_ids,
            .output_tokens = &output.tokens,
            .scratch = scratch,
            .indices = indices,
        },
    );

    output.count = result.tokens_generated;

    std.debug.print(
        "Generated {d}/{d} tokens " ++
            "(prefill {d:.1} ms)\n\n",
        .{
            result.tokens_generated,
            MAX_OUTPUT_TOKENS,
            nanosToMs(result.prefill_ns),
        },
    );

    return output;
}

// ============================================================
// Capture mode
// ============================================================

/// Print generated tokens as a Zig array literal for
/// pasting back into this file.
fn printCapture(
    output: *const GoldenOutput,
    tokenizer: *const nn.Tokenizer,
) !void {
    std.debug.assert(output.count <= MAX_OUTPUT_TOKENS);
    std.debug.assert(MAX_OUTPUT_TOKENS > 0);

    const count = output.count;
    const tokens = output.tokens[0..count];

    // Format the array literal into a stack buffer.
    var buf: [MAX_DECODE_BUF]u8 = undefined;
    var pos: usize = 0;

    pos += (std.fmt.bufPrint(
        buf[pos..],
        "Golden output ({d} tokens generated):\n\n" ++
            "const GOLDEN_TOKEN_COUNT: u32 = {d};\n\n" ++
            "const GOLDEN_TOKENS = " ++
            "[GOLDEN_TOKEN_COUNT]u32{{\n    ",
        .{ count, count },
    ) catch return error.OutputBufferTooSmall).len;

    pos += formatTokenList(tokens, buf[pos..]);

    pos += (std.fmt.bufPrint(
        buf[pos..],
        "\n}};\n\n",
        .{},
    ) catch return error.OutputBufferTooSmall).len;

    // Decode and append the generated text.
    var decode_buf: [MAX_DECODE_BUF]u8 = undefined;
    const decoded_len = try tokenizer.decode(
        tokens,
        &decode_buf,
    );
    std.debug.assert(decoded_len <= MAX_DECODE_BUF);

    pos += (std.fmt.bufPrint(
        buf[pos..],
        "Generated text: {s}\n",
        .{decode_buf[0..decoded_len]},
    ) catch return error.OutputBufferTooSmall).len;

    // Write to stdout in one call.
    _ = std.posix.write(
        std.posix.STDOUT_FILENO,
        buf[0..pos],
    ) catch {};
}

/// Format the token ID list as comma-separated values,
/// wrapping every 10 entries.  Returns bytes written.
fn formatTokenList(
    tokens: []const u32,
    buf: []u8,
) usize {
    std.debug.assert(buf.len > 0);

    var pos: usize = 0;
    for (tokens, 0..) |tok, i| {
        if (i > 0 and i % 10 == 0) {
            const nl = std.fmt.bufPrint(
                buf[pos..],
                "\n    ",
                .{},
            ) catch break;
            pos += nl.len;
        }
        const sep: []const u8 =
            if (i + 1 < tokens.len) ", " else ",";
        const entry = std.fmt.bufPrint(
            buf[pos..],
            "{d}{s}",
            .{ tok, sep },
        ) catch break;
        pos += entry.len;
    }
    return pos;
}

// ============================================================
// Verify mode
// ============================================================

/// Compare generated tokens against the golden sequence.
/// Exits with code 1 on mismatch, 0 on success.
fn verifyGolden(output: *const GoldenOutput) void {
    std.debug.assert(output.count <= MAX_OUTPUT_TOKENS);

    // Check token count first.
    if (output.count != GOLDEN_TOKEN_COUNT) {
        std.debug.print(
            "\n\x1b[31mQ4 golden output test FAILED" ++
                "\x1b[0m\n" ++
                "  token count: expected {d}, " ++
                "got {d}\n",
            .{ GOLDEN_TOKEN_COUNT, output.count },
        );
        std.process.exit(1);
    }

    // Compare each token.
    var mismatch_count: u32 = 0;
    const count: usize = @intCast(output.count);

    for (
        output.tokens[0..count],
        GOLDEN_TOKENS[0..count],
        0..,
    ) |got, expected, i| {
        if (got != expected) {
            mismatch_count += 1;
            std.debug.print(
                "  token[{d}]: expected {d}, " ++
                    "got {d}\n",
                .{ i, expected, got },
            );
        }
    }

    if (mismatch_count == 0) {
        std.debug.print(
            "\x1b[32mQ4 golden output test PASSED " ++
                "({d}/{d} tokens match)\x1b[0m\n",
            .{ GOLDEN_TOKEN_COUNT, GOLDEN_TOKEN_COUNT },
        );
    } else {
        std.debug.print(
            "\n\x1b[31mQ4 golden output test FAILED" ++
                "\x1b[0m\n" ++
                "{d}/{d} tokens mismatched\n",
            .{ mismatch_count, GOLDEN_TOKEN_COUNT },
        );
        std.process.exit(1);
    }
}

// ============================================================
// Model and tokenizer loading
// ============================================================

/// Allocate the model and load safetensor weights.
fn loadModel(
    model_dir: [:0]const u8,
    device: *nn.Device,
) !Q4Model {
    std.debug.assert(model_dir.len > 0);
    std.debug.assert(model_dir.len < MAX_PATH_LEN);

    std.debug.print(
        "Allocating model buffers...\n",
        .{},
    );

    var model: Q4Model = undefined;
    try model.init(device.obj);

    std.debug.print("Loading weights...\n", .{});

    var shard_storage: [MAX_SHARDS][MAX_PATH_LEN]u8 =
        undefined;
    var shard_slices: [MAX_SHARDS][]const u8 = undefined;
    const shard_count = try findSafetensorShards(
        model_dir,
        &shard_storage,
        &shard_slices,
    );
    try model.loadWeights(
        shard_slices[0..shard_count],
    );

    return model;
}

/// Build tokenizer path and initialise.
fn loadTokenizer(
    model_dir: [:0]const u8,
) !nn.Tokenizer {
    std.debug.assert(model_dir.len > 0);
    std.debug.assert(model_dir.len < MAX_PATH_LEN);

    std.debug.print("Loading tokenizer...\n", .{});

    var path_buf: [MAX_PATH_LEN]u8 = undefined;
    const path = try std.fmt.bufPrint(
        &path_buf,
        "{s}/tokenizer.json",
        .{@as([]const u8, model_dir)},
    );

    var tokenizer: nn.Tokenizer = undefined;
    try tokenizer.init(std.heap.page_allocator, path);

    return tokenizer;
}

// ============================================================
// CLI argument parsing
// ============================================================

/// Scan argv for --capture or --diagnose flag.
fn checkRunMode() RunMode {
    var iter = std.process.args();
    _ = iter.next(); // Skip argv[0].

    while (iter.next()) |arg| {
        std.debug.assert(arg.len > 0);
        if (std.mem.eql(u8, arg, "--capture")) {
            return .capture;
        }
        if (std.mem.eql(u8, arg, "--diagnose")) {
            return .diagnose;
        }
        if (std.mem.eql(u8, arg, "--unfused")) {
            return .unfused;
        }
    }

    return .verify;
}

/// Resolve model directory from CLI args, local data/,
/// or default to ~/models/qwen3-1.7b-q4.
fn resolveModelDir() [:0]const u8 {
    var iter = std.process.args();
    _ = iter.next(); // Skip argv[0].

    // First positional argument overrides the default.
    // Skip flags (anything starting with '--').
    while (iter.next()) |arg| {
        std.debug.assert(arg.len > 0);
        if (!std.mem.startsWith(u8, arg, "--")) {
            return arg;
        }
    }

    // Try data/qwen3-1.7b-q4 relative to cwd (works
    // when running from the nnmetal project root via
    // `zig build run-bonsai-q4-golden`).
    const local_path = "data/qwen3-1.7b-q4";
    if (std.fs.cwd().access(
        local_path,
        .{},
    )) |_| {
        return local_path;
    } else |_| {}

    // Fall back to ~/models/qwen3-1.7b-q4.
    const home = std.posix.getenv("HOME") orelse {
        std.debug.print(
            "error: HOME not set and no model dir " ++
                "provided\n",
            .{},
        );
        std.process.exit(1);
    };
    std.debug.assert(home.len > 0);

    // Build the default path into a static buffer so
    // the returned slice lives for the program lifetime.
    const S = struct {
        var buf: [MAX_PATH_LEN]u8 = undefined;
        var sentinel_buf: [MAX_PATH_LEN + 1]u8 =
            undefined;
    };
    const path = std.fmt.bufPrint(
        &S.buf,
        "{s}/models/qwen3-1.7b-q4",
        .{home},
    ) catch {
        std.debug.print(
            "error: HOME path too long\n",
            .{},
        );
        std.process.exit(1);
    };

    // Copy into sentinel-terminated buffer.
    @memcpy(S.sentinel_buf[0..path.len], path);
    S.sentinel_buf[path.len] = 0;
    return S.sentinel_buf[0..path.len :0];
}

// ============================================================
// Safetensors shard discovery
// ============================================================

/// Scan `dir_path` for *.safetensors files and populate
/// `path_storage` and `path_slices`.  Returns the number
/// of shards found.  Shards are sorted alphabetically
/// for deterministic loading order.
fn findSafetensorShards(
    dir_path: []const u8,
    path_storage: *[MAX_SHARDS][MAX_PATH_LEN]u8,
    path_slices: *[MAX_SHARDS][]const u8,
) !u32 {
    std.debug.assert(dir_path.len > 0);
    std.debug.assert(dir_path.len < MAX_PATH_LEN);

    var dir = std.fs.cwd().openDir(
        dir_path,
        .{ .iterate = true },
    ) catch |err| {
        std.debug.print(
            "error: cannot open model directory: " ++
                "{s} ({})\n",
            .{ dir_path, err },
        );
        return err;
    };
    defer dir.close();

    var count: u32 = 0;
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        const name = entry.name;
        if (!std.mem.endsWith(
            u8,
            name,
            ".safetensors",
        )) {
            continue;
        }
        if (count >= MAX_SHARDS) {
            std.debug.print(
                "error: too many shards (max {d})\n",
                .{MAX_SHARDS},
            );
            return error.TooManyShards;
        }
        const path = std.fmt.bufPrint(
            &path_storage[count],
            "{s}/{s}",
            .{ dir_path, name },
        ) catch {
            return error.PathTooLong;
        };
        path_slices[count] = path;
        count += 1;
    }

    if (count == 0) {
        std.debug.print(
            "error: no .safetensors files in {s}\n",
            .{dir_path},
        );
        return error.NoSafetensorFiles;
    }

    // Sort for deterministic shard loading order.
    std.sort.pdq(
        []const u8,
        path_slices[0..count],
        {},
        pathLessThan,
    );

    std.debug.print(
        "Found {d} safetensor shard(s)\n",
        .{count},
    );
    return count;
}

fn pathLessThan(
    _: void,
    a: []const u8,
    b: []const u8,
) bool {
    return std.mem.order(u8, a, b) == .lt;
}

// ============================================================
// Utility
// ============================================================

fn nanosToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

// ============================================================
// Diagnostic mode — compare with MLX reference
// ============================================================

const DIAG_SEP = "=" ** 60;

/// Run a manual prefill, then dump embedding values and
/// logits for comparison with the MLX reference script
/// (reference/mlx_q4_compare.py).
fn runDiagnose(
    model: *Q4Model,
    device: *nn.Device,
    pipelines: *nn.TransformerPipelines,
    prompt_ids: []const u32,
) void {
    std.debug.assert(prompt_ids.len > 0);
    std.debug.assert(
        prompt_ids.len < Config.max_context_length,
    );

    var args = model.forwardDecodeArgs(0, 0);

    std.debug.print(
        "\n{s}\n  DIAGNOSE: processing {d} prompt" ++
            " tokens\n{s}\n",
        .{ DIAG_SEP, prompt_ids.len, DIAG_SEP },
    );

    // ── Step 1: Forward pass for first token.
    args.token_id = prompt_ids[0];
    args.position = 0;
    {
        const cmd = device.beginCommandBufferUnretained();
        const enc = device.beginCompute(cmd);
        transformer.forwardDecode(
            Config,
            device,
            enc,
            pipelines,
            args,
        );
        enc.msgSend(
            void,
            "memoryBarrierWithScope:",
            .{@as(c_ulong, 1)},
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }

    dumpLogits(
        "Logits after token 0",
        model.logits.asSlice(),
    );

    // Dump the f32 residual after the full forward pass
    // for token 0 (= hidden state after all 28 layers).
    dumpResidual(
        "Residual after token 0 (all 28 layers)",
        model.residual.asSlice(),
    );

    // ── Step 2: Prefill remaining tokens.
    for (prompt_ids[1..], 1..) |tok, i| {
        args.token_id = tok;
        args.position = @intCast(i);
        const cmd = device.beginCommandBufferUnretained();
        const enc = device.beginCompute(cmd);
        transformer.forwardDecode(
            Config,
            device,
            enc,
            pipelines,
            args,
        );
        enc.msgSend(
            void,
            "memoryBarrierWithScope:",
            .{@as(c_ulong, 1)},
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }

    dumpLogits(
        "Logits after full prefill (last pos)",
        model.logits.asSlice(),
    );

    // Dump the f32 residual after full prefill.
    dumpResidual(
        "Residual after full prefill (last pos)",
        model.residual.asSlice(),
    );

    // ── Step 3: Dump Q4 weight metadata.
    dumpQ4BufferInfo("Embedding", model.embedding);
    dumpQ4FirstRow("Embedding row 0", model.embedding);

    dumpQ4BufferInfo("Layer 0 q_proj", model.q_proj[0]);
    dumpQ4FirstRow(
        "Layer 0 q_proj row 0",
        model.q_proj[0],
    );

    // ── Step 4: CPU-side embedding dequant for token 0.
    dumpEmbeddingCPU(model, prompt_ids[0]);

    // ── Step 5: GPU embedding verification — dispatch
    //    embedding_lookup_q4 alone, read back output,
    //    compare with CPU dequant.
    verifyGPUEmbedding(model, device, pipelines, prompt_ids[0]);

    // ── Step 5b: Run layer-0 operations step by step on
    //    the fresh embedding, dumping intermediates for
    //    comparison with MLX layer-0 values.
    stepByStepLayer0(model, device, pipelines);

    // ── Step 6: Dump norm_out (final norm output = LM head
    //    input) from the position-0 forward pass.
    dumpNormOut(
        "Final norm_out after token 0 (LM head input)",
        model,
    );

    // ── Step 7: CPU reference Q4 matmul for the LM head.
    //    After forwardDecode, norm_out holds the input to
    //    the LM head, and logits holds the GPU output.
    //    Compute the same matmul on CPU and compare.
    cpuQ4MatmulCheck(model);

    std.debug.print(
        "\n{s}\n  DIAGNOSE COMPLETE\n" ++
            "  Compare with: python reference/" ++
            "mlx_q4_compare.py\n{s}\n\n",
        .{ DIAG_SEP, DIAG_SEP },
    );
}

/// Read f32 logits from buffer and print top 20.
fn dumpLogits(label: []const u8, logits: []f32) void {
    std.debug.assert(logits.len >= Config.vocab_size);
    const vs = logits[0..Config.vocab_size];

    // Find top 20 by repeated linear scan.
    const TOP_N = 20;
    var top_idx: [TOP_N]u32 = undefined;
    var top_val: [TOP_N]f32 = undefined;
    for (0..TOP_N) |rank| {
        var best_i: u32 = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (vs, 0..) |v, idx| {
            var skip = false;
            for (top_idx[0..rank]) |prev| {
                if (prev == @as(u32, @intCast(idx))) {
                    skip = true;
                    break;
                }
            }
            if (skip) continue;
            if (v > best_v) {
                best_v = v;
                best_i = @intCast(idx);
            }
        }
        top_idx[rank] = best_i;
        top_val[rank] = best_v;
    }

    // Stats.
    var sum: f64 = 0.0;
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    for (vs) |v| {
        sum += @as(f64, v);
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    const mean_v: f32 = @floatCast(
        sum / @as(f64, @floatFromInt(Config.vocab_size)),
    );

    std.debug.print(
        "\n{s}\n  {s}\n" ++
            "  vocab_size={d}  min={d:.4}" ++
            "  max={d:.4}  mean={d:.4}\n" ++
            "{s}\n" ++
            "  Rank     Index        Logit\n" ++
            "  ----  --------  -----------\n",
        .{
            DIAG_SEP,
            label,
            Config.vocab_size,
            min_v,
            max_v,
            mean_v,
            DIAG_SEP,
        },
    );
    for (0..TOP_N) |rank| {
        std.debug.print(
            "  {d:4}  {d:8}  {d:11.4}\n",
            .{ rank, top_idx[rank], top_val[rank] },
        );
    }
}

/// Dump first 16 values of the f32 residual buffer, plus
/// stats (min, max, mean) across all hidden_size elements.
fn dumpResidual(
    label: []const u8,
    residual: []f32,
) void {
    std.debug.assert(residual.len >= Config.hidden_size);
    const h = residual[0..Config.hidden_size];

    var sum: f64 = 0.0;
    var min_v: f32 = std.math.inf(f32);
    var max_v: f32 = -std.math.inf(f32);
    for (h) |v| {
        sum += @as(f64, v);
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    const mean_v: f32 = @floatCast(
        sum / @as(f64, @floatFromInt(Config.hidden_size)),
    );

    std.debug.print(
        "\n{s}\n  {s}\n" ++
            "  min={d:.6}  max={d:.6}  mean={d:.6}\n" ++
            "{s}\n",
        .{ DIAG_SEP, label, min_v, max_v, mean_v, DIAG_SEP },
    );
    for (0..16) |i| {
        std.debug.print(
            "  [{d:4}] = {d:.6}\n",
            .{ i, h[i] },
        );
    }
}

/// Dump first 16 values of the f16 norm_out buffer.
fn dumpNormOut(
    label: []const u8,
    model: *const Q4Model,
) void {
    const raw = model.norm_out.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const f16_ptr: [*]const f16 = @ptrCast(
        @alignCast(raw),
    );

    std.debug.print(
        "\n{s}\n  {s}\n{s}\n",
        .{ DIAG_SEP, label, DIAG_SEP },
    );
    for (0..16) |i| {
        const v: f32 = @floatCast(f16_ptr[i]);
        std.debug.print(
            "  [{d:4}] = {d:.6}\n",
            .{ i, v },
        );
    }
}

/// Dump Q4Buffer metadata.
fn dumpQ4BufferInfo(
    label: []const u8,
    buf: metal.Q4Buffer,
) void {
    std.debug.print(
        "\n  {s}: packed_count={d}, group_size={d}" ++
            ", nibbleBytes={d}, numGroups={d}" ++
            ", scaleOff={d}, biasOff={d}" ++
            ", total={d}\n",
        .{
            label,
            buf.packed_count,
            buf.group_size,
            buf.nibbleBytes(),
            buf.numGroups(),
            buf.scaleOffset(),
            buf.biasOffset(),
            buf.byte_len,
        },
    );
}

/// CPU-side dequant of first 8 weights from row 0 of a
/// Q4Buffer, for cross-checking with MLX.
fn dumpQ4FirstRow(
    label: []const u8,
    buf: metal.Q4Buffer,
) void {
    const raw_ptr = buf.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const raw: [*]const u8 = @ptrCast(raw_ptr);

    // First uint32 of packed nibbles (8 nibbles).
    const word_ptr: *const u32 = @ptrCast(
        @alignCast(raw),
    );
    const word0 = word_ptr.*;

    // First scale and bias.
    const scale_off = buf.scaleOffset();
    const scale_ptr: *const f16 = @ptrCast(
        @alignCast(raw + scale_off),
    );
    const bias_off = buf.biasOffset();
    const bias_ptr: *const f16 = @ptrCast(
        @alignCast(raw + bias_off),
    );
    const scale: f32 = @floatCast(scale_ptr.*);
    const bias: f32 = @floatCast(bias_ptr.*);

    std.debug.print(
        "\n  {s}:\n" ++
            "  word0=0x{x:0>8}  scale={d:.6}" ++
            "  bias={d:.6}\n",
        .{ label, word0, scale, bias },
    );

    for (0..8) |ni| {
        const shift: u5 = @intCast(ni * 4);
        const nibble = (word0 >> shift) & 0xF;
        const dequant = scale *
            @as(f32, @floatFromInt(nibble)) + bias;
        std.debug.print(
            "    col {d}: nibble={d:2}" ++
                "  dequant={d:.6}\n",
            .{ ni, nibble, dequant },
        );
    }
}

/// CPU reference Q4 matmul: read the norm_out (f16)
/// input and the embedding Q4 buffer (= LM head when
/// tie_word_embeddings), compute dot products on CPU
/// for specific rows, and compare with the GPU logits.
fn cpuQ4MatmulCheck(model: *const Q4Model) void {
    const H = Config.hidden_size;
    const GS = Config.group_size;
    const groups_per_row: u32 = H / GS;
    const nib_row_bytes: u32 = H / 2;

    // Read the f16 norm_out buffer (LM head input).
    const norm_ptr = model.norm_out.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const norm_f16: [*]const f16 = @ptrCast(
        @alignCast(norm_ptr),
    );

    // Read the f32 logits buffer (GPU output).
    const gpu_logits = model.logits.asSlice();

    // Read the Q4 embedding buffer (LM head weights).
    const buf = model.embedding;
    const raw_ptr = buf.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const raw: [*]const u8 = @ptrCast(raw_ptr);
    const scale_base = buf.scaleOffset();
    const bias_base = buf.biasOffset();
    const sc_all: [*]const f16 = @ptrCast(
        @alignCast(raw + scale_base),
    );
    const bi_all: [*]const f16 = @ptrCast(
        @alignCast(raw + bias_base),
    );

    // Test rows: include top MLX logit (151667) and
    // top nnmetal logit (198), plus rows 0 and 94.
    const test_rows = [_]u32{ 0, 94, 198, 151667 };

    std.debug.print(
        "\n{s}\n  CPU vs GPU Q4 matmul (LM head)" ++
            "\n  K={d}, group_size={d}" ++
            "\n{s}\n" ++
            "       Row     CPU logit     GPU logit" ++
            "           Diff\n" ++
            "  --------  --------------" ++
            "  --------------  --------------\n",
        .{ DIAG_SEP, H, GS, DIAG_SEP },
    );

    for (test_rows) |row| {
        if (row >= Config.vocab_size) continue;

        // Pointer to this row's packed nibbles.
        const row_nibs = raw +
            @as(usize, row) * nib_row_bytes;

        // Compute dot(dequant(W[row]), norm_out) in f64.
        var dot: f64 = 0.0;
        var col: u32 = 0;
        while (col < H) : (col += 1) {
            // Extract nibble.
            const word_idx = col / 8;
            const nib_shift: u5 = @intCast(
                (col % 8) * 4,
            );
            const word_ptr: *const u32 = @ptrCast(
                @alignCast(row_nibs + word_idx * 4),
            );
            const nibble: u32 =
                (word_ptr.* >> nib_shift) & 0xF;

            // Scale and bias for this group.
            const grp: u32 = col / GS;
            const sc_idx: usize =
                @as(usize, row) * groups_per_row + grp;
            const scale: f64 =
                @floatCast(sc_all[sc_idx]);
            const bias: f64 =
                @floatCast(bi_all[sc_idx]);

            // Dequantize.
            const w: f64 = scale *
                @as(f64, @floatFromInt(nibble)) + bias;

            // Input.
            const x: f64 =
                @floatCast(norm_f16[col]);

            dot += w * x;
        }

        const cpu_val: f32 = @floatCast(dot);
        const gpu_val: f32 = gpu_logits[row];
        const diff: f32 = cpu_val - gpu_val;

        std.debug.print(
            "  {d:8}  {d:14.4}  {d:14.4}  {d:14.6}\n",
            .{ row, cpu_val, gpu_val, diff },
        );
    }

    // Also dump first 8 values of norm_out for
    // cross-reference with MLX.
    std.debug.print(
        "\n  norm_out (LM head input, first 8):\n",
        .{},
    );
    for (0..8) |i| {
        const v: f32 = @floatCast(norm_f16[i]);
        std.debug.print(
            "    [{d}] = {d:.6}\n",
            .{ i, v },
        );
    }
}

/// Dispatch embedding_lookup_q4 alone in a fresh command
/// buffer, read back the f32 output, and compare with
/// the CPU dequant to verify the GPU kernel.
fn verifyGPUEmbedding(
    model: *Q4Model,
    device: *nn.Device,
    pipelines: *nn.TransformerPipelines,
    token_id: u32,
) void {
    std.debug.assert(token_id < Config.vocab_size);

    const H = Config.hidden_size;
    const GS = Config.group_size;
    const groups_per_row: u32 = H / GS;
    const nib_row_bytes: u32 = H / 2;

    // Write token_id into token_ids buffer.
    const tid_raw = model.token_ids.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const tid_u32: [*]u32 = @ptrCast(
        @alignCast(tid_raw),
    );
    tid_u32[0] = token_id;

    // Zero the residual buffer for a clean read.
    const res_slice = model.residual.asSlice();
    for (res_slice[0..H]) |*v| v.* = 0.0;

    // Dispatch embedding_lookup_q4 only.
    const cmd = device.beginCommandBufferUnretained();
    const enc = device.beginCompute(cmd);

    // buffer(0) = token_ids
    enc.msgSend(void, "setBuffer:offset:atIndex:", .{
        model.token_ids.obj.value,
        @as(c_ulong, 0),
        @as(c_ulong, 0),
    });
    // buffer(1,2,3) = Q4 embedding (nibs, scales, biases)
    metal.setQ4Buffer(enc, model.embedding, 1);
    // buffer(4) = output (residual)
    enc.msgSend(void, "setBuffer:offset:atIndex:", .{
        model.residual.obj.value,
        @as(c_ulong, 0),
        @as(c_ulong, 4),
    });
    // buffer(5) = dims
    const embed_dims = transformer.EmbedDims{
        .vocab_size = Config.vocab_size,
        .hidden_size = H,
        .num_tokens = 1,
        .group_size = GS,
    };
    metal.setBytes(
        enc,
        transformer.EmbedDims,
        &embed_dims,
        5,
    );
    device.dispatch1D(
        enc,
        pipelines.embedding_lookup_q4,
        H,
    );

    enc.msgSend(
        void,
        "memoryBarrierWithScope:",
        .{@as(c_ulong, 1)},
    );
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);

    // Read GPU output.
    const gpu_out = model.residual.asSlice();

    // CPU dequant for comparison.
    const buf = model.embedding;
    const raw_ptr = buf.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const raw: [*]const u8 = @ptrCast(raw_ptr);
    const sc_all: [*]const f16 = @ptrCast(
        @alignCast(raw + buf.scaleOffset()),
    );
    const bi_all: [*]const f16 = @ptrCast(
        @alignCast(raw + buf.biasOffset()),
    );
    const row: u32 = token_id;
    const row_nibs = raw + @as(usize, row) * nib_row_bytes;

    std.debug.print(
        "\n{s}\n  GPU vs CPU embedding for token {d}" ++
            " (first 16 dims)\n{s}\n" ++
            "     Col      GPU val      CPU val" ++
            "           Diff\n" ++
            "  ------  -----------  -----------" ++
            "  -------------\n",
        .{ DIAG_SEP, token_id, DIAG_SEP },
    );

    var max_diff: f32 = 0.0;
    for (0..H) |col| {
        const grp: u32 = @intCast(col / GS);
        const sc_idx: usize =
            @as(usize, row) * groups_per_row + grp;
        const scale: f32 = @floatCast(sc_all[sc_idx]);
        const bias: f32 = @floatCast(bi_all[sc_idx]);

        const word_idx = col / 8;
        const nib_shift: u5 = @intCast((col % 8) * 4);
        const word_ptr: *const u32 = @ptrCast(
            @alignCast(row_nibs + word_idx * 4),
        );
        const nibble = (word_ptr.* >> nib_shift) & 0xF;
        const cpu_val = scale *
            @as(f32, @floatFromInt(nibble)) + bias;

        const gpu_val = gpu_out[col];
        const diff = gpu_val - cpu_val;
        const abs_diff = if (diff < 0) -diff else diff;
        if (abs_diff > max_diff) max_diff = abs_diff;

        if (col < 16) {
            std.debug.print(
                "  {d:6}  {d:11.6}  {d:11.6}" ++
                    "  {d:13.8}\n",
                .{ col, gpu_val, cpu_val, diff },
            );
        }
    }

    std.debug.print(
        "\n  Max abs diff across all {d} dims: {d:.8}\n",
        .{ H, max_diff },
    );
}

/// Run layer-0 operations step by step on the embedding
/// that verifyGPUEmbedding left in the residual buffer.
/// Dispatches RMSNorm and Q4 QMV individually, reading
/// back after each to compare with MLX reference values.
fn stepByStepLayer0(
    model: *Q4Model,
    device: *nn.Device,
    pipelines: *nn.TransformerPipelines,
) void {
    const H = Config.hidden_size;

    // Struct matching Metal's RMSNormDims.
    const NormDims = extern struct {
        hidden_size: u32,
        num_tokens: u32,
        eps: f32,
    };
    // Struct matching Metal's QMVDims.
    const MVDims = extern struct {
        M: u32,
        K: u32,
        group_size: u32,
    };

    // ── A: GPU RMSNorm(embedding) → norm_out ──────
    {
        const dims = NormDims{
            .hidden_size = H,
            .num_tokens = 1,
            .eps = 1e-6,
        };
        const cmd = device.beginCommandBufferUnretained();
        const enc = device.beginCompute(cmd);
        enc.msgSend(
            void,
            "setBuffer:offset:atIndex:",
            .{
                model.residual.obj.value,
                @as(c_ulong, 0),
                @as(c_ulong, 0),
            },
        );
        enc.msgSend(
            void,
            "setBuffer:offset:atIndex:",
            .{
                model.attn_norm[0].obj.value,
                @as(c_ulong, 0),
                @as(c_ulong, 1),
            },
        );
        enc.msgSend(
            void,
            "setBuffer:offset:atIndex:",
            .{
                model.norm_out.obj.value,
                @as(c_ulong, 0),
                @as(c_ulong, 2),
            },
        );
        metal.setBytes(enc, NormDims, &dims, 3);
        const grid = metal.MTLSize{
            .width = 1,
            .height = 1,
            .depth = 1,
        };
        const group = metal.MTLSize{
            .width = 256,
            .height = 1,
            .depth = 1,
        };
        device.dispatchCustom(
            enc,
            pipelines.rms_norm_f16out,
            grid,
            group,
        );
        enc.msgSend(
            void,
            "memoryBarrierWithScope:",
            .{@as(c_ulong, 1)},
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }

    // Read and dump norm_out (f16).
    const norm_raw = model.norm_out.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const norm_f16: [*]const f16 = @ptrCast(
        @alignCast(norm_raw),
    );
    std.debug.print(
        "\n{s}\n  Layer 0: GPU RMSNorm(embedding)" ++
            " -> norm_out\n" ++
            "  (compare with MLX attn norm output)" ++
            "\n{s}\n",
        .{ DIAG_SEP, DIAG_SEP },
    );
    for (0..16) |i| {
        const v: f32 = @floatCast(norm_f16[i]);
        std.debug.print(
            "  [{d:4}] = {d:.6}\n",
            .{ i, v },
        );
    }

    // ── B: GPU Q4 QMV q_proj(norm_out) → Q ───────
    {
        const dims = MVDims{
            .M = Config.query_dim,
            .K = H,
            .group_size = Config.group_size,
        };
        const pipeline =
            device.spec_q4mv_f16io orelse unreachable;
        const cmd = device.beginCommandBufferUnretained();
        const enc = device.beginCompute(cmd);
        metal.setQ4Buffer(enc, model.q_proj[0], 0);
        enc.msgSend(
            void,
            "setBuffer:offset:atIndex:",
            .{
                model.norm_out.obj.value,
                @as(c_ulong, 0),
                @as(c_ulong, 3),
            },
        );
        enc.msgSend(
            void,
            "setBuffer:offset:atIndex:",
            .{
                model.q.obj.value,
                @as(c_ulong, 0),
                @as(c_ulong, 4),
            },
        );
        metal.setBytes(enc, MVDims, &dims, 5);
        const rows_per_tg: u32 = 32;
        const tg_count =
            (dims.M + rows_per_tg - 1) / rows_per_tg;
        const grid = metal.MTLSize{
            .width = @as(c_ulong, tg_count),
            .height = 1,
            .depth = 1,
        };
        const group = metal.MTLSize{
            .width = 512,
            .height = 1,
            .depth = 1,
        };
        device.dispatchCustom(
            enc,
            pipeline,
            grid,
            group,
        );
        enc.msgSend(
            void,
            "memoryBarrierWithScope:",
            .{@as(c_ulong, 1)},
        );
        enc.msgSend(void, "endEncoding", .{});
        device.commitAndWait(cmd);
    }

    // Read and dump Q output (f16).
    const q_raw = model.q.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const q_f16: [*]const f16 = @ptrCast(
        @alignCast(q_raw),
    );
    std.debug.print(
        "\n{s}\n  Layer 0: GPU Q4 q_proj(norm_out)" ++
            " -> Q\n" ++
            "  (compare with MLX Q projection)" ++
            "\n{s}\n",
        .{ DIAG_SEP, DIAG_SEP },
    );
    for (0..16) |i| {
        const v: f32 = @floatCast(q_f16[i]);
        std.debug.print(
            "  [{d:4}] = {d:.6}\n",
            .{ i, v },
        );
    }
}

/// CPU-side embedding dequant for a single token,
/// reading directly from the Q4 embedding buffer
/// (no GPU dispatch needed).
fn dumpEmbeddingCPU(
    model: *const Q4Model,
    token_id: u32,
) void {
    std.debug.assert(token_id < Config.vocab_size);

    const buf = model.embedding;
    const raw_ptr = buf.obj.msgSend(
        *anyopaque,
        "contents",
        .{},
    );
    const raw: [*]const u8 = @ptrCast(raw_ptr);

    const H = Config.hidden_size;
    const GS = Config.group_size;
    const row: u32 = token_id;

    // Nibble bytes per row = H / 2.
    const nib_row_bytes: u32 = H / 2;
    const row_start = @as(usize, row) * nib_row_bytes;

    // Scales and biases start after all nibbles.
    const scale_base = buf.scaleOffset();
    const bias_base = buf.biasOffset();
    const groups_per_row: u32 = H / GS;

    std.debug.print(
        "\n{s}\n  CPU embedding dequant for" ++
            " token {d} (first 16 dims)\n{s}\n",
        .{ DIAG_SEP, token_id, DIAG_SEP },
    );

    // Cast nibble region to uint32 array.
    const nibs: [*]const u8 = raw + row_start;

    for (0..16) |col| {
        // Which group does this column belong to?
        const grp: u32 = @intCast(col / GS);
        const sc_idx = @as(usize, row) *
            groups_per_row + grp;

        // Read scale and bias (f16).
        const sc_ptr: [*]const f16 = @ptrCast(
            @alignCast(raw + scale_base),
        );
        const bi_ptr: [*]const f16 = @ptrCast(
            @alignCast(raw + bias_base),
        );
        const scale: f32 = @floatCast(sc_ptr[sc_idx]);
        const bias: f32 = @floatCast(bi_ptr[sc_idx]);

        // Extract the nibble.  Two nibbles per byte,
        // but MLX packs 8 nibbles per uint32 in the
        // pattern: nibble i at bits [i*4 .. i*4+3].
        // Byte-level: col/2, low/high nibble.
        // But the uint32 layout means we should read
        // uint32 words and extract by bit shift.
        const word_idx = col / 8;
        const nib_in_word: u5 = @intCast(
            (col % 8) * 4,
        );
        const word_ptr: *const u32 = @ptrCast(
            @alignCast(nibs + word_idx * 4),
        );
        const nibble = (word_ptr.* >> nib_in_word) & 0xF;
        const val = scale *
            @as(f32, @floatFromInt(nibble)) + bias;

        std.debug.print(
            "  [{d:4}] = {d:.6}\n",
            .{ col, val },
        );
    }
}
