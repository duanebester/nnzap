//! Golden output test for Bonsai 1.7B inference.
//!
//! Loads the real model, runs a short prompt with greedy
//! decoding, and verifies the output token IDs match a
//! hardcoded golden sequence.  This is the correctness
//! gate for the autonomous optimisation agent — it tests
//! the ENTIRE pipeline (embedding, all 28 decoder blocks,
//! final norm, LM head, sampling) without constraining
//! how the internals are implemented.
//!
//! Usage:
//!   zig build run-bonsai-golden               # verify
//!   zig build run-bonsai-golden -- --capture   # print golden tokens

const std = @import("std");
const nn = @import("nn");

// ── Config aliases ───────────────────────────────────
const Config = nn.Bonsai1_7B;
const BonsaiModel = nn.Model(Config);
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
///   zig build run-bonsai-golden -- --capture
const GOLDEN_TOKEN_COUNT: u32 = 11;

const GOLDEN_TOKENS = [GOLDEN_TOKEN_COUNT]u32{
    151667, 198, 151668, 271, 785, 6722, 315, 9625, 374, 12095,
    13,
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

pub fn main() !void {
    const capture_mode = checkCaptureFlag();
    const model_dir = resolveModelDir();

    std.debug.print(
        "\n\x1b[1m[nnmetal] Bonsai 1.7B Golden Test" ++
            "\x1b[0m\n" ++
            "model:   {s}\n" ++
            "mode:    {s}\n" ++
            "prompt:  \"{s}\"\n" ++
            "max tok: {d}\n\n",
        .{
            model_dir,
            if (capture_mode) "capture" else "verify",
            GOLDEN_PROMPT,
            MAX_OUTPUT_TOKENS,
        },
    );

    var device: nn.Device = undefined;
    try device.init();

    var pipelines: nn.TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    try nn.specialized_qmv.initOnDevice(
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

    const output = try runGenerate(
        &model_val,
        &device,
        &pipelines,
        prompt_ids[0..prompt_len],
        &tokenizer,
    );

    if (capture_mode) {
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
    model: *BonsaiModel,
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
            "\n\x1b[31mGolden output test FAILED" ++
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
            "\x1b[32mGolden output test PASSED " ++
                "({d}/{d} tokens match)\x1b[0m\n",
            .{ GOLDEN_TOKEN_COUNT, GOLDEN_TOKEN_COUNT },
        );
    } else {
        std.debug.print(
            "\n\x1b[31mGolden output test FAILED" ++
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
) !BonsaiModel {
    std.debug.assert(model_dir.len > 0);
    std.debug.assert(model_dir.len < MAX_PATH_LEN);

    std.debug.print(
        "Allocating model buffers...\n",
        .{},
    );

    var model: BonsaiModel = undefined;
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

/// Scan argv for the --capture flag.
fn checkCaptureFlag() bool {
    var iter = std.process.args();
    _ = iter.next(); // Skip argv[0].

    while (iter.next()) |arg| {
        std.debug.assert(arg.len > 0);
        if (std.mem.eql(u8, arg, "--capture")) {
            return true;
        }
    }

    return false;
}

/// Resolve model directory from CLI args or default to
/// ~/models/bonsai-1.7b.
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

    // Fall back to ~/models/bonsai-1.7b.
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
        "{s}/models/bonsai-1.7b",
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
/// of shards found.  Shards are sorted alphabetically for
/// deterministic loading order.
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
