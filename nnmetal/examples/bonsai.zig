//! Bonsai 1.7B inference — CLI entry point (Step 2f).
//!
//! Loads a Bonsai 1-bit model from safetensors, tokenizes
//! a prompt, and streams generated text to stdout.
//!
//! Usage:
//!   zig build run-bonsai -- <model_dir> [options]
//!
//! Options:
//!   --prompt <text>       (default: "Hello!")
//!   --max-tokens <n>      (default: 256)
//!   --temperature <f>     0 = greedy (default: 0.0)
//!   --top-k <n>           0 = disabled (default: 0)
//!   --top-p <f>           (default: 1.0)
//!   --seed <n>            (default: 42)
//!
//! The model directory must contain one or more
//! *.safetensors weight files and a tokenizer.json.

const std = @import("std");
const nn = @import("nn");

// ── Config aliases ───────────────────────────────────
const Config = nn.Bonsai1_7B;
const BonsaiModel = nn.Model(Config);
const transformer = nn.transformer;

// ── Limits ───────────────────────────────────────────
const MAX_SHARDS = 16;
const MAX_PATH_LEN = 4096;
const MAX_PROMPT_TOKENS = 8192;
const DEFAULT_MAX_GENERATE = 256;
const DECODE_BUF_LEN = 4096;

const USAGE =
    \\usage: bonsai <model_dir> [options]
    \\
    \\options:
    \\  --prompt <text>       Prompt text (default: Hello!)
    \\  --max-tokens <n>      Max tokens (default: 256)
    \\  --temperature <f>     0 = greedy (default: 0.0)
    \\  --top-k <n>           0 = disabled (default: 0)
    \\  --top-p <f>           Nucleus threshold (default: 1.0)
    \\  --seed <n>            Random seed (default: 42)
    \\
;

const CliArgs = struct {
    model_dir: [:0]const u8,
    prompt: [:0]const u8,
    max_tokens: u32,
    sampling: transformer.SamplingParams,
};

// ============================================================
// Entry point
// ============================================================

pub fn main() !void {
    // Parse command-line arguments.
    const cli = parseArgs();

    std.debug.print(
        "\n\x1b[1m[nnmetal] Bonsai 1.7B\x1b[0m\n" ++
            "model:       {s}\n" ++
            "prompt:      \"{s}\"\n" ++
            "max_tokens:  {d}\n" ++
            "temperature: {d:.2}  " ++
            "top_k: {d}  " ++
            "top_p: {d:.2}  " ++
            "seed: {d}\n\n",
        .{
            cli.model_dir,
            cli.prompt,
            cli.max_tokens,
            cli.sampling.temperature,
            cli.sampling.top_k,
            cli.sampling.top_p,
            cli.sampling.seed,
        },
    );

    // ── Metal device and pipelines ───────────────────
    std.debug.print("Initializing Metal...\n", .{});

    var device: nn.Device = undefined;
    try device.init();

    var pipelines: nn.TransformerPipelines = undefined;
    try pipelines.init(device.obj);

    // Compile specialized QMV kernels with model dimensions
    // baked in as constexpr for full loop unrolling.
    try nn.specialized_qmv.initOnDevice(
        &device,
        Config.hidden_size,
        Config.intermediate_size,
        Config.group_size,
    );

    // ── Model allocation and weight loading ──────────
    std.debug.print("Allocating model buffers...\n", .{});

    var model: BonsaiModel = undefined;
    try model.init(device.obj);
    defer model.deinit();

    std.debug.print("Loading weights...\n", .{});

    var shard_storage: [MAX_SHARDS][MAX_PATH_LEN]u8 =
        undefined;
    var shard_slices: [MAX_SHARDS][]const u8 = undefined;
    const shard_count = try findSafetensorShards(
        cli.model_dir,
        &shard_storage,
        &shard_slices,
    );
    try model.loadWeights(
        shard_slices[0..shard_count],
    );

    // ── Tokenizer ────────────────────────────────────
    std.debug.print("Loading tokenizer...\n", .{});

    var tokenizer_path_buf: [MAX_PATH_LEN]u8 = undefined;
    const tokenizer_path = try std.fmt.bufPrint(
        &tokenizer_path_buf,
        "{s}/tokenizer.json",
        .{@as([]const u8, cli.model_dir)},
    );

    var tokenizer: nn.Tokenizer = undefined;
    try tokenizer.init(
        std.heap.page_allocator,
        tokenizer_path,
    );
    defer tokenizer.deinit();

    // ── Encode prompt with chat template ─────────────
    var prompt_ids: [MAX_PROMPT_TOKENS]u32 = undefined;
    const prompt_len = try tokenizer.applyChatTemplate(
        cli.prompt,
        &prompt_ids,
    );
    std.debug.print(
        "Prompt: {d} tokens\n\n",
        .{prompt_len},
    );

    // ── EOS token set ────────────────────────────────
    // Qwen3 uses two stop tokens: the base EOS and
    // the chat-template <|im_end|>.
    const eos_ids = [_]u32{
        tokenizer.eos_token_id,
        tokenizer.im_end_token_id,
    };

    // ── Sampling scratch (heap, init-time only) ──────
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

    var prng = std.Random.DefaultPrng.init(
        cli.sampling.seed,
    );
    const rng = prng.random();

    // ── Chunked prefill ──────────────────────────────
    // Process prompt tokens one at a time through the
    // full decode path.  Slower than batched prefill
    // (qmm), but correct and simple.  Batched prefill
    // is a Step 3 optimisation.
    const t_prefill_start = std.time.nanoTimestamp();
    for (prompt_ids[0..prompt_len], 0..) |tok, pos| {
        dispatchOneDecode(
            &device,
            &pipelines,
            &model,
            tok,
            @intCast(pos),
        );
    }
    const t_prefill_end = std.time.nanoTimestamp();

    // Sample first token from prefill logits.
    var logits_slice = model.logits.asSlice();
    var next_token = transformer.sampleToken(
        logits_slice,
        cli.sampling,
        scratch,
        indices,
        rng,
    );

    // ── Streaming decode ─────────────────────────────
    const t_decode_start = std.time.nanoTimestamp();
    var position: u32 = @intCast(prompt_len);
    var generated: u32 = 0;
    var decode_buf: [DECODE_BUF_LEN]u8 = undefined;

    while (generated < cli.max_tokens) {
        // Check for EOS or context-length exhaustion.
        const is_eos = for (eos_ids) |eid| {
            if (next_token == eid) break true;
        } else false;
        if (is_eos) break;
        if (position >= Config.max_context_length) break;

        // Decode the token to text and stream it.
        const token_arr = [_]u32{next_token};
        const nbytes = tokenizer.decode(
            &token_arr,
            &decode_buf,
        ) catch |err| {
            std.debug.print(
                "\ndecode error for token {d}: {}\n",
                .{ next_token, err },
            );
            break;
        };
        writeStdout(decode_buf[0..nbytes]);

        generated += 1;

        // Forward pass for the next position.
        dispatchOneDecode(
            &device,
            &pipelines,
            &model,
            next_token,
            position,
        );
        position += 1;

        // Sample the next token.
        logits_slice = model.logits.asSlice();
        next_token = transformer.sampleToken(
            logits_slice,
            cli.sampling,
            scratch,
            indices,
            rng,
        );
    }
    const t_decode_end = std.time.nanoTimestamp();

    // ── Timing summary ───────────────────────────────
    const prefill_ns: u64 =
        @intCast(t_prefill_end - t_prefill_start);
    const decode_ns: u64 =
        @intCast(t_decode_end - t_decode_start);
    const prefill_ms = nanosToMs(prefill_ns);
    const decode_ms = nanosToMs(decode_ns);

    const prefill_tps: f64 = if (prefill_ms > 0.0)
        @as(f64, @floatFromInt(prompt_len)) /
            (prefill_ms / 1000.0)
    else
        0.0;

    const decode_tps: f64 = if (decode_ms > 0.0)
        @as(f64, @floatFromInt(generated)) /
            (decode_ms / 1000.0)
    else
        0.0;

    std.debug.print(
        "\n\n" ++
            "--- timing " ++
            "-----------------------------------\n" ++
            "prefill: {d} tokens in {d:.1} ms " ++
            "({d:.1} tok/s)\n" ++
            "decode:  {d} tokens in {d:.1} ms " ++
            "({d:.1} tok/s)\n",
        .{
            prompt_len,
            prefill_ms,
            prefill_tps,
            generated,
            decode_ms,
            decode_tps,
        },
    );
}

// ============================================================
// Stdout helper
// ============================================================

/// Write raw bytes directly to stdout.  Uses POSIX write
/// to avoid buffered-writer ceremony.  Generated text goes
/// to stdout so it can be piped; all diagnostic output uses
/// std.debug.print (stderr).
fn writeStdout(bytes: []const u8) void {
    if (bytes.len == 0) return;
    _ = std.posix.write(
        std.posix.STDOUT_FILENO,
        bytes,
    ) catch {};
}

// ============================================================
// GPU dispatch helper
// ============================================================

/// Run a single-token forward decode pass (embedding →
/// N blocks → final norm → LM head) and wait for the
/// GPU to finish.  After this returns, `model.logits`
/// contains valid [vocab_size] f32 values.
fn dispatchOneDecode(
    device: *const nn.Device,
    pipelines: *const nn.TransformerPipelines,
    model: *const BonsaiModel,
    token_id: u32,
    position: u32,
) void {
    std.debug.assert(token_id < Config.vocab_size);
    std.debug.assert(
        position < Config.max_context_length,
    );

    const args = model.forwardDecodeArgs(
        token_id,
        position,
    );

    const cmd = device.beginCommandBufferUnretained();
    const enc = device.beginCompute(cmd);
    transformer.forwardDecode(
        Config,
        device,
        enc,
        pipelines,
        args,
    );
    // End encoding before commit (Rule 26).
    enc.msgSend(void, "endEncoding", .{});
    device.commitAndWait(cmd);
}

// ============================================================
// CLI argument parsing
// ============================================================

/// Parse command-line arguments into a `CliArgs` struct.
/// Exits with usage message on invalid input.
fn parseArgs() CliArgs {
    var iter = std.process.args();
    _ = iter.next(); // Skip argv[0] (program name).

    var result = CliArgs{
        .model_dir = ".",
        .prompt = "Hello!",
        .max_tokens = DEFAULT_MAX_GENERATE,
        .sampling = .{
            .temperature = 0.0,
            .top_k = 0,
            .top_p = 1.0,
            .seed = 42,
        },
    };

    // First positional argument: model directory.
    const first = iter.next() orelse
        printUsageAndExit();
    if (first.len > 0 and first[0] == '-') {
        // No positional arg — treat as flag below.
        std.debug.print(
            "error: model directory is required\n",
            .{},
        );
        printUsageAndExit();
    }
    result.model_dir = first;

    // Named options.
    while (iter.next()) |arg| {
        if (eql(arg, "--prompt")) {
            result.prompt = iter.next() orelse
                printUsageAndExit();
        } else if (eql(arg, "--max-tokens")) {
            result.max_tokens = parseU32(
                iter.next() orelse printUsageAndExit(),
            );
        } else if (eql(arg, "--temperature")) {
            result.sampling.temperature = parseF32(
                iter.next() orelse printUsageAndExit(),
            );
        } else if (eql(arg, "--top-k")) {
            result.sampling.top_k = parseU32(
                iter.next() orelse printUsageAndExit(),
            );
        } else if (eql(arg, "--top-p")) {
            result.sampling.top_p = parseF32(
                iter.next() orelse printUsageAndExit(),
            );
        } else if (eql(arg, "--seed")) {
            result.sampling.seed = parseU64(
                iter.next() orelse printUsageAndExit(),
            );
        } else {
            std.debug.print(
                "error: unknown option: {s}\n",
                .{arg},
            );
            printUsageAndExit();
        }
    }

    return result;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn parseU32(s: []const u8) u32 {
    return std.fmt.parseInt(u32, s, 10) catch {
        std.debug.print(
            "error: invalid u32: {s}\n",
            .{s},
        );
        printUsageAndExit();
    };
}

fn parseU64(s: []const u8) u64 {
    return std.fmt.parseInt(u64, s, 10) catch {
        std.debug.print(
            "error: invalid u64: {s}\n",
            .{s},
        );
        printUsageAndExit();
    };
}

fn parseF32(s: []const u8) f32 {
    return std.fmt.parseFloat(f32, s) catch {
        std.debug.print(
            "error: invalid f32: {s}\n",
            .{s},
        );
        printUsageAndExit();
    };
}

fn printUsageAndExit() noreturn {
    std.debug.print("{s}", .{USAGE});
    std.process.exit(1);
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
        if (!std.mem.endsWith(u8, name, ".safetensors")) {
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
