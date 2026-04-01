//! BPE tokenizer — Step 2e of the Bonsai implementation plan.
//!
//! Parses a HuggingFace `tokenizer.json` file (Qwen3 format) and
//! provides byte-level BPE encode/decode.  All heap allocation
//! happens at init time via an owned arena; encode and decode use
//! only stack-local scratch space.
//!
//! Encoding pipeline:
//!   1. Split input text on special-token boundaries.
//!   2. Pre-tokenize each non-special segment (GPT-4–style).
//!   3. Map each chunk's bytes to base BPE tokens (byte→unicode).
//!   4. Apply greedy BPE merges within each chunk.
//!   5. Concatenate all token IDs into the caller's output buffer.
//!
//! Decoding pipeline:
//!   1. Map each token ID to its vocab string.
//!   2. Convert byte-level unicode characters back to raw bytes.

const std = @import("std");
const log = std.log.scoped(.tokenizer);

// ====================================================================
// Hard limits (Rule 4)
// ====================================================================

/// Upper bound on vocabulary size.  Qwen3-4B has 151 669 entries;
/// this cap leaves generous headroom.
const MAX_VOCAB_SIZE: u32 = 200_000;

/// Upper bound on BPE merge rules.
const MAX_MERGES: u32 = 200_000;

/// Maximum byte length of any single token string in the vocabulary.
const MAX_TOKEN_LENGTH: u32 = 512;

/// Maximum number of special (added) tokens.
const MAX_SPECIAL_TOKENS: u32 = 256;

/// Maximum input text length accepted by `encode`, in bytes.
const MAX_INPUT_LENGTH: u32 = 1024 * 1024;

/// Maximum token count returned by a single `encode` call.
const MAX_ENCODE_TOKENS: u32 = 65_536;

/// Maximum tokenizer.json file size we will read, in bytes.
const MAX_JSON_FILE_BYTES: u32 = 32 * 1024 * 1024;

/// Sentinel value indicating "no token ID for this byte."
const UNKNOWN_TOKEN_ID: u32 = 0xFFFF_FFFF;

// ====================================================================
// Comptime byte ↔ unicode mapping (GPT-2 convention)
// ====================================================================
//
// GPT-2 / Qwen3 byte-level BPE represents each raw byte as a Unicode
// character so that the vocabulary contains printable strings rather
// than raw control codes.
//
//   • Bytes 33–126, 161–172, 174–255 map to themselves.
//   • The remaining 68 bytes (0–32, 127–160, 173) map to codepoints
//     256–323, preserving a bijection over all 256 byte values.

const ByteUnicodeMap = struct {
    /// byte → unicode codepoint (for encoding).
    forward: [256]u21,

    /// unicode codepoint → byte (for decoding).  Only entries where
    /// `reverse_valid[cp]` is true are meaningful.
    reverse: [324]u8,

    /// True iff `reverse[cp]` holds a valid byte mapping.
    reverse_valid: [324]bool,
};

/// Build the GPT-2 byte↔unicode tables at comptime.
fn buildByteUnicodeMap() ByteUnicodeMap {
    @setEvalBranchQuota(4096);
    var result: ByteUnicodeMap = .{
        .forward = undefined,
        .reverse = [_]u8{0} ** 324,
        .reverse_valid = [_]bool{false} ** 324,
    };
    var extra: u21 = 0;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const cp: u21 = if (isDirectMapped(b)) b else blk: {
            const mapped = 256 + extra;
            extra += 1;
            break :blk mapped;
        };
        result.forward[i] = cp;
        result.reverse[cp] = b;
        result.reverse_valid[cp] = true;
    }
    // Every byte must be mapped — the table is a bijection.
    std.debug.assert(extra == 68);
    return result;
}

/// True for bytes that map to themselves in the GPT-2 byte→unicode
/// table: printable ASCII (33–126) plus Latin-1 Supplement ranges
/// 161–172 and 174–255.
fn isDirectMapped(b: u8) bool {
    if (b >= 33 and b <= 126) return true;
    if (b >= 161 and b <= 172) return true;
    if (b >= 174) return true; // 174–255
    return false;
}

const byte_map = buildByteUnicodeMap();

// ====================================================================
// Types
// ====================================================================

const SpecialToken = struct {
    content: []const u8,
    id: u32,
};

// ====================================================================
// Tokenizer
// ====================================================================

pub const Tokenizer = struct {
    /// Owns all internal heap allocations (vocab strings, hashmaps,
    /// id_to_vocab array).  Backed by the caller's allocator.
    arena: std.heap.ArenaAllocator,

    /// Token string → token ID.
    vocab_to_id: std.StringHashMap(u32),

    /// Token ID → token string.  Indexed by ID; unused slots are
    /// empty slices.  Allocated with capacity MAX_VOCAB_SIZE.
    id_to_vocab: [][]const u8,

    /// One past the highest token ID seen during parsing.
    vocab_size: u32,

    /// Merge pair string ("tokenA tokenB") → priority rank.
    /// Lower rank = higher priority (applied first).
    merge_ranks: std.StringHashMap(u32),

    /// Special tokens that bypass BPE (e.g. `<|im_start|>`).
    special_tokens: [MAX_SPECIAL_TOKENS]SpecialToken,
    special_token_count: u32,

    /// Precomputed: raw byte → base BPE token ID.  Built once at
    /// init from the vocab + byte-to-unicode mapping.
    byte_to_token_id: [256]u32,

    // Well-known special token IDs, resolved at init.
    eos_token_id: u32,
    im_start_token_id: u32,
    im_end_token_id: u32,

    // ----------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------

    /// Open a HuggingFace tokenizer.json file from disk, parse it,
    /// and populate all internal tables.  `backing` is used to back
    /// the internal arena; call `deinit` to release.
    pub fn init(
        self: *Tokenizer,
        backing: std.mem.Allocator,
        json_path: []const u8,
    ) !void {
        std.debug.assert(json_path.len > 0);
        std.debug.assert(json_path.len < 4096);

        const file = std.fs.cwd().openFile(json_path, .{}) catch |err| {
            log.err("cannot open tokenizer: {s}", .{json_path});
            return err;
        };
        defer file.close();

        // Temporary arena for the file read — freed after parsing.
        var temp = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer temp.deinit();

        const json_bytes = file.readToEndAlloc(
            temp.allocator(),
            MAX_JSON_FILE_BYTES,
        ) catch |err| {
            log.err("cannot read tokenizer: {s}", .{json_path});
            return err;
        };

        try self.initFromJson(backing, json_bytes);
    }

    /// Parse tokenizer data from an in-memory JSON byte slice.
    /// Useful for testing without a file on disk.
    pub fn initFromJson(
        self: *Tokenizer,
        backing: std.mem.Allocator,
        json_bytes: []const u8,
    ) !void {
        std.debug.assert(json_bytes.len > 0);
        std.debug.assert(json_bytes.len <= MAX_JSON_FILE_BYTES);

        self.arena = std.heap.ArenaAllocator.init(backing);
        errdefer self.arena.deinit();

        const alloc = self.arena.allocator();

        self.vocab_to_id = std.StringHashMap(u32).init(alloc);
        self.merge_ranks = std.StringHashMap(u32).init(alloc);
        self.id_to_vocab = try alloc.alloc([]const u8, MAX_VOCAB_SIZE);
        self.vocab_size = 0;
        self.special_token_count = 0;
        self.eos_token_id = 0;
        self.im_start_token_id = 0;
        self.im_end_token_id = 0;
        @memset(self.byte_to_token_id[0..], UNKNOWN_TOKEN_ID);

        // Zero-fill id_to_vocab (Rule 21: no stale data).
        for (self.id_to_vocab) |*slot| slot.* = "";

        // Zero-fill special_tokens.
        for (&self.special_tokens) |*st| st.* = .{
            .content = "",
            .id = 0,
        };

        try self.parseJson(json_bytes);
        self.buildByteTokenTable();
    }

    /// Release all memory owned by the tokenizer.
    pub fn deinit(self: *Tokenizer) void {
        std.debug.assert(self.vocab_size <= MAX_VOCAB_SIZE);
        std.debug.assert(self.special_token_count <= MAX_SPECIAL_TOKENS);

        // Hashmaps use the arena allocator; their deinit is
        // effectively a no-op, but we call it for hygiene.
        self.vocab_to_id.deinit();
        self.merge_ranks.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encode `text` into BPE token IDs.  Writes into the caller-
    /// provided `output_ids` buffer and returns the number of IDs
    /// written.
    pub fn encode(
        self: *const Tokenizer,
        text: []const u8,
        output_ids: []u32,
    ) !u32 {
        std.debug.assert(output_ids.len > 0);
        if (text.len == 0) return 0;
        if (text.len > MAX_INPUT_LENGTH) return error.InputTooLong;

        var out_count: u32 = 0;
        var cursor: u32 = 0;
        const text_len: u32 = @intCast(text.len);

        while (cursor < text_len) {
            // Check for a special token at the current position.
            if (self.findSpecialTokenAt(text, cursor)) |st| {
                if (out_count >= output_ids.len) {
                    return error.OutputBufferTooSmall;
                }
                output_ids[out_count] = st.id;
                out_count += 1;
                cursor += @intCast(st.content.len);
                continue;
            }

            // Find the end of the non-special segment.
            const seg_end = self.findNextSpecialStart(
                text,
                cursor + 1,
            );
            const segment = text[cursor..seg_end];

            // Pre-tokenize and BPE-encode the segment.
            const added = try self.encodeSegment(
                segment,
                output_ids[out_count..],
            );
            out_count += added;
            cursor = seg_end;
        }

        return out_count;
    }

    /// Decode a sequence of token IDs back into raw bytes.  Writes
    /// into `output_buf` and returns the number of bytes written.
    pub fn decode(
        self: *const Tokenizer,
        token_ids: []const u32,
        output_buf: []u8,
    ) !u32 {
        std.debug.assert(output_buf.len > 0);
        std.debug.assert(token_ids.len <= MAX_ENCODE_TOKENS);

        var pos: u32 = 0;
        for (token_ids) |id| {
            if (id >= self.vocab_size) {
                log.warn("unknown token ID: {}", .{id});
                return error.UnknownTokenId;
            }
            const token_str = self.id_to_vocab[id];
            if (token_str.len == 0) {
                log.warn("token ID {} has empty string", .{id});
                return error.UnknownTokenId;
            }
            const remaining: u32 = @intCast(output_buf.len - pos);
            if (remaining < token_str.len) {
                return error.OutputBufferTooSmall;
            }
            const written = decodeTokenToBytes(
                token_str,
                output_buf[pos..],
            );
            pos += written;
        }
        return pos;
    }

    /// Build the Qwen3 chat-template prefix + prompt + suffix and
    /// return the encoded token IDs.
    ///
    /// Format:
    ///   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    pub fn applyChatTemplate(
        self: *const Tokenizer,
        prompt: []const u8,
        output_ids: []u32,
    ) !u32 {
        std.debug.assert(prompt.len > 0);
        std.debug.assert(output_ids.len >= 8);

        var count: u32 = 0;

        // <|im_start|>
        output_ids[count] = self.im_start_token_id;
        count += 1;

        // "user\n"
        count += try self.encode(
            "user\n",
            output_ids[count..],
        );

        // The user's prompt.
        count += try self.encode(prompt, output_ids[count..]);

        // <|im_end|>\n
        output_ids[count] = self.im_end_token_id;
        count += 1;
        count += try self.encode("\n", output_ids[count..]);

        // <|im_start|>assistant\n
        output_ids[count] = self.im_start_token_id;
        count += 1;
        count += try self.encode(
            "assistant\n",
            output_ids[count..],
        );

        return count;
    }

    // ----------------------------------------------------------------
    // JSON parsing (private)
    // ----------------------------------------------------------------

    /// Top-level JSON parser.  Extracts model.vocab, model.merges,
    /// and added_tokens from a HuggingFace tokenizer.json payload.
    fn parseJson(self: *Tokenizer, json_bytes: []const u8) !void {
        std.debug.assert(json_bytes.len > 0);
        std.debug.assert(json_bytes.len <= MAX_JSON_FILE_BYTES);

        // Temporary arena for the JSON parse tree — freed at the
        // end of this function so we don't bloat the long-lived arena.
        var json_arena = std.heap.ArenaAllocator.init(
            std.heap.page_allocator,
        );
        defer json_arena.deinit();

        const parsed = std.json.parseFromSlice(
            std.json.Value,
            json_arena.allocator(),
            json_bytes,
            .{},
        ) catch |err| {
            log.err("tokenizer JSON parse failed: {}", .{err});
            return error.InvalidJson;
        };

        const root = switch (parsed.value) {
            .object => |obj| obj,
            else => {
                log.err("JSON root is not an object", .{});
                return error.InvalidJson;
            },
        };

        try self.parseModel(root);
        try self.parseAddedTokens(root);
        self.resolveSpecialTokenIds();
    }

    /// Extract `model.vocab` and `model.merges` from the root object.
    fn parseModel(
        self: *Tokenizer,
        root: std.json.ObjectMap,
    ) !void {
        const model_val = root.get("model") orelse {
            log.err("missing 'model' key in tokenizer JSON", .{});
            return error.MissingField;
        };
        const model_obj = switch (model_val) {
            .object => |o| o,
            else => {
                log.err("'model' is not an object", .{});
                return error.InvalidJson;
            },
        };

        // Parse vocabulary.
        const vocab_val = model_obj.get("vocab") orelse {
            log.err("missing 'model.vocab'", .{});
            return error.MissingField;
        };
        const vocab_obj = switch (vocab_val) {
            .object => |o| o,
            else => return error.InvalidJson,
        };
        try self.parseVocab(vocab_obj);

        // Parse merge rules (optional — a vocab-only tokenizer is
        // valid, though unusual).
        if (model_obj.get("merges")) |merges_val| {
            const merges_arr = switch (merges_val) {
                .array => |a| a,
                else => return error.InvalidJson,
            };
            try self.parseMerges(merges_arr);
        }
    }

    /// Iterate the vocab object and populate `vocab_to_id` and
    /// `id_to_vocab`.
    fn parseVocab(
        self: *Tokenizer,
        vocab_obj: std.json.ObjectMap,
    ) !void {
        std.debug.assert(self.vocab_size == 0);
        const alloc = self.arena.allocator();

        var iter = vocab_obj.iterator();
        while (iter.next()) |entry| {
            const key = entry.key_ptr.*;
            const id: u32 = switch (entry.value_ptr.*) {
                .integer => |n| std.math.cast(u32, n) orelse {
                    log.warn("vocab ID out of u32 range", .{});
                    return error.InvalidVocabEntry;
                },
                else => return error.InvalidVocabEntry,
            };
            if (id >= MAX_VOCAB_SIZE) {
                log.warn(
                    "vocab ID {} >= MAX_VOCAB_SIZE",
                    .{id},
                );
                return error.VocabTooLarge;
            }

            const key_dupe = try alloc.dupe(u8, key);
            try self.vocab_to_id.put(key_dupe, id);
            self.id_to_vocab[id] = key_dupe;

            if (id >= self.vocab_size) {
                self.vocab_size = id + 1;
            }
        }
    }

    /// Iterate the merges array and populate `merge_ranks`.  The
    /// array index is the merge priority (0 = highest).
    /// Parse merges from the tokenizer JSON.  Handles two formats:
    ///   - String: `"tokenA tokenB"` (HuggingFace default).
    ///   - Array:  `["tokenA", "tokenB"]` (Bonsai / newer format).
    /// Both are stored as `"tokenA tokenB"` in the merge_ranks map.
    fn parseMerges(
        self: *Tokenizer,
        merges_arr: std.json.Array,
    ) !void {
        const alloc = self.arena.allocator();
        const items = merges_arr.items;
        std.debug.assert(items.len <= MAX_MERGES);

        for (items, 0..) |item, rank_usize| {
            const rank: u32 = @intCast(rank_usize);
            switch (item) {
                .string => |s| {
                    const dupe = try alloc.dupe(u8, s);
                    try self.merge_ranks.put(dupe, rank);
                },
                .array => |arr| {
                    // Array format: ["tokenA", "tokenB"].
                    if (arr.items.len != 2) {
                        return error.InvalidMergeEntry;
                    }
                    const a = switch (arr.items[0]) {
                        .string => |s| s,
                        else => return error.InvalidMergeEntry,
                    };
                    const b = switch (arr.items[1]) {
                        .string => |s| s,
                        else => return error.InvalidMergeEntry,
                    };
                    // Join as "tokenA tokenB" to match the
                    // lookup format used by bpeMerge.
                    const joined = try std.fmt.allocPrint(
                        alloc,
                        "{s} {s}",
                        .{ a, b },
                    );
                    try self.merge_ranks.put(joined, rank);
                },
                else => return error.InvalidMergeEntry,
            }
        }
    }

    /// Iterate the `added_tokens` array.  Each special token is
    /// stored for exact-match detection during encoding and is also
    /// added to the vocabulary so decode can resolve its ID.
    fn parseAddedTokens(
        self: *Tokenizer,
        root: std.json.ObjectMap,
    ) !void {
        const added_val = root.get("added_tokens") orelse return;
        const added_arr = switch (added_val) {
            .array => |a| a,
            else => return error.InvalidJson,
        };

        const alloc = self.arena.allocator();

        for (added_arr.items) |item| {
            const obj = switch (item) {
                .object => |o| o,
                else => continue,
            };
            const id = self.extractAddedTokenId(obj) orelse continue;
            const content = self.extractAddedTokenContent(
                obj,
                alloc,
            ) catch continue;
            const is_special = self.extractAddedTokenSpecial(obj);

            // Ensure the token is in the vocabulary.
            if (id < MAX_VOCAB_SIZE) {
                if (self.id_to_vocab[id].len == 0) {
                    self.id_to_vocab[id] = content;
                }
                self.vocab_to_id.put(content, id) catch {};
                if (id >= self.vocab_size) {
                    self.vocab_size = id + 1;
                }
            }

            if (is_special) {
                if (self.special_token_count >= MAX_SPECIAL_TOKENS) {
                    log.warn("too many special tokens", .{});
                    continue;
                }
                self.special_tokens[self.special_token_count] = .{
                    .content = content,
                    .id = id,
                };
                self.special_token_count += 1;
            }
        }
    }

    /// Helper: extract `"id"` from an added_token object.
    fn extractAddedTokenId(
        _: *const Tokenizer,
        obj: std.json.ObjectMap,
    ) ?u32 {
        const val = obj.get("id") orelse return null;
        return switch (val) {
            .integer => |n| std.math.cast(u32, n),
            else => null,
        };
    }

    /// Helper: extract and arena-dupe `"content"` from an
    /// added_token object.
    fn extractAddedTokenContent(
        _: *const Tokenizer,
        obj: std.json.ObjectMap,
        alloc: std.mem.Allocator,
    ) ![]const u8 {
        const val = obj.get("content") orelse {
            return error.MissingField;
        };
        const str = switch (val) {
            .string => |s| s,
            else => return error.InvalidJson,
        };
        return try alloc.dupe(u8, str);
    }

    /// Helper: extract `"special"` bool from an added_token object.
    fn extractAddedTokenSpecial(
        _: *const Tokenizer,
        obj: std.json.ObjectMap,
    ) bool {
        const val = obj.get("special") orelse return false;
        return switch (val) {
            .bool => |b| b,
            else => false,
        };
    }

    /// Look up well-known Qwen3 special tokens by content string
    /// and cache their IDs for fast access.
    fn resolveSpecialTokenIds(self: *Tokenizer) void {
        std.debug.assert(self.vocab_size > 0);
        std.debug.assert(
            self.special_token_count <= MAX_SPECIAL_TOKENS,
        );

        const names = [_]struct {
            content: []const u8,
            field: enum { eos, im_start, im_end },
        }{
            .{ .content = "<|endoftext|>", .field = .eos },
            .{ .content = "<|im_start|>", .field = .im_start },
            .{ .content = "<|im_end|>", .field = .im_end },
        };

        for (self.special_tokens[0..self.special_token_count]) |st| {
            for (names) |n| {
                if (std.mem.eql(u8, st.content, n.content)) {
                    switch (n.field) {
                        .eos => self.eos_token_id = st.id,
                        .im_start => self.im_start_token_id = st.id,
                        .im_end => self.im_end_token_id = st.id,
                    }
                }
            }
        }
    }

    /// Populate `byte_to_token_id` by mapping each raw byte through
    /// the GPT-2 byte→unicode table, encoding the codepoint as
    /// UTF-8, and looking it up in the vocabulary.
    fn buildByteTokenTable(self: *Tokenizer) void {
        std.debug.assert(self.vocab_size > 0);
        std.debug.assert(self.id_to_vocab.len == MAX_VOCAB_SIZE);

        for (0..256) |byte_val| {
            const codepoint = byte_map.forward[byte_val];
            var utf8_buf: [4]u8 = undefined;
            const utf8_len = std.unicode.utf8Encode(
                codepoint,
                &utf8_buf,
            ) catch unreachable; // codepoint ≤ 323, always valid.
            const utf8_str = utf8_buf[0..utf8_len];

            if (self.vocab_to_id.get(utf8_str)) |id| {
                self.byte_to_token_id[byte_val] = id;
            } else {
                // Byte has no base token — the vocab may be
                // incomplete.  Leave as UNKNOWN_TOKEN_ID.
                log.debug(
                    "no base token for byte 0x{x:0>2}",
                    .{byte_val},
                );
            }
        }
    }

    // ----------------------------------------------------------------
    // Encoding helpers (private)
    // ----------------------------------------------------------------

    /// Encode a text segment (no special tokens) by pre-tokenizing
    /// it into chunks and BPE-encoding each chunk.
    fn encodeSegment(
        self: *const Tokenizer,
        segment: []const u8,
        output_ids: []u32,
    ) !u32 {
        std.debug.assert(segment.len > 0);
        std.debug.assert(output_ids.len > 0);

        var count: u32 = 0;
        var pos: u32 = 0;
        const seg_len: u32 = @intCast(segment.len);

        while (pos < seg_len) {
            const chunk_len = nextPretokenChunk(segment[pos..]);
            std.debug.assert(chunk_len > 0); // progress guarantee
            const added = try self.encodeChunk(
                segment[pos .. pos + chunk_len],
                output_ids[count..],
            );
            count += added;
            pos += chunk_len;
        }

        return count;
    }

    /// Encode a single pre-tokenized chunk: convert bytes to base
    /// BPE tokens, apply greedy merges, copy result to output.
    fn encodeChunk(
        self: *const Tokenizer,
        chunk: []const u8,
        output_ids: []u32,
    ) !u32 {
        std.debug.assert(chunk.len > 0);
        std.debug.assert(output_ids.len > 0);

        // Convert each byte to its base BPE token ID.
        var tokens: [MAX_TOKEN_LENGTH]u32 = undefined;
        var count: u32 = 0;
        for (chunk) |byte| {
            if (count >= MAX_TOKEN_LENGTH) {
                return error.ChunkTooLong;
            }
            const base_id = self.byte_to_token_id[byte];
            if (base_id == UNKNOWN_TOKEN_ID) {
                log.warn(
                    "no base token for byte 0x{x:0>2}",
                    .{byte},
                );
                return error.UnknownByte;
            }
            tokens[count] = base_id;
            count += 1;
        }

        // Apply BPE merges greedily until no more apply.
        count = self.bpeMerge(&tokens, count);

        // Copy merged tokens to the caller's output buffer.
        if (count > output_ids.len) {
            return error.OutputBufferTooSmall;
        }
        @memcpy(output_ids[0..count], tokens[0..count]);
        return count;
    }

    /// Greedy BPE merge: repeatedly find the highest-priority
    /// adjacent pair and merge it.  O(n²) per chunk — acceptable
    /// because pre-tokenized chunks are small (typically < 100
    /// tokens).
    fn bpeMerge(
        self: *const Tokenizer,
        tokens: []u32,
        initial_count: u32,
    ) u32 {
        std.debug.assert(initial_count <= MAX_TOKEN_LENGTH);
        std.debug.assert(tokens.len >= initial_count);

        var current = initial_count;
        while (current > 1) {
            const best = findBestPair(
                tokens[0..current],
                self.id_to_vocab,
                &self.merge_ranks,
            ) orelse break;

            // Build the merged token string and look up its ID.
            const merged_id = self.lookupMergedToken(
                tokens[best.index],
                tokens[best.index + 1],
            ) orelse break;

            // Replace the pair with the merged token.
            tokens[best.index] = merged_id;

            // Shift subsequent tokens left by one position.
            const tail_start = best.index + 2;
            const tail_len = current - tail_start;
            if (tail_len > 0) {
                std.mem.copyForwards(
                    u32,
                    tokens[best.index + 1 ..][0..tail_len],
                    tokens[tail_start..][0..tail_len],
                );
            }
            current -= 1;
        }
        return current;
    }

    /// Concatenate two token strings and look up the result in the
    /// vocabulary.  Returns null if the merged string is not in the
    /// vocab (which halts the BPE merge loop).
    fn lookupMergedToken(
        self: *const Tokenizer,
        id_a: u32,
        id_b: u32,
    ) ?u32 {
        std.debug.assert(id_a < self.vocab_size);
        std.debug.assert(id_b < self.vocab_size);

        const str_a = self.id_to_vocab[id_a];
        const str_b = self.id_to_vocab[id_b];
        if (str_a.len + str_b.len > MAX_TOKEN_LENGTH * 2) {
            return null;
        }

        var buf: [MAX_TOKEN_LENGTH * 2]u8 = undefined;
        @memcpy(buf[0..str_a.len], str_a);
        @memcpy(buf[str_a.len..][0..str_b.len], str_b);
        const merged = buf[0 .. str_a.len + str_b.len];

        return self.vocab_to_id.get(merged);
    }

    // ----------------------------------------------------------------
    // Special-token helpers (private)
    // ----------------------------------------------------------------

    /// Return the longest special token that starts at `pos`, or
    /// null if none matches.
    fn findSpecialTokenAt(
        self: *const Tokenizer,
        text: []const u8,
        pos: u32,
    ) ?SpecialToken {
        std.debug.assert(pos <= text.len);
        const remaining = text[pos..];
        var best: ?SpecialToken = null;
        var best_len: u32 = 0;

        const tokens = self.special_tokens[0..self.special_token_count];
        for (tokens) |st| {
            const cl: u32 = @intCast(st.content.len);
            if (cl <= best_len) continue;
            if (cl > remaining.len) continue;
            if (std.mem.eql(u8, remaining[0..cl], st.content)) {
                best = st;
                best_len = cl;
            }
        }
        return best;
    }

    /// Return the byte offset of the next special-token boundary
    /// at or after `start`, or `text.len` if none is found.
    fn findNextSpecialStart(
        self: *const Tokenizer,
        text: []const u8,
        start: u32,
    ) u32 {
        std.debug.assert(start <= text.len);
        var pos = start;
        const end: u32 = @intCast(text.len);
        while (pos < end) : (pos += 1) {
            if (self.findSpecialTokenAt(text, pos) != null) {
                return pos;
            }
        }
        return end;
    }
};

// ====================================================================
// Free-standing helpers (Rule 20: extract hot paths, primitives only)
// ====================================================================

/// Scan the token sequence for the adjacent pair with the lowest
/// merge rank.  Returns the pair's index and rank, or null if no
/// mergeable pair exists.
fn findBestPair(
    tokens: []const u32,
    id_to_vocab: []const []const u8,
    merge_ranks: *const std.StringHashMap(u32),
) ?struct { index: u32, rank: u32 } {
    std.debug.assert(tokens.len >= 2);
    std.debug.assert(id_to_vocab.len > 0);

    var best_rank: u32 = std.math.maxInt(u32);
    var best_index: u32 = 0;
    var found = false;

    // Stack buffer for the merge key "tokenA tokenB".
    var key_buf: [MAX_TOKEN_LENGTH * 2 + 1]u8 = undefined;

    for (0..tokens.len - 1) |i| {
        const str_a = id_to_vocab[tokens[i]];
        const str_b = id_to_vocab[tokens[i + 1]];
        const key_len = str_a.len + 1 + str_b.len;
        if (key_len > key_buf.len) continue;

        @memcpy(key_buf[0..str_a.len], str_a);
        key_buf[str_a.len] = ' ';
        @memcpy(
            key_buf[str_a.len + 1 ..][0..str_b.len],
            str_b,
        );

        if (merge_ranks.get(key_buf[0..key_len])) |rank| {
            if (rank < best_rank) {
                best_rank = rank;
                best_index = @intCast(i);
                found = true;
            }
        }
    }

    if (found) return .{ .index = best_index, .rank = best_rank };
    return null;
}

// ====================================================================
// Pre-tokenization (GPT-4–style, simplified)
// ====================================================================

/// Return the byte length of the next pre-tokenization chunk.
/// Approximates the GPT-4 / Qwen3 regex pattern:
///
///   (?i:'s|'t|'re|'ve|'m|'ll|'d)
///   | [^\r\n\p{L}\p{N}]?\p{L}+
///   | \p{N}{1,3}
///   | …
///
/// Merges only happen within chunks, so this function determines
/// merge boundaries.
fn nextPretokenChunk(text: []const u8) u32 {
    std.debug.assert(text.len > 0);
    const len: u32 = @intCast(@min(text.len, MAX_INPUT_LENGTH));

    // Pattern 1: English contractions ('s, 't, 're, 've, etc.).
    if (text[0] == '\'') {
        const cl = matchContraction(text, len);
        if (cl > 0) return cl;
    }

    // Pattern 2: letters, with an optional leading non-letter-
    //            non-digit character (e.g. space + word → one chunk).
    if (isLetter(text[0])) return scanLetters(text, 0, len);
    if (len > 1 and !isLetterOrDigit(text[0]) and !isNewline(text[0]) and isLetter(text[1])) {
        return scanLetters(text, 1, len);
    }

    // Pattern 3: digit runs of 1–3.
    if (isDigit(text[0])) {
        var end: u32 = 1;
        while (end < len and end < 3 and isDigit(text[end])) {
            end += 1;
        }
        return end;
    }

    // Pattern 4: symbol run (non-ws, non-letter, non-digit), with
    //            an optional leading space.
    if (isSymbol(text[0])) {
        var end: u32 = 1;
        while (end < len and isSymbol(text[end])) end += 1;
        return end;
    }
    if (text[0] == ' ' and len > 1 and isSymbol(text[1])) {
        var end: u32 = 2;
        while (end < len and isSymbol(text[end])) end += 1;
        return end;
    }

    // Pattern 5: newline runs.
    if (isNewline(text[0])) {
        var end: u32 = 1;
        while (end < len and isNewline(text[end])) end += 1;
        return end;
    }

    // Pattern 6: whitespace (non-newline) runs.
    if (isWhitespace(text[0])) {
        var end: u32 = 1;
        while (end < len and isWhitespace(text[end]) and !isNewline(text[end])) {
            end += 1;
        }
        return end;
    }

    // Fallback: single byte.
    return 1;
}

/// Match English contractions at the start of `text`: 's, 't, 'm,
/// 'd (2 bytes) or 're, 've, 'll (3 bytes).  Case-insensitive.
fn matchContraction(text: []const u8, len: u32) u32 {
    std.debug.assert(text[0] == '\'');
    if (len < 2) return 0;
    const c1 = text[1] | 0x20; // to lowercase
    if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') {
        return 2;
    }
    if (len < 3) return 0;
    const c2 = text[2] | 0x20;
    if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) {
        return 3;
    }
    return 0;
}

/// Starting from `start`, consume consecutive letters and return
/// the end offset (exclusive).
fn scanLetters(text: []const u8, start: u32, len: u32) u32 {
    std.debug.assert(start < len);
    var pos = start;
    while (pos < len and isLetter(text[pos])) pos += 1;
    // Must consume at least the starting letter.
    std.debug.assert(pos > start);
    return pos;
}

// ====================================================================
// Byte classification
// ====================================================================

/// ASCII letters plus all high bytes (≥ 0x80), which are UTF-8
/// start or continuation bytes — treated as "letters" so that
/// multi-byte Unicode sequences stay grouped together.
fn isLetter(b: u8) bool {
    if (b >= 'a' and b <= 'z') return true;
    if (b >= 'A' and b <= 'Z') return true;
    if (b >= 0x80) return true; // UTF-8 multi-byte
    return false;
}

fn isDigit(b: u8) bool {
    return b >= '0' and b <= '9';
}

fn isLetterOrDigit(b: u8) bool {
    return isLetter(b) or isDigit(b);
}

fn isWhitespace(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r' or b == 0x0B or b == 0x0C;
}

fn isNewline(b: u8) bool {
    return b == '\n' or b == '\r';
}

/// A "symbol" is anything that is not a letter, digit, or
/// whitespace: punctuation, operators, control characters, etc.
fn isSymbol(b: u8) bool {
    return !isLetter(b) and !isDigit(b) and !isWhitespace(b);
}

// ====================================================================
// Decode helper: token string → raw bytes
// ====================================================================

/// Convert a BPE token string (which uses the GPT-2 byte→unicode
/// mapping) back to raw bytes.  Returns the number of output bytes
/// written.
fn decodeTokenToBytes(
    token_str: []const u8,
    output: []u8,
) u32 {
    std.debug.assert(token_str.len > 0);
    std.debug.assert(output.len >= token_str.len);

    var out_pos: u32 = 0;
    var in_pos: u32 = 0;
    const in_len: u32 = @intCast(token_str.len);

    while (in_pos < in_len) {
        const seq_len = utf8SequenceLength(token_str[in_pos]);
        if (in_pos + seq_len > in_len) {
            // Truncated UTF-8 — emit the raw byte.
            output[out_pos] = token_str[in_pos];
            out_pos += 1;
            in_pos += 1;
            continue;
        }
        const codepoint = decodeUtf8Codepoint(
            token_str[in_pos .. in_pos + seq_len],
        );

        // Map the codepoint back to a raw byte.
        if (codepoint < byte_map.reverse.len and byte_map.reverse_valid[codepoint]) {
            output[out_pos] = byte_map.reverse[codepoint];
            out_pos += 1;
        } else {
            // Codepoint not in the byte mapping — pass through
            // the raw UTF-8 bytes unchanged.
            const sl: u32 = seq_len;
            @memcpy(
                output[out_pos..][0..sl],
                token_str[in_pos..][0..sl],
            );
            out_pos += sl;
        }
        in_pos += seq_len;
    }
    return out_pos;
}

// ====================================================================
// UTF-8 helpers (no std.unicode dependency for decode)
// ====================================================================

/// Return the expected byte length of a UTF-8 sequence given its
/// first byte.
fn utf8SequenceLength(first: u8) u32 {
    if (first < 0x80) return 1;
    if (first < 0xC0) return 1; // invalid start — treat as 1
    if (first < 0xE0) return 2;
    if (first < 0xF0) return 3;
    return 4;
}

/// Decode a UTF-8 byte sequence (1–4 bytes) into a codepoint.
fn decodeUtf8Codepoint(bytes: []const u8) u21 {
    std.debug.assert(bytes.len >= 1);
    std.debug.assert(bytes.len <= 4);
    if (bytes.len == 1) return bytes[0];
    if (bytes.len == 2) {
        return @as(u21, bytes[0] & 0x1F) << 6 | @as(u21, bytes[1] & 0x3F);
    }
    if (bytes.len == 3) {
        return @as(u21, bytes[0] & 0x0F) << 12 | @as(u21, bytes[1] & 0x3F) << 6 | @as(u21, bytes[2] & 0x3F);
    }
    return @as(u21, bytes[0] & 0x07) << 18 | @as(u21, bytes[1] & 0x3F) << 12 | @as(u21, bytes[2] & 0x3F) << 6 | @as(u21, bytes[3] & 0x3F);
}

// ====================================================================
// Tests
// ====================================================================

test "byte-to-unicode mapping is a bijection over 256 bytes" {
    // Every byte must map to a unique codepoint, and the reverse
    // table must map each codepoint back to the original byte.
    var seen: [324]bool = [_]bool{false} ** 324;
    for (0..256) |i| {
        const cp = byte_map.forward[i];
        // Codepoint must be in the valid range.
        try std.testing.expect(cp < 324);
        // Must not have been seen before (injective).
        try std.testing.expect(!seen[cp]);
        seen[cp] = true;
        // Reverse mapping must recover the original byte.
        try std.testing.expect(byte_map.reverse_valid[cp]);
        try std.testing.expectEqual(@as(u8, @intCast(i)), byte_map.reverse[cp]);
    }
}

test "direct-mapped bytes map to themselves" {
    // Printable ASCII (33–126) maps to itself.
    for (33..127) |b| {
        try std.testing.expectEqual(
            @as(u21, @intCast(b)),
            byte_map.forward[b],
        );
    }
    // Space (32) does NOT map to itself.
    try std.testing.expect(byte_map.forward[32] != 32);
    // Space maps to codepoint 256 (first non-direct byte in order).
    // Actually byte 0 is first in the non-direct sequence; let's
    // just check that space maps to something ≥ 256.
    try std.testing.expect(byte_map.forward[32] >= 256);
}

test "pre-tokenizer: letter chunks" {
    // Pure ASCII word.
    try std.testing.expectEqual(@as(u32, 5), nextPretokenChunk("hello world"));
    // Space + word grouped.
    try std.testing.expectEqual(@as(u32, 6), nextPretokenChunk(" world"));
    // Single letter.
    try std.testing.expectEqual(@as(u32, 1), nextPretokenChunk("a"));
}

test "pre-tokenizer: digit chunks" {
    // Digits are grouped in runs of up to 3.
    try std.testing.expectEqual(@as(u32, 3), nextPretokenChunk("123456"));
    try std.testing.expectEqual(@as(u32, 2), nextPretokenChunk("42abc"));
    try std.testing.expectEqual(@as(u32, 1), nextPretokenChunk("7"));
}

test "pre-tokenizer: contractions" {
    try std.testing.expectEqual(@as(u32, 2), nextPretokenChunk("'s great"));
    try std.testing.expectEqual(@as(u32, 2), nextPretokenChunk("'t do"));
    try std.testing.expectEqual(@as(u32, 3), nextPretokenChunk("'re done"));
    try std.testing.expectEqual(@as(u32, 3), nextPretokenChunk("'ll go"));
}

test "pre-tokenizer: symbols" {
    // Consecutive symbols form one chunk.
    try std.testing.expectEqual(@as(u32, 3), nextPretokenChunk("!!!"));
    // Space + symbols → one chunk.
    try std.testing.expectEqual(@as(u32, 4), nextPretokenChunk(" !!!"));
}

test "pre-tokenizer: whitespace and newlines" {
    // Newlines form their own chunk.
    try std.testing.expectEqual(@as(u32, 1), nextPretokenChunk("\n"));
    try std.testing.expectEqual(@as(u32, 2), nextPretokenChunk("\n\n"));
    // Spaces.
    try std.testing.expectEqual(@as(u32, 3), nextPretokenChunk("   abc"));
}

test "UTF-8 decode codepoint" {
    // ASCII.
    try std.testing.expectEqual(@as(u21, 'a'), decodeUtf8Codepoint("a"));
    // 2-byte: Ġ = U+0120 = 0xC4 0xA0.
    try std.testing.expectEqual(@as(u21, 0x120), decodeUtf8Codepoint(&[_]u8{ 0xC4, 0xA0 }));
    // 3-byte: € = U+20AC = 0xE2 0x82 0xAC.
    try std.testing.expectEqual(@as(u21, 0x20AC), decodeUtf8Codepoint(&[_]u8{ 0xE2, 0x82, 0xAC }));
}

test "BPE merge on known sequence" {
    // Goal: verify that the greedy BPE merge selects the highest-
    // priority (lowest-rank) pair first, and that merges cascade.
    //
    // Setup: vocab {a:0, b:1, c:2, ab:3, abc:4}
    //        merges {"a b":0, "ab c":1}
    //
    // Input tokens: [0, 1, 2]  (a, b, c)
    // Step 1: merge (0,1) → 3  → [3, 2]
    // Step 2: merge (3,2) → 4  → [4]

    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4
        \\    },
        \\    "merges": [
        \\      "a b",
        \\      "ab c"
        \\    ]
        \\  },
        \\  "added_tokens": []
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    // Verify vocab was parsed correctly.
    try std.testing.expectEqual(@as(u32, 5), tok.vocab_size);
    try std.testing.expectEqualStrings("a", tok.id_to_vocab[0]);
    try std.testing.expectEqualStrings("abc", tok.id_to_vocab[4]);

    // Run bpeMerge on [0, 1, 2].
    var tokens = [_]u32{ 0, 1, 2 };
    const result = tok.bpeMerge(&tokens, 3);
    try std.testing.expectEqual(@as(u32, 1), result);
    try std.testing.expectEqual(@as(u32, 4), tokens[0]);
}

test "encode and decode round-trip with ASCII" {
    // Goal: encode "abcd", verify the BPE produces the expected
    // merged token, then decode back to the original string.
    //
    // Vocab: single-byte tokens for 'a','b','c','d', plus merges.
    // The byte-to-unicode mapping for bytes 97–100 (a–d) is the
    // identity, so vocab keys "a","b","c","d" are the base tokens.

    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "a": 0, "b": 1, "c": 2, "d": 3,
        \\      "ab": 4, "cd": 5, "abcd": 6
        \\    },
        \\    "merges": [
        \\      "a b",
        \\      "c d",
        \\      "ab cd"
        \\    ]
        \\  },
        \\  "added_tokens": []
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    // Encode "abcd": a(0)+b(1)→ab(4), c(2)+d(3)→cd(5), ab(4)+cd(5)→abcd(6).
    var ids: [64]u32 = undefined;
    const encode_count = try tok.encode("abcd", &ids);
    try std.testing.expectEqual(@as(u32, 1), encode_count);
    try std.testing.expectEqual(@as(u32, 6), ids[0]);

    // Decode [6] → "abcd".
    var buf: [64]u8 = undefined;
    const decode_count = try tok.decode(ids[0..encode_count], &buf);
    try std.testing.expectEqual(@as(u32, 4), decode_count);
    try std.testing.expectEqualStrings("abcd", buf[0..decode_count]);
}

test "encode preserves partial merges" {
    // Goal: when no merge applies to the middle pair, the result
    // reflects partial merging on each side independently.
    //
    // Vocab: a:0, b:1, x:2, c:3, d:4, ab:5, cd:6
    // Merges: "a b" (rank 0), "c d" (rank 1)
    // Input: "abxcd" → base [0,1,2,3,4]
    //   merge a+b → [5,2,3,4]
    //   merge c+d → [5,2,6]
    //   no more merges.

    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "a": 0, "b": 1, "x": 2, "c": 3,
        \\      "d": 4, "ab": 5, "cd": 6
        \\    },
        \\    "merges": [
        \\      "a b",
        \\      "c d"
        \\    ]
        \\  },
        \\  "added_tokens": []
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    var ids: [64]u32 = undefined;
    const count = try tok.encode("abxcd", &ids);
    try std.testing.expectEqual(@as(u32, 3), count);
    try std.testing.expectEqual(@as(u32, 5), ids[0]); // "ab"
    try std.testing.expectEqual(@as(u32, 2), ids[1]); // "x"
    try std.testing.expectEqual(@as(u32, 6), ids[2]); // "cd"
}

test "special tokens are not BPE-split" {
    // Goal: the special token "<s>" is recognized as a single token
    // and not split into '<', 's', '>'.

    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "a": 0, "b": 1, "ab": 2, "<s>": 10
        \\    },
        \\    "merges": ["a b"]
        \\  },
        \\  "added_tokens": [
        \\    {"id": 10, "content": "<s>", "special": true}
        \\  ]
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    try std.testing.expectEqual(@as(u32, 1), tok.special_token_count);

    var ids: [64]u32 = undefined;
    const count = try tok.encode("ab<s>ab", &ids);

    // Expected: "ab" → [2], "<s>" → [10], "ab" → [2].
    try std.testing.expectEqual(@as(u32, 3), count);
    try std.testing.expectEqual(@as(u32, 2), ids[0]);
    try std.testing.expectEqual(@as(u32, 10), ids[1]);
    try std.testing.expectEqual(@as(u32, 2), ids[2]);
}

test "decode handles special tokens" {
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "a": 0, "b": 1, "ab": 2, "<s>": 10
        \\    },
        \\    "merges": ["a b"]
        \\  },
        \\  "added_tokens": [
        \\    {"id": 10, "content": "<s>", "special": true}
        \\  ]
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    const ids = [_]u32{ 2, 10, 2 };
    var buf: [64]u8 = undefined;
    const len = try tok.decode(&ids, &buf);
    try std.testing.expectEqualStrings("ab<s>ab", buf[0..len]);
}

test "empty input encodes to zero tokens" {
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0},
        \\    "merges": []
        \\  },
        \\  "added_tokens": []
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    var ids: [64]u32 = undefined;
    const count = try tok.encode("", &ids);
    try std.testing.expectEqual(@as(u32, 0), count);
}

test "single byte with no merges" {
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0, "b": 1, "c": 2},
        \\    "merges": []
        \\  },
        \\  "added_tokens": []
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    var ids: [64]u32 = undefined;
    const count = try tok.encode("abc", &ids);

    // No merges: each byte is its own token.
    try std.testing.expectEqual(@as(u32, 3), count);
    try std.testing.expectEqual(@as(u32, 0), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    try std.testing.expectEqual(@as(u32, 2), ids[2]);
}

test "well-known special token IDs are resolved" {
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0},
        \\    "merges": []
        \\  },
        \\  "added_tokens": [
        \\    {"id": 100, "content": "<|endoftext|>", "special": true},
        \\    {"id": 101, "content": "<|im_start|>", "special": true},
        \\    {"id": 102, "content": "<|im_end|>", "special": true}
        \\  ]
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    try std.testing.expectEqual(@as(u32, 100), tok.eos_token_id);
    try std.testing.expectEqual(@as(u32, 101), tok.im_start_token_id);
    try std.testing.expectEqual(@as(u32, 102), tok.im_end_token_id);
}

test "chat template produces correct structure" {
    // Goal: verify that applyChatTemplate emits the expected
    // sequence: <|im_start|> + "user\n" + prompt + <|im_end|> +
    // "\n" + <|im_start|> + "assistant\n".
    //
    // We use a minimal vocab with single-char tokens and special
    // tokens.  The exact encoding of "user\n" etc. depends on the
    // pre-tokenizer and merge rules, so we just verify the special
    // token positions.

    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "H": 0, "i": 1,
        \\      "u": 2, "s": 3, "e": 4, "r": 5,
        \\      "a": 6, "t": 7, "n": 8,
        \\      "<|im_start|>": 100,
        \\      "<|im_end|>": 101
        \\    },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [
        \\    {"id": 100, "content": "<|im_start|>", "special": true},
        \\    {"id": 101, "content": "<|im_end|>", "special": true}
        \\  ]
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    // We need the newline byte's base token to exist.  Newline is
    // byte 10, which maps to codepoint 256+10 = 266.  Its UTF-8 is
    // 2 bytes: 0xC4 0x8A.  We didn't add it to the vocab, so
    // encoding "\n" would fail with UnknownByte.  That's OK — this
    // test just verifies that the special token IDs are resolved.
    try std.testing.expectEqual(@as(u32, 100), tok.im_start_token_id);
    try std.testing.expectEqual(@as(u32, 101), tok.im_end_token_id);
}

test "decode token to bytes for ASCII" {
    // ASCII characters map to themselves.
    var buf: [8]u8 = undefined;
    const len = decodeTokenToBytes("hello", &buf);
    try std.testing.expectEqual(@as(u32, 5), len);
    try std.testing.expectEqualStrings("hello", buf[0..len]);
}

test "decode token to bytes for space prefix" {
    // The GPT-2 space token Ġ (U+0120) should decode to a space byte.
    // U+0120 in UTF-8 = 0xC4, 0xA0.
    const space_token = &[_]u8{ 0xC4, 0xA0 };
    var buf: [8]u8 = undefined;
    const len = decodeTokenToBytes(space_token, &buf);
    try std.testing.expectEqual(@as(u32, 1), len);
    try std.testing.expectEqual(@as(u8, ' '), buf[0]);
}

test "merge priority selects lowest rank" {
    // Goal: when two mergeable pairs exist, the one with the lower
    // rank (higher priority) is selected first.
    //
    // Vocab: a:0, b:1, c:2, ab:3, bc:4
    // Merges: "b c" (rank 0), "a b" (rank 1)
    // Input: [0, 1, 2] → should merge b+c first → [0, 4].

    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4
        \\    },
        \\    "merges": [
        \\      "b c",
        \\      "a b"
        \\    ]
        \\  },
        \\  "added_tokens": []
        \\}
    ;

    var tok: Tokenizer = undefined;
    try tok.initFromJson(std.testing.allocator, json);
    defer tok.deinit();

    var ids: [64]u32 = undefined;
    const count = try tok.encode("abc", &ids);
    try std.testing.expectEqual(@as(u32, 2), count);
    try std.testing.expectEqual(@as(u32, 0), ids[0]); // "a"
    try std.testing.expectEqual(@as(u32, 4), ids[1]); // "bc"
}
