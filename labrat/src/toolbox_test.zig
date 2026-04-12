//! External tests for pure functions in toolbox.zig.
//!
//! Goal: cover extractFunctionName, collectFunctionNames,
//! extractJsonFloat, sanitizeBranchName, truncateAtSentence,
//! countBracesOnLine, and extractLineRange with edge cases
//! not covered by the inline tests.
//!
//! The tested functions are marked `pub` in toolbox.zig so
//! that this external test file can import them.  Zig 0.15
//! does not expose non-pub declarations through @import,
//! even in test builds.

const std = @import("std");
const toolbox = @import("toolbox.zig");
const tools = @import("tools.zig");

// Re-bind pub declarations under short names for brevity.
const extractFunctionName = toolbox.extractFunctionName;
const collectFunctionNames = toolbox.collectFunctionNames;
const extractJsonFloat = toolbox.extractJsonFloat;
const sanitizeBranchName = toolbox.sanitizeBranchName;
const truncateAtSentence = toolbox.truncateAtSentence;
const countBracesOnLine = toolbox.countBracesOnLine;
const extractLineRange = toolbox.extractLineRange;
const FunctionEntry = toolbox.FunctionEntry;
const MAX_OUTLINE_FUNCTIONS = toolbox.MAX_OUTLINE_FUNCTIONS;

// ============================================================
// extractFunctionName
// ============================================================

test "extractFunctionName: pub fn declaration" {
    // Goal: verify that `pub fn init(` is detected and
    // the name "init" is returned.
    const result = extractFunctionName(
        "pub fn init(self: *Self) void {",
    );
    try std.testing.expectEqualStrings(
        "init",
        result.?,
    );
}

test "extractFunctionName: private fn declaration" {
    // Goal: verify that bare `fn helper(` works.
    const result = extractFunctionName(
        "fn helper(x: u32) u32 {",
    );
    try std.testing.expectEqualStrings(
        "helper",
        result.?,
    );
}

test "extractFunctionName: Metal kernel with void return" {
    // Goal: Metal `kernel void name(` pattern.
    const result = extractFunctionName(
        "kernel void matmul(" ++ "device float* a [[buffer(0)]])",
    );
    try std.testing.expectEqualStrings(
        "matmul",
        result.?,
    );
}

test "extractFunctionName: Metal kernel with half return" {
    // Goal: Metal kernel with non-void return type.
    const result = extractFunctionName(
        "kernel half reduce(device half* data)",
    );
    try std.testing.expectEqualStrings(
        "reduce",
        result.?,
    );
}

test "extractFunctionName: non-function line" {
    // Goal: a variable declaration returns null.
    const result = extractFunctionName(
        "const x = 42;",
    );
    try std.testing.expect(result == null);
}

test "extractFunctionName: empty string" {
    // Goal: empty input returns null (len < 4 guard).
    const result = extractFunctionName("");
    try std.testing.expect(result == null);
}

test "extractFunctionName: fn keyword with no paren" {
    // Goal: `"fn "` alone (3 chars, < 4) returns null.
    const result = extractFunctionName("fn ");
    try std.testing.expect(result == null);
}

test "extractFunctionName: fn keyword no paren long" {
    // Goal: `"fn something"` with no `(` returns null.
    const result = extractFunctionName(
        "fn something",
    );
    try std.testing.expect(result == null);
}

test "extractFunctionName: comment with fn — known false positive" {
    // Goal: document that a commented-out function
    // declaration is detected as a function.  The parser
    // does not skip comment prefixes, so this is a known
    // false positive.
    const result = extractFunctionName(
        "// fn commented(x: u32) void {",
    );
    try std.testing.expectEqualStrings(
        "commented",
        result.?,
    );
}

test "extractFunctionName: kernel with no space after return type" {
    // Goal: `kernel void(` has no name (space+1 == len
    // of after, or paren == 0).  The return type is
    // "void" but the paren immediately follows with no
    // name.
    const result = extractFunctionName(
        "kernel void(",
    );
    // after = "void(", space at 4, name_part = "(", paren = 0.
    try std.testing.expect(result == null);
}

test "extractFunctionName: kernel with only return type" {
    // Goal: `"kernel void"` — no second space, no paren.
    const result = extractFunctionName(
        "kernel void",
    );
    // after = "void", space not found → null.
    try std.testing.expect(result == null);
}

// ============================================================
// collectFunctionNames
// ============================================================

test "collectFunctionNames: empty content" {
    // Goal: empty string produces zero entries.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const entries = collectFunctionNames(arena, "");
    try std.testing.expectEqual(
        @as(usize, 0),
        entries.len,
    );
}

test "collectFunctionNames: single function" {
    // Goal: one function line produces one entry with
    // correct name and line number 1.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const content = "fn hello(x: u32) void {}\n";
    const entries = collectFunctionNames(arena, content);

    try std.testing.expectEqual(
        @as(usize, 1),
        entries.len,
    );
    try std.testing.expectEqualStrings(
        "hello",
        entries[0].name,
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        entries[0].line,
    );
}

test "collectFunctionNames: multiple functions" {
    // Goal: multiple function lines produce entries with
    // correct names and ascending line numbers.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const content =
        \\const std = @import("std");
        \\
        \\fn alpha(a: u32) void {}
        \\
        \\pub fn beta(b: bool) u8 {}
        \\
        \\fn gamma() void {}
    ;
    const entries = collectFunctionNames(arena, content);

    try std.testing.expectEqual(
        @as(usize, 3),
        entries.len,
    );
    try std.testing.expectEqualStrings(
        "alpha",
        entries[0].name,
    );
    try std.testing.expectEqual(
        @as(u32, 3),
        entries[0].line,
    );
    try std.testing.expectEqualStrings(
        "beta",
        entries[1].name,
    );
    try std.testing.expectEqual(
        @as(u32, 5),
        entries[1].line,
    );
    try std.testing.expectEqualStrings(
        "gamma",
        entries[2].name,
    );
    try std.testing.expectEqual(
        @as(u32, 7),
        entries[2].line,
    );
}

test "collectFunctionNames: non-function lines skipped" {
    // Goal: lines without function signatures are
    // silently ignored — only real declarations appear.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const content =
        \\const x = 42;
        \\var y: bool = true;
        \\// Just a comment.
        \\return error.Fail;
    ;
    const entries = collectFunctionNames(arena, content);
    try std.testing.expectEqual(
        @as(usize, 0),
        entries.len,
    );
}

test "collectFunctionNames: mixed Zig and Metal" {
    // Goal: both Zig fn and Metal kernel declarations
    // are collected in one pass.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const content =
        "fn zigFunc(a: u32) void {}\n" ++
        "kernel void metalFunc(" ++
        "device float* buf)\n";
    const entries = collectFunctionNames(arena, content);

    try std.testing.expectEqual(
        @as(usize, 2),
        entries.len,
    );
    try std.testing.expectEqualStrings(
        "zigFunc",
        entries[0].name,
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        entries[0].line,
    );
    try std.testing.expectEqualStrings(
        "metalFunc",
        entries[1].name,
    );
    try std.testing.expectEqual(
        @as(u32, 2),
        entries[1].line,
    );
}

// ============================================================
// extractJsonFloat
// ============================================================

test "extractJsonFloat: simple float value" {
    // Goal: basic key-value extraction.
    const result = extractJsonFloat(
        \\{"score": 42.5}
    , "score");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(
        @as(f64, 42.5),
        result.?,
        0.001,
    );
}

test "extractJsonFloat: second key in object" {
    // Goal: the scanner finds the right key when
    // multiple keys are present.
    const result = extractJsonFloat(
        \\{"a": 1.0, "b": 2.0}
    , "b");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(
        @as(f64, 2.0),
        result.?,
        0.001,
    );
}

test "extractJsonFloat: key not found" {
    // Goal: missing key returns null.
    const result = extractJsonFloat(
        \\{"score": 42.5}
    , "missing");
    try std.testing.expect(result == null);
}

test "extractJsonFloat: negative number" {
    // Goal: leading minus sign is parsed correctly.
    const result = extractJsonFloat(
        \\{"val": -3.14}
    , "val");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(
        @as(f64, -3.14),
        result.?,
        0.001,
    );
}

test "extractJsonFloat: integer value" {
    // Goal: whole numbers without a decimal point parse
    // as float.
    const result = extractJsonFloat(
        \\{"count": 100}
    , "count");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(
        @as(f64, 100.0),
        result.?,
        0.001,
    );
}

test "extractJsonFloat: scientific notation" {
    // Goal: exponent form (1.5e2 = 150) is handled.
    const result = extractJsonFloat(
        \\{"x": 1.5e2}
    , "x");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(
        @as(f64, 150.0),
        result.?,
        0.001,
    );
}

test "extractJsonFloat: empty JSON" {
    // Goal: empty string returns null without crashing.
    const result = extractJsonFloat("", "key");
    try std.testing.expect(result == null);
}

test "extractJsonFloat: tight bounds no spaces" {
    // Goal: `{"k":9}` — minimal whitespace, value at
    // the boundary before closing brace.
    const result = extractJsonFloat(
        \\{"k":9}
    , "k");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(
        @as(f64, 9.0),
        result.?,
        0.001,
    );
}

test "extractJsonFloat: key is substring of another key" {
    // Goal: searching for "a" should not match "ab".
    // The parser checks for a closing quote after the
    // key, so "ab" is rejected when looking for "a".
    const result = extractJsonFloat(
        \\{"ab": 1.0, "a": 2.0}
    , "a");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(
        @as(f64, 2.0),
        result.?,
        0.001,
    );
}

// ============================================================
// sanitizeBranchName
// ============================================================

test "sanitizeBranchName: already clean" {
    // Goal: alphanumeric-with-dashes passes through.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = sanitizeBranchName(
        arena,
        "hello-world",
    );
    try std.testing.expectEqualStrings(
        "hello-world",
        result,
    );
}

test "sanitizeBranchName: spaces and punctuation" {
    // Goal: spaces become dashes, trailing punctuation
    // is replaced and then the trailing dash is trimmed.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = sanitizeBranchName(
        arena,
        "Hello World!",
    );
    try std.testing.expectEqualStrings(
        "Hello-World",
        result,
    );
}

test "sanitizeBranchName: consecutive non-alnum collapsed" {
    // Goal: multiple consecutive spaces collapse into a
    // single dash.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = sanitizeBranchName(
        arena,
        "foo  bar",
    );
    try std.testing.expectEqualStrings(
        "foo-bar",
        result,
    );
}

test "sanitizeBranchName: all non-alnum returns unnamed" {
    // Goal: when every character is non-alphanumeric,
    // result after collapse and trim is empty, so
    // "unnamed" is returned.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = sanitizeBranchName(arena, "!!!");
    try std.testing.expectEqualStrings(
        "unnamed",
        result,
    );
}

test "sanitizeBranchName: long string truncated to 60" {
    // Goal: input longer than 60 characters is capped
    // before sanitization.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // 70 'a' characters — should be truncated to 60.
    const long = "a" ** 70;
    const result = sanitizeBranchName(arena, long);
    try std.testing.expectEqual(
        @as(usize, 60),
        result.len,
    );
    // Every character is 'a', so no dashes inserted.
    try std.testing.expectEqualStrings(
        "a" ** 60,
        result,
    );
}

test "sanitizeBranchName: simple alphanumeric" {
    // Goal: pure lowercase passes through unchanged.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = sanitizeBranchName(arena, "simple");
    try std.testing.expectEqualStrings(
        "simple",
        result,
    );
}

test "sanitizeBranchName: leading non-alnum" {
    // Goal: leading special characters become a dash
    // that remains (only trailing dash is trimmed).
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = sanitizeBranchName(
        arena,
        "##hello",
    );
    try std.testing.expectEqualStrings(
        "-hello",
        result,
    );
}

// ============================================================
// truncateAtSentence
// ============================================================

test "truncateAtSentence: short text returned as-is" {
    // Goal: when text fits within max_len, return it
    // unmodified.
    const text = "Hello world.";
    const result = truncateAtSentence(text, 100);
    try std.testing.expectEqualStrings(text, result);
}

test "truncateAtSentence: cuts at sentence boundary" {
    // Goal: truncation occurs at the '.' that ends
    // "Goodbye world." because the next character is
    // a space.
    //
    // "Hello world. Goodbye world. The end."
    //  0         1111111111222222222233333333
    //  0123456789012345678901234567890123456
    //
    // Position 26 is '.', position 27 is ' '.
    // With max_len=28, pos starts at 28, decrements
    // to 26, finds '.', next char is ' ', returns
    // text[0..27].
    const text =
        "Hello world. Goodbye world. The end.";
    const result = truncateAtSentence(text, 28);
    try std.testing.expectEqualStrings(
        "Hello world. Goodbye world.",
        result,
    );
}

test "truncateAtSentence: no sentence enders — hard cut" {
    // Goal: when no sentence-ending punctuation exists,
    // falls back to hard truncation at max_len.
    const text = "abcdefghijklmnopqrstuvwxyz";
    const result = truncateAtSentence(text, 10);
    try std.testing.expectEqualStrings(
        "abcdefghij",
        result,
    );
}

test "truncateAtSentence: ender at exact boundary" {
    // Goal: when a sentence ender is at position
    // max_len - 1 (the last position checked), it is
    // accepted via the `pos + 1 >= max_len` branch.
    //
    // "Stop here. Go on."
    //  01234567890
    // Position 9 is '.', max_len = 10.
    // pos starts at 10, decrements.  pos=9: '.' found,
    // pos + 1 = 10 >= max_len(10), returns text[0..10].
    const text = "Stop here. Go on.";
    const result = truncateAtSentence(text, 10);
    try std.testing.expectEqualStrings(
        "Stop here.",
        result,
    );
}

test "truncateAtSentence: exclamation mark as ender" {
    // Goal: '!' is a valid sentence ender.
    const text = "Wow! That is amazing really.";
    const result = truncateAtSentence(text, 10);
    try std.testing.expectEqualStrings(
        "Wow!",
        result,
    );
}

test "truncateAtSentence: question mark as ender" {
    // Goal: '?' is a valid sentence ender.
    const text = "Why? Because I said so.";
    const result = truncateAtSentence(text, 10);
    try std.testing.expectEqualStrings(
        "Why?",
        result,
    );
}

test "truncateAtSentence: closing paren as ender" {
    // Goal: ')' is a valid sentence ender.
    const text = "See (note) for details here.";
    const result = truncateAtSentence(text, 12);
    try std.testing.expectEqualStrings(
        "See (note)",
        result,
    );
}

test "truncateAtSentence: ender not followed by space" {
    // Goal: a '.' not followed by space or newline is
    // skipped unless it is at the boundary.
    // "a.b.c.d.e.f.g" — all dots followed by letters.
    // max_len=8: text[0..8] = "a.b.c.d."
    // pos=7 is '.', pos+1=8 >= max_len(8) → accepted.
    const text = "a.b.c.d.e.f.g";
    const result = truncateAtSentence(text, 8);
    try std.testing.expectEqualStrings(
        "a.b.c.d.",
        result,
    );
}

// ============================================================
// countBracesOnLine — additional edge cases
// ============================================================

test "countBracesOnLine: empty line" {
    // Goal: empty input does not change depth.
    const depth = countBracesOnLine("", 0);
    try std.testing.expectEqual(@as(i32, 0), depth);
}

test "countBracesOnLine: empty line preserves depth" {
    // Goal: non-zero starting depth is preserved.
    const depth = countBracesOnLine("", 3);
    try std.testing.expectEqual(@as(i32, 3), depth);
}

test "countBracesOnLine: triple open brace" {
    // Goal: three consecutive open braces add 3.
    const depth = countBracesOnLine("{{{", 0);
    try std.testing.expectEqual(@as(i32, 3), depth);
}

test "countBracesOnLine: balanced with content" {
    // Goal: `if (x) { return; }` has one open and one
    // close — net zero change.
    const depth = countBracesOnLine(
        "if (x) { return; }",
        0,
    );
    try std.testing.expectEqual(@as(i32, 0), depth);
}

test "countBracesOnLine: mixed braces and text" {
    // Goal: `"{ a { b } }"` has 2 opens and 2 closes.
    const depth = countBracesOnLine(
        "{ a { b } }",
        0,
    );
    try std.testing.expectEqual(@as(i32, 0), depth);
}

test "countBracesOnLine: negative depth from unmatched close" {
    // Goal: more closes than opens can drive depth
    // negative (to -1, which the assertion allows).
    const depth = countBracesOnLine("}", 0);
    try std.testing.expectEqual(@as(i32, -1), depth);
}

// ============================================================
// extractLineRange — additional edge cases
// ============================================================

test "extractLineRange: end past content returns available" {
    // Goal: when end_line exceeds the number of lines in
    // content, return all lines from start_line onward.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const content = "alpha\nbeta\ngamma";
    const result = extractLineRange(
        arena,
        content,
        2,
        100,
    );
    try std.testing.expectEqualStrings(
        "beta\ngamma",
        result,
    );
}

test "extractLineRange: empty content" {
    // Goal: empty input returns empty string.
    // splitScalar on "" yields one empty slice at
    // line 1, which is within [1,1], so we append "".
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = extractLineRange(arena, "", 1, 1);
    try std.testing.expectEqualStrings("", result);
}

test "extractLineRange: single char no newline" {
    // Goal: content "x" (no trailing newline) is one
    // line, and extracting line 1 returns "x".
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = extractLineRange(arena, "x", 1, 1);
    try std.testing.expectEqualStrings("x", result);
}

test "extractLineRange: middle range of five lines" {
    // Goal: extracting lines 2-4 from a 5-line file.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const content = "one\ntwo\nthree\nfour\nfive\n";
    const result = extractLineRange(
        arena,
        content,
        2,
        4,
    );
    try std.testing.expectEqualStrings(
        "two\nthree\nfour",
        result,
    );
}

test "extractLineRange: last line without trailing newline" {
    // Goal: content without a trailing newline still
    // yields the last line correctly.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const content = "aaa\nbbb\nccc";
    const result = extractLineRange(
        arena,
        content,
        3,
        3,
    );
    try std.testing.expectEqualStrings("ccc", result);
}
