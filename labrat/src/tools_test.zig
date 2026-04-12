// Comprehensive tests for the pure functions in tools.zig.
//
// Goal: exercise every public helper with positive, negative, and
// edge-case inputs so that regressions surface immediately.

const std = @import("std");
const tools = @import("tools.zig");

// ============================================================
// 1. eql
// ============================================================

test "eql returns true for identical strings" {
    try std.testing.expect(tools.eql("hello", "hello"));
}

test "eql returns false for different strings" {
    try std.testing.expect(!tools.eql("hello", "world"));
}

test "eql returns true for two empty strings" {
    try std.testing.expect(tools.eql("", ""));
}

test "eql returns false when same prefix but different lengths" {
    try std.testing.expect(!tools.eql("abc", "abcd"));
    try std.testing.expect(!tools.eql("abcd", "abc"));
}

// ============================================================
// 2. startsWith
// ============================================================

test "startsWith returns true on prefix match" {
    try std.testing.expect(tools.startsWith("hello world", "hello"));
}

test "startsWith returns false on mismatch" {
    try std.testing.expect(!tools.startsWith("hello", "world"));
}

test "startsWith returns true for empty prefix" {
    // Every string starts with the empty string.
    try std.testing.expect(tools.startsWith("anything", ""));
}

test "startsWith returns false for empty haystack with non-empty prefix" {
    try std.testing.expect(!tools.startsWith("", "x"));
}

test "startsWith returns true when haystack equals prefix exactly" {
    try std.testing.expect(tools.startsWith("exact", "exact"));
}

// ============================================================
// 3. indexOf
// ============================================================

test "indexOf finds needle in the middle" {
    try std.testing.expectEqual(@as(?usize, 6), tools.indexOf("hello world", "world"));
}

test "indexOf returns null when needle is absent" {
    try std.testing.expectEqual(@as(?usize, null), tools.indexOf("hello", "xyz"));
}

test "indexOf finds empty needle at position zero" {
    try std.testing.expectEqual(@as(?usize, 0), tools.indexOf("hello", ""));
}

test "indexOf finds needle at start" {
    try std.testing.expectEqual(@as(?usize, 0), tools.indexOf("abcdef", "abc"));
}

test "indexOf finds needle at end" {
    try std.testing.expectEqual(@as(?usize, 3), tools.indexOf("abcdef", "def"));
}

// ============================================================
// 4. truncate
// ============================================================

test "truncate returns full string when shorter than max" {
    const result = tools.truncate("hi", 10);
    try std.testing.expect(tools.eql(result, "hi"));
}

test "truncate clips string when longer than max" {
    const result = tools.truncate("hello world", 5);
    try std.testing.expect(tools.eql(result, "hello"));
}

test "truncate returns full string at exact length" {
    const result = tools.truncate("abc", 3);
    try std.testing.expect(tools.eql(result, "abc"));
}

test "truncate handles single character max" {
    const result = tools.truncate("hello", 1);
    try std.testing.expect(tools.eql(result, "h"));
}

test "truncate returns empty string when input is empty" {
    const result = tools.truncate("", 5);
    try std.testing.expect(tools.eql(result, ""));
}

// ============================================================
// 5. trimCR
// ============================================================

test "trimCR strips trailing carriage return" {
    const result = tools.trimCR("hello\r");
    try std.testing.expect(tools.eql(result, "hello"));
}

test "trimCR leaves line without CR unchanged" {
    const result = tools.trimCR("hello");
    try std.testing.expect(tools.eql(result, "hello"));
}

test "trimCR returns empty for empty input" {
    const result = tools.trimCR("");
    try std.testing.expect(tools.eql(result, ""));
}

test "trimCR strips only CR from a lone CR" {
    const result = tools.trimCR("\r");
    try std.testing.expect(tools.eql(result, ""));
}

test "trimCR strips only the last CR when multiple are present" {
    // "\r\r" → "\r" (only the trailing one is removed).
    const result = tools.trimCR("\r\r");
    try std.testing.expect(tools.eql(result, "\r"));
}

// ============================================================
// 6. leadingSpaces
// ============================================================

test "leadingSpaces returns zero for no leading spaces" {
    try std.testing.expectEqual(@as(usize, 0), tools.leadingSpaces("abc"));
}

test "leadingSpaces counts leading spaces" {
    try std.testing.expectEqual(@as(usize, 4), tools.leadingSpaces("    abc"));
}

test "leadingSpaces returns full length for all-spaces string" {
    try std.testing.expectEqual(@as(usize, 3), tools.leadingSpaces("   "));
}

test "leadingSpaces returns zero for empty string" {
    try std.testing.expectEqual(@as(usize, 0), tools.leadingSpaces(""));
}

test "leadingSpaces returns zero when line starts with tab" {
    // Only literal space characters are counted.
    try std.testing.expectEqual(@as(usize, 0), tools.leadingSpaces("\tabc"));
}

// ============================================================
// 7. splitOnce
// ============================================================

test "splitOnce splits at first separator" {
    const result = tools.splitOnce("key=value", '=').?;
    try std.testing.expect(tools.eql(result[0], "key"));
    try std.testing.expect(tools.eql(result[1], "value"));
}

test "splitOnce returns null when separator is absent" {
    try std.testing.expectEqual(
        @as(?[2][]const u8, null),
        tools.splitOnce("nosep", '='),
    );
}

test "splitOnce with separator at start yields empty left part" {
    const result = tools.splitOnce("=value", '=').?;
    try std.testing.expect(tools.eql(result[0], ""));
    try std.testing.expect(tools.eql(result[1], "value"));
}

test "splitOnce with separator at end yields empty right part" {
    const result = tools.splitOnce("key=", '=').?;
    try std.testing.expect(tools.eql(result[0], "key"));
    try std.testing.expect(tools.eql(result[1], ""));
}

test "splitOnce splits at first of multiple separators" {
    const result = tools.splitOnce("a=b=c", '=').?;
    try std.testing.expect(tools.eql(result[0], "a"));
    try std.testing.expect(tools.eql(result[1], "b=c"));
}

// ============================================================
// 8. extractAfterEq
// ============================================================

test "extractAfterEq extracts value from const declaration" {
    const line = "const batch_size: u32 = 42;";
    const result = tools.extractAfterEq(line, "const batch_size").?;
    try std.testing.expect(tools.eql(result, "42"));
}

test "extractAfterEq returns null for wrong prefix" {
    const line = "const batch_size: u32 = 42;";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractAfterEq(line, "const learning_rate"),
    );
}

test "extractAfterEq returns null when no equals sign" {
    const line = "const batch_size: u32;";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractAfterEq(line, "const batch_size"),
    );
}

test "extractAfterEq returns null when no semicolon" {
    const line = "const batch_size: u32 = 42";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractAfterEq(line, "const batch_size"),
    );
}

test "extractAfterEq returns null when semicolon precedes equals" {
    const line = "const x; = 42";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractAfterEq(line, "const x"),
    );
}

test "extractAfterEq trims spaces around value" {
    const line = "const x: u32 =  99  ;";
    const result = tools.extractAfterEq(line, "const x").?;
    try std.testing.expect(tools.eql(result, "99"));
}

// ============================================================
// 9. extractField
// ============================================================

test "extractField extracts value ending with comma" {
    const line = ".batch_size = 128,";
    const result = tools.extractField(line, ".batch_size = ").?;
    try std.testing.expect(tools.eql(result, "128"));
}

test "extractField extracts value ending with closing brace" {
    const line = ".batch_size = 256}";
    const result = tools.extractField(line, ".batch_size = ").?;
    try std.testing.expect(tools.eql(result, "256"));
}

test "extractField returns null when needle not found" {
    const line = ".learning_rate = 0.01,";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractField(line, ".batch_size = "),
    );
}

test "extractField returns null when value is immediately a terminator" {
    // Needle ends right at a comma → end == 0 → null.
    const line = ".batch_size = ,";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractField(line, ".batch_size = "),
    );
}

test "extractField extracts value ending with space" {
    const line = ".batch_size = 64 extra";
    const result = tools.extractField(line, ".batch_size = ").?;
    try std.testing.expect(tools.eql(result, "64"));
}

// ============================================================
// 10. extractDotField
// ============================================================

test "extractDotField extracts dot-prefixed enum value" {
    const line = ".act = .relu,";
    const result = tools.extractDotField(line, ".act = .").?;
    try std.testing.expect(tools.eql(result, "relu"));
}

test "extractDotField extracts value ending with brace" {
    const line = ".act = .sigmoid}";
    const result = tools.extractDotField(line, ".act = .").?;
    try std.testing.expect(tools.eql(result, "sigmoid"));
}

test "extractDotField returns null when needle not found" {
    const line = ".loss = .mse,";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractDotField(line, ".act = ."),
    );
}

test "extractDotField returns null when value immediately terminates" {
    const line = ".act = .,";
    try std.testing.expectEqual(
        @as(?[]const u8, null),
        tools.extractDotField(line, ".act = ."),
    );
}

// ============================================================
// 11. jsonEscape
// ============================================================

test "jsonEscape passes through a plain string unchanged" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.jsonEscape(arena, "hello");
    try std.testing.expect(tools.eql(result, "hello"));
}

test "jsonEscape escapes embedded double quotes" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.jsonEscape(arena, "say \"hi\"");
    try std.testing.expect(tools.eql(result, "say \\\"hi\\\""));
}

test "jsonEscape escapes backslashes" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.jsonEscape(arena, "a\\b");
    try std.testing.expect(tools.eql(result, "a\\\\b"));
}

test "jsonEscape escapes newlines" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.jsonEscape(arena, "line1\nline2");
    try std.testing.expect(tools.eql(result, "line1\\nline2"));
}

test "jsonEscape handles empty string" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.jsonEscape(arena, "");
    try std.testing.expect(tools.eql(result, ""));
}

// ============================================================
// 12. truncateAndEscape
// ============================================================

test "truncateAndEscape returns full input when short" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.truncateAndEscape(
        arena,
        "short",
        100,
    );
    try std.testing.expect(tools.eql(result, "short"));
}

test "truncateAndEscape keeps tail when input exceeds max" {
    // Input: "abcdefghij" (10 bytes), max 5 → keeps "fghij".
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.truncateAndEscape(
        arena,
        "abcdefghij",
        5,
    );
    try std.testing.expect(tools.eql(result, "fghij"));
}

test "truncateAndEscape handles empty input" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.truncateAndEscape(arena, "", 10);
    try std.testing.expect(tools.eql(result, ""));
}

test "truncateAndEscape escapes special chars in tail" {
    // The truncated tail contains a newline that must be
    // escaped.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const result = try tools.truncateAndEscape(
        arena,
        "err\n",
        100,
    );
    try std.testing.expect(tools.eql(result, "err\\n"));
}

// ============================================================
// 13. sortStrings
// ============================================================

test "sortStrings sorts already-sorted array" {
    var items = [_][]const u8{ "alpha", "beta", "gamma" };
    tools.sortStrings(&items);
    try std.testing.expect(tools.eql(items[0], "alpha"));
    try std.testing.expect(tools.eql(items[1], "beta"));
    try std.testing.expect(tools.eql(items[2], "gamma"));
}

test "sortStrings sorts reverse-sorted array" {
    var items = [_][]const u8{ "gamma", "beta", "alpha" };
    tools.sortStrings(&items);
    try std.testing.expect(tools.eql(items[0], "alpha"));
    try std.testing.expect(tools.eql(items[1], "beta"));
    try std.testing.expect(tools.eql(items[2], "gamma"));
}

test "sortStrings handles single element" {
    var items = [_][]const u8{"only"};
    tools.sortStrings(&items);
    try std.testing.expect(tools.eql(items[0], "only"));
}

test "sortStrings handles empty slice" {
    var items = [_][]const u8{};
    tools.sortStrings(&items);
    try std.testing.expectEqual(@as(usize, 0), items.len);
}

test "sortStrings handles duplicates" {
    var items = [_][]const u8{ "b", "a", "b", "a" };
    tools.sortStrings(&items);
    try std.testing.expect(tools.eql(items[0], "a"));
    try std.testing.expect(tools.eql(items[1], "a"));
    try std.testing.expect(tools.eql(items[2], "b"));
    try std.testing.expect(tools.eql(items[3], "b"));
}

// ============================================================
// 14. isBenchmarkFile
// ============================================================

test "isBenchmarkFile matches prefix and .json suffix" {
    const prefixes = [_][]const u8{ "mnist_", "bonsai_" };
    try std.testing.expect(
        tools.isBenchmarkFile("mnist_001.json", &prefixes),
    );
}

test "isBenchmarkFile rejects wrong suffix" {
    const prefixes = [_][]const u8{"mnist_"};
    try std.testing.expect(
        !tools.isBenchmarkFile("mnist_001.txt", &prefixes),
    );
}

test "isBenchmarkFile rejects wrong prefix" {
    const prefixes = [_][]const u8{"mnist_"};
    try std.testing.expect(
        !tools.isBenchmarkFile("bonsai_001.json", &prefixes),
    );
}

test "isBenchmarkFile matches minimal name with prefix" {
    const prefixes = [_][]const u8{"m"};
    try std.testing.expect(
        tools.isBenchmarkFile("m.json", &prefixes),
    );
}

test "isBenchmarkFile rejects bare .json without matching prefix" {
    const prefixes = [_][]const u8{"mnist_"};
    try std.testing.expect(
        !tools.isBenchmarkFile(".json", &prefixes),
    );
}

// ============================================================
// 15. buildArchString
// ============================================================

test "buildArchString formats two-layer network" {
    // Goal: 784->128->10 for a typical MNIST architecture.
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const layers = [_]tools.ArchLayer{
        .{ .input_size = 784, .output_size = 128 },
        .{ .input_size = 128, .output_size = 10 },
    };
    const result = tools.buildArchString(arena, &layers);
    try std.testing.expect(tools.eql(result, "784->128->10"));
}

test "buildArchString formats single-layer network" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const layers = [_]tools.ArchLayer{
        .{ .input_size = 784, .output_size = 10 },
    };
    const result = tools.buildArchString(arena, &layers);
    try std.testing.expect(tools.eql(result, "784->10"));
}

test "buildArchString returns (empty) for no layers" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const layers = [_]tools.ArchLayer{};
    const result = tools.buildArchString(arena, &layers);
    try std.testing.expect(tools.eql(result, "(empty)"));
}

test "buildArchString formats three-layer network" {
    var arena_state = std.heap.ArenaAllocator.init(
        std.testing.allocator,
    );
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const layers = [_]tools.ArchLayer{
        .{ .input_size = 784, .output_size = 256 },
        .{ .input_size = 256, .output_size = 128 },
        .{ .input_size = 128, .output_size = 10 },
    };
    const result = tools.buildArchString(arena, &layers);
    try std.testing.expect(
        tools.eql(result, "784->256->128->10"),
    );
}
