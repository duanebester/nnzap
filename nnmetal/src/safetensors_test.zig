const std = @import("std");
const safetensors = @import("safetensors.zig");

const SafetensorsFile = safetensors.SafetensorsFile;
const Dtype = safetensors.Dtype;

/// Build a minimal valid safetensors file in memory for testing.
/// Returns a stack-allocated buffer and its used length.
fn buildTestFile(
    comptime json_str: []const u8,
    comptime data_size: u32,
) [8 + json_str.len + data_size]u8 {
    comptime std.debug.assert(json_str.len > 0);
    const total = 8 + json_str.len + data_size;
    var buf: [total]u8 = undefined;

    // Write header length as u64 LE.
    const header_len: u64 = json_str.len;
    @memcpy(buf[0..8], std.mem.asBytes(&std.mem.nativeToLittle(
        u64,
        header_len,
    )));

    // Write JSON header.
    @memcpy(buf[8..][0..json_str.len], json_str);

    // Zero-fill data region (Rule 21: no stale data).
    @memset(buf[8 + json_str.len ..], 0);

    return buf;
}

test "parse safetensors header from bytes" {
    const json =
        \\{"tensor_a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    ;
    // 2 × 3 × 4 bytes = 24 bytes of F32 data.
    const file_bytes = buildTestFile(json, 24);

    var sf: SafetensorsFile = undefined;
    try sf.initFromBytes(&file_bytes);

    // Exactly one tensor parsed.
    try std.testing.expectEqual(@as(u32, 1), sf.count);

    const desc = &sf.tensors[0];
    try std.testing.expectEqualStrings("tensor_a", desc.name);
    try std.testing.expectEqual(Dtype.f32, desc.dtype);
    try std.testing.expectEqual(@as(u32, 2), desc.rank);
    try std.testing.expectEqual(@as(u32, 2), desc.dims[0]);
    try std.testing.expectEqual(@as(u32, 3), desc.dims[1]);
    try std.testing.expectEqual(@as(usize, 24), desc.data.len);
    try std.testing.expectEqual(@as(u32, 6), desc.elementCount());
    try std.testing.expectEqual(@as(u32, 24), desc.sizeBytes());
}

test "getTensor returns correct descriptor" {
    const json =
        \\{"weight": {"dtype": "F16", "shape": [4],
        \\"data_offsets": [0, 8]},
        \\"bias": {"dtype": "F32", "shape": [2],
        \\"data_offsets": [8, 16]}}
    ;
    // 4 × 2 bytes (F16) + 2 × 4 bytes (F32) = 16 bytes total.
    const file_bytes = buildTestFile(json, 16);

    var sf: SafetensorsFile = undefined;
    try sf.initFromBytes(&file_bytes);

    try std.testing.expectEqual(@as(u32, 2), sf.count);

    // Look up "bias" by name.
    const bias = sf.getTensor("bias") orelse {
        return error.TestUnexpectedResult;
    };
    try std.testing.expectEqualStrings("bias", bias.name);
    try std.testing.expectEqual(Dtype.f32, bias.dtype);
    try std.testing.expectEqual(@as(u32, 1), bias.rank);
    try std.testing.expectEqual(@as(u32, 2), bias.dims[0]);
    try std.testing.expectEqual(@as(usize, 8), bias.data.len);

    // Look up "weight" by name.
    const weight = sf.getTensor("weight") orelse {
        return error.TestUnexpectedResult;
    };
    try std.testing.expectEqualStrings("weight", weight.name);
    try std.testing.expectEqual(Dtype.f16, weight.dtype);

    // Non-existent tensor returns null.
    try std.testing.expect(sf.getTensor("nonexistent") == null);
}

test "data offsets are validated" {
    // End offset (999) exceeds the 24-byte data region.
    const json =
        \\{"bad": {"dtype": "F32", "shape": [2], "data_offsets": [0, 999]}}
    ;
    const file_bytes = buildTestFile(json, 24);

    var sf: SafetensorsFile = undefined;
    const result = sf.initFromBytes(&file_bytes);
    try std.testing.expectError(
        error.DataOffsetOutOfBounds,
        result,
    );
}

test "metadata key is skipped" {
    const json =
        \\{"real_tensor": {"dtype": "U8", "shape": [4],
        \\"data_offsets": [0, 4]},
        \\"__metadata__": {"format": "test",
        \\"version": "1.0"}}
    ;
    const file_bytes = buildTestFile(json, 4);

    var sf: SafetensorsFile = undefined;
    try sf.initFromBytes(&file_bytes);

    // Only the real tensor, not __metadata__.
    try std.testing.expectEqual(@as(u32, 1), sf.count);
    try std.testing.expectEqualStrings(
        "real_tensor",
        sf.tensors[0].name,
    );
    try std.testing.expectEqual(Dtype.u8, sf.tensors[0].dtype);

    // Confirm metadata is not findable as a tensor.
    try std.testing.expect(
        sf.getTensor("__metadata__") == null,
    );
}

test "Dtype.fromString round-trips" {
    try std.testing.expectEqual(Dtype.u8, try Dtype.fromString("U8"));
    try std.testing.expectEqual(
        Dtype.f16,
        try Dtype.fromString("F16"),
    );
    try std.testing.expectEqual(
        Dtype.f32,
        try Dtype.fromString("F32"),
    );
    try std.testing.expectEqual(
        Dtype.bf16,
        try Dtype.fromString("BF16"),
    );
    try std.testing.expectError(
        error.UnsupportedDtype,
        Dtype.fromString("I32"),
    );
}

test "Dtype.sizeBytes" {
    try std.testing.expectEqual(@as(u32, 1), Dtype.u8.sizeBytes());
    try std.testing.expectEqual(@as(u32, 2), Dtype.f16.sizeBytes());
    try std.testing.expectEqual(@as(u32, 2), Dtype.bf16.sizeBytes());
    try std.testing.expectEqual(@as(u32, 4), Dtype.f32.sizeBytes());
}
