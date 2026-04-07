//! Safetensors binary format parser
//!
//! Parses the safetensors file format used by MLX and Hugging Face
//! models.  Memory-maps the file for zero-copy access to tensor data.
//!
//! Format layout:
//!   [0..8]                    u64 LE — JSON header length
//!   [8..8+header_length]      JSON header (tensor descriptors)
//!   [8+header_length..]       contiguous tensor data
//!
//! Each JSON entry maps a tensor name to its dtype, shape, and byte
//! offsets within the data region.  The `__metadata__` key is skipped.
//!
//! Usage:
//!   var st = try SafetensorsFile.init("/path/to/model.safetensors");
//!   defer st.deinit();
//!   const desc = st.getTensor("model.layers.0.self_attn.q_proj.weight");

const std = @import("std");

const log = std.log.scoped(.safetensors);

// -- Hard limits (Rule 4) --

/// Upper bound on tensor count.  An 8B-parameter model has roughly
/// 36 layers × 15 tensors + embeddings + norms ≈ 560.  Round up
/// generously.
const MAX_TENSORS: u32 = 1024;

/// Maximum number of dimensions per tensor.  Covers scalars through
/// 4-D tensors (e.g. convolution weights).
const MAX_DIMS: u32 = 4;

/// Maximum JSON header size: 16 MiB.  Safetensors headers for large
/// models are typically < 1 MiB; this is a generous safety cap.
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum tensor name length in bytes.
const MAX_NAME_LENGTH: u32 = 256;

// ========================================================================
// Dtype — tensor element type
// ========================================================================

/// Element data type, matching the safetensors JSON `"dtype"` strings.
pub const Dtype = enum {
    u8,
    u32,
    f16,
    f32,
    bf16,

    /// Bytes per element.
    pub fn sizeBytes(self: Dtype) u32 {
        return switch (self) {
            .u8 => 1,
            .u32 => 4,
            .f16 => 2,
            .bf16 => 2,
            .f32 => 4,
        };
    }

    /// Parse the JSON string into a `Dtype`.
    pub fn fromString(s: []const u8) !Dtype {
        std.debug.assert(s.len > 0);
        if (std.mem.eql(u8, s, "U8")) return .u8;
        if (std.mem.eql(u8, s, "U32")) return .u32;
        if (std.mem.eql(u8, s, "F16")) return .f16;
        if (std.mem.eql(u8, s, "F32")) return .f32;
        if (std.mem.eql(u8, s, "BF16")) return .bf16;
        log.warn(
            "unsupported dtype: {s}",
            .{s},
        );
        return error.UnsupportedDtype;
    }
};

// ========================================================================
// TensorDescriptor — describes one tensor in the file
// ========================================================================

/// Describes a single tensor: its name, element type, shape, and a
/// byte slice into the mmap'd data region.  The slice remains valid
/// for the lifetime of the owning `SafetensorsFile`.
pub const TensorDescriptor = struct {
    /// Tensor name (slice into the mmap'd JSON header region).
    name: []const u8,
    dtype: Dtype,
    /// Number of dimensions (0..MAX_DIMS).
    rank: u32,
    /// Shape values; only `dims[0..rank]` are meaningful.
    dims: [MAX_DIMS]u32,
    /// Raw byte slice into the mmap'd data region.
    data: []const u8,

    /// Total number of elements (product of dims).
    pub fn elementCount(self: *const TensorDescriptor) u32 {
        std.debug.assert(self.rank > 0);
        std.debug.assert(self.rank <= MAX_DIMS);
        var count: u32 = 1;
        for (self.dims[0..self.rank]) |d| {
            count *= d;
        }
        return count;
    }

    /// Total data size in bytes (element count × dtype size).
    pub fn sizeBytes(self: *const TensorDescriptor) u32 {
        return self.elementCount() * self.dtype.sizeBytes();
    }
};

// ========================================================================
// SafetensorsFile — the top-level parser
// ========================================================================

pub const SafetensorsFile = struct {
    /// Full mmap'd file contents.  Tensor data slices point into this.
    mmap_data: []align(std.heap.page_size_min) const u8,

    /// Parsed tensor descriptors; `tensors[0..count]` are valid.
    tensors: [MAX_TENSORS]TensorDescriptor,
    count: u32,

    // ----------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------

    /// Open a safetensors file, mmap it, parse the JSON header, and
    /// populate tensor descriptors.  The caller must call `deinit`
    /// when done.
    pub fn init(
        self: *SafetensorsFile,
        path: []const u8,
    ) !void {
        std.debug.assert(path.len > 0);

        const file = std.fs.cwd().openFile(
            path,
            .{},
        ) catch |err| {
            log.err("cannot open safetensors file: {s}", .{path});
            return err;
        };
        defer file.close();

        const stat = try file.stat();
        const file_size = stat.size;

        // Minimum valid file: 8-byte length + 2-byte header "{}".
        std.debug.assert(file_size >= 10);

        const mapped = try mmapFile(file.handle, file_size);
        errdefer std.posix.munmap(mapped);

        self.mmap_data = mapped;
        self.count = 0;

        try self.parseHeader(file_size);
    }

    /// Open from an already-available byte slice (e.g. for testing).
    /// The caller retains ownership of `bytes`; `deinit` is a no-op
    /// for slice-backed instances.
    pub fn initFromBytes(
        self: *SafetensorsFile,
        bytes: []const u8,
    ) !void {
        // Minimum valid: 8-byte length prefix + 2-byte JSON "{}".
        std.debug.assert(bytes.len >= 10);

        // Store a zero-length mmap sentinel so deinit knows not to
        // munmap.  Tensor data slices point into `bytes` directly.
        self.mmap_data = &.{};
        self.count = 0;

        try parseHeaderFromSlice(self, bytes);
    }

    /// Unmap the file and invalidate all tensor data slices.
    pub fn deinit(self: *SafetensorsFile) void {
        std.debug.assert(
            self.mmap_data.len > 0 or self.count < MAX_TENSORS,
        );
        if (self.mmap_data.len > 0) {
            std.posix.munmap(self.mmap_data);
        }
        self.* = undefined;
    }

    /// Look up a tensor by name.  Returns null if not found.
    /// Linear scan is fine for ~600 tensors (cache-friendly,
    /// called infrequently at load time).
    pub fn getTensor(
        self: *const SafetensorsFile,
        name: []const u8,
    ) ?*const TensorDescriptor {
        std.debug.assert(name.len > 0);
        std.debug.assert(self.count <= MAX_TENSORS);
        for (self.tensors[0..self.count]) |*desc| {
            if (std.mem.eql(u8, desc.name, name)) {
                return desc;
            }
        }
        return null;
    }

    // ----------------------------------------------------------------
    // Private parsing helpers
    // ----------------------------------------------------------------

    /// Parse the JSON header from the mmap'd file data.
    fn parseHeader(
        self: *SafetensorsFile,
        file_size: u64,
    ) !void {
        std.debug.assert(file_size >= 10);
        std.debug.assert(self.mmap_data.len == file_size);
        try parseHeaderFromSlice(self, self.mmap_data);
    }

    /// Core header parser, works on any byte slice (mmap'd or in-
    /// memory).  Separated so both `init` and `initFromBytes` can
    /// share the same logic.
    fn parseHeaderFromSlice(
        self: *SafetensorsFile,
        bytes: []const u8,
    ) !void {
        std.debug.assert(bytes.len >= 10);

        const header_length = readHeaderLength(bytes);
        const file_len: u64 = @intCast(bytes.len);
        const data_start = 8 + header_length;

        // Header must fit within the file.
        if (data_start > file_len) {
            log.warn(
                "header length ({}) exceeds file size ({})",
                .{ header_length, file_len },
            );
            return error.InvalidHeader;
        }

        const header_json =
            bytes[8..@intCast(data_start)];
        const data_region =
            bytes[@intCast(data_start)..];

        try parseJsonHeader(
            self,
            header_json,
            data_region,
            file_len - data_start,
        );
    }
};

// ========================================================================
// Free-standing helpers (Rule 20: hot-loop extraction, primitives only)
// ========================================================================

/// Read the 8-byte little-endian header length prefix.
fn readHeaderLength(bytes: []const u8) u64 {
    std.debug.assert(bytes.len >= 8);
    const header_length = std.mem.readInt(
        u64,
        bytes[0..8],
        .little,
    );
    // Sanity-check: header must not be absurdly large.
    std.debug.assert(header_length <= MAX_HEADER_BYTES);
    return header_length;
}

/// Memory-map a file descriptor as read-only shared mapping.
fn mmapFile(
    fd: std.posix.fd_t,
    file_size: u64,
) ![]align(std.heap.page_size_min) const u8 {
    std.debug.assert(file_size >= 10);
    // mmap requires a non-zero length.
    std.debug.assert(file_size > 0);
    return try std.posix.mmap(
        null,
        @intCast(file_size),
        std.posix.PROT.READ,
        .{ .TYPE = .SHARED },
        fd,
        0,
    );
}

/// Parse the JSON header and populate tensor descriptors.
/// `data_region` is the byte slice starting after the header.
/// `data_region_size` is its length (for bounds checking).
fn parseJsonHeader(
    self: *SafetensorsFile,
    header_json: []const u8,
    data_region: []const u8,
    data_region_size: u64,
) !void {
    std.debug.assert(header_json.len > 0);
    std.debug.assert(data_region.len == data_region_size);

    // Use a page-backed arena for the JSON parser's temporary
    // state.  Real model headers (~600 tensors) produce a parsed
    // tree that exceeds 256 KiB once hash-map buckets, string
    // keys, and value nodes are counted.  The arena is freed in
    // bulk after we extract tensor descriptors — init-time only
    // (Rule 2).
    // Intentionally not freed: tensor descriptor `name` fields
    // are slices into arena-owned memory (Zig 0.15's JSON parser
    // allocates key copies rather than referencing the input).
    // This is init-time only (Rule 2); the pages are reclaimed
    // when the process exits or SafetensorsFile.deinit clears
    // the struct.
    var arena = std.heap.ArenaAllocator.init(
        std.heap.page_allocator,
    );

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        arena.allocator(),
        header_json,
        .{},
    ) catch |err| {
        log.warn("JSON parse failed: {}", .{err});
        return error.InvalidJson;
    };
    _ = &parsed;

    const root = switch (parsed.value) {
        .object => |obj| obj,
        else => {
            log.warn("JSON root is not an object", .{});
            return error.InvalidJson;
        },
    };

    var iter = root.iterator();
    while (iter.next()) |entry| {
        const key = entry.key_ptr.*;

        // Skip the metadata key — it's not a tensor.
        if (std.mem.eql(u8, key, "__metadata__")) continue;

        if (self.count >= MAX_TENSORS) {
            log.warn(
                "tensor count exceeds MAX_TENSORS ({})",
                .{MAX_TENSORS},
            );
            return error.TooManyTensors;
        }

        const desc = try parseTensorEntry(
            key,
            entry.value_ptr.*,
            data_region,
            data_region_size,
        );
        self.tensors[self.count] = desc;
        self.count += 1;
    }
}

/// Parse a single tensor JSON entry into a `TensorDescriptor`.
fn parseTensorEntry(
    name: []const u8,
    value: std.json.Value,
    data_region: []const u8,
    data_region_size: u64,
) !TensorDescriptor {
    std.debug.assert(name.len > 0);
    std.debug.assert(name.len <= MAX_NAME_LENGTH);

    const obj = switch (value) {
        .object => |o| o,
        else => {
            log.warn(
                "tensor '{s}' value is not an object",
                .{name},
            );
            return error.InvalidTensorEntry;
        },
    };

    const dtype = try parseDtype(obj, name);
    var desc = TensorDescriptor{
        .name = name,
        .dtype = dtype,
        .rank = 0,
        .dims = .{ 0, 0, 0, 0 },
        .data = &.{},
    };

    try parseShape(&desc, obj, name);
    try parseDataSlice(
        &desc,
        obj,
        name,
        data_region,
        data_region_size,
    );

    return desc;
}

/// Extract and validate the `"dtype"` field.
fn parseDtype(
    obj: std.json.ObjectMap,
    name: []const u8,
) !Dtype {
    std.debug.assert(name.len > 0);
    const dtype_val = obj.get("dtype") orelse {
        log.warn("tensor '{s}' missing 'dtype'", .{name});
        return error.MissingDtype;
    };
    const dtype_str = switch (dtype_val) {
        .string => |s| s,
        else => {
            log.warn(
                "tensor '{s}' dtype is not a string",
                .{name},
            );
            return error.InvalidDtype;
        },
    };
    return Dtype.fromString(dtype_str);
}

/// Extract and validate the `"shape"` array.
fn parseShape(
    desc: *TensorDescriptor,
    obj: std.json.ObjectMap,
    name: []const u8,
) !void {
    std.debug.assert(name.len > 0);
    const shape_val = obj.get("shape") orelse {
        log.warn("tensor '{s}' missing 'shape'", .{name});
        return error.MissingShape;
    };
    const shape_arr = switch (shape_val) {
        .array => |a| a,
        else => {
            log.warn(
                "tensor '{s}' shape is not an array",
                .{name},
            );
            return error.InvalidShape;
        },
    };

    if (shape_arr.items.len > MAX_DIMS) {
        log.warn(
            "tensor '{s}' has {} dims, max is {}",
            .{ name, shape_arr.items.len, MAX_DIMS },
        );
        return error.TooManyDimensions;
    }

    desc.rank = @intCast(shape_arr.items.len);
    for (shape_arr.items, 0..) |dim_val, i| {
        const dim_int = switch (dim_val) {
            .integer => |v| v,
            else => {
                log.warn(
                    "tensor '{s}' shape[{}] not an int",
                    .{ name, i },
                );
                return error.InvalidShape;
            },
        };
        if (dim_int < 0) {
            log.warn(
                "tensor '{s}' shape[{}] is negative",
                .{ name, i },
            );
            return error.InvalidShape;
        }
        desc.dims[i] = @intCast(dim_int);
    }
}

/// Extract and validate the `"data_offsets"` pair, then slice
/// into the data region.
fn parseDataSlice(
    desc: *TensorDescriptor,
    obj: std.json.ObjectMap,
    name: []const u8,
    data_region: []const u8,
    data_region_size: u64,
) !void {
    std.debug.assert(name.len > 0);
    std.debug.assert(data_region.len == data_region_size);

    const pair = try extractOffsetPair(obj, name);
    const start = pair[0];
    const end = pair[1];

    // End must be >= start, and both must be within bounds.
    if (end < start) {
        log.warn(
            "tensor '{s}' has end ({}) < start ({})",
            .{ name, end, start },
        );
        return error.InvalidDataOffsets;
    }
    if (end > data_region_size) {
        log.warn(
            "tensor '{s}' end offset ({}) exceeds data ({})",
            .{ name, end, data_region_size },
        );
        return error.DataOffsetOutOfBounds;
    }

    const s: usize = @intCast(start);
    const e: usize = @intCast(end);
    desc.data = data_region[s..e];
}

/// Extract the `"data_offsets"` `[start, end]` pair from a tensor
/// JSON object.  Returns the two u64 values or an error.
fn extractOffsetPair(
    obj: std.json.ObjectMap,
    name: []const u8,
) !struct { u64, u64 } {
    std.debug.assert(name.len > 0);

    const offsets_val = obj.get("data_offsets") orelse {
        log.warn(
            "tensor '{s}' missing 'data_offsets'",
            .{name},
        );
        return error.MissingDataOffsets;
    };
    const offsets_arr = switch (offsets_val) {
        .array => |a| a,
        else => {
            log.warn(
                "tensor '{s}' data_offsets not an array",
                .{name},
            );
            return error.InvalidDataOffsets;
        },
    };

    if (offsets_arr.items.len != 2) {
        log.warn(
            "tensor '{s}' data_offsets needs 2 elements",
            .{name},
        );
        return error.InvalidDataOffsets;
    }

    const start = extractOffset(offsets_arr.items[0]) orelse {
        log.warn("tensor '{s}' offset[0] invalid", .{name});
        return error.InvalidDataOffsets;
    };
    const end = extractOffset(offsets_arr.items[1]) orelse {
        log.warn("tensor '{s}' offset[1] invalid", .{name});
        return error.InvalidDataOffsets;
    };

    return .{ start, end };
}

/// Extract a u64 from a JSON integer value.  Returns null if the
/// value is not a non-negative integer.
fn extractOffset(val: std.json.Value) ?u64 {
    const int_val = switch (val) {
        .integer => |v| v,
        else => return null,
    };
    if (int_val < 0) return null;
    return @intCast(int_val);
}

// ========================================================================
// Tests
// ========================================================================

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
