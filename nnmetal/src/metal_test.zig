const std = @import("std");
const metal = @import("metal.zig");
const Device = metal.Device;

test "memory_budget is half of recommendedMaxWorkingSetSize" {
    // Goal: confirm the budget stored at init is exactly half of
    // what Metal reports as its recommended working set ceiling.
    // Methodology: init a Device, re-query recommendedMaxWorkingSetSize
    // via msgSend, and compare against the stored field.
    var device: Device = undefined;
    try device.init();

    const recommended: u64 = @intCast(device.obj.msgSend(
        c_ulong,
        "recommendedMaxWorkingSetSize",
        .{},
    ));

    // Budget must be positive — a zero budget would reject every
    // allocation including the first parameter buffer.
    try std.testing.expect(device.memory_budget_bytes > 0);

    // Budget must equal exactly half of the Metal recommendation.
    try std.testing.expectEqual(
        recommended / 2,
        device.memory_budget_bytes,
    );
}

test "createBuffer succeeds within budget" {
    // Goal: a small allocation (1,000 floats = 4 KB) must succeed,
    // return the correct length, and be zero-filled on creation
    // (buffer bleed prevention — stale data must never be visible).
    var device: Device = undefined;
    try device.init();

    var buf = try device.createBuffer(1_000);
    defer buf.deinit();

    try std.testing.expectEqual(@as(u32, 1_000), buf.len);

    const slice = buf.asSlice();
    try std.testing.expectEqual(@as(usize, 1_000), slice.len);

    for (slice) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "createBuffer returns MetalMemoryBudgetExceeded when over limit" {
    // Goal: exercise the fail-fast error path in createBuffer.
    // Methodology: shrink the budget to 1 byte so any allocation
    // exceeds it, assert the correct error is returned, then restore
    // the budget via defer so subsequent tests are unaffected.
    var device: Device = undefined;
    try device.init();

    const saved_budget = device.memory_budget_bytes;
    device.memory_budget_bytes = 1;
    defer device.memory_budget_bytes = saved_budget;

    try std.testing.expectError(
        error.MetalMemoryBudgetExceeded,
        device.createBuffer(1),
    );
}
