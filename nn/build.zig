const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── zig-objc dependency (for Metal API calls) ─────────────────────
    const objc_dep = b.dependency("zig_objc", .{
        .target = target,
        .optimize = optimize,
    });

    // ── nn library module ─────────────────────────────────────────────
    const mod = b.addModule("nn", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("objc", objc_dep.module("objc"));
    mod.linkFramework("Metal", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("CoreGraphics", .{});
    mod.link_libc = true;

    // ── MNIST example ─────────────────────────────────────────────────
    const mnist_example = b.addExecutable(.{
        .name = "mnist",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/mnist.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(mnist_example);

    const run_step = b.step("run", "Run the MNIST example");
    const run_cmd = b.addRunArtifact(mnist_example);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // ── Tests ─────────────────────────────────────────────────────────
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const example_tests = b.addTest(.{
        .root_module = mnist_example.root_module,
    });
    const run_example_tests = b.addRunArtifact(example_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_example_tests.step);
}
