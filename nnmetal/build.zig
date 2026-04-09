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

    // ── 1-bit MNIST integration test ──────────────────────────────────
    const mnist_1bit = b.addExecutable(.{
        .name = "mnist_1bit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/mnist_1bit.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(mnist_1bit);

    const run_1bit_step = b.step("run-1bit", "Run the 1-bit MNIST integration test");
    const run_1bit_cmd = b.addRunArtifact(mnist_1bit);
    run_1bit_step.dependOn(&run_1bit_cmd.step);
    run_1bit_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_1bit_cmd.addArgs(args);
    }

    // ── Inference benchmark ───────────────────────────────────────────
    const infer_bench = b.addExecutable(.{
        .name = "inference_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/inference_bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(infer_bench);

    const run_infer_step = b.step("run-infer", "Run the inference benchmark");
    const run_infer_cmd = b.addRunArtifact(infer_bench);
    run_infer_step.dependOn(&run_infer_cmd.step);
    run_infer_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_infer_cmd.addArgs(args);
    }

    // ── Bonsai 1.7B CLI ───────────────────────────────────────────────
    const bonsai = b.addExecutable(.{
        .name = "bonsai",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/bonsai.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai);

    const run_bonsai_step = b.step("run-bonsai", "Run the Bonsai 1.7B inference CLI");
    const run_bonsai_cmd = b.addRunArtifact(bonsai);
    run_bonsai_step.dependOn(&run_bonsai_cmd.step);
    run_bonsai_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bonsai_cmd.addArgs(args);
    }

    // ── Bonsai 1.7B benchmark ─────────────────────────────────────────
    const bonsai_bench = b.addExecutable(.{
        .name = "bonsai_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/bonsai_bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_bench);

    const run_bonsai_bench_step = b.step(
        "run-bonsai-bench",
        "Run the Bonsai 1.7B inference benchmark",
    );
    const run_bonsai_bench_cmd = b.addRunArtifact(bonsai_bench);
    run_bonsai_bench_step.dependOn(&run_bonsai_bench_cmd.step);
    run_bonsai_bench_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bonsai_bench_cmd.addArgs(args);
    }

    // ── Bonsai 1.7B Q4 benchmark ──────────────────────────────────────
    const bonsai_q4_bench = b.addExecutable(.{
        .name = "bonsai_q4_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/bonsai_q4_bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_q4_bench);

    const run_bonsai_q4_bench_step = b.step(
        "run-bonsai-q4-bench",
        "Run the Qwen3 1.7B Q4 inference benchmark",
    );
    const run_bonsai_q4_bench_cmd = b.addRunArtifact(bonsai_q4_bench);
    run_bonsai_q4_bench_step.dependOn(&run_bonsai_q4_bench_cmd.step);
    run_bonsai_q4_bench_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bonsai_q4_bench_cmd.addArgs(args);
    }

    // ── Bonsai 1.7B golden output test ────────────────────────────────
    const bonsai_golden = b.addExecutable(.{
        .name = "bonsai_golden",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/bonsai_golden.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_golden);

    const run_bonsai_golden_step = b.step(
        "run-bonsai-golden",
        "Run the Bonsai 1.7B golden output test",
    );
    const run_bonsai_golden_cmd = b.addRunArtifact(bonsai_golden);
    run_bonsai_golden_step.dependOn(&run_bonsai_golden_cmd.step);
    run_bonsai_golden_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bonsai_golden_cmd.addArgs(args);
    }

    // ── Bonsai 1.7B Q4 golden output test ─────────────────────────────
    const bonsai_q4_golden = b.addExecutable(.{
        .name = "bonsai_q4_golden",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/bonsai_q4_golden.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_q4_golden);

    const run_bonsai_q4_golden_step = b.step(
        "run-bonsai-q4-golden",
        "Run the Bonsai 1.7B Q4 golden output test",
    );
    const run_bonsai_q4_golden_cmd = b.addRunArtifact(bonsai_q4_golden);
    run_bonsai_q4_golden_step.dependOn(&run_bonsai_q4_golden_cmd.step);
    run_bonsai_q4_golden_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bonsai_q4_golden_cmd.addArgs(args);
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
