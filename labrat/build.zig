const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Shared modules ────────────────────────────────────────────────
    const tools_module = b.createModule(.{
        .root_source_file = b.path("src/tools.zig"),
        .target = target,
        .optimize = optimize,
    });

    const toolbox_module = b.createModule(.{
        .root_source_file = b.path("src/toolbox.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tools.zig", .module = tools_module },
        },
    });

    const api_client_module = b.createModule(.{
        .root_source_file = b.path("src/api_client.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tools.zig", .module = tools_module },
        },
    });

    const agent_core_module = b.createModule(.{
        .root_source_file = b.path("src/agent_core.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "api_client.zig", .module = api_client_module },
            .{ .name = "tools.zig", .module = tools_module },
        },
    });

    // ── MNIST researcher CLI ──────────────────────────────────────────
    const mnist_researcher = b.addExecutable(.{
        .name = "mnist_researcher",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mnist_researcher.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
                .{ .name = "toolbox.zig", .module = toolbox_module },
            },
        }),
    });
    b.installArtifact(mnist_researcher);

    const mr_step = b.step("mnist-researcher", "MNIST training research and benchmarking");
    const mr_cmd = b.addRunArtifact(mnist_researcher);
    mr_step.dependOn(&mr_cmd.step);
    mr_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        mr_cmd.addArgs(args);
    }

    // ── Bonsai researcher CLI ─────────────────────────────────────────
    const bonsai_researcher = b.addExecutable(.{
        .name = "bonsai_researcher",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_researcher.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
                .{ .name = "toolbox.zig", .module = toolbox_module },
            },
        }),
    });
    b.installArtifact(bonsai_researcher);

    const br_step = b.step("bonsai-researcher", "Bonsai inference research and benchmarking");
    const br_cmd = b.addRunArtifact(bonsai_researcher);
    br_step.dependOn(&br_cmd.step);
    br_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        br_cmd.addArgs(args);
    }

    // ── Bonsai Q4 researcher CLI ──────────────────────────────────────
    const bonsai_q4_researcher = b.addExecutable(.{
        .name = "bonsai_q4_researcher",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_q4_researcher.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
                .{ .name = "toolbox.zig", .module = toolbox_module },
            },
        }),
    });
    b.installArtifact(bonsai_q4_researcher);

    const bq4r_step = b.step("bonsai-q4-researcher", "Bonsai Q4 inference research and benchmarking");
    const bq4r_cmd = b.addRunArtifact(bonsai_q4_researcher);
    bq4r_step.dependOn(&bq4r_cmd.step);
    bq4r_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        bq4r_cmd.addArgs(args);
    }

    // ── MNIST agent (LLM experiment runner) ───────────────────────────
    const mnist_agent = b.addExecutable(.{
        .name = "mnist_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mnist_agent.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "agent_core.zig", .module = agent_core_module },
            },
        }),
    });
    b.installArtifact(mnist_agent);

    const ma_step = b.step("mnist-agent", "Autonomous MNIST experiment runner");
    const ma_cmd = b.addRunArtifact(mnist_agent);
    ma_step.dependOn(&ma_cmd.step);
    ma_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        ma_cmd.addArgs(args);
    }

    // ── Bonsai agent (LLM engine optimiser) ───────────────────────────
    const bonsai_agent = b.addExecutable(.{
        .name = "bonsai_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_agent.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "agent_core.zig", .module = agent_core_module },
            },
        }),
    });
    b.installArtifact(bonsai_agent);

    const ba_step = b.step("bonsai-agent", "Autonomous engine optimisation runner");
    const ba_cmd = b.addRunArtifact(bonsai_agent);
    ba_step.dependOn(&ba_cmd.step);
    ba_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        ba_cmd.addArgs(args);
    }

    // ── Bonsai Q4 agent (LLM engine optimiser) ────────────────────────
    const bonsai_q4_agent = b.addExecutable(.{
        .name = "bonsai_q4_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_q4_agent.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "agent_core.zig", .module = agent_core_module },
            },
        }),
    });
    b.installArtifact(bonsai_q4_agent);

    const bq4a_step = b.step("bonsai-q4-agent", "Autonomous Q4 engine optimisation runner");
    const bq4a_cmd = b.addRunArtifact(bonsai_q4_agent);
    bq4a_step.dependOn(&bq4a_cmd.step);
    bq4a_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        bq4a_cmd.addArgs(args);
    }

    // ── Tests ─────────────────────────────────────────────────────────
    const toolbox_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/toolbox.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
            },
        }),
    });
    const run_toolbox_tests = b.addRunArtifact(
        toolbox_tests,
    );
    const test_step = b.step(
        "test",
        "Run toolbox unit tests",
    );
    test_step.dependOn(&run_toolbox_tests.step);
}
