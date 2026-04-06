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

    const api_client_module = b.createModule(.{
        .root_source_file = b.path("src/api_client.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tools.zig", .module = tools_module },
        },
    });

    const ollama_client_module = b.createModule(.{
        .root_source_file = b.path("src/ollama_client.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "api_client.zig", .module = api_client_module },
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

    // ── MNIST research CLI ────────────────────────────────────────────
    const mnist_research = b.addExecutable(.{
        .name = "mnist_research",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mnist_research.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
            },
        }),
    });
    b.installArtifact(mnist_research);

    const mr_step = b.step("mnist-research", "MNIST training research and benchmarking");
    const mr_cmd = b.addRunArtifact(mnist_research);
    mr_step.dependOn(&mr_cmd.step);
    mr_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        mr_cmd.addArgs(args);
    }

    // ── Bonsai research CLI ───────────────────────────────────────────
    const bonsai_research = b.addExecutable(.{
        .name = "bonsai_research",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_research.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
            },
        }),
    });
    b.installArtifact(bonsai_research);

    const br_step = b.step("bonsai-research", "Bonsai inference research and benchmarking");
    const br_cmd = b.addRunArtifact(bonsai_research);
    br_step.dependOn(&br_cmd.step);
    br_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        br_cmd.addArgs(args);
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
                .{ .name = "ollama_client.zig", .module = ollama_client_module },
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

    // ── Tests ─────────────────────────────────────────────────────────
    // (No library tests — these are standalone CLI tools)
}
