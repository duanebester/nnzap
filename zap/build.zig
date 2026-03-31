const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Shared tools module ───────────────────────────────────────────
    const tools_module = b.createModule(.{
        .root_source_file = b.path("src/tools.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── Autoresearch CLI ──────────────────────────────────────────────
    const autoresearch = b.addExecutable(.{
        .name = "autoresearch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/autoresearch.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
            },
        }),
    });
    b.installArtifact(autoresearch);

    const ar_step = b.step("autoresearch", "Benchmark comparison and sweep");
    const ar_cmd = b.addRunArtifact(autoresearch);
    ar_step.dependOn(&ar_cmd.step);
    ar_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        ar_cmd.addArgs(args);
    }

    // ── Engine research CLI ───────────────────────────────────────────
    const engine_research = b.addExecutable(.{
        .name = "engine_research",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/engine_research.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
            },
        }),
    });
    b.installArtifact(engine_research);

    const er_step = b.step("engine-research", "Engine code research and benchmarking");
    const er_cmd = b.addRunArtifact(engine_research);
    er_step.dependOn(&er_cmd.step);
    er_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        er_cmd.addArgs(args);
    }

    // ── Agent (LLM experiment runner) ─────────────────────────────────
    const agent = b.addExecutable(.{
        .name = "agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/agent.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(agent);

    const agent_step = b.step("agent", "Autonomous experiment runner");
    const agent_cmd = b.addRunArtifact(agent);
    agent_step.dependOn(&agent_cmd.step);
    agent_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        agent_cmd.addArgs(args);
    }

    // ── Engine agent (LLM engine optimiser) ───────────────────────────
    const engine_agent = b.addExecutable(.{
        .name = "engine_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/engine_agent.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(engine_agent);

    const ea_step = b.step("engine-agent", "Autonomous engine optimisation runner");
    const ea_cmd = b.addRunArtifact(engine_agent);
    ea_step.dependOn(&ea_cmd.step);
    ea_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        ea_cmd.addArgs(args);
    }

    // ── Tests ─────────────────────────────────────────────────────────
    // (No library tests — these are standalone CLI tools)
}
