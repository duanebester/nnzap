const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── zig-objc dependency (for Metal API calls) ─────────────────────────────────────────────
    const objc_dep = b.dependency("zig_objc", .{
        .target = target,
        .optimize = optimize,
    });

    // ── nnzap library module ───────────────────────────────────────────────────────────────────
    const mod = b.addModule("nnzap", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("objc", objc_dep.module("objc"));
    mod.linkFramework("Metal", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("CoreGraphics", .{});
    mod.link_libc = true;

    // ── Main executable (demos / tests) ────────────────────────────────────────────────────────
    const exe = b.addExecutable(.{
        .name = "nnzap",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nnzap", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the demo");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // ── Autoresearch script (Rule 25 — tooling in Zig) ─────────────────────────────────────────
    const autoresearch = b.addExecutable(.{
        .name = "autoresearch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("scripts/autoresearch.zig"),
            .target = target,
            .optimize = optimize,
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

    // ── Engine research script (Rule 25 — tooling in Zig) ──────────────────────────────────────
    const engine_research = b.addExecutable(.{
        .name = "engine_research",
        .root_module = b.createModule(.{
            .root_source_file = b.path("scripts/engine_research.zig"),
            .target = target,
            .optimize = optimize,
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

    // ── Agent script (Rule 25 — tooling in Zig) ────────────────────────────────────────────────
    const agent = b.addExecutable(.{
        .name = "agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("scripts/agent.zig"),
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

    // ── Engine agent script (Rule 25 — tooling in Zig) ─────────────────────────────────────────
    const engine_agent = b.addExecutable(.{
        .name = "engine_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("scripts/engine_agent.zig"),
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

    // ── Tests ──────────────────────────────────────────────────────────────────────────────────
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
