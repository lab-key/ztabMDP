const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ztabMDP_module = b.createModule(.{
        .root_source_file = b.path("src/ztabMDP.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "ztabMDP",
        .root_module = ztabMDP_module,
        .linkage = .static,
    });
    b.installArtifact(lib);

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/ztabMDP.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "ztabMDP", .module = ztabMDP_module },
        },
    });

    const main_tests = b.addTest(.{
        .root_module = test_module,
    });

    const run_tests = b.step("test", "Run ztabMDP tests");
    run_tests.dependOn(&main_tests.step);

    const gridworld_module = b.createModule(.{
        .root_source_file = b.path("src/example.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "ztabMDP", .module = ztabMDP_module },
        },
    });

    const gridworld_exe = b.addExecutable(.{
        .name = "gridworld_offline",
        .root_module = gridworld_module,
    });
    b.installArtifact(gridworld_exe);

    const run_gridworld_cmd = b.addRunArtifact(gridworld_exe);
    const run_gridworld_step = b.step("run-gridworld", "Run the gridworld offline example");
    run_gridworld_step.dependOn(&run_gridworld_cmd.step);
}
