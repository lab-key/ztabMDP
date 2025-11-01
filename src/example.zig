const std = @import("std");
const ztmdp = @import("ztabMDP");

const Allocator = std.mem.Allocator;
const fs = std.fs;
const io = std.io;

const Grid = struct {
    x: u32,
    y: u32,
    reward: ztmdp.ztmdp_value_t,
    occupied: bool,

    pub fn hash(self: Grid) u64 {
        var hasher = std.hash.Fnv1a_64.init();
        hasher.update(std.mem.asBytes(&self.x));
        hasher.update(std.mem.asBytes(&self.y));
        return hasher.final();
    }

    pub fn eql(self: Grid, other: Grid) bool {
        return self.x == other.x and self.y == other.y;
    }

    pub fn less(self: Grid, other: Grid) bool {
        if (self.y < other.y) return true;
        if (self.y > other.y) return false;
        return self.x < other.x;
    }
};

const GridContext = struct {
    pub fn hash(self: @This(), key: Grid) u64 {
        _ = self;
        return key.hash();
    }

    pub fn eql(self: @This(), a: Grid, b: Grid) bool {
        _ = self;
        return a.eql(b);
    }
};

const Direction = struct {
    dir: u32, // 0: up, 1: right, 2: down, 3: left

    pub fn hash(self: Direction) u64 {
        var hasher = std.hash.Fnv1a_64.init();
        hasher.update(std.mem.asBytes(&self.dir));
        return hasher.final();
    }

    pub fn eql(self: Direction, other: Direction) bool {
        return self.dir == other.dir;
    }

    pub fn less(self: Direction, other: Direction) bool {
        return self.dir < other.dir;
    }
};

const DirectionContext = struct {
    pub fn hash(self: @This(), key: Direction) u64 {
        _ = self;
        return key.hash();
    }

    pub fn eql(self: @This(), a: Direction, b: Direction) bool {
        _ = self;
        return a.eql(b);
    }
};

const GridConcept = ztmdp.TraitConcept(Grid){
    .hash = Grid.hash,
    .equal = Grid.eql,
    .less = Grid.less,
};

const DirectionConcept = ztmdp.TraitConcept(Direction){
    .hash = Direction.hash,
    .equal = Direction.eql,
    .less = Direction.less,
};

const State = ztmdp.State(Grid, GridConcept);
const Action = ztmdp.Action(Direction, DirectionConcept);
const Link = ztmdp.Link(State, Action);
const Policy = ztmdp.Policy(State, Action);
const MarkovChain = ztmdp.MarkovChain(Link);
const QLearning = ztmdp.QLearning(Policy, MarkovChain);

const World = struct {
    blocks: std.HashMap(Grid, void, GridContext, std.hash_map.default_max_load_percentage),

    pub fn init(allocator: Allocator) World {
        return .{ .blocks = std.HashMap(Grid, void, GridContext, std.hash_map.default_max_load_percentage).init(allocator) };
    }

    pub fn deinit(self: *World) void {
        self.blocks.deinit();
    }
};

fn populate(allocator: Allocator) !World {
    var world = World.init(allocator);
    const file_content = try std.fs.cwd().readFileAlloc(allocator, "gridworld.txt", 10 * 1024); // 10KB limit
    defer allocator.free(file_content);

    var lines = std.mem.splitScalar(u8, file_content, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var tokenizer = std.mem.splitAny(u8, line, " \t\r");
        const x = try std.fmt.parseInt(u32, tokenizer.next().?, 10);
        const y = try std.fmt.parseInt(u32, tokenizer.next().?, 10);
        const occupied_str = tokenizer.next().?;
        const occupied = std.mem.eql(u8, occupied_str, "1");
        const r = try std.fmt.parseFloat(f64, tokenizer.next().?);

        try world.blocks.put(Grid{ .x = x, .y = y, .reward = r, .occupied = occupied }, {});
    }
    return world;
}

const RandDirectionError = error{InvalidMove};

fn randDirection(world: *const World, rnd: std.Random, current: Grid) !struct { dir: Direction, grid: Grid } {
    while (true) {
        var x = current.x;
        var y = current.y;
        const d = rnd.intRangeAtMost(u32, 0, 3);

        switch (d) {
            0 => y -%= 1,
            1 => x += 1,
            2 => y += 1,
            3 => x -%= 1,
            else => unreachable,
        }

        const next_grid_query = Grid{ .x = x, .y = y, .reward = 0, .occupied = false };
        if (world.blocks.getEntry(next_grid_query)) |entry| {
            const found_grid = entry.key_ptr.*;
            if (!found_grid.occupied) {
                return .{ .dir = Direction{ .dir = d }, .grid = found_grid };
            }
        }
    }
}

fn explore(allocator: Allocator, world: *const World, rnd: std.Random, start: Grid) !MarkovChain {
    var episode = MarkovChain.init(allocator);
    var stop = false;

    std.debug.print("starting exploration from: {d},{d}\n", .{ start.x, start.y });
    var curr = start;
    var state_now = State.init(curr, curr.reward);

    while (!stop) {
        const next_move = try randDirection(world, rnd, curr);
        const action_now = Action.init(next_move.dir);
        try episode.addLink(Link.init(state_now, action_now));

        curr = next_move.grid;
        state_now = State.init(curr, curr.reward);

        if (curr.reward == -1 or curr.reward == 1) {
            stop = true;
        }
        std.debug.print("coord: {d},{d} = {d}\n", .{ curr.x, curr.y, curr.reward });
    }

    const action_empty = Action.init(Direction{ .dir = 100 });
    try episode.addLink(Link.init(state_now, action_empty));
    return episode;
}

fn on_policy(world: *const World, policy_map: *Policy, start: Grid) void {
    var curr = start;
    std.debug.print("starting from: {d},{d} = {d}\n", .{ curr.x, curr.y, curr.reward });
    var state_t = State.init(curr, curr.reward);

    while (true) {
        const best_action = policy_map.getBestAction(state_t);
        if (best_action) |action| {
            switch (action.trait_data.dir) {
                0 => curr.y -%= 1,
                1 => curr.x += 1,
                2 => curr.y += 1,
                3 => curr.x -%= 1,
                else => unreachable,
            }

            if (world.blocks.getEntry(curr)) |entry| {
                curr = entry.key_ptr.*;
                const state_n = State.init(curr, curr.reward);
                std.debug.print("coord: {d},{d} = {d}\n", .{ curr.x, curr.y, curr.reward });
                state_t = state_n;
                if (curr.reward == -1.0 or curr.reward == 1.0) {
                    break;
                }
            } else {
                std.debug.print("Fell off the world!\n", .{});
                break;
            }
        } else {
            std.debug.print("No policy for this state.\n", .{});
            break;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) {
            std.log.err("Memory leak detected!", .{});
        }
    }
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const rnd = prng.random();

    var world = try populate(allocator);
    defer world.deinit();

    const start_grid = Grid{ .x = 1, .y = 8, .reward = 0, .occupied = false };

    var policies = Policy.init(allocator);
    defer policies.deinit();

    var args = std.process.args();
    _ = args.next(); // Skip executable name
    var command: ?[]const u8 = null;
    if (args.next()) |arg| {
        command = arg;
    }

    if (command) |cmd| {
        if (std.mem.eql(u8, cmd, "load")) {
            std.debug.print("Loading policy from gridworld.policy...\n", .{});
            var file = try fs.cwd().openFile("gridworld.policy", .{ .mode = .read_only });
            defer file.close();
            policies = try Policy.fromText(allocator, &file, Grid, Direction);
            std.debug.print("Policy loaded. Running on-policy algorithm.\n", .{});
            on_policy(&world, &policies, start_grid);
            return;
        } else {
            std.debug.print("Unknown command: {s}. Training new policy.\n", .{cmd});
        }
    }

    // Training logic
    std.debug.print("Training new policy...\n", .{});
    var episodes = std.array_list.Managed(MarkovChain).init(allocator);
    defer {
        for (episodes.items) |*ep| {
            ep.deinit();
        }
        episodes.deinit();
    }

    std.debug.print("---" ++ " Exploring ---" ++ "\n", .{});
    while (true) {
        var episode = try explore(allocator, &world, rnd, start_grid);
        try episodes.append(episode);

        const last_link = episode.getLink(episode.size() - 1).?;
        if (last_link.state.reward == 1.0) {
            break;
        }
    }

    std.debug.print("---" ++ " Training ---" ++ "\n", .{});
    var learner = QLearning.init(0.9, 0.9);
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        for (episodes.items) |episode| {
            try learner.processEpisode(episode, &policies);
        }
    }

    std.debug.print("Training complete.\n", .{});

    // Save policy logic
    std.debug.print("Saving policy to gridworld.policy...\n", .{});
    var file = try fs.cwd().createFile("gridworld.policy", .{});
    defer file.close();

    try policies.toText(&file);
    std.debug.print("Policy saved.\n", .{});

    std.debug.print("---" ++ " Running on-policy ---" ++ "\n", .{});
    on_policy(&world, &policies, start_grid);
}
