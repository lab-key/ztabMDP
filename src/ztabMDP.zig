const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ztmdp_value_t = f64;

pub const ztabMDPObjectType = enum {
    state,
    action,
};

// Context for std.io.GenericReader to wrap std.fs.File
const FileReadContext = struct {
    file: *std.fs.File,
};

// Read function for std.io.GenericReader
fn fileReadFn(context: FileReadContext, buffer: []u8) !usize {
    return context.file.read(buffer);
}

// Context for std.io.GenericWriter to wrap std.fs.File
const FileWriteContext = struct {
    file: *std.fs.File,
};

// Write function for std.io.GenericWriter
fn fileWriteFn(context: FileWriteContext, bytes: []const u8) !usize {
    return context.file.write(bytes);
}

// TraitConcept defines the interface for user-defined trait data
// It replaces the C function pointers for hash, equal, and less.
pub fn TraitConcept(comptime T: type) type {
    return struct {
        hash: fn (trait: T) u64,
        equal: fn (trait1: T, trait2: T) bool,
        less: fn (trait1: T, trait2: T) bool,
    };
}

// State represents a state in the reinforcement learning environment.
// It is generic over the user-defined 'Trait' type and requires a 'concept'
// struct that provides the necessary operations (hash, equal, less).
pub fn State(comptime Trait: type, comptime concept: TraitConcept(Trait)) type {
    return struct {
        object_type: ztabMDPObjectType = .state,
        trait_data: Trait,
        reward: ztmdp_value_t,

        pub fn init(trait_data: Trait, reward: ztmdp_value_t) @This() {
            return .{
                .trait_data = trait_data,
                .reward = reward,
            };
        }

        pub fn hash(self: @This()) u64 {
            return concept.hash(self.trait_data);
        }

        pub fn equals(self: @This(), other: @This()) bool {
            return concept.equal(self.trait_data, other.trait_data);
        }

        pub fn less(self: @This(), other: @This()) bool {
            return concept.less(self.trait_data, other.trait_data);
        }

        pub fn setReward(self: *@This(), new_reward: ztmdp_value_t) void {
            self.reward = new_reward;
        }
    };
}

// Action represents an action in the reinforcement learning environment.
// Similar to State, it is generic over the user-defined 'Trait' type and
// requires a 'concept' struct for operations.
pub fn Action(comptime Trait: type, comptime concept: TraitConcept(Trait)) type {
    return struct {
        object_type: ztabMDPObjectType = .action,
        trait_data: Trait,

        pub fn init(trait_data: Trait) @This() {
            return .{
                .trait_data = trait_data,
            };
        }

        pub fn hash(self: @This()) u64 {
            return concept.hash(self.trait_data);
        }

        pub fn equals(self: @This(), other: @This()) bool {
            return concept.equal(self.trait_data, other.trait_data);
        }

        pub fn less(self: @This(), other: @This()) bool {
            return concept.less(self.trait_data, other.trait_data);
        }
    };
}

// Link represents a transition from a state via an action.
pub fn Link(comptime StateType: type, comptime ActionType: type) type {
    return struct {
        state: StateType,
        action: ActionType,

        pub fn init(state: StateType, action: ActionType) @This() {
            return .{
                .state = state,
                .action = action,
            };
        }

        pub fn equals(self: @This(), other: @This()) bool {
            return self.state.equals(other.state) and self.action.equals(other.action);
        }

        pub fn less(self: @This(), other: @This()) bool {
            if (self.action.less(other.action)) return true;
            if (self.action.equals(other.action)) return self.state.less(other.state);
            return false;
        }
    };
}

// Context for State in std.HashMap
pub fn StateContext(comptime StateType: type) type {
    return struct {
        pub fn hash(self: @This(), state: StateType) u64 {
            _ = self;
            return state.hash();
        }
        pub fn eql(self: @This(), state1: StateType, state2: StateType) bool {
            _ = self;
            return state1.equals(state2);
        }
    };
}

// Context for Action in std.HashMap
pub fn ActionContext(comptime ActionType: type) type {
    return struct {
        pub fn hash(self: @This(), action: ActionType) u64 {
            _ = self;
            return action.hash();
        }
        pub fn eql(self: @This(), action1: ActionType, action2: ActionType) bool {
            _ = self;
            return action1.equals(action2);
        }
    };
}

// Policy stores the Q-values for state-action pairs.
pub fn Policy(comptime StateType: type, comptime ActionType: type) type {
    const ActionMap = std.HashMap(ActionType, ztmdp_value_t, ActionContext(ActionType), std.hash_map.default_max_load_percentage);
    const StateMap = std.HashMap(StateType, ActionMap, StateContext(StateType), std.hash_map.default_max_load_percentage);

    return struct {
        allocator: Allocator,
        policies: StateMap,

        pub fn init(allocator: Allocator) @This() {
            return .{
                .allocator = allocator,
                .policies = StateMap.init(allocator),
            };
        }

        pub fn deinit(self: *@This()) void {
            var state_iterator = self.policies.iterator();
            while (state_iterator.next()) |entry| {
                entry.value_ptr.*.deinit(); // Deinit inner ActionMap
            }
            self.policies.deinit();
        }

        pub fn update(self: *@This(), state: StateType, action: ActionType, q_value: ztmdp_value_t) !void {
            const get_or_put_result = try self.policies.getOrPut(state);
            var action_map = get_or_put_result.value_ptr;

            if (!get_or_put_result.found_existing) {
                action_map.* = ActionMap.init(self.allocator);
            }

            try action_map.put(action, q_value);
        }

        pub fn getValue(self: @This(), state: StateType, action: ActionType) ?ztmdp_value_t {
            if (self.policies.get(state)) |action_map| {
                return action_map.get(action);
            } else {
                return null;
            }
        }

        pub fn getBestValue(self: @This(), state: StateType) ?ztmdp_value_t {
            if (self.policies.get(state)) |action_map| {
                if (action_map.count() == 0) return null;

                var max_value: ztmdp_value_t = -std.math.floatMax(ztmdp_value_t);
                var action_iterator = action_map.iterator();
                while (action_iterator.next()) |entry| {
                    if (entry.value_ptr.* > max_value) {
                        max_value = entry.value_ptr.*;
                    }
                }
                return max_value;
            } else {
                return null;
            }
        }

        pub fn getBestAction(self: @This(), state: StateType) ?ActionType {
            if (self.policies.get(state)) |action_map| {
                if (action_map.count() == 0) return null;

                var max_value: ztmdp_value_t = -std.math.floatMax(ztmdp_value_t);
                var best_action: ?ActionType = null;

                var action_iterator = action_map.iterator();
                while (action_iterator.next()) |entry| {
                    if (entry.value_ptr.* > max_value) {
                        max_value = entry.value_ptr.*;
                        best_action = entry.key_ptr.*;
                    }
                }
                return best_action;
            } else {
                return null;
            }
        }

        pub fn getActions(self: @This(), state: StateType) !std.array_list.Managed(ActionType) {
            var actions = std.array_list.Managed(ActionType).init(self.allocator);
            if (self.policies.get(state)) |action_map| {
                var action_iterator = action_map.iterator();
                while (action_iterator.next()) |entry| {
                    try actions.append(entry.key_ptr.*);
                }
            }
            return actions;
        }

        pub fn toText(self: @This(), file: *std.fs.File) !void {
            const context = FileWriteContext{ .file = file };
            const generic_writer = std.io.GenericWriter(FileWriteContext, anyerror, fileWriteFn){ .context = context };

            var state_iterator = self.policies.iterator();
            while (state_iterator.next()) |state_entry| {
                var action_iterator = state_entry.value_ptr.*.iterator();
                while (action_iterator.next()) |action_entry| {
                    try std.fmt.format(generic_writer, "{d},{d},{d};{d};{d}\n", .{
                        state_entry.key_ptr.*.trait_data.x,
                        state_entry.key_ptr.*.trait_data.y,
                        state_entry.key_ptr.*.reward,
                        action_entry.key_ptr.*.trait_data.dir,
                        action_entry.value_ptr.*,
                    });
                }
            }
        }

        pub fn fromText(allocator: Allocator, file: *std.fs.File, comptime GridType: type, comptime DirectionType: type) !@This() {
            var policy = @This().init(allocator);
            var line_buf = std.array_list.Managed(u8).init(allocator);
            defer line_buf.deinit();

            const context = FileReadContext{ .file = file };
            var generic_reader = std.io.GenericReader(FileReadContext, anyerror, fileReadFn){ .context = context };

            while (true) {
                line_buf.clearRetainingCapacity(); // Clear for the new line
                var byte_buf: [1]u8 = undefined;
                var eof = false;
                while (true) {
                    const bytes_read = generic_reader.read(&byte_buf) catch |err| switch (err) {
                        error.EndOfStream => {
                            eof = true;
                            break;
                        },
                        else => return err,
                    };
                    if (bytes_read == 0) {
                        eof = true;
                        break;
                    }
                    const byte = byte_buf[0];
                    if (byte == '\n') {
                        break;
                    }
                    try line_buf.append(byte);
                }

                if (line_buf.items.len == 0 and eof) {
                    break; // End of file and no more data
                }

                const line_slice = line_buf.items;
                if (line_slice.len == 0) continue;

                var tokenizer = std.mem.splitScalar(u8, line_slice, ';');
                const state_part = tokenizer.next().?;
                const action_part = tokenizer.next().?;
                const value_part = tokenizer.next().?;

                var state_tokenizer = std.mem.splitScalar(u8, state_part, ',');
                const state_x = try std.fmt.parseInt(u32, state_tokenizer.next().?, 10);
                const state_y = try std.fmt.parseInt(u32, state_tokenizer.next().?, 10);
                const state_reward = try std.fmt.parseFloat(ztmdp_value_t, state_tokenizer.next().?);

                const action_dir = try std.fmt.parseInt(u32, action_part, 10);
                const q_value = try std.fmt.parseFloat(ztmdp_value_t, value_part);

                const state_trait = GridType{ .x = state_x, .y = state_y, .reward = state_reward, .occupied = false }; // Assuming occupied is always false for deserialization
                const state = StateType.init(state_trait, state_reward);
                const action_trait = DirectionType{ .dir = action_dir };
                const action = ActionType.init(action_trait);

                try policy.update(state, action, q_value);
            }
            return policy;
        }
    };
}

pub fn MarkovChain(comptime LinkType: type) type {
    return struct {
        allocator: Allocator,
        links: std.array_list.Managed(LinkType),

        pub fn init(allocator: Allocator) @This() {
            return .{
                .allocator = allocator,
                .links = std.array_list.Managed(LinkType).init(allocator),
            };
        }

        pub fn deinit(self: *@This()) void {
            self.links.deinit();
        }

        pub fn addLink(self: *@This(), link: LinkType) !void {
            try self.links.append(link);
        }

        pub fn getLink(self: @This(), index: usize) ?LinkType {
            if (index < self.links.items.len) {
                return self.links.items[index];
            } else {
                return null;
            }
        }

        pub fn size(self: @This()) usize {
            return self.links.items.len;
        }
    };
}

pub fn QLearning(comptime PolicyType: type, comptime MarkovChainType: type) type {
    return struct {
        alpha: ztmdp_value_t,
        gamma: ztmdp_value_t,

        pub fn init(alpha: ztmdp_value_t, gamma: ztmdp_value_t) @This() {
            return .{
                .alpha = alpha,
                .gamma = gamma,
            };
        }

        pub fn qValue(self: @This(), episode: MarkovChainType, index: usize, policy_map: *PolicyType) !ztmdp_value_t {
            const step = episode.getLink(index) orelse return error.InvalidIndex;

            const q_current: ztmdp_value_t = policy_map.getValue(step.state, step.action) orelse 0.0;

            var updated_q_value: ztmdp_value_t = undefined;

            if (index < episode.size() - 1) {
                const next_step = episode.getLink(index + 1) orelse return error.InvalidIndex;

                const q_next_best = policy_map.getBestValue(next_step.state) orelse 0.0;

                const reward = step.state.reward;

                updated_q_value = q_current + self.alpha * (reward + (self.gamma * q_next_best) - q_current);
            } else {
                // Terminal state
                updated_q_value = step.state.reward;
            }

            return updated_q_value;
        }

        pub fn processEpisode(self: @This(), episode: MarkovChainType, policy_map: *PolicyType) !void {
            for (0..episode.size()) |i| {
                const updated_q_value = try self.qValue(episode, i, policy_map);
                const step = episode.getLink(i) orelse return error.InvalidIndex;
                try policy_map.update(step.state, step.action, updated_q_value);
            }
        }
    };
}

pub fn QProbabilistic(comptime PolicyType: type, comptime MarkovChainType: type) type {
    const StateType = PolicyType.StateType; // Extract StateType from PolicyType
    const ActionType = PolicyType.ActionType; // Extract ActionType from PolicyType

    const StateFreqMap = std.HashMap(StateType, usize, StateContext(StateType), std.hash_map.default_max_load_percentage);
    const ActionObservationMap = std.HashMap(ActionType, StateFreqMap, ActionContext(ActionType), std.hash_map.default_max_load_percentage);
    const ObservationMemory = std.HashMap(StateType, ActionObservationMap, StateContext(StateType), std.hash_map.default_max_load_percentage);

    return struct {
        allocator: Allocator,
        gamma: ztmdp_value_t,
        memory: ObservationMemory,

        pub fn init(allocator: Allocator, gamma: ztmdp_value_t) @This() {
            return .{
                .allocator = allocator,
                .gamma = gamma,
                .memory = ObservationMemory.init(allocator),
            };
        }

        pub fn deinit(self: *@This()) void {
            var state_iterator = self.memory.iterator();
            while (state_iterator.next()) |state_entry| {
                var action_iterator = state_entry.value.iterator();
                while (action_iterator.next()) |action_entry| {
                    action_entry.value.deinit(); // Deinit inner StateFreqMap
                }
                state_entry.value.deinit(); // Deinit ActionObservationMap
            }
            self.memory.deinit();
        }

        pub fn qValue(self: @This(), episode: MarkovChainType, index: usize, policy_map: *PolicyType) !ztmdp_value_t {
            const step = episode.getLink(index) orelse return error.InvalidIndex;

            if (index == episode.size() - 1) {
                // Terminal state
                return step.state.reward;
            }

            const next_step = episode.getLink(index + 1) orelse return error.InvalidIndex;
            const q_next_best = policy_map.getBestValue(next_step.state) orelse 0.0;
            const reward = step.state.reward;

            // Accessing transition frequencies from memory
            const action_obs_map = self.memory.get(step.state) orelse return reward; // Default to immediate reward if no observations for this state
            const state_freq_map = action_obs_map.get(step.action) orelse return reward; // Default to immediate reward if no observations for this state-action pair
            const frequency_ptr = state_freq_map.get(next_step.state) orelse return reward; // Default to immediate reward if no observation for this state-action-next_state transition

            const total_observations_for_state_action = state_freq_map.count();
            const prob = @as(ztmdp_value_t, @floatFromInt(frequency_ptr.*)) / @as(ztmdp_value_t, @floatFromInt(total_observations_for_state_action));

            const r_expected = prob * reward;

            return r_expected + (self.gamma * (q_next_best * prob));
        }

        pub fn processEpisode(self: *@This(), episode: MarkovChainType, policy_map: *PolicyType) !void {
            // First pass: Update transition frequencies in memory
            for (0..episode.size()) |i| {
                if (i == episode.size() - 1) break; // Skip terminal state

                const current_step = episode.getLink(i) orelse return error.InvalidIndex;
                const next_step = episode.getLink(i + 1) orelse return error.InvalidIndex;

                var action_obs_map = (try self.memory.getOrPut(current_step.state)).value_ptr;
                if (!action_obs_map.found) {
                    action_obs_map.* = ActionObservationMap.init(self.allocator);
                }

                var state_freq_map = (try action_obs_map.getOrPut(current_step.action)).value_ptr;
                if (!state_freq_map.found) {
                    state_freq_map.* = StateFreqMap.init(self.allocator);
                }

                const frequency = (try state_freq_map.getOrPut(next_step.state)).value_ptr;
                frequency.* += 1;
            }

            // Second pass: Update policy map using calculated Q-values
            for (0..episode.size()) |i| {
                const updated_q_value = try self.qValue(episode, i, policy_map);
                const step = episode.getLink(i) orelse return error.InvalidIndex;
                try policy_map.update(step.state, step.action, updated_q_value);
            }
        }
    };
}

test "ztmdp basic types and policy" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) {
            std.log.err("Memory leak detected in test!", .{});
        }
    }
    const allocator = gpa.allocator();

    // Dummy Trait and TraitConcept for testing
    const IntTrait = u32;
    const IntTraitConcept = TraitConcept(IntTrait){
        .hash = struct {
            fn hashFn(trait: IntTrait) u64 {
                var hasher = std.hash.Fnv1a64.init();
                hasher.update(std.mem.asBytes(&trait));
                return hasher.final();
            }
        }.hashFn,
        .equal = std.math.eql,
        .less = std.math.lt,
    };

    const MyState = State(IntTrait, IntTraitConcept);
    const MyAction = Action(IntTrait, IntTraitConcept);
    const MyLink = Link(MyState, MyAction);
    const MyPolicy = Policy(MyState, MyAction);
    const MyMarkovChain = MarkovChain(MyLink);
    const MyQLearning = QLearning(MyPolicy, MyMarkovChain);
    const MyQProbabilistic = QProbabilistic(MyPolicy, MyMarkovChain);

    // Test State
    const state1 = MyState.init(1, 0.0);
    const state2 = MyState.init(2, 1.0);
    const state3 = MyState.init(1, 0.5);

    try std.testing.expect(state1.equals(state3));
    try std.testing.expect(!state1.equals(state2));
    try std.testing.expect(state1.hash() == state3.hash());
    try std.testing.expect(state1.less(state2));
    var state1_mut = state1; // Need mutable copy to set reward
    state1_mut.setReward(10.0);
    try std.testing.expect(state1_mut.reward == 10.0);

    // Test Action
    const action1 = MyAction.init(10);
    const action2 = MyAction.init(20);
    const action3 = MyAction.init(10);

    try std.testing.expect(action1.equals(action3));
    try std.testing.expect(!action1.equals(action2));
    try std.testing.expect(action1.hash() == action3.hash());
    try std.testing.expect(action1.less(action2));

    // Test Link
    const link1 = MyLink.init(state1, action1);
    const link2 = MyLink.init(state2, action2);
    const link3 = MyLink.init(state1, action1);

    try std.testing.expect(link1.equals(link3));
    try std.testing.expect(!link1.equals(link2));
    try std.testing.expect(link1.less(link2));

    // Test Policy
    var policy = MyPolicy.init(allocator);
    defer policy.deinit();

    try policy.update(state1, action1, 0.5);
    try policy.update(state1, action2, 0.8);
    try policy.update(state2, action1, 0.2);

    try std.testing.expectEqual(0.5, policy.getValue(state1, action1).?);
    try std.testing.expectEqual(0.8, policy.getValue(state1, action2).?);
    try std.testing.expectEqual(0.2, policy.getValue(state2, action1).?);
    try std.testing.expect(policy.getValue(state2, action2) == null);

    try std.testing.expectEqual(0.8, policy.getBestValue(state1).?);
    try std.testing.expect(policy.getBestAction(state1).?.equals(action2));

    const actions_for_state1 = try policy.getActions(state1);
    defer actions_for_state1.deinit();
    try std.testing.expectEqual(2, actions_for_state1.items.len);
    try std.testing.expect(actions_for_state1.items[0].equals(action1) or actions_for_state1.items[0].equals(action2));
    try std.testing.expect(actions_for_state1.items[1].equals(action1) or actions_for_state1.items[1].equals(action2));

    // Test MarkovChain
    var episode = MyMarkovChain.init(allocator);
    defer episode.deinit();

    try episode.addLink(link1);
    try episode.addLink(link2);

    try std.testing.expectEqual(2, episode.size());
    try std.testing.expect(episode.getLink(0).?.equals(link1));
    try std.testing.expect(episode.getLink(1).?.equals(link2));
    try std.testing.expect(episode.getLink(2) == null);

    // Test QLearning (simple scenario)
    const q_learning = MyQLearning.init(0.1, 0.9);
    var policy_q = MyPolicy.init(allocator);
    defer policy_q.deinit();

    const state_q1 = MyState.init(1, 0.0);
    const state_q2 = MyState.init(2, 10.0); // Terminal state with reward
    const action_q1 = MyAction.init(100);

    const link_q1 = MyLink.init(state_q1, action_q1);
    const link_q2 = MyLink.init(state_q2, action_q1);

    var episode_q = MyMarkovChain.init(allocator);
    defer episode_q.deinit();
    try episode_q.addLink(link_q1);
    try episode_q.addLink(link_q2);

    try q_learning.processEpisode(episode_q, &policy_q);
    const q_value_q1 = policy_q.getValue(state_q1, action_q1).?;
    // Expected Q-value for state_q1, action_q1: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
    // Initial Q(s,a) = 0.0
    // reward = 0.0 (from state_q1)
    // max(Q(s',a')) = 10.0 (from state_q2, terminal reward)
    // updated_q_value = 0.0 + 0.1 * (0.0 + (0.9 * 10.0) - 0.0) = 0.1 * 9.0 = 0.9
    try std.testing.expectApproxEqAbs(0.9, q_value_q1, 0.0001);

    // Test QProbabilistic (simple scenario)
    var q_probabilistic = MyQProbabilistic.init(allocator, 0.9);
    defer q_probabilistic.deinit();
    var policy_qp = MyPolicy.init(allocator);
    defer policy_qp.deinit();

    const state_qp1 = MyState.init(1, 0.0);
    const state_qp2 = MyState.init(2, 10.0);
    const state_qp3 = MyState.init(3, 5.0);
    const action_qp1 = MyAction.init(100);
    const action_qp2 = MyAction.init(200);

    const link_qp1 = MyLink.init(state_qp1, action_qp1);
    const link_qp2 = MyLink.init(state_qp2, action_qp1);
    const link_qp3 = MyLink.init(state_qp1, action_qp2);
    const link_qp4 = MyLink.init(state_qp3, action_qp2);

    var episode_qp1 = MyMarkovChain.init(allocator);
    defer episode_qp1.deinit();
    try episode_qp1.addLink(link_qp1);
    try episode_qp1.addLink(link_qp2);

    var episode_qp2 = MyMarkovChain.init(allocator);
    defer episode_qp2.deinit();
    try episode_qp2.addLink(link_qp3);
    try episode_qp2.addLink(link_qp4);

    // Process first episode to build memory
    try q_probabilistic.processEpisode(episode_qp1, &policy_qp);
    // Process second episode to build memory and update policy
    try q_probabilistic.processEpisode(episode_qp2, &policy_qp);

    // After processing two episodes:
    // (state_qp1, action_qp1) -> state_qp2 (frequency 1)
    // (state_qp1, action_qp2) -> state_qp3 (frequency 1)

    // For (state_qp1, action_qp1):
    // prob = 1/1 = 1.0
    // r_expected = 1.0 * state_qp1.reward (0.0) = 0.0
    // q_next_best (for state_qp2) = 10.0
    // updated_q_value = 0.0 + (0.9 * (10.0 * 1.0)) = 9.0
    const q_value_qp1 = policy_qp.getValue(state_qp1, action_qp1).?;
    try std.testing.expectApproxEqAbs(9.0, q_value_qp1, 0.0001);

    // For (state_qp1, action_qp2):
    // prob = 1/1 = 1.0
    // r_expected = 1.0 * state_qp1.reward (0.0) = 0.0
    // q_next_best (for state_qp3) = 5.0
    // updated_q_value = 0.0 + (0.9 * (5.0 * 1.0)) = 4.5
    const q_value_qp2 = policy_qp.getValue(state_qp1, action_qp2).?;
    try std.testing.expectApproxEqAbs(4.5, q_value_qp2, 0.0001);

    // Best action for state_qp1 should be action_qp1
    try std.testing.expect(policy_qp.getBestAction(state_qp1).?.equals(action_qp1));
}
