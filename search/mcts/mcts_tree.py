import math
import numpy as np

DEBUG = False

class MCTSNode:
    def __init__(self, content, parent, timestep, is_leaf=False, p=1):
        self.content = content
        self.value = -1
        self.rewards = [] # backprob rewards
        self.tree = parent.tree if parent else None
        self.parent = parent
        self.actions = []
        self.rollouts = [] # rollouts traj
        self.timestep = timestep # at which step, this node is expanded
        self.is_leaf = is_leaf # terminate
        self.p = p # probability of this node, currently only use 1 or 0, where 0 meaning too messy, abort the node
        self.is_expanded = False

    def get_depth(self):
        return len(self.return_path()) + 1

    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content]
        return self.parent.return_path() + [self.content]

    def q(self):
        return np.mean(self.rewards) if self.rewards else 0

    def n(self):
        return len(self.rewards) + 1 # to avoid too large impact of diveristy
    
    def calc_ucb(self, c=1.414):
        if self.parent is None: # this actually will not be activate
            return self.q()
        else:
            return self.q() + c * self.p * (math.log(self.parent.n()) / self.n()) ** 0.5 # notice at least 1 term in self.rewards

    def best_child(self, c=1.414):
        return max(self.actions, key=lambda x: x.calc_ucb(c))


class MCTSTree:
    def __init__(self, question, answer, config):
        self.question = question
        self.answer = answer # provided, but will not used when searching
        self.all_nodes = []
        self.config = config
        self.root = self.init_root_node() # wait init

    def init_root_node(self):
        root = MCTSNode(None, None, 0, False, p=1)
        root.tree = self
        self.all_nodes.append(root)
        return root

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def mcts_select(self):
        """
        start from root everytime
        """
        if not self.root.actions:
            curr_node, need_expand = self.root, True
        else:
            curr_node = self.root
            need_expand = False
            while not curr_node.is_leaf:
                need_expand = True if len(curr_node.actions) == 0 else any([not child.is_expanded for child in curr_node.actions]) # has child not been expanded
                if need_expand: # budget not used up
                    break
                else:
                    best_child_node = curr_node.best_child(self.config.c)
                    curr_node = best_child_node
        if DEBUG:
            print("** Selection **")
            print("Node Content:", (curr_node.content, need_expand))
            print("Node Reward:", (curr_node.value, curr_node.rewards))
        return curr_node, need_expand

    def mcts_expand(self, node, timestep):
        """
        expand static number of children. strategy:
        1) use rollout results if possible to save computation
        2) expand util use all budget
        3) return the node for simulation
        """
        if DEBUG:
            print("** Expansion **")
        if node.actions: # we have detect all children, then just expand
            for action in node.actions:
                if not action.is_expanded:
                    if DEBUG:
                        print("Node Content (Expand):", (action.content,))
                    action.is_expanded = True
                    return action
        # if no child, then first get all childrens
        node_budget = self.config.root_budget if node.parent is None else self.config.node_budget
        to_end = node.get_depth() >= self.config.max_depth or node.q() >= self.config.conf_high_value
        actions = []
        while len(actions) < node_budget:
            action = self.config.get_next_step(self.question, node.return_path(), False) if not to_end else self.config.get_full_traj(self.question, node.return_path(), False)
            actions.append(action)
            new_child_node = MCTSNode(action, node, timestep, is_leaf = self.config.is_terminal(action), p = self.config.prior(action) if not to_end else 1)
            self.all_nodes.append(new_child_node)
            node.actions.append(new_child_node)
            if DEBUG:
                print("Node Content (Generate):", (new_child_node.content,))
        
        node.actions[0].is_expanded = True
        if DEBUG:
            print("Node Content (Expand):", (node.actions[0].content,))
        return node.actions[0]

    def mcts_simulate(self, node):
        """
        rollouts
        """
        return [self.config.get_full_traj(self.question, node.return_path()) for _ in range(self.config.n_rollouts - len(node.rollouts))]

    def mcts_backpropagate(self, node, reward):
        curr_node = node
        while True:
            if curr_node is None:
                break
            curr_node.rewards.append(reward)
            curr_node = curr_node.parent

    def run_mcts(self):
        timestep, n_terminals = 0, 0
        while (timestep < self.config.min_search_time or n_terminals < self.config.min_terminals) and timestep < self.config.max_search_time:
            timestep += 1
            if DEBUG:
                print(f"==== TIMESTEP {timestep} ====")
            selected_node, need_expand = self.mcts_select()
            if need_expand:
                new_node = self.mcts_expand(selected_node, timestep)
                if new_node.value < 0:
                    new_node.value = self.config.get_value(self.question, new_node.return_path())
                if new_node.is_leaf:
                    rollouts = []
                    n_terminals += 1
                else:
                    rollouts = self.mcts_simulate(new_node) if self.config.n_rollouts > 0 else None
                    rollouts = self.config.compute_reward(self.question, new_node.return_path(), rollouts)
                    new_node.rollouts += rollouts # rollout with value
                reward = new_node.value * (1 - self.config.alpha) + np.mean([rollout["value"] for rollout in new_node.rollouts]) * self.config.alpha if new_node.rollouts else new_node.value
                if DEBUG:
                    print("** Simulation **")
                    print("Node Reward:", (new_node.value, reward))
                    print("Rollouts:", [(rollout["text"][:50] + "..." + rollout["text"][-50:], rollout["value"]) for rollout in rollouts])
                self.mcts_backpropagate(new_node, reward)
            else: # is terminal
                self.mcts_backpropagate(selected_node, selected_node.rewards[0]) # directly backprop
                n_terminals += 0.1 # trick, may remove in future