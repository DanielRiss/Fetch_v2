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


class VirtualMCTSNode:
    def __init__(self, sub_nodes, virtual_parent):
        self.parent = virtual_parent
        self.sub_nodes = sorted(sub_nodes, key=lambda x: x.value, reverse=True)
        self.actions = []
        self.value = sub_nodes[0].value # max
        self.rewards = []
        self.is_leaf = self.sub_nodes[0].is_leaf
        self.is_expanded = False
        self.content = [node.content for node in sub_nodes]
        self.p = 1

    def get_depth(self):
        return self.sub_nodes[0].get_depth()

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
        self.all_virtual_nodes = []
        self.config = config
        self.root = self.init_root_node() # wait init

    def init_root_node(self):
        root = MCTSNode(None, None, 0, False, p=1)
        root.tree = self
        self.all_nodes.append(root)
        virtual_root = VirtualMCTSNode([root], None)
        self.all_virtual_nodes.append(virtual_root)
        return virtual_root

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
        3) merge redundant node
        4) return the node for simulation
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
        # for more vote, assign more budget, this step is optional. if the expansion budget is enough, we can just comment this line
        node_budget = node_budget + round(min(len(node.sub_nodes) * 0.5, node_budget))
        to_end = node.get_depth() >= self.config.max_depth or node.q() >= self.config.conf_high_value
        # "_" denote concrete node or action
        _actions = []
        while len(_actions) < node_budget:
            _node = node.sub_nodes[len(_actions) % len(node.sub_nodes)] # seq find next node
            _action = self.config.get_next_step(self.question, _node.return_path(), False) if not to_end else self.config.get_full_traj(self.question, _node.return_path(), False)
            _new_child_node = MCTSNode(_action, _node, timestep, is_leaf = self.config.is_terminal(_action), p = self.config.prior(_action) if not to_end else (1 if self.config.is_terminal(_action) else 0))
            if _new_child_node.p > 0:
                _actions.append(_new_child_node)
            self.all_nodes.append(_new_child_node)
            if DEBUG:
                print("Node Content (Generate):", (_new_child_node.content,))
        
        # merge
        for _action in _actions:
            _action.value = self.config.get_value(self.question, _action.return_path())
        texts = [_action.content for _action in _actions]
        labels = self.config.cluster(texts)
        clusters = {}
        for _action, label in zip(_actions, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(_action)
        node.actions = []
        for label, _actions_ in clusters.items():
            action = VirtualMCTSNode(_actions_, node)
            # reassignment of p, give more weight to the most popular cluster
            action.p = 0.8 + 0.2 * len(_actions_) / len(_actions) # basic acc
            node.actions.append(action)
        if DEBUG:
            print("Merge, Keep", len(node.actions))
            print("Cluster:", [(action.content, action.value) for action in node.actions])
        
        # node.actions = sorted(node.actions, key=lambda x: len(x.sub_nodes), reverse=True)
        node.actions[0].is_expanded = True
        if DEBUG:
            print("Node Content (Expand):", (node.actions[0].content,))
        return node.actions[0]

    def mcts_simulate(self, node):
        """
        rollouts
        """
        rollouts = []
        for i in range(self.config.n_rollouts):
            _node = node.sub_nodes[i % len(node.sub_nodes)] # need exp
            rollout = self.config.get_full_traj(self.question, _node.return_path())
            value = self.config.get_value(self.question, _node.return_path() + [rollout])
            rollout = {"text": rollout, "value": value}
            _node.rollouts.append(rollout)
            rollouts.append(rollout)
        return rollouts

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
                if new_node.is_leaf:
                    rollouts = []
                    n_terminals += 1
                else:
                    rollouts = self.mcts_simulate(new_node) if self.config.n_rollouts > 0 else None
                reward = new_node.value * (1 - self.config.alpha) + np.mean([rollout["value"] for rollout in rollouts]) * self.config.alpha if rollouts else new_node.value
                if DEBUG:
                    print("** Simulation **")
                    print("Node Reward:", (new_node.value, reward))
                    print("Rollouts:", [(rollout["text"][:50] + "..." + rollout["text"][-50:] if len(rollout["text"]) > 100 else rollout["text"], rollout["value"]) for rollout in rollouts])
                self.mcts_backpropagate(new_node, reward)
            else: # is terminal
                self.mcts_backpropagate(selected_node, selected_node.rewards[0]) # directly backprop
                n_terminals += 0.1 # trick, may remove in future