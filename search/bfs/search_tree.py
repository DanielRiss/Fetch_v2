class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.value = value
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf

    def get_depth(self):
        return len(self.return_path()) + 1

    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content]
        return self.parent.return_path() + [self.content]

    def print_path(self):
        return "\n".join(self.return_path())

class Tree:
    def __init__(self, question, answer, additional_info):
        self.question = question
        self.answer = answer # provided, but will not used when searching
        self.all_nodes = []
        self.root = None # wait init
        self.additional_info = additional_info

    def init_root_node(self, value):
        self.root = Node(None, value, None, 0, self)
        self.all_nodes.append(self.root)

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def add_node(self, content, value, parent, timestep, is_leaf=False):
        node = Node(content, value, parent, timestep, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def select_next_node(self, timestep):
        # latest timestep and highest value
        available_nodes = [node for node in self.all_nodes if node.timestep == timestep]
        if len(available_nodes) == 0:
            return None # end previously
            # raise Exception("No node with timestep {}".format(timestep))
        selected_node = max(available_nodes, key=lambda x: x.value)
        if selected_node.is_leaf: # search done
            return None
        return selected_node

    def select_best_node(self):
        available_nodes = [node for node in self.all_nodes if len(node.children) < 1] # have not been explored
        if len(available_nodes) == 0:
            return None # end previously
            # raise Exception("No available node")
        best_node = max(available_nodes, key=lambda x: x.value)
        if best_node.is_leaf: # search done
            return None
        return best_node

    def return_best_path(self, use_greedy=True):
        leaf_nodes = [node for node in self.all_nodes if node.is_leaf]
        if leaf_nodes:
            state = max(leaf_nodes, key=lambda x: x.value)
            return state.print_path(), state.value
        return None, None