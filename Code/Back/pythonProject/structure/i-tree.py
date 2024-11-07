import json
import hyperplane_set

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.utilityR = None
        self.left = None
        self.right = None

# 生成二叉树
def generate_tree(depth, current_value=1):
    if depth == 0:
        return None
    root = TreeNode(current_value)
    root.left = generate_tree(depth - 1, current_value * 2)
    root.right = generate_tree(depth - 1, current_value * 2 + 1)
    return root

# 将二叉树转为字典，便于序列化
def tree_to_dict(node):
    if node is None:
        return None
    return {
        'value': node.value,
        'left': tree_to_dict(node.left),
        'right': tree_to_dict(node.right)
    }

# 存储二叉树
def store_tree(tree, filename='binary_tree.json'):
    with open(filename, 'w') as file:
        json.dump(tree_to_dict(tree), file)

# 读取二叉树
def load_tree(filename='binary_tree.json'):
    with open(filename, 'r') as file:
        return json.load(file)

# 生成并存储树
depth = 3
root = generate_tree(depth)
tree_dict = tree_to_dict(root)
store_tree(tree_dict)