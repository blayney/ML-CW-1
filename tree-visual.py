# %%
import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with your data)
X = np.array([[2, 3], [2, 2], [1, 5], [4, 1], [3, 3]])
y = np.array([0, 1, 1, 0, 1])

# Define a decision tree manually (you should build your tree model)
node_dictionary = {
    "head" : ["attribute", "value", "l_branch", "r_branch"]
}

# Create a simple decision tree (replace with your own tree)
decision_tree = {
    "0" : ["x0", "20.3", "1", "2"],
    "1" : ["x1", "35.0", "3", "4"],
    "2" : ["x2", "37.6", "5", "6"],
    "3" : ["x3", "41.0", "leaf1", "leaf3"],
    "4" : ["x4", "45.4", "leaf4", "leaf2"],
    "5" : ["x5", "56.8", "leaf1", "leaf4"],
    "6" : ["x6", "73.2", "leaf2", "leaf3"],
    "leaf1" : ["leaf", "1", None, None],
    "leaf2" : ["leaf", "2", None, None],
    "leaf3" : ["leaf", "3", None, None],
    "leaf4" : ["leaf", "4", None, None],
    
}
# dx and dy spacing 
def plot_decision_tree(tree, node, x, y, dx, dy, depth):
    if node in tree:
        feature, threshold, left, right = tree[node]
        if left is None and right is None:
            plt.text(x, y, f"leaf {threshold}\nDepth: {depth}", fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
        else:
            plt.text(x, y, f"{feature} <= {threshold}", fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
        if left in tree:
            # Adjust the vertical positions for boxes and branches
            x_left = x - dx
            y_left = y - dy
            plt.plot([x, x_left], [y, y - dy], color='black')
            plot_decision_tree(tree, left, x_left, y_left, dx / 2, dy, depth + 1)
        if right in tree:
            # Adjust the vertical positions for boxes and branches
            x_right = x + dx
            y_right = y - dy
            plt.plot([x, x_right], [y, y - dy], color='black')
            plot_decision_tree(tree, right, x_right, y_right, dx / 2, dy, depth + 1)
    else:
        plt.text(x, y, f"leaf {threshold}\nDepth: {depth}", fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightgreen'))

plt.figure(figsize=(12, 8))
plot_decision_tree(decision_tree, "0", x=0, y=0, dx=10, dy=5, depth=0)
plt.axis('off')
plt.show()

# %%
