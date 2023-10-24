import numpy as np
import matplotlib.pyplot as plt

node_dictionary = {
}

# Load our clean dataset
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')

# Load our noisy dataset
noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

# Splitting our clean dataset into features and labels
clean_features = clean_data[:, :-1]
clean_labels = clean_data[:, -1].astype(int)

# def decision_tree_learning():
print(noisy_data.shape)
print(clean_data.shape)
print(clean_features.shape)
print(clean_labels.shape)

print(node_dictionary)

dataset = np.loadtxt('wifi_db/clean_dataset.txt')

def calculate_entropy(labels):
    unique_labels = np.unique(labels)
    entropy = 0
    for label in unique_labels:
        p = np.sum(labels == label) / len(labels)
        entropy -= p * np.log2(p)
    return entropy

def get_max_IG_emitter(dataset):

    features = dataset[:, :-1]
    labels = dataset[:, -1]
    
    total_entropy = calculate_entropy(labels)
    
    max_IG = -99999.999 
    max_IG_feature = -999  
    
    for col in range(features.shape[1]):
        unique_values = np.unique(features[:, col])
        weighted_entropy = 0
        
        for val in unique_values:
            subset_labels = labels[features[:, col] == val]
            p_val = len(subset_labels) / len(labels)
            weighted_entropy += p_val * calculate_entropy(subset_labels)
        
        IG = total_entropy - weighted_entropy
        
        if IG > max_IG:
            max_IG = IG
            max_IG_feature = col

    return max_IG_feature, max_IG

def get_threshold_value(dataset, max_IG_col):
    features = dataset[:, max_IG_col]
    labels = dataset[:, -1]

    total_entropy = calculate_entropy(labels)

    max_IG_for_value = -np.inf
    best_threshold = None
    prev_value = None

    for value in np.unique(features):

        if (prev_value != None):
            current_threshold = (value + prev_value) / 2
        else:
            current_threshold = value
        
        labels_below_threshold = labels[features <= current_threshold]
        labels_above_threshold = labels[features > current_threshold]
        
        p_below = len(labels_below_threshold) / len(labels)
        p_above = len(labels_above_threshold) / len(labels)

        weighted_entropy = (p_below * calculate_entropy(labels_below_threshold) +
                            p_above * calculate_entropy(labels_above_threshold))

        IG_for_value = total_entropy - weighted_entropy

        if IG_for_value > max_IG_for_value:
            max_IG_for_value = IG_for_value
            best_threshold = current_threshold

        prev_value = value  

    return best_threshold, max_IG_for_value

def find_split(dataset):
    # Assuming you have the following two functions defined elsewhere
    max_IG_col, max_IG_of_emitter = get_max_IG_emitter(dataset)
    threshold, max_IG_for_value = get_threshold_value(dataset, max_IG_col)

    # Sorting the dataset by the column max_IG_col using argsort
    sorted_indices = np.argsort(dataset[:, max_IG_col])
    sorted_dataset = dataset[sorted_indices]

    # Splitting the dataset into two parts based on the threshold
    split_index = np.searchsorted(sorted_dataset[:, max_IG_col], threshold, side='right')
    splitR = sorted_dataset[:split_index]
    splitL = sorted_dataset[split_index:]

    print("Threshold Value: ", threshold)
    print("Threshold IG: ", max_IG_for_value)
    print("Emitter Number: ", max_IG_col)
    print("Emitter IG: ", max_IG_of_emitter)

    return threshold, max_IG_col, splitL, splitR




def decision_tree_learning(training_dataset, depth):
    print(training_dataset)
    flag = False
    value = training_dataset[0, -1]
    for item in training_dataset:
        if item[-1] != value:
            flag = True
            break
    if flag != True:
        tmp = "leaf" + str(depth)
        return {tmp : ["leaf", depth, None, None]}, depth
    else:
        split, attributeSplit, l_dataset, r_dataset = find_split(training_dataset)
        newVal = "X" + str(attributeSplit) +  " < " + str(split)
        tmpTree = {newVal : [attributeSplit, split, "l_branch", "r_branch"] }
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        tmpKey = tmpTree[newVal]
        tmpKey[2] = next(iter(l_branch))
        tmpTree[newVal] = tmpKey
        tmpTree.update(l_branch)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
        tmpKey = tmpTree[newVal]
        tmpKey[3] = next(iter(r_branch))
        tmpTree[newVal] = tmpKey
        tmpTree.update(r_branch)
        return tmpTree, max(l_depth, r_depth)

tmp_dictionary, _ = decision_tree_learning(clean_data, 0)
node_dictionary.update(tmp_dictionary)
print(node_dictionary)

# Original one by Jackson
# -----------------------
# def plot_decision_tree(tree, node, x, y, dx, dy, depth):
#     if node in tree:
#         feature, threshold, left, right = tree[node]
#         if left is None and right is None:
#             plt.text(x, y, f"Leaf {threshold}", fontsize=5, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
#         else:
#             plt.text(x, y, f"X{feature} <= {threshold}", fontsize=5, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
        
#         if left in tree:
#             x_left = x - dx * (1 + depth)
#             y_left = y - dy
#             plt.plot([x, x_left], [y, y - dy], color='black')
#             plot_decision_tree(tree, left, x_left, y_left, dx / 2, dy, depth + 1)
        
#         if right in tree:
#             x_right = x + dx * (1 + depth)
#             y_right = y - dy
#             plt.plot([x, x_right], [y, y - dy], color='black')
#             plot_decision_tree(tree, right, x_right, y_right, dx / 2, dy, depth + 1)


# #Updated one by Alvi
# #--------------------
# ...

# Updated tree plotting function
def plot_decision_tree(tree, node, x, y, dx, dy, depth=0):
    if node in tree:
        feature, threshold, left, right = tree[node]
        boxprops = dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.3')  # Adjusted node appearance
        
        if left is None and right is None:
            plt.text(x, y, f"Leaf\n{threshold}", fontsize=8, ha='center', va='center', bbox=boxprops)
        else:
            plt.text(x, y, f"E{feature}\n<= {threshold:.2f}", fontsize=8, ha='center', va='center', bbox=boxprops)  # Adjusted clarity of text
        
        if left in tree:
            x_left = x - dx / (2**depth)
            y_left = y - dy
            plt.plot([x, x_left], [y, y_left], color='black')
            plot_decision_tree(tree, left, x_left, y_left, dx, dy, depth + 1)
        
        if right in tree:
            x_right = x + dx / (2**depth)
            y_right = y - dy
            plt.plot([x, x_right], [y, y_right], color='black')
            plot_decision_tree(tree, right, x_right, y_right, dx, dy, depth + 1)

plt.figure(figsize=(20, 10))  # Adjusted figure size
plot_decision_tree(node_dictionary, next(iter(node_dictionary)), x=0, y=0, dx=20, dy=5, depth=0)  # Adjusted dx and dy for spacing
plt.axis('off')
plt.tight_layout()
plt.show()
