import numpy as np
import matplotlib.pyplot as plt


def get_entropy(labels):
    unique_labels = np.unique(labels)
    entropy = 0
    for label in unique_labels:
        p = np.sum(labels == label) / len(labels)
        entropy -= p * np.log2(p)
    return entropy


def get_best_emitter(dataset):
    features, labels = dataset[:, :-1], dataset[:, -1]
    total_entropy = calculate_entropy(labels)
    max_IG = float('-inf')
    max_IG_feature = None

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
    features, labels = dataset[:, max_IG_col], dataset[:, -1]
    total_entropy = calculate_entropy(labels)

    max_IG_for_value = float('-inf')
    best_threshold, prev_value = None, None

    for value in sorted(np.unique(features)):
        current_threshold = value if prev_value is None else (value + prev_value) / 2
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

def get_labelled_col(dataset, column):
    strengths_set = np.array([])
    strengths_labels = np.array([])
    strengths_set = dataset[:, column]
    strengths_labels = dataset[:, -1]

    strengths = np.column_stack((strengths_set, strengths_labels))

    sorted_indices = np.argsort(strengths[:, 0])
    sorted_strengths = strengths[sorted_indices]

    return sorted_strengths

def get_IG_from_split(old_set, a, b):
    return get_entropy(old_set) - ((get_entropy(a[:, 1]) * a.shape[0] + get_entropy(b[:, 1]) * b.shape[0]) / a.shape[0])


def find_split(dataset):
    #print("called split")
    best_emitter = -1
    best_IG = -1
    best_threshold = -10000

    col_n = 0
    equal_either_side_flag = False
    for n in range(0, dataset.shape[1]-1):
        sorted_labelled = get_labelled_col(dataset, col_n)
        #col_entropy = get_entropy(sorted_labelled[:, -1])
        row_n = 0
        previous_row = []
        for row_n in range(len(sorted_labelled) - 1):  # Loop to length - 1 so we don't exceed the bounds
            current_row = sorted_labelled[row_n]
            next_row = sorted_labelled[row_n + 1]

            if current_row[-1] != next_row[-1]:
                #print("found a change")

                if row_n == 0:  # Special case for the first row
                    above = sorted_labelled[:1]
                    below = sorted_labelled[1:]
                else:
                    above = sorted_labelled[:row_n + 1]
                    below = sorted_labelled[row_n + 1:]

                split_IG = get_IG_from_split(dataset[:, col_n], above, below)
                if split_IG > best_IG:
                    best_IG = split_IG
                    best_emitter = col_n
                    print("setting threshold to the mean of ", current_row, " and ", next_row)
                    best_threshold = (current_row[0] + next_row[0]) / 2
                    if current_row[0] == next_row[0]:
                        equal_either_side_flag = True
            row_n += 1        
        col_n += 1
    lpy = []
    rpy = []
    #print("Dataset before split:", dataset)
    is_upper_of_issue = False
    for row in dataset:
        #print("Value in best emitter:", row[best_emitter], "Threshold:", best_threshold)

        if row[best_emitter] < best_threshold:
            lpy.append(row)
        elif row[best_emitter] == best_threshold and equal_either_side_flag:
            if is_upper_of_issue:
                lpy.append(row)
            else:
                rpy.append(row)
            is_upper_of_issue = not is_upper_of_issue  # Toggle the flag
        else:
            rpy.append(row)
    
    l = np.asarray(lpy)
    r = np.asarray(rpy)

    #print("Best emitter:", best_emitter)
    #print("Best threshold:", best_threshold)

    return best_threshold, best_emitter, l, r

def decision_tree_learning(training_dataset, depth):
    if len(np.unique(training_dataset[:, -1])) == 1:
        label_value = int(training_dataset[0, -1])
        return {f'R {label_value}': [None, label_value, None, None]}, depth

    best_threshold, best_emitter, l_dataset, r_dataset = find_split(training_dataset)

    newVal = f"X{best_emitter} < {best_threshold}"

    tmpTree = {newVal: [best_emitter, best_threshold, "l_branch", "r_branch"]}

    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

    tmpTree[newVal][2], tmpTree[newVal][3] = next(iter(l_branch)), next(iter(r_branch))
    tmpTree.update(l_branch)
    tmpTree.update(r_branch)
    
    return tmpTree, max(l_depth, r_depth)


def plot_decision_tree(tree, node, x, y, dx, dy, depth=0):
    if node in tree:
        feature, threshold, left, right = tree[node]
        boxprops = dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.3')

        plt.text(x, y, f"R{threshold}" if left is None and right is None else f"E{feature} > \n{threshold:.2f}",
                 fontsize=8, ha='center', va='center', bbox=boxprops)

        if left in tree:
            plot_decision_tree(tree, left, x - dx / (2**depth), y - dy, dx, dy, depth + 1)
            plt.plot([x, x - dx / (2**depth)], [y, y - dy], color='black')

        if right in tree:
            plot_decision_tree(tree, right, x + dx / (2**depth), y - dy, dx, dy, depth + 1)
            plt.plot([x, x + dx / (2**depth)], [y, y - dy], color='black')




dataset = np.loadtxt('wifi_db/clean_dataset.txt')

tmp_dictionary, _ = decision_tree_learning(dataset, 0)

node_dictionary = {}
node_dictionary.update(tmp_dictionary)

plt.figure(figsize=(20, 10))
plot_decision_tree(node_dictionary, next(iter(node_dictionary)), x=0, y=0, dx=20, dy=5, depth=0)
plt.tight_layout()
plt.show()