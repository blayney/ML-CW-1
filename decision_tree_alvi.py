import numpy as np
import matplotlib.pyplot as plt


def calculate_entropy(labels):
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


def find_split(dataset):
    max_IG_col, _ = get_best_emitter(dataset)
    threshold, _ = get_threshold_value(dataset, max_IG_col)

    sorted_indices = np.argsort(dataset[:, max_IG_col])
    sorted_dataset = dataset[sorted_indices]

    split_index = np.searchsorted(sorted_dataset[:, max_IG_col], threshold, side='right')
    splitL, splitR = sorted_dataset[:split_index], sorted_dataset[split_index:]

    return threshold, max_IG_col, splitL, splitR


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


def predict_room_number(data, tree):

    current_node = next(iter(tree))
    
    while True:
        feature, threshold, left, right = tree[current_node]
        
        if left is None and right is None:
            label = threshold
            return label

        if data[feature] < threshold:
            current_node = left
        else:
            current_node = right


def run_model(dataset, tree):
    
    predicted_labels = [-1] * len(dataset)

    for i in range(len(dataset)):
        predicted_labels[i] = predict_room_number(dataset[i], tree)

    return np.array(predicted_labels)


def compute_accuracy(true_labels, predicted_labels):

    correct_predictions = 0

    print('')
    print('----------------------------Wrong Predictions----------------------------')
    for i in range(len(true_labels)):
      if true_labels[i] == predicted_labels[i]:
        correct_predictions += 1
      else:
        print('Dataset Row: '+ str(i) + '     Actual: ' + str(true_labels[i]) + '     Predicted: ' + str(predicted_labels[i])) #just to see which data went wrong 
    print('---------------------------------------------------------------------------')

    return float(correct_predictions/len(true_labels))


def split_dataset(dataset, train_ratio=0.8):

    np.random.shuffle(dataset)
    
    split_idx = int(len(dataset) * train_ratio)
    
    training_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    return np.array(training_data), np.array(test_data)





#############################################################################################################    

if __name__ == '__main__': 

    full_dataset = np.loadtxt('wifi_db/clean_dataset.txt')

    training_dataset, test_dataset = split_dataset(full_dataset, 0.8)

    tmp_dictionary, _ = decision_tree_learning(training_dataset, 0)

    node_dictionary = {}
    node_dictionary.update(tmp_dictionary)

    predicted_labels = run_model(test_dataset[:,:-1], node_dictionary)
    print('Prediction Accuracy: ', compute_accuracy(test_dataset[:,-1], predicted_labels))

    # plt.figure(figsize=(20, 10))
    # plot_decision_tree(node_dictionary, next(iter(node_dictionary)), x=0, y=0, dx=20, dy=5, depth=0)
    # plt.tight_layout()
    # plt.show()