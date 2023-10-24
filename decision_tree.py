import numpy as np
import matplotlib.pyplot as plt

## For debugging:
with open(r'C:\Users\sohai\Desktop\out_data.txt', 'w') as file:
    pass
##-----------------------------------------------------------------

node_dictionary = {
}

# Load the dataset
dataset = np.loadtxt('wifi_db/clean_dataset.txt')

features = dataset[:, :-1]
labels = dataset[:, -1].astype(int)

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
    splitL = sorted_dataset[:split_index]
    splitR = sorted_dataset[split_index:]

    return threshold, max_IG_col, splitL, splitR




def decision_tree_learning(training_dataset, depth):
    flag = False
    value = training_dataset[0, -1]
    for item in training_dataset:
        if item[-1] != value:
            flag = True
            break
        else:
            flag = False
    if flag == False:
        tmp = 'R ' + str(int(value))
        return {tmp : [None, int(value), None, None]}, depth
    else:
        best_threshold, best_emitter, l_dataset, r_dataset = find_split(training_dataset)

## For debugging:
        with open(r'C:\Users\sohai\Desktop\out_data.txt', 'a') as file:
            file.write('MAIN:' + '\n')
            file.write(str(training_dataset) + '\n\n')
            file.write('E'+str(best_emitter)+' > '+ str(best_threshold)+'\n\n')
            file.write('RIGHT:' + '\n')
            file.write(str(r_dataset) + '\n')
            file.write('LEFT:' + '\n')
            file.write(str(l_dataset) + '\n\n\n')
##-----------------------------------------------------------------

        newVal = "X" + str(best_emitter) +  " < " + str(best_threshold)

        tmpTree = {newVal : [best_emitter, best_threshold, "l_branch", "r_branch"] }

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

tmp_dictionary, _ = decision_tree_learning(dataset, 0)
node_dictionary.update(tmp_dictionary)

def plot_decision_tree(tree, node, x, y, dx, dy, depth=0):
    if node in tree:
        feature, threshold, left, right = tree[node]
        boxprops = dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.3') 
        
        if left is None and right is None:
            plt.text(x, y, f"R{threshold}", fontsize=8, ha='center', va='center', bbox=boxprops)
        else:
            plt.text(x, y, f"E{feature} > \n{threshold:.2f}", fontsize=8, ha='center', va='center', bbox=boxprops)  
        
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

plt.figure(figsize=(20, 10)) 
plot_decision_tree(node_dictionary, next(iter(node_dictionary)), x=0, y=0, dx=20, dy=5, depth=0) 
plt.tight_layout()
plt.show()
