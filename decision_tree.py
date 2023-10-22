import numpy as np
import matplotlib as mt

node_dictionary = {
    "head" : ["attribute", "value", "l_branch", "r_branch"]
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

def find_split():
    return

unique_leaf_id = 0

def decision_tree_learning(training_dataset, depth):
    flag = False
    value = training_dataset[0][len(training_dataset[0]) - 1]
    for item in training_dataset:
        if item[len(item) - 1] != value:
            flag = True
    if flag == True:
        tmp = "leaf: " + unique_leaf_id
        unique_leaf_id += 1
        return {tmp : ["leaf", depth, None, None]}, depth
    else:
        split, attributeSplit, l_dataset, r_dataset = find_split(training_dataset)
        newVal = attributeSplit +  " < " + str(split)
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
