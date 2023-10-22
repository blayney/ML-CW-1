import numpy as np
import matplotlib as mt

node_dictionary = {
    "leaf0" : ["leaf", 0, None, None],
    "leaf1" : ["leaf", 1, None, None],
    "leaf2" : ["leaf", 2, None, None],
    "leaf3" : ["leaf", 3, None, None]
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
    
def decision_tree_learning(training_dataset, depth):
    flag = False
    value = training_dataset[0][len(training_dataset[0]) - 1]
    for item in training_dataset:
        if item[len(item) - 1] != value:
            flag = True
    if flag == True:
        tmp = "leaf" + depth
        return {tmp : ["leaf", depth, None, None]}, depth
    else:
        split, attributeSplit, l_dataset, r_dataset = find_split(training_dataset)
        newVal = attributeSplit +  " < " + str(split)
        tmpTree = {newVal : [attributeSplit, split, "l_branch", "r_branch"] }
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        tmpKey = tmpTree[newVal]
        tmpKey[2] = next(iter(l_branch))
        tmpTree[newVal] = tmpKey
        if("leaf" not in next(iter(l_branch))):
           tmpTree.update(l_branch)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
        tmpKey = tmpTree[newVal]
        tmpKey[3] = next(iter(r_branch))
        tmpTree[newVal] = tmpKey
        if("leaf" not in next(iter(r_branch))):
           tmpTree.update(r_branch)
        return tmpTree, max(l_depth, r_depth)

tmp_dictionary, _ = decision_tree_learning(clean_data, 0)
node_dictionary.update(tmp_dictionary)
