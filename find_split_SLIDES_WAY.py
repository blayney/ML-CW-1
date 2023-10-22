import numpy as np
import matplotlib as mt

dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

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

    print("Emitter Number: ", max_IG_col)
    print("Threshold Value: ", threshold)
    print("Threshold IG: ", max_IG_for_value)
    # print("Emitter IG: ", max_IG_of_emitter)
    print(splitL.shape)
    print(splitR.shape)

    return threshold, max_IG_col, splitL, splitR



print("")
print("Head")
threshold, max_IG_col, split_L, split_R = find_split(dataset)

print("")
print("Right Wing")
find_split(split_R)

print("")
print("Left Wing")
find_split(split_L)

print("")
print("Calling on right wing")
threshold, max_IG_col, split_L, split_R = find_split(split_R)
print("")
print("Right-Right Wing")
find_split(split_R)

print("")
print("Right-Left Wing")
find_split(split_L)