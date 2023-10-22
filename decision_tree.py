import numpy as np
import matplotlib as mt

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