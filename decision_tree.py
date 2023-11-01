import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, label=None, left_child=None, right_child=None):
        self.feature = feature
        self.threshold = threshold
        self.max_tree_depth = 0
        self.label = label
        self.left_child = left_child
        self.right_child = right_child


class DecisionTreeModel:

# Initialization and basic utility method:
    def __init__(self, dataset_path, folds=10, depth_limit=None):
        self.dataset_path = dataset_path
        self.folds = folds
        self.depth_limit = depth_limit
        self.full_dataset = np.loadtxt(self.dataset_path)
        self.entropy_values = []
                        

# Core decision tree building and prediction methods:
    def calculate_entropy(self, labels):
        unique_labels = np.unique(labels)
        entropy = 0
        for label in unique_labels:
            p = np.sum(labels == label) / len(labels)
            entropy -= p * np.log2(p)
        return entropy


    def get_best_feature_and_threshold(self, dataset):
        features, labels = dataset[:, :-1], dataset[:, -1]
        total_entropy = self.calculate_entropy(labels)
        max_IG = float('-inf')
        best_threshold, best_feature = None, None

        for col in range(features.shape[1]):
            unique_values = np.unique(features[:, col])
            
            if len(unique_values) == 1:
                continue
            
            prev_value = None
            for value in unique_values:
                current_threshold = value if prev_value is None else (value + prev_value) / 2
                
                labels_below_threshold = labels[features[:, col] <= current_threshold]
                labels_above_threshold = labels[features[:, col] > current_threshold]

                p_below = len(labels_below_threshold) / len(labels)
                p_above = len(labels_above_threshold) / len(labels)

                weighted_entropy = (p_below * self.calculate_entropy(labels_below_threshold) +
                                    p_above * self.calculate_entropy(labels_above_threshold))

                IG = total_entropy - weighted_entropy

                if IG > max_IG:
                    max_IG = IG
                    best_threshold = current_threshold
                    best_feature = col

                prev_value = value

        return best_feature, best_threshold


    def find_split(self, dataset):
        best_feature, threshold = self.get_best_feature_and_threshold(dataset)
        splitR = dataset[dataset[:, best_feature] > threshold]
        splitL = dataset[dataset[:, best_feature] <= threshold]
        return threshold, best_feature, splitL, splitR


    def decision_tree_learning(self, training_dataset, current_depth, depth_limit):

        current_entropy = self.calculate_entropy(training_dataset[:, -1])
        self.entropy_values.append((current_depth, current_entropy))

        if depth_limit is not None and current_depth == depth_limit:
            most_common_label = int(np.bincount(training_dataset[:, -1].astype(int)).argmax())
            return Node(label=most_common_label), current_depth

        if len(np.unique(training_dataset[:, -1])) == 1:
            label_value = int(training_dataset[0, -1])
            return Node(label=label_value), current_depth

        best_threshold, best_emitter, l_dataset, r_dataset = self.find_split(training_dataset)
        left_node, left_depth = self.decision_tree_learning(l_dataset, current_depth + 1, depth_limit)
        right_node, right_depth = self.decision_tree_learning(r_dataset, current_depth + 1, depth_limit)

        return Node(feature=best_emitter, threshold=best_threshold, left_child=left_node, right_child=right_node), max(left_depth, right_depth)


    def find_class(self, sample, node):
        if node.label is not None:
            return node.label

        if sample[node.feature] <= node.threshold:
            return self.find_class(sample, node.left_child)
        else:
            return self.find_class(sample, node.right_child)

    
    def run_model(self, dataset, tree):

        predicted_labels = [-1] * len(dataset)

        for i in range(len(dataset)):
            predicted_labels[i] = self.find_class(dataset[i], tree)

        return np.array(predicted_labels)


# Evaluation and metrics computation methods:
    def evaluate(self, test_db, trained_tree): #function not being used anywhere, but the spec said we had to implement, plz check @Jackson / @Will
        predicted_labels = self.run_model(test_db[:,:-1], self.root_node)
        true_labels = test_db[:,-1]

        correct_predictions = 0

        for i in range(len(true_labels)):
            if true_labels[i] == predicted_labels[i]:
                correct_predictions += 1

        return float(correct_predictions/len(true_labels))


    def do_macro_avergage(self, data):
        size = len(data)
        sum = 0
        for i in range (size):
            sum += data[i]

        return np.round(sum / size, 5)


    def compute_metrics(self, confusion_matrix):
        num_classes = confusion_matrix.shape[0]
        recalls = []
        precisions = []
        f1_scores = []

        for i in range(num_classes):
            TP = confusion_matrix[i][i]
            FN = sum(confusion_matrix[i, :]) - TP
            FP = sum(confusion_matrix[:, i]) - TP
            
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)

            print(f"Class {i+1} --->     Recall: {recall:.5f}    Precision: {precision:.5f}     F1-Measure: {f1:.5f}")

        return recalls, precisions, f1_scores


    def run_cross_validation(self, dataset, folds=10):

        np.random.shuffle(dataset)

        fold_size = len(dataset) // folds

        confusion_matrix = np.zeros((4, 4))

        for i in range(folds):
            test_data = dataset[i * fold_size : (i + 1) * fold_size]
            
            train_data = np.concatenate((dataset[: i * fold_size], dataset[(i + 1) * fold_size :]), axis=0)
            
            self.root_node, _ = self.decision_tree_learning(train_data, 0, self.depth_limit)
            predicted_labels = self.run_model(test_data[:, :-1], self.root_node)

            true_labels = test_data[:, -1]

            for j in range(len(true_labels)):
                actual = int(true_labels[j]) - 1
                predicted = int(predicted_labels[j]) - 1
                confusion_matrix[actual][predicted] += 1

        final_confusion_matrix = np.round(confusion_matrix / folds).astype(int)

        return np.array(final_confusion_matrix)


    def compute_accuracy(self, confusion_matrix):

        total_sum = 0
        true_sum = 0

        for i in range(4):
            for j in range(4):
                total_sum += confusion_matrix[i][j]
                if i == j:
                    true_sum += confusion_matrix[i][j]

        accuracy = float(true_sum) / total_sum

        return np.round(accuracy, 5)


# Visualization and plotting methods:
    def plot_decision_tree(self, node, x=0, y=0, dx=1, dy=1, depth=0):
        if depth == 0:  
            plt.figure(figsize=(20, 10))

        if node.label is not None:
            boxprops = dict(facecolor='orange', edgecolor='black', boxstyle='round,pad=0.3')
            plt.text(x, y, f"R{node.label}", fontsize=8, ha='center', va='center', bbox=boxprops)
            return

        if node.feature is None or node.threshold is None:
            return

        boxprops = dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.3')
        plt.text(x, y, f"E{node.feature} > \n{node.threshold:.2f}", fontsize=8, ha='center', va='center', bbox=boxprops)


        if node.left_child:
            self.plot_decision_tree(node.left_child, x - dx / (2**depth), y - dy, dx, dy, depth + 1)
            plt.plot([x, x - dx / (2**depth)], [y, y - dy], color='black')
        
        if node.right_child:
            self.plot_decision_tree(node.right_child, x + dx / (2**depth), y - dy, dx, dy, depth + 1)
            plt.plot([x, x + dx / (2**depth)], [y, y - dy], color='black')
        
        if depth == 0:  
            plt.tight_layout()
            plt.savefig('Tree.png')


    def plot_confusion_matrix(self, matrix):

        title="Confusion Matrix \n"

        fig, ax = plt.subplots()
        im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.YlOrBr)

        ax.figure.colorbar(im, ax=ax)

        classes = ['1', '2', '3', '4']

        ax.set(xticks=np.arange(matrix.shape[1]),
                yticks=np.arange(matrix.shape[0]),
                xticklabels=classes, yticklabels=classes,
                ylabel='Actual Classes',
                xlabel='Predicted Classes')

        ax.set_title(title, weight='bold')

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
                
        fig.tight_layout()
        plt.savefig('Confusion_Matrix.png')


    def plot_loss(self, entropy_values):
        depths = []
        entropies = []

        for item in entropy_values:
            depths.append(item[0])
            entropies.append(item[1])
            
        max_depth = max(depths)

        avg_entropies = []
        for d in range(max_depth + 1):

            total_entropy = 0
            count = 0
            for i in range(len(depths)):
                if depths[i] == d:
                    total_entropy += entropies[i]
                    count += 1
            avg_entropies.append(total_entropy / count)

        plt.figure(figsize=(10, 6))
        plt.plot(range(max_depth + 1), avg_entropies, marker='o')
        plt.xlabel('Depth of Tree')
        plt.ylabel('Average Entropy (Loss)')
        plt.title('Loss vs. Depth of Decision Tree')
        plt.grid(True)
        plt.savefig('Loss_vs_Depth.png')


# Driver method:
    def run(self):
        self.root_node, self.tree_depth = self.decision_tree_learning(self.full_dataset, 0, self.depth_limit)

        print('\nMaximum Depth of the Trained Tree:    ', self.tree_depth)

        confusion_matrix = self.run_cross_validation(self.full_dataset)
        algorithm_accuracy = self.compute_accuracy(confusion_matrix)
        print('\nAlgorithm Accuracy:    ', algorithm_accuracy)

        print("\nCross Validation Classification Metrics for Each Class:        ")
        recalls, precisions, f1_scores = self.compute_metrics(confusion_matrix)

        macro_averaged_recall = self.do_macro_avergage(recalls)
        print('\nMacro-Averaged Recall:    ', macro_averaged_recall)

        macro_averaged_precision = self.do_macro_avergage(precisions)
        print('Macro-Averaged Precision:    ', macro_averaged_precision)

        macro_averaged_f1_score = self.do_macro_avergage(f1_scores)
        print('Macro-Averaged F1-Score:    ', macro_averaged_f1_score)
        print(' ')

        self.plot_decision_tree(self.root_node) 
        self.plot_confusion_matrix(confusion_matrix)
        self.plot_loss(self.entropy_values)



# Main script execution:
if __name__ == '__main__':
    dataset_path='wifi_db/clean_dataset.txt'
    model = DecisionTreeModel(dataset_path, folds=10)
    model.run()