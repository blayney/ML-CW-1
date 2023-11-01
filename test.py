import numpy as np
import matplotlib.pyplot as plt

class DecisionTreeModel:

# Initialization and basic utility methods:
    def __init__(self, dataset_path, folds=10, depth_limit=None):
        self.dataset_path = dataset_path
        self.folds = folds
        self.depth_limit = depth_limit
        self.full_dataset = np.loadtxt(self.dataset_path)
        self.node_dictionary = {}
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
            sorted_indices = np.argsort(features[:, col])
            sorted_features, sorted_labels = features[sorted_indices, col], labels[sorted_indices]
            
            labels_below = []
            labels_above = sorted_labels.tolist()

            for idx in range(len(sorted_features) - 1):
                labels_below.append(sorted_labels[idx])
                labels_above.remove(sorted_labels[idx])

                if sorted_features[idx] == sorted_features[idx + 1]:
                    continue

                p_below = len(labels_below) / len(labels)
                p_above = len(labels_above) / len(labels)

                weighted_entropy = (p_below * self.calculate_entropy(np.array(labels_below)) +
                                    p_above * self.calculate_entropy(np.array(labels_above)))

                IG = total_entropy - weighted_entropy

                if IG > max_IG:
                    max_IG = IG
                    best_threshold = (sorted_features[idx] + sorted_features[idx + 1]) / 2
                    best_feature = col

        return best_feature, best_threshold

    def find_split(self, dataset):
        best_feature, threshold = self.get_best_feature_and_threshold(dataset)
        return threshold, best_feature, dataset[dataset[:, best_feature] <= threshold], dataset[dataset[:, best_feature] > threshold]

    def decision_tree_learning(self, training_dataset, current_depth, depth_limit, entropy_values=[]):
        current_entropy = self.calculate_entropy(training_dataset[:, -1])
        entropy_values.append((current_depth, current_entropy))

        if len(np.unique(training_dataset[:, -1])) == 1:
            label_value = int(training_dataset[0, -1])
            return {f'R {label_value}': [None, label_value, None, None]}, current_depth

        if depth_limit is not None and current_depth == depth_limit:
            most_common_label = int(np.bincount(training_dataset[:, -1].astype(int)).argmax())
            return {f'R {most_common_label}': [None, most_common_label, None, None]}, current_depth

        best_threshold, best_feature, l_dataset, r_dataset = self.find_split(training_dataset)
        newVal = f"X{best_feature} < {best_threshold}"

        tmpTree = {newVal: [best_feature, best_threshold, "l_branch", "r_branch"]}

        l_branch, l_depth = self.decision_tree_learning(l_dataset, current_depth + 1, depth_limit, entropy_values)
        r_branch, r_depth = self.decision_tree_learning(r_dataset, current_depth + 1, depth_limit, entropy_values)

        tmpTree[newVal][2], tmpTree[newVal][3] = next(iter(l_branch)), next(iter(r_branch))
        tmpTree.update(l_branch)
        tmpTree.update(r_branch)

        return tmpTree, max(l_depth, r_depth)



    def run_model(self, dataset, tree):

        def get_class(data):
            current_node = next(iter(tree))

            while True:
                feature, threshold, left, right = tree[current_node]

                if left is None and right is None:
                    return threshold

                if data[feature] < threshold:
                    current_node = left
                else:
                    current_node = right

        predicted_labels = [get_class(data) for data in dataset]

        return np.array(predicted_labels)



# Evaluation and metrics computation methods:
    def evaluate(self, test_db, trained_tree):
        predicted_labels = self.run_model(test_db[:,:-1], self.node_dictionary)
        true_labels = test_db[:,-1]

        correct_predictions = 0

        for i in range(len(true_labels)):
            if true_labels[i] == predicted_labels[i]:
                correct_predictions += 1

        return float(correct_predictions/len(true_labels))


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

            print(f"Class {i+1} --->     Recall: {recall:.3f}    Precision: {precision:.3f}     F1-Measure: {f1:.3f}")

        return recalls, precisions, f1_scores


    def run_cross_validation(self, dataset, folds=10):

        np.random.seed(7000)

        np.random.shuffle(dataset)

        fold_size = len(dataset) // folds

        confusion_matrix = np.zeros((4, 4))

        total_accuracy_of_all_trees = 0

        for i in range(folds):
            test_data = dataset[i * fold_size : (i + 1) * fold_size]
            
            train_data = np.concatenate((dataset[: i * fold_size], dataset[(i + 1) * fold_size :]), axis=0)
            
            tmp_dictionary, _ = self.decision_tree_learning(train_data, 0, self.depth_limit)
            node_dictionary = {}
            node_dictionary.update(tmp_dictionary)

            predicted_labels = self.run_model(test_data[:, :-1], node_dictionary)
            true_labels = test_data[:, -1]

            total_accuracy_of_all_trees += self.evaluate(test_data, node_dictionary)

            for j in range(len(true_labels)):
                actual = int(true_labels[j]) - 1
                predicted = int(predicted_labels[j]) - 1
                confusion_matrix[actual][predicted] += 1

        final_confusion_matrix = np.round(confusion_matrix / folds).astype(int)
        average_accuracy = float(total_accuracy_of_all_trees) / folds

        return np.array(final_confusion_matrix), average_accuracy


    def compute_accuracy(self, confusion_matrix):

        total_sum = 0
        true_sum = 0

        for i in range(4):
            for j in range(4):
                total_sum += confusion_matrix[i][j]
                if i == j:
                    true_sum += confusion_matrix[i][j]

        accuracy = float(true_sum) / total_sum

        return np.round(accuracy, 3)


# Visualization and plotting methods:
    def plot_decision_tree(self, node_dictionary):
        def prepare_visual_decision_tree(tree, node, x, y, dx, dy, depth=0):
            if node in tree:
                feature, threshold, left, right = tree[node]
                boxprops = dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.3')

                plt.text(x, y, f"R{threshold}" if left is None and right is None else f"E{feature} > \n{threshold:.2f}",
                            fontsize=8, ha='center', va='center', bbox=boxprops)

                if left in tree:
                    prepare_visual_decision_tree(tree, left, x - dx / (2**depth), y - dy, dx, dy, depth + 1)
                    plt.plot([x, x - dx / (2**depth)], [y, y - dy], color='black')

                if right in tree:
                    prepare_visual_decision_tree(tree, right, x + dx / (2**depth), y - dy, dx, dy, depth + 1)
                    plt.plot([x, x + dx / (2**depth)], [y, y - dy], color='black')

        plt.figure(figsize=(20, 10))
        prepare_visual_decision_tree(node_dictionary, next(iter(node_dictionary)), x=0, y=0, dx=20, dy=5, depth=0)
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
        # plt.show()


# Driver method:
    def run(self):
        entropy_values = []
        tmp_dictionary, max_depth_achieved = self.decision_tree_learning(self.full_dataset, current_depth=0, depth_limit=self.depth_limit, entropy_values=entropy_values)
        self.node_dictionary.update(tmp_dictionary)

        print('Model Maximum Depth: ', max_depth_achieved)
        
        model_accuracy = self.evaluate(self.full_dataset, self.node_dictionary)
        print('Model Accuracy: ', model_accuracy)

        confusion_matrix, average_accuracy = self.run_cross_validation(self.full_dataset)
        algorithm_accuracy = self.compute_accuracy(confusion_matrix)
        print('Algorithm Accuracy: ', algorithm_accuracy, ' ', average_accuracy)

        self.compute_metrics(confusion_matrix)

        self.plot_decision_tree(self.node_dictionary)
        self.plot_confusion_matrix(confusion_matrix)
        self.plot_loss(entropy_values)





# Main script execution:
if __name__ == '__main__':
    dataset_path='wifi_db/clean_dataset.txt'
    model = DecisionTreeModel(dataset_path, folds=10, depth_limit=None)
    model.run()