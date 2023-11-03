import numpy as np
import matplotlib.pyplot as plt

class Node:
    """ Class for each individual node in the decision tree. """

    def __init__(self, feature=None, threshold=None, label=None, left_child=None, right_child=None):
        """ Initialize a decision tree node.

        Args:
            feature (int or None): The emitter used for splitting the dataset at this node,
                None when we know the room.
            threshold (float or None): The dB value for splitting the dataset based on the 
                emitter, None when we know the room.
            label (int or None): The label for this node, which contains information about 
                the decision needing to be made or the room number.
            left_child (Node or None): The left child node (subtree). Occurs when there is 
                a decision and contains the split dataset, None when we know the room.
            right_child (Node or None): The right child node (subtree). Occurs when there 
                is a decision and contains the other part of the split dataset, None when 
                we know the room.

        Attributes:
            max_tree_depth (int): The maximum depth of the decision tree. Used for pruning.

        Returns: 
            None
        """

        self.feature = feature
        self.threshold = threshold
        self.max_tree_depth = 0
        self.label = label
        self.left_child = left_child
        self.right_child = right_child


class DecisionTreeModel:
    """ Class used to store the whole decision tree. """

    # Initialization and basic utility method:
    def __init__(self, dataset_path, folds=10, depth_limit=None):
        """ Initialize an instance of DecisionTreeModel, reads in dataset and stores it in
            full_dataset.

        Args:
            dataset_path (str): The filepath to the dataset file.
            folds (int, optional): The number of folds to be used in cross-validation. 
                Defaults to 10.
            depth_limit (int or None, optional): The maximum depth allowed for the decision
                tree. Defaults to None.

        Attributes:
            dataset_path (str): The path to the dataset file.
            folds (int): The number of folds to be used in cross-validation.
            depth_limit (int or None): The maximum depth allowed for the decision tree.
            full_dataset (numpy.ndarray): A NumPy array containing the full dataset loaded 
                from 'dataset_path'.
            entropy_values (list): A list to store entropy values. ###

        Returns:
            None
        """

        self.dataset_path = dataset_path
        self.folds = folds
        self.depth_limit = depth_limit
        self.full_dataset = np.loadtxt(self.dataset_path)
        self.entropy_values = []
                        

    # Core decision tree building and prediction methods:
    def calculate_entropy(self, labels): 
        """ Calculate the entropy of a set of labels. ###

        Args:
            labels (numpy.ndarray): An array containing room numbers for data instances. ###

        Returns:
            float: The entropy value for the given set of room numbers.
        """

        unique_labels = np.unique(labels)
        entropy = 0
        for label in unique_labels:
            p = np.sum(labels == label) / len(labels)
            entropy -= p * np.log2(p)
        return entropy


    def get_best_feature_and_threshold(self, dataset):
        """ Find the best feature and threshold to split the dataset based on information 
            gain.

        Args:
            dataset (numpy.ndarray): The dataset containing the measurements in dB 
                (features) and room numbers (labels).

        Returns:
            tuple: A tuple (best_feature, best_threshold) representing the best feature 
                index and threshold value that maximize information gain when splitting 
                the dataset.
        """

        features, labels = dataset[:, :-1], dataset[:, -1] # extract measurements and rooms from the dataset
        total_entropy = self.calculate_entropy(labels) # calculates total entropy of entire dataset
        
        # initialise variables to keep track of the best information gain and its features/threshold
        max_IG = float('-inf')
        best_threshold, best_feature = None, None

        # iterate over each column in the dataset
        for col in range(features.shape[1]):
            unique_values = np.unique(features[:, col]) # finds unique values in current feature column
            
            if len(unique_values) == 1: # if there is only one unique value, skip the feature, gets rid of error on single row datasets
                continue
            
            prev_value = None
            for value in unique_values:
                current_threshold = value if prev_value is None else (value + prev_value) / 2 # calculate threshold of current iteration
                
                # split rooms based on current threshold
                labels_below_threshold = labels[features[:, col] <= current_threshold]
                labels_above_threshold = labels[features[:, col] > current_threshold]

                # calculate probabilities of samples falling above and below threshold
                p_below = len(labels_below_threshold) / len(labels)
                p_above = len(labels_above_threshold) / len(labels)

                # calculate weighted entropy for this split
                weighted_entropy = (p_below * self.calculate_entropy(labels_below_threshold) +
                                    p_above * self.calculate_entropy(labels_above_threshold))

                # calculate information gain for this split
                IG = total_entropy - weighted_entropy

                # update best feature, threshold if current split has higher information gain
                if IG > max_IG:
                    max_IG = IG
                    best_threshold = current_threshold
                    best_feature = col

                prev_value = value

        return best_feature, best_threshold


    def find_split(self, dataset):
        """ Find the best split for the dataset based on the given criteria.

        Args:
            dataset (numpy.ndarray): The dataset containing features and labels.

        Returns:
            tuple: A tuple (threshold, best_feature, splitL, splitR) representing the best split
            information for the dataset.
                - threshold (float): The threshold value used for the split.
                - best_feature (int): The index of the best feature for splitting.
                - splitL (numpy.ndarray): The dataset split with data points where the best feature
                    is less than or equal to the threshold.
                - splitR (numpy.ndarray): The dataset split with data points where the best feature
                    is greater than the threshold.
        """

        best_feature, threshold = self.get_best_feature_and_threshold(dataset)

        splitR = dataset[dataset[:, best_feature] > threshold]
        splitL = dataset[dataset[:, best_feature] <= threshold]

        return threshold, best_feature, splitL, splitR


    def decision_tree_learning(self, training_dataset, current_depth, depth_limit):
        """ Recursively construct a decision tree using the training dataset.

        Args:
            training_dataset (numpy.ndarray): The dataset used for training the decision tree.
            current_depth (int): The current depth of the decision tree during recursion.
            depth_limit (int or None): The maximum depth allowed for the decision tree. Set to None for no limit.

        Returns:
            tuple: A tuple (node, tree_depth) representing the constructed decision tree node
            and the depth of the tree.
                - node (Node): The root node of the decision tree or its subtree.
                - tree_depth (int): The depth of the constructed decision tree.
        """

        # calculate the current entropy of the dataset and add it to array
        current_entropy = self.calculate_entropy(training_dataset[:, -1])
        self.entropy_values.append((current_depth, current_entropy))

        # check if depth limit has been reached, exit condition/base case and picks most common room as label
        if depth_limit is not None and current_depth == depth_limit:
            most_common_label = int(np.bincount(training_dataset[:, -1].astype(int)).argmax())
            return Node(label=most_common_label), current_depth

        # check if all labels are the same and returns node with this label
        if len(np.unique(training_dataset[:, -1])) == 1:
            label_value = int(training_dataset[0, -1])
            return Node(label=label_value), current_depth
        
        # find best split and recursively create left and right sub-decision-trees
        best_threshold, best_emitter, l_dataset, r_dataset = self.find_split(training_dataset)
        left_node, left_depth = self.decision_tree_learning(l_dataset, current_depth + 1, depth_limit)
        right_node, right_depth = self.decision_tree_learning(r_dataset, current_depth + 1, depth_limit)

        # returns this node with the decision made and maximum depth of its children
        return Node(feature=best_emitter, threshold=best_threshold, left_child=left_node, right_child=right_node), max(left_depth, right_depth)


    def find_class(self, sample, node):
        """ Traverse the decision tree to find the class label for a given sample.

        Args:
            sample (numpy.ndarray): The sample for which to predict the class label.
            node (Node): The root node of the decision tree or its subtree.

        Returns:
            int: The predicted room for the given sample based on the decision tree.
        """

        # if current node is a leaf, return its label (room number)
        if node.label is not None:
            return node.label

        # traverse left or right subtree based on sample's feature value
        if sample[node.feature] <= node.threshold:
            return self.find_class(sample, node.left_child)
        else:
            return self.find_class(sample, node.right_child)

    
    def run_model(self, dataset, tree):
        """ Use the decision tree model to predict labels for a dataset.

        Args:
            dataset (numpy.ndarray): The dataset for which to make predictions.
            tree (Node): The root node of the decision tree model.

        Returns:
            numpy.ndarray: An array containing predicted class labels for each sample in the dataset.
        """

        # initialise list to store labels
        predicted_labels = [-1] * len(dataset)

        # iterate over dataset and predict class label for each sample
        for i in range(len(dataset)):
            predicted_labels[i] = self.find_class(dataset[i], tree)

        # convert list of predicted labels to a numpy array
        return np.array(predicted_labels)


    # Evaluation, cross validation and metrics computation methods:
    def evaluate(self, test_db, trained_tree): # This evaluate(...) function is not being called/used anymore
                                               # as we are computing all metrics from confusion matrix now. 
                                               # However, it is still kept here as the CourseWork Specification
                                               # document mentioned that we shall write this function.
        """ Evaluate the accuracy of a trained decision tree on a test dataset.

        Args:
            test_db (numpy.ndarray): The test dataset containing features and true class labels.
            trained_tree (Node): The trained decision tree to be evaluated.

        Returns:
            float: The accuracy of the trained decision tree on the test dataset as a floating-point value
                in the range [0.0, 1.0].
        """

        # use the trained decision tree to predict class labels for the test dataset
        predicted_labels = self.run_model(test_db[:,:-1], self.root_node)

        # extract true class labels from test dataset
        true_labels = test_db[:,-1]

        # initialise counter for correct predictions
        correct_predictions = 0

        # count how many labels were correct
        for i in range(len(true_labels)):
            if true_labels[i] == predicted_labels[i]:
                correct_predictions += 1

        # calculate and return accuracy
        return float(correct_predictions/len(true_labels))


    def do_macro_average(self, data):
        """Calculate the macro-avergage of a list of values.

        Args:
            data (list): A list of numerical values for which the macro-avergage is to be calculated.

        Returns:
            float: The macro-avergage of the provided list of values, rounded to 5 decimal places.
        """

        size = len(data)

        sum = 0

        # calculate sum of all values in the list
        for i in range (size):
            sum += data[i]

        return np.round(sum / size, 5)


    def compute_metrics(self, confusion_matrix):
        """ Calculate various classification metrics for multiple classes based on a confusion matrix.
        This method calculates recall, precision, and F1-score for each class using the given confusion matrix.
        It then returns lists of these metrics for all classes, providing insights into the model's performance
        on a per-class basis.

        Args:
            confusion_matrix (numpy.ndarray): A confusion matrix representing the model's performance
            in a multi-class classification task.

        Returns:
            tuple: A tuple containing lists of recall, precision, and F1-score for each class.
                - recalls (list): A list of recall values for each class.
                - precisions (list): A list of precision values for each class.
                - f1_scores (list): A list of F1-score values for each class.
        """

        # get number of classes from the confusion matrix
        num_classes = confusion_matrix.shape[0]

        # used to store data for each class
        recalls = []
        precisions = []
        f1_scores = []

        # iterate over each class to calculate metrics
        for i in range(num_classes):
            # calculates true positives, false negatives and false positives
            TP = confusion_matrix[i][i]
            FN = sum(confusion_matrix[i, :]) - TP
            FP = sum(confusion_matrix[:, i]) - TP
            
            # calculate recall, precision & f1-score for current class
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            # updates lists with values
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)

            # print metrics for current class
            print(f"Class {i+1} --->     Recall: {recall:.5f}    Precision: {precision:.5f}     F1-Measure: {f1:.5f}")

        return recalls, precisions, f1_scores


    def run_cross_validation(self, dataset, folds=10):
        """ Perform k-fold cross-validation and return the averaged confusion matrix.

        Args:
            dataset (numpy.ndarray): The dataset to be used for cross-validation.
            folds (int, optional): The number of folds for cross-validation. Defaults to 10.

        Returns:
            numpy.ndarray: The averaged confusion matrix based on k-fold cross-validation.
        """

        # shuffle dataset to ensure randomness in fold assignments
        np.random.shuffle(dataset)

        # determine size of each fold
        fold_size = len(dataset) // folds

        # initialise confusion matrix to store results
        confusion_matrix = np.zeros((4, 4))

        # perform k-fold cross validation
        for i in range(folds):
            # split dataset into training & testing sets for current fold
            test_data = dataset[i * fold_size : (i + 1) * fold_size]
            train_data = np.concatenate((dataset[: i * fold_size], dataset[(i + 1) * fold_size :]), axis=0)
            
            # train decision tree on training data for current fold
            self.root_node, _ = self.decision_tree_learning(train_data, 0, self.depth_limit)
            
            # use trained model to make predictions on test data
            predicted_labels = self.run_model(test_data[:, :-1], self.root_node)

            # extract true labels from test data
            true_labels = test_data[:, -1]

            # update confusion matrix for current fold
            for j in range(len(true_labels)):
                actual = int(true_labels[j]) - 1
                predicted = int(predicted_labels[j]) - 1
                confusion_matrix[actual][predicted] += 1

        # calculate averaged confusion matrix
        final_confusion_matrix = np.round(confusion_matrix / folds).astype(int)

        return np.array(final_confusion_matrix)


    def compute_accuracy(self, confusion_matrix):
        """ Calculate the accuracy based on the confusion matrix.

        Args:
            confusion_matrix (numpy.ndarray): The confusion matrix representing the model's performance.

        Returns:
            float: The accuracy of the model, rounded to 5 decimal places.
        """
        # initialise counters to track total, correctly predicted instances
        total_sum = 0
        true_sum = 0

        # iterate over confusion matrix to count total, correct predictions
        for i in range(4):
            for j in range(4):
                total_sum += confusion_matrix[i][j]
                if i == j:
                    true_sum += confusion_matrix[i][j]

        # calculates accuracy as a percentage
        accuracy = float(true_sum) / total_sum

        # rounds accuracy to 5 d.p.
        return np.round(accuracy, 5)


    # Visualization and plotting methods:
    def plot_decision_tree(self, node, x=0, y=0, dx=1, dy=1, depth=0):
        """ Plot a visual representation of the decision tree structure. 
        This method visualizes the structure of the decision tree by plotting nodes and their relationships.
        Each node's feature and threshold values are displayed on the plot, and branching paths are drawn
        to connect parent and child nodes. The resulting visualization is saved as 'Tree.png' if `depth` is 0.

        Args:
            node (Node): The current node in the decision tree to visualize.
            x (float, optional): The x-coordinate of the current node's position. Defaults to 0.
            y (float, optional): The y-coordinate of the current node's position. Defaults to 0.
            dx (float, optional): The change in the x-coordinate for each level of the tree. Defaults to 1.
            dy (float, optional): The change in the y-coordinate for each level of the tree. Defaults to 1.
            depth (int, optional): The current depth level of the tree. Defaults to 0.
        """
        if depth == 0:  
            # create a new figure for the plot at the beginning of the tree
            plt.figure(figsize=(20, 10))

        # if current node is a leaf, display label with an orange box
        if node.label is not None:
            boxprops = dict(facecolor='orange', edgecolor='black', boxstyle='round,pad=0.3')
            plt.text(x, y, f"R{node.label}", fontsize=8, ha='center', va='center', bbox=boxprops)
            return

        # exit if current node haas no useful information
        if node.feature is None or node.threshold is None:
            return

        # display feature and threshold for current node with light yellow box in format X0 > 36
        boxprops = dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.3')
        plt.text(x, y, f"E{node.feature} > \n{node.threshold:.2f}", fontsize=8, ha='center', va='center', bbox=boxprops)

        # if left child, recursively plot left subtree & draw connecting line
        if node.left_child:
            self.plot_decision_tree(node.left_child, x - dx / (2**depth), y - dy, dx, dy, depth + 1)
            plt.plot([x, x - dx / (2**depth)], [y, y - dy], color='black')
        
        # if right chilf, recursively plot right subtree & draw connecting line
        if node.right_child:
            self.plot_decision_tree(node.right_child, x + dx / (2**depth), y - dy, dx, dy, depth + 1)
            plt.plot([x, x + dx / (2**depth)], [y, y - dy], color='black')
        
        # if depth is 0 finalise plot, save visualisation as 'Tree.png'
        if depth == 0:  
            plt.tight_layout()
            plt.savefig('Tree.png')
            # plt.show()


    def plot_confusion_matrix(self, matrix):
        """ Plot a confusion matrix for model evaluation. This method generates
        a graphical representation of a confusion matrix for model evaluation. 
        The matrix is displayed as a heatmap with labeled axes for actual and
        predicted classes. The visualisation is saved as 'Confusion_Matrix.png'
        
        Args:
            matrix (numpy.ndarray): confusion matrix for visualisation
        """

        title="Confusion Matrix \n"

        # creates new figure, axis for plot
        fig, ax = plt.subplots()

        # displays confusion matrix as heatmap with colormap
        im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.YlOrBr)

        # adds colorbar to plot
        ax.figure.colorbar(im, ax=ax)

        # defines class labels for axes
        classes = ['1', '2', '3', '4']

        # set labels and title for plot
        ax.set(xticks=np.arange(matrix.shape[1]),
                yticks=np.arange(matrix.shape[0]),
                xticklabels=classes, yticklabels=classes,
                ylabel='Actual Classes',
                xlabel='Predicted Classes')

        ax.set_title(title, weight='bold')

        # adds text labels to cells of confusion matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
                
        fig.tight_layout()
        plt.savefig('Confusion_Matrix.png')
        # plt.show()


    def plot_loss(self, entropy_values):
        """ Plots average entropy (loss) vs depth of decision tree. Resulting plot saved
        as 'Loss_vs_Depth.png'.
        
        Args: 
            entropy_values (list): A list of tuples where each contains (depth, entropy).
        """


        depths = []
        entropies = []

        for item in entropy_values:
            depths.append(item[0])
            entropies.append(item[1])
            
        max_depth = max(depths)

        # calculate average entropies for each depth
        avg_entropies = []
        for d in range(max_depth + 1):
            total_entropy = 0
            count = 0

            # calculate total entropy and count for current depth
            for i in range(len(depths)):
                if depths[i] == d:
                    total_entropy += entropies[i]
                    count += 1

            # calculate and store average entropy
            avg_entropies.append(total_entropy / count)

        # create a plot with depth on x-axis and average entropy on y-axis
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
        """ Runs the entire decision tree algorithm and generates evaluation results.
        These are then visualised
        """

        # build decision tree & get maximum depth of it
        self.root_node, self.tree_depth = self.decision_tree_learning(self.full_dataset, 0, self.depth_limit)

        # output maximum depth
        print('\nMaximum Depth of the Trained Tree:    ', self.tree_depth)

        # perform cross-validation and obtain confusion matrix
        confusion_matrix = self.run_cross_validation(self.full_dataset)

        # calculate algorithm accuracy and output it
        algorithm_accuracy = self.compute_accuracy(confusion_matrix)
        print('\nAlgorithm Accuracy:    ', algorithm_accuracy)

        # calculate and output classification metrics for each class
        print("\nCross Validation Classification Metrics for Each Class:        ")
        recalls, precisions, f1_scores = self.compute_metrics(confusion_matrix)
        
        # calculate and output macro-averaged recall
        macro_averaged_recall = self.do_macro_average(recalls)
        print('\nMacro-Averaged Recall:    ', macro_averaged_recall)

        # calculate and output macro-averaged precision
        macro_averaged_precision = self.do_macro_average(precisions)
        print('Macro-Averaged Precision:    ', macro_averaged_precision)

        # calculate and output macro-averaged f1-score
        macro_averaged_f1_score = self.do_macro_average(f1_scores)
        print('Macro-Averaged F1-Score:    ', macro_averaged_f1_score)
        print(' ')

        # generate visualisations for decision tree, confusion matrix, loss graph
        self.plot_decision_tree(self.root_node) 
        self.plot_confusion_matrix(confusion_matrix)
        self.plot_loss(self.entropy_values)



# Main for script execution:
if __name__ == '__main__':
    dataset_path='wifi_db/clean_dataset.txt'
    model = DecisionTreeModel(dataset_path, folds=10)
    model.run()