# Decision Tree Model for Wi-Fi Locationing

This project provides a decision tree-based model for predicting room numbers based on Wi-Fi signal strengths. 

## Prerequisites

Before running the code, ensure you have the following dependencies:

- Python 3
- NumPy 1.24.3
- Matplotlib 3.7.2

To install the dependencies, run:

```py
pip install matplotlib==3.7.2 numpy==1.24.3
```

## Data

Make sure you update the `__main__` section of `decision_tree.py` file:

```py
# Main script execution:
if __name__ == '__main__':
    dataset_path='wifi_db/clean_dataset.txt'
    model = DecisionTreeModel(dataset_path, train_ratio=0.8, folds=10)
    model.run()

```
- `dataset_path` : The file path of the dataset you wish to load and run the program on.
- `train_ratio` : The ratio at which training and testing data will be split from the complete dataset.
- `folds` : Number of cross validation folds to perform on the dataset.

## How to Run

1. Download the file `decision_tree.py` to your local machine.
2. Update the  `__main__` as required.
3. Execute the main script:

    ```
    python decision_tree.py
    ```

This will do the followings:
- Train the Decision Tree Model on the given dataset's training portion as split by the choosen `train_ratio`.
- Evaluate the model's accuracy by running the test portion of the dataset as split by the choosen `train_ratio`.
- Run cross validation based on the choosen number of `folds`.
- Generate necessary plots and show various cross validation classification metrics.



## Outputs

Upon successful execution, you will get:

1. A visualization of the decision tree saved as `Tree.png`.
2. A confusion matrix saved as `Confusion_Matrix.png`.
3. A plot representing loss vs. depth of the tree saved as `Loss_vs_Depth.png`.
4. The maximum depth of the initial  Decision Tree Model produced, printed on console.
5. The accuracy of the initial  Decision Tree Model produced, printed on console.
6. Various Cross Validation Classification Metrics printed on the console, such as recall, precision, F1-measure for each class, and the overall accuracy of the algorithm.

The `.png` images produced will be saved in the same directory where the script is executed.


---