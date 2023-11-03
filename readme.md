# Decision Tree Model for Room Locationing based on Wi-Fi Reception

This project provides a decision tree-based model for predicting room numbers based on the receiving Wi-Fi signal strengths. 

## Prerequisites

Before running the code, ensure you have the following dependencies installed on the Lab Machines:

- Python 3.10.12
- NumPy 1.24.3
- Matplotlib 3.7.2

To install the dependencies, run:

```py
pip install matplotlib==3.7.2 numpy==1.24.3
```

## Data

Make sure you update the `__main__` section of the `decision_tree.py` file:

```py
# Main for script execution:
if __name__ == '__main__':
    dataset_path='wifi_db/clean_dataset.txt'
    model = DecisionTreeModel(dataset_path, folds=10)
    model.run()

```
- `dataset_path` : File path of the dataset to load and run the program on.
- `folds` : Number of cross validation folds to perform on the dataset.

## How to Run

1. Download the file `decision_tree.py` to your local machine.
2. Update the  `__main__` as required with the dataset's file path.
3. Execute the main script:

    ```
    python decision_tree.py
    ```

This will do the followings:
- Build a Decision Tree Model on the entire dataset.
- Run cross validation based on the choosen number of `folds`.
- Generate necessary plots and compute various cross validation classification metrics.



## Outputs

Upon successful execution, you will get:

1. A visualization of the Decision Tree Model trained on the full dataset as `Tree.png`.
2. A confusion matrix saved as `Confusion_Matrix.png` generated from the k-fold cross validation.
3. A plot representing loss vs. depth of the tree saved as `Loss_vs_Depth.png` showing data-subsets become purer with increasing tree depth.
4. The maximum depth of the initial  Decision Tree Model produced, printed on console.
5. The accuracy of the algorithm computed from k-fold cross validation, printed on console.
6. Cross validation classification metrics such as recall, precision and F1-measure for each class, printed on console.
7. The macro-averaged values of the above metrics from each class, printed on console. 

The `.png` images produced will be saved in the same directory where the script is being executed.


---
