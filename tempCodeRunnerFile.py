# Main script execution:
if __name__ == '__main__':
    dataset_path='wifi_db/clean_dataset.txt'
    model = DecisionTreeModel(dataset_path, train_ratio=0.8, folds=10)
    model.run()
