    # def evaluate(self, test_db, trained_tree):
    #     predicted_labels = self.run_model(test_db[:,:-1], self.root_node)
    #     true_labels = test_db[:,-1]

    #     correct_predictions = 0

    #     for i in range(len(true_labels)):
    #         if true_labels[i] == predicted_labels[i]:
    #             correct_predictions += 1

    #     return float(correct_predictions/len(true_labels))