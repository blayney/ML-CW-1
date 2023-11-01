    def get_best_feature_and_threshold(self, dataset):
        features, labels = dataset[:, :-1], dataset[:, -1]
        total_entropy = self.calculate_entropy(labels)
        max_IG = float('-inf')
        best_threshold, best_feature = None, None

        for col in range(features.shape[1]):
            unique_values = sorted(np.unique(features[:, col]))
            
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
