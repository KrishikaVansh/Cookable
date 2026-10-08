"""
Multinomial Naive Bayes — implemented from scratch with numpy.
No sklearn. Works on raw count features (not tf-idf normalized).
"""
import numpy as np


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha            # Laplace smoothing
        self.classes_ = None
        self.class_log_prior_ = None  # log P(class)
        self.feature_log_prob_ = None  # log P(word | class), shape (n_classes, n_features)

    def fit(self, X, y):
        """
        X: (n_samples, n_features) raw counts (bag-of-words, NOT tf-idf)
        y: (n_samples,) integer class labels
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            # prior: fraction of samples belonging to this class
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)

            # word counts for this class + Laplace smoothing
            word_counts = X_c.sum(axis=0) + self.alpha
            total_count = word_counts.sum()
            self.feature_log_prob_[idx] = np.log(word_counts / total_count)

        return self

    def predict_log_proba(self, X):
        # log P(class|X) proportional to log P(class) + sum(count_i * log P(word_i|class))
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        log_probs = self.predict_log_proba(X)
        # softmax for interpretable probabilities
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
