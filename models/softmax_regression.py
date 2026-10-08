"""
Multi-class Softmax (Logistic) Regression — implemented from scratch.
Trained with mini-batch gradient descent, no sklearn/autograd.
"""
import numpy as np


class SoftmaxRegression:
    def __init__(self, lr=0.1, epochs=200, batch_size=256, l2=1e-4, verbose=True):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.verbose = verbose
        self.W = None
        self.b = None
        self.classes_ = None
        self.history_ = []

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)  # numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y] = 1
        return oh

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # map labels to 0..n_classes-1 indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y])
        Y = self._one_hot(y_idx, n_classes)

        # Xavier init
        self.W = np.random.randn(n_features, n_classes) * np.sqrt(1.0 / n_features)
        self.b = np.zeros(n_classes)

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            X_shuf, Y_shuf = X[perm], Y[perm]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuf[start:end]
                Y_batch = Y_shuf[start:end]
                m = X_batch.shape[0]

                # forward
                logits = X_batch @ self.W + self.b
                probs = self._softmax(logits)

                # gradients (cross-entropy loss + L2 reg)
                grad_logits = (probs - Y_batch) / m
                grad_W = X_batch.T @ grad_logits + self.l2 * self.W
                grad_b = grad_logits.sum(axis=0)

                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                loss = self._cross_entropy_loss(X, Y)
                self.history_.append(loss)
                print(f"  [SoftmaxRegression] epoch {epoch:3d} | loss {loss:.4f}")

        return self

    def _cross_entropy_loss(self, X, Y):
        logits = X @ self.W + self.b
        probs = self._softmax(logits)
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.sum(Y * np.log(probs), axis=1))

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
