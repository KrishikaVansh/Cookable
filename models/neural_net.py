"""
Small feedforward Neural Network — 1 hidden layer, manual forward + backprop.
No PyTorch/TensorFlow/autograd. Pure numpy.
"""
import numpy as np


class NeuralNetClassifier:
    def __init__(self, hidden_size=64, lr=0.05, epochs=300, batch_size=256, l2=1e-4, verbose=True):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.verbose = verbose
        self.classes_ = None
        self.params = {}
        self.history_ = []

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(np.float32)

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, n_classes):
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y] = 1
        return oh

    def _init_params(self, n_features, n_classes):
        h = self.hidden_size
        # He init, good default for ReLU hidden layers
        self.params["W1"] = np.random.randn(n_features, h) * np.sqrt(2.0 / n_features)
        self.params["b1"] = np.zeros(h)
        self.params["W2"] = np.random.randn(h, n_classes) * np.sqrt(2.0 / h)
        self.params["b2"] = np.zeros(n_classes)

    def _forward(self, X):
        Z1 = X @ self.params["W1"] + self.params["b1"]
        A1 = self._relu(Z1)
        Z2 = A1 @ self.params["W2"] + self.params["b2"]
        A2 = self._softmax(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def _backward(self, cache, Y):
        m = cache["X"].shape[0]

        dZ2 = (cache["A2"] - Y) / m                       # (m, n_classes)
        dW2 = cache["A1"].T @ dZ2 + self.l2 * self.params["W2"]
        db2 = dZ2.sum(axis=0)

        dA1 = dZ2 @ self.params["W2"].T
        dZ1 = dA1 * self._relu_deriv(cache["Z1"])
        dW1 = cache["X"].T @ dZ1 + self.l2 * self.params["W1"]
        db1 = dZ1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y])
        Y = self._one_hot(y_idx, n_classes)

        self._init_params(n_features, n_classes)

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            X_shuf, Y_shuf = X[perm], Y[perm]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuf[start:end]
                Y_batch = Y_shuf[start:end]

                _, cache = self._forward(X_batch)
                grads = self._backward(cache, Y_batch)

                for param in self.params:
                    self.params[param] -= self.lr * grads[param]

            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                loss = self._loss(X, Y)
                self.history_.append(loss)
                print(f"  [NeuralNet] epoch {epoch:3d} | loss {loss:.4f}")

        return self

    def _loss(self, X, Y):
        A2, _ = self._forward(X)
        A2 = np.clip(A2, 1e-12, 1.0)
        return -np.mean(np.sum(Y * np.log(A2), axis=1))

    def predict_proba(self, X):
        A2, _ = self._forward(X)
        return A2

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
