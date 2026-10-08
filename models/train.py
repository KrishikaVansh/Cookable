"""
Trains the 3 from-scratch cuisine classifiers (Naive Bayes, Softmax Regression,
Neural Net) on the "What's Cooking" style dataset (ingredients + cuisine label only).

Usage:
    python train.py --data ../data/train.json
"""
import json
import argparse
import pickle
import numpy as np

from features import Vectorizer
from naive_bayes import NaiveBayesClassifier
from softmax_regression import SoftmaxRegression
from neural_net import NeuralNetClassifier


def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    ingredients = [d["ingredients"] for d in data]
    cuisines = [d["cuisine"] for d in data]
    return ingredients, cuisines


def train_test_split(X_ingredients, y, test_ratio=0.15, seed=42):
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train = [X_ingredients[i] for i in train_idx]
    X_test = [X_ingredients[i] for i in test_idx]
    y_train = np.array([y[i] for i in train_idx])
    y_test = np.array([y[i] for i in test_idx])
    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main(data_path, out_dir):
    print("Loading data...")
    ingredients, cuisines = load_data(data_path)
    print(f"  {len(ingredients)} recipes, {len(set(cuisines))} cuisines")

    classes = sorted(set(cuisines))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = [label_to_idx[c] for c in cuisines]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(ingredients, y)

    print("\nBuilding vocabulary + vectorizing (TF-IDF, from scratch)...")
    vectorizer = Vectorizer(use_tfidf=True, min_df=3)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"  Vocab size: {len(vectorizer.vocab)}")

    count_vectorizer = Vectorizer(use_tfidf=False, min_df=3)
    X_train_counts = count_vectorizer.fit_transform(X_train_raw)
    X_test_counts = count_vectorizer.transform(X_test_raw)

    print("\n--- Training Naive Bayes ---")
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(X_train_counts, y_train)
    nb_acc = accuracy(y_test, nb.predict(X_test_counts))
    print(f"Naive Bayes test accuracy: {nb_acc:.4f}")

    print("\n--- Training Softmax Regression ---")
    softmax = SoftmaxRegression(lr=0.5, epochs=150, batch_size=256, l2=1e-4)
    softmax.fit(X_train, y_train)
    softmax_acc = accuracy(y_test, softmax.predict(X_test))
    print(f"Softmax Regression test accuracy: {softmax_acc:.4f}")

    print("\n--- Training Neural Network ---")
    nn = NeuralNetClassifier(hidden_size=128, lr=0.3, epochs=150, batch_size=256, l2=1e-4)
    nn.fit(X_train, y_train)
    nn_acc = accuracy(y_test, nn.predict(X_test))
    print(f"Neural Net test accuracy: {nn_acc:.4f}")

    print("\n=== Summary ===")
    print(f"Naive Bayes:        {nb_acc:.4f}")
    print(f"Softmax Regression: {softmax_acc:.4f}")
    print(f"Neural Network:     {nn_acc:.4f}")

    print("\nSaving models + vectorizers...")
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    artifacts = {
        "vectorizer_tfidf": vectorizer,
        "vectorizer_counts": count_vectorizer,
        "naive_bayes": nb,
        "softmax_regression": softmax,
        "neural_net": nn,
        "idx_to_label": idx_to_label,
    }
    with open(f"{out_dir}/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved to {out_dir}/artifacts.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/train.json")
    parser.add_argument("--out", default="../data")
    args = parser.parse_args()
    main(args.data, args.out)
