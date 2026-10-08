"""
Feature engineering — built entirely from scratch (no sklearn).
Converts ingredient lists into numeric vectors using Bag-of-Words / TF-IDF.
"""
import numpy as np
import re
from collections import Counter


def clean_ingredient(text):
    """Normalize an ingredient string: lowercase, strip numbers/punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Vectorizer:
    """
    Hand-built Bag-of-Words + TF-IDF vectorizer.
    Fit on a list of ingredient-lists (each a list of strings),
    transforms to a numpy matrix of shape (n_samples, vocab_size).
    """

    def __init__(self, use_tfidf=True, min_df=2):
        self.use_tfidf = use_tfidf
        self.min_df = min_df
        self.vocab = {}          # token -> index
        self.idf = None          # idf values, shape (vocab_size,)

    def _tokenize_recipe(self, ingredients):
        """Turn a list of ingredient strings into a list of individual word tokens."""
        tokens = []
        for ing in ingredients:
            tokens.extend(clean_ingredient(ing).split())
        return tokens

    def fit(self, recipes):
        """
        recipes: list of ingredient-lists, e.g. [["chicken breast", "garlic"], ...]
        """
        doc_freq = Counter()
        all_tokens = set()

        tokenized_docs = []
        for recipe in recipes:
            tokens = set(self._tokenize_recipe(recipe))
            tokenized_docs.append(tokens)
            doc_freq.update(tokens)
            all_tokens.update(tokens)

        # keep only tokens appearing in at least min_df documents (removes noise/typos)
        kept_tokens = sorted([t for t in all_tokens if doc_freq[t] >= self.min_df])
        self.vocab = {tok: i for i, tok in enumerate(kept_tokens)}

        n_docs = len(recipes)
        vocab_size = len(self.vocab)

        if self.use_tfidf:
            self.idf = np.zeros(vocab_size)
            for tok, idx in self.vocab.items():
                df = doc_freq[tok]
                # smoothed idf, same formula sklearn uses: log((1+n)/(1+df)) + 1
                self.idf[idx] = np.log((1 + n_docs) / (1 + df)) + 1

        return self

    def transform(self, recipes):
        n_docs = len(recipes)
        vocab_size = len(self.vocab)
        X = np.zeros((n_docs, vocab_size), dtype=np.float32)

        for i, recipe in enumerate(recipes):
            tokens = self._tokenize_recipe(recipe)
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                if tok in self.vocab:
                    X[i, self.vocab[tok]] = cnt

        if self.use_tfidf:
            X = X * self.idf  # broadcast tf * idf
            # L2 normalize each row (standard tf-idf practice)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        return X

    def fit_transform(self, recipes):
        self.fit(recipes)
        return self.transform(recipes)
