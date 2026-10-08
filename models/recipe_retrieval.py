"""
Recipe Retrieval Engine — from scratch, sparse inverted-index TF-IDF search.

Scales to 50k+ recipes without a giant dense matrix:
  * each ingredient word has a "postings list": which recipes contain it + its weight
  * a query only touches the postings of the words the user typed
  * cosine similarity is accumulated manually with numpy (no sklearn)

Matching uses the clean NER ingredient names (e.g. "brown sugar"), while the
full lines ("1 c. firmly packed brown sugar") are kept for display.
"""
import math
from collections import Counter

import numpy as np

from features import clean_ingredient

# words that don't identify an ingredient (units, prep words, filler)
STOPWORDS = {
    "cup", "cups", "c", "tablespoon", "tablespoons", "tbsp", "teaspoon", "teaspoons",
    "tsp", "gram", "grams", "g", "kg", "ounce", "ounces", "oz", "pound", "pounds",
    "lb", "lbs", "ml", "l", "liter", "liters", "pinch", "dash", "package", "packages",
    "pkg", "can", "cans", "jar", "box", "carton", "container", "clove", "cloves",
    "slice", "slices", "piece", "pieces", "chopped", "sliced", "diced", "minced",
    "crushed", "grated", "shredded", "fresh", "large", "small", "medium", "whole",
    "ground", "finely", "coarsely", "optional", "to", "taste", "of", "and", "or",
    "a", "the", "into", "for", "divided", "plus", "more", "room", "temperature",
    "cut", "halves", "half", "frozen", "cooked", "boiled", "melted", "soft", "firm",
    "packed", "drained", "inch", "thick", "extra",
}


def stem(tok):
    """Tiny plural normalizer applied to BOTH recipes and queries (eggs -> egg)."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("oes"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def essential_tokens(text):
    """'2 cups chopped fresh Basil leaves' -> {'basil', 'leaf'}"""
    toks = clean_ingredient(text).split()
    return {stem(t) for t in toks if t not in STOPWORDS and len(t) > 1}


class RecipeRetriever:
    def __init__(self, min_df=2, n_candidates=50, coverage_weight=0.5):
        self.min_df = min_df
        self.n_candidates = n_candidates        # cosine shortlist size
        self.coverage_weight = coverage_weight  # blend of cosine vs. ingredient coverage
        self.recipes = []
        self.vocab = {}       # token -> id
        self.idf = None       # (vocab,)
        self.postings = {}    # token id -> (doc_ids int32 array, weights float32 array)

    # ---------- indexing ----------
    @staticmethod
    def _recipe_tokens(recipe):
        names = recipe.get("ner") or recipe["ingredients"]
        toks = set()
        for n in names:
            toks |= essential_tokens(n)
        return toks

    def fit(self, recipes):
        self.recipes = recipes
        n_docs = len(recipes)

        doc_tokens = [self._recipe_tokens(r) for r in recipes]
        df = Counter()
        for toks in doc_tokens:
            df.update(toks)

        kept = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab = {t: i for i, t in enumerate(kept)}
        self.idf = np.array(
            [math.log((1 + n_docs) / (1 + df[t])) + 1 for t in kept], dtype=np.float32
        )

        # build postings lists (binary tf * idf, L2-normalized per recipe)
        post_docs = {i: [] for i in range(len(kept))}
        post_w = {i: [] for i in range(len(kept))}
        for d, toks in enumerate(doc_tokens):
            ids = [self.vocab[t] for t in toks if t in self.vocab]
            if not ids:
                continue
            norm = math.sqrt(sum(float(self.idf[i]) ** 2 for i in ids))
            for i in ids:
                post_docs[i].append(d)
                post_w[i].append(float(self.idf[i]) / norm)

        self.postings = {
            i: (np.array(post_docs[i], dtype=np.int32), np.array(post_w[i], dtype=np.float32))
            for i in post_docs if post_docs[i]
        }
        return self

    # ---------- search ----------
    def query(self, user_ingredients, top_k=5):
        user_tokens = set()
        for ing in user_ingredients:
            user_tokens |= essential_tokens(ing)

        q_ids = [self.vocab[t] for t in user_tokens if t in self.vocab]
        if not q_ids:
            return []

        q_norm = math.sqrt(sum(float(self.idf[i]) ** 2 for i in q_ids))
        scores = np.zeros(len(self.recipes), dtype=np.float32)
        for i in q_ids:
            if i in self.postings:
                docs, w = self.postings[i]
                scores[docs] += w * (float(self.idf[i]) / q_norm)   # cosine accumulation

        # shortlist by cosine, then re-rank by blend of cosine + coverage
        n_cand = min(self.n_candidates, len(scores))
        cand = np.argpartition(-scores, n_cand - 1)[:n_cand]
        cand = cand[scores[cand] > 0]

        results = []
        for d in cand:
            r = self.recipes[d]
            names = r.get("ner") or r["ingredients"]
            matched, missing = [], []
            for name in names:
                (matched if essential_tokens(name) & user_tokens else missing).append(name)
            coverage = len(matched) / len(names) if names else 0.0
            final = (1 - self.coverage_weight) * float(scores[d]) + self.coverage_weight * coverage
            results.append({
                "title": r["title"],
                "instructions": r.get("instructions", ""),
                "cuisine": r.get("cuisine"),
                "similarity": float(scores[d]),
                "ingredient_match_pct": coverage,
                "matched_ingredients": matched,
                "missing_ingredients": missing,
                "all_ingredients": r["ingredients"],
                "_score": final,
            })

        results.sort(key=lambda x: -x["_score"])
        for x in results:
            del x["_score"]
        return results[:top_k]
