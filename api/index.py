"""
Vercel serverless entry point (also runs locally with uvicorn).

Local:   uvicorn api.index:app --reload --port 8000
Vercel:  deployed automatically from /api (see vercel.json)

Trained models are loaded lazily on the first request and cached, because
serverless platforms don't reliably run startup hooks.
"""
import os
import sys
import pickle

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "models"))   # needed so pickled classes can be imported
from diet_rules import tag_diet  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")

app = FastAPI(title="Pantry API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache = {}


def _load(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_models():
    """Load (once) and return (cuisine_models, recipe_index)."""
    if "loaded" not in _cache:
        _cache["cuisine"] = _load("artifacts.pkl")
        _cache["recipes"] = _load("recipe_artifacts.pkl")
        _cache["loaded"] = True
    return _cache["cuisine"], _cache["recipes"]


class PredictRequest(BaseModel):
    ingredients: list[str]
    top_k_recipes: int = 5


def top_words_for_prediction(vectorizer, log_prob_row, x_vector, top_n=5):
    idx_to_word = {i: w for w, i in vectorizer.vocab.items()}
    nonzero_idx = np.nonzero(x_vector)[0]
    contributions = [(idx_to_word[i], log_prob_row[i] * x_vector[i]) for i in nonzero_idx if i in idx_to_word]
    contributions.sort(key=lambda x: -x[1])
    return [w for w, _ in contributions[:top_n]]


@app.post("/api/predict")
def predict(req: PredictRequest):
    artifacts, recipe_artifacts = get_models()
    ingredients = req.ingredients
    response = {}

    if artifacts is not None:
        idx_to_label = artifacts["idx_to_label"]

        x_counts = artifacts["vectorizer_counts"].transform([ingredients])
        nb_probs = artifacts["naive_bayes"].predict_proba(x_counts)[0]
        nb_idx = int(np.argmax(nb_probs))

        x_tfidf = artifacts["vectorizer_tfidf"].transform([ingredients])
        sm_probs = artifacts["softmax_regression"].predict_proba(x_tfidf)[0]
        sm_idx = int(np.argmax(sm_probs))

        nn_probs = artifacts["neural_net"].predict_proba(x_tfidf)[0]
        nn_idx = int(np.argmax(nn_probs))

        response["cuisine_predictions"] = {
            "naive_bayes": {"cuisine": idx_to_label[nb_idx], "confidence": float(nb_probs[nb_idx])},
            "softmax_regression": {"cuisine": idx_to_label[sm_idx], "confidence": float(sm_probs[sm_idx])},
            "neural_net": {"cuisine": idx_to_label[nn_idx], "confidence": float(nn_probs[nn_idx])},
        }
        response["top_contributing_ingredients"] = top_words_for_prediction(
            artifacts["vectorizer_counts"],
            artifacts["naive_bayes"].feature_log_prob_[nb_idx],
            x_counts[0],
        )

    if recipe_artifacts is not None:
        response["recipe_matches"] = recipe_artifacts["retriever"].query(ingredients, top_k=req.top_k_recipes)

    response["diet_tags"] = tag_diet(ingredients)
    return response


@app.get("/api/health")
def health():
    artifacts, recipe_artifacts = get_models()
    return {"status": "ok", "cuisine_models_loaded": artifacts is not None,
            "recipe_index_loaded": recipe_artifacts is not None}
