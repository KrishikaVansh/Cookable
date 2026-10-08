"""
Builds the recipe retrieval index from a recipe dataset (title + ingredients + instructions).

This is a SEPARATE dataset from train.json (which only has ingredients + cuisine,
no dish names or instructions). See README for dataset sources.

Usage:
    python train_recipes.py --data ../data/recipes.json
"""
import json
import argparse
import pickle

from recipe_retrieval import RecipeRetriever


def load_recipes(path):
    """
    Expected format: list of dicts, e.g.
    [
      {
        "title": "Classic Margherita Pizza",
        "ingredients": ["2 cups flour", "1 cup tomato sauce", "fresh mozzarella", ...],
        "instructions": "1. Preheat oven... 2. Roll out dough... 3. ...",
        "cuisine": "italian"   // optional
      },
      ...
    ]
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main(data_path, out_dir):
    print("Loading recipe data...")
    recipes = load_recipes(data_path)
    print(f"  {len(recipes)} recipes loaded")

    print("\nBuilding retrieval index (TF-IDF + cosine similarity, from scratch)...")
    retriever = RecipeRetriever(min_df=2 if len(recipes) > 1000 else 1)
    retriever.fit(recipes)
    print(f"  Vocab size: {len(retriever.vocab)}")

    print("\nSanity check — querying with a sample ingredient list...")
    sample_ingredients = (recipes[0].get("ner") or recipes[0]["ingredients"])[:3]
    results = retriever.query(sample_ingredients, top_k=3)
    for r in results:
        print(f"  {r['title']}  (similarity={r['similarity']:.3f}, "
              f"match={r['ingredient_match_pct']*100:.0f}%)")

    with open(f"{out_dir}/recipe_artifacts.pkl", "wb") as f:
        pickle.dump({"retriever": retriever}, f)
    print(f"\nSaved to {out_dir}/recipe_artifacts.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/recipes.json")
    parser.add_argument("--out", default="../data")
    args = parser.parse_args()
    main(args.data, args.out)
