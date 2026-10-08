"""
Converts the RecipeNLG CSV into data/recipes.json for the retrieval engine.

RecipeNLG columns: index, title, ingredients, directions, link, source, NER
 - ingredients / directions / NER are JSON-encoded lists stored as strings
 - NER = clean ingredient names (best for matching)
 - ingredients = full lines with quantities (best for display)

The full file has ~2.2M recipes, which is too many for a free-tier deployment,
so we randomly sample --max-recipes of them (streaming, low memory).

Usage:
    python convert_recipenlg.py --csv ../data/full_dataset.csv --out ../data/recipes.json --max-recipes 50000
"""
import csv
import json
import sys
import random
import argparse

csv.field_size_limit(2**31 - 1)  # large cells; 2**31-1 is safe on Windows (sys.maxsize overflows there)


def parse_list(cell):
    try:
        val = json.loads(cell)
        return [str(x).strip() for x in val if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        return []


def main(csv_path, out_path, max_recipes, seed, min_ner, max_ner):
    rng = random.Random(seed)
    kept = []
    seen_titles = set()
    n_read = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # first column is an unnamed index column ("") — we ignore it
        for row in reader:
            n_read += 1

            title = (row.get("title") or "").strip()
            ingredients = parse_list(row.get("ingredients", ""))
            directions = parse_list(row.get("directions", ""))
            ner = parse_list(row.get("NER", ""))

            # quality filters
            if not title or not directions:
                continue
            if not (min_ner <= len(ner) <= max_ner):
                continue
            key = title.lower()
            if key in seen_titles:
                continue

            record = {
                "title": title,
                "ingredients": ingredients,
                "ner": ner,
                "instructions": "\n".join(f"{i}. {s}" for i, s in enumerate(directions, 1)),
            }

            # reservoir sampling so we get a uniform sample of the whole file,
            # not just the first N rows (the file is ordered by source)
            if len(kept) < max_recipes:
                kept.append(record)
                seen_titles.add(key)
            else:
                j = rng.randint(0, n_read - 1)
                if j < max_recipes:
                    kept[j] = record
                    seen_titles.add(key)

            if n_read % 200000 == 0:
                print(f"  ...read {n_read:,} rows, kept {len(kept):,}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"Done. Read {n_read:,} rows -> wrote {len(kept):,} recipes to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to RecipeNLG csv")
    p.add_argument("--out", default="../data/recipes.json")
    p.add_argument("--max-recipes", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-ner", type=int, default=3, help="drop recipes with fewer ingredients")
    p.add_argument("--max-ner", type=int, default=20, help="drop recipes with more ingredients")
    a = p.parse_args()
    main(a.csv, a.out, a.max_recipes, a.seed, a.min_ner, a.max_ner)
