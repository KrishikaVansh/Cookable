# Cookable — turn what you have into what you cook

Add the ingredients you have → get dishes you can cook, what you're missing, the likely
cuisine, and diet checks. Installable as an app (PWA) and deployable on Vercel.

## Project layout
```
api/index.py            FastAPI serverless function (/api/predict, /api/health)
models/                 all ML code, written from scratch with numpy (no sklearn)
public/                 website: index.html (cook), about.html, manifest, service worker, icons
data/                   trained .pkl files live here (commit these!)
vercel.json             tells Vercel to bundle models/ + data/ with the API
requirements.txt        runtime deps for Vercel   (requirements-dev.txt = local extras)
```

## Train locally (once)
```powershell
cd models
python convert_recipenlg.py --csv "C:\path\to\RecipeNLG_dataset.csv" --out ..\data\recipes.json --max-recipes 20000
python train_recipes.py --data ..\data\recipes.json --out ..\data
python train.py --data ..\data\train.json --out ..\data
```
This creates `data/artifacts.pkl` and `data/recipe_artifacts.pkl`.

## Run locally
```powershell
pip install -r requirements-dev.txt
uvicorn api.index:app --reload --port 8000      # from the project root
```
Open `public/index.html` in a browser (it talks to localhost:8000 when opened as a file).

## Deploy to Vercel
1. **Pin numpy.** Run `pip show numpy`, then set that exact version in `requirements.txt`
   (e.g. `numpy==2.1.3`). The .pkl files must be loaded by the same numpy major version
   that created them.
2. **Check sizes.** Each .pkl must be under 100 MB (GitHub limit); the whole function
   bundle must stay under 250 MB. If `recipe_artifacts.pkl` is too big, re-run the
   converter with a smaller `--max-recipes`.
3. Push the project to a GitHub repo (the .pkl files must be included; raw datasets are ignored).
4. On vercel.com: **Add New → Project → import the repo**. Framework preset: **Other**.
   No build command or output directory needed. Click **Deploy**.
5. Visit `https://<your-app>.vercel.app/api/health` — both values should be `true`.

CLI alternative: `npm i -g vercel`, then `vercel` (preview) and `vercel --prod`.

## Installing as an app
- **Android / desktop Chrome & Edge:** an "Install app" button appears in the top bar
  (or use the browser's install icon).
- **iPhone / iPad:** Safari → Share → **Add to Home Screen**.

## Notes
- The first request after a quiet period is slower (the function loads the models).
- Diet labels are keyword-based guides, not allergy guarantees.
