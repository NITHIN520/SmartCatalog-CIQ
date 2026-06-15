# Category Tagging Automation — Complete Guide

---

## What This System Does

Takes a product name like `"Corona Extra Lager Can"` and automatically predicts:
- **Category** → `Beer`
- **Sub-Category** → `Light Lager`

It learns over time — the more products you run through it, the smarter it gets.

---

## Project File Structure

```
Category_Tagging-Automation/
│
├── main.py                          ← Entry point for all commands
├── requirements.txt                 ← Python packages needed
├── .env                             ← Your API keys and settings (create from .env.example)
├── .env.example                     ← Template for .env
│
├── my_products.json                 ← YOUR reference data (sub_category → category)
├── products_input.csv               ← Products you want to tag
├── products_output.csv              ← Results after batch prediction
│
├── src/
│   ├── config.py                    ← All settings (thresholds, file paths)
│   │
│   ├── data/
│   │   ├── dataset_manager.py       ← Reads/writes category_dataset.csv
│   │   ├── vector_store.py          ← Builds and queries faiss_index + faiss_meta
│   │   └── fuzzy_matcher.py         ← Fuzzy string matching logic
│   │
│   ├── ml/
│   │   └── classifier.py            ← XGBoost ML model (trains + predicts)
│   │
│   ├── llm/
│   │   └── llm_predictor.py         ← Sends product to OpenAI GPT and gets category
│   │
│   └── pipeline/
│       ├── predictor.py             ← Main logic: runs all 6 layers in order
│       ├── trainer.py               ← Handles seeding, retraining, corrections
│       ├── sub_category_predictor.py← Finds best matching sub_category
│       └── text_preprocessor.py    ← Keyword rules + noise word stripping
│
└── data/
    ├── raw/
    │   └── category_dataset.csv     ← The system's learned memory (grows over time)
    │
    └── trained_model/
        ├── faiss_index.pkl          ← TF-IDF similarity math (vector index)
        ├── faiss_meta.json          ← Labels for vector index (product → sub_category → category)
        ├── ml_classifier.joblib     ← Trained XGBoost ML model
        ├── label_encoder.joblib     ← Encodes category names for ML model
        └── tfidf_vectorizer.joblib  ← TF-IDF vectorizer used by ML model
```

---

## Key Data Files Explained

### `my_products.json` — Your Reference Data
This is the ground truth you provide. The system uses this to learn from.
```json
[
  {"sub_category": "Craft Beer",    "category": "Beer"},
  {"sub_category": "Light Lager",   "category": "Beer"},
  {"sub_category": "Cider",         "category": "Hard Beverage"},
  {"sub_category": "Non-alcoholic Beer", "category": "Non-alcoholic Beer"}
]
```
Update this file whenever you have new sub_categories to add.

---

### `category_dataset.csv` — The System's Memory
Every product that passes through the system gets stored here.
```
product_name               | predicted_sub_category | category           | source
Craft Beer                 | Craft Beer             | Beer               | seed
Light Lager                | Light Lager            | Beer               | seed
Corona Extra Lager Can     | Light Lager            | Beer               | keyword
Heineken Draught           | Classic Lager          | Beer               | llm
```
- `seed` = came from your `my_products.json`
- `keyword` = matched by keyword scan
- `fuzzy` = matched by fuzzy string similarity
- `vector` = matched by vector similarity
- `ml` = predicted by ML model
- `llm` = predicted by OpenAI GPT
- `human` = you manually corrected it

---

### `faiss_meta.json` — Vector Index Labels
Stores the labels used by the similarity search layer.
```json
[
  {"product_name": "Light Lager", "predicted_sub_category": "Light Lager", "category": "Beer"},
  {"product_name": "Cider",       "predicted_sub_category": "Cider",       "category": "Hard Beverage"}
]
```
Rebuilds automatically when you run `seed`, `retrain`, or `correct`.

---

### `products_input.csv` — Your Input File
Put the product names you want to tag here:
```
product_name
Corona Extra Lager Can
Heineken 0.0 Non-Alcoholic Beer
Smirnoff Vodka
```

---

### `products_output.csv` — Your Results File
Generated after running batch. Contains predictions:
```
product_name               | predicted_category | predicted_sub_category | category_confidence | sub_category_confidence
Corona Extra Lager Can     | Beer               | Light Lager            | 0.90                | 0.62
Heineken 0.0               | Non-alcoholic Beer | Non-alcoholic Beer     | 0.90                | 0.85
```

---

## How Prediction Works (Step by Step)

When you run a product name through the system, it goes through 6 layers in order.
It stops at the first layer that gives a confident answer.

```
Product Name: "Corona Extra Lager Can"
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 1: Exact Match                                             │
│ Checks: Is this product name already in category_dataset.csv?   │
│ If YES → return stored category + sub_category instantly         │
│ If NO  → go to next layer                                        │
└──────────────────────────────────────────────────────────────────┘
         │ miss
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 2: Keyword Scan (text_preprocessor.py)                     │
│ Checks: Does the name contain known words like                   │
│         "lager", "ale", "ipa", "whiskey", "wine" etc?           │
│ If YES → return category based on keyword                        │
│ If NO  → go to next layer                                        │
│ Note: Keywords are hardcoded, they do NOT learn automatically    │
└──────────────────────────────────────────────────────────────────┘
         │ miss
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 3: Fuzzy String Match (fuzzy_matcher.py)                   │
│ Checks: Is this name ≥85% similar to any known product name?    │
│ Uses RapidFuzz string matching                                   │
│ Example: "Caft Beer" matches "Craft Beer" (typo tolerance)       │
│ If match found → return that product's category                  │
│ If no match   → go to next layer                                 │
└──────────────────────────────────────────────────────────────────┘
         │ miss
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 4: Vector Similarity Search (vector_store.py)              │
│ Checks: Is this name ≥80% semantically similar to any known one? │
│ Uses TF-IDF cosine similarity (faiss_index.pkl + faiss_meta.json)│
│ Example: "IPA Craft" matches "Craft Beer" semantically           │
│ If match found → return category + sub_category from meta        │
│ If no match   → go to next layer                                 │
└──────────────────────────────────────────────────────────────────┘
         │ miss
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 5: ML Model (classifier.py)                                │
│ XGBoost model trained on all known product names                 │
│ Predicts category with a confidence score                        │
│ If confidence ≥75% → return prediction                           │
│ If confidence <75% → go to next layer                            │
│ Note: Learns automatically when you run retrain                  │
└──────────────────────────────────────────────────────────────────┘
         │ low confidence
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 6: LLM — OpenAI GPT-4o-mini (llm_predictor.py)            │
│ Sends the product name + your full reference list to GPT         │
│ GPT reads the list and reasons which category fits best          │
│ Returns category + confidence                                    │
│ Requires OPENAI_API_KEY in .env file                             │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ SUB-CATEGORY PREDICTION (sub_category_predictor.py)              │
│ After category is found, finds the closest matching sub_category │
│ from your reference list within that category                    │
│ Uses: Exact → Fuzzy → Vector → LLM → Fallback                   │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ STORE RESULT (dataset_manager.py)                                │
│ Saves to category_dataset.csv:                                   │
│   product_name, predicted_sub_category, category, source,        │
│   confidence, sub_confidence, timestamp                          │
│ Next time same product comes → found in Layer 1 instantly        │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
    products_output.csv ✅
```

---

## Speed and Cost of Each Layer

| Layer | Speed | Cost | Learns? |
|---|---|---|---|
| Exact Match | Instant | Free | ✅ Grows with every prediction |
| Keyword Scan | Instant | Free | ❌ Hardcoded, never updates |
| Fuzzy Match | Instant | Free | ✅ Grows with every prediction |
| Vector Search | Fast | Free | ✅ Rebuilds on retrain |
| ML Model | Fast | Free | ✅ Retrains periodically |
| LLM (GPT) | ~1 second | Paid | ❌ (but saves result to dataset) |

Over time: more products in dataset → more Exact/Fuzzy hits → fewer LLM calls → lower cost.

---

## All Commands

### Setup (Run Once)
```bash
# 1. Create virtual environment
python3.13 -m venv .venv

# 2. Install packages
.venv/bin/pip install -r requirements.txt

# 3. Create .env file and add your OpenAI key
cp .env.example .env
# Open .env and add: OPENAI_API_KEY=sk-your-key-here
```

---

### `seed` — Load your reference data and train first model
```bash
.venv/bin/python main.py seed --file my_products.json
```
**What it does:**
- Reads `my_products.json`
- Writes seed rows to `category_dataset.csv`
- Trains the ML model
- Builds `faiss_meta.json` and `faiss_index.pkl`

**When to run:**
- First time you set up the system
- When you add new sub_categories to `my_products.json`

---

### `batch` — Tag a list of products
```bash
.venv/bin/python main.py batch products_input.csv --output products_output.csv
```
**What it does:**
- Reads every product name from `products_input.csv`
- Runs each through all 6 prediction layers
- Writes results to `products_output.csv`
- Appends new predictions to `category_dataset.csv`

**When to run:**
- Every time you have a new list of products to tag

**Options:**
```bash
# Don't save predictions to dataset (dry run)
.venv/bin/python main.py batch products_input.csv --output products_output.csv --no-store
```

---

### `predict` — Tag a single product (quick test)
```bash
.venv/bin/python main.py predict "Corona Extra Lager Can"
```
**What it does:**
- Runs prediction for one product
- Prints result in terminal
- Saves to dataset

**When to run:**
- Quick testing of a single product

---

### `correct` — Fix a wrong prediction
```bash
.venv/bin/python main.py correct "ProductName" "CorrectCategory"
```
Example:
```bash
.venv/bin/python main.py correct "Corona Extra Lager Can" "Beer"
```
**What it does:**
- Updates `category_dataset.csv` with the correct label
- Immediately retrains the ML model
- Rebuilds `faiss_meta.json`

**When to run:**
- When `products_output.csv` shows a wrong prediction
- Correction takes effect immediately for future predictions

---

### `retrain` — Retrain model with all accumulated data
```bash
.venv/bin/python main.py retrain
```
**What it does:**
- Reads all rows from `category_dataset.csv` (seed + predicted)
- Retrains the ML model on everything
- Rebuilds `faiss_meta.json` and `faiss_index.pkl`

**When to run:**
- After running many batch predictions
- To make the ML model smarter with new data
- Auto-triggers after 20+ new predictions by default

---

### `stats` — See dataset statistics
```bash
.venv/bin/python main.py stats
```
Shows: total rows, category breakdown, source counts, human corrections.

---

### `status` — Check system is ready
```bash
.venv/bin/python main.py status
```
Shows: dataset loaded, ML model trained, LLM available, thresholds.

---

## Recommended Workflow

```
STEP 1 — First time only
  Edit my_products.json with your sub_category → category reference list
  Run: python main.py seed --file my_products.json

STEP 2 — Tag your products
  Add product names to products_input.csv
  Run: python main.py batch products_input.csv --output products_output.csv

STEP 3 — Review results
  Open products_output.csv
  Look for low confidence scores or wrong predictions

STEP 4 — Fix wrong predictions
  Run: python main.py correct "wrong product name" "correct category"

STEP 5 — Retrain after many predictions
  Run: python main.py retrain

STEP 6 — Repeat from Step 2
  The system gets smarter with every cycle
```

---

## Settings (`.env` file)

| Setting | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Your OpenAI key for LLM layer |
| `OPENAI_MODEL` | `gpt-4o-mini` | Which GPT model to use |
| `ML_CONFIDENCE_THRESHOLD` | `0.75` | Min ML confidence to trust ML result |
| `FUZZY_MATCH_THRESHOLD` | `85` | Min fuzzy score (0-100) to trust fuzzy match |
| `MIN_SAMPLES_TO_RETRAIN` | `20` | New predictions needed before auto-retrain |

---

## Self-Learning Loop

```
Run 1: "Quilmes Cristal" → LLM says Beer → stored in dataset
Run 2: "Quilmes Cristal" → Exact match found → instant, free ✅
Run 3: "Quilmes Cristal Light" → Fuzzy match finds "Quilmes Cristal" → free ✅
Run 4: After retrain → ML model now knows this pattern → free ✅
```

The system gets faster and cheaper with every run.
After enough data, the LLM is rarely needed.
