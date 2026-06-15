<div align="center">

```
 ██████╗ █████╗ ████████╗███████╗ ██████╗  ██████╗ ██████╗ ██╗   ██╗
██╔════╝██╔══██╗╚══██╔══╝██╔════╝██╔════╝ ██╔═══██╗██╔══██╗╚██╗ ██╔╝
██║     ███████║   ██║   █████╗  ██║  ███╗██║   ██║██████╔╝ ╚████╔╝
██║     ██╔══██║   ██║   ██╔══╝  ██║   ██║██║   ██║██╔══██╗  ╚██╔╝
╚██████╗██║  ██║   ██║   ███████╗╚██████╔╝╚██████╔╝██║  ██║   ██║
 ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝
          T A G G I N G   A U T O M A T I O N
```

### *Drop in a product name. Get back a category. The system learns from every single run.*

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Engine-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)
![CLI](https://img.shields.io/badge/CLI-Typer%20%2B%20Rich-00A896?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

</div>

---

## What Does It Do?

Feed it a raw product name. It returns a **Category** and **Sub-Category** — automatically, in milliseconds, using a 6-layer AI pipeline that gets smarter with every prediction.

```
INPUT   →   "Corona Extra Lager Can"

OUTPUT  →   Category     : Beer
            Sub-Category : Light Lager
            Confidence   : 94.2%
            Source       : ml
```

The more products you run through it, the smarter it gets. **Zero manual effort after setup.**

---

## The 6-Layer Intelligence Pipeline

Every product name passes through these layers **in order**, stopping at the first confident match:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PRODUCT NAME ENTERS HERE                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────▼─────────────────┐
              │  LAYER 1 · Exact Match            │  ← Instant · Free
              │  Already seen this product?       │
              └──────┬───────────────────┬────────┘
                  HIT│                MISS│
                     ▼                   │
              ✅ Return instantly        │
                                         ▼
              ┌──────────────────────────────────┐
              │  LAYER 2 · Keyword Scan           │  ← Instant · Free
              │  "lager", "ipa", "whiskey"…       │
              └──────┬───────────────────┬────────┘
                  HIT│                MISS│
                     ▼                   │
              ✅ Return by rule          │
                                         ▼
              ┌──────────────────────────────────┐
              │  LAYER 3 · Fuzzy Match            │  ← Instant · Free
              │  "Caft Beer" → "Craft Beer" ✓     │
              │  RapidFuzz ≥ 85% similarity       │
              └──────┬───────────────────┬────────┘
                  HIT│                MISS│
                     ▼                   │
              ✅ Typo-tolerant match     │
                                         ▼
              ┌──────────────────────────────────┐
              │  LAYER 4 · Vector Search          │  ← Fast · Free
              │  TF-IDF cosine similarity         │
              │  Finds semantically close names   │
              └──────┬───────────────────┬────────┘
                  HIT│                MISS│
                     ▼                   │
              ✅ Semantic match          │
                                         ▼
              ┌──────────────────────────────────┐
              │  LAYER 5 · ML Classifier          │  ← Fast · Free
              │  XGBoost trained on all past data │
              │  Returns category + confidence    │
              └──────┬───────────────────┬────────┘
          ≥75% conf. │         < 75% conf│
                     ▼                   │
              ✅ ML prediction           │
                                         ▼
              ┌──────────────────────────────────┐
              │  LAYER 6 · LLM Fallback           │  ← ~1s · Paid
              │  GPT-4o-mini reasons from your    │
              │  full reference list              │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  SUB-CATEGORY PREDICTOR           │
              │  Exact → Fuzzy → Vector → LLM    │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  STORE RESULT → DATASET GROWS    │  ← Self-learning
              │  Auto-retrain after 20 new rows  │
              └──────────────┬───────────────────┘
                             │
                             ▼
                    products_output.csv ✅
```

> **The more data it accumulates, the more the cheaper/faster top layers handle requests — and the fewer expensive LLM calls are needed.**

---

## The Self-Learning Loop

```
Run 1:  "Quilmes Cristal"       → LLM answers (paid)  → saved to dataset
Run 2:  "Quilmes Cristal"       → Exact match (free)  ← remembers it ✅
Run 3:  "Quilmes Cristal Light" → Fuzzy match (free)  ← similar name ✅
Run 4+: After retrain           → ML handles it (free) ← fully learned ✅
```

Every cycle: **faster, cheaper, smarter.**

---

## Layer Performance at a Glance

| Layer | Speed | API Cost | Self-Learns? |
|---|---|---|---|
| Exact Match | ⚡ Instant | Free | ✅ Grows with every prediction |
| Keyword Scan | ⚡ Instant | Free | ❌ Hardcoded rules |
| Fuzzy Match | ⚡ Instant | Free | ✅ Grows with every prediction |
| Vector Search | 🚀 Fast | Free | ✅ Rebuilds on retrain |
| ML Classifier | 🚀 Fast | Free | ✅ Retrains periodically |
| LLM (GPT-4o-mini) | ⏱ ~1 second | Paid | ❌ (but saves result) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set your OPENAI_API_KEY
```

### 3. Seed your reference data and train the first model

```bash
python main.py seed --file my_products.json
```

Or use the built-in beverage demo data:

```bash
python main.py seed
```

### 4. Tag your products

```bash
python main.py batch products_input.csv --output products_output.csv
```

That's it. Open `products_output.csv` and your products are categorized.

---

## Commands

### `predict` — Tag a single product

```bash
python main.py predict "IPA Beer"
python main.py predict "Sparkling Wine" "Kombucha" "Energy Drinks"
```

Output (rendered in terminal with colors):

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Product / Input   ┃ Category    ┃ Sub-Category   ┃ Cat Conf ┃ Sub Conf ┃ Source ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ IPA Beer          │ Beer        │ Craft Beer     │  91.4%   │  78.2%   │ ml     │
│ Sparkling Wine    │ Wine        │ Sparkling Wine │  88.7%   │  84.1%   │ vector │
│ Kombucha          │ Non-alc…    │ Kombucha       │  76.3%   │  61.5%   │ llm    │
└───────────────────┴─────────────┴────────────────┴──────────┴──────────┴────────┘
```

---

### `batch` — Tag a whole CSV

```bash
python main.py batch products_input.csv --output products_output.csv

# Dry run — predict without saving to dataset
python main.py batch products_input.csv --output results.csv --no-store
```

Your input CSV needs a `product_name` or `sub_category` column:

```csv
product_name
Corona Extra Lager Can
Heineken 0.0 Non-Alcoholic Beer
Jack Daniel's Tennessee Whiskey
```

Output columns written in real-time as each row is processed:

```csv
product_name, predicted_category, predicted_sub_category,
category_confidence, sub_category_confidence,
category_source, sub_category_source, needs_review
```

The `needs_review` column is automatically flagged `YES` for low-confidence or LLM-sourced predictions.

---

### `correct` — Fix a wrong prediction

```bash
python main.py correct "BeatBox Wine Cocktail" "Ready to Drink"
```

This does three things instantly:
1. Updates the stored label in the dataset
2. Retrains the ML model
3. Rebuilds the vector index

**This is the most powerful way to improve accuracy.**

---

### `retrain` — Force the model to learn from all accumulated data

```bash
python main.py retrain
```

Run this after every ~50–100 new batch predictions to keep the ML model sharp.
Auto-triggers after 20 new predictions by default (configurable).

---

### `seed` — Load reference data and bootstrap the model

```bash
# From your own JSON
python main.py seed --file my_products.json

# Use built-in beverage demo data
python main.py seed
```

Your `my_products.json` format:

```json
[
  {"sub_category": "Craft Beer",       "category": "Beer"},
  {"sub_category": "Light Lager",      "category": "Beer"},
  {"sub_category": "Cider",            "category": "Hard Beverage"},
  {"sub_category": "Sparkling Water",  "category": "Non-alcoholic Beverage"}
]
```

---

### `stats` / `status` — Inspect the system

```bash
python main.py stats     # Dataset size, category breakdown, source counts
python main.py status    # Model trained? LLM available? Thresholds?
```

---

## Recommended Workflow

```
STEP 1 · One time only
  └─ Edit my_products.json   →  add your sub_category → category mappings
  └─ python main.py seed --file my_products.json

STEP 2 · Tag your products
  └─ Fill products_input.csv with product names
  └─ python main.py batch products_input.csv --output products_output.csv

STEP 3 · Review results
  └─ Open products_output.csv
  └─ Look at needs_review = YES rows
  └─ Check low confidence scores

STEP 4 · Fix wrong predictions
  └─ python main.py correct "product name" "correct category"

STEP 5 · Retrain periodically
  └─ python main.py retrain   (every 50–100 new predictions)

STEP 6 · Repeat from Step 2
  └─ The system gets smarter every cycle  ♻
```

---

## Project Structure

```
Category_Tagging-Automation/
│
├── main.py                           ← CLI entry point (all commands live here)
├── requirements.txt                  ← Python packages
├── .env                              ← Your secrets (never commit this)
├── .env.example                      ← Template — copy to .env
├── my_products.json                  ← Your ground-truth reference data
├── products_input.csv                ← Products to tag
├── products_output.csv               ← Tagged results
│
├── src/
│   ├── config.py                     ← All settings, env-driven
│   │
│   ├── data/
│   │   ├── dataset_manager.py        ← Reads / writes category_dataset.csv
│   │   ├── vector_store.py           ← Builds & queries the vector index
│   │   └── fuzzy_matcher.py          ← RapidFuzz string matching
│   │
│   ├── ml/
│   │   └── classifier.py             ← TF-IDF + XGBoost pipeline
│   │
│   ├── llm/
│   │   └── llm_predictor.py          ← OpenAI GPT-4o-mini fallback
│   │
│   └── pipeline/
│       ├── predictor.py              ← 6-layer decision waterfall
│       ├── trainer.py                ← Bootstrap, retrain, corrections
│       ├── sub_category_predictor.py ← Sub-category resolution
│       └── text_preprocessor.py     ← Keyword rules + text cleaning
│
└── data/
    ├── raw/
    │   └── category_dataset.csv      ← The system's growing memory
    └── trained_model/
        ├── ml_classifier.joblib      ← Trained XGBoost model
        ├── label_encoder.joblib      ← Category label encoder
        ├── faiss_index.pkl           ← Vector similarity index
        └── faiss_meta.json           ← Labels for the vector index
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI key — only used by Layer 6 |
| `OPENAI_MODEL` | `gpt-4o-mini` | GPT model for LLM fallback |
| `ML_CONFIDENCE_THRESHOLD` | `0.75` | Min ML confidence to trust prediction |
| `FUZZY_MATCH_THRESHOLD` | `85` | Min fuzzy score (0–100) to trust string match |
| `MIN_SAMPLES_TO_RETRAIN` | `20` | New predictions needed to trigger auto-retrain |

---

## Tech Stack

| Component | Technology |
|---|---|
| CLI Framework | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) |
| ML Model | [XGBoost](https://xgboost.readthedocs.io/) + TF-IDF (scikit-learn) |
| Vector Search | scikit-learn cosine similarity |
| Fuzzy Matching | [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) |
| LLM Fallback | [OpenAI GPT-4o-mini](https://platform.openai.com/) |
| Serialization | [joblib](https://joblib.readthedocs.io/) |
| Data | [pandas](https://pandas.pydata.org/) |

---

## Files You Ever Manually Edit

Only **three** files require human input — everything else is managed automatically:

| File | When to Edit |
|---|---|
| `my_products.json` | Adding new sub-category → category mappings |
| `products_input.csv` | Loading a new batch of product names to tag |
| `src/pipeline/text_preprocessor.py` | Adding custom keyword rules (optional) |

---

<div align="center">

**Built for scale. Designed to learn. Zero maintenance after setup.**

</div>
