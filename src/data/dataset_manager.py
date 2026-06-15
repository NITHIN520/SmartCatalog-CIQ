"""
Dataset Manager
---------------
Single source of truth for all labelled (product_name → sub_category → category) pairs.
Handles loading, appending, deduplication, and human-correction updates.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config import DATASET_PATH

logger = logging.getLogger(__name__)

COLUMNS = [
    "product_name",           # input text (full product name)
    "predicted_sub_category", # closest known sub_category from seed list
    "category",               # top-level predicted category
    "source",                 # "seed" | "keyword" | "ml" | "llm" | "human"
    "confidence",             # category confidence (float 0-1)
    "sub_confidence",         # sub_category confidence (float 0-1)
    "timestamp",
    "corrected",              # bool – True if a human overrode this row
]


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def load() -> pd.DataFrame:
    """Load the dataset. Returns empty DataFrame if file doesn't exist."""
    if not DATASET_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(DATASET_PATH, dtype=str)
    # Backwards-compat: rename old 'sub_category' column if present
    if "sub_category" in df.columns and "product_name" not in df.columns:
        df = df.rename(columns={"sub_category": "product_name"})
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["sub_confidence"] = pd.to_numeric(df["sub_confidence"], errors="coerce")
    df["corrected"] = df["corrected"].map({"True": True, "False": False, True: True, False: False})
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate product_name rows, keeping the highest-priority source.
    Priority: human > seed > llm > batch_cache > vector > fuzzy > ml > keyword
    """
    if df.empty:
        return df
    priority = {"human": 0, "seed": 1, "llm": 2, "batch_cache": 3, "vector": 4, "fuzzy": 5, "ml": 6, "keyword": 7}
    df = df.copy()
    df["_pri"] = df["source"].map(priority).fillna(8)
    df = df.sort_values("_pri").drop_duplicates(subset="product_name", keep="first")
    df = df.drop(columns=["_pri"]).reset_index(drop=True)
    return df


def save(df: pd.DataFrame) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = _deduplicate(df)
    df.to_csv(DATASET_PATH, index=False)
    logger.debug("Dataset saved (%d rows).", len(df))


def seed_from_list(records: list[dict]) -> pd.DataFrame:
    """
    Bootstrap the dataset from a list of dicts:
      [{"sub_category": "Craft Beer", "category": "Beer"}, ...]
    Only inserts rows that don't already exist.
    Returns the final full DataFrame.
    """
    df = load()
    existing_keys = set(df["product_name"].str.lower())

    new_rows = []
    for r in records:
        name = r["sub_category"].strip()   # seed uses sub_category as the product_name
        key = name.lower()
        if key not in existing_keys:
            new_rows.append(
                {
                    "product_name": name,
                    "predicted_sub_category": name,  # for seed rows, sub_category = itself
                    "category": r["category"].strip(),
                    "source": "seed",
                    "confidence": None,
                    "sub_confidence": None,
                    "timestamp": _now(),
                    "corrected": False,
                }
            )
            existing_keys.add(key)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save(df)
        logger.info("Seeded %d new rows into dataset.", len(new_rows))
    return df


def append_prediction(
    product_name: str,
    category: str,
    source: str,
    confidence: Optional[float] = None,
    predicted_sub_category: Optional[str] = None,
    sub_confidence: Optional[float] = None,
) -> None:
    """
    Store a new prediction result.
    - If an authoritative (seed/human) row already exists → skip entirely.
    - If a non-authoritative row already exists → UPDATE it in-place (no duplicate).
    - Otherwise → append a new row.
    """
    df = load()
    key = product_name.strip().lower()
    mask = df["product_name"].str.lower() == key

    if mask.any():
        existing_source = df.loc[mask, "source"].iloc[0]
        if existing_source in ("seed", "human"):
            logger.debug("Skipping – authoritative label exists for '%s'.", product_name)
            return
        # Update the existing row in-place instead of adding a duplicate
        df.loc[mask, "category"] = category.strip()
        df.loc[mask, "source"] = source
        df.loc[mask, "confidence"] = confidence
        df.loc[mask, "predicted_sub_category"] = (predicted_sub_category or "").strip()
        df.loc[mask, "sub_confidence"] = sub_confidence
        df.loc[mask, "timestamp"] = _now()
        logger.debug("Updated existing row for '%s' (source=%s).", product_name, source)
    else:
        row = pd.DataFrame(
            [
                {
                    "product_name": product_name.strip(),
                    "predicted_sub_category": (predicted_sub_category or "").strip(),
                    "category": category.strip(),
                    "source": source,
                    "confidence": confidence,
                    "sub_confidence": sub_confidence,
                    "timestamp": _now(),
                    "corrected": False,
                }
            ]
        )
        df = pd.concat([df, row], ignore_index=True)

    save(df)


def apply_human_correction(product_name: str, correct_category: str) -> bool:
    """
    Human override: update the category for a given product_name.
    Returns True if a row was found and updated.
    """
    df = load()
    mask = df["product_name"].str.lower() == product_name.strip().lower()
    if not mask.any():
        logger.warning("No existing row for '%s' – adding as new human row.", product_name)
        row = pd.DataFrame(
            [
                {
                    "product_name": product_name.strip(),
                    "predicted_sub_category": product_name.strip(),
                    "category": correct_category.strip(),
                    "source": "human",
                    "confidence": 1.0,
                    "sub_confidence": 1.0,
                    "timestamp": _now(),
                    "corrected": True,
                }
            ]
        )
        df = pd.concat([df, row], ignore_index=True)
        save(df)
        return True

    df.loc[mask, "category"] = correct_category.strip()
    df.loc[mask, "source"] = "human"
    df.loc[mask, "confidence"] = 1.0
    df.loc[mask, "sub_confidence"] = 1.0
    df.loc[mask, "corrected"] = True
    df.loc[mask, "timestamp"] = _now()
    save(df)
    logger.info("Human correction applied for '%s' → '%s'.", product_name, correct_category)
    return True


def get_training_data() -> tuple[list[str], list[str]]:
    """
    Returns (X, y) lists for ML training.
    Prioritises human > seed rows; deduplicates by product_name.
    """
    df = load()
    if df.empty:
        return [], []

    priority = {"human": 0, "seed": 1, "llm": 2, "ml": 3}
    df["_pri"] = df["source"].map(priority).fillna(3)
    df = df.sort_values("_pri").drop_duplicates(subset="product_name", keep="first")
    df = df.drop(columns=["_pri"])

    return df["product_name"].tolist(), df["category"].tolist()


def dataset_stats() -> dict:
    df = load()
    stats = {
        "total_rows": len(df),
        "unique_categories": df["category"].nunique() if not df.empty else 0,
        "source_counts": df["source"].value_counts().to_dict() if not df.empty else {},
        "human_corrections": int(df["corrected"].sum()) if not df.empty else 0,
    }
    return stats
