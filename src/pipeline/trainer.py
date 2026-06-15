"""
Retraining / Feedback Loop Manager
------------------------------------
Handles:
  - Initial training from seed data
  - Triggered retraining when new samples accumulate
  - Rebuilding vector index after retraining
  - Applying human corrections and triggering retraining

This is the MOST IMPORTANT part of the self-learning loop.
"""
from __future__ import annotations

import logging
from datetime import datetime

from src.config import MIN_SAMPLES_TO_RETRAIN
from src.data import dataset_manager, fuzzy_matcher, vector_store
from src.ml import classifier

logger = logging.getLogger(__name__)


def _refresh_lookup_tables() -> None:
    """Sync fuzzy matcher and vector store from the current dataset."""
    X, y = dataset_manager.get_training_data()
    if X:
        # Also pull predicted_sub_category to store in the vector index
        df = dataset_manager.load()
        priority = {"human": 0, "seed": 1, "llm": 2, "ml": 3}
        df["_pri"] = df["source"].map(priority).fillna(3)
        df = df.sort_values("_pri").drop_duplicates(subset="product_name", keep="first")
        sub_cats = df["predicted_sub_category"].tolist()

        fuzzy_matcher.update_lookup(X, y)
        vector_store.build_index(X, y, sub_categories=sub_cats)
    else:
        logger.warning("No training data found – lookup tables not updated.")


def bootstrap(seed_records: list[dict]) -> dict:
    """
    One-time (or idempotent) bootstrap:
      1. Seed dataset with ground-truth records
      2. Train ML model
      3. Build vector index

    Parameters
    ----------
    seed_records : List of {"sub_category": ..., "category": ...} dicts.
    """
    logger.info("Bootstrapping system with %d seed records …", len(seed_records))

    # 1. Seed dataset
    dataset_manager.seed_from_list(seed_records)

    # 2. Train / retrain
    metrics = _train_and_refresh()
    return {**metrics, "seeded": len(seed_records)}


def _train_and_refresh() -> dict:
    X, y = dataset_manager.get_training_data()
    if not X:
        return {"status": "no_data"}

    metrics = classifier.train(X, y)
    _refresh_lookup_tables()
    logger.info("System refreshed. Dataset size=%d.", len(X))
    return metrics


def maybe_retrain() -> dict:
    """
    Check if enough new ML/LLM predictions have accumulated since last retrain.
    If so, retrain the model.
    Called automatically after each batch of predictions (optional).
    """
    df = dataset_manager.load()
    if df.empty:
        return {"status": "no_data"}

    # Count rows added by ml/llm (not yet from a retrain-triggered pass)
    new_rows = df[df["source"].isin(["ml", "llm", "vector", "fuzzy"])]
    if len(new_rows) >= MIN_SAMPLES_TO_RETRAIN:
        logger.info(
            "Retraining triggered: %d new samples accumulated (threshold=%d).",
            len(new_rows), MIN_SAMPLES_TO_RETRAIN,
        )
        return _train_and_refresh()

    logger.debug(
        "Retraining not triggered: %d / %d samples.",
        len(new_rows), MIN_SAMPLES_TO_RETRAIN,
    )
    return {"status": "skipped", "new_samples": len(new_rows), "threshold": MIN_SAMPLES_TO_RETRAIN}


def force_retrain() -> dict:
    """Unconditionally retrain from the full current dataset."""
    logger.info("Force retraining …")
    return _train_and_refresh()


def apply_correction(sub_category: str, correct_category: str) -> dict:
    """
    Human correction workflow:
      1. Update the dataset with the correct label
      2. Immediately retrain the model so the correction is reflected now
    """
    updated = dataset_manager.apply_human_correction(sub_category, correct_category)
    if updated:
        logger.info(
            "Human correction: '%s' → '%s'. Retraining …", sub_category, correct_category
        )
        metrics = _train_and_refresh()
        return {**metrics, "correction_applied": True}
    return {"correction_applied": False}
