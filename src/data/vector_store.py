"""
Vector Store (TF-IDF + Cosine Similarity)
------------------------------------------
Lightweight semantic similarity index using sklearn's TF-IDF + cosine
similarity. No external model download, no GPU/MPS, no segfaults.

For short category-style text (1-5 words) this performs comparably to
full sentence-transformers while being much faster to initialise.

Used as the *semantic lookup layer* before hitting the ML model or LLM.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    VECTOR_INDEX_PATH,
    VECTOR_META_PATH,
    VECTOR_SIMILARITY_THRESHOLD,
    VECTOR_TOP_K,
)

logger = logging.getLogger(__name__)

# Swap the binary FAISS file for a pickle of the sklearn matrix
_TFIDF_PATH = VECTOR_INDEX_PATH.with_suffix(".pkl")

_vectorizer = None       # lazy-loaded TfidfVectorizer
_matrix = None           # sparse TF-IDF matrix (n_items × vocab)
_meta: list[dict] = []  # [{sub_category, category}, ...]


# ── Build / Rebuild ───────────────────────────────────────────────────────────

def build_index(
    product_names: list[str],
    categories: list[str],
    sub_categories: list[str] | None = None,
) -> None:
    """
    (Re)build the TF-IDF similarity index from the full labelled dataset.
    Called during seeding and after retraining.
    sub_categories: the predicted_sub_category for each entry (optional).
    """
    global _vectorizer, _matrix, _meta

    if not product_names:
        logger.warning("No data to build vector index.")
        return

    from sklearn.feature_extraction.text import TfidfVectorizer

    vect = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=20_000,
        sublinear_tf=True,
    )
    matrix = vect.fit_transform(product_names)

    _sub = sub_categories or product_names   # for seed rows sub_category == product_name
    meta = [
        {"product_name": pn, "predicted_sub_category": sc, "category": cat}
        for pn, sc, cat in zip(product_names, _sub, categories)
    ]

    # Persist
    _TFIDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TFIDF_PATH, "wb") as f:
        pickle.dump({"vectorizer": vect, "matrix": matrix, "meta": meta}, f)
    VECTOR_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    _vectorizer, _matrix, _meta = vect, matrix, meta
    logger.info("TF-IDF vector index built with %d entries.", len(meta))


def _ensure_loaded() -> None:
    global _vectorizer, _matrix, _meta
    if _vectorizer is not None:
        return
    if _TFIDF_PATH.exists():
        with open(_TFIDF_PATH, "rb") as f:
            data = pickle.load(f)
        _vectorizer = data["vectorizer"]
        _matrix = data["matrix"]
        _meta = data["meta"]
        logger.debug("TF-IDF index loaded (%d entries).", len(_meta))


# ── Query ─────────────────────────────────────────────────────────────────────

def lookup(sub_category: str) -> Optional[dict]:
    """
    Search the index for the closest known sub_category.
    Returns dict {sub_category, category, score} or None if below threshold.
    """
    _ensure_loaded()
    if _vectorizer is None or _matrix is None:
        return None

    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = _vectorizer.transform([sub_category])
    scores = cosine_similarity(query_vec, _matrix).flatten()

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= VECTOR_SIMILARITY_THRESHOLD and best_idx < len(_meta):
        hit = _meta[best_idx]
        logger.debug(
            "Vector hit: '%s' → '%s' → '%s' (score=%.3f)",
            sub_category, hit.get("predicted_sub_category"), hit["category"], best_score,
        )
        return {
            "product_name": hit.get("product_name", ""),
            "predicted_sub_category": hit.get("predicted_sub_category", hit.get("product_name", "")),
            "category": hit["category"],
            "score": best_score,
        }

    logger.debug(
        "Vector miss for '%s' (best score=%.3f < %.2f).",
        sub_category, best_score, VECTOR_SIMILARITY_THRESHOLD,
    )
    return None


def top_k_lookup(sub_category: str, k: int = VECTOR_TOP_K) -> list[dict]:
    """Returns top-k results regardless of threshold (for debugging)."""
    _ensure_loaded()
    if _vectorizer is None or _matrix is None:
        return []

    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = _vectorizer.transform([sub_category])
    scores = cosine_similarity(query_vec, _matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:k]

    return [
        {**_meta[i], "score": float(scores[i])}
        for i in top_indices
        if i < len(_meta)
    ]
