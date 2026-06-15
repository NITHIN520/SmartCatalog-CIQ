"""
Fuzzy String Matcher (RapidFuzz)
---------------------------------
Lightweight string-similarity layer that complements the vector store.
Catches typos and minor misspellings without needing embeddings.
"""
from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz, process

from src.config import FUZZY_MATCH_THRESHOLD

logger = logging.getLogger(__name__)

# In-memory lookup table: lower(sub_category) → category
_lookup: dict[str, str] = {}


def update_lookup(product_names: list[str], categories: list[str]) -> None:
    """Rebuild the in-memory fuzzy lookup table."""
    global _lookup
    _lookup = {pn.lower(): cat for pn, cat in zip(product_names, categories)}
    logger.debug("Fuzzy lookup table updated (%d entries).", len(_lookup))


def match(sub_category: str) -> Optional[dict]:
    """
    Attempt a fuzzy match.
    Returns dict {sub_category, category, score} or None.
    """
    if not _lookup:
        return None

    result = process.extractOne(
        sub_category.lower(),
        list(_lookup.keys()),
        scorer=fuzz.token_sort_ratio,
    )
    if result is None:
        return None

    matched_key, score, _ = result
    if score >= FUZZY_MATCH_THRESHOLD:
        logger.debug(
            "Fuzzy hit: '%s' → '%s' → '%s' (score=%d)",
            sub_category, matched_key, _lookup[matched_key], score,
        )
        return {
            "sub_category": matched_key,
            "category": _lookup[matched_key],
            "score": score / 100.0,
        }

    logger.debug("Fuzzy miss for '%s' (score=%d).", sub_category, score)
    return None
