"""
Prediction Orchestrator
------------------------
Implements the full decision waterfall:

  Input sub_category
       │
       ▼
  1. Exact / Fuzzy match  ──hit──► return (source=dataset)
       │ miss
       ▼
  2. Vector similarity search ──hit──► return (source=vector)
       │ miss
       ▼
  3. ML Classifier
       │ high confidence ──────────► return (source=ml)
       │ low confidence (keep hint)
       ▼
  4. LLM fallback ────────────────► return (source=llm)
       │ error
       ▼
  5. Best-effort: return lowest-confidence ML or "Unknown"

After each prediction the result is stored in the dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.data import dataset_manager, fuzzy_matcher, vector_store
from src.ml import classifier
from src.llm import llm_predictor
from src.pipeline.text_preprocessor import clean_product_name, extract_category_hint
from src.pipeline.sub_category_predictor import predict_sub_category

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    sub_category: str
    category: str
    confidence: float
    source: str                          # dataset | keyword | vector | fuzzy | ml | llm | fallback
    predicted_sub_category: str = ""     # best-matching sub_category from the known list
    sub_confidence: float = 0.0
    sub_source: str = ""
    low_confidence: bool = False
    steps_tried: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_known_categories() -> list[str]:
    return llm_predictor.get_known_categories_from_dataset()


def _enrich_with_sub_category(result: PredictionResult, cleaned_name: str) -> PredictionResult:
    """Attach predicted_sub_category to an already-resolved PredictionResult."""
    sub = predict_sub_category(
        product_name=result.sub_category,
        category=result.category,
        cleaned_name=cleaned_name,
    )
    result.predicted_sub_category = sub["sub_category"]
    result.sub_confidence = sub["sub_confidence"]
    result.sub_source = sub["sub_source"]
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def predict(sub_category: str, store_result: bool = True) -> PredictionResult:
    """
    Predict the category for a single sub_category string.

    Parameters
    ----------
    sub_category : The input text (e.g. "Craft Beer").
    store_result : Whether to persist the prediction to the dataset.
                   Set False for dry-runs / evaluation.
    """
    sub_category = sub_category.strip()
    steps: list[str] = []

    # ── Pre-processing: keyword hint + cleaned text ───────────────────────────
    keyword_hint = extract_category_hint(sub_category)
    cleaned_input = clean_product_name(sub_category)
    # Use cleaned text for fuzzy/vector/ML; keep original for display & storage
    lookup_text = cleaned_input if cleaned_input != sub_category else sub_category
    logger.debug("Input: '%s' → cleaned: '%s', hint: %s", sub_category, lookup_text, keyword_hint)

    # ── Layer 1: Exact dataset lookup ─────────────────────────────────────────
    steps.append("dataset_exact")
    X, y = dataset_manager.get_training_data()   # X = product_names, y = categories
    lower_map = {x.lower(): cat for x, cat in zip(X, y)}
    exact = lower_map.get(sub_category.lower()) or lower_map.get(lookup_text.lower())
    if exact:
        logger.info("[DATASET-EXACT] '%s' → '%s'", sub_category, exact)
        return _enrich_with_sub_category(PredictionResult(
            sub_category=sub_category,
            category=exact,
            confidence=1.0,
            source="dataset",
            steps_tried=steps,
        ), lookup_text)

    # ── Layer 1b: Keyword shortcut ────────────────────────────────────────────
    if keyword_hint:
        known_cats_set = set(_get_known_categories())
        if keyword_hint in known_cats_set:
            logger.info("[KEYWORD] '%s' → '%s'", sub_category, keyword_hint)
            result = _enrich_with_sub_category(PredictionResult(
                sub_category=sub_category,
                category=keyword_hint,
                confidence=0.90,
                source="keyword",
                steps_tried=steps,
            ), lookup_text)
            if store_result:
                dataset_manager.append_prediction(
                    sub_category, keyword_hint, "keyword", 0.90,
                    predicted_sub_category=result.predicted_sub_category,
                    sub_confidence=result.sub_confidence,
                )
            return result

    # ── Layer 2: Fuzzy string matching ────────────────────────────────────────
    steps.append("fuzzy")
    fuzzy_result = fuzzy_matcher.match(lookup_text)
    if fuzzy_result:
        logger.info(
            "[FUZZY] '%s' → '%s' (score=%.2f)",
            sub_category, fuzzy_result["category"], fuzzy_result["score"],
        )
        result = _enrich_with_sub_category(PredictionResult(
            sub_category=sub_category,
            category=fuzzy_result["category"],
            confidence=fuzzy_result["score"],
            source="fuzzy",
            steps_tried=steps,
        ), lookup_text)
        if store_result:
            dataset_manager.append_prediction(
                sub_category, result.category, "fuzzy", result.confidence,
                predicted_sub_category=result.predicted_sub_category,
                sub_confidence=result.sub_confidence,
            )
        return result

    # ── Layer 3: Vector semantic search ──────────────────────────────────────
    steps.append("vector")
    vec_result = vector_store.lookup(lookup_text)
    if vec_result:
        logger.info(
            "[VECTOR] '%s' → '%s' (score=%.3f)",
            sub_category, vec_result["category"], vec_result["score"],
        )
        result = _enrich_with_sub_category(PredictionResult(
            sub_category=sub_category,
            category=vec_result["category"],
            confidence=vec_result["score"],
            source="vector",
            steps_tried=steps,
        ), lookup_text)
        if store_result:
            dataset_manager.append_prediction(
                sub_category, result.category, "vector", result.confidence,
                predicted_sub_category=result.predicted_sub_category,
                sub_confidence=result.sub_confidence,
            )
        return result

    # ── Layer 4: ML classifier ────────────────────────────────────────────────
    ml_hint: Optional[str] = None
    steps.append("ml")
    ml_result = classifier.predict(lookup_text)
    if ml_result and not ml_result.get("low_confidence"):
        logger.info(
            "[ML] '%s' → '%s' (confidence=%.3f)",
            sub_category, ml_result["category"], ml_result["confidence"],
        )
        result = _enrich_with_sub_category(PredictionResult(
            sub_category=sub_category,
            category=ml_result["category"],
            confidence=ml_result["confidence"],
            source="ml",
            steps_tried=steps,
        ), lookup_text)
        if store_result:
            dataset_manager.append_prediction(
                sub_category, result.category, "ml", result.confidence,
                predicted_sub_category=result.predicted_sub_category,
                sub_confidence=result.sub_confidence,
            )
        return result

    if ml_result:
        ml_hint = ml_result["category"]
        logger.debug("ML low-confidence hint: '%s'", ml_hint)

    # ── Layer 5: LLM fallback ─────────────────────────────────────────────────
    steps.append("llm")
    known_cats = _get_known_categories()
    llm_result = llm_predictor.predict(sub_category, known_cats, ml_hint=ml_hint)
    if llm_result:
        logger.info(
            "[LLM] '%s' → '%s' (confidence=%.3f)",
            sub_category, llm_result["category"], llm_result["confidence"],
        )
        result = _enrich_with_sub_category(PredictionResult(
            sub_category=sub_category,
            category=llm_result["category"],
            confidence=llm_result["confidence"],
            source="llm",
            low_confidence=llm_result["confidence"] < 0.5,
            steps_tried=steps,
        ), lookup_text)
        if store_result:
            dataset_manager.append_prediction(
                sub_category, result.category, "llm", result.confidence,
                predicted_sub_category=result.predicted_sub_category,
                sub_confidence=result.sub_confidence,
            )
        return result

    # ── Layer 6: Hard fallback ────────────────────────────────────────────────
    steps.append("fallback")
    if ml_hint:
        category, confidence = ml_hint, ml_result["confidence"]
    elif known_cats:
        category, confidence = known_cats[0], 0.0
    else:
        category, confidence = "Unknown", 0.0

    logger.warning("[FALLBACK] '%s' → '%s'", sub_category, category)
    return _enrich_with_sub_category(PredictionResult(
        sub_category=sub_category,
        category=category,
        confidence=confidence,
        source="fallback",
        low_confidence=True,
        steps_tried=steps,
    ), lookup_text)


def predict_batch(sub_categories: list[str], store_result: bool = True) -> list[PredictionResult]:
    """
    Predict categories for a list of sub_categories.

    Uses an in-batch cache: if a product is very similar to one already
    predicted in this batch (rapidfuzz score ≥ 92), it reuses the category
    instead of making a redundant LLM call.
    """
    from rapidfuzz import process, fuzz

    results: list[PredictionResult] = []
    # Maps already-seen product names → their PredictionResult (for cache lookup)
    batch_cache: dict[str, PredictionResult] = {}

    for sc in sub_categories:
        sc_stripped = sc.strip()

        # Check in-batch cache first (only applies when LLM would be called)
        cached: Optional[PredictionResult] = None
        if batch_cache:
            best = process.extractOne(
                sc_stripped,
                list(batch_cache.keys()),
                scorer=fuzz.token_sort_ratio,
            )
            if best and best[1] >= 92:
                cached = batch_cache[best[0]]
                logger.info(
                    "[BATCH-CACHE] '%s' → '%s' (matched '%s', score=%d)",
                    sc_stripped, cached.category, best[0], best[1],
                )

        if cached is not None:
            # Reuse category/sub from cache but create a fresh result for this product
            result = _enrich_with_sub_category(PredictionResult(
                sub_category=sc_stripped,
                category=cached.category,
                confidence=cached.confidence,
                source="batch_cache",
                steps_tried=["batch_cache"],
            ), clean_product_name(sc_stripped))
            if store_result:
                dataset_manager.append_prediction(
                    sc_stripped, result.category, "batch_cache", result.confidence,
                    predicted_sub_category=result.predicted_sub_category,
                    sub_confidence=result.sub_confidence,
                )
        else:
            result = predict(sc_stripped, store_result=store_result)
            batch_cache[sc_stripped] = result

        results.append(result)

    return results
