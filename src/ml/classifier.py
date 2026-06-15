"""
ML Classifier
-------------
Two-stage pipeline:
  1. TF-IDF (char n-grams + word n-grams) → sparse features
  2. XGBoost classifier → category probabilities

Falls back to Logistic Regression if XGBoost is unavailable.
Persisted to disk with joblib so it survives restarts.
"""
from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import (
    ML_CONFIDENCE_THRESHOLD,
    ML_MODEL_PATH,
    LABEL_ENCODER_PATH,
    VECTORIZER_PATH,
)

logger = logging.getLogger(__name__)

_pipeline: Optional[Pipeline] = None
_label_encoder: Optional[LabelEncoder] = None


def _build_pipeline() -> Pipeline:
    try:
        from xgboost import XGBClassifier
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
        )
        logger.info("Using XGBoost classifier.")
    except Exception:
        # XGBoost may raise XGBoostError (missing libomp) or ImportError
        classifier = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
        logger.info("XGBoost not available – using Logistic Regression.")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=10_000,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# ── Training ──────────────────────────────────────────────────────────────────

def train(X: list[str], y: list[str]) -> dict:
    """
    Train (or retrain) the classifier on (X, y) pairs.
    Saves model artifacts to disk.
    Returns a metrics dict.
    """
    global _pipeline, _label_encoder

    if len(set(y)) < 2:
        logger.warning("Need at least 2 classes to train. Got: %s", set(y))
        return {"status": "skipped", "reason": "insufficient_classes"}

    from sklearn.model_selection import cross_val_score

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipe = _build_pipeline()
    pipe.fit(X, y_enc)

    # Cross-validation (only if we have enough samples per class)
    min_class_count = min(np.bincount(y_enc))
    cv_folds = min(5, min_class_count)
    if cv_folds >= 2:
        scores = cross_val_score(pipe, X, y_enc, cv=cv_folds, scoring="accuracy")
        cv_acc = float(scores.mean())
    else:
        cv_acc = None

    _pipeline = pipe
    _label_encoder = le

    # Persist
    joblib.dump(pipe, ML_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    metrics = {
        "status": "trained",
        "samples": len(X),
        "classes": le.classes_.tolist(),
        "cv_accuracy": cv_acc,
    }
    cv_str = f"{cv_acc:.3f}" if cv_acc is not None else "n/a (too few samples)"
    logger.info("ML model trained. cv_accuracy=%s, classes=%s", cv_str, le.classes_.tolist())
    return metrics


# ── Loading ───────────────────────────────────────────────────────────────────

def _load_model() -> bool:
    global _pipeline, _label_encoder
    if ML_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
        _pipeline = joblib.load(ML_MODEL_PATH)
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.debug("ML model loaded from disk.")
        return True
    return False


def is_trained() -> bool:
    if _pipeline is not None:
        return True
    return _load_model()


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(sub_category: str) -> Optional[dict]:
    """
    Predict category for a sub_category string.
    Returns dict {category, confidence, source="ml"} or None if:
    - Model not trained yet
    - Confidence below ML_CONFIDENCE_THRESHOLD
    """
    global _pipeline, _label_encoder
    if _pipeline is None:
        if not _load_model():
            logger.debug("ML model not available.")
            return None

    proba = _pipeline.predict_proba([sub_category])[0]
    best_idx = int(np.argmax(proba))
    confidence = float(proba[best_idx])
    category = _label_encoder.inverse_transform([best_idx])[0]

    logger.debug(
        "ML prediction: '%s' → '%s' (confidence=%.3f, threshold=%.2f)",
        sub_category, category, confidence, ML_CONFIDENCE_THRESHOLD,
    )

    if confidence >= ML_CONFIDENCE_THRESHOLD:
        return {"category": category, "confidence": confidence, "source": "ml"}

    # Return low-confidence result as hint (pipeline decides whether to use it)
    return {
        "category": category,
        "confidence": confidence,
        "source": "ml",
        "low_confidence": True,
    }


def predict_batch(sub_categories: list[str]) -> list[Optional[dict]]:
    """Batch predict – returns list aligned with input."""
    return [predict(sc) for sc in sub_categories]
