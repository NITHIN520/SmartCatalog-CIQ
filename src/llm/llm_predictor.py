"""
LLM Predictor
-------------
Uses Groq (free) as the primary LLM.
Falls back to OpenAI if OPENAI_API_KEY is set and Groq fails.

Groq is free, fast, and supports Llama 3.3 70B which is very accurate
for product categorisation tasks.

Get a free Groq key at: https://console.groq.com
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from src.config import (
    GROQ_API_KEY, GROQ_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

_groq_client = None
_openai_client = None

# Rate-limit guard: Groq free tier allows ~30 req/min → 1 req every 2 seconds
_GROQ_MIN_INTERVAL: float = 2.0
_last_groq_call_time: float = 0.0


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY is not set in .env")
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


_SYSTEM_PROMPT = """You are a product categorization expert.
Given a product name, assign it to exactly ONE of the provided client categories.

Rules:
- Return ONLY valid JSON: {"category": "<chosen category>", "confidence": <0.0-1.0>}
- "category" must be one of the listed categories – do not invent new ones
- "confidence" is your certainty (0.0 = guessing, 1.0 = certain)
- If none fits at all, pick the closest one and set confidence below 0.5
"""


def _build_user_prompt(product_name: str, known_categories: list[str], ml_hint: Optional[str]) -> str:
    cat_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(known_categories))
    prompt = (
        f"Available categories:\n{cat_list}\n\n"
        f'Product name: "{product_name}"\n\n'
        "Return JSON only."
    )
    if ml_hint:
        prompt += f"\n(ML model hinted: '{ml_hint}', but with low confidence)"
    return prompt


def _parse_response(raw: str, known_categories: list[str]) -> Optional[dict]:
    """Parse LLM JSON response and validate the category."""
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in LLM response: {raw!r}")

    result = json.loads(json_match.group())
    category = result.get("category", "").strip()
    confidence = float(result.get("confidence", 0.5))

    # Validate against allowed categories (case-insensitive)
    lower_map = {c.lower(): c for c in known_categories}
    canonical = lower_map.get(category.lower())

    if canonical is None:
        # Fuzzy fallback within known categories
        from rapidfuzz import process, fuzz
        best = process.extractOne(category, known_categories, scorer=fuzz.token_sort_ratio)
        canonical = best[0] if best else known_categories[0]
        confidence = max(0.1, confidence - 0.2)
        logger.warning("LLM returned unknown category '%s' → mapped to '%s'", category, canonical)

    return {"category": canonical, "confidence": confidence, "source": "llm"}


def _call_groq(messages: list[dict]) -> str:
    global _last_groq_call_time
    elapsed = time.time() - _last_groq_call_time
    if elapsed < _GROQ_MIN_INTERVAL:
        wait = _GROQ_MIN_INTERVAL - elapsed
        logger.debug("Rate-limit guard: sleeping %.2fs before Groq call.", wait)
        time.sleep(wait)

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=100,
    )
    _last_groq_call_time = time.time()
    return response.choices[0].message.content.strip()


def _call_openai(messages: list[dict]) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def predict(
    product_name: str,
    known_categories: list[str],
    ml_hint: Optional[str] = None,
) -> Optional[dict]:
    """
    Predict the category for product_name using Groq (free) → OpenAI (fallback).

    Returns dict {category, confidence, source="llm"} or None on error.
    """
    if not known_categories:
        logger.error("No known categories provided to LLM predictor.")
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(product_name, known_categories, ml_hint)},
    ]

    # ── Try Groq first (free) ─────────────────────────────────────────────────
    if GROQ_API_KEY:
        try:
            raw = _call_groq(messages)
            logger.debug("Groq raw response: %s", raw)
            result = _parse_response(raw, known_categories)
            logger.info("Groq: '%s' → '%s' (confidence=%.2f)", product_name, result["category"], result["confidence"])
            return result
        except Exception as exc:
            logger.warning("Groq failed: %s — trying OpenAI fallback.", exc)

    # ── Fallback to OpenAI ────────────────────────────────────────────────────
    if OPENAI_API_KEY:
        try:
            raw = _call_openai(messages)
            logger.debug("OpenAI raw response: %s", raw)
            result = _parse_response(raw, known_categories)
            logger.info("OpenAI: '%s' → '%s' (confidence=%.2f)", product_name, result["category"], result["confidence"])
            return result
        except Exception as exc:
            logger.error("OpenAI also failed: %s", exc)

    logger.error("No LLM available. Set GROQ_API_KEY in .env — free at console.groq.com")
    return None


def get_known_categories_from_dataset() -> list[str]:
    """Pull unique categories from the stored dataset."""
    from src.data.dataset_manager import load
    df = load()
    if df.empty:
        return []
    return sorted(df["category"].unique().tolist())
