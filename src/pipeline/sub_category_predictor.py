"""
Sub-Category Predictor
-----------------------
Given a product name AND an already-predicted category, find the best
matching sub_category from the known list for that category.

Layers (in order):
  1. Exact match against known sub_categories in the category
  2. Fuzzy string match (RapidFuzz)
  3. TF-IDF cosine similarity (vector store)
  4. LLM – generates a sub_category from the known list
  5. Fallback – return the most common sub_category in that category
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SUB_CONF_THRESHOLD = 0.55   # minimum similarity to trust vector/fuzzy result


def _get_sub_categories_for(category: str) -> list[str]:
    """
    Return canonical sub_categories for a given category.
    Only uses seed/human rows so that stored product names don't pollute the list.
    For seed rows, predicted_sub_category equals the product_name (which IS the sub_category).
    """
    from src.data.dataset_manager import load
    df = load()
    if df.empty:
        return []
    authoritative_sources = {"seed", "human"}
    mask = (
        (df["category"].str.lower() == category.lower()) &
        (df["source"].isin(authoritative_sources))
    )
    return df.loc[mask, "predicted_sub_category"].drop_duplicates().tolist()


def _keyword_best(product_name: str, candidates: list[str]) -> Optional[tuple[str, float]]:
    """
    Check if any sub-category name (or its key words) appear directly in the
    product name. Prefers longer/more-specific matches.

    Examples:
      "Angry Orchard Hard Cider Beer"   → "Cider"         (word "cider" found)
      "White Claw Hard Seltzer 6pk"     → "Hard Seltzers" (stem "seltz" found)
      "Truly Hard Seltzer Variety Pack" → "Hard Seltzers"
    """
    import re

    name_lower = product_name.lower()
    name_words = set(re.findall(r"[a-z]+", name_lower))

    # Words too generic to discriminate between candidates
    # "beer" and "ale" appear in almost every product name so they must not
    # be used to match "Craft Beer" against unrelated products like Beck's.
    _SKIP = {"beer", "ale", "pack", "packs", "cans", "bottles", "that", "from", "with", "fl", "oz"}

    # Beer-style aliases: product name terms that unambiguously imply a sub-category.
    # These bridge the gap when the candidate name itself ("Craft Beer") doesn't
    # contain the style word ("IPA", "Stout") that appears in the product name.
    _STYLE_ALIASES: dict[str, list[str]] = {
        "Craft Beer": [
            "ipa", "stout", "porter", "hazy", "saison", "sour", "pale",
            "hefeweizen", "weiss", "weizen", "amber", "bock", "barleywine",
            "session", "wheat", "kolsch", "tripel", "dubbel", "pilsner", "pilsener",
        ],
        "Light Lager": ["light", "lite"],
        "Classic Lager": ["lager", "cerveza"],
        "Hard Seltzers": ["seltzer", "seltzers"],
        "Cider": ["cider"],
        "Ready to Drink": ["rtd"],
        "Flavored Hard Beverages": ["cocktail", "punch", "lemonade"],
    }

    best_match: Optional[str] = None
    best_score: int = 0  # higher = more confident

    for candidate in candidates:
        cand_lower = candidate.lower()

        # Layer A: full phrase substring match (highest priority)
        if cand_lower in name_lower:
            score = 100 + len(cand_lower)   # longer phrase = more specific
            if score > best_score:
                best_match, best_score = candidate, score
            continue

        # Layer A2: style-alias match — e.g. "ipa" in product → "Craft Beer"
        aliases = _STYLE_ALIASES.get(candidate, [])
        alias_matches = [a for a in aliases if re.search(r"\b" + re.escape(a) + r"\b", name_lower)]
        if alias_matches:
            # score = 90 + length of longest alias matched (longer = more specific)
            score = 90 + max(len(a) for a in alias_matches)
            if score > best_score:
                best_match, best_score = candidate, score
            continue

        # Layer B: word-level match with prefix tolerance (handles "seltzer" ↔ "seltzers")
        cand_words = [
            w for w in re.findall(r"[a-z]+", cand_lower)
            if len(w) >= 4 and w not in _SKIP
        ]
        matched_lengths: list[int] = []
        for cw in cand_words:
            prefix = cw[:5]
            for nw in name_words:
                if len(nw) >= 4 and (nw.startswith(prefix) or cw.startswith(nw[:5])):
                    matched_lengths.append(len(cw))  # track length of matched word
                    break

        if matched_lengths:
            # Score = number of matched words * 10 + total chars matched
            # Longer/more-specific keywords win ties
            score = len(matched_lengths) * 10 + sum(matched_lengths)
            if score > best_score:
                best_match, best_score = candidate, score

    return (best_match, 0.92) if best_match else None


def _fuzzy_best(product_name: str, candidates: list[str]) -> Optional[tuple[str, float]]:
    import re
    from rapidfuzz import process, fuzz

    # Strip tokens that appear in almost every product name AND in candidate
    # names (e.g. "beer"), which inflate fuzzy scores and cause false positives
    # like "Beck's Beer 4 Pack" → "Craft Beer" just because of shared "beer".
    _GENERIC = re.compile(r"\b(beer|ale|brew|brewing|brewed)\b", re.I)

    name_stripped = _GENERIC.sub("", product_name).strip()
    if not name_stripped:
        name_stripped = product_name

    candidates_stripped = [_GENERIC.sub("", c).strip() or c for c in candidates]

    result = process.extractOne(
        name_stripped.lower(),
        [c.lower() for c in candidates_stripped],
        scorer=fuzz.token_set_ratio,
    )
    if result is None:
        return None
    _, score, idx = result
    return candidates[idx], score / 100.0


def _dataset_similar_best(product_name: str, category: str) -> Optional[tuple[str, float]]:
    """
    Find the most similar product already in the dataset (same category)
    and return its sub-category. This catches cases like "Angry Orchard
    Crisp Light" where the word "cider" is absent but similar products
    (e.g. "Angry Orchard Crisp Apple Hard Cider") are already tagged.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from src.data.dataset_manager import load

    df = load()
    if df.empty:
        return None

    # Only use rows from reliable sources to avoid learning from past wrong predictions
    _RELIABLE = {"seed", "human", "keyword"}
    mask = (
        (df["category"].str.lower() == category.lower())
        & df["predicted_sub_category"].notna()
        & (df["predicted_sub_category"] != "")
        & df["source"].isin(_RELIABLE)
    )

    df_cat = df[mask].drop_duplicates(subset="product_name")
    if len(df_cat) < 2:
        return None

    corpus = df_cat["product_name"].tolist() + [product_name]
    try:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        matrix = vect.fit_transform(corpus)
        query_vec = matrix[-1]
        candidate_matrix = matrix[:-1]
        scores = cosine_similarity(query_vec, candidate_matrix).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_sub = df_cat.iloc[best_idx]["predicted_sub_category"]
        logger.debug(
            "Dataset-sim sub-cat: '%s' → '%s' via '%s' (score=%.3f)",
            product_name, best_sub, df_cat.iloc[best_idx]["product_name"], best_score,
        )
        return best_sub, best_score
    except Exception as exc:
        logger.debug("Dataset-sim sub_category search failed: %s", exc)
        return None


def _vector_best(product_name: str, candidates: list[str]) -> Optional[tuple[str, float]]:
    """TF-IDF cosine similarity between product_name and each candidate."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not candidates:
        return None

    corpus = candidates + [product_name]
    try:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        matrix = vect.fit_transform(corpus)
        query_vec = matrix[-1]          # last row = product_name
        candidate_matrix = matrix[:-1]
        scores = cosine_similarity(query_vec, candidate_matrix).flatten()
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], float(scores[best_idx])
    except Exception as exc:
        logger.debug("Vector sub_category search failed: %s", exc)
        return None


def _llm_predict(product_name: str, category: str, candidates: list[str]) -> Optional[str]:
    """Ask Groq (or OpenAI fallback) to pick the best sub_category."""
    import json, re as _re
    from src.config import GROQ_API_KEY, GROQ_MODEL, OPENAI_API_KEY, OPENAI_MODEL

    if not GROQ_API_KEY and not OPENAI_API_KEY:
        return None

    cat_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        f"Product: \"{product_name}\"\n"
        f"Category: {category}\n\n"
        f"Known sub-categories:\n{cat_list}\n\n"
        "Pick the BEST matching sub-category from the list above.\n"
        'Return ONLY JSON: {"sub_category": "<exact name from list>"}'
    )
    messages = [
        {"role": "system", "content": "You are a product categorisation expert. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]

    def _parse(raw: str) -> Optional[str]:
        m = _re.search(r"\{.*?\}", raw, _re.DOTALL)
        if m:
            data = json.loads(m.group())
            sc = data.get("sub_category", "").strip()
            # Validate against known candidates (case-insensitive)
            lower_map = {c.lower(): c for c in candidates}
            return lower_map.get(sc.lower(), sc if sc else None)
        return None

    # Try Groq first
    if GROQ_API_KEY:
        try:
            from src.llm.llm_predictor import _call_groq
            raw = _call_groq(messages)
            sc = _parse(raw)
            if sc:
                logger.debug("Groq sub-cat: '%s' → '%s'", product_name, sc)
                return sc
        except Exception as exc:
            logger.debug("Groq sub-cat failed: %s", exc)

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        try:
            from src.llm.llm_predictor import _call_openai
            raw = _call_openai(messages)
            sc = _parse(raw)
            if sc:
                logger.debug("OpenAI sub-cat: '%s' → '%s'", product_name, sc)
                return sc
        except Exception as exc:
            logger.debug("OpenAI sub-cat failed: %s", exc)

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def predict_sub_category(
    product_name: str,
    category: str,
    cleaned_name: Optional[str] = None,
) -> dict:
    """
    Predict the sub_category for product_name given its category.

    Returns
    -------
    dict with keys:
      sub_category   : str
      sub_confidence : float (0–1)
      sub_source     : str  (exact | fuzzy | vector | llm | fallback)
    """
    # Use original product_name for keyword/exact matching (preserves meaningful
    # words like "light", "classic" that the noise cleaner may strip).
    # Use cleaned_name for fuzzy/vector layers where shorter text works better.
    keyword_name = product_name
    lookup_name = cleaned_name or product_name
    candidates = _get_sub_categories_for(category)

    if not candidates:
        return {"sub_category": "Unknown", "sub_confidence": 0.0, "sub_source": "fallback"}

    # ── 1. Exact match ────────────────────────────────────────────────────────
    lower_map = {c.lower(): c for c in candidates}
    if keyword_name.lower() in lower_map or lookup_name.lower() in lower_map:
        sc = lower_map.get(keyword_name.lower()) or lower_map.get(lookup_name.lower())
        return {"sub_category": sc, "sub_confidence": 1.0, "sub_source": "exact"}

    # ── 1b. Keyword / token match (uses ORIGINAL name to preserve "light" etc.) ─
    kw_result = _keyword_best(keyword_name, candidates)
    if kw_result:
        sc, score = kw_result
        logger.debug("Sub-category keyword hit: '%s' → '%s'", product_name, sc)
        return {"sub_category": sc, "sub_confidence": score, "sub_source": "keyword"}

    # ── 2. Fuzzy match ────────────────────────────────────────────────────────
    fuzzy_result = _fuzzy_best(lookup_name, candidates)
    if fuzzy_result and fuzzy_result[1] >= _SUB_CONF_THRESHOLD:
        sc, score = fuzzy_result
        logger.debug("Sub-category fuzzy hit: '%s' → '%s' (%.2f)", product_name, sc, score)
        return {"sub_category": sc, "sub_confidence": score, "sub_source": "fuzzy"}

    # ── 3. Vector / TF-IDF match ──────────────────────────────────────────────
    vec_result = _vector_best(lookup_name, candidates)
    if vec_result and vec_result[1] >= _SUB_CONF_THRESHOLD:
        sc, score = vec_result
        logger.debug("Sub-category vector hit: '%s' → '%s' (%.2f)", product_name, sc, score)
        return {"sub_category": sc, "sub_confidence": score, "sub_source": "vector"}

    # ── 4. Dataset similarity search ──────────────────────────────────────────
    # Finds similar products already in the dataset for the same category.
    # Catches e.g. "Angry Orchard Crisp Light" → "Cider" because similar
    # "Angry Orchard Crisp Apple Hard Cider" products are already tagged.
    ds_result = _dataset_similar_best(keyword_name, category)
    if ds_result and ds_result[1] >= _SUB_CONF_THRESHOLD:
        sc, score = ds_result
        if sc in candidates:   # only accept if it's a known sub-category
            return {"sub_category": sc, "sub_confidence": score, "sub_source": "dataset_sim"}

    # ── 5. LLM ────────────────────────────────────────────────────────────────
    llm_sc = _llm_predict(product_name, category, candidates)
    if llm_sc:
        logger.debug("Sub-category LLM: '%s' → '%s'", product_name, llm_sc)
        return {"sub_category": llm_sc, "sub_confidence": 0.80, "sub_source": "llm"}

    # ── 6. Best available fallback ────────────────────────────────────────────
    # Use dataset similarity even below threshold rather than a random guess
    if ds_result and ds_result[0] in candidates:
        sc, score = ds_result
        return {"sub_category": sc, "sub_confidence": score, "sub_source": "fallback"}

    if fuzzy_result:
        sc, score = fuzzy_result
        return {"sub_category": sc, "sub_confidence": score, "sub_source": "fallback"}

    return {"sub_category": candidates[0], "sub_confidence": 0.0, "sub_source": "fallback"}
