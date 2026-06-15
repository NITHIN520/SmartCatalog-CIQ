"""
Text Preprocessor
------------------
Extracts the most category-relevant keywords from a full product name
before passing it to the ML / vector / fuzzy layers.

Example:
  "Corona Extra Lager Can 12 x 330ml"
        → "Extra Lager Beer"          (matched keywords)
        → fed into ML model / vector store

This bridges the gap between short sub-category training data
and long real-world product name inputs.
"""
from __future__ import annotations

import re

# ── Keyword → Category hint map ───────────────────────────────────────────────
# Order matters: more specific patterns first

_KEYWORD_RULES: list[tuple[re.Pattern, str]] = [
    # Non-alcoholic beer
    (re.compile(r"\b(non.?alcohol|alcohol.?free|0\.0%|0%\s*abv|sunbrew)\b", re.I), "Non-alcoholic Beer"),

    # Hard beverage
    (re.compile(r"\b(hard\s+seltzer|hard\s+selt|hard\s+lem|hard\s+tea|hard\s+cider|hard\s+kombucha|hard\s+ginger|malt\s+bev|rtd|ready.to.drink|spiked)\b", re.I), "Hard Beverage"),

    # Beer keywords
    (re.compile(r"\b(lager|ale|ipa|stout|porter|pilsner|pilsener|bock|saison|cerveza|beer|weiss|weizen|hefe|cider(?!\s+vin))\b", re.I), "Beer"),

    # Wine
    (re.compile(r"\b(wine|chardonnay|merlot|cabernet|sauvignon|pinot|riesling|ros[eé]|prosecco|champagne|sparkling\s+wine|shiraz|malbec|zinfandel)\b", re.I), "Wine"),

    # Spirits
    (re.compile(r"\b(vodka|whisky|whiskey|bourbon|scotch|rum|tequila|gin|brandy|cognac|liqueur|mezcal|absinthe|schnapps|triple\s+sec|vermouth)\b", re.I), "Spirits"),

    # Non-alcoholic beverages
    (re.compile(r"\b(water|juice|soda|cola|coffee|tea(?!\s+hard)|energy\s+drink|sports\s+drink|lemonade|kombucha(?!\s+hard)|smoothie|shake|protein|vitamin)\b", re.I), "Non-alcoholic Beverage"),
]

# Packaging / size words to strip before feeding to ML
_NOISE_PATTERN = re.compile(
    r"\b(\d+\s*(x|pk|pack|ml|cl|l|oz|fl\.?\s*oz|can|cans|bottle|bottles|"
    r"longneck|tallboy|stubby|keg|case|litre|liter|pint|quart)|"
    r"premium|select|special|reserve|vintage|"
    r"limited|edition|fresh|natural|organic|"
    r"imported|import|domestic|draft|draught)\b",
    re.I,
)


def extract_category_hint(product_name: str) -> str | None:
    """
    Returns the best category hint from the product name, or None.
    Used to guide the vector / ML layer when the product name is long.
    """
    for pattern, category in _KEYWORD_RULES:
        if pattern.search(product_name):
            return category
    return None


def clean_product_name(product_name: str) -> str:
    """
    Strip packaging noise and keep only meaningful tokens.
    Result is shorter and closer to how sub-categories are named.

    Example:
      "Corona Extra Lager Can 12 x 330ml" → "Corona Lager"
    """
    cleaned = _NOISE_PATTERN.sub(" ", product_name)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else product_name
