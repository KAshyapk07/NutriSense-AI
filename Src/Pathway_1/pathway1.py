import re
import pandas as pd
from rapidfuzz import fuzz, process

MIN_COMPOSITE_SCORE = 75
TOP_K_CANDIDATES = 5

WEIGHT_TOKENSET = 0.6
WEIGHT_WRATIO = 0.4

VARIANT_SCORE_DELTA = 10  
MAX_VARIANTS = 4         

def clean_minimal(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

NEGATIVE_TOKEN_PAIRS = {
    ("idli", "dosa"),
    ("roti", "naan"),
    ("pulao", "khichdi"),
    ("kheer", "payasam")
}

def negative_pair_penalty(q_tokens, c_tokens):
    penalty = 0
    for a, b in NEGATIVE_TOKEN_PAIRS:
        if a in q_tokens and b in c_tokens:
            penalty += 20
        if b in q_tokens and a in c_tokens:
            penalty += 20
    return penalty

def composite_score(q_clean, c_clean):
    ts = fuzz.token_set_ratio(q_clean, c_clean)
    wr = fuzz.WRatio(q_clean, c_clean)

    q_tokens = set(q_clean.split())
    c_tokens = set(c_clean.split())

    penalty = negative_pair_penalty(q_tokens, c_tokens)

    score = (
        WEIGHT_TOKENSET * ts +
        WEIGHT_WRATIO * wr
    ) - penalty

    return score

def to_python_type(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def pathway_1_lookup(recipe_name: str, df: pd.DataFrame):
    """
    Pathway 1 (Multi-output):
    - Input: extracted recipe name
    - Output: list of full dataset-backed results
    """

    query = recipe_name.lower().strip()

    NAME_COL = "best_match_clean"
    DISPLAY_COL = "final_food_name"

    choices = df[NAME_COL].astype(str).tolist()

    candidates = process.extract(
        query,
        choices,
        scorer=fuzz.token_set_ratio,
        limit=TOP_K_CANDIDATES
    )

    scored = []

    for cand, _, idx in candidates:
        score = composite_score(query, cand)
        if score >= MIN_COMPOSITE_SCORE:
            scored.append((idx, score))

    if not scored:
        return {
            "status": "NOT_FOUND",
            "results": []
        }

    # sort by score (desc)
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    seen_names = set()

    nutrition_cols = [
        "Calories (kcal)", "Carbohydrates (g)", "Protein (g)",
        "Fats (g)", "Free Sugar (g)", "Fibre (g)",
        "Sodium (mg)", "Calcium (mg)", "Iron (mg)",
        "Vitamin C (mg)", "Folate (µg)"
    ]

    for idx, score in scored:
        row = df.iloc[idx]
        name = str(row[DISPLAY_COL]).strip()

        # deduplicate by display name
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        nutrition = {
            col: to_python_type(row[col])
            for col in nutrition_cols
            if col in df.columns
        }

        results.append({
            "recipe_name": name,
            "confidence": round(min(score / 100, 0.95), 2),
            "nutrition": nutrition,
            "ingredients": row.get("TranslatedIngredients"),
            "instructions": row.get("TranslatedInstructions"),
            "meta": {
                "cuisine": row.get("Cuisine"),
                "total_time": to_python_type(row.get("TotalTimeInMins"))
            }
        })

        if len(results) >= MAX_VARIANTS + 1:  # primary + variants
            break

    return {
        "status": "FOUND",
        "results": results
    }