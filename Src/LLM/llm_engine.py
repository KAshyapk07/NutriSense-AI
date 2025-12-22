class LLMEngine:
    """
    LLM reasoning engine for NutriSense AI.

    Responsibilities:
    - Modify recipes based on health constraints
    - Compare nutrition between two dishes
    - Estimate nutrition when dataset lookup fails

    IMPORTANT:
    - Dataset values are ground truth
    - LLM must not hallucinate exact nutrition values
    """

    def __init__(self, llm_client):
        self.llm = llm_client

# MODIFICATION PATHWAY

    def modify_recipe(
        self,
        dish_name: str,
        nutrition: dict,
        ingredients: str,
        method: str,
        user_constraint: str
    ) -> dict:
        """
        Modify an existing recipe while preserving dish identity.
        """

        prompt = f"""
Dish: {dish_name}

Original Nutrition (ground truth):
{nutrition}

Original Ingredients:
{ingredients}

Original Cooking Method:
{method}

User Constraint:
{user_constraint}

TASK:
- Modify the recipe to satisfy the constraint
- Keep dish identity intact
- Do NOT invent exact nutrition numbers
- Explain changes clearly

FORMAT YOUR RESPONSE AS:
1. Modified Ingredients
2. Modified Cooking Method
3. Changes Summary (bullet points)
4. Qualitative Nutrition Impact
"""

        response = self.llm.generate(prompt)

        return {
            "pathway": "modification",
            "dish": dish_name,
            "constraint": user_constraint,
            "llm_response": response,
            "estimated": False,
            "source": "dataset + llm"
        }

    #  COMPARISON PATHWAY

    def compare_dishes(
        self,
        dish_a: str,
        nutrition_a: dict,
        dish_b: str,
        nutrition_b: dict,
        user_goal: str | None = None
    ) -> dict:
        """
        Compare two dishes using dataset-backed nutrition.
        """

        prompt = f"""
Dish A: {dish_a}
Nutrition A:
{nutrition_a}

Dish B: {dish_b}
Nutrition B:
{nutrition_b}

User Goal:
{user_goal if user_goal else "General health"}

RULES:
- Use provided nutrition values as ground truth
- Do NOT fabricate numbers
- Compare calories, macros, and key micronutrients
- Decide which dish is better for the given goal
- Mention trade-offs

FORMAT YOUR RESPONSE AS:
1. Summary Comparison
2. Better Choice (for the goal)
3. Trade-offs
"""

        response = self.llm.generate(prompt)

        return {
            "pathway": "comparison",
            "dish_a": dish_a,
            "dish_b": dish_b,
            "goal": user_goal,
            "llm_response": response,
            "estimated": False,
            "source": "dataset + llm"
        }

    # ESTIMATOR / FALLBACK PATHWAY

    def estimate_nutrition(
        self,
        user_query: str,
        closest_dish: str | None = None,
        reference_nutrition: dict | None = None
    ) -> dict:
        """
        Estimate nutrition when dataset lookup fails.
        """

        prompt = f"""
User Query:
{user_query}

Closest Known Dish:
{closest_dish if closest_dish else "None"}

Reference Nutrition (if available):
{reference_nutrition if reference_nutrition else "None"}

TASK:
- Clearly state that values are ESTIMATES
- Provide nutrition as approximate ranges
- Assign a confidence level: High / Medium / Low
- Do NOT claim medical accuracy

FORMAT YOUR RESPONSE AS:
1. Estimated Nutrition (ranges)
2. Reference Dish Used
3. Confidence Level
4. Disclaimer
"""

        response = self.llm.generate(prompt)

        return {
            "pathway": "estimation",
            "query": user_query,
            "reference_dish": closest_dish,
            "llm_response": response,
            "estimated": True,
            "source": "llm_estimate"
        }
