class LLMEngine:
    """
    LLM reasoning engine for NutriSense AI.
    Now returns structured data that the frontend can render beautifully.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

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
        Returns structured data for frontend.
        """

        prompt = f"""You are a nutrition expert. Modify this recipe based on the user's request.

ORIGINAL RECIPE:
Dish: {dish_name}

Nutrition (per serving):
{self._format_nutrition(nutrition)}

Ingredients:
{ingredients}

Cooking Method:
{method}

USER REQUEST:
{user_constraint}

TASK:
1. Suggest ingredient substitutions to meet the constraint
2. Modify the cooking method if needed
3. Explain how this affects nutrition (qualitatively - do NOT invent exact numbers)
4. Keep the dish recognizable

FORMAT YOUR RESPONSE CLEARLY WITH SECTIONS:
**Modified Ingredients:**
[List changes here]

**Modified Cooking Method:**
[Describe changes here]

**Nutritional Impact:**
[Explain how nutrition changes - use terms like "reduced", "lower", "higher" - NO exact numbers]

**Tips:**
[Any additional advice]
"""

        response = self.llm.generate(prompt)

        return {
            "recipe_name": f"{dish_name} ({user_constraint})",
            "nutrition": nutrition,  # Keep original for reference
            "ingredients": "See modified version below",
            "instructions": "See modified version below",
            "llm_response": response,
            "pathway": "modification",
            "constraint": user_constraint,
            "estimated": False,
            "source": "dataset + llm_modification"
        }

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
        Returns structured comparison for frontend.
        """

        prompt = f"""You are a nutrition expert comparing two dishes. Use the provided data.

DISH A: {dish_a}
Nutrition (per serving):
{self._format_nutrition(nutrition_a)}

DISH B: {dish_b}
Nutrition (per serving):
{self._format_nutrition(nutrition_b)}

USER GOAL: {user_goal if user_goal else "General healthy eating"}

IMPORTANT RULES:
- Use ONLY the provided nutrition values
- Do NOT invent numbers
- Compare based on the user's goal
- Mention trade-offs

FORMAT YOUR RESPONSE WITH CLEAR SECTIONS:

**Summary:**
[Brief comparison of both dishes]

**For {user_goal if user_goal else "general health"}:**
[Which dish is better and why]

**Key Differences:**
- Calories: [comparison]
- Protein: [comparison]  
- Carbs: [comparison]
- Fats: [comparison]

**Trade-offs:**
[What you gain/lose with each choice]

**Recommendation:**
[Final verdict]
"""

        response = self.llm.generate(prompt)

        return {
            "dish_a": dish_a,
            "nutrition_a": nutrition_a,
            "dish_b": dish_b,
            "nutrition_b": nutrition_b,
            "llm_response": response,
            "pathway": "comparison",
            "goal": user_goal,
            "estimated": False,
            "source": "dataset + llm_comparison"
        }

    def estimate_nutrition(self, user_query: str) -> dict:
        """
        Estimate nutrition when dataset lookup fails.
        Returns structured estimates with clear disclaimers.
        """

        prompt = f"""You are a nutrition expert. The user asked about a dish not in our database.

USER QUERY: "{user_query}"

TASK:
Provide helpful nutritional information, but you MUST:
1. State clearly this is an ESTIMATE
2. Give approximate ranges (not exact numbers)
3. Explain what the dish typically contains
4. Provide general nutritional context
5. Suggest similar dishes in Indian cuisine

FORMAT YOUR RESPONSE:

**About {user_query}:**
[Brief description of the dish]

**Estimated Nutrition (per serving):**
⚠️ These are approximate values based on typical recipes
- Calories: ~[range] kcal
- Protein: ~[range]g
- Carbohydrates: ~[range]g  
- Fats: ~[range]g
- Fiber: ~[range]g

**Key Ingredients:**
[List main ingredients]

**Nutritional Highlights:**
[What's good/noteworthy about this dish]

**Similar Dishes in Database:**
[Suggest 2-3 similar dishes we DO have data for]

**Disclaimer:**
These values are estimates. For precise nutrition data, consult a nutritionist or use dishes from our verified database.
"""

        response = self.llm.generate(prompt)

        return {
            "recipe_name": f"{user_query} (Estimated)",
            "nutrition": {
                "⚠️ Estimated Values": "See below"
            },
            "ingredients": "See estimated details below",
            "instructions": "Not available - estimated dish",
            "llm_response": response,
            "pathway": "estimation",
            "estimated": True,
            "accuracy": 50.0,  # Lower confidence for estimates
            "source": "llm_estimation"
        }

    def _format_nutrition(self, nutrition: dict) -> str:
        """Helper to format nutrition dict for LLM prompts"""
        lines = []
        for key, value in nutrition.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)