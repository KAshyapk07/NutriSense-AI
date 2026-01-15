class LLMEngine:
    """
    LLM reasoning engine for NutriSense AI.
    Now returns structured data that the frontend can render beautifully.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def estimate_single_dish_nutrition(self, dish_name: str) -> dict:
        """
        Estimate nutrition for a single dish when not in database.
        Returns nutrition dict compatible with comparison.
        """
        
        prompt = f"""You are a nutrition expert. Provide estimated nutrition for this Indian dish.

DISH: {dish_name}

TASK:
Provide realistic estimated nutrition values based on typical recipes for ONE SERVING.

IMPORTANT: Return ONLY the nutrition values in this EXACT format (no extra text):

Calories: [number] kcal
Protein: [number]g
Carbohydrates: [number]g
Fats: [number]g
Fiber: [number]g

Base your estimates on:
- Typical serving size (1 plate/serving)
- Common ingredients and preparation methods
- Standard Indian cuisine recipes

Example format:
Calories: 250 kcal
Protein: 8g
Carbohydrates: 35g
Fats: 10g
Fiber: 4g

Respond ONLY with the nutrition values in the exact format shown above. No additional text.
"""

        response = self.llm.generate(prompt).strip()
        
        # Parse the LLM response into a nutrition dict
        nutrition = {}
        for line in response.split('\n'):
            line = line.strip()
            if ':' in line and line:
                try:
                    key, value = line.split(':', 1)
                    nutrition[key.strip()] = value.strip()
                except:
                    continue
        
        # Fallback if parsing fails
        if not nutrition:
            nutrition = {
                "Calories": "~250 kcal",
                "Protein": "~8g",
                "Carbohydrates": "~35g",
                "Fats": "~10g",
                "Fiber": "~3g"
            }
        
        return nutrition

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
        user_goal: str | None = None,
        is_a_estimated: bool = False,
        is_b_estimated: bool = False
    ) -> dict:
        """
        Compare two dishes using dataset-backed nutrition OR estimates.
        Returns structured comparison for frontend.
        """
        
        # Add estimation disclaimer if needed
        estimation_note = ""
        if is_a_estimated and is_b_estimated:
            estimation_note = "\n NOTE: Both dishes use estimated nutrition values based on typical recipes."
        elif is_a_estimated:
            estimation_note = f"\n NOTE: {dish_a} uses estimated nutrition values."
        elif is_b_estimated:
            estimation_note = f"\n NOTE: {dish_b} uses estimated nutrition values."

        prompt = f"""You are a nutrition expert comparing two dishes. Use the provided data.

DISH A: {dish_a}
Nutrition (per serving):
{self._format_nutrition(nutrition_a)}

DISH B: {dish_b}
Nutrition (per serving):
{self._format_nutrition(nutrition_b)}

USER GOAL: {user_goal if user_goal else "General healthy eating"}
{estimation_note}

IMPORTANT RULES:
- Use ONLY the provided nutrition values
- Do NOT invent numbers
- Compare based on the user's goal
- Mention trade-offs
{f"- Note that some values are estimates and may vary by recipe" if (is_a_estimated or is_b_estimated) else ""}

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
            "estimated": is_a_estimated or is_b_estimated,
            "accuracy": 85.0 if not (is_a_estimated or is_b_estimated) else 60.0,
            "source": "dataset + llm_comparison" if not (is_a_estimated or is_b_estimated) else "mixed/estimated + llm_comparison"
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
 These are approximate values based on typical recipes
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
                " Estimated Values": "See below"
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