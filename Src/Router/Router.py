from __future__ import annotations

import asyncio
import json
import re
import traceback

from Src.Pathway_1.pathway1 import pathway_1_lookup as search_recipe
from Src.neo4j_client import Neo4jClient


COMPARE_PATTERNS = [
    r'\bvs\.?\b', r'\bversus\b', r'\bcompare\b', r'\bcomparison\b',
    r'\bbetter\s+than\b', r'\bhealthier\s+than\b',
    r'\bor\b(?=.*\b(?:which|healthier|better|calories)\b)',
]

MODIFY_KEYWORDS = [
    'less', 'low', 'reduce', 'without', 'no ', 'vegan', 'vegetarian',
    'healthy version', 'healthier', 'substitute', 'replace', 'gluten free',
    'sugar free', 'keto', 'diet', 'light version', 'make it',
    'calorie', 'protein rich', 'high protein', 'low fat', 'low carb',
    'oil free', 'dairy free',
]

COMPARE_REGEX = re.compile('|'.join(COMPARE_PATTERNS), re.IGNORECASE)


class NutriSenseRouter:
    def __init__(self, neo4j_client, llm_engine, image_model=None):
        self.neo4j_client = neo4j_client
        self.engine = llm_engine
        self.image_model = image_model

    def _rule_based_classify(self, query: str):
        q = query.lower().strip()

        if ' and ' in q:
            dishes = self._extract_compare_dishes(q)
            if dishes and len(dishes) == 2:
                if len(dishes[0]) > 2 and len(dishes[1]) > 2:
                    return {"pathway": "COMPARE", "dishes": dishes, "constraint": None}

        if COMPARE_REGEX.search(q):
            dishes = self._extract_compare_dishes(q)
            if dishes and len(dishes) >= 2:
                return {"pathway": "COMPARE", "dishes": dishes, "constraint": None}

        for kw in MODIFY_KEYWORDS:
            if kw in q:
                dish = self._extract_dish_for_modify(q, kw)
                constraint = self._extract_constraint(q)
                return {
                    "pathway": "MODIFY",
                    "dishes": [dish] if dish else [q],
                    "constraint": constraint,
                }

        dish = self._clean_dish_name(q)
        return {"pathway": "EXTRACT", "dishes": [dish if dish else q], "constraint": None}

    def _extract_compare_dishes(self, q: str):
        for splitter in [r'\s+vs\.?\s+', r'\s+versus\s+', r'\s+compare\s+',
                         r'\s+and\s+', r'\s+or\s+']:
            parts = re.split(splitter, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                a = self._clean_dish_name(parts[0])
                b = self._clean_dish_name(parts[1])
                if a and b:
                    return [a, b]
        return None

    @staticmethod
    def _extract_dish_for_modify(q: str, keyword: str):
        idx = q.find(keyword)
        before = q[:idx].strip() if idx > 0 else ""
        for filler in ['make', 'prepare', 'cook', 'give me', 'i want', 'can you', 'how to', 'create']:
            before = re.sub(rf'\b{filler}\b', '', before, flags=re.IGNORECASE).strip()
        return before if before else q

    @staticmethod
    def _extract_constraint(q: str):
        for kw in MODIFY_KEYWORDS:
            if kw in q:
                idx = q.find(kw)
                return q[idx:].strip()
        return q

    @staticmethod
    def _clean_dish_name(text: str):
        text = text.strip()
        for filler in ['what is in', 'what is', 'nutrition of', 'nutrients in',
                       'tell me about', 'details of', 'info about', 'about',
                       'how much', 'nutritional value of', 'give me',
                       'i want', 'show me', 'compare', 'which is']:
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE).strip()
        return text

    def classify_intent(self, query):
        if not query or query.strip() == "":
            return {"pathway": "EXTRACT", "dishes": [""], "constraint": None}

        result = self._rule_based_classify(query)
        if result:
            print(f"[Router] {result['pathway']} | Dishes: {result['dishes']}")
            return result

        return self._llm_classify(query)

    def _llm_classify(self, query):
        prompt = f"""Analyze this food-related user query and classify its intent.

Query: "{query}"

Categories:
- "EXTRACT": User is asking about ONE specific food/dish (this is the DEFAULT)
- "COMPARE": User EXPLICITLY asks to compare TWO or more foods (must use words like "vs", "compare", "versus", "or which is better")
- "MODIFY": User wants to change/modify a recipe (must use words like "less", "low", "without", "vegan", "healthier version")

CRITICAL RULES:
1. If the query mentions only ONE dish, ALWAYS return EXTRACT — never invent a second dish
2. COMPARE requires the user to EXPLICITLY name two dishes
3. When in doubt, choose EXTRACT
4. The "dishes" array must contain ONLY dishes that the user actually mentioned

Return ONLY valid JSON (no markdown, no explanation):
{{"pathway": "EXTRACT", "dishes": ["dish name"], "constraint": null}}"""

        try:
            response = self.engine.llm.generate(prompt)

            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                pathway = data.get('pathway', 'EXTRACT')
                dishes = data.get('dishes', [query])
                constraint = data.get('constraint')

                if pathway == "COMPARE" and len(dishes) < 2:
                    pathway = "EXTRACT"

                if pathway == "COMPARE" and len(dishes) >= 2:
                    q_lower = query.lower()
                    hallucinated = [d for d in dishes if d.lower() not in q_lower]
                    if hallucinated:
                        pathway = "EXTRACT"
                        dishes = [d for d in dishes if d.lower() in q_lower] or [query]

                result = {
                    "pathway": pathway,
                    "dishes": dishes if dishes else [query],
                    "constraint": constraint,
                }
                print(f"[Router] {result['pathway']} | Dishes: {result['dishes']}")
                return result
            else:
                raise ValueError("No valid JSON in LLM response")

        except Exception as e:
            print(f"[Router] Classification error: {e}, defaulting to EXTRACT")
            return {"pathway": "EXTRACT", "dishes": [query], "constraint": None}
        
    def execute(self, text_query=None, image_input=None):
        try:
            if image_input is not None:
                dish_name, img_conf = self.image_model.predict(image_input)
                return self.handle_extraction(dish_name, override_conf=img_conf)

            if not text_query or text_query.strip() == "":
                return {
                    'error': 'Please provide a valid query',
                    'llm_response': 'I need a question about food to help you!'
                }

            intent = self.classify_intent(text_query)
            dishes = intent.get('dishes', [])
            constraint = intent.get('constraint')
            pathway = intent.get('pathway', 'EXTRACT')

            if pathway == "COMPARE" and len(dishes) >= 2:
                return self.handle_comparison(dishes, constraint)
            elif pathway == "MODIFY":
                target_dish = dishes[0] if dishes else text_query
                return self.handle_modification(target_dish, constraint)
            else:
                target = dishes[0] if dishes else text_query
                return self.handle_extraction(target)

        except Exception as e:
            traceback.print_exc()
            return {
                'error': str(e),
                'llm_response': f'An error occurred: {str(e)}'
            }

    def handle_extraction(self, name, override_conf=None):
        try:
            if not name or name.strip() == "":
                return self.engine.estimate_nutrition("general healthy meal")

            res = search_recipe(name, self.neo4j_client)

            if res and res.get('status') == "FOUND" and res.get('results'):
                out = res['results'][0].copy()
                if override_conf is not None:
                    out['accuracy'] = float(override_conf * 100)
                elif 'confidence' in out:
                    out['accuracy'] = float(out['confidence'] * 100)
                else:
                    out['accuracy'] = 85.0
                return out

            return self.engine.estimate_nutrition(name)

        except Exception as e:
            return self.engine.estimate_nutrition(name)

    def handle_modification(self, name, constraint):
        try:
            if not name or name.strip() == "":
                return {
                    'error': 'No dish specified',
                    'llm_response': 'Please specify which dish you want to modify.'
                }

            res = search_recipe(name, self.neo4j_client)

            if res and res.get('status') == "FOUND" and res.get('results'):
                d = res['results'][0]
                out = self.engine.modify_recipe(
                    d.get('recipe_name', name),
                    d.get('nutrition', {}),
                    d.get('ingredients', 'Not available'),
                    d.get('instructions', 'Not available'),
                    constraint
                )
                out['accuracy'] = float(d.get('confidence', 0.85) * 100)
                return out

            constraint_text = f" with {constraint}" if constraint else ""
            return self.engine.estimate_nutrition(f"{name}{constraint_text}")

        except Exception as e:
            constraint_text = f" with {constraint}" if constraint else ""
            return self.engine.estimate_nutrition(f"{name}{constraint_text}")

    def handle_comparison(self, dishes, goal):
        try:
            if len(dishes) < 2:
                return {
                    'error': 'Need two dishes to compare',
                    'llm_response': 'Please specify two dishes to compare.'
                }

            res_a = search_recipe(dishes[0], self.neo4j_client)
            found_a = res_a and res_a.get('status') == "FOUND" and res_a.get('results')

            if found_a:
                dish_a_name = res_a['results'][0].get('recipe_name', dishes[0])
                nutrition_a = res_a['results'][0].get('nutrition', {})
                is_a_estimated = False
            else:
                dish_a_name = dishes[0]
                nutrition_a = self.engine.estimate_single_dish_nutrition(dishes[0])
                is_a_estimated = True

            res_b = search_recipe(dishes[1], self.neo4j_client)
            found_b = res_b and res_b.get('status') == "FOUND" and res_b.get('results')

            if found_b:
                dish_b_name = res_b['results'][0].get('recipe_name', dishes[1])
                nutrition_b = res_b['results'][0].get('nutrition', {})
                is_b_estimated = False
            else:
                dish_b_name = dishes[1]
                nutrition_b = self.engine.estimate_single_dish_nutrition(dishes[1])
                is_b_estimated = True

            out = self.engine.compare_dishes(
                dish_a_name, nutrition_a,
                dish_b_name, nutrition_b,
                goal,
                is_a_estimated=is_a_estimated,
                is_b_estimated=is_b_estimated
            )

            if not is_a_estimated and not is_b_estimated:
                conf_a = res_a['results'][0].get('confidence', 0.85)
                conf_b = res_b['results'][0].get('confidence', 0.85)
                out['accuracy'] = float((conf_a + conf_b) / 2 * 100)
            elif is_a_estimated and is_b_estimated:
                out['accuracy'] = 55.0
            else:
                out['accuracy'] = 70.0

            return out

        except Exception as e:
            traceback.print_exc()
            goal_text = f" for {goal}" if goal else ""
            return self.engine.estimate_nutrition(f"Compare {dishes[0]} and {dishes[1]}{goal_text}")

    # Async execute methods.
    # Neo4j queries and TF inference run in a thread pool via asyncio.to_thread.
    # LLM calls use generate_async (httpx) and do not block the event loop.

    async def execute_async(
        self, text_query: str = "", image_input: str | None = None
    ) -> dict:
        try:
            if image_input is not None:
                dish_name, img_conf = await asyncio.to_thread(
                    self.image_model.predict, image_input
                )
                return await self._async_handle_extraction(dish_name, override_conf=img_conf)

            if not text_query or text_query.strip() == "":
                return {
                    "error": "Please provide a valid query",
                    "llm_response": "I need a question about food to help you!",
                }

            # classify_intent may fall back to the sync LLM — run in a thread
            intent = await asyncio.to_thread(self.classify_intent, text_query)
            dishes = intent.get("dishes", [])
            constraint = intent.get("constraint")
            pathway = intent.get("pathway", "EXTRACT")

            if pathway == "COMPARE" and len(dishes) >= 2:
                return await self._async_handle_comparison(dishes, constraint)
            elif pathway == "MODIFY":
                target = dishes[0] if dishes else text_query
                return await self._async_handle_modification(target, constraint)
            else:
                target = dishes[0] if dishes else text_query
                return await self._async_handle_extraction(target)

        except Exception as exc:
            traceback.print_exc()
            return {"error": str(exc), "llm_response": f"An error occurred: {exc}"}

    async def _async_handle_extraction(
        self, name: str, override_conf: float | None = None
    ) -> dict:
        try:
            if not name or name.strip() == "":
                return await self.engine.estimate_nutrition_async("general healthy meal")

            res = await asyncio.to_thread(search_recipe, name, self.neo4j_client)

            if res and res.get("status") == "FOUND" and res.get("results"):
                out = res["results"][0].copy()
                if override_conf is not None:
                    out["accuracy"] = float(override_conf * 100)
                elif "confidence" in out:
                    out["accuracy"] = float(out["confidence"] * 100)
                else:
                    out["accuracy"] = 85.0
                return out

            return await self.engine.estimate_nutrition_async(name)

        except Exception:
            return await self.engine.estimate_nutrition_async(name)

    async def _async_handle_modification(self, name: str, constraint: str) -> dict:
        try:
            if not name or name.strip() == "":
                return {
                    "error": "No dish specified",
                    "llm_response": "Please specify which dish you want to modify.",
                }

            res = await asyncio.to_thread(search_recipe, name, self.neo4j_client)

            if res and res.get("status") == "FOUND" and res.get("results"):
                d = res["results"][0]
                out = await self.engine.modify_recipe_async(
                    d.get("recipe_name", name),
                    d.get("nutrition", {}),
                    d.get("ingredients", "Not available"),
                    d.get("instructions", "Not available"),
                    constraint,
                )
                out["accuracy"] = float(d.get("confidence", 0.85) * 100)
                return out

            constraint_text = f" with {constraint}" if constraint else ""
            return await self.engine.estimate_nutrition_async(f"{name}{constraint_text}")

        except Exception:
            constraint_text = f" with {constraint}" if constraint else ""
            return await self.engine.estimate_nutrition_async(f"{name}{constraint_text}")

    async def _async_handle_comparison(self, dishes: list, goal: str) -> dict:
        try:
            if len(dishes) < 2:
                return {
                    "error": "Need two dishes to compare",
                    "llm_response": "Please specify two dishes to compare.",
                }

            res_a, res_b = await asyncio.gather(
                asyncio.to_thread(search_recipe, dishes[0], self.neo4j_client),
                asyncio.to_thread(search_recipe, dishes[1], self.neo4j_client),
            )

            found_a = res_a and res_a.get("status") == "FOUND" and res_a.get("results")
            found_b = res_b and res_b.get("status") == "FOUND" and res_b.get("results")

            if found_a:
                dish_a_name = res_a["results"][0].get("recipe_name", dishes[0])
                nutrition_a = res_a["results"][0].get("nutrition", {})
                is_a_estimated = False
            else:
                dish_a_name = dishes[0]
                nutrition_a = await self.engine.estimate_single_dish_nutrition_async(dishes[0])
                is_a_estimated = True

            if found_b:
                dish_b_name = res_b["results"][0].get("recipe_name", dishes[1])
                nutrition_b = res_b["results"][0].get("nutrition", {})
                is_b_estimated = False
            else:
                dish_b_name = dishes[1]
                nutrition_b = await self.engine.estimate_single_dish_nutrition_async(dishes[1])
                is_b_estimated = True

            out = await self.engine.compare_dishes_async(
                dish_a_name, nutrition_a,
                dish_b_name, nutrition_b,
                goal,
                is_a_estimated=is_a_estimated,
                is_b_estimated=is_b_estimated,
            )

            if not is_a_estimated and not is_b_estimated:
                conf_a = res_a["results"][0].get("confidence", 0.85)
                conf_b = res_b["results"][0].get("confidence", 0.85)
                out["accuracy"] = float((conf_a + conf_b) / 2 * 100)
            elif is_a_estimated and is_b_estimated:
                out["accuracy"] = 55.0
            else:
                out["accuracy"] = 70.0

            return out

        except Exception as exc:
            traceback.print_exc()
            goal_text = f" for {goal}" if goal else ""
            return await self.engine.estimate_nutrition_async(
                f"Compare {dishes[0]} and {dishes[1]}{goal_text}"
            )