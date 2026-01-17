from Src.Pathway_1.pathway1 import pathway_1_lookup as search_recipe
import json


class NutriSenseRouter:
    def __init__(self, df, llm_engine, image_model=None):
        self.df = df
        self.engine = llm_engine
        self.image_model = image_model

    def classify_intent(self, query):
        """Uses Llama to decide the pathway - simplified prompt that works."""
        if not query or query.strip() == "":
            return {"pathway": "EXTRACT", "dishes": [""], "constraint": None}
            
        prompt = f"""Analyze this user query: "{query}"

Categorize into:
- "COMPARE": User mentions TWO or MORE foods (e.g., "dosa vs idli", "compare chicken and paneer")
- "MODIFY": User wants a healthier version/modification (e.g., "less oil", "vegan", "low calorie")
- "EXTRACT": User asks about ONE food only (e.g., "what is in dosa", "nutrition of idli")

Rules:
- COMPARE only if TWO+ dishes are explicitly mentioned
- MODIFY if words like "less", "low", "without", "vegan", "healthy version"
- EXTRACT for everything else (default)

Return ONLY a JSON object:
{{"pathway": "COMPARE"|"MODIFY"|"EXTRACT", "dishes": ["dish1", "dish2"], "constraint": "modifier text or null"}}"""

        try:
            response = self.engine.llm.generate(prompt)
            
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                
                # Validation: Enforce rules
                pathway = data.get('pathway', 'EXTRACT')
                dishes = data.get('dishes', [query])
                constraint = data.get('constraint')
                
                # Force correction if LLM makes mistakes
                if pathway == "COMPARE" and len(dishes) < 2:
                    print(f" LLM said COMPARE but only found {len(dishes)} dishes, changing to EXTRACT")
                    pathway = "EXTRACT"
                
                result = {
                    "pathway": pathway,
                    "dishes": dishes if dishes else [query],
                    "constraint": constraint
                }
                
                print(f" Intent: {result['pathway']} | Dishes: {result['dishes']} | Constraint: {result['constraint']}")
                return result
            else:
                raise ValueError("No valid JSON in response")
                
        except Exception as e:
            print(f" Classification error: {e}, defaulting to EXTRACT")
            return {"pathway": "EXTRACT", "dishes": [query], "constraint": None}
        
    def execute(self, text_query=None, image_input=None):
        """Main execution router for all pathways"""
        try:
            # PATHWAY 4: IMAGE
            if image_input is not None:
                print(f" Processing image: {image_input}")
                dish_name, img_conf = self.image_model.predict(image_input)
                print(f" Image detected: {dish_name} (confidence: {img_conf:.2%})")
                return self.handle_extraction(dish_name, override_conf=img_conf)

            # Validate text query
            if not text_query or text_query.strip() == "":
                return {
                    'error': 'Please provide a valid query',
                    'llm_response': 'I need a question about food to help you!'
                }

            # TEXT PROCESSING
            intent = self.classify_intent(text_query)
            
            dishes = intent.get('dishes', [])
            constraint = intent.get('constraint')
            pathway = intent.get('pathway', 'EXTRACT')

            print(f" Routing to: {pathway}")

            # Route to appropriate handler
            if pathway == "COMPARE" and len(dishes) >= 2:
                return self.handle_comparison(dishes, constraint)
            
            elif pathway == "MODIFY":
                target_dish = dishes[0] if dishes else text_query
                return self.handle_modification(target_dish, constraint)
            
            else:  # EXTRACT
                target = dishes[0] if dishes else text_query
                return self.handle_extraction(target)
                
        except Exception as e:
            print(f" Router execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'llm_response': f'An error occurred: {str(e)}'
            }

    def handle_extraction(self, name, override_conf=None):
        """PATHWAY 1: Extract nutrition info for a single dish"""
        try:
            if not name or name.strip() == "":
                return self.engine.estimate_nutrition("general healthy meal")
            
            res = search_recipe(name, self.df)
            
            if res and res.get('status') == "FOUND" and res.get('results'):
                out = res['results'][0].copy()
                
                # Handle accuracy
                if override_conf is not None:
                    out['accuracy'] = float(override_conf * 100)
                elif 'confidence' in out:
                    out['accuracy'] = float(out['confidence'] * 100)
                else:
                    out['accuracy'] = 85.0
                
                print(f" Found in database: {out.get('recipe_name')}")
                return out
            
            # Fallback to LLM
            print(f" Not in database: {name}, using LLM")
            return self.engine.estimate_nutrition(name)
            
        except Exception as e:
            print(f" Extraction error: {e}")
            return self.engine.estimate_nutrition(name)

    def handle_modification(self, name, constraint):
        """PATHWAY 2: Modify a recipe with constraints"""
        try:
            if not name or name.strip() == "":
                return {
                    'error': 'No dish specified',
                    'llm_response': 'Please specify which dish you want to modify.'
                }
            
            res = search_recipe(name, self.df)
            
            if res and res.get('status') == "FOUND" and res.get('results'):
                d = res['results'][0]
                
                recipe_name = d.get('recipe_name', name)
                nutrition = d.get('nutrition', {})
                ingredients = d.get('ingredients', 'Not available')
                instructions = d.get('instructions', 'Not available')
                
                print(f" Modifying: {recipe_name} with constraint: {constraint}")
                
                out = self.engine.modify_recipe(
                    recipe_name, 
                    nutrition, 
                    ingredients, 
                    instructions, 
                    constraint
                )
                
                out['accuracy'] = float(d.get('confidence', 0.85) * 100)
                return out
            
            # Fallback
            print(f" Not in database for modification: {name}")
            constraint_text = f" with {constraint}" if constraint else ""
            return self.engine.estimate_nutrition(f"{name}{constraint_text}")
            
        except Exception as e:
            print(f" Modification error: {e}")
            constraint_text = f" with {constraint}" if constraint else ""
            return self.engine.estimate_nutrition(f"{name}{constraint_text}")

    def handle_comparison(self, dishes, goal):
        """PATHWAY 3: Compare two dishes - with estimation fallback"""
        try:
            if len(dishes) < 2:
                print(f" Comparison needs 2 dishes, got {len(dishes)}")
                return {
                    'error': 'Need two dishes to compare',
                    'llm_response': 'Please specify two dishes to compare.'
                }
            
            print(f" Comparing: {dishes[0]} vs {dishes[1]}")
            
            # Try to find dish A in database
            res_a = search_recipe(dishes[0], self.df)
            found_a = res_a and res_a.get('status') == "FOUND" and res_a.get('results')
            
            if found_a:
                dish_a_data = res_a['results'][0]
                dish_a_name = dish_a_data.get('recipe_name', dishes[0])
                nutrition_a = dish_a_data.get('nutrition', {})
                is_a_estimated = False
                print(f" Dish A found in database: {dish_a_name}")
            else:
                dish_a_name = dishes[0]
                nutrition_a = self.engine.estimate_single_dish_nutrition(dishes[0])
                is_a_estimated = True
                print(f" Dish A not found, estimating: {dish_a_name}")
            
            # Try to find dish B in database
            res_b = search_recipe(dishes[1], self.df)
            found_b = res_b and res_b.get('status') == "FOUND" and res_b.get('results')
            
            if found_b:
                dish_b_data = res_b['results'][0]
                dish_b_name = dish_b_data.get('recipe_name', dishes[1])
                nutrition_b = dish_b_data.get('nutrition', {})
                is_b_estimated = False
                print(f" Dish B found in database: {dish_b_name}")
            else:
                dish_b_name = dishes[1]
                nutrition_b = self.engine.estimate_single_dish_nutrition(dishes[1])
                is_b_estimated = True
                print(f" Dish B not found, estimating: {dish_b_name}")
            
            # Now compare (works whether estimated or not!)
            out = self.engine.compare_dishes(
                dish_a_name, 
                nutrition_a,
                dish_b_name, 
                nutrition_b, 
                goal,
                is_a_estimated=is_a_estimated,
                is_b_estimated=is_b_estimated
            )
            
            # Calculate average confidence
            if not is_a_estimated and not is_b_estimated:
                conf_a = res_a['results'][0].get('confidence', 0.85)
                conf_b = res_b['results'][0].get('confidence', 0.85)
                out['accuracy'] = float((conf_a + conf_b) / 2 * 100)
            elif is_a_estimated and is_b_estimated:
                out['accuracy'] = 55.0  # Both estimated
            else:
                out['accuracy'] = 70.0  # One estimated
            
            print(f" Comparison complete (A estimated: {is_a_estimated}, B estimated: {is_b_estimated})")
            return out
            
        except Exception as e:
            print(f" Comparison error: {e}")
            import traceback
            traceback.print_exc()
            goal_text = f" for {goal}" if goal else ""
            return self.engine.estimate_nutrition(f"Compare {dishes[0]} and {dishes[1]}{goal_text}")