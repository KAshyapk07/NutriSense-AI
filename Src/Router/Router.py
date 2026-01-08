from Src.Pathway_1 import pathway1
import json
from Src.LLM.llm_client import OllamaLLMClient as LLMClient
from Src.LLM.llm_engine import LLMEEngine as llm_engine  


class NutriSenseRouter:
    def __init__(self, df, llm_engine, image_model=None):
        self.df = df
        self.engine = llm_engine
        self.image_model = image_model # Your trained CNN

    def classify_intent(self, query):
        """Uses Llama to decide the pathway."""
        prompt = f"""
        Analyze this user query: "{query}"
        Categorize into:
        - "COMPARE": Mentioning two or more foods.
        - "MODIFY": Asking for a healthy version/change (e.g. 'less oil', 'vegan').
        - "EXTRACT": Just asking for info on one food.

        Return ONLY a JSON object:
        {{"pathway": "COMPARE"|"MODIFY"|"EXTRACT", "dishes": ["dish1", "dish2"], "constraint": "modifier text or null"}}
        """
        response = self.engine.llm.generate(prompt)
        try:
            # Extract JSON from potential LLM chatter
            data = json.loads(response[response.find("{"):response.rfind("}")+1])
            return data
        except:
            return {"pathway": "EXTRACT", "dishes": [query], "constraint": None}
        
    def execute(self, text_query=None, image_input=None):
         # PATHWAY 4: IMAGE
        if image_input is not None:
            dish_name, img_conf = self.image_model.predict(image_input) 
            return self.handle_extraction(dish_name, override_conf=img_conf)

        # TEXT PROCESSING
        intent = self.classify_intent(text_query)
    
        # Use .get('constraint') instead of ['constraint'] to avoid KeyErrors
        dishes = intent.get('dishes', [])
        constraint = intent.get('constraint') # Defaults to None if missing

        if intent.get('pathway') == "COMPARE" and len(dishes) >= 2:
         return self.handle_comparison(dishes, constraint)
    
        elif intent.get('pathway') == "MODIFY":
         # Make sure there is at least one dish before accessing index 0
            target_dish = dishes[0] if dishes else text_query
            return self.handle_modification(target_dish, constraint)
    
        else:
            target = dishes[0] if dishes else text_query
            return self.handle_extraction(target)

    # Logic Handlers
    def handle_extraction(self, name, override_conf=None):
        res = pathway1(name, self.df)
        if res['status'] == "FOUND":
            out = res['results'][0]
            out['accuracy'] = (override_conf * 100) if override_conf else (out['confidence'] * 100)
            return out
        return self.engine.estimate_nutrition(name)

    def handle_modification(self, name, constraint):
        res = pathway1(name, self.df)
        if res['status'] == "FOUND":
            d = res['results'][0]
            out = self.engine.modify_recipe(d['recipe_name'], d['nutrition'], d['ingredients'], d['instructions'], constraint)
            out['accuracy'] = d['confidence'] * 100
            return out
        return self.engine.estimate_nutrition(f"{name} with {constraint}")

    def handle_comparison(self, dishes, goal):
        res_a = pathway1(dishes[0], self.df)
        res_b = pathway1(dishes[1], self.df)
        if res_a['status'] == "FOUND" and res_b['status'] == "FOUND":
            out = self.engine.compare_dishes(res_a['results'][0]['recipe_name'], res_a['results'][0]['nutrition'],
                                             res_b['results'][0]['recipe_name'], res_b['results'][0]['nutrition'], goal)
            out['accuracy'] = ((res_a['results'][0]['confidence'] + res_b['results'][0]['confidence'])/2) * 100
            return out
        return self.engine.estimate_nutrition(f"Compare {dishes[0]} and {dishes[1]}")