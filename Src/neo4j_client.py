
import os
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()



class Neo4jClient:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "Neo4j credentials not found. Please set NEO4J_URI, "
                "NEO4J_USER, and NEO4J_PASSWORD environment variables."
            )
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def search_recipes_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('recipe_name_fulltext', $query)
                YIELD node, score
                WITH node, score
                ORDER BY score DESC
                LIMIT $limit
                MATCH (node)-[:BELONGS_TO]->(c:Cuisine)
                RETURN 
                    node.id AS id,
                    node.name AS recipe_original,
                    node.food_name AS food_name,
                    node.prep_time_mins AS prep_time_mins,
                    node.instructions AS instructions,
                    node.raw_ingredients AS raw_ingredients,
                    node.calories AS calories,
                    node.carbohydrates AS carbohydrates,
                    node.protein AS protein,
                    node.fats AS fats,
                    node.free_sugar AS free_sugar,
                    node.fibre AS fibre,
                    node.sodium AS sodium,
                    node.calcium AS calcium,
                    node.iron AS iron,
                    node.vitamin_c AS vitamin_c,
                    node.folate AS folate,
                    node.composite_score AS composite_score,
                    c.name AS cuisine,
                    score AS search_score
                ORDER BY score DESC
                """,
                query=query, limit=limit
            )
            recipes = []
            for record in result:
                recipes.append(dict(record))
            return recipes

    def get_recipe_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recipe {name: $name})-[:BELONGS_TO]->(c:Cuisine)
                RETURN 
                    r.id AS id,
                    r.name AS recipe_original,
                    r.food_name AS food_name,
                    r.prep_time_mins AS prep_time_mins,
                    r.instructions AS instructions,
                    r.raw_ingredients AS raw_ingredients,
                    r.calories AS calories,
                    r.carbohydrates AS carbohydrates,
                    r.protein AS protein,
                    r.fats AS fats,
                    r.free_sugar AS free_sugar,
                    r.fibre AS fibre,
                    r.sodium AS sodium,
                    r.calcium AS calcium,
                    r.iron AS iron,
                    r.vitamin_c AS vitamin_c,
                    r.folate AS folate,
                    r.composite_score AS composite_score,
                    c.name AS cuisine
                LIMIT 1
                """,
                name=name
            )
            record = result.single()
            return dict(record) if record else None

    def get_all_recipe_names(self) -> List[Dict[str, str]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recipe)
                RETURN r.name AS name, r.food_name AS food_name, r.composite_score AS composite_score
                """
            )
            return [dict(record) for record in result]

    def get_recipe_by_food_name(self, food_name: str) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recipe)-[:BELONGS_TO]->(c:Cuisine)
                WHERE toLower(r.food_name) CONTAINS toLower($food_name)
                RETURN 
                    r.id AS id,
                    r.name AS recipe_original,
                    r.food_name AS food_name,
                    r.prep_time_mins AS prep_time_mins,
                    r.instructions AS instructions,
                    r.raw_ingredients AS raw_ingredients,
                    r.calories AS calories,
                    r.carbohydrates AS carbohydrates,
                    r.protein AS protein,
                    r.fats AS fats,
                    r.free_sugar AS free_sugar,
                    r.fibre AS fibre,
                    r.sodium AS sodium,
                    r.calcium AS calcium,
                    r.iron AS iron,
                    r.vitamin_c AS vitamin_c,
                    r.folate AS folate,
                    r.composite_score AS composite_score,
                    c.name AS cuisine
                """,
                food_name=food_name
            )
            return [dict(record) for record in result]

    def get_stats(self) -> Dict[str, int]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recipe) WITH count(r) AS recipes
                MATCH (i:Ingredient) WITH recipes, count(i) AS ingredients
                MATCH (c:Cuisine) WITH recipes, ingredients, count(c) AS cuisines
                MATCH (ic:ImageClass) WITH recipes, ingredients, cuisines, count(ic) AS image_classes
                RETURN recipes, ingredients, cuisines, image_classes
                """
            )
            record = result.single()
            return dict(record) if record else {"recipes": 0, "ingredients": 0, "cuisines": 0, "image_classes": 0}
