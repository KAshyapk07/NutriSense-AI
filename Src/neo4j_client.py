
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
        """Return aggregate node counts for both food clusters (used by /health endpoint)."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recipe)       WITH count(r)  AS recipes
                MATCH (i:Ingredient)   WITH recipes,  count(i)  AS ingredients
                MATCH (c:Cuisine)      WITH recipes,  ingredients, count(c)  AS cuisines
                MATCH (ic:ImageClass)  WITH recipes,  ingredients, cuisines,  count(ic) AS image_classes
                MATCH (fp:FoodProduct) WITH recipes,  ingredients, cuisines,  image_classes, count(fp) AS food_products
                MATCH (b:Brand)        WITH recipes,  ingredients, cuisines,  image_classes, food_products, count(b) AS brands
                MATCH (cat:Category)   WITH recipes,  ingredients, cuisines,  image_classes, food_products, brands, count(cat) AS categories
                MATCH (at:AllergenTag) WITH recipes,  ingredients, cuisines,  image_classes, food_products, brands, categories, count(at) AS allergen_tags
                RETURN recipes, ingredients, cuisines, image_classes,
                       food_products, brands, categories, allergen_tags
                """
            )
            record = result.single()
            return dict(record) if record else {
                "recipes": 0, "ingredients": 0, "cuisines": 0, "image_classes": 0,
                "food_products": 0, "brands": 0, "categories": 0, "allergen_tags": 0,
            }

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Cluster B â€” FoodProduct queries
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def search_products_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search against the product_name_fulltext index (Cluster B)."""
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('product_name_fulltext', $query)
                YIELD node, score
                ORDER BY score DESC
                LIMIT $limit
                OPTIONAL MATCH (node)-[:MADE_BY]->(b:Brand)
                OPTIONAL MATCH (node)-[:IN_CATEGORY]->(cat:Category)
                RETURN
                    node.id               AS id,
                    node.name             AS name,
                    node.generic_name     AS generic_name,
                    node.brand            AS brand,
                    node.category         AS category,
                    node.serving_size     AS serving_size,
                    node.serving_quantity AS serving_quantity,
                    node.nova_group       AS nova_group,
                    node.nutriscore_grade AS nutriscore_grade,
                    node.calories_100g    AS calories_100g,
                    node.fat_100g         AS fat_100g,
                    node.saturated_fat_100g AS saturated_fat_100g,
                    node.carbohydrates_100g AS carbohydrates_100g,
                    node.sugars_100g      AS sugars_100g,
                    node.fiber_100g       AS fiber_100g,
                    node.proteins_100g    AS proteins_100g,
                    node.sodium_100g      AS sodium_100g,
                    node.calcium_100g     AS calcium_100g,
                    node.iron_100g        AS iron_100g,
                    node.vitamin_c_100g   AS vitamin_c_100g,
                    node.folate_100g      AS folate_100g,
                    node.image_url        AS image_url,
                    b.name                AS brand_node,
                    cat.name              AS category_node,
                    score                 AS search_score
                ORDER BY score DESC
                """,
                query=query, limit=limit,
            )
            return [dict(r) for r in result]

    def get_product_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Case-insensitive substring match on FoodProduct.name (Cluster B)."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (fp:FoodProduct)
                WHERE toLower(fp.name) CONTAINS toLower($name)
                OPTIONAL MATCH (fp)-[:MADE_BY]->(b:Brand)
                OPTIONAL MATCH (fp)-[:IN_CATEGORY]->(cat:Category)
                RETURN
                    fp.id               AS id,
                    fp.name             AS name,
                    fp.generic_name     AS generic_name,
                    fp.brand            AS brand,
                    fp.category         AS category,
                    fp.serving_size     AS serving_size,
                    fp.serving_quantity AS serving_quantity,
                    fp.nova_group       AS nova_group,
                    fp.nutriscore_grade AS nutriscore_grade,
                    fp.calories_100g    AS calories_100g,
                    fp.fat_100g         AS fat_100g,
                    fp.saturated_fat_100g AS saturated_fat_100g,
                    fp.carbohydrates_100g AS carbohydrates_100g,
                    fp.sugars_100g      AS sugars_100g,
                    fp.fiber_100g       AS fiber_100g,
                    fp.proteins_100g    AS proteins_100g,
                    fp.sodium_100g      AS sodium_100g,
                    fp.calcium_100g     AS calcium_100g,
                    fp.iron_100g        AS iron_100g,
                    fp.vitamin_c_100g   AS vitamin_c_100g,
                    fp.folate_100g      AS folate_100g,
                    fp.image_url        AS image_url,
                    b.name              AS brand_node,
                    cat.name            AS category_node
                LIMIT 1
                """,
                name=name,
            )
            record = result.single()
            return dict(record) if record else None

    def get_all_product_names(self) -> List[Dict[str, str]]:
        """Return all FoodProduct names (used by fuzzy lookup layer)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (fp:FoodProduct) RETURN fp.id AS id, fp.name AS name, fp.brand AS brand"
            )
            return [dict(r) for r in result]

    def get_products_by_category(self, category: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return products belonging to a given Category node (Cluster B)."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (fp:FoodProduct)-[:IN_CATEGORY]->(cat:Category {name: $category})
                OPTIONAL MATCH (fp)-[:MADE_BY]->(b:Brand)
                RETURN fp.id AS id, fp.name AS name, fp.brand AS brand,
                       fp.calories_100g AS calories_100g, fp.proteins_100g AS proteins_100g,
                       fp.carbohydrates_100g AS carbohydrates_100g, fp.fat_100g AS fat_100g,
                       fp.nutriscore_grade AS nutriscore_grade, b.name AS brand_node
                LIMIT $limit
                """,
                category=category, limit=limit,
            )
            return [dict(r) for r in result]

    def get_allergen_info(self, item_name: str, is_product: bool = False) -> List[str]:
        """
        Return list of AllergenTag names for a Recipe or FoodProduct.
        Traverses: (node)-[:CONTAINS]->(Ingredient)-[:IS_ALLERGEN]->(AllergenTag)
        """
        label = "FoodProduct" if is_product else "Recipe"
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n:{label})
                WHERE toLower(n.name) CONTAINS toLower($name)
                MATCH (n)-[:CONTAINS]->(i:Ingredient)-[:IS_ALLERGEN]->(a:AllergenTag)
                RETURN DISTINCT a.name AS allergen
                ORDER BY allergen
                """,
                name=item_name,
            )
            return [r["allergen"] for r in result]
