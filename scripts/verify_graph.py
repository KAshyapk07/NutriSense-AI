"""Verify the Neo4j graph data after migration."""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

with driver.session() as s:
    queries = {
        "Recipe nodes": "MATCH (r:Recipe) RETURN count(r) AS c",
        "Ingredient nodes": "MATCH (i:Ingredient) RETURN count(i) AS c",
        "Cuisine nodes": "MATCH (c:Cuisine) RETURN count(c) AS c",
        "ImageClass nodes": "MATCH (ic:ImageClass) RETURN count(ic) AS c",
        "CONTAINS rels": "MATCH ()-[r:CONTAINS]->() RETURN count(r) AS c",
        "BELONGS_TO rels": "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) AS c",
        "MAPS_TO rels": "MATCH ()-[r:MAPS_TO]->() RETURN count(r) AS c",
    }
    print("=== GRAPH SUMMARY ===")
    for label, query in queries.items():
        result = s.run(query)
        count = result.single()["c"]
        print(f"  {label}: {count}")

    print()
    print("=== SAMPLE: Paneer Butter Masala ===")
    result = s.run("""
        MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient)
        WHERE r.name CONTAINS 'Paneer Butter Masala'
        WITH r, collect(i.name) AS ingredients
        MATCH (r)-[:BELONGS_TO]->(c:Cuisine)
        RETURN r.name AS recipe, r.calories AS cal, r.protein AS protein,
               r.fats AS fats, c.name AS cuisine, ingredients
        LIMIT 1
    """)
    for record in result:
        print(f"  Recipe: {record['recipe']}")
        print(f"  Cuisine: {record['cuisine']}")
        print(f"  Calories: {record['cal']}, Protein: {record['protein']}g, Fats: {record['fats']}g")
        print(f"  Ingredients: {', '.join(record['ingredients'])}")

    print()
    print("=== CUISINE DISTRIBUTION (top 10) ===")
    result = s.run("""
        MATCH (c:Cuisine)<-[:BELONGS_TO]-(r:Recipe)
        RETURN c.name AS cuisine, count(r) AS recipes
        ORDER BY recipes DESC LIMIT 10
    """)
    for record in result:
        print(f"  {record['cuisine']}: {record['recipes']} recipes")

    print()
    print("=== IMAGE CLASS MAPPINGS (sample) ===")
    result = s.run("""
        MATCH (ic:ImageClass)-[:MAPS_TO]->(r:Recipe)
        RETURN ic.name AS image_class, r.name AS recipe
        LIMIT 10
    """)
    for record in result:
        print(f"  {record['image_class']} -> {record['recipe']}")

    print()
    print("=== HIGH PROTEIN RECIPES (>30g) ===")
    result = s.run("""
        MATCH (r:Recipe)-[:BELONGS_TO]->(c:Cuisine)
        WHERE r.protein > 30
        RETURN r.name AS recipe, r.protein AS protein, c.name AS cuisine
        ORDER BY r.protein DESC LIMIT 5
    """)
    for record in result:
        print(f"  {record['recipe']} ({record['cuisine']}): {record['protein']}g protein")

    print()
    print("=== RECIPES WITH INGREDIENT: 'paneer' ===")
    result = s.run("""
        MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient {name: 'paneer'})
        RETURN r.name AS recipe LIMIT 5
    """)
    for record in result:
        print(f"  {record['recipe']}")

driver.close()
print("\n[SUCCESS] Verification complete!")
