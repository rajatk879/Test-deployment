"""
Setup Text2SQL database with MC4 schema
"""
import os
import sys
import json

# Add parent directory to path to find Text2SQL_V2
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add Text2SQL_V2 to path
text2sql_path = os.path.join(parent_dir, "Text2SQL_V2")
if text2sql_path not in sys.path:
    sys.path.insert(0, text2sql_path)

# Now import from Text2SQL_V2 modules (they're in the path)
from core.db_builder import build_database, execute_sql
from core.schema_loader import SchemaLoader

# MC4 Schema for Text2SQL
# Get absolute paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(backend_dir, "datasets")

MC4_SCHEMA = [
    # Layer 1 — Master / Dimension tables
    {"table_name": "dim_mill", "path": os.path.join(datasets_dir, "dim_mill.csv")},
    {"table_name": "dim_recipe", "path": os.path.join(datasets_dir, "dim_recipe.csv")},
    {"table_name": "dim_flour_type", "path": os.path.join(datasets_dir, "dim_flour_type.csv")},
    {"table_name": "dim_sku", "path": os.path.join(datasets_dir, "dim_sku.csv")},
    {"table_name": "dim_wheat_type", "path": os.path.join(datasets_dir, "dim_wheat_type.csv")},
    {"table_name": "dim_country", "path": os.path.join(datasets_dir, "dim_country.csv")},
    # Layer 2 — Mapping tables
    {"table_name": "map_flour_recipe", "path": os.path.join(datasets_dir, "map_flour_recipe.csv")},
    {"table_name": "map_recipe_mill", "path": os.path.join(datasets_dir, "map_recipe_mill.csv")},
    {"table_name": "map_sku_flour", "path": os.path.join(datasets_dir, "map_sku_flour.csv")},
    {"table_name": "map_recipe_wheat", "path": os.path.join(datasets_dir, "map_recipe_wheat.csv")},
    {"table_name": "map_wheat_country", "path": os.path.join(datasets_dir, "map_wheat_country.csv")},
    # Layer 3 — Fact / Transactional tables
    {"table_name": "fact_sku_forecast", "path": os.path.join(datasets_dir, "fact_sku_forecast.csv")},
    {"table_name": "fact_bulk_flour_requirement", "path": os.path.join(datasets_dir, "fact_bulk_flour_requirement.csv")},
    {"table_name": "fact_recipe_demand", "path": os.path.join(datasets_dir, "fact_recipe_demand.csv")},
    {"table_name": "fact_mill_capacity", "path": os.path.join(datasets_dir, "fact_mill_capacity.csv")},
    {"table_name": "fact_mill_schedule_daily", "path": os.path.join(datasets_dir, "fact_mill_schedule_daily.csv")},
    {"table_name": "fact_mill_recipe_plan", "path": os.path.join(datasets_dir, "fact_mill_recipe_plan.csv")},
    {"table_name": "fact_wheat_requirement", "path": os.path.join(datasets_dir, "fact_wheat_requirement.csv")},
    {"table_name": "fact_waste_metrics", "path": os.path.join(datasets_dir, "fact_waste_metrics.csv")},
    {"table_name": "raw_material_prices", "path": os.path.join(datasets_dir, "raw_material_prices.csv")},
    # Layer 4 — KPI Snapshot
    {"table_name": "fact_kpi_snapshot", "path": os.path.join(datasets_dir, "fact_kpi_snapshot.csv")},
    # Time
    {"table_name": "time_dimension", "path": os.path.join(datasets_dir, "time_dimension.csv")},
    # Recipe Mix (helper)
    {"table_name": "recipe_mix", "path": os.path.join(datasets_dir, "recipe_mix.csv")},
]

def setup_database():
    """Setup SQLite database for Text2SQL"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(parent_dir, "Text2SQL_V2", "chatbot.db")
    
    print("🔄 Building MC4 database for Text2SQL...")
    build_database(MC4_SCHEMA, db_path)
    
    print("✅ Database setup complete!")
    print(f"📁 Database location: {db_path}")
    
    # Load schema for metadata
    schema_loader = SchemaLoader(MC4_SCHEMA)
    loaded_schema = schema_loader.load()
    
    # SchemaLoader.load() returns a list, so just use MC4_SCHEMA for table names
    print(f"📊 Loaded {len(MC4_SCHEMA)} tables:")
    for table_info in MC4_SCHEMA:
        print(f"   - {table_info['table_name']}")

if __name__ == "__main__":
    setup_database()
