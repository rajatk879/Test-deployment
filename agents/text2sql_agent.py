# from utils.llm_factory import load_llm
# from langchain_core.prompts import PromptTemplate

# class Text2SQLAgent:
#     def __init__(self, db_path, schema, schema_metadata):
#         self.db_path = db_path
#         self.schema = schema
#         self.schema_metadata = schema_metadata or {}
#         self.llm = load_llm(temp=0)

#         schema_str = "\n".join(
#             [f"Table {t['table_name']}: {', '.join(t['columns'])}" for t in schema]
#         )

#         self.prompt = PromptTemplate.from_template(
#             """
# You are an expert SQL generator. You ALWAYS return only pure SQL without explanation.

# Database schema:
# {schema}

# Question: {question}

# SQL:
#             """
#         )

#         self.schema_text = schema_str

#     # ---- This is the actual SQL generator ----
#     def generate_sql(self, question: str):
#         p = self.prompt.format(schema=self.schema_text, question=question)
#         response = self.llm.invoke(p)

#         # Gemini returns .content, OpenAI returns string
#         sql = response.content if hasattr(response, "content") else str(response)

#         # Clean backticks / markdown formatting
#         sql = sql.replace("```sql", "").replace("```", "").strip()

#         return sql

#     def run(self, question: str):
#         """
#         Wrapper method used by app.py
#         """
#         return self.generate_sql(question)

from utils.llm_factory import load_llm
from langchain_core.prompts import PromptTemplate


class Text2SQLAgent:
    def __init__(self, db_path, schema, schema_metadata=None):
        self.db_path = db_path
        self.schema = schema
        self.schema_metadata = schema_metadata or {}
        self.llm = load_llm(temp=0)

        self.schema_text = self._build_schema_text()

#         self.prompt = PromptTemplate.from_template(
#             """
# You are an expert SQL generator.

# STRICT RULES:
# - Return ONLY valid SQLite SQL
# - NO explanations
# - NO markdown
# - NO comments

# CRITICAL FORECAST RULE:
# - NEVER filter forecasts directly on `timestamp`
# - ALWAYS compute forecasted time using:
#   datetime(timestamp, '+' || forecast_hour_offset || ' hours')
# - Any date range in the user question applies to the computed forecast time

# RULES:
# - If filtering on textual identifiers (store_id, dc_id, sku_id, city, location):
#   - Always convert the attribute to LOWER() for case-insensitive matching
#   - Use LIKE with wildcards unless the question specifies an exact ID
# - Assume user names (e.g. "dubai") may be partial or case-insensitive
# - Never guess numeric IDs
# - Always generate a unique "order_id" when inserting into order_log

# DATABASE SCHEMA WITH SEMANTICS:
# {schema}

# Question:
# {question}

# SQL:
#             """
#         )

        self.prompt = PromptTemplate.from_template(
    """
You are an expert SQLite SQL generator for a flour milling and production planning database (MC4 domain).

CRITICAL ANTI-HALLUCINATION RULES (MUST FOLLOW):
- ONLY use tables and columns that are EXPLICITLY listed in the schema below
- NEVER invent, guess, or assume column names, table names, or values
- If a column or table is NOT in the schema, it DOES NOT EXIST - do not use it
- ONLY use sample values provided in the metadata - do not invent values
- If unsure about a column name, check the exact spelling in the schema
- If a user asks about data not in the schema, you must inform them it's not available (but still return valid SQL if possible)

STRICT OUTPUT RULES:
- Return ONLY valid SQLite SQL
- NO explanations
- NO markdown
- NO comments
- NO text before or after the SQL

WRITE PERMISSIONS:
- This database is READ-ONLY for all tables
- NEVER INSERT, UPDATE, or DELETE any table
- All tables are for querying only

DOMAIN CONTEXT (MC4 - Flour Milling):
- This database tracks flour production, mill operations, recipes, SKU forecasts, and raw material (wheat) supply
- Key entities: Mills (M1, M2, M3), Recipes (R1, R2, R3, etc.), SKUs (SKU001, SKU002, etc.), Flour Types (Superior, Bakery, Patent, Brown, Superior Brown)
- Forecasts are in TONS (not units), dates are in YYYY-MM-DD format
- Mill operations track scheduled hours, capacity, utilization, and production schedules

FORECAST QUERIES:
- For sku_forecast table: The date column represents the forecast date directly
- Filter by date range using: WHERE date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
- Aggregate forecast_tons using SUM() for totals
- Join with sku_master to get flour_type, brand, pack_size_kg details

FILTERING RULES:
- For mill_id: Use exact match (M1, M2, M3) - case sensitive
- For recipe_id: Use exact match (R1, R2, R3, etc.) - case sensitive
- For sku_id: Use exact match (SKU001, SKU002, etc.) - case sensitive, or use LOWER() for case-insensitive if user provides partial match
- For flour_type: Use case-insensitive partial match: LOWER(flour_type) LIKE '%value%'
- For country (in raw_material): Use case-insensitive partial match: LOWER(country) LIKE '%value%'
- For dates: Use 'YYYY-MM-DD' format, e.g., '2020-01-01'

VALUE VALIDATION:
- Use ONLY the example_values and allowed_values provided in the schema metadata
- For mill_id: Use format 'M1', 'M2', 'M3' (exact match)
- For recipe_id: Use format 'R1', 'R2', 'R3', etc. (exact match)
- For sku_id: Use format 'SKU001', 'SKU002', etc. (exact match preferred, or case-insensitive if needed)
- For flour_type: Use 'Superior', 'Bakery', 'Patent', 'Brown', 'Superior Brown' (case-insensitive match)
- For period formats: Use 'YYYY-MM' for months, 'YYYY-W##' for weeks, YYYY for years

TIME DIMENSION USAGE:
- Join with time_dimension table for time-based filtering (weekend, Ramadan, Hajj, Eid flags)
- Use time_dimension for week/month/year grouping when needed
- Example: JOIN time_dimension td ON your_table.date = td.date WHERE td.is_weekend = 1

AGGREGATION GUIDELINES:
- Use SUM() for tons, hours, quantities that should be totaled
- Use AVG() for rates, percentages, or when averaging across records
- Use COUNT() for counting records
- Group by appropriate dimensions (date, mill_id, recipe_id, flour_type, etc.)

DATABASE SCHEMA WITH COMPLETE METADATA:
{schema}

Question:
{question}

SQL:
"""
)


    # ---------------------------------------------
    # Build schema with column descriptions
    # ---------------------------------------------
    def _build_schema_text(self):
        blocks = []

        # ---------------------------------------------
        # Global rules (very important for LLM behavior)
        # ---------------------------------------------
        global_rules = self.schema_metadata.get("global_rules", {})

        if global_rules:
            rules_block = ["GLOBAL SQL & FORECAST RULES:"]

            for k, v in global_rules.items():
                if isinstance(v, list):
                    rules_block.append(f"- {k}:")
                    for item in v:
                        rules_block.append(f"    - {item}")
                else:
                    rules_block.append(f"- {k}: {v}")

            blocks.append("\n".join(rules_block))

        # ---------------------------------------------
        # Table-level schema
        # ---------------------------------------------
        tables_meta = self.schema_metadata.get("tables", {})

        for table in self.schema:
            tname = table["table_name"]
            cols = table["columns"]

            meta = tables_meta.get(tname, {})
            table_desc = meta.get("description", "")
            forecast_semantics = meta.get("forecast_semantics", {})

            col_lines = []
            col_meta = meta.get("columns", {})

            for c in cols:
                c_info = col_meta.get(c, {})

                line = f"- {c}"

                if isinstance(c_info, dict):
                    parts = []

                    if c_info.get("description"):
                        parts.append(f"Description: {c_info['description']}")

                    if c_info.get("type"):
                        parts.append(f"Type: {c_info['type']}")

                    if c_info.get("semantic_role"):
                        parts.append(f"Semantic role: {c_info['semantic_role']}")

                    if c_info.get("matching_rule"):
                        parts.append(f"SQL matching rule: {c_info['matching_rule']}")

                    if c_info.get("sql_rule_sqlite"):
                        parts.append(f"SQLite SQL rule: {c_info['sql_rule_sqlite']}")

                    if c_info.get("aggregation_rule"):
                        parts.append(f"Aggregation rule: {c_info['aggregation_rule']}")

                    if c_info.get("location_encoding"):
                        parts.append(f"Location encoding: {c_info['location_encoding']}")

                    # Include example values to prevent hallucination
                    if c_info.get("example_values"):
                        examples = ", ".join([str(v) for v in c_info["example_values"][:5]])
                        parts.append(f"Example values: {examples}")

                    if c_info.get("distinct_values_sample"):
                        values = ", ".join([str(v) for v in c_info["distinct_values_sample"][:10]])
                        parts.append(f"Valid values (sample): {values}")

                    if c_info.get("allowed_values"):
                        allowed = ", ".join([str(v) for v in c_info["allowed_values"]])
                        parts.append(f"Allowed values: {allowed}")

                    if c_info.get("distinct_values"):
                        values = ", ".join([str(v) for v in c_info["distinct_values"][:10]])
                        parts.append(f"Valid values (examples): {values}")

                    if c_info.get("semantic_hints"):
                        hints = " | ".join(c_info["semantic_hints"])
                        parts.append(f"Semantic hints: {hints}")

                    if parts:
                        line += "\n    " + "\n    ".join(parts)

                col_lines.append(line)

            # ---------------------------------------------
            # Forecast semantics (critical!)
            # ---------------------------------------------
            forecast_block = ""
            if forecast_semantics:
                forecast_block = (
                    "\nForecast semantics:\n"
                    f"- Definition: {forecast_semantics.get('definition', '')}\n"
                    f"- SQLite expression: {forecast_semantics.get('sql_expression_sqlite', '')}"
                )

            block = (
                f"Table: {tname}\n"
                f"Description: {table_desc}\n"
                f"{forecast_block}\n"
                f"Columns:\n" +
                "\n".join(col_lines)
            )

            blocks.append(block.strip())

        return "\n\n".join(blocks)



    # ---------------------------------------------
    # SQL generation
    # ---------------------------------------------
    def generate_sql(self, question: str):
        prompt = self.prompt.format(
            schema=self.schema_text,
            question=question
        )

        response = self.llm.invoke(prompt)
        sql = response.content if hasattr(response, "content") else str(response)

        # Cleanup
        sql = sql.replace("```sql", "").replace("```", "").strip()

        return sql
    
    def _normalize_like_patterns(self, sql: str) -> str:
        import re

        def repl(match):
            col = match.group(1)
            val = match.group(2)
            tokens = val.lower().split()
            pattern = "%" + "%".join(tokens) + "%"
            return f"LOWER({col}) LIKE '{pattern}'"

        sql = re.sub(
            r"LOWER\((store_id|dc_id)\)\s+LIKE\s+'%([^']+)%'",
            repl,
            sql,
            flags=re.IGNORECASE
        )
        return sql


    def _apply_forecast_time(self, sql: str) -> str:
        if "forecast_hour_offset" in sql and "timestamp BETWEEN" in sql:
            sql = sql.replace(
                "timestamp BETWEEN",
                "datetime(timestamp, '+' || forecast_hour_offset || ' hours') BETWEEN"
            )
        return sql


    def run(self, question: str):
        sql = self.generate_sql(question)
        sql = sql.replace("ILIKE", "LIKE")  # SQLite safety
        sql = self._normalize_like_patterns(sql)
        sql = self._apply_forecast_time(sql)
        return sql

