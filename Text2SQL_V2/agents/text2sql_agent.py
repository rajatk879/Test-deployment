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
You are an expert SQLite SQL generator.

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

WRITE PERMISSIONS (VERY IMPORTANT):
- INSERT is allowed ONLY for these tables:
  1) order_log
     Columns (in order):
     - date
     - sku_id
     - store_id
     - sales_channel
     - actual_sales_units

  2) raw_material_log
     Columns (in order):
     - date
     - raw_material
     - opening_inventory
     - inflow_quantity
     - consumed_quantity
     - closing_inventory
     - safety_stock

- NEVER INSERT into any other table
- NEVER UPDATE any table
- NEVER DELETE any table

CRITICAL FORECAST RULE:
- For forecast tables (sku_daily_forecast_7day, sku_daily_forecast_30day):
  - For SIMPLE queries asking for "7-day forecast" or "30-day forecast" or "forecasted units" WITHOUT a specific date range:
    * Just filter by sku_id and store_id (forecast_horizon is optional since table name already indicates the horizon)
    * Sum the forecast_units directly - DO NOT compute forecast_date
    * Example: SELECT SUM(forecast_units) as total_forecast FROM sku_daily_forecast_7day WHERE sku_id = 'WL-SKU-001' AND store_id = 'Store_01'
  - For queries asking for forecasts IN A SPECIFIC DATE RANGE:
    * Compute forecast_date using: date(date, '+' || CAST(REPLACE(forecast_horizon, 'day', '') AS INTEGER) || ' days')
    * Filter on the computed forecast_date, not the original date column
    * Example: WHERE date(date, '+' || CAST(REPLACE(forecast_horizon, 'day', '') AS INTEGER) || ' days') BETWEEN '2026-01-01' AND '2026-01-31'

FILTERING RULES:
- For text identifiers (store_id, sku_id, raw_material, product_id):
  - Use LOWER(column) for case-insensitive matching
  - Use LIKE '%value%' unless exact match is specified
  - Example: LOWER(store_id) LIKE '%store_01%' or store_id = 'Store_01' for exact match

VALUE VALIDATION:
- Use ONLY the example_values and allowed_values provided in the schema metadata
- For store_id: Use format 'Store_XX' (e.g., 'Store_01', 'Store_02')
- For sku_id: Use format 'WL-SKU-XXX' (e.g., 'WL-SKU-001')
- For product_id: Use format 'WL-PROD-XXX' (e.g., 'WL-PROD-101')
- For sales_channel: Use 'E-Commerce' or 'Offline Retail' (exact match)
- For raw_material: Use values like 'Leather_FG', 'Rubber_Sole', 'EVA_Foam' (case-insensitive match)

DATA AVAILABILITY RULE (MANDATORY):
•⁠  ⁠This database DOES NOT contain actual sales transactions.
•⁠  ⁠All SKU-related numbers are FORECASTED demand only.
•⁠  ⁠There is NO sales_channel data.
•⁠  ⁠If a question asks about actual sales, revenue, orders, or sales channels:
  - Do NOT generate SQL
  - Return an empty SELECT (SELECT NULL WHERE 1=0)

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

