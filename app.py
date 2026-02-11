# from flask import Flask, request, jsonify
# import json, os

# # DB & Schema
# from core.db_builder import build_database, execute_sql
# from core.schema_loader import SchemaLoader

# # Agents
# from agents.text2sql_agent import Text2SQLAgent
# from agents.summarizer_agent import SummarizerAgent

# # Utilities
# from utils.intent import wants_chart
# from utils.persist import persist_order_log

# # Email & Summary
# from mailer import send_success_email
# from summary_generator import generate_llm_summary

# app = Flask(__name__)

# # -----------------------------
# # LOAD SCHEMA
# # -----------------------------
# schema = [
#     {"table_name": "sku_daily_forecasts", "path": "datasets/sku_daily_forecast.csv"},
#     {"table_name": "order_log", "path": "datasets/order_log.csv"},
#     {"table_name": "sku_daily_sales", "path": "datasets/sku_daily_sales.csv"},
#     {"table_name": "raw_material_inventory", "path": "datasets/raw_material_inventory.csv"},

# ]

# schema_loader = SchemaLoader(schema)
# loaded_schema = schema_loader.load()

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# db_path = os.path.join(BASE_DIR, "local.db")
# build_database(schema, db_path)

# with open("schema_metadata.json", "r") as f:
#     schema_metadata = json.load(f)

# t2s = Text2SQLAgent(db_path, loaded_schema, schema_metadata)
# summarizer = SummarizerAgent()

# # -----------------------------
# # API Endpoint
# # -----------------------------
# @app.route("/query", methods=["POST"])
# def query():
#     body = request.json
#     question = body.get("question", "")

#     sql = t2s.run(question)
#     result = execute_sql(db_path, sql)

#     # WRITE (INSERT / UPDATE)
#     if isinstance(result, int):
#         email_content = generate_llm_summary(sql, result)

#         persist_order_log(db_path)

#         send_success_email(
#             subject=email_content["subject"],
#             body=email_content["body"]
#         )

#         return jsonify({
#             "sql": sql,
#             "rows_affected": result,
#             "email_subject": email_content["subject"],
#             "email_body": email_content["body"]
#         })

#     # READ (SELECT)
#     summary = summarizer.summarize(question, result)
#     data = result.to_dict(orient="records")

#     if wants_chart(question) and not result.empty:
#         viz, mime = summarizer.generate_viz(question, result)
#     else:
#         viz, mime = None, None

#     return jsonify({
#         "sql": sql,
#         "data": data,
#         "summary": summary,
#         "viz": viz,
#         "mime": mime
#     })

# if __name__ == "__main__":
#     app.run(debug=True)
