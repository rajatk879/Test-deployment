#chatbot_api.py

import json, os
from core.db_builder import build_database, execute_sql
from core.schema_loader import SchemaLoader
from agents.text2sql_agent import Text2SQLAgent
from agents.summarizer_agent import SummarizerAgent
from utils.intent import wants_chart
from utils.persist import persist_order_log
import os
from mailer import send_success_email
from summary_generator import generate_llm_summary
import logging
import matplotlib
matplotlib.use("Agg")  # ✅ MUST be before pyplot

import matplotlib.pyplot as plt


def setup_logger():
    """
    Sets up a logger with a console handler.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set minimum log level

    # Prevent adding multiple handlers if setup_logger is called multiple times
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Handler log level

        # Create formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger
logger = setup_logger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MC4 Schema - datasets are in backend/datasets (now that we're in backend/)
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# MC4 Schema for Text2SQL chatbot
schema = [
    {
        "table_name": "sku_forecast",
        "path": os.path.join(DATASETS_DIR, "sku_forecast.csv"),
    },
    {
        "table_name": "sku_master",
        "path": os.path.join(DATASETS_DIR, "sku_master.csv"),
    },
    {
        "table_name": "recipe_master",
        "path": os.path.join(DATASETS_DIR, "recipe_master.csv"),
    },
    {
        "table_name": "recipe_eligibility",
        "path": os.path.join(DATASETS_DIR, "recipe_eligibility.csv"),
    },
    {
        "table_name": "mill_master",
        "path": os.path.join(DATASETS_DIR, "mill_master.csv"),
    },
    {
        "table_name": "mill_load",
        "path": os.path.join(DATASETS_DIR, "mill_load.csv"),
    },
    {
        "table_name": "mill_load_weekly",
        "path": os.path.join(DATASETS_DIR, "mill_load_weekly.csv"),
    },
    {
        "table_name": "mill_load_monthly",
        "path": os.path.join(DATASETS_DIR, "mill_load_monthly.csv"),
    },
    {
        "table_name": "mill_load_yearly",
        "path": os.path.join(DATASETS_DIR, "mill_load_yearly.csv"),
    },
    {
        "table_name": "mill_capacity",
        "path": os.path.join(DATASETS_DIR, "mill_capacity.csv"),
    },
    {
        "table_name": "mill_recipe_schedule",
        "path": os.path.join(DATASETS_DIR, "mill_recipe_schedule.csv"),
    },
    {
        "table_name": "mill_recipe_rates",
        "path": os.path.join(DATASETS_DIR, "mill_recipe_rates.csv"),
    },
    {
        "table_name": "bulk_flour_demand",
        "path": os.path.join(DATASETS_DIR, "bulk_flour_demand.csv"),
    },
    {
        "table_name": "recipe_demand",
        "path": os.path.join(DATASETS_DIR, "recipe_demand.csv"),
    },
    {
        "table_name": "recipe_mix",
        "path": os.path.join(DATASETS_DIR, "recipe_mix.csv"),
    },
    {
        "table_name": "raw_material",
        "path": os.path.join(DATASETS_DIR, "raw_material.csv"),
    },
    {
        "table_name": "time_dimension",
        "path": os.path.join(DATASETS_DIR, "time_dimension.csv"),
    },
]



# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# schema = [
#     {
#         "table_name": "sku_daily_forecast",
#         "path": os.path.join(DATASETS_DIR, "sku_daily_forecast.csv")
#     },
#     {
#         "table_name": "order_log",
#         "path": os.path.join(DATASETS_DIR, "order_log.csv")
#     },
#     {
#         "table_name": "sku_daily_sales",
#         "path": os.path.join(DATASETS_DIR, "sku_daily_sales.csv")
#     },
#     {
#         "table_name": "raw_material_inventory",
#         "path": os.path.join(DATASETS_DIR, "raw_material_inventory.csv")
#     },
# ]

# Load schema - filter out files that don't exist
existing_schema = []
for table_info in schema:
    if os.path.exists(table_info["path"]):
        existing_schema.append(table_info)
    else:
        logger.warning(f"⚠️ CSV file not found, skipping: {table_info['path']}")

if not existing_schema:
    logger.error("❌ No valid schema files found! Please run data_generator.py first.")
    raise FileNotFoundError("No MC4 datasets found. Run: cd backend && python data_generator.py")

schema_loader = SchemaLoader(existing_schema)
loaded_schema = schema_loader.load()

db_path = os.path.join(BASE_DIR, "chatbot.db")
# Only build database if it doesn't exist or if explicitly needed
if not os.path.exists(db_path):
    logger.info("🔄 Building chatbot database...")
    build_database(existing_schema, db_path)
    logger.info("✅ Database built successfully")
else:
    logger.info(f"✅ Using existing database: {db_path}")

with open(os.path.join(BASE_DIR, "schema_metadata.json")) as f:
    schema_metadata = json.load(f)

t2s = Text2SQLAgent(db_path, loaded_schema, schema_metadata)
summarizer = SummarizerAgent()


def run_chatbot_query(question: str):
    sql = t2s.run(question)
    result = execute_sql(db_path, sql)

    # ==================================================
    # INSERT handling (order_log / raw_material_log)
    # ==================================================
    if isinstance(result, int):
        rows_affected = result

        table = (
            "raw_material_log"
            if "raw_material_log" in sql.lower()
            else "order_log"
        )

        # --------------------
        # Generate LLM email summary (MANDATORY - retry until success)
        # --------------------
        email_content = None
        summary_retries = 3
        
        for summary_attempt in range(1, summary_retries + 1):
            try:
                logger.info(f"📧 Generating email summary (attempt {summary_attempt}/{summary_retries})...")
                email_content = generate_llm_summary(
                    sql_query=sql,
                    row_count=rows_affected
                )
                
                # Validate that email_content is a dict with required keys
                if not isinstance(email_content, dict):
                    raise ValueError(f"Invalid email_content type: {type(email_content)}, expected dict")
                
                if "subject" not in email_content or "body" not in email_content:
                    raise ValueError(f"Missing required keys in email_content. Got: {list(email_content.keys())}")
                
                if not email_content.get("subject") or not email_content.get("body"):
                    raise ValueError("Empty subject or body in email_content")
                
                logger.info("✅ Email summary generated successfully")
                break  # Success - exit retry loop
                
            except Exception as e:
                logger.warning(f"⚠️ Email summary generation attempt {summary_attempt}/{summary_retries} failed: {str(e)}")
                if summary_attempt < summary_retries:
                    import time
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    # All retries exhausted - summary generation is mandatory
                    logger.error(f"❌ Email summary generation failed after {summary_retries} attempts. Email will NOT be sent.")
                    logger.error(f"   Last error: {type(e).__name__}: {e}")
                    email_content = None

        # --------------------
        # Persist correct CSV
        # --------------------
        try:
            persist_order_log(db_path, table)
            logger.info(f"✅ {table} persisted successfully")
        except Exception as e:
            logger.warning(f"Failed to persist {table}: {str(e)}")

        # --------------------
        # Send success email (ONLY if summary was generated successfully)
        # --------------------
        if email_content:
            try:
                email_sent = send_success_email(
                    subject=email_content["subject"],
                    body=email_content["body"]
                )
                if email_sent:
                    logger.info("✅ Success email sent")
                else:
                    logger.error("❌ Email sending failed - check mailer logs for details")
            except Exception as e:
                logger.error(f"❌ Failed to send email: {str(e)}", exc_info=True)
        else:
            logger.error("❌ Email NOT sent - summary generation failed and is mandatory")

        return {
            "sql": sql,
            "summary": f"✅ Successfully inserted {rows_affected} record(s) into {table}.",
            "rows_affected": rows_affected,
            "email_subject": email_content["subject"] if email_content else None,
            "email_body": email_content["body"] if email_content else None,
            "data": [],
            "viz": None,
            "mime": None
        }

    # ==================================================
    # SELECT handling
    # ==================================================
    df = result

    # ---------- Summary ----------
    if df.empty:
        summary = f"No data found for your query: '{question}'."
        data = []
    else:
        try:
            summary = summarizer.summarize(question, df)
        except Exception as e:
            logger.warning(f"Summarization failed: {str(e)}")
            summary = f"Query returned {len(df)} row(s)."

        data = df.to_dict(orient="records")

    # ---------- Visualization (Step 4) ----------
    viz, mime = None, None
    if wants_chart(question) and not df.empty:
        try:
            logger.info(f"Generating visualization for: {question}")
            viz, mime = summarizer.generate_viz(question, df)
            logger.info("✅ Visualization generated successfully")
        except Exception as e:
            logger.warning(f"Visualization generation failed: {str(e)}")
            viz, mime = None, None

    # Map backend response to frontend expected format
    return {
        "sql_query": sql,
        "answer": summary,
        "data": data,
        "chart": viz,  # Frontend expects 'chart' not 'viz'
        "mime": mime  # Keep mime for reference, but frontend uses chart with data URI
    }