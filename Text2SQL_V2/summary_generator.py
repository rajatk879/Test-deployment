# import json
# import re
# from Text2SQL_V2.config import config
# from Text2SQL_V2.utils.llm_factory import load_llm
# import logging

# def setup_logger():
#     """
#     Sets up a logger with a console handler.
#     """
#     # Create a logger
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)  # Set minimum log level

#     # Prevent adding multiple handlers if setup_logger is called multiple times
#     if not logger.handlers:
#         # Create console handler
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.DEBUG)  # Handler log level

#         # Create formatter for log messages
#         formatter = logging.Formatter(
#             '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         )
#         console_handler.setFormatter(formatter)

#         # Add handler to logger
#         logger.addHandler(console_handler)

#     return logger
# logger = setup_logger()

# # -----------------------------
# # Load LLM
# # -----------------------------
# def get_llm(max_tokens: int):
#     # Reload model with token limit for retries
#     return load_llm(temp=0, max_tokens=max_tokens)

# logger.info(f"🤖 LLM Provider: {config.LLM_PROVIDER}")

# # -----------------------------
# # LLM PROMPT
# # -----------------------------
# PROMPT_TEMPLATE = """
# Role:
# You are an expert business communication assistant.

# Task:
# You will be given a SQL query. Convert it into a clear and professional email alert.

# Instructions:
# 1. Understand what the SQL query is doing (filters, metrics, time window).
# 2. Write:
#    - A short subject line.
#    - A concise, well-structured email body in HTML.
#    - Use <b> for important phrases.
#    - Use a clear header and short paragraphs.
#    - Don't include any technical jargon.
#    - Use a professional and positive tone.
#    - Don't mention "forecasted date".
#    - Don't repeat yourself.
#    - Don't make any recommendations or suggestions.
#    - Don't use word like "Manual".
#    - Subject and body must have proper html tags.
# 3. Keep language business-friendly and action-oriented.
# 4. Do NOT mention SQL explicitly in the email body.
# 5. Your entire JSON response must be complete and under 500 tokens.

# Output Format (strict):
# Return ONLY a valid JSON object:

# {{
#   "subject": "<email subject>",
#   "body": "<HTML formatted email body>"
# }}

# Rules:
# - No markdown.
# - Only valid HTML in body (e.g., <h3>, <p>, <b>, <br>).
# - Do not wrap HTML in code blocks.
# - No explanation.
# - Output only JSON.

# For Example:
# {{
#   "subject": "Order Successfully Created for Dubai Hypermarkets",
#   "body": "<h3>Order Confirmation</h3><p>This is to confirm that your request has been <b>successfully processed</b>.</p><p>A new order for <b>SKU_101</b> has been created based on the total predicted demand across all Dubai hypermarkets.</p><p><b>Status:</b> Completed successfully.</p>"
# }}


# SQL Query:
# {operation}

# Row Count: {rows}
# """

# # -----------------------------
# # Robust JSON extractor
# # -----------------------------
# def extract_json(text: str) -> dict:
#     text = text.strip()
#     start = text.find("{")
#     end = text.rfind("}")

#     if start == -1 or end == -1 or end <= start:
#         raise ValueError("No valid JSON boundaries found")

#     json_text = text[start:end + 1]
#     return json.loads(json_text)

# # -----------------------------
# # Fallback
# # -----------------------------
# def fallback_summary(row_count: int | None) -> dict:
#     return {
#         "subject": "Operation Completed Successfully",
#         "body": f"{row_count or 0} record(s) were successfully processed."
#     }

# # -----------------------------
# # LLM Call (NO API)
# # -----------------------------
# def call_llm(prompt: str, llm) -> str:
#     response = llm.invoke(prompt)
#     return response.content

# # -----------------------------
# # Main Summary Generator
# # -----------------------------
# def generate_llm_summary(sql_query: str, row_count: int | None = None) -> dict:
#     prompt = PROMPT_TEMPLATE.format(
#         operation=sql_query,
#         rows=row_count or 0
#     )

#     max_tokens_list = [800, 700, 600]

#     for attempt, max_tokens in enumerate(max_tokens_list, start=1):
#         try:
#             logger.info(f"🚀 Calling LLM (attempt {attempt}, max_tokens={max_tokens})...")
#             llm = get_llm(max_tokens)

#             raw_text = call_llm(prompt, llm)

#             logger.info("🧠 RAW LLM OUTPUT ↓↓↓")
#             logger.info(raw_text)
#             logger.info("🧠 RAW LLM OUTPUT ↑↑↑")

#             parsed = extract_json(raw_text)

#             return {
#                 "subject": parsed.get("subject", f"Operation Completed ({row_count or 0} rows)"),
#                 "body": parsed.get("body", f"{row_count or 0} record(s) were successfully processed.")
#             }

#         except Exception as e:
#             logger.info(f"❌ LLM attempt {attempt} failed:", e)

#     logger.info("⚠️ All LLM attempts failed. Using fallback.")
#     return fallback_summary(row_count)



import json
import re
from config import config
from utils.llm_factory import load_llm
import logging

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# -----------------------------
# Load LLM
# -----------------------------
def get_llm():
    """
    Loads the LLM with appropriate token limits for email summaries.
    Email summaries should be concise, so we use a reasonable token limit.
    """
    # Use 2000 tokens to ensure we get complete responses (subject + body)
    return load_llm(temp=0, max_tokens=2000)

logger.info(f"🤖 LLM Provider: {config.LLM_PROVIDER}")

# -----------------------------
# LLM PROMPT (Woodland Consumption Focused)
# -----------------------------
PROMPT_TEMPLATE = """
Role:
You are an expert business communication assistant for Woodland AI.

Task:
You will be given a SQL query regarding material consumption or demand. Convert it into a professional email alert.

Instructions:
1. Subject line: Professional and action-oriented.
2. Body:
   - Maximum 2 short sentences
   - Plain HTML only (<p>, <b>)
   - Do NOT exceed 40 words
3. Context: Focus on "Consumption Demand" and "Material Readiness".
4. Constraint: Do NOT mention SQL, technical terms, or "forecasted date".
5. Format: Return ONLY a valid JSON object.

Output Format (STRICT):
{{
  "subject": "<subject>",
  "body": "<HTML formatted body>"
}}

SQL Query:
{operation}

Row Count: {rows}
"""

# -----------------------------
# Robust JSON extractor
# -----------------------------
def extract_json(text: str) -> dict:
    # Remove markdown code fences if present
    text = re.sub(r"  ⁠json|⁠  ", "", text).strip()

    # Find JSON boundaries
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        logger.error(f"❌ No valid JSON boundaries found. Text preview: {text[:200]}")
        raise ValueError("No valid JSON boundaries found in LLM output")

    json_text = text[start:end + 1]
    
    try:
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed JSON is not a dict: {type(parsed)}")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error: {e}. JSON text: {json_text[:200]}")
        raise ValueError(f"Invalid JSON format: {e}")


# -----------------------------
# Fallback (Consumption Focused)
# -----------------------------
def fallback_summary(row_count: int | None) -> dict:
    return {
        "subject": "Consumption Data Update Successful",
        "body": f"<h3>System Update</h3><p>Successfully processed <b>{row_count or 0}</b> record(s) related to consumption demand and order logs.</p>"
    }

# -----------------------------
# LLM Call
# -----------------------------
def call_llm(prompt: str, llm) -> str:
    """
    Call LLM and extract response content.
    
    Args:
        prompt: The prompt to send to the LLM
        llm: The LLM instance
        
    Returns:
        str: The response content
        
    Raises:
        Exception: If LLM call fails or response is invalid
    """
    try:
        response = llm.invoke(prompt)
        
        # Handle different response formats
        if hasattr(response, 'content'):
            content = response.content
        elif hasattr(response, 'text'):
            content = response.text
        elif isinstance(response, str):
            content = response
        else:
            # Try to convert to string
            content = str(response)
        
        if not content or not content.strip():
            raise ValueError("LLM returned empty response")
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"❌ LLM invoke failed: {type(e).__name__}: {e}")
        raise

# -----------------------------
# Main Summary Generator
# -----------------------------
def generate_llm_summary(sql_query: str, row_count: int | None = None) -> dict:
    """
    Generate email summary using LLM. Always returns a dict with 'subject' and 'body'.
    
    Args:
        sql_query: The SQL query that was executed
        row_count: Number of rows affected
        
    Returns:
        dict: Always returns a dict with 'subject' and 'body' keys
    """
    # Ensure we always return a valid dict, even if everything fails
    try:
        prompt = PROMPT_TEMPLATE.format(
            operation=sql_query,
            rows=row_count or 0
        )

        # Retry loop - try up to 3 times
        last_exception = None
        for attempt in range(1, 4):
            try:
                logger.info(f"🚀 Calling LLM (attempt {attempt}/3)...")
                llm = get_llm()

                raw_text = call_llm(prompt, llm)

                logger.debug("🧠 RAW LLM OUTPUT ↓↓↓")
                logger.debug(raw_text)
                logger.debug("🧠 RAW LLM OUTPUT ↑↑↑")

                # Parse JSON from LLM response
                parsed = extract_json(raw_text)

                # Validate parsed result
                if not isinstance(parsed, dict):
                    raise ValueError(f"Parsed result is not a dict: {type(parsed)}")
                
                # Ensure required keys exist
                result = {
                    "subject": parsed.get("subject", "Inventory Data Update"),
                    "body": parsed.get("body", f"Updated {row_count or 0} consumption records.")
                }
                
                # Validate result
                if not result.get("subject") or not result.get("body"):
                    raise ValueError("Missing required keys in parsed result")
                
                logger.info(f"✅ LLM summary generated successfully (attempt {attempt})")
                return result

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                error_msg = str(e)
                logger.warning(f"❌ LLM attempt {attempt}/3 failed: {error_type}: {error_msg}")
                if attempt < 3:
                    logger.debug(f"   Retrying... (attempt {attempt + 1}/3)")
                else:
                    logger.debug(f"   All attempts exhausted. Last error: {error_type}: {error_msg}")

    except Exception as e:
        # Catch any unexpected errors in the outer try block
        logger.error(f"❌ Unexpected error in generate_llm_summary: {type(e).__name__}: {e}", exc_info=True)
    
    # Always return fallback if we get here
    logger.info("⚠️ Using fallback summary (LLM generation failed)")
    return fallback_summary(row_count)