import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    GOOGLE_MODEL = "models/gemini-2.5-flash"
    OPENAI_MODEL = "gpt-4o-mini"

    if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
        print("⚠️ WARNING: GOOGLE_API_KEY is missing")

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("⚠️ WARNING: OPENAI_API_KEY is missing")

config = Config()
