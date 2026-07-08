import os
from reset import reset_all

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
else:
    reset_all(gemini_api_key=GEMINI_API_KEY)
