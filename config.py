import os

STANDALONE_KEYWORDS = [
    "standalone financial statements",
    "standalone balance sheet",
    "standalone statement of profit",
    "statement of profit and loss",
    "balance sheet",
    "notes to financial statements",
]

CONSOLIDATED_KEYWORDS = [
    "consolidated financial statements",
    "consolidated balance sheet",
    "consolidated statement of profit",
]

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "data", "chromadb")
COLLECTION_NAME = "cam_documents"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

LLM_PROVIDER   = os.environ.get("LLM_PROVIDER", "gemini")

CAM_MAX_PARALLEL_SECTIONS = int(os.environ.get("CAM_MAX_PARALLEL_SECTIONS", 3))
CAM_MAX_REVIEW_RETRIES = int(os.environ.get("CAM_MAX_REVIEW_RETRIES", 1))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o")

AZURE_OPENAI_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")