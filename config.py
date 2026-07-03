import os

STANDALONE_KEYWORDS = [
    "standalone financial statements",
    "standalone balance sheet",
    "standalone statement of profit",
    "statement of profit and loss",   # add this
    "balance sheet",                   # add this
    "notes to financial statements",   # add this
]


CONSOLIDATED_KEYWORDS = [
    "consolidated financial statements",
    "consolidated balance sheet",
    "consolidated statement of profit",
]

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "data", "chromadb")
COLLECTION_NAME = "cam_documents"

TEST_PDF_PATH = r"C:\Users\samya\OneDrive\Documents\GitHub\CamGeneration\durlax\Annual_Report_2024-25.pdf"        # e.g. "C:/Users/samya/.../AnnualReport.pdf"
TEST_COMPANY_NAME = "Durlax Top Surface Limited"    # e.g. "Durlax Top Surface Limited"
TEST_FISCAL_YEAR = 2025
AUDIT_TEST_PATH = r"C:\Users\samya\OneDrive\Documents\GitHub\CamGeneration\durlax\Audit_Report_2024-25.pdf"

# -- Groq LLM Config -----------------------------------------
GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")
GROQ_MODEL     = "openai/gpt-oss-120b"