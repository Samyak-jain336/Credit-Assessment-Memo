import os

STANDALONE_KEYWORDS = [
    "standalone financial statements",
    "standalone balance sheet",
    "standalone statement of profit",
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