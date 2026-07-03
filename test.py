import pdfplumber
from ingestion.pdf_utils import extract_table_page, extract_mixed_page, classify_page

PDF_PATH = r"C:\Users\samya\OneDrive\Documents\GitHub\CamGeneration\durlax\Annual_Report_2024-25.pdf"

PAGES_TO_CHECK = [47, 48, 49]  # 0-indexed, so physical pages 48, 49, 50

with pdfplumber.open(PDF_PATH) as pdf:
    for i in PAGES_TO_CHECK:
        page = pdf.pages[i]
        text = page.extract_text() or ""
        page_type = classify_page(text)
        print(f"\n{'='*60}")
        print(f"physical page {i+1} -> classified as: {page_type}")
        print(f"{'='*60}")

        if page_type == "table":
            extracted = extract_table_page(page)
        elif page_type == "mixed":
            extracted = extract_mixed_page(page)
        else:
            extracted = text

        print(extracted)