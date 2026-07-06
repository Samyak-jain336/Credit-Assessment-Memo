"""
test_ingestion.py — End-to-end test script for the annual report
ingestion pipeline. Run this file directly to process one PDF,
store the chunks in ChromaDB, and inspect the results.

This is a throwaway test script, not part of the core pipeline.
It lives at the project root: CreditAssessmentMemo/test_ingestion.py
"""

import os

from config import TEST_PDF_PATH, TEST_COMPANY_NAME, TEST_FISCAL_YEAR, AUDIT_TEST_PATH
from ingestion.annual_report import run_annual_report_ingestion
from ingestion.audit_report import run_audit_report_ingestion
from vector_store import init_vector_store, add_chunks, query_chunks


# Gemini API key is read from an environment variable rather than
# hardcoded, so the key never gets committed to git by accident.
# Set it before running: setx GEMINI_API_KEY "your-key-here" (Windows)
# or export GEMINI_API_KEY="your-key-here" (Mac/Linux), then restart terminal.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def print_chunk_stats(chunks: list[dict]) -> None:
    """Print a breakdown of chunk counts by page_type and statement_type.

    This gives a quick sanity check on the ingestion output before
    spending time on manual inspection — e.g. are there way too many
    or way too few chunks, and are statement types being detected at all.
    """
    total = len(chunks)
    print(f"\n=== CHUNK STATS ===")
    print(f"Total chunks: {total}")

    if total == 0:
        # If this happens, either the PDF has no extractable text
        # (scanned/image PDF) or every page got classified as "skip".
        print("No chunks produced — check the PDF and thresholds.")
        return

    # Count how many chunks fall into each page_type category
    # (table / mixed / narrative) so we can see the document's shape.
    page_type_counts = {}
    for chunk in chunks:
        pt = chunk["page_type"]
        page_type_counts[pt] = page_type_counts.get(pt, 0) + 1

    print("\nBy page_type:")
    for page_type, count in page_type_counts.items():
        print(f"  {page_type}: {count}")

    # Count how many chunks fall into each statement_type category
    # (standalone / consolidated / unknown) to verify the carry-forward
    # keyword detection is actually catching the right sections.
    statement_type_counts = {}
    for chunk in chunks:
        st = chunk["statement_type"]
        statement_type_counts[st] = statement_type_counts.get(st, 0) + 1

    print("\nBy statement_type:")
    for statement_type, count in statement_type_counts.items():
        print(f"  {statement_type}: {count}")


def print_sample_chunks(chunks: list[dict], n: int = 3) -> None:
    """Print the first few chunks of each page_type so you can visually
    inspect extraction quality — especially whether table serialization
    reads correctly like "Revenue for FY2024 is 5907.39".
    """
    print(f"\n=== SAMPLE CHUNKS (up to {n} per page_type) ===")

    # Group chunks by page_type so we can show a few examples of each
    # kind of extraction (table, mixed, narrative) rather than just
    # the first N chunks overall, which might all be the same type.
    seen_counts = {"table": 0, "mixed": 0, "narrative": 0}

    for chunk in chunks:
        pt = chunk["page_type"]
        if seen_counts.get(pt, 0) >= n:
            continue

        seen_counts[pt] = seen_counts.get(pt, 0) + 1

        print(f"\n--- {pt.upper()} | page {chunk['page_number']} | "
              f"statement_type={chunk['statement_type']} ---")
        # Truncate long chunks so the terminal output stays readable
        preview = chunk["text"][:400]
        print(preview)


def run_test_queries(collection) -> None:
    """Run a handful of representative queries against the ingested
    document and print the top results. This checks retrieval quality —
    does asking for revenue actually surface the revenue table chunk?
    """
    # These queries are deliberately generic financial terms that
    # should exist in almost any Indian SME annual report, so this
    # test works regardless of which company's PDF you're testing.
    test_queries = [
        "revenue from operations",
        "profit after tax",
        "borrowings and long term debt",
        "auditor qualification",
    ]

    print("\n=== TEST QUERIES ===")

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Restrict results to the company we just ingested, so if the
        # collection has other companies' data from previous test runs,
        # we don't get cross-company noise in the results.
        results = query_chunks(
            collection,
            query_text=query,
            filters={"company_name": TEST_COMPANY_NAME},
            n_results=2,
        )

        if not results:
            print("  No results found.")
            continue

        for i, result in enumerate(results, start=1):
            meta = result["metadata"]
            print(f"  [{i}] page={meta['page_number']} "
                  f"type={meta['page_type']} "
                  f"distance={result['distance']:.4f}")
            # Short preview so you can eyeball whether the match is relevant
            print(f"      {result['text'][:150]}")


def main():
    """Run the full parser agent for Durlax."""

    import os
    from parser_agent import run_parser_agent

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return

    run_parser_agent(
        folder_path=r"C:\Users\samya\OneDrive\Documents\GitHub\CreditAssessmentMemo\Inputs\Durlax Top Surface Limited",
        company_name="Durlax Top Surface Limited",
        fiscal_year=2025,
        gemini_api_key=GEMINI_API_KEY,
    )


if __name__ == "__main__":
    main()