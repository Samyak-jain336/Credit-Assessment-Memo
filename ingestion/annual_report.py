"""
annual_report.py — Orchestrates the full annual report PDF ingestion pipeline.

Produces a flat list of chunk dicts ready for a vector store. No LLM calls,
no database calls, no vector store calls. Pure orchestration of pdf_utils
functions and pdfplumber page iteration.
"""

import pdfplumber

from config import STANDALONE_KEYWORDS, CONSOLIDATED_KEYWORDS

from ingestion.pdf_utils import (
    detect_two_column_layout,
    split_two_column_page,
    classify_page,
    extract_table_page,
    extract_narrative_page,
    extract_mixed_page,
    chunk_narrative_text,
    chunk_table_text,
)


def load_pdf(filepath: str):
    """Open a PDF file using pdfplumber and return the PDF object.

    The caller is responsible for closing the returned object. This function
    intentionally does not use a context manager so that the PDF remains open
    across the full pipeline and is closed by run_annual_report_ingestion.
    """
    return pdfplumber.open(filepath)


def detect_statement_type(page_text: str) -> str:
    """Determine whether a page belongs to standalone or consolidated statements.

    Lowercases the page text and checks against STANDALONE_KEYWORDS first, then
    CONSOLIDATED_KEYWORDS (imported from config). Returns 'standalone',
    'consolidated', or 'unknown'. Standalone is checked first because some
    reports include the word 'consolidated' in notes on standalone pages.
    """
    lowered = page_text.lower()

    for keyword in STANDALONE_KEYWORDS:
        if keyword in lowered:
            return "standalone"

    for keyword in CONSOLIDATED_KEYWORDS:
        if keyword in lowered:
            return "consolidated"

    return "unknown"


def preprocess_pages(pdf) -> list[dict]:
    """Convert a pdfplumber PDF into a flat list of logical page dicts.

    Detects two-column layouts and splits them into two separate logical pages,
    each with its own sequential logical_page_num. Single-column pages pass
    through as-is. Returns dicts with keys: page_obj, page_text,
    logical_page_num, original_page_num, is_split_half.
    """
    result = []
    logical_page_num = 1

    for original_page_num, page in enumerate(pdf.pages, start=1):
        raw_text = page.extract_text() or ""

        if detect_two_column_layout(page):
            # Two A4 pages printed side-by-side on one physical page.
            # Split into left and right halves, each becoming its own
            # logical page so downstream classification and extraction
            # operate on single-column text rather than garbled merged text.
            left_text, right_text = split_two_column_page(page)

            result.append({
                "page_obj": page,
                "page_text": left_text,
                "logical_page_num": logical_page_num,
                "original_page_num": original_page_num,
                "is_split_half": True,
            })
            result.append({
                "page_obj": page,
                "page_text": right_text,
                "logical_page_num": logical_page_num + 1,
                "original_page_num": original_page_num,
                "is_split_half": True,
            })
            logical_page_num += 2
        else:
            # Standard single-column page — pass through as-is with
            # the raw extracted text for classification downstream.
            result.append({
                "page_obj": page,
                "page_text": raw_text,
                "logical_page_num": logical_page_num,
                "original_page_num": original_page_num,
                "is_split_half": False,
            })
            logical_page_num += 1

    return result


def extract_and_classify_pages(logical_pages: list[dict]) -> list[dict]:
    """Classify each logical page and extract its text content.

    Uses a carry-forward state machine for statement_type: once a page declares
    'standalone' or 'consolidated', all subsequent pages inherit that type until
    a new declaration is found. Returns dicts with keys: text, page_number,
    page_type, statement_type. Skipped and empty pages are omitted.
    """
    result = []
    last_statement_type = "unknown"

    for logical_page in logical_pages:
        page_text = logical_page["page_text"]
        page_type = classify_page(page_text)

        if page_type == "skip":
            continue

        # Update the carry-forward statement type whenever a page
        # explicitly mentions standalone or consolidated keywords.
        # This persists across pages so that financial tables (which
        # rarely repeat the heading) inherit the correct section label.
        detected = detect_statement_type(page_text)
        if detected != "unknown":
            last_statement_type = detected

        # For two-column split pages, the page_obj points to the FULL
        # physical page, so extract_table_page / extract_mixed_page
        # would return un-cropped data from both columns. Fall back to
        # narrative-style extraction using the already-cropped page_text
        # string. In Indian annual reports, two-column pages are always
        # narrative (MDA, Director's Report), never financial tables.
        is_split = logical_page["is_split_half"]

        if page_type == "table" and not is_split:
            extracted = extract_table_page(logical_page["page_obj"])
        elif page_type == "mixed" and not is_split:
            extracted = extract_mixed_page(logical_page["page_obj"])
        else:
            extracted = extract_narrative_page(page_text)

        if is_split and page_type in ("table", "mixed"):
            page_type = "narrative"

        if not extracted or not extracted.strip():
            continue

        result.append({
            "text": extracted,
            "page_number": logical_page["logical_page_num"],
            "page_type": page_type,
            "statement_type": last_statement_type,
        })

    return result


def chunk_all_pages(extracted_pages: list[dict]) -> list[dict]:
    """Chunk all extracted pages into a flat list of chunk dicts.

    Table pages become a single chunk (never split) to preserve row-column
    relationships. Narrative and mixed pages are split into overlapping
    1000-char chunks. A global chunk index runs across all pages so every
    chunk in the final output has a unique, sequential chunk_index.
    """
    result = []
    global_chunk_idx = 0

    for extracted_page in extracted_pages:
        page_type = extracted_page["page_type"]
        text = extracted_page["text"]
        page_num = extracted_page["page_number"]

        if page_type == "table":
            # Tables are kept as a single chunk so that row-column
            # relationships (e.g. "Revenue for FY2024 is 5907.39")
            # are not broken across chunk boundaries.
            chunks = chunk_table_text(text, page_num, global_chunk_idx)
        else:
            # Narrative and mixed pages are split into overlapping
            # fixed-size windows. The chunk_index values returned by
            # chunk_narrative_text start at 0, so we remap them to
            # continue the global sequence.
            chunks = chunk_narrative_text(text, page_num)
            for chunk in chunks:
                chunk["chunk_index"] = global_chunk_idx + chunk["chunk_index"]

        # Attach page-level metadata (page_type and statement_type)
        # to each chunk so downstream consumers know the source
        # classification without needing to look up the original page.
        for chunk in chunks:
            chunk["page_type"] = extracted_page["page_type"]
            chunk["statement_type"] = extracted_page["statement_type"]

        result.extend(chunks)
        global_chunk_idx += len(chunks)

    return result


def run_annual_report_ingestion(
    filepath: str,
    company_name: str,
    fiscal_year: int,
) -> list[dict]:
    """Top-level orchestrator: ingest an annual report PDF into chunk dicts.

    Opens the PDF, preprocesses pages (handling two-column splits), classifies
    and extracts each page, chunks the text, and attaches company metadata.
    Returns a flat list of dicts matching the pipeline chunk schema with keys:
    text, chunk_index, page_number, page_type, statement_type, company_name,
    fiscal_year, document_type.
    """
    pdf = load_pdf(filepath)

    logical_pages = preprocess_pages(pdf)
    extracted_pages = extract_and_classify_pages(logical_pages)
    chunks = chunk_all_pages(extracted_pages)

    # Stamp every chunk with the company-level metadata that was
    # provided at ingestion time. These fields are constant across
    # the entire document and are used downstream for filtering
    # and attribution when querying the vector store.
    for chunk in chunks:
        chunk["company_name"] = company_name
        chunk["fiscal_year"] = fiscal_year
        chunk["document_type"] = "annual_report"

    pdf.close()

    return chunks
