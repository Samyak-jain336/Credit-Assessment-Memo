"""
annual_report.py — Orchestrates the full annual report PDF ingestion pipeline.

Produces a flat list of chunk dicts ready for a vector store. No LLM calls,
no database calls, no vector store calls. Pure orchestration of pdf_utils
functions and pdfplumber page iteration.
"""

import pdfplumber

from config import STANDALONE_KEYWORDS, CONSOLIDATED_KEYWORDS

from ingestion.pdf_utils import (
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

def _detect_column_layout(page) -> str:
    """Detect whether a page has a two-column layout.

    Splits the page into left and right halves using page.within_bbox,
    checks if both halves have more than 150 characters of stripped text,
    computes the ratio of min to max length, and returns 'two_column' if
    both conditions pass (ratio > 0.3), otherwise returns 'single'.
    """
    width = page.width
    height = page.height

    left_crop = page.within_bbox((0, 0, width / 2, height))
    right_crop = page.within_bbox((width / 2, 0, width, height))

    left_text = (left_crop.extract_text() or "").strip()
    right_text = (right_crop.extract_text() or "").strip()

    left_len = len(left_text)
    right_len = len(right_text)

    # Raised thresholds to avoid false positives on this PDF format:
    # - Many Indian annual reports have a decorative sidebar
    #   ("CORPORATE OVERVIEW | STATUTORY REPORTS | FINANCIAL STATEMENTS")
    #   on every page that reads as a right-hand column but is not
    #   actual two-column content. Raising min chars to 400 per side
    #   filters these out since the sidebar is typically < 200 chars.
    # - Ratio raised to 0.6 to avoid splitting landscape/rotated table
    #   pages down the middle, which produces reversed garbled text
    #   (e.g. "erusolcsiD" instead of "Disclosure").
    if left_len > 400 and right_len > 400:
        ratio = min(left_len, right_len) / max(left_len, right_len)
        if ratio > 0.6:
            return "two_column"

    return "single"


def _extract_single_column(page) -> str:
    """Extract text from a single-column page."""
    return page.extract_text() or ""


def _extract_two_column(page) -> str:
    """Extract text from a two-column page by reading left then right.

    Extracts left half (x: 0 to width/2) and right half (x: width/2 to width)
    using page.within_bbox, then returns left text stripped + newline + right
    text stripped.
    """
    width = page.width
    height = page.height

    left_crop = page.within_bbox((0, 0, width / 2, height))
    right_crop = page.within_bbox((width / 2, 0, width, height))

    left_text = (left_crop.extract_text() or "").strip()
    right_text = (right_crop.extract_text() or "").strip()

    return left_text + "\n" + right_text


def preprocess_pages(pdf) -> list[dict]:
    """Convert a pdfplumber PDF into a flat list of logical page dicts.

    Two-column layout detection/splitting is deferred for now — every
    physical page is treated as a single logical page. Returns dicts with
    keys: page_obj, page_text, logical_page_num, original_page_num,
    is_split_half (always False).
    """
    result = []
    logical_page_num = 1

    for original_page_num, page in enumerate(pdf.pages, start=1):
        layout = _detect_column_layout(page)
        if layout == "two_column":
            raw_text = _extract_two_column(page)
        else:
            raw_text = _extract_single_column(page)

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

        detected = detect_statement_type(page_text)
        if detected != "unknown":
            last_statement_type = detected

        if page_type == "table":
            extracted = extract_table_page(logical_page["page_obj"])
        elif page_type == "mixed":
            extracted = extract_mixed_page(logical_page["page_obj"])
        else:
            extracted = extract_narrative_page(page_text)

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
