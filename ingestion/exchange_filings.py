"""
exchange_filings.py — Ingestion pipeline for NSE exchange filing PDFs.

Reads all PDFs from a folder, infers document type from filename,
extracts and chunks text, and stores everything in ChromaDB.
No LLM calls, no database calls — pure pdfplumber + ChromaDB.

Entry point: run_exchange_filings_ingestion(folder_path, company_id, gemini_api_key)
"""

import re
import base64
import time
from pathlib import Path

import pdfplumber
import fitz  # pymupdf — pip install pymupdf pillow
import google.genai as genai

from vector_store import init_vector_store, add_chunks
from config import GEMINI_API_KEY, GEMINI_MODEL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Keywords that indicate a page contains financial statement data (balance
# sheet, P&L, annexure etc.) — these pages are only skipped when they are
# also in landscape orientation, since landscape + financial keywords almost
# always means a wide-format table that doesn't chunk well as narrative text.
# Intentionally empty — exchange filing PDFs (board meeting outcomes,
# quarterly results) ARE the financial data source for recent periods.
# Skipping landscape financial pages here silently drops FY26 data
# that has no other source in the pipeline.
LANDSCAPE_SKIP_KEYWORDS = []


# ---------------------------------------------------------------------------
# FUNCTION 1 — Infer document type from the PDF filename
# ---------------------------------------------------------------------------

def infer_document_type(filename: str) -> str:
    """Map a PDF filename to a document_type string for ChromaDB metadata.

    Lowercases and strips punctuation/spaces before matching so that
    filenames like "Outcome_of_BM.pdf" and "OutcomeOfBM.pdf" both match.
    First match wins — order matters.
    """
    # Strip punctuation, underscores, spaces and lowercase for matching
    cleaned = re.sub(r"[^a-z0-9]", "", filename.lower())

    # Shareholders check runs BEFORE board_meeting_outcome because
    # "outcome" is too broad — "EGM_outcome_voting.pdf" should match
    # shareholders_meeting, not board_meeting_outcome.
    if "votingresult" in cleaned or "shareholder" in cleaned or "egm" in cleaned or "agm" in cleaned:
        return "shareholders_meeting"

    if "outcomeofbm" in cleaned or "outcome" in cleaned:
        return "board_meeting_outcome"

    if "preferential" in cleaned or "warrant" in cleaned:
        return "preferential_issue"

    if "creditrating" in cleaned or "rating" in cleaned:
        return "credit_rating"

    # Catch-all ensures no file is silently dropped — every PDF in the
    # folder gets ingested even if we can't infer a specific type.
    return "exchange_filing"


# ---------------------------------------------------------------------------
# FUNCTION 2 — Decide whether a page should be skipped
# ---------------------------------------------------------------------------

def should_skip_page(page) -> bool:
    """Return True if a page is landscape AND contains financial statement keywords.

    We use a dual-condition check (landscape + keyword) rather than skipping
    all landscape pages, because some legitimate content like wide voting
    result tables is presented in landscape without being a financial
    statement. Only pages that are BOTH landscape AND contain known
    financial-statement keywords are skipped.
    """
    # Condition 1: page must be in landscape orientation
    is_landscape = page.width > page.height
    if not is_landscape:
        return False

    # Condition 2: page text must contain at least one financial keyword
    text = (page.extract_text() or "").lower()
    for keyword in LANDSCAPE_SKIP_KEYWORDS:
        if keyword in text:
            return True

    return False


# ---------------------------------------------------------------------------
# FUNCTION 3 — Extract filing date from first page text
# ---------------------------------------------------------------------------

# Common date formats found in NSE exchange filing cover letters.
# Example: "Thursday, April 16, 2026" or "16th April 2026" or "April 16, 2026".
_DATE_PATTERNS = [
    # "Thursday, April 16, 2026" or "April 16, 2026"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?[,\s]*"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}[,\s]+\d{4})",
    # "16th April 2026" or "16 April 2026"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"[,\s]+\d{4})",
]


def extract_filing_date(first_page_text: str) -> str | None:
    """Try to extract a date string from the first page of a filing.

    Searches for common date formats in NSE cover letters. Returns the
    raw matched date string (e.g. "April 16, 2026") or None if no date found.
    The caller can parse this further if needed.
    """
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, first_page_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_text_via_gemini_vision(filepath: str) -> tuple[list[dict], str | None]:
    """Fallback extractor for scanned/image-based PDF pages using Gemini vision.

    Converts each page to a PNG at 200 DPI using pymupdf, sends it to
    Gemini with a structured extraction prompt, and returns page dicts
    in the same format as extract_text_from_pdf(). Used automatically
    when pdfplumber extracts text from fewer than 3 pages — the signal
    that the PDF is scanned rather than text-based.

    Retries each page up to 3 times with 5s/10s/15s backoff (same
    pattern as vector_store.py and llm_utils.py call_gemini()).
    """
    pages = []
    filing_date = None
    client = genai.Client(api_key=GEMINI_API_KEY)

    doc = fitz.open(filepath)

    for page_num, page in enumerate(doc, start=1):
        # Render page to PNG at 200 DPI — sufficient resolution for
        # dense financial tables without excessive image size.
        mat = fitz.Matrix(200/72, 200/72)
        pix = page.get_pixmap(matrix=mat)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()

        prompt = (
            f"This is page {page_num} of an Indian company's NSE exchange "
            f"filing PDF. Extract ALL text exactly as it appears, preserving:\n"
            f"- All financial figures and their row/column labels\n"
            f"- Table structure: use | to separate columns\n"
            f"- Headers, subheaders, and period labels (e.g. Year Ended 31.03.2026)\n"
            f"- Dates, reference numbers, and signatory details\n\n"
            f"If this page contains a financial statement (P&L, Balance Sheet, "
            f"Cash Flow), extract every single row with all period values shown.\n\n"
            f"Return only the extracted text. No commentary, no markdown fences."
        )

        last_error = None
        extracted = ""

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        {
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": img_b64,
                                    }
                                },
                                {"text": prompt},
                            ]
                        }
                    ],
                )
                extracted = response.text.strip()
                break
            except Exception as e:
                last_error = e
                if attempt == 2:
                    print(f"  Gemini vision failed page {page_num} after 3 attempts: {e}")
                else:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                    print(f"  Gemini vision page {page_num} attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)

        if not extracted:
            continue

        # Try to extract filing date from first page text
        if page_num == 1 and filing_date is None:
            filing_date = extract_filing_date(extracted)

        pages.append({
            "text": extracted,
            "page_number": page_num,
        })

    doc.close()
    return pages, filing_date


# ---------------------------------------------------------------------------
# FUNCTION 4 — Extract text from all pages of a single PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> tuple[list[dict], str | None]:
    """Open a PDF and extract cleaned text from each non-skipped page.

    Returns a tuple of (pages, filing_date) where pages is a list of dicts
    with keys: text, page_number (1-indexed), and filing_date is a date
    string extracted from the first page (or None if not found).
    Skipped pages (landscape financial tables) are logged with a warning.
    Empty pages are skipped silently.
    """
    pages = []
    filing_date = None

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Skip landscape pages that contain financial statement data —
            # these are wide-format tables that don't chunk well as text.
            if should_skip_page(page):
                print(f"  Skipping landscape financial page {page_num} in {Path(filepath).name}")
                continue

            raw_text = page.extract_text()
            if not raw_text:
                # Empty page (scanned image, blank, etc.) — skip silently
                continue

            # Clean the text: strip each line, drop blank lines, rejoin.
            # This removes stray whitespace from PDF extraction artefacts.
            lines = raw_text.split("\n")
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            cleaned_text = "\n".join(cleaned_lines)

            if not cleaned_text:
                continue

            # Extract filing date from the first page's text — exchange
            # filings typically have the date in the cover letter.
            if page_num == 1 and filing_date is None:
                filing_date = extract_filing_date(cleaned_text)

            pages.append({
                "text": cleaned_text,
                "page_number": page_num,
            })

    # If pdfplumber extracted text from fewer than 3 pages, this is
    # almost certainly a scanned/image PDF. Fall back to Gemini vision
    # OCR which handles image-embedded financial tables correctly.
    # The threshold is 3 (not 0) because cover letter pages are
    # text-based even when the financial annexures are scanned images.
    if len(pages) < 3:
        print(f"  Only {len(pages)} text page(s) via pdfplumber — "
              f"falling back to Gemini vision OCR...")
        return extract_text_via_gemini_vision(filepath)

    return pages, filing_date


# ---------------------------------------------------------------------------
# FUNCTION 5 — Chunk extracted pages into overlapping windows
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict],
    source_filename: str,
    company_id: str,
    document_type: str,
    filing_date: str | None = None,
    chunk_offset: int = 0,
) -> list[dict]:
    """Slide a fixed-size window across each page's text to produce chunks.

    Each chunk carries metadata compatible with add_chunks() in vector_store.py.
    company_name is set to company_id so the existing add_chunks() function
    works without modification — it expects a company_name field for building
    the chunk ID string.

    chunk_offset is added to the per-file chunk index so that chunk_index
    values are globally unique across the entire ingestion run. This prevents
    ID collisions in add_chunks() when two files share the same document_type
    (e.g. two "exchange_filing" PDFs would otherwise both start at c0).
    """
    chunks = []
    local_idx = 0  # Sequential index within this file
    step = CHUNK_SIZE - CHUNK_OVERLAP  # How far the window advances each iteration

    for page_dict in pages:
        text = page_dict["text"]
        page_number = page_dict["page_number"]

        # Slide the window across the page text
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            # Skip very short chunks (< 50 chars) — these are usually
            # trailing fragments with no useful content.
            if len(chunk_text.strip()) < 50:
                start += step
                continue

            chunks.append({
                "text": chunk_text,
                # chunk_offset ensures globally unique indices across all
                # files in a single ingestion run, preventing ID collisions.
                "chunk_index": chunk_offset + local_idx,
                "page_number": page_number,
                # Exchange filings are always narrative text — no table
                # serialization or mixed-page handling needed.
                "page_type": "narrative",
                # No standalone/consolidated distinction for exchange filings.
                "statement_type": "unknown",
                # company_name is set to company_id so add_chunks() can build
                # its ID string without modification to vector_store.py.
                "company_name": company_id,
                # Exchange filings are not year-specific. fiscal_year=0 is
                # used as an integer sentinel so add_chunks() can include it
                # in the chunk ID string without type errors. The ID format
                # is "{company}_{doctype}_{fiscal_year}_p{page}_c{chunk}" so
                # 0 produces valid, non-colliding IDs like "..._0_p1_c0".
                "fiscal_year": 0,
                "document_type": document_type,
                "source_filename": source_filename,
                # Date from the filing's cover letter, or None if not found.
                "filing_date": filing_date,
            })

            local_idx += 1
            start += step

    return chunks


# ---------------------------------------------------------------------------
# FUNCTION 6 — Top-level orchestrator
# ---------------------------------------------------------------------------

def run_exchange_filings_ingestion(
    folder_path: str,
    company_id: str,
    gemini_api_key: str,
) -> int:
    """Ingest all PDFs in a folder into ChromaDB as exchange filing chunks.

    Returns the total number of chunks stored across all files.
    """
    # Find all PDFs in the folder
    pdf_files = sorted(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        print(f"WARNING: No PDF files found in {folder_path}")
        return 0

    # Initialise ChromaDB with Gemini embeddings
    collection, _ = init_vector_store(api_key=gemini_api_key)

    total_chunks = 0
    files_processed = 0

    for pdf_path in pdf_files:
        filename = pdf_path.name  # Just the filename, not the full path
        print(f"\nProcessing: {filename}")

        # Infer document type from filename pattern
        document_type = infer_document_type(filename)
        print(f"  Inferred type: {document_type}")

        # Extract text from all non-skipped pages, plus filing date
        pages, filing_date = extract_text_from_pdf(str(pdf_path))

        if filing_date:
            print(f"  Filing date: {filing_date}")
        else:
            print(f"  Filing date: not found")

        if not pages:
            print(f"  WARNING: No extractable pages in {filename} — skipping")
            continue

        # Chunk the extracted pages with metadata. chunk_offset ensures
        # globally unique chunk indices across all files in this run.
        chunks = chunk_pages(
            pages=pages,
            source_filename=filename,
            company_id=company_id,
            document_type=document_type,
            filing_date=filing_date,
            chunk_offset=total_chunks,
        )

        if not chunks:
            print(f"  WARNING: No chunks produced from {filename} — skipping")
            continue

        # Store chunks in ChromaDB (embedding happens here via the API)
        add_chunks(collection, chunks)

        print(f"  {len(chunks)} chunks stored for {filename}")
        total_chunks += len(chunks)
        files_processed += 1

    # Final summary
    print(f"\n=== EXCHANGE FILINGS INGESTION COMPLETE ===")
    print(f"  Files processed: {files_processed}/{len(pdf_files)}")
    print(f"  Total chunks stored: {total_chunks}")

    return total_chunks
