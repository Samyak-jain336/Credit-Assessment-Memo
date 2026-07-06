"""
bank_statement.py — Dual-path ingestion pipeline for bank statement files.

Path 1: Extract raw text → chunk → push to ChromaDB (for RAG pattern queries).
Path 2: LLM call on full extracted text → extract structured summary →
        insert into MySQL bank_statements table.

Supports PDF, Excel (.xlsx/.xls), and CSV bank statement formats.

Entry point: run_bank_statement_ingestion(filepath, company_name, gemini_api_key)
"""

import json
import re
import os
from datetime import date
from pathlib import Path

import pdfplumber
import pandas as pd

from db import get_connection
from llm_utils import call_llm_json
from vector_store import init_vector_store, add_chunks


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Helpers — safe type casting (same pattern as saverisk.py)
# ---------------------------------------------------------------------------

def _cast_decimal(val):
    """Cast a value to float for DECIMAL columns, or None if missing/null."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _cast_int(val, default=None):
    """Cast a value to int for INT/BIGINT columns, or default if missing/null."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# STEP 1 — Extract raw text from any supported file format
# ---------------------------------------------------------------------------

def extract_raw_text(filepath: str) -> str:
    """Extract all text from a bank statement file regardless of format.

    Supports .pdf, .xlsx, .xls, and .csv. This format-agnostic extraction
    is intentional — the LLM handles interpretation of the content, not
    the parser. We just need clean text to feed into the prompt.
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".pdf":
        # PDF: iterate pages, extract text from each, join with double newline
        pages = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text)
        return "\n\n".join(pages)

    elif ext in (".xlsx", ".xls"):
        # Excel: read all sheets, convert each to string representation.
        # Using sheet_name=None returns a dict of {sheet_name: DataFrame}.
        sheets = pd.read_excel(filepath, sheet_name=None, header=None)
        parts = []
        for sheet_name, df in sheets.items():
            parts.append(f"=== Sheet: {sheet_name} ===")
            parts.append(df.to_string())
        return "\n\n".join(parts)

    elif ext == ".csv":
        # CSV: single sheet, convert directly to string
        df = pd.read_csv(filepath, header=None)
        return df.to_string()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# STEP 2 — Chunk text for ChromaDB storage
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    company_name: str,
    source_filename: str,
) -> list[dict]:
    """Slide a fixed-size window across the text to produce chunks for ChromaDB.

    Each chunk carries metadata compatible with add_chunks() in vector_store.py.
    """
    chunks = []
    chunk_idx = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP  # How far the window advances

    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text_str = text[start:end]

        # Skip very short trailing fragments (< 50 chars)
        if len(chunk_text_str.strip()) < 50:
            start += step
            continue

        chunks.append({
            "text": chunk_text_str,
            "chunk_index": chunk_idx,
            # page_number is 0 because bank statements don't have meaningful
            # page boundaries for ChromaDB retrieval purposes — the text is
            # extracted as one continuous stream regardless of source format.
            "page_number": 0,
            "page_type": "narrative",
            "statement_type": "unknown",
            "company_name": company_name,
            # fiscal_year is 0 because bank statements are not tied to a
            # specific fiscal year — they cover arbitrary date ranges.
            "fiscal_year": 0,
            "document_type": "bank_statement",
            "source_filename": source_filename,
        })

        chunk_idx += 1
        start += step

    return chunks


# ---------------------------------------------------------------------------
# STEP 3 — Extract structured summary via LLM
# ---------------------------------------------------------------------------

def extract_summary_via_llm(raw_text: str, source_filename: str) -> dict:
    """Ask the LLM to extract a structured summary from bank statement text.

    Returns a dict with bank_name, account_type, account_number, balances,
    transaction totals, and any other noteworthy findings.
    """
    # Header: account details, bank name, statement period
    header_section = raw_text[:2000]
    # Tail: TOTALS row, closing balance, interest charges
    tail_section = raw_text[-2000:]
    # Middle sample: enough transactions for EMI/bounce pattern detection
    mid_start = max(2000, len(raw_text) // 2 - 1000)
    mid_section = raw_text[mid_start:mid_start + 2000]

    prompt = f"""You are a financial analyst extracting structured data from a bank statement.

Source file: {source_filename}

Here is text extracted from the bank statement in three sections:

--- HEADER SECTION (account details, bank name, period) ---
{header_section}

--- MIDDLE SECTION (sample transactions for pattern detection) ---
{mid_section}

--- TAIL SECTION (totals row, closing balance, charges) ---
{tail_section}

Extract the following fields and return as a JSON object:

{{
  "bank_name": "Name of the bank",
  "account_type": "Current / Savings / OD / CC etc.",
  "account_number": "Account number as string",
  "sanctioned_limit": null or numeric (for OD/CC accounts),
  "statement_period_from": "YYYY-MM-DD",
  "statement_period_to": "YYYY-MM-DD",
  "currency": "INR",
  "unit": "Rupees",
  "avg_monthly_balance": null or numeric,
  "total_credits": null or numeric (sum of all credits if stated),
  "total_debits": null or numeric (sum of all debits if stated),
  "closing_balance": null or numeric,
  "emi_count": 0,
  "bounce_count": 0,
  "other_findings": {{}}
}}

Rules:
- Return ONLY valid JSON, no explanation, no markdown.
- All numeric fields should be plain numbers without commas (e.g. 415382.50 not 4,15,382.50).
- If a field cannot be determined, set it to null.
- emi_count and bounce_count must be integers, not null — use 0 if none found.
- other_findings must be a JSON object, not null — use {{}} if nothing extra.
  Put any noteworthy fields not in the schema above here (e.g. overdraft limit,
  interest charged, bank charges, processing fees).
"""

    return call_llm_json(prompt)


# ---------------------------------------------------------------------------
# STEP 4 — Insert structured summary into MySQL
# ---------------------------------------------------------------------------

def insert_into_db(
    company_name: str,
    summary: dict,
    source_filename: str,
) -> int:
    """Insert the LLM-extracted summary into the bank_statements table.

    Looks up company_id first — raises ValueError if the company doesn't
    exist yet (run saverisk ingestion first to create the company row).
    Returns company_id for logging.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Look up company_id — company must already exist from saverisk ingestion
    cursor.execute("SELECT id FROM companies WHERE company_name = %s", (company_name,))
    row = cursor.fetchone()

    if not row:
        cursor.close()
        conn.close()
        raise ValueError(
            f"Company not found in DB: {company_name}. "
            "Run saverisk ingestion first."
        )

    company_id = row[0]

    # Parse date fields safely — LLM should return YYYY-MM-DD format
    # but might return null or an unparseable string.
    def _parse_date(val):
        if not val:
            return None
        try:
            return date.fromisoformat(val)
        except (ValueError, TypeError):
            return None

    period_from = _parse_date(summary.get("statement_period_from"))
    period_to = _parse_date(summary.get("statement_period_to"))

    # Build other_findings JSON — ensure it's always a valid JSON string
    other_findings = summary.get("other_findings")
    if not other_findings or not isinstance(other_findings, dict):
        other_findings_json = "{}"
    else:
        other_findings_json = json.dumps(other_findings, default=str)

    # Insert — no ON DUPLICATE KEY UPDATE because a company can have
    # multiple bank statements (different months, different banks).
    cursor.execute(
        """
        INSERT INTO bank_statements (
            company_id, bank_name, account_type, account_number,
            sanctioned_limit, statement_period_from, statement_period_to,
            currency, unit, avg_monthly_balance, total_credits, total_debits,
            closing_balance, emi_count, bounce_count, source_filename,
            other_findings
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            company_id,
            summary.get("bank_name"),
            summary.get("account_type"),
            summary.get("account_number"),
            _cast_decimal(summary.get("sanctioned_limit")),
            period_from,
            period_to,
            summary.get("currency", "INR"),
            summary.get("unit", "Rupees"),
            _cast_decimal(summary.get("avg_monthly_balance")),
            _cast_decimal(summary.get("total_credits")),
            _cast_decimal(summary.get("total_debits")),
            _cast_decimal(summary.get("closing_balance")),
            # emi_count and bounce_count default to 0, never null
            _cast_int(summary.get("emi_count"), default=0),
            _cast_int(summary.get("bounce_count"), default=0),
            source_filename,
            other_findings_json,
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()

    return company_id


# ---------------------------------------------------------------------------
# STEP 5 — Top-level orchestrator
# ---------------------------------------------------------------------------

def run_bank_statement_ingestion(
    filepath: str,
    company_name: str,
    gemini_api_key: str,
) -> dict:
    """Ingest a bank statement file into both ChromaDB and MySQL.

    Path 1: Raw text → chunks → ChromaDB (for RAG queries).
    Path 2: Raw text → LLM extraction → MySQL bank_statements table.

    Returns the LLM-extracted summary dict for verification.
    """
    source_filename = Path(filepath).name  # Just the filename, not full path
    ext = Path(filepath).suffix.lower()

    print(f"Processing bank statement: {source_filename}")

    # Step 1: Extract raw text from whatever format we received
    text = extract_raw_text(filepath)
    print(f"  Extracted {len(text)} characters from {ext} file")

    # Step 2: Chunk the text for ChromaDB storage
    chunks = chunk_text(text, company_name, source_filename)
    print(f"  {len(chunks)} chunks produced")

    # Step 3: Store chunks in ChromaDB (embedding happens here via Gemini API)
    collection, _ = init_vector_store(api_key=gemini_api_key)
    add_chunks(collection, chunks)
    print(f"  Chunks stored in ChromaDB")

    # Step 4: Ask the LLM to extract a structured summary
    summary = extract_summary_via_llm(text, source_filename)
    print(f"  LLM summary extracted:")
    for key, value in summary.items():
        # Print each field except other_findings (can be noisy)
        if key != "other_findings":
            print(f"    {key}: {value}")

    # Step 5: Insert the structured summary into MySQL
    company_id = insert_into_db(company_name, summary, source_filename)
    print(f"  Summary inserted into MySQL (company_id={company_id})")

    print(f"\n=== BANK STATEMENT INGESTION COMPLETE ===")

    return summary
