"""
parser_agent.py — Orchestrates all five ingestion pipelines for a single company.

Runs each document-type ingestion in sequence, tracking progress in the
ingestion_runs MySQL table. Each flag starts at 0 and is set to 1 only
after successful completion — if the agent crashes mid-run, the next
invocation skips already-completed steps automatically.

Entry point: run_parser_agent(folder_path, company_name, fiscal_year, gemini_api_key)

Expected folder structure under folder_path:
    annual_report/      → single .pdf
    audit_report/       → single .pdf
    saverisk/           → single .xlsx or .xls
    exchange_filings/   → one or more .pdf files
    bank_statements/    → one or more .pdf / .xlsx / .xls / .csv files
"""

import os
from pathlib import Path

from db import get_connection
from ingestion.annual_report import run_annual_report_ingestion
from ingestion.audit_report import run_audit_report_ingestion
from ingestion.saverisk import run_saverisk_ingestion
from ingestion.exchange_filings import run_exchange_filings_ingestion
from ingestion.bank_statement import run_bank_statement_ingestion
from vector_store import init_vector_store, add_chunks


# ---------------------------------------------------------------------------
# Flag column names — used by _get_or_create_run and _set_flag
# ---------------------------------------------------------------------------

_FLAG_COLUMNS = [
    "annual_report_parsed",
    "audit_report_parsed",
    "screener_parsed",
    "exchange_filings_parsed",
    "bank_statements_parsed",
]


# ---------------------------------------------------------------------------
# STEP 1 — Get or create an ingestion_runs row for this company
# ---------------------------------------------------------------------------

def _get_or_create_run(cursor, company_name: str) -> dict:
    """Return the ingestion_runs row for this company, creating it if needed.

    Returns a dict with keys matching _FLAG_COLUMNS, each value 0 or 1.
    """
    cursor.execute(
        "SELECT * FROM ingestion_runs WHERE company_name = %s",
        (company_name,),
    )
    row = cursor.fetchone()

    if row is None:
        # First run for this company — insert a fresh row with all flags = 0
        cursor.execute(
            "INSERT INTO ingestion_runs (company_name) VALUES (%s)",
            (company_name,),
        )
        # Re-fetch to get the full row with defaults applied
        cursor.execute(
            "SELECT * FROM ingestion_runs WHERE company_name = %s",
            (company_name,),
        )
        row = cursor.fetchone()

    # Build a dict from column names. cursor.description gives us the
    # column names so we don't have to hardcode ordinal positions.
    col_names = [desc[0] for desc in cursor.description]
    row_dict = dict(zip(col_names, row))

    # Return only the flag columns the caller cares about
    return {flag: row_dict[flag] for flag in _FLAG_COLUMNS}


# ---------------------------------------------------------------------------
# STEP 2 — Set a single flag to 1 after successful ingestion
# ---------------------------------------------------------------------------

def _set_flag(cursor, company_name: str, flag_name: str) -> None:
    """Mark a single ingestion step as completed in the database.

    Uses an f-string for the column name because MySQL doesn't support
    parameterised column names. This is safe — flag_name comes from
    internal code (_FLAG_COLUMNS), never from user input.
    """
    cursor.execute(
        f"UPDATE ingestion_runs SET {flag_name} = 1 WHERE company_name = %s",
        (company_name,),
    )


# ---------------------------------------------------------------------------
# STEP 3 — Find exactly one file in a subfolder by extension
# ---------------------------------------------------------------------------

def _find_single_file(subfolder: Path, extensions: list[str]):
    """Look for exactly one file in subfolder matching any of the given extensions.

    Returns the Path if exactly one match is found.
    Returns None (with a warning) if zero or multiple matches are found.
    Multiple matches are ambiguous — force the user to clean up rather
    than guessing which file to use.
    """
    matches = []
    if subfolder.exists():
        for ext in extensions:
            matches.extend(subfolder.glob(f"*{ext}"))

    if len(matches) == 0:
        print(f"  WARNING: No files with extensions {extensions} found in {subfolder}")
        return None

    if len(matches) > 1:
        print(f"  WARNING: Multiple files found in {subfolder} — cannot auto-select:")
        for m in sorted(matches):
            print(f"    - {m.name}")
        return None

    return matches[0]


# ---------------------------------------------------------------------------
# STEP 4 — Main orchestrator
# ---------------------------------------------------------------------------

def run_parser_agent(
    folder_path: str,
    company_name: str,
    fiscal_year: int,
    gemini_api_key: str,
) -> None:
    """Run all five ingestion pipelines in sequence for a single company.

    Tracks progress in ingestion_runs — skips already-completed steps.
    Each step is independently wrapped in try/except so one failure
    never blocks subsequent steps.
    """
    base = Path(folder_path)

    # Build expected subfolder paths
    annual_report_dir    = base / "annual_report"
    audit_report_dir     = base / "audit_report"
    saverisk_dir         = base / "saverisk"
    exchange_filings_dir = base / "exchange_filings"
    bank_statements_dir  = base / "bank_statements"

    # Open a single MySQL connection for the entire run — flag updates
    # are committed individually so partial progress is always saved.
    conn = get_connection()
    cursor = conn.cursor()

    # Get or create the ingestion_runs row for this company
    run_state = _get_or_create_run(cursor, company_name)
    conn.commit()

    print(f"\n{'='*60}")
    print(f"PARSER AGENT — {company_name}")
    print(f"{'='*60}")
    print(f"Folder: {folder_path}")
    print(f"Fiscal year: {fiscal_year}")
    print(f"\nCurrent ingestion state:")
    for flag, val in run_state.items():
        status = "DONE ✓" if val else "PENDING"
        print(f"  {flag}: {status}")
    print()

    # ------------------------------------------------------------------
    # 1. ANNUAL REPORT
    # ------------------------------------------------------------------
    if run_state["annual_report_parsed"]:
        print("[1/5] Annual report: SKIPPED (already parsed)")
    else:
        print("[1/5] Annual report: STARTING...")
        try:
            if not annual_report_dir.exists():
                print(f"  WARNING: Folder not found: {annual_report_dir} — skipping")
            else:
                filepath = _find_single_file(annual_report_dir, [".pdf"])
                if filepath:
                    chunks = run_annual_report_ingestion(
                        filepath=str(filepath),
                        company_name=company_name,
                        fiscal_year=fiscal_year,
                    )
                    collection, _ = init_vector_store(api_key=gemini_api_key)
                    add_chunks(collection, chunks)
                    print(f"  {len(chunks)} chunks stored in ChromaDB")
                    _set_flag(cursor, company_name, "annual_report_parsed")
                    conn.commit()
                    print("  Annual report ingestion COMPLETE ✓")
        except Exception as e:
            print(f"  ERROR in annual report ingestion: {e}")

    # ------------------------------------------------------------------
    # 2. AUDIT REPORT
    # ------------------------------------------------------------------
    if run_state["audit_report_parsed"]:
        print("[2/5] Audit report: SKIPPED (already parsed)")
    else:
        print("[2/5] Audit report: STARTING...")
        try:
            if not audit_report_dir.exists():
                print(f"  WARNING: Folder not found: {audit_report_dir} — skipping")
            else:
                filepath = _find_single_file(audit_report_dir, [".pdf"])
                if filepath:
                    chunks = run_audit_report_ingestion(
                        filepath=str(filepath),
                        company_name=company_name,
                        fiscal_year=fiscal_year,
                    )
                    collection, _ = init_vector_store(api_key=gemini_api_key)
                    add_chunks(collection, chunks)
                    print(f"  {len(chunks)} chunks stored in ChromaDB")
                    _set_flag(cursor, company_name, "audit_report_parsed")
                    conn.commit()
                    print("  Audit report ingestion COMPLETE ✓")
        except Exception as e:
            print(f"  ERROR in audit report ingestion: {e}")

    # ------------------------------------------------------------------
    # 3. SCREENER / SAVERISK (Excel financial data)
    # ------------------------------------------------------------------
    if run_state["screener_parsed"]:
        print("[3/5] Screener (saverisk): SKIPPED (already parsed)")
    else:
        print("[3/5] Screener (saverisk): STARTING...")
        try:
            if not saverisk_dir.exists():
                print(f"  WARNING: Folder not found: {saverisk_dir} — skipping")
            else:
                filepath = _find_single_file(saverisk_dir, [".xlsx", ".xls"])
                if filepath:
                    run_saverisk_ingestion(
                        filepath=str(filepath),
                        company_name=company_name,
                    )
                    _set_flag(cursor, company_name, "screener_parsed")
                    conn.commit()
                    print("  Screener ingestion COMPLETE ✓")
        except Exception as e:
            print(f"  ERROR in screener ingestion: {e}")

    # ------------------------------------------------------------------
    # 4. EXCHANGE FILINGS (folder of PDFs)
    # ------------------------------------------------------------------
    if run_state["exchange_filings_parsed"]:
        print("[4/5] Exchange filings: SKIPPED (already parsed)")
    else:
        print("[4/5] Exchange filings: STARTING...")
        try:
            if not exchange_filings_dir.exists():
                print(f"  WARNING: Folder not found: {exchange_filings_dir} — skipping")
            else:
                # Check if any PDFs exist before calling the ingestion function
                pdfs = list(exchange_filings_dir.glob("*.pdf"))
                if not pdfs:
                    print(f"  WARNING: No PDF files in {exchange_filings_dir} — skipping")
                else:
                    run_exchange_filings_ingestion(
                        folder_path=str(exchange_filings_dir),
                        company_id=company_name,
                        gemini_api_key=gemini_api_key,
                    )
                    _set_flag(cursor, company_name, "exchange_filings_parsed")
                    conn.commit()
                    print("  Exchange filings ingestion COMPLETE ✓")
        except Exception as e:
            print(f"  ERROR in exchange filings ingestion: {e}")

    # ------------------------------------------------------------------
    # 5. BANK STATEMENTS (multiple files of various formats)
    # ------------------------------------------------------------------
    if run_state["bank_statements_parsed"]:
        print("[5/5] Bank statements: SKIPPED (already parsed)")
    else:
        print("[5/5] Bank statements: STARTING...")
        try:
            if not bank_statements_dir.exists():
                print(f"  WARNING: Folder not found: {bank_statements_dir} — skipping")
            else:
                # Collect all supported bank statement files
                bank_files = []
                for ext in [".pdf", ".xlsx", ".xls", ".csv"]:
                    bank_files.extend(bank_statements_dir.glob(f"*{ext}"))
                bank_files = sorted(bank_files)

                if not bank_files:
                    print(f"  WARNING: No bank statement files in {bank_statements_dir} — skipping")
                else:
                    all_succeeded = True
                    for bf in bank_files:
                        try:
                            print(f"\n  Processing: {bf.name}")
                            run_bank_statement_ingestion(
                                filepath=str(bf),
                                company_name=company_name,
                                gemini_api_key=gemini_api_key,
                            )
                        except Exception as file_err:
                            # If any single file fails, stop processing remaining
                            # files and do NOT set the flag — partial completion
                            # must be retried from scratch.
                            print(f"  ERROR processing {bf.name}: {file_err}")
                            all_succeeded = False
                            break

                    # Only mark complete if every file succeeded
                    if all_succeeded:
                        _set_flag(cursor, company_name, "bank_statements_parsed")
                        conn.commit()
                        print("  Bank statements ingestion COMPLETE ✓")
        except Exception as e:
            print(f"  ERROR in bank statements ingestion: {e}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    # Re-fetch the final state to show accurate results
    final_state = _get_or_create_run(cursor, company_name)

    print(f"\n{'='*60}")
    print(f"PARSER AGENT — FINAL SUMMARY")
    print(f"{'='*60}")
    for flag, val in final_state.items():
        status = "DONE ✓" if val else "INCOMPLETE ✗"
        print(f"  {flag}: {status}")
    print()

    # Clean up the connection
    cursor.close()
    conn.close()
