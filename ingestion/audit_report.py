"""
audit_report.py — Ingestion pipeline for standalone audit report PDFs.

Reuses the same PDF extraction, classification, and chunking logic as
annual_report.py. The only difference is document_type is stamped as
"audit_report" so chunks can be filtered separately in the vector store.
"""

from ingestion.annual_report import run_annual_report_ingestion


def run_audit_report_ingestion(
    filepath: str,
    company_name: str,
    fiscal_year: int,
) -> list[dict]:
    """Ingest an audit report PDF into chunk dicts.

    Delegates entirely to run_annual_report_ingestion() with document_type
    overwritten to "audit_report" so downstream vector store queries can
    filter audit chunks separately from annual report chunks.
    """
    chunks = run_annual_report_ingestion(
        filepath=filepath,
        company_name=company_name,
        fiscal_year=fiscal_year,
    )

    # Overwrite the document_type stamped by run_annual_report_ingestion()
    # since that function hardcodes "annual_report". Every other field
    # (company_name, fiscal_year, chunk_index, page_type etc.) is correct.
    for chunk in chunks:
        chunk["document_type"] = "audit_report"

    return chunks
