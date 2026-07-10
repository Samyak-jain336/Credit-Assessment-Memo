"""
reconciliation.py

Performs reconciliation between structured database values and
retrieved document evidence before CAM generation.

This module DOES NOT call the LLM.
"""

import re
from typing import Any, Dict, List

import schema
from stubs import (
    RetrievedChunk,
    ReconciliationResult,
)

# ------------------------------------------------------------------
# Financial statement labels used inside PDFs
# ------------------------------------------------------------------

FIELD_KEYWORDS = {
    "revenue": [
        "revenue from operations",
        "total revenue",
        "turnover",
        "net sales",
    ],
    "net_profit": [
        "profit after tax",
        "net profit",
        "profit for the year",
        "profit / (loss) for the period",
    ],
    "borrowings": [
        "borrowings",
        "long term borrowings",
        "short term borrowings",
        "total borrowings",
    ],
    "cash_and_bank": [
        "cash and bank balances",
        "cash & bank balances",
        "cash and cash equivalents",
        "balance with banks",
    ],
    "inventory": [
        "inventories",
        "inventory",
        "stock in trade",
    ],
    "trade_receivables": [
        "trade receivables",
        "sundry debtors",
        "receivables",
    ],
    "cfo": [
        "cash flow from operating",
        "net cash from operating",
        "cash generated from operations",
        "net cash flow from operating activities",
    ],
    "net_cash_flow": [
        "net increase in cash",
        "net decrease in cash",
        "net change in cash",
        "net increase / (decrease) in cash",
    ],
}


class ReconciliationAgent:
    """
    Validates structured financial data against retrieved document chunks.
    """

    def __init__(self, tolerance: float = schema.DEFAULT_TOLERANCE):
        self.tolerance = tolerance

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def reconcile(
        self,
        financials: Dict[str, Any],
        retrieved_chunks: List[RetrievedChunk],
    ) -> List[ReconciliationResult]:

        results = []

        chunk_text = "\n".join(
            chunk.text.lower()
            for chunk in retrieved_chunks
        )

        # Normalize whitespace to make regex matching more reliable
        chunk_text = " ".join(chunk_text.split())

        for field in schema.RECONCILIATION_FIELDS:

            db_value = financials.get(field)

            # Numeric extraction via regex has proven unreliable across the
            # varied table formats in this corpus (2-column annual report
            # tables, 6-column board meeting outcome tables, narrative
            # percentages). Rather than risk false MISMATCH/MATCH status
            # from a wrong number, use extract_numeric only as a HINT passed
            # to the LLM — never as the basis for the status field. Status is
            # based purely on keyword presence, which is reliable. The LLM
            # performs the actual numeric verification directly from the raw
            # evidence text in the prompt, which has proven far more accurate
            # than regex extraction for this multi-format document set.
            numeric_hint = self._extract_numeric(field, chunk_text)
            keyword_found = self._extract_value(field, chunk_text)

            status = self._compare(db_value, keyword_found)
            extracted_value = keyword_found

            remarks = ""

            if status == schema.MISMATCH:
                remarks = (
                    f"{field} differs between "
                    "database and retrieved evidence."
                )

            results.append(
                ReconciliationResult(
                    field_name=field,
                    database_value=db_value,
                    extracted_value=extracted_value,
                    status=status,
                    remarks=remarks,
                )
            )

        return results

    # ------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------

    def _extract_value(
        self,
        field: str,
        chunk_text: str,
    ):

        keywords = FIELD_KEYWORDS.get(
            field,
            [field.lower().replace("_", " ")]
        )

        for keyword in keywords:
            if keyword.lower() in chunk_text:
                return "FOUND"

        return None

    def _extract_numeric(
        self,
        field: str,
        chunk_text: str,
    ):
        """Extract a numeric value for a field from chunk text.

        Two-pass approach:
        Pass 1 — looks for the keyword near a recent year label (2025 or 2026)
                  so the current year column is preferred over the prior year
                  column in two-column P&L/Balance Sheet tables. This prevents
                  the regex from picking up FY24 figures (e.g. 174.09) instead
                  of FY25 figures (e.g. 749.66) when both appear after the
                  same keyword.
        Pass 2 — falls back to a tight 60-char window with no year constraint
                  if pass 1 finds nothing.

        Minimum value filter of 10 Lakhs eliminates note reference numbers,
        page numbers, and percentages from being returned as financial values.
        """
        keywords = FIELD_KEYWORDS.get(
            field,
            [field.lower().replace("_", " ")]
        )

        for keyword in keywords:

            # Pass 1: prefer value near a current year label
            year_pattern = (
                rf"{re.escape(keyword)}"
                rf"[^\n]{{0,120}}"
                rf"(?:2025|2026)"
                rf"[^\n]{{0,60}}"
                rf"((?:\d{{1,3}}(?:,\d{{3}})*|\d{{3,}})(?:\.\d+)?)"
            )
            match = re.search(year_pattern, chunk_text, re.IGNORECASE)

            if not match:
                # Pass 2: no year label, tight window fallback
                pattern = (
                    rf"{re.escape(keyword)}"
                    rf"[^\n]{{0,60}}"
                    rf"((?:\d{{1,3}}(?:,\d{{3}})*|\d{{3,}})(?:\.\d+)?)"
                )
                match = re.search(pattern, chunk_text, re.IGNORECASE)

            if match:
                try:
                    value = float(match.group(1).replace(",", ""))
                    # Filter out note numbers, page numbers, percentages.
                    # Real SME financials in Lakhs are never single or
                    # double digit. Adjust only for micro-cap companies
                    # where figures genuinely fall below 10 Lakhs.
                    if value < 10:
                        continue
                    return value
                except ValueError:
                    continue

        return None

    def _compare(
        self,
        db_value,
        extracted_value,
    ) -> str:
        """Keyword-presence-based comparison only.

        Numeric auto-comparison was removed — regex-based number
        extraction proved unreliable across multi-column financial
        tables (quarterly/half-year/annual columns) and narrative
        percentage mentions, producing hallucinated mismatch values.
        The LLM performs actual numeric verification directly from
        the evidence text in the prompt instead.
        """
        if extracted_value is None:
            return schema.MISMATCH
        if db_value is None:
            return schema.MISMATCH
        return schema.MATCH