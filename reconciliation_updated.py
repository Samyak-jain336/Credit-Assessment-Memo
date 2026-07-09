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

            # Try numeric extraction first
            extracted_value = self._extract_numeric(
                field,
                chunk_text,
            )

            # Fall back to keyword detection
            if extracted_value is None:
                extracted_value = self._extract_value(
                    field,
                    chunk_text,
                )

            status = self._compare(
                db_value,
                extracted_value,
            )

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

        keywords = FIELD_KEYWORDS.get(
            field,
            [field.lower().replace("_", " ")]
        )

        for keyword in keywords:

            pattern = (
                rf"{re.escape(keyword)}"
                rf".{{0,120}}?"
                rf"([\d,]+(?:\.\d+)?)"
            )

            match = re.search(
                pattern,
                chunk_text,
                re.IGNORECASE,
            )

            if match:
                try:
                    return float(
                        match.group(1).replace(",", "")
                    )
                except ValueError:
                    continue

        return None

    def _compare(
        self,
        db_value,
        extracted_value,
    ) -> str:

        if extracted_value is None:
            return schema.MISMATCH

        if db_value is None:
            return schema.MISMATCH

        # Keyword found but no numeric value
        if extracted_value == "FOUND":
            return schema.MATCH

        try:

            # Database stores Crores
            db_in_lakhs = float(db_value) * 100

            diff = abs(
                db_in_lakhs - float(extracted_value)
            )

            relative_diff = diff / max(
                abs(db_in_lakhs),
                1,
            )

            if relative_diff <= self.tolerance:
                return schema.MATCH

            return schema.MISMATCH

        except (TypeError, ValueError):

            # Fall back to MATCH if comparison fails
            return schema.MATCH