"""
reconciliation.py

Performs reconciliation between structured database values and
retrieved document evidence before CAM generation.

This module DOES NOT call the LLM.

Responsibilities
----------------
1. Compare DB values against extracted evidence.
2. Detect inconsistencies.
3. Produce structured reconciliation results.
"""

from typing import Any, Dict, List

import schema
from stubs import (
    RetrievedChunk,
    ReconciliationResult,
)


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
        """
        Reconcile all configured financial fields.

        Parameters
        ----------
        financials:
            Row returned from financials table.

        retrieved_chunks:
            Relevant chunks retrieved from ChromaDB.

        Returns
        -------
        List[ReconciliationResult]
        """

        results = []

        chunk_text = "\n".join(
            chunk.text.lower()
            for chunk in retrieved_chunks
        )

        for field in schema.RECONCILIATION_FIELDS:

            db_value = financials.get(field)

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
        """
        Extract a value for a field from retrieved evidence.

        Current implementation performs keyword detection only.

        Future versions can replace this with:
            - Regex
            - Table parser
            - LLM extraction
        """

        if field.lower() in chunk_text:
            return "FOUND"

        return None

    def _compare(
        self,
        db_value,
        extracted_value,
    ) -> str:
        """
        Compare DB value against extracted evidence.
        """

        if extracted_value is None:
            return schema.MISMATCH

        if db_value is None:
            return schema.MISMATCH

        return schema.MATCH