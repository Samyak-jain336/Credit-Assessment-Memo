"""
data_access.py

Centralized database access layer for the CAM generation pipeline.

This module is responsible ONLY for reading data from MySQL.
No business logic or LLM calls should live here.

All CAM agents should fetch structured information through this
module rather than issuing SQL directly.
"""

from typing import Any, Dict, List, Optional

from db import get_connection
import schema


class DataAccess:
    """Helper class for reading structured information from MySQL."""

    def __init__(self):
        self.conn = get_connection()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    # ---------------------------------------------------------
    # Company
    # ---------------------------------------------------------

    def get_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor(dictionary=True)

        cursor.execute(
            f"""
            SELECT *
            FROM {schema.COMPANIES_TABLE}
            WHERE company_name=%s
            LIMIT 1
            """,
            (company_name,),
        )

        row = cursor.fetchone()
        cursor.close()

        return row

    # ---------------------------------------------------------
    # Financials
    # ---------------------------------------------------------

    def get_financials(
        self,
        company_id: int,
        fiscal_year: int,
    ) -> Optional[Dict[str, Any]]:

        cursor = self.conn.cursor(dictionary=True)

        cursor.execute(
            f"""
            SELECT *
            FROM {schema.FINANCIALS_TABLE}
            WHERE company_id=%s
            AND fiscal_year=%s
            LIMIT 1
            """,
            (
                company_id,
                fiscal_year,
            ),
        )

        row = cursor.fetchone()
        cursor.close()

        return row

    # ---------------------------------------------------------
    # Market Data
    # ---------------------------------------------------------

    def get_market_data(
        self,
        company_id: int,
    ) -> List[Dict[str, Any]]:

        cursor = self.conn.cursor(dictionary=True)

        cursor.execute(
            f"""
            SELECT *
            FROM {schema.MARKET_DATA_TABLE}
            WHERE company_id=%s
            ORDER BY as_of_date DESC
            """,
            (company_id,),
        )

        rows = cursor.fetchall()

        cursor.close()

        return rows

    # ---------------------------------------------------------
    # Bank Statements
    # ---------------------------------------------------------

    def get_bank_statements(
        self,
        company_id: int,
    ) -> List[Dict[str, Any]]:

        cursor = self.conn.cursor(dictionary=True)

        cursor.execute(
            f"""
            SELECT *
            FROM {schema.BANK_STATEMENTS_TABLE}
            WHERE company_id=%s
            """,
            (company_id,),
        )

        rows = cursor.fetchall()

        cursor.close()

        return rows

    # ---------------------------------------------------------
    # Convenience Method
    # ---------------------------------------------------------

    def load_company_bundle(
        self,
        company_name: str,
        fiscal_year: int,
    ) -> Dict[str, Any]:
        """
        Returns every structured dataset required for CAM generation.
        """

        company = self.get_company(company_name)

        if not company:
            raise ValueError(f"Company '{company_name}' not found.")

        company_id = company["id"]

        return {
            "company": company,
            "financials": self.get_financials(
                company_id,
                fiscal_year,
            ),
            "market_data": self.get_market_data(
                company_id,
            ),
            "bank_statements": self.get_bank_statements(
                company_id,
            ),
        }