"""
cam_langgraph.py

Main orchestration layer for CAM generation.

This module coordinates the complete CAM generation pipeline:

    Database
        ↓
    Data Access
        ↓
    Vector Retrieval
        ↓
    Reconciliation
        ↓
    Section Generation
        ↓
    DOCX Writer

The design intentionally mirrors a future LangGraph workflow where each
method can be converted into an independent graph node.
"""

from dotenv import load_dotenv
import os
from typing import List

import schema

from data_access import DataAccess
from reconciliation import ReconciliationAgent
from section_generation import SectionGenerator
from vector_store import init_vector_store, query_chunks
from docx_writer import DocumentWriter
from stubs import RetrievedChunk
from parser_agent import run_parser_agent

load_dotenv()
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")



class CAMOrchestrator:
    """
    Coordinates the entire CAM generation workflow.
    """

    def __init__(self, gemini_api_key: str, collection=None):
        if collection is None:
            collection, _ = init_vector_store(api_key=gemini_api_key)
        self.collection = collection

        self.db = DataAccess()
        self.reconciliation = ReconciliationAgent()
        self.generator = SectionGenerator()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def generate_cam(
        self,
        company_name: str,
        fiscal_year: int,
        folder_path: str,
        gemini_api_key: str,
    ):
        """
        Execute the complete CAM generation pipeline.
        """

        print(f"Ensuring ingestion is complete for {company_name}...")
        run_parser_agent(
            folder_path=folder_path,
            company_name=company_name,
            fiscal_year=fiscal_year,
            gemini_api_key=gemini_api_key,
        )

        print(f"Generating CAM for {company_name} ({fiscal_year})")

        company_bundle = self.load_company_data(
            company_name,
            fiscal_year,
        )

        retrieved_chunks = self.retrieve_chunks(
            company_name,
            fiscal_year,
        )

        reconciliation_results = self.reconcile(
            company_bundle,
            retrieved_chunks,
        )

        sections = self.generate_sections(
            company_bundle,
            retrieved_chunks,
            reconciliation_results,
        )

        output = self.write_document(
            company_name,
            fiscal_year,
            sections,
            reconciliation_results,
        )

        return output

    # ---------------------------------------------------------
    # Pipeline Nodes
    # ---------------------------------------------------------

    def load_company_data(
        self,
        company_name,
        fiscal_year,
    ):
        """
        Load structured company data from MySQL.
        """

        return self.db.load_company_bundle(
            company_name,
            fiscal_year,
        )

    def build_company_context(
        self,
        company_bundle: dict,
    ) -> str:
        """Build a structured factual context block from MySQL data.

        Injected into every section prompt as the single source of
        truth for financial figures, unit, and company identity.
        Prevents the LLM from re-deriving these facts independently
        in each section and producing contradictions or unit errors.

        All values come from MySQL — no hardcoding. Unit warning
        appears only for companies whose financials table stores a
        non-standard unit (e.g. 'hundreds', 'thousands') so the LLM
        knows not to re-convert already-converted Crores figures,
        and knows how to convert raw chunk figures it encounters.
        """
        company = company_bundle.get("company") or {}
        financials = company_bundle.get("financials")
        bank_stmts = company_bundle.get("bank_statements") or []

        lines = [
            "=== VERIFIED COMPANY FACTS (pre-converted, from internal database) ===",
            f"Applicant: {company.get('company_name', 'N/A')}",
            f"CIN: {company.get('cin', 'N/A')}",
        ]

        if financials:
            unit = financials.get("unit") or "Crores"
            currency = financials.get("currency") or "INR"
            fiscal_year = financials.get("fiscal_year", "N/A")

            lines.append(f"Fiscal Year: {fiscal_year}")
            lines.append(f"All figures below are in: {currency} Crores")
            lines.append(f"(Original source unit in documents: {unit})")
            lines.append("")

            def fmt(val):
                return str(val) if val is not None else "N/A"

            lines.append(f"Revenue:           {fmt(financials.get('revenue'))} Crores")
            lines.append(f"Net Profit:        {fmt(financials.get('net_profit'))} Crores")
            lines.append(f"Borrowings:        {fmt(financials.get('borrowings'))} Crores")
            lines.append(f"Cash and Bank:     {fmt(financials.get('cash_and_bank'))} Crores")
            lines.append(f"CFO:               {fmt(financials.get('cfo'))} Crores")
            lines.append(f"Net Cash Flow:     {fmt(financials.get('net_cash_flow'))} Crores")
            lines.append(f"Trade Receivables: {fmt(financials.get('trade_receivables'))} Crores")
            lines.append(f"Inventory:         {fmt(financials.get('inventory'))} Crores")

            # Unit warning — only appears when source documents use a
            # non-standard unit. Derived from MySQL, no hardcoding.
            if unit.lower() not in ("crores", "lakhs"):
                # Compute conversion factor from unit string dynamically
                unit_lower = unit.lower().strip()
                if "hundred" in unit_lower:
                    factor_desc = "divide by 10,000,000"
                elif "thousand" in unit_lower:
                    factor_desc = "divide by 100,000"
                elif "million" in unit_lower:
                    factor_desc = "divide by 10"
                else:
                    factor_desc = f"convert from '{unit}' to Crores"

                lines.append("")
                lines.append(
                    f"UNIT WARNING: Source documents report in '{unit}'. "
                    f"The figures ABOVE are already converted to Crores by "
                    f"the ingestion pipeline — do NOT convert them again. "
                    f"When you encounter raw figures from evidence chunks "
                    f"(not from the list above), {factor_desc} to get Crores. "
                    f"Always state (Standalone) or (Consolidated) alongside "
                    f"any figure from evidence to distinguish the two sets."
                )

        if bank_stmts:
            lines.append("")
            lines.append(f"Bank statements on file: {len(bank_stmts)}")
            for bs in bank_stmts:
                period_from = bs.get("statement_period_from", "?")
                period_to = bs.get("statement_period_to", "?")
                lines.append(
                    f"  - {bs.get('bank_name', 'Unknown Bank')} | "
                    f"{bs.get('account_type', '?')} | "
                    f"A/C: {bs.get('account_number', '?')} | "
                    f"{period_from} to {period_to}"
                )

        lines.append("=== END VERIFIED FACTS ===")
        return "\n".join(lines)

    def retrieve_chunks(
        self,
        company_name,
        fiscal_year,
    ):
        """
        Retrieve relevant evidence from ChromaDB.

        Returns a dict of {section_title: List[RetrievedChunk]} using
        the per-section queries defined in schema.SECTION_QUERIES.
        """

        print("Retrieving supporting evidence per section...")
        section_chunks = {}

        # Sections where mixing standalone and consolidated chunks produces
        # contradictory or inflated figures. Restricted to standalone only.
        STANDALONE_ONLY_SECTIONS = {
            "Financial Analysis",
            "Risk Assessment",
            "Recommendation",
        }

        for title, queries in schema.SECTION_QUERIES.items():
            chunks = []
            seen_ids = set()

            # Apply statement_type filter for financial sections to prevent
            # consolidated group figures from contaminating standalone analysis.
            if title in STANDALONE_ONLY_SECTIONS:
                section_filter = {
                    "company_name": company_name,
                    "statement_type": "standalone",
                }
            else:
                section_filter = {"company_name": company_name}

            for q in queries:
                results = query_chunks(
                    self.collection,
                    query_text=q,
                    filters=section_filter,
                    n_results=2,
                )
                for r in results:
                    uid = (
                        r["metadata"]["page_number"],
                        r["metadata"]["chunk_index"],
                        r["metadata"]["document_type"],
                    )
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)
                    chunks.append(
                        RetrievedChunk(
                            text=r["text"],
                            metadata=r["metadata"],
                            distance=r["distance"],
                        )
                    )

            section_chunks[title] = chunks
            print(f"  {title}: {len(chunks)} chunks")

        return section_chunks

    def reconcile(
        self,
        company_bundle,
        retrieved_chunks,
    ):
        """
        Compare DB values with retrieved evidence.
        """

        financials = company_bundle["financials"]
        if financials is None:
            print("WARNING: No financials found — skipping reconciliation.")
            return []

        # retrieved_chunks is {section_title: List[RetrievedChunk]} —
        # reconciliation expects a flat list, so flatten all sections.
        # Deduplicate by (page_number, chunk_index, document_type) so
        # the same chunk appearing in multiple sections isn't double-counted.
        seen = set()
        flat_chunks = []
        for chunks in retrieved_chunks.values():
            for chunk in chunks:
                uid = (
                    chunk.metadata["page_number"],
                    chunk.metadata["chunk_index"],
                    chunk.metadata["document_type"],
                )
                if uid not in seen:
                    seen.add(uid)
                    flat_chunks.append(chunk)

        return self.reconciliation.reconcile(
            financials,
            flat_chunks,
        )

    def generate_sections(
        self,
        company_bundle,
        section_chunks,
        reconciliation_results,
    ):
        """
        Generate all CAM sections.
        """

        titles = list(schema.CAM_SECTIONS.values())

        # Build the grounded DB context block and inject into every section.
        # Without this the LLM has no authoritative figures and reads raw
        # chunk numbers (which may be in Lakhs or from a subsidiary) as facts.
        company_context = self.build_company_context(company_bundle)

        return self.generator.generate_all_sections(
            titles,
            section_chunks,
            reconciliation_results,
            company_name=company_bundle["company"]["company_name"],
            company_context=company_context,
        )

    def write_document(
        self,
        company_name,
        fiscal_year,
        sections,
        reconciliation_results=None,
    ):
        """
        Write final CAM report as a .docx file.
        """

        print("Writing report...")

        output_path = f"{company_name.replace(' ', '_')}_CAM.docx"

        writer = DocumentWriter()
        writer.write(
            company_name,
            fiscal_year,
            sections,
            output_path,
            reconciliation_results,
        )

        return output_path

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self):
        self.db.close()


if __name__ == "__main__":

    orchestrator = CAMOrchestrator(
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    )

    orchestrator.generate_cam(

        company_name="FSN E-Commerce",

        fiscal_year=2025,

        folder_path=r"Inputs\FSN E-Commerce",

        gemini_api_key=os.environ.get("GEMINI_API_KEY"),

    )