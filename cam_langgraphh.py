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

        for title, queries in schema.SECTION_QUERIES.items():
            chunks = []
            seen_ids = set()

            for q in queries:
                results = query_chunks(
                    self.collection,
                    query_text=q,
                    filters={"company_name": company_name},
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

        return self.generator.generate_all_sections(
            titles,
            section_chunks,
            reconciliation_results,
        )

    def write_document(
        self,
        company_name,
        fiscal_year,
        sections,
    ):
        """
        Write final CAM report as a .docx file.
        """

        print("Writing report...")

        output_path = f"{company_name.replace(' ', '_')}_CAM.docx"

        writer = DocumentWriter()
        writer.write(company_name, fiscal_year, sections, output_path)

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

        company_name="Durlax Top Surface Limited",

        fiscal_year=2025,

        folder_path=r"C:\Users\samya\OneDrive\Documents\GitHub\CreditAssessmentMemo\Inputs\Durlax Top Surface Limited",

        gemini_api_key=os.environ.get("GEMINI_API_KEY"),

    )