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

from typing import List

import schema

from data_access import DataAccess
from reconciliation import ReconciliationAgent
from section_generation import SectionGenerator

# Future imports
# from retrieval import RetrievalAgent
# from docx_writer import DocumentWriter


class CAMOrchestrator:
    """
    Coordinates the entire CAM generation workflow.
    """

    def __init__(self, collection=None):
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
    ):
        """
        Execute the complete CAM generation pipeline.
        """

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

        Placeholder until retrieval module is implemented.
        """

        print("Retrieving supporting evidence...")

        # TODO:
        # query_chunks(...)
        return []

    def reconcile(
        self,
        company_bundle,
        retrieved_chunks,
    ):
        """
        Compare DB values with retrieved evidence.
        """

        return self.reconciliation.reconcile(
            company_bundle["financials"],
            retrieved_chunks,
        )

    def generate_sections(
        self,
        company_bundle,
        retrieved_chunks,
        reconciliation_results,
    ):
        """
        Generate all CAM sections.
        """

        titles = list(schema.CAM_SECTIONS.values())

        return self.generator.generate_all_sections(
            titles,
            retrieved_chunks,
            reconciliation_results,
        )

    def write_document(
        self,
        company_name,
        sections,
    ):
        """
        Write final CAM report.

        Placeholder until docx_writer.py is completed.
        """

        print("Writing report...")

        # Future:
        #
        # writer = DocumentWriter()
        #
        # return writer.write(...)
        #

        return f"{company_name}_CAM.docx"

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self):
        self.db.close()


if __name__ == "__main__":

    orchestrator = CAMOrchestrator()

    orchestrator.generate_cam(

        company_name="Durlax Top Surface Limited",

        fiscal_year=2025,

    )