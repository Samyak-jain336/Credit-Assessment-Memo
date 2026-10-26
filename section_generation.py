"""
section_generation.py

Generates individual CAM sections using:

- Retrieved evidence
- Reconciliation results
- Shared prompt templates
- Provider-agnostic LLM via llm_utils.call_llm_primary
Each section is generated independently so the orchestration layer
(LangGraph) can run them sequentially or in parallel.
"""

import re
import json
from typing import List

from llm_utils import call_llm_primary

from prompts import (
    SYSTEM_PROMPT,
    build_section_prompt,
)

from stubs import (
    CAMSection,
    RetrievedChunk,
    ReconciliationResult,
)


class SectionGenerator:
    """
    Generates one CAM section using retrieved evidence.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def generate_section(
        self,
        title: str,
        retrieved_chunks: List[RetrievedChunk],
        reconciliation_results: List[ReconciliationResult],
        company_name: str = "",
        company_context: str = "",
    ) -> CAMSection:
        """
        Generate a single CAM section.

        For sections listed in schema.SECTION_TABLE_SCHEMA, the LLM
        response is expected to contain both a narrative block and a
        <TABLE>...</TABLE> JSON block. The TABLE block is extracted,
        parsed, and stored in CAMSection.table_data. It is stripped
        from the narrative so it does not appear twice in the docx.

        If the TABLE block is missing or malformed JSON, table_data
        is set to None and the section ships as prose only — the
        narrative is never blocked by a table parse failure.
        """

        user_prompt = build_section_prompt(
            title,
            retrieved_chunks,
            reconciliation_results,
            company_name=company_name,
            company_context=company_context,
        )

        full_prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + user_prompt
        )

        response = call_llm_primary(full_prompt)

        # Extract structured table block if present
        table_data = None
        match = re.search(r'<TABLE>(.*?)</TABLE>', response, re.DOTALL)
        if match:
            try:
                table_data = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                # Malformed JSON — ship prose only, never crash the section
                table_data = None
            # Strip TABLE block from narrative regardless of parse result
            response = response[:match.start()] + response[match.end():]

        return CAMSection(
            title=title,
            content=response.strip(),
            evidence=retrieved_chunks,
            table_data=table_data,
        )

    # ---------------------------------------------------------
    # Batch Generation
    # ---------------------------------------------------------

    def generate_all_sections(
        self,
        section_titles: List[str],
        retrieved_chunks: List[RetrievedChunk],
        reconciliation_results: List[ReconciliationResult],
        company_name: str = "",
        company_context: str = "",
    ) -> List[CAMSection]:
        """
        Generate every requested CAM section.

        All reconciliation results are passed to every section.
        Section-level relevance filtering is handled downstream
        in prompts.py via NON_FINANCIAL_SECTIONS.
        """

        sections = []

        # If retrieved_chunks is already grouped by section, use it.
        # Otherwise every section receives the same evidence (current behaviour).
        if isinstance(retrieved_chunks, dict):
            section_chunks = retrieved_chunks
        else:
            section_chunks = {
                title: retrieved_chunks
                for title in section_titles
            }

        for title in section_titles:

            chunks_for_section = section_chunks.get(title, [])

            # Pass all reconciliation results to every section.
            # ReconciliationResult has no section_relevance field — the old
            # filter was silently returning [] for every section, meaning the
            # LLM generated all sections with zero reconciliation context.
            # Section-level filtering is already handled in prompts.py via
            # NON_FINANCIAL_SECTIONS, which instructs the LLM to ignore
            # reconciliation data where it is not relevant.
            relevant_reconciliation = reconciliation_results

            section = self.generate_section(
                title,
                chunks_for_section,
                relevant_reconciliation,
                company_name=company_name,
                company_context=company_context,
            )

            sections.append(section)

        return sections