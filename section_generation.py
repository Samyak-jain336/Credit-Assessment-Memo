"""
section_generation.py

Generates individual CAM sections using:

- Retrieved evidence
- Reconciliation results
- Shared prompt templates
- Groq LLM

Each section is generated independently so the orchestration layer
(LangGraph) can run them sequentially or in parallel.
"""

import re
import json
from typing import List

from llm_utils import call_gemini

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
        )

        full_prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + user_prompt
        )

        response = call_gemini(full_prompt)

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
    ) -> List[CAMSection]:
        """
        Generate every requested CAM section.

        Only reconciliation results relevant to the current section
        are passed to the LLM.
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

            # Only send reconciliation results relevant to this section.
            relevant_reconciliation = [
                result
                for result in reconciliation_results
                if title in result.section_relevance
            ]

            section = self.generate_section(
                title,
                chunks_for_section,
                relevant_reconciliation,
            )

            sections.append(section)

        return sections