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

        return CAMSection(
            title=title,
            content=response,
            evidence=retrieved_chunks,
        )

    # ---------------------------------------------------------
    # Batch Generation
    # ---------------------------------------------------------

    def generate_all_sections(
        self,
        section_titles: List[str],
        section_chunks: dict,
        reconciliation_results: List[ReconciliationResult],
    ) -> List[CAMSection]:
        """
        Generate every requested CAM section.
        """

        sections = []

        for title in section_titles:

            chunks_for_section = section_chunks.get(title, [])

            section = self.generate_section(
                title,
                chunks_for_section,
                reconciliation_results,
            )

            sections.append(section)

        return sections