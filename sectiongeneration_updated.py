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