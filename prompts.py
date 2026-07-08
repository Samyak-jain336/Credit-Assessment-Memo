"""
prompts.py

Prompt templates for CAM generation.

Centralising prompts here keeps the rest of the pipeline free of
large prompt strings and makes prompt updates easy.
"""

from typing import List
from stubs import RetrievedChunk, ReconciliationResult


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def format_chunks(chunks: List[RetrievedChunk]) -> str:
    """Convert retrieved chunks into readable context."""

    output = []

    for i, chunk in enumerate(chunks, start=1):

        output.append(
            f"""
Evidence {i}

{chunk.text}
"""
        )

    return "\n".join(output)


def format_reconciliation(
    results: List[ReconciliationResult],
) -> str:
    """Convert reconciliation results into text."""

    lines = []

    for result in results:

        lines.append(

            f"""
Field : {result.field_name}

Database : {result.database_value}

Evidence : {result.extracted_value}

Status : {result.status}

Remarks : {result.remarks}
"""
        )

    return "\n".join(lines)


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an experienced credit analyst preparing a Credit Assessment Memo.

Use only the supplied evidence.

Never invent financial numbers.

If evidence is insufficient, explicitly state that.

Write in a professional banking tone.
"""


# ---------------------------------------------------------
# Section Prompt
# ---------------------------------------------------------

def build_section_prompt(
    section_title: str,
    retrieved_chunks: List[RetrievedChunk],
    reconciliation: List[ReconciliationResult],
) -> str:
    """
    Build the prompt used to generate one CAM section.
    """

    evidence = format_chunks(retrieved_chunks)

    reconciliation_text = format_reconciliation(
        reconciliation
    )

    return f"""
Generate the CAM section:

Section:
{section_title}

Evidence:
{evidence}

Reconciliation Results:
{reconciliation_text}

Instructions:

1. Use only supplied evidence.

2. Mention inconsistencies where relevant.

3. Keep the response concise.

4. Do not fabricate values.

5. Produce polished business English.
"""


# ---------------------------------------------------------
# Final Review Prompt
# ---------------------------------------------------------

FINAL_REVIEW_PROMPT = """
Review the complete Credit Assessment Memo.

Check:

• Internal consistency

• Grammar

• Financial terminology

• Duplicate statements

• Missing information

Do not invent facts.

Only improve clarity and presentation.
"""