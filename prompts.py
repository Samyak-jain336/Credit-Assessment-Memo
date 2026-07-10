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
    """Format reconciliation results for the LLM prompt.

    Includes an explicit unit-conversion note so the LLM understands
    that database values are in Crores and document evidence is in Lakhs.
    1 Crore = 100 Lakhs — the LLM must account for this before flagging
    a mismatch. A value of 7.50 Cr and 750 Lakhs are the same figure.
    """

    lines = [
        "NOTE: The 'Status' field below (MATCH/MISMATCH) is based only on "
        "whether the field's keyword was found in evidence — it is NOT a "
        "verified numeric comparison. You must independently verify the "
        "actual numbers using the Evidence section above. Do not repeat "
        "the Status field's MISMATCH label as fact without checking the "
        "real figures yourself.\n"
        "NOTE: Database values are stored in CRORES. "
        "Financial statement evidence is expressed in LAKHS. "
        "1 Crore = 100 Lakhs. A value is a genuine match if it reconciles "
        "after multiplying the database value by 100 — verify this using "
        "the actual evidence numbers, not the Status label.\n"
        "NOTE: cash_and_bank in the database refers to the Balance Sheet "
        "line item 'Cash and Bank Balances', NOT the closing cash balance on the "
        "Cash Flow Statement. These are legitimately different figures in Indian "
        "financial statements due to bank overdraft classification. Do not treat "
        "this difference as a mismatch — note the distinction instead.\n"
        "NOTE: If evidence contains figures from different reporting periods "
        "(e.g. quarterly, half-yearly, and full-year columns in the same table), "
        "NEVER compare a partial-period figure (quarter/half-year) against a "
        "full-year database value and call it a mismatch. Always identify which "
        "column corresponds to the full financial year before comparing.\n"
    ]

    for result in results:

        lines.append(
            f"""
Field : {result.field_name}

Database (Crores) : {result.database_value}

Evidence (Lakhs)  : {result.extracted_value}

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

    # Sections that should not reference financial reconciliation results —
    # they lack financial evidence and will falsely report mismatches.
    NON_FINANCIAL_SECTIONS = {
        "Applicant Overview",
        "Company Background",
        "Banking Conduct",
        "Tax and Statutory Compliance",
        "Collateral",
    }

    if section_title == "Recommendation":
        reconciliation_instruction = (
            "6. Base the recommendation on the complete financial picture "
            "drawn from the Financial Analysis and Risk Assessment sections. "
            "Do NOT refuse a recommendation due to data mismatches — "
            "the reconciliation flags are unit-conversion artefacts (Crores vs Lakhs), "
            "not actual data gaps. Provide a definitive credit recommendation."
        )
    elif section_title == "Data Consistency Review":
        reconciliation_instruction = (
            "6. For each field marked MISMATCH: before reporting it as a genuine "
            "inconsistency, first check whether it is a unit-scaling difference "
            "(database in Crores, evidence in Lakhs, multiply db value × 100) or "
            "a label mismatch (e.g. cash_and_bank on Balance Sheet vs closing cash "
            "on Cash Flow Statement — these are different line items, not a mismatch). "
            "Only report as a true mismatch if the values differ after unit conversion. "
            "For each field, state: the database value, the evidence value, whether "
            "they reconcile after unit conversion, and your conclusion."
        )
    elif section_title in NON_FINANCIAL_SECTIONS:
        reconciliation_instruction = (
            "6. This section covers non-financial topics. "
            "Do NOT reference reconciliation results or flag missing financial data here. "
            "Confine your response strictly to the evidence provided above."
        )
    else:
        reconciliation_instruction = (
            "6. Reference reconciliation results where directly relevant to this section. "
            "Remember database values are in Crores, evidence is in Lakhs (1 Cr = 100 Lakhs). "
            "Do not flag a unit-scaled match as a mismatch."
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

6. If evidence spans multiple financial years, always lead with the most
   recent year. Explicitly call out year-on-year trends where figures
   from multiple years are available.

7. Do NOT state that information is unavailable if the evidence above
   contains it. Read all evidence blocks carefully before concluding
   anything is missing. If a Board of Directors list, KMP details,
   loan schedule, or capital raising event appears in any evidence
   block, include it in the analysis.

8. For cash and bank balances, distinguish between freely available
   cash and restricted cash (e.g. margin money held as fixed deposit
   against letters of credit). If the evidence contains notes
   breaking down the composition, include that detail.

{reconciliation_instruction}
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