from dataclasses import dataclass, field
from typing import Any

@dataclass
class ReconciliationResult:
    """
    Stores the result of one reconciliation check.
    """

    field_name: str
    database_value: Any
    extracted_value: Any
    status: str
    remarks: str = ""

    # Sections that should receive this reconciliation result.
    # Non-financial sections should not receive financial mismatch flags.
    section_relevance: list[str] = field(
        default_factory=lambda: [
            "Financial Analysis",
            "Data Consistency Review",
            "Recommendation",
        ]
    )