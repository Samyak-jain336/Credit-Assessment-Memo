"""
stubs.py

Shared data structures used across the CAM generation pipeline.

These dataclasses act as the common contract between:
    • Data Access
    • Reconciliation
    • RAG Retrieval
    • Section Generation
    • LangGraph

No business logic should live here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# Retrieved RAG Chunk
# ============================================================

@dataclass
class RetrievedChunk:
    """Single chunk returned from ChromaDB."""

    text: str

    metadata: Dict[str, Any]

    distance: float


# ============================================================
# Reconciliation Result
# ============================================================

@dataclass
class ReconciliationResult:
    """Stores the result of one reconciliation check."""

    field_name: str

    database_value: Any

    extracted_value: Any

    status: str

    remarks: str = ""

    section_relevance: list[str] = field(
        default_factory=lambda: [
            "Financial Analysis",
            "Data Consistency Review",
            "Recommendation",
        ]
    )


# ============================================================
# CAM Section
# ============================================================

@dataclass
class CAMSection:
    """Represents one generated CAM section."""

    title: str

    content: str

    evidence: List[RetrievedChunk] = field(default_factory=list)

    table_data: Optional[List[Dict[str, Any]]] = None


# ============================================================
# Company Bundle
# ============================================================

@dataclass
class CompanyBundle:
    """
    Structured information fetched from MySQL.

    Populated by DataAccess.load_company_bundle().
    """

    company: Dict[str, Any]

    financials: Dict[str, Any]

    market_data: List[Dict[str, Any]]

    bank_statements: List[Dict[str, Any]]


