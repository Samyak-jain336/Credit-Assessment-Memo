"""LangGraph workflow for CAM generation."""

import json
import operator
import os
import re
import time
from datetime import datetime
from functools import wraps
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

try:
    from langgraph.types import Send
except ImportError:
    from langgraph.constants import Send

import schema
from config import (
    GEMINI_API_KEY,
    CAM_MAX_PARALLEL_SECTIONS,
    CAM_MAX_REVIEW_RETRIES,
)
from data_access import DataAccess
from docx_writer import DocumentWriter
from parser_agent import run_parser_agent
from reconciliation import ReconciliationAgent
from section_generation import SectionGenerator
from stubs import CAMSection, ReconciliationResult, RetrievedChunk
from vector_store import init_vector_store, query_chunks

load_dotenv()
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")

# Sections restricted to standalone-only chunks
STANDALONE_ONLY_SECTIONS = {
    "Financial Analysis",
    "Risk Assessment",
    "Recommendation",
}

# Rendered by docx_writer directly from reconciliation results
CODE_RENDERED_SECTIONS = {"Data Consistency Review"}

MIN_SECTION_CHARS = 300
MIN_CHUNKS_FOR_MISSING_CHECK = 3

CURRENCY_PATTERN = re.compile(r"(\$\s?\d|\bUSD\s?\d|\bUS\$)")

MISSING_PHRASES = (
    "not provided",
    "not been provided",
    "not available in the evidence",
    "insufficient evidence",
    "no evidence",
    "not included in the",
)


def merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer: merge dict updates from parallel branches."""
    return {**(left or {}), **(right or {})}


class CAMState(TypedDict, total=False):
    company_name: str
    fiscal_year: int
    folder_path: str

    company_bundle: Optional[Dict[str, Any]]
    company_context: str
    section_chunks: Dict[str, List[RetrievedChunk]]
    reconciliation_results: List[ReconciliationResult]

    sections: Annotated[Dict[str, CAMSection], merge_dicts]

    pending_titles: List[str]
    review_feedback: Dict[str, List[str]]
    review_attempts: int
    unresolved_issues: Dict[str, List[str]]

    output_path: Optional[str]
    report_path: Optional[str]
    node_timings: Annotated[Dict[str, float], merge_dicts]
    errors: Annotated[List[str], operator.add]


def timed(name: str):
    """Record a node's wall-clock duration in node_timings."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            result = dict(fn(state) or {})
            timings = dict(result.get("node_timings", {}))
            timings[name] = round(time.perf_counter() - start, 2)
            result["node_timings"] = timings
            return result
        return wrapper
    return decorator


def build_company_context(company_bundle: dict) -> str:
    """Build a structured factual context block from MySQL data."""
    company = company_bundle.get("company") or {}
    financials = company_bundle.get("financials")
    bank_stmts = company_bundle.get("bank_statements") or []

    lines = [
        "=== VERIFIED COMPANY FACTS (pre-converted, from internal database) ===",
        f"Applicant: {company.get('company_name', 'N/A')}",
        f"CIN: {company.get('cin', 'N/A')}",
    ]

    if financials:
        unit = financials.get("unit") or "Crores"
        currency = financials.get("currency") or "INR"
        fiscal_year = financials.get("fiscal_year", "N/A")

        lines.append(f"Fiscal Year: {fiscal_year}")
        lines.append(f"All figures below are in: {currency} Crores")
        lines.append(f"(Original source unit in documents: {unit})")
        lines.append("")

        def fmt(val):
            return str(val) if val is not None else "N/A"

        lines.append(f"Revenue:           {fmt(financials.get('revenue'))} Crores")
        lines.append(f"Net Profit:        {fmt(financials.get('net_profit'))} Crores")
        lines.append(f"Borrowings:        {fmt(financials.get('borrowings'))} Crores")
        lines.append(f"Cash and Bank:     {fmt(financials.get('cash_and_bank'))} Crores")
        lines.append(f"CFO:               {fmt(financials.get('cfo'))} Crores")
        lines.append(f"Net Cash Flow:     {fmt(financials.get('net_cash_flow'))} Crores")
        lines.append(f"Trade Receivables: {fmt(financials.get('trade_receivables'))} Crores")
        lines.append(f"Inventory:         {fmt(financials.get('inventory'))} Crores")

        if unit.lower() not in ("crores", "lakhs"):
            unit_lower = unit.lower().strip()
            if "hundred" in unit_lower:
                factor_desc = "divide by 10,000,000"
            elif "thousand" in unit_lower:
                factor_desc = "divide by 100,000"
            elif "million" in unit_lower:
                factor_desc = "divide by 10"
            else:
                factor_desc = f"convert from '{unit}' to Crores"

            lines.append("")
            lines.append(
                f"UNIT WARNING: Source documents report in '{unit}'. "
                f"The figures ABOVE are already converted to Crores by "
                f"the ingestion pipeline — do NOT convert them again. "
                f"When you encounter raw figures from evidence chunks "
                f"(not from the list above), {factor_desc} to get Crores. "
                f"Always state (Standalone) or (Consolidated) alongside "
                f"any figure from evidence to distinguish the two sets."
            )

    if bank_stmts:
        lines.append("")
        lines.append(f"Bank statements on file: {len(bank_stmts)}")
        for bs in bank_stmts:
            period_from = bs.get("statement_period_from", "?")
            period_to = bs.get("statement_period_to", "?")
            lines.append(
                f"  - {bs.get('bank_name', 'Unknown Bank')} | "
                f"{bs.get('account_type', '?')} | "
                f"A/C: {bs.get('account_number', '?')} | "
                f"{period_from} to {period_to}"
            )

    lines.append("=== END VERIFIED FACTS ===")
    return "\n".join(lines)


def _chunk_uid(metadata: dict) -> tuple:
    """Unique key for a chunk, used for deduplication."""
    return (
        metadata["page_number"],
        metadata["chunk_index"],
        metadata["document_type"],
    )


def review_section(
    section: CAMSection,
    chunk_count: int,
    mismatch_fields: List[str],
) -> List[str]:
    """Deterministic quality checks for one generated section."""
    issues = []
    content = (section.content or "").strip()
    lowered = content.lower()

    if len(content) < MIN_SECTION_CHARS:
        issues.append(
            f"The section is empty or too short ({len(content)} characters). "
            f"Write a complete analysis using the supplied evidence."
        )
        return issues

    if CURRENCY_PATTERN.search(content):
        issues.append(
            "The draft expresses figures in US dollars. Express every figure "
            "in INR using Crores or Lakhs, exactly as given in the verified "
            "facts and evidence."
        )

    if chunk_count >= MIN_CHUNKS_FOR_MISSING_CHECK:
        hits = [p for p in MISSING_PHRASES if p in lowered]
        if hits:
            issues.append(
                f"The draft states information is missing ({', '.join(hits)}) "
                f"although {chunk_count} evidence blocks were supplied. Re-read "
                f"every evidence block and use what is present. Only state that "
                f"information is unavailable if no evidence block contains it."
            )

    if section.title in schema.SECTION_TABLE_SCHEMA and not section.table_data:
        issues.append(
            "The structured table is missing or is not valid JSON. After the "
            "narrative, append a valid JSON array inside <TABLE></TABLE> tags "
            "using exactly the required keys."
        )

    if (
        section.title == "Recommendation"
        and mismatch_fields
        and "conditional" not in lowered
    ):
        issues.append(
            f"These fields could not be verified: {', '.join(mismatch_fields)}. "
            f"The recommendation must not be unconditional. For each field, "
            f"state: 'Conditional approval recommended, subject to independent "
            f"verification of [field_name].'"
        )

    return issues


def _context_with_feedback(base_context: str, feedback: List[str]) -> str:
    """Append reviewer feedback to the verified-facts block for a retry."""
    if not feedback:
        return base_context
    points = "\n".join(f"- {item}" for item in feedback)
    return (
        f"{base_context}\n\n"
        f"=== REVIEWER FEEDBACK ON YOUR PREVIOUS DRAFT OF THIS SECTION ===\n"
        f"Fix every point below in this new draft:\n{points}\n"
        f"=== END REVIEWER FEEDBACK ==="
    )


def build_graph(collection=None):
    """Build and compile the CAM generation graph."""
    if collection is None:
        collection, _ = init_vector_store(api_key=GEMINI_API_KEY)

    reconciliation_agent = ReconciliationAgent()
    generator = SectionGenerator()
    section_titles = list(schema.CAM_SECTIONS.values())

    @timed("ingest")
    def ingest_node(state: CAMState) -> dict:
        """Run Track A ingestion."""
        company_name = state["company_name"]
        print(f"Ensuring ingestion is complete for {company_name}...")

        try:
            run_parser_agent(
                folder_path=state["folder_path"],
                company_name=company_name,
                fiscal_year=state["fiscal_year"],
                gemini_api_key=GEMINI_API_KEY,
            )
        except Exception as e:
            return {"errors": [f"ingest: {e}"]}

        return {"errors": []}

    @timed("load_data")
    def load_data_node(state: CAMState) -> dict:
        """Load company bundle from MySQL and build verified-facts context."""
        company_name = state["company_name"]
        fiscal_year = state["fiscal_year"]
        print(f"Generating CAM for {company_name} ({fiscal_year})")

        db = DataAccess()
        try:
            bundle = db.load_company_bundle(company_name, fiscal_year)
        except ValueError as e:
            return {"company_bundle": None, "errors": [f"load_data: {e}"]}
        finally:
            db.close()

        return {
            "company_bundle": bundle,
            "company_context": build_company_context(bundle),
        }

    def route_after_load(state: CAMState) -> str:
        """Stop the graph if the company was not found."""
        if state.get("company_bundle"):
            return "retrieve"
        print("ERROR: Company data not found — stopping CAM generation.")
        return END

    @timed("retrieve")
    def retrieve_node(state: CAMState) -> dict:
        """Retrieve evidence from ChromaDB per section."""
        company_name = state["company_name"]
        print("Retrieving supporting evidence per section...")
        section_chunks = {}

        for title, queries in schema.SECTION_QUERIES.items():
            chunks = []
            seen_ids = set()

            if title in STANDALONE_ONLY_SECTIONS:
                section_filter = {
                    "$and": [
                        {"company_name": {"$eq": company_name}},
                        {"statement_type": {"$eq": "standalone"}},
                    ]
                }
            else:
                section_filter = {"company_name": {"$eq": company_name}}

            for q in queries:
                results = query_chunks(
                    collection,
                    query_text=q,
                    filters=section_filter,
                    n_results=2,
                )
                for r in results:
                    uid = _chunk_uid(r["metadata"])
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

        return {"section_chunks": section_chunks}

    @timed("reconcile")
    def reconcile_node(state: CAMState) -> dict:
        """Keyword-based reconciliation and review loop init."""
        loop_init = {
            "pending_titles": section_titles,
            "review_feedback": {},
            "review_attempts": 0,
        }

        financials = state["company_bundle"].get("financials")
        if financials is None:
            print("WARNING: No financials found — skipping reconciliation.")
            return {
                **loop_init,
                "reconciliation_results": [],
                "errors": ["reconcile: no financials found"],
            }

        seen = set()
        flat_chunks = []
        for chunks in state["section_chunks"].values():
            for chunk in chunks:
                uid = _chunk_uid(chunk.metadata)
                if uid not in seen:
                    seen.add(uid)
                    flat_chunks.append(chunk)

        results = reconciliation_agent.reconcile(financials, flat_chunks)
        return {**loop_init, "reconciliation_results": results}

    def dispatch_sections(state: CAMState) -> List[Send]:
        """Create one parallel task per pending section."""
        feedback = state.get("review_feedback") or {}
        company_name = state["company_bundle"]["company"]["company_name"]

        return [
            Send(
                "generate_section",
                {
                    "title": title,
                    "chunks": state["section_chunks"].get(title, []),
                    "reconciliation_results": state["reconciliation_results"],
                    "company_name": company_name,
                    "company_context": _context_with_feedback(
                        state["company_context"],
                        feedback.get(title, []),
                    ),
                },
            )
            for title in state["pending_titles"]
        ]

    def generate_section_node(payload: dict) -> dict:
        """Generate a single CAM section."""
        title = payload["title"]
        start = time.perf_counter()

        try:
            section = generator.generate_section(
                title,
                payload["chunks"],
                payload["reconciliation_results"],
                company_name=payload["company_name"],
                company_context=payload["company_context"],
            )
            errors = []
        except Exception as e:
            print(f"  ERROR generating '{title}': {e}")
            section = CAMSection(title=title, content="", evidence=payload["chunks"])
            errors = [f"generate_section[{title}]: {e}"]

        elapsed = round(time.perf_counter() - start, 2)
        print(f"  Generated: {title} ({elapsed}s)")

        return {
            "sections": {title: section},
            "node_timings": {f"section:{title}": elapsed},
            "errors": errors,
        }

    @timed("review")
    def review_node(state: CAMState) -> dict:
        """Deterministic review of generated sections."""
        attempts = state.get("review_attempts", 0)
        mismatch_fields = [
            r.field_name
            for r in state.get("reconciliation_results", [])
            if r.status == schema.MISMATCH
        ]

        failed = {}
        for title in state["pending_titles"]:
            if title in CODE_RENDERED_SECTIONS:
                continue
            section = state["sections"].get(title)
            if section is None:
                failed[title] = ["Section was not generated."]
                continue
            issues = review_section(
                section,
                len(state["section_chunks"].get(title, [])),
                mismatch_fields,
            )
            if issues:
                failed[title] = issues

        print(f"Review round {attempts + 1}: "
              f"{len(state['pending_titles']) - len(failed)} passed, "
              f"{len(failed)} failed")
        for title, issues in failed.items():
            print(f"  {title}: {len(issues)} issue(s)")

        if failed and attempts < CAM_MAX_REVIEW_RETRIES:
            return {
                "pending_titles": list(failed.keys()),
                "review_feedback": failed,
                "review_attempts": attempts + 1,
            }

        return {
            "pending_titles": [],
            "review_feedback": {},
            "unresolved_issues": failed,
        }

    def route_after_review(state: CAMState):
        """Retry failed sections, or move on to writing."""
        if state.get("pending_titles"):
            return dispatch_sections(state)
        return "write"

    @timed("write")
    def write_node(state: CAMState) -> dict:
        """Write the final CAM .docx and a JSON run report."""
        print("Writing report...")
        company_name = state["company_name"]
        output_path = f"{company_name.replace(' ', '_')}_CAM.docx"
        report_path = output_path.replace(".docx", "_run_report.json")

        ordered_sections = [
            state["sections"][title]
            for title in section_titles
            if title in state["sections"]
        ]

        DocumentWriter().write(
            company_name,
            state["fiscal_year"],
            ordered_sections,
            output_path,
            state["reconciliation_results"],
        )

        recon = state.get("reconciliation_results", [])
        report = {
            "company_name": company_name,
            "fiscal_year": state["fiscal_year"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_docx": output_path,
            "chunks_per_section": {
                title: len(chunks)
                for title, chunks in state["section_chunks"].items()
            },
            "reconciliation": {
                "match": sum(1 for r in recon if r.status == schema.MATCH),
                "mismatch": sum(1 for r in recon if r.status == schema.MISMATCH),
                "mismatch_fields": [
                    r.field_name for r in recon if r.status == schema.MISMATCH
                ],
            },
            "review_retries_used": state.get("review_attempts", 0),
            "unresolved_review_issues": state.get("unresolved_issues", {}),
            "node_timings_seconds": state.get("node_timings", {}),
            "errors": state.get("errors", []),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        if report["unresolved_review_issues"]:
            print("WARNING: Some sections still have review issues — "
                  f"see {report_path}")

        return {"output_path": output_path, "report_path": report_path}

    # Wiring
    graph = StateGraph(CAMState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("load_data", load_data_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("generate_section", generate_section_node)
    graph.add_node("review", review_node)
    graph.add_node("write", write_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "load_data")
    graph.add_conditional_edges(
        "load_data",
        route_after_load,
        {"retrieve": "retrieve", END: END},
    )
    graph.add_edge("retrieve", "reconcile")
    graph.add_conditional_edges("reconcile", dispatch_sections, ["generate_section"])
    graph.add_edge("generate_section", "review")
    graph.add_conditional_edges("review", route_after_review, ["generate_section", "write"])
    graph.add_edge("write", END)

    return graph.compile()


def _initial_state(company_name: str, fiscal_year: int, folder_path: str) -> dict:
    return {
        "company_name": company_name,
        "fiscal_year": fiscal_year,
        "folder_path": folder_path,
        "sections": {},
        "node_timings": {},
        "errors": [],
    }


def _run_config() -> dict:
    return {"max_concurrency": CAM_MAX_PARALLEL_SECTIONS}


def run_cam(
    company_name: str,
    fiscal_year: int,
    folder_path: str,
    collection=None,
) -> Optional[str]:
    """Run the full CAM graph. Returns the .docx path or None."""
    app = build_graph(collection)
    final_state = app.invoke(
        _initial_state(company_name, fiscal_year, folder_path),
        config=_run_config(),
    )

    if final_state.get("errors"):
        print("\nIssues during run:")
        for err in final_state["errors"]:
            print(f"  - {err}")

    return final_state.get("output_path")


def export_graph_diagram(path: str = "cam_graph.mmd") -> str:
    """Write the graph as Mermaid code."""
    mermaid = build_graph().get_graph().draw_mermaid()
    with open(path, "w", encoding="utf-8") as f:
        f.write(mermaid)
    return path


class CAMOrchestrator:
    """Backward-compatible wrapper for FastAPI callers."""

    def __init__(self, collection=None):
        self.app = build_graph(collection)

    def generate_cam(
        self,
        company_name: str,
        fiscal_year: int,
        folder_path: str,
    ):
        final_state = self.app.invoke(
            _initial_state(company_name, fiscal_year, folder_path),
            config=_run_config(),
        )
        return final_state.get("output_path")

    def close(self):
        pass


if __name__ == "__main__":

    run_cam(
        company_name="C.E. Info Systems Ltd",
        fiscal_year=2025,
        folder_path=r"Inputs\C.E. Info Systems Ltd",
    )