"""
api/main.py — FastAPI application for Credit Assessment Memo (CAM) generation.

Exposes REST endpoints for the full CAM pipeline:
    1. Upload company documents
    2. Run ingestion (parser_agent.run_parser_agent)
    3. Generate CAM report (CAMOrchestrator.generate_cam)
    4. Track ingestion status via ingestion_runs table
    5. Review unrecognised-format flags (other_findings JSON)
    6. Download generated .docx reports
"""

import json
import os
import sys
import shutil
from pathlib import Path
from uuid import uuid4
from typing import List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
import mysql.connector

# ---------------------------------------------------------------------------
# Ensure the project root is importable so we can reach parser_agent,
# cam_langgraphh, db, schema, config, etc. from inside api/.
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CAM Generation API",
    description="REST API for Credit Assessment Memo ingestion and generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory task tracker
# ---------------------------------------------------------------------------

# Maps run_id → {"status": str, "error": str | None}
_task_status: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------


def _get_db_connection():
    """Return a MySQL connection using env vars."""
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        database=os.environ.get("DB_NAME", "creditmemo"),
    )


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def _run_ingestion(company_id: str, run_id: str):
    """Background task: run the full ingestion pipeline.

    Calls parser_agent.run_parser_agent with exact signature:
        run_parser_agent(folder_path, company_name, fiscal_year, gemini_api_key)

    The folder_path points to uploads/{company_id} which should contain
    subfolders: annual_report/, audit_report/, saverisk/, exchange_filings/,
    bank_statements/.
    """
    from config import GEMINI_API_KEY

    _task_status[run_id] = {"status": "running", "error": None}
    try:
        from parser_agent import run_parser_agent

        folder_path = os.path.join(UPLOADS_DIR, company_id)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Upload folder not found: {folder_path}. "
                f"Upload files to /upload/{company_id} first."
            )

        # fiscal_year defaults to current year; callers can extend this later.
        fiscal_year = int(os.environ.get("DEFAULT_FISCAL_YEAR", 2025))

        run_parser_agent(
            folder_path=folder_path,
            company_name=company_id,
            fiscal_year=fiscal_year,
            gemini_api_key=GEMINI_API_KEY,
        )

        _task_status[run_id] = {"status": "completed", "error": None}
    except Exception as exc:
        _task_status[run_id] = {"status": "failed", "error": str(exc)}


def _run_cam_generation(company_id: str, run_id: str):
    """Background task: generate a CAM report.

    Calls CAMOrchestrator.generate_cam with exact signature:
        generate_cam(company_name, fiscal_year, folder_path)

    The orchestrator internally calls run_parser_agent to ensure
    ingestion is complete, then produces a .docx report.
    """
    _task_status[run_id] = {"status": "running", "error": None}
    try:
        from cam_langgraphh import CAMOrchestrator

        folder_path = os.path.join(UPLOADS_DIR, company_id)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Upload folder not found: {folder_path}. "
                f"Upload files to /upload/{company_id} first."
            )

        fiscal_year = int(os.environ.get("DEFAULT_FISCAL_YEAR", 2025))

        orchestrator = CAMOrchestrator()
        try:
            output_path = orchestrator.generate_cam(
                company_name=company_id,
                fiscal_year=fiscal_year,
                folder_path=folder_path,
            )

            # Move the generated file to outputs/{company_id}_{run_id}.docx
            final_path = os.path.join(OUTPUTS_DIR, f"{company_id}_{run_id}.docx")
            shutil.move(output_path, final_path)

            _task_status[run_id] = {"status": "completed", "error": None}
        finally:
            orchestrator.close()

    except Exception as exc:
        _task_status[run_id] = {"status": "failed", "error": str(exc)}


# ===================================================================
# ENDPOINTS
# ===================================================================


# 1. POST /upload/{company_id}
# -------------------------------------------------------------------


@app.post("/upload/{company_id}")
async def upload_files(company_id: str, files: List[UploadFile] = File(...)):
    """Accept multiple files and save them to uploads/{company_id}/.

    Files should be uploaded with their relative subfolder paths encoded
    in the filename (e.g. "annual_report/report.pdf"). If no subfolder
    prefix is present, files are saved directly under {company_id}/.
    """
    company_dir = os.path.join(UPLOADS_DIR, company_id)
    os.makedirs(company_dir, exist_ok=True)

    saved = []
    for f in files:
        # Preserve subfolder structure if encoded in filename
        dest = os.path.join(company_dir, f.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)

        saved.append(f.filename)

    return {"company_id": company_id, "uploaded_files": saved, "count": len(saved)}


# 2. POST /ingest/{company_id}
# -------------------------------------------------------------------


@app.post("/ingest/{company_id}")
async def ingest(company_id: str, background_tasks: BackgroundTasks):
    """Run the ingestion pipeline as a background task.

    Returns a run_id (uuid4) and initial status immediately.
    Poll GET /status/{run_id} for progress.
    """
    run_id = str(uuid4())
    _task_status[run_id] = {"status": "queued", "error": None}

    background_tasks.add_task(_run_ingestion, company_id, run_id)

    return {"run_id": run_id, "status": "queued"}


# 3. POST /generate-cam/{company_id}
# -------------------------------------------------------------------


@app.post("/generate-cam/{company_id}")
async def generate_cam(company_id: str, background_tasks: BackgroundTasks):
    """Run CAM generation as a background task.

    Output is saved to outputs/{company_id}_{run_id}.docx.
    Returns a run_id and initial status immediately.
    """
    run_id = str(uuid4())
    _task_status[run_id] = {"status": "queued", "error": None}

    background_tasks.add_task(_run_cam_generation, company_id, run_id)

    return {"run_id": run_id, "status": "queued"}


# 4. GET /status/{run_id}
# -------------------------------------------------------------------


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """Query ingestion status.

    First checks the in-memory task tracker. If not found there,
    falls back to the ingestion_runs table which tracks per-step
    completion flags.

    ingestion_runs columns (from db.py / schema.py):
        id, company_name,
        annual_report_parsed, audit_report_parsed,
        screener_parsed, exchange_filings_parsed,
        bank_statements_parsed,
        created_at, updated_at
    """
    # Check in-memory status first (covers both ingestion and CAM tasks)
    if run_id in _task_status:
        return {
            "run_id": run_id,
            **_task_status[run_id],
        }

    # Fall back: try to interpret run_id as a company_name lookup
    # in ingestion_runs for backward compatibility
    conn = _get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, company_name,
                   annual_report_parsed,
                   audit_report_parsed,
                   screener_parsed,
                   exchange_filings_parsed,
                   bank_statements_parsed,
                   created_at,
                   updated_at
            FROM ingestion_runs
            WHERE company_name = %s
            LIMIT 1
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row:
            return row

        raise HTTPException(
            status_code=404,
            detail=f"No task or ingestion run found for run_id '{run_id}'.",
        )
    finally:
        conn.close()


# 5. GET /flags/{company_id}
# -------------------------------------------------------------------


@app.get("/flags/{company_id}")
async def get_flags(company_id: str):
    """Return unrecognised-format flags for human-in-the-loop review.

    The pipeline stores unmapped / unrecognised data in the
    `other_findings` JSON column on both the `financials` and
    `bank_statements` tables. This endpoint collects them all.

    company_id here is treated as company_name. We resolve it to
    companies.id to query the related tables.
    """
    conn = _get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)

        # Resolve company_name → companies.id
        cursor.execute(
            "SELECT id FROM companies WHERE company_name = %s LIMIT 1",
            (company_id,),
        )
        company_row = cursor.fetchone()
        if not company_row:
            cursor.close()
            raise HTTPException(
                status_code=404,
                detail=f"Company '{company_id}' not found in companies table.",
            )

        cid = company_row["id"]

        # --- Financials other_findings ---
        cursor.execute(
            """
            SELECT fiscal_year, other_findings
            FROM financials
            WHERE company_id = %s
            """,
            (cid,),
        )
        financials_rows = cursor.fetchall()

        financials_flags = []
        for row in financials_rows:
            findings = row.get("other_findings")
            if findings:
                # other_findings is a JSON column; mysql.connector may
                # return it as a string or already-parsed dict.
                if isinstance(findings, str):
                    findings = json.loads(findings)
                if findings:  # skip empty dicts
                    financials_flags.append({
                        "fiscal_year": row["fiscal_year"],
                        "other_findings": findings,
                    })

        # --- Bank statements other_findings ---
        cursor.execute(
            """
            SELECT bank_name, account_number, source_filename, other_findings
            FROM bank_statements
            WHERE company_id = %s
            """,
            (cid,),
        )
        bank_rows = cursor.fetchall()

        bank_flags = []
        for row in bank_rows:
            findings = row.get("other_findings")
            if findings:
                if isinstance(findings, str):
                    findings = json.loads(findings)
                if findings:
                    bank_flags.append({
                        "bank_name": row["bank_name"],
                        "account_number": row["account_number"],
                        "source_filename": row.get("source_filename"),
                        "other_findings": findings,
                    })

        cursor.close()

        return {
            "company_id": company_id,
            "financials_flags": financials_flags,
            "bank_statement_flags": bank_flags,
            "total_flag_count": len(financials_flags) + len(bank_flags),
        }

    finally:
        conn.close()


# 6. GET /download/{company_id}/{run_id}
# -------------------------------------------------------------------


@app.get("/download/{company_id}/{run_id}")
async def download_report(company_id: str, run_id: str):
    """Return the generated .docx CAM report as a file download."""
    filename = f"{company_id}_{run_id}.docx"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    if not os.path.isfile(filepath):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Report not found: {filename}. "
                f"Ensure CAM generation completed for company '{company_id}' "
                f"with run_id '{run_id}'."
            ),
        )

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
