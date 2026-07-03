"""
saverisk.py — Ingestion pipeline for Screener "Data Sheet" Excel exports.

Reads the fixed-layout "Data Sheet" tab, maps known rows to the financials
schema, asks an LLM to resolve any unmapped rows, detects currency/unit,
and writes everything into the MySQL database (companies, financials,
market_data tables).

Single entry point: run_saverisk_ingestion(filepath, company_name)
"""

import json
from datetime import date

import pandas as pd

from db import get_connection
from llm_utils import call_llm_json


# ---------------------------------------------------------------------------
# Row-label → schema-field mappings for each section of the Data Sheet.
# Labels must match the Excel cell text exactly (case-sensitive).
# ---------------------------------------------------------------------------

PNL_ROW_MAP = {
    "Sales":                "revenue",
    "Raw Material Cost":    "raw_material_cost",
    "Change in Inventory":  "change_in_inventory",
    "Power and Fuel":       "power_and_fuel",
    "Other Mfr. Exp":       "other_mfr_exp",
    "Employee Cost":        "employee_cost",
    "Selling and admin":    "selling_and_admin",
    "Other Expenses":       "other_expenses",
    "Other Income":         "other_income",
    "Depreciation":         "depreciation",
    "Interest":             "interest",
    "Profit before tax":    "profit_before_tax",
    "Tax":                  "tax",
    "Net profit":           "net_profit",
}

BS_ROW_MAP = {
    "Equity Share Capital":     "equity_share_capital",
    "Reserves":                 "reserves",
    "Borrowings":               "borrowings",
    "Other Liabilities":        "other_liabilities",
    "Net Block":                "net_block",
    "Capital Work in Progress": "capital_wip",
    "Investments":              "investments",
    "Other Assets":             "other_assets",
    "Receivables":              "trade_receivables",
    "Inventory":                "inventory",
    "Cash & Bank":              "cash_and_bank",
}

CF_ROW_MAP = {
    "Cash from Operating Activity": "cfo",
    "Cash from Investing Activity": "cfi",
    "Cash from Financing Activity": "cff",
    "Net Cash Flow":                "net_cash_flow",
}

SHARES_ROW_MAP = {
    "No. of Equity Shares": "no_of_shares",
}

# Labels that appear in the spreadsheet but carry no extractable financial
# data — silently skip these rather than flagging them as "unmapped".
KNOWN_SKIP_LABELS = {
    "PROFIT & LOSS", "BALANCE SHEET", "CASH FLOW:", "PRICE:", "DERIVED:",
    "Quarters", "Report Date", "Total", "Face value", "New Bonus Shares",
    "Dividend Amount", "Adjusted Equity Shares in Cr", "Operating Profit",
    "Expenses", "META", "COMPANY NAME", "LATEST VERSION", "CURRENT VERSION",
    "Number of shares", "Face Value", "Current Price", "Market Capitalization",
}

# All schema fields that live in the financials table (used for validation
# and for building the INSERT column list).
ALL_SCHEMA_FIELDS = (
    list(PNL_ROW_MAP.values())
    + list(BS_ROW_MAP.values())
    + list(CF_ROW_MAP.values())
    + list(SHARES_ROW_MAP.values())
)


# ---------------------------------------------------------------------------
# STEP 1 — Read the raw "Data Sheet" from the Excel file
# ---------------------------------------------------------------------------

def read_datasheet(filepath: str) -> pd.DataFrame:
    """Read the 'Data Sheet' tab from the Screener Excel export.

    Uses header=None so every cell is accessible by integer index —
    the spreadsheet has no consistent header row.
    """
    df = pd.read_excel(filepath, sheet_name="Data Sheet", header=None)
    return df


# ---------------------------------------------------------------------------
# STEP 2 — Extract company-level metadata from fixed cell positions
# ---------------------------------------------------------------------------

def _safe_get(df: pd.DataFrame, row: int, col: int):
    """Safely read a single cell, returning None for NaN or out-of-bounds."""
    try:
        val = df.iloc[row, col]
        if pd.isna(val):
            return None
        return val
    except Exception:
        return None


def extract_meta(df: pd.DataFrame) -> dict:
    """Pull company metadata from fixed cells in the Data Sheet.

    These positions are hard-coded to match the Screener export layout:
      row 0, col 1 → screener_name
      row 6, col 1 → face_value
      row 7, col 1 → current_price
      row 8, col 1 → market_cap
    """
    return {
        "screener_name": _safe_get(df, 0, 1),
        "face_value":    _safe_get(df, 6, 1),
        "current_price": _safe_get(df, 7, 1),
        "market_cap":    _safe_get(df, 8, 1),
    }


# ---------------------------------------------------------------------------
# STEP 3 — Extract fiscal year columns from the report-date row
# ---------------------------------------------------------------------------

def extract_fiscal_years(df: pd.DataFrame, report_date_row: int) -> list[tuple]:
    """Read the report-date row and convert each valid date to a fiscal year.

    Iterates all columns in the given row, skips NaN cells, and converts
    datetime values to fiscal year integers via pd.Timestamp(val).year.
    Returns a list of (col_index, fiscal_year) tuples.
    """
    year_columns = []

    for col_idx in range(df.shape[1]):
        val = _safe_get(df, report_date_row, col_idx)
        if val is None:
            continue
        try:
            # Convert whatever Excel gives us (datetime, string, etc.)
            # into a pandas Timestamp so we can extract the year.
            ts = pd.Timestamp(val)
            fiscal_year = ts.year
            year_columns.append((col_idx, fiscal_year))
        except Exception:
            # Not a parseable date — skip (e.g. the row label itself)
            continue

    return year_columns


# ---------------------------------------------------------------------------
# STEP 4 — Extract a section of rows into mapped + unmapped dicts
# ---------------------------------------------------------------------------

def extract_section(
    df: pd.DataFrame,
    row_map: dict,
    year_columns: list[tuple],
    start_row: int,
    end_row: int,
) -> tuple[dict, dict]:
    """Extract one section (P&L / BS / CF / Shares) from the Data Sheet.

    For each row between start_row and end_row (inclusive):
      - If the label matches a key in row_map → store value under the
        mapped schema field name, keyed by fiscal year.
      - If the label is in KNOWN_SKIP_LABELS → silently skip.
      - Otherwise → store in the unmapped dict for later LLM resolution.

    Returns (mapped, unmapped) where:
      mapped[fiscal_year][schema_field] = value
      unmapped[label][fiscal_year]      = value
    """
    mapped = {}    # {fiscal_year: {schema_field: value}}
    unmapped = {}  # {label: {fiscal_year: value}}

    for row_idx in range(start_row, end_row + 1):
        label = _safe_get(df, row_idx, 0)
        if label is None:
            continue

        # Normalise label to string and strip whitespace
        label = str(label).strip()
        if not label:
            continue

        if label in KNOWN_SKIP_LABELS:
            # Header rows, totals, meta labels — not financial data
            continue

        if label in row_map:
            # Known mapping — extract values for every fiscal year column
            schema_field = row_map[label]
            for col_idx, fiscal_year in year_columns:
                val = _safe_get(df, row_idx, col_idx)
                mapped.setdefault(fiscal_year, {})[schema_field] = val
        else:
            # Unknown label — stash for LLM resolution in step 5
            for col_idx, fiscal_year in year_columns:
                val = _safe_get(df, row_idx, col_idx)
                unmapped.setdefault(label, {})[fiscal_year] = val

    return mapped, unmapped


def _merge_dicts(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay into base at the fiscal_year → field level."""
    for fy, fields in overlay.items():
        base.setdefault(fy, {}).update(fields)
    return base


def _merge_unmapped(base: dict, overlay: dict) -> dict:
    """Merge unmapped label dicts across sections."""
    for label, year_vals in overlay.items():
        base.setdefault(label, {}).update(year_vals)
    return base


# ---------------------------------------------------------------------------
# STEP 5 — Use the LLM to detect currency/unit and resolve unmapped labels
# ---------------------------------------------------------------------------

def detect_currency_unit_and_resolve_unmapped(
    meta: dict,
    merged_mapped: dict,
    unmapped: dict,
) -> tuple[str, str, dict, dict]:
    """Ask the LLM to detect currency/unit and try to map unknown labels.

    Sends sample mapped data and all unmapped labels to the LLM. Expects
    a JSON response with currency, unit, and a resolved dict mapping each
    unmapped label to either a schema field name or null.

    Returns (currency, unit, final_mapped, other_findings).
    """
    # Build a sample of mapped data for context (pick a few representative fields)
    sample_fields = ["revenue", "net_profit", "borrowings", "cash_and_bank"]
    sample_data = {}
    for fy, fields in merged_mapped.items():
        sample_data[fy] = {
            k: v for k, v in fields.items()
            if k in sample_fields and v is not None
        }

    # Build the unmapped summary: label → {year: value}
    unmapped_summary = {}
    for label, year_vals in unmapped.items():
        unmapped_summary[label] = {
            str(fy): v for fy, v in year_vals.items() if v is not None
        }

    # List of valid schema fields the LLM can map to
    valid_fields = ALL_SCHEMA_FIELDS

    prompt = f"""You are a financial data analyst. I am ingesting a Screener.in "Data Sheet" Excel export for a company.

Company metadata:
  screener_name: {meta.get('screener_name')}

Here is a sample of ALREADY MAPPED financial data (field → value by fiscal year):
{json.dumps(sample_data, indent=2, default=str)}

Here are UNMAPPED row labels that I could not match to my schema. For each label I show the values by fiscal year:
{json.dumps(unmapped_summary, indent=2, default=str)}

Valid schema fields you can map to:
{json.dumps(valid_fields)}

Return a JSON object with exactly this structure:
{{
  "currency": "<3-letter currency code, e.g. INR>",
  "unit": "<unit of the numbers, e.g. Crores or Lakhs>",
  "resolved": {{
    "<unmapped label>": "<schema_field_name>" or null
  }}
}}

Rules for "resolved":
- If you are confident an unmapped label maps to one of the valid schema fields, set the value to that field name.
- If the label does not clearly map to any schema field, set the value to null.
- Every unmapped label listed above must appear as a key in "resolved".
"""

    llm_response = call_llm_json(prompt)

    currency = llm_response.get("currency", "INR")
    unit = llm_response.get("unit", "Crores")
    resolved = llm_response.get("resolved", {})

    # Process the LLM's resolved mappings
    other_findings = {}  # labels that couldn't be mapped → stay as other_findings

    for label, schema_field in resolved.items():
        if schema_field is not None and schema_field in valid_fields:
            # LLM confidently mapped this label → merge into mapped data
            if label in unmapped:
                for fy, val in unmapped[label].items():
                    merged_mapped.setdefault(fy, {})[schema_field] = val
                print(f"  Resolved: '{label}' → {schema_field}")
        else:
            # LLM couldn't map it → keep in other_findings
            if label in unmapped:
                other_findings[label] = unmapped[label]
                print(f"  WARNING: unmapped label kept in other_findings: '{label}'")

    # Catch any unmapped labels the LLM didn't mention in resolved
    for label in unmapped:
        if label not in resolved:
            other_findings[label] = unmapped[label]
            print(f"  WARNING: unmapped label not in LLM response: '{label}'")

    return currency, unit, merged_mapped, other_findings


# ---------------------------------------------------------------------------
# STEP 6 — Write everything into the MySQL database
# ---------------------------------------------------------------------------

def _cast_decimal(val):
    """Cast a value to float for DECIMAL columns, or None if NaN/missing."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


def _cast_bigint(val):
    """Cast a value to int for BIGINT columns, or None if NaN/missing."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def insert_into_db(
    company_name: str,
    meta: dict,
    merged_mapped: dict,
    other_findings: dict,
    currency: str,
    unit: str,
) -> int:
    """Insert/update companies, financials, and market_data tables.

    Uses ON DUPLICATE KEY UPDATE so re-ingestion overwrites rather than
    duplicates. Returns the company_id for logging.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- 6a. Upsert company ---
    # Check if company already exists to avoid duplicate inserts.
    # No unique key on company_name yet — so we check manually first.
    cursor.execute("SELECT id FROM companies WHERE company_name = %s", (company_name,))
    row = cursor.fetchone()

    if row:
        # Company already exists — update screener_name and face_value
        company_id = row[0]
        cursor.execute(
            """
            UPDATE companies
            SET screener_name = %s, face_value = %s
            WHERE id = %s
            """,
            (meta.get("screener_name"), _cast_decimal(meta.get("face_value")), company_id),
        )
    else:
        # New company — insert fresh row
        cursor.execute(
            """
            INSERT INTO companies (company_name, screener_name, face_value)
            VALUES (%s, %s, %s)
            """,
            (company_name, meta.get("screener_name"), _cast_decimal(meta.get("face_value"))),
        )
        company_id = cursor.lastrowid

    # --- 6b. Upsert financials for each fiscal year ---
    for fiscal_year, fields in merged_mapped.items():
        # Build the other_findings JSON for this specific year
        year_other = {}
        for label, year_vals in other_findings.items():
            if fiscal_year in year_vals and year_vals[fiscal_year] is not None:
                year_other[label] = year_vals[fiscal_year]

        # Convert to JSON string (empty dict → "{}" not NULL)
        other_findings_json = json.dumps(
            year_other, default=str
        ) if year_other else "{}"

        cursor.execute(
            """
            INSERT INTO financials (
                company_id, fiscal_year, currency, unit,
                revenue, raw_material_cost, change_in_inventory,
                power_and_fuel, other_mfr_exp, employee_cost,
                selling_and_admin, other_expenses, other_income,
                depreciation, interest, profit_before_tax, tax, net_profit,
                equity_share_capital, reserves, borrowings, other_liabilities,
                net_block, capital_wip, investments, other_assets,
                trade_receivables, inventory, cash_and_bank,
                cfo, cfi, cff, net_cash_flow,
                no_of_shares, other_findings
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
            ON DUPLICATE KEY UPDATE
                currency              = VALUES(currency),
                unit                  = VALUES(unit),
                revenue               = VALUES(revenue),
                raw_material_cost     = VALUES(raw_material_cost),
                change_in_inventory   = VALUES(change_in_inventory),
                power_and_fuel        = VALUES(power_and_fuel),
                other_mfr_exp         = VALUES(other_mfr_exp),
                employee_cost         = VALUES(employee_cost),
                selling_and_admin     = VALUES(selling_and_admin),
                other_expenses        = VALUES(other_expenses),
                other_income          = VALUES(other_income),
                depreciation          = VALUES(depreciation),
                interest              = VALUES(interest),
                profit_before_tax     = VALUES(profit_before_tax),
                tax                   = VALUES(tax),
                net_profit            = VALUES(net_profit),
                equity_share_capital  = VALUES(equity_share_capital),
                reserves              = VALUES(reserves),
                borrowings            = VALUES(borrowings),
                other_liabilities     = VALUES(other_liabilities),
                net_block             = VALUES(net_block),
                capital_wip           = VALUES(capital_wip),
                investments           = VALUES(investments),
                other_assets          = VALUES(other_assets),
                trade_receivables     = VALUES(trade_receivables),
                inventory             = VALUES(inventory),
                cash_and_bank         = VALUES(cash_and_bank),
                cfo                   = VALUES(cfo),
                cfi                   = VALUES(cfi),
                cff                   = VALUES(cff),
                net_cash_flow         = VALUES(net_cash_flow),
                no_of_shares          = VALUES(no_of_shares),
                other_findings        = VALUES(other_findings)
            """,
            (
                company_id,
                int(fiscal_year),
                currency,
                unit,
                # P&L fields
                _cast_decimal(fields.get("revenue")),
                _cast_decimal(fields.get("raw_material_cost")),
                _cast_decimal(fields.get("change_in_inventory")),
                _cast_decimal(fields.get("power_and_fuel")),
                _cast_decimal(fields.get("other_mfr_exp")),
                _cast_decimal(fields.get("employee_cost")),
                _cast_decimal(fields.get("selling_and_admin")),
                _cast_decimal(fields.get("other_expenses")),
                _cast_decimal(fields.get("other_income")),
                _cast_decimal(fields.get("depreciation")),
                _cast_decimal(fields.get("interest")),
                _cast_decimal(fields.get("profit_before_tax")),
                _cast_decimal(fields.get("tax")),
                _cast_decimal(fields.get("net_profit")),
                # Balance Sheet fields
                _cast_decimal(fields.get("equity_share_capital")),
                _cast_decimal(fields.get("reserves")),
                _cast_decimal(fields.get("borrowings")),
                _cast_decimal(fields.get("other_liabilities")),
                _cast_decimal(fields.get("net_block")),
                _cast_decimal(fields.get("capital_wip")),
                _cast_decimal(fields.get("investments")),
                _cast_decimal(fields.get("other_assets")),
                _cast_decimal(fields.get("trade_receivables")),
                _cast_decimal(fields.get("inventory")),
                _cast_decimal(fields.get("cash_and_bank")),
                # Cash Flow fields
                _cast_decimal(fields.get("cfo")),
                _cast_decimal(fields.get("cfi")),
                _cast_decimal(fields.get("cff")),
                _cast_decimal(fields.get("net_cash_flow")),
                # Shares
                _cast_bigint(fields.get("no_of_shares")),
                # Other findings JSON
                other_findings_json,
            ),
        )

    # --- 6c. Upsert market data ---
    cursor.execute(
        """
        INSERT INTO market_data (company_id, as_of_date, current_price, market_cap)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            current_price = VALUES(current_price),
            market_cap    = VALUES(market_cap)
        """,
        (
            company_id,
            date.today(),
            _cast_decimal(meta.get("current_price")),
            _cast_decimal(meta.get("market_cap")),
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()

    return company_id


# ---------------------------------------------------------------------------
# STEP 7 — Top-level orchestrator
# ---------------------------------------------------------------------------

def run_saverisk_ingestion(filepath: str, company_name: str):
    """Ingest a Screener "Data Sheet" Excel export into the MySQL database.

    Orchestrates the full pipeline: read → extract meta → extract fiscal
    years → extract sections → LLM resolution → database insert.
    """
    # Step 1: Read raw datasheet
    print("[1/6] Reading Data Sheet...")
    df = read_datasheet(filepath)
    print(f"  Sheet shape: {df.shape[0]} rows × {df.shape[1]} cols")

    # Step 2: Extract company metadata from fixed cells
    print("[2/6] Extracting metadata...")
    meta = extract_meta(df)
    print(f"  screener_name: {meta['screener_name']}")
    print(f"  face_value: {meta['face_value']}")
    print(f"  current_price: {meta['current_price']}")
    print(f"  market_cap: {meta['market_cap']}")

    # Step 3: Extract fiscal year columns from P&L report date row (row 15)
    print("[3/6] Extracting fiscal years...")
    year_columns = extract_fiscal_years(df, report_date_row=15)
    fiscal_years = [fy for _, fy in year_columns]
    print(f"  Found {len(year_columns)} fiscal years: {fiscal_years}")

    # Step 4: Extract all four sections using their respective row maps
    print("[4/6] Extracting financial sections...")

    # P&L: rows 14–38
    pnl_mapped, pnl_unmapped = extract_section(df, PNL_ROW_MAP, year_columns, 14, 38)
    print(f"  P&L: {len(pnl_mapped)} years mapped, {len(pnl_unmapped)} unmapped labels")

    # Balance Sheet: rows 54–78
    bs_mapped, bs_unmapped = extract_section(df, BS_ROW_MAP, year_columns, 54, 78)
    print(f"  BS:  {len(bs_mapped)} years mapped, {len(bs_unmapped)} unmapped labels")

    # Cash Flow: rows 79–88
    cf_mapped, cf_unmapped = extract_section(df, CF_ROW_MAP, year_columns, 79, 88)
    print(f"  CF:  {len(cf_mapped)} years mapped, {len(cf_unmapped)} unmapped labels")

    # Shares: rows 54–78 (shares row lives inside the BS section)
    shares_mapped, shares_unmapped = extract_section(df, SHARES_ROW_MAP, year_columns, 54, 78)
    print(f"  Shares: {len(shares_mapped)} years mapped, {len(shares_unmapped)} unmapped labels")

    # Merge all mapped data into one dict keyed by fiscal year
    merged_mapped = {}
    _merge_dicts(merged_mapped, pnl_mapped)
    _merge_dicts(merged_mapped, bs_mapped)
    _merge_dicts(merged_mapped, cf_mapped)
    _merge_dicts(merged_mapped, shares_mapped)

    # Merge all unmapped labels across sections
    all_unmapped = {}
    _merge_unmapped(all_unmapped, pnl_unmapped)
    _merge_unmapped(all_unmapped, bs_unmapped)
    _merge_unmapped(all_unmapped, cf_unmapped)
    _merge_unmapped(all_unmapped, shares_unmapped)

    print(f"  Total merged: {len(merged_mapped)} fiscal years, {len(all_unmapped)} unmapped labels")

    # Step 5: Ask LLM to detect currency/unit and resolve unmapped labels
    print("[5/6] Detecting currency/unit and resolving unmapped labels via LLM...")
    currency, unit, merged_mapped, other_findings = (
        detect_currency_unit_and_resolve_unmapped(meta, merged_mapped, all_unmapped)
    )
    print(f"  Currency: {currency}, Unit: {unit}")
    print(f"  Remaining other_findings labels: {len(other_findings)}")

    # Step 6: Insert everything into the database
    print("[6/6] Inserting into database...")
    company_id = insert_into_db(
        company_name, meta, merged_mapped, other_findings, currency, unit
    )

    # Final summary
    print("\n=== INGESTION COMPLETE ===")
    print(f"  Company: {company_name} (id={company_id})")
    print(f"  Fiscal years inserted: {len(merged_mapped)} — {sorted(merged_mapped.keys())}")
    print(f"  Currency: {currency}, Unit: {unit}")
    if other_findings:
        print(f"  WARNING: {len(other_findings)} labels stored in other_findings:")
        for label in other_findings:
            print(f"    - {label}")
    else:
        print("  All labels mapped — no other_findings.")
