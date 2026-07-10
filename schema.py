"""
schema.py

Central location for database table names, field names and reconciliation
configuration used throughout the CAM Generation pipeline.

No table or column names should be hardcoded anywhere else.
"""

# ============================================================
# TABLE NAMES
# ============================================================

COMPANIES_TABLE = "companies"
FINANCIALS_TABLE = "financials"
BANK_STATEMENTS_TABLE = "bank_statements"
MARKET_DATA_TABLE = "market_data"
INGESTION_RUNS_TABLE = "ingestion_runs"

# Optional table (create during integration if required)
INCONSISTENCIES_TABLE = "inconsistencies"

# ============================================================
# COMPANY
# ============================================================

COMPANY_ID = "id"
COMPANY_NAME = "company_name"
SCREENER_NAME = "screener_name"
CIN = "cin"
FACE_VALUE = "face_value"

# ============================================================
# FINANCIALS
# ============================================================

FISCAL_YEAR = "fiscal_year"

REVENUE = "revenue"
RAW_MATERIAL_COST = "raw_material_cost"
CHANGE_IN_INVENTORY = "change_in_inventory"
POWER_AND_FUEL = "power_and_fuel"
OTHER_MFR_EXP = "other_mfr_exp"

EMPLOYEE_COST = "employee_cost"
SELLING_AND_ADMIN = "selling_and_admin"
OTHER_EXPENSES = "other_expenses"

OTHER_INCOME = "other_income"

DEPRECIATION = "depreciation"
INTEREST = "interest"

PROFIT_BEFORE_TAX = "profit_before_tax"
TAX = "tax"
NET_PROFIT = "net_profit"

EQUITY_SHARE_CAPITAL = "equity_share_capital"
RESERVES = "reserves"
BORROWINGS = "borrowings"
OTHER_LIABILITIES = "other_liabilities"

NET_BLOCK = "net_block"
CAPITAL_WIP = "capital_wip"
INVESTMENTS = "investments"
OTHER_ASSETS = "other_assets"

TRADE_RECEIVABLES = "trade_receivables"
INVENTORY = "inventory"
CASH_AND_BANK = "cash_and_bank"

CFO = "cfo"
CFI = "cfi"
CFF = "cff"
NET_CASH_FLOW = "net_cash_flow"

NUMBER_OF_SHARES = "no_of_shares"

OTHER_FINDINGS = "other_findings"

# ============================================================
# BANK STATEMENTS
# ============================================================

BANK_NAME = "bank_name"
ACCOUNT_TYPE = "account_type"
ACCOUNT_NUMBER = "account_number"

SANCTIONED_LIMIT = "sanctioned_limit"

AVG_MONTHLY_BALANCE = "avg_monthly_balance"

TOTAL_CREDITS = "total_credits"
TOTAL_DEBITS = "total_debits"

CLOSING_BALANCE = "closing_balance"

EMI_COUNT = "emi_count"
BOUNCE_COUNT = "bounce_count"

# ============================================================
# RECONCILIATION
# ============================================================

RECONCILIATION_FIELDS = [

    REVENUE,
    NET_PROFIT,
    BORROWINGS,
    CASH_AND_BANK,
    INVENTORY,
    TRADE_RECEIVABLES,
    CFO,
    NET_CASH_FLOW,

]

DEFAULT_TOLERANCE = 0.01

MATCH = "match"
MISMATCH = "mismatch"

# ============================================================
# DOCUMENT TYPES
# ============================================================

ANNUAL_REPORT = "annual_report"
AUDIT_REPORT = "audit_report"
SAVE_RISK = "save_risk"

# ============================================================
# CHROMA FILTERS
# ============================================================

STANDALONE = "standalone"
CONSOLIDATED = "consolidated"
UNKNOWN = "unknown"

# ============================================================
# CAM SECTION TITLES
# ============================================================

CAM_SECTIONS = {

    1: "Applicant Overview",
    2: "Company Background",
    3: "Financial Analysis",
    4: "Banking Conduct",
    5: "Tax and Statutory Compliance",
    6: "Audit and Compliance",
    7: "Risk Assessment",
    8: "Collateral",
    9: "Data Consistency Review",
    10: "Recommendation",

}

SECTION_QUERIES = {
    "Applicant Overview": [
        "company overview",
        "registered office address",
        "constitution of business",
        "managing director name",
        "chief financial officer name",
        "company secretary name",
        "key managerial personnel",
        "remuneration key managerial",
    ],
    "Company Background": [
        "incorporation history",
        "business activities",
        "managing director chairman",
        "board of directors members",
        "key managerial personnel designation",
        "directors report",
        "independent director appointed",
        "whole time director executive",
    ],
    "Financial Analysis": [
        "revenue from operations",
        "profit after tax",
        "cash flow from operations",
        "total income",
        "profit and loss account",
        "balance sheet as at",
        "net cash flow from operating activities",
        "cash and bank balances",
        "margin money fixed deposit",
        "notes to accounts",
    ],
    "Banking Conduct": [
        "average monthly balance",
        "cheque bounce",
        "emi deduction",
        "term loan vehicle machinery",
        "secured loans bank",
        "repayment schedule",
    ],
    "Tax and Statutory Compliance": [
        "income tax return",
        "gst return",
        "statutory dues",
    ],
    "Audit and Compliance": [
        "auditor qualification and audit opinion",
        "emphasis of matter",
    ],
    "Risk Assessment": [
        "borrowings and long term debt",
        "contingent liabilities",
        "related party transactions",
        "net profit",
        "cash flow from operations",
        "debt to equity ratio",
        "initial public offering ipo capital raised",
        "equity share capital rights issue proceeds",
        "letter of credit margin money",
    ],
    "Collateral": [
        "collateral security",
        "hypothecation",
        "valuation report",
    ],
    "Data Consistency Review": [],
    "Recommendation": [
        "credit recommendation",
        "debt to equity ratio",
        "current ratio",
        "cash flow from operations",
    ],
}

SECTION_TABLE_SCHEMA = {
    "Financial Analysis": ["metric", "fy25", "fy24", "change"],
    "Banking Conduct": ["lender", "facility_type", "amount_lakhs", "roi", "security"],
    "Risk Assessment": ["ratio", "fy25", "fy24", "trend"],
}