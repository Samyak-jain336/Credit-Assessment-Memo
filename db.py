"""
db.py — MySQL connection and schema initialisation for the CAM pipeline.

Handles all database concerns: connecting, creating tables if they don't
exist, and providing a reusable connection getter. No ingestion logic here.

Requires the following variables in .env:
    DB_HOST      e.g. localhost
    DB_PORT      e.g. 3306
    DB_USER      e.g. root
    DB_PASSWORD  e.g. yourpassword
    DB_NAME      e.g. creditmemo
"""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Return a live MySQL connection using credentials from .env.

    Raises mysql.connector.Error if the connection fails.
    """
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        database=os.environ.get("DB_NAME", "creditmemo"),
    )


def init_db():
    """Create all pipeline tables if they do not already exist.

    Safe to call on every run — uses CREATE TABLE IF NOT EXISTS so
    existing data is never dropped. Call this once at pipeline startup
    before any inserts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            company_name    VARCHAR(255) NOT NULL,
            screener_name   VARCHAR(255),
            cin             VARCHAR(50),
            face_value      DECIMAL(10,2)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financials (
            id                      INT AUTO_INCREMENT PRIMARY KEY,
            company_id              INT NOT NULL,
            fiscal_year             INT NOT NULL,
            currency                VARCHAR(20) NOT NULL DEFAULT 'INR',
            unit                    VARCHAR(20) NOT NULL,

            -- P&L
            revenue                 DECIMAL(15,2),
            raw_material_cost       DECIMAL(15,2),
            change_in_inventory     DECIMAL(15,2),
            power_and_fuel          DECIMAL(15,2),
            other_mfr_exp           DECIMAL(15,2),
            employee_cost           DECIMAL(15,2),
            selling_and_admin       DECIMAL(15,2),
            other_expenses          DECIMAL(15,2),
            other_income            DECIMAL(15,2),
            depreciation            DECIMAL(15,2),
            interest                DECIMAL(15,2),
            profit_before_tax       DECIMAL(15,2),
            tax                     DECIMAL(15,2),
            net_profit              DECIMAL(15,2),

            -- Balance Sheet
            equity_share_capital    DECIMAL(15,2),
            reserves                DECIMAL(15,2),
            borrowings              DECIMAL(15,2),
            other_liabilities       DECIMAL(15,2),
            net_block               DECIMAL(15,2),
            capital_wip             DECIMAL(15,2),
            investments             DECIMAL(15,2),
            other_assets            DECIMAL(15,2),
            trade_receivables       DECIMAL(15,2),
            inventory               DECIMAL(15,2),
            cash_and_bank           DECIMAL(15,2),

            -- Cash Flow
            cfo                     DECIMAL(15,2),
            cfi                     DECIMAL(15,2),
            cff                     DECIMAL(15,2),
            net_cash_flow           DECIMAL(15,2),

            -- Shares
            no_of_shares            BIGINT,

            -- Unmapped or unrecognised rows from source file stored as JSON
            -- so no data is silently dropped during ingestion.
            other_findings          JSON,

            FOREIGN KEY (company_id) REFERENCES companies(id),
            UNIQUE KEY uq_company_year (company_id, fiscal_year)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            company_id      INT NOT NULL,
            as_of_date      DATE NOT NULL,
            current_price   DECIMAL(10,2),
            market_cap      DECIMAL(15,2),
            FOREIGN KEY (company_id) REFERENCES companies(id),
            UNIQUE KEY uq_company_date (company_id, as_of_date)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bank_statements (
            id                      INT AUTO_INCREMENT PRIMARY KEY,
            company_id              INT NOT NULL,
            bank_name               VARCHAR(255),
            account_type            VARCHAR(100),
            account_number          VARCHAR(100),
            sanctioned_limit        DECIMAL(15,2),
            statement_period_from   DATE,
            statement_period_to     DATE,
            currency                VARCHAR(20) DEFAULT 'INR',
            unit                    VARCHAR(20) NOT NULL,
            avg_monthly_balance     DECIMAL(15,2),
            total_credits           DECIMAL(15,2),
            total_debits            DECIMAL(15,2),
            closing_balance         DECIMAL(15,2),
            emi_count               INT,
            bounce_count            INT,
            source_filename         VARCHAR(255),
            other_findings          JSON,
            FOREIGN KEY (company_id) REFERENCES companies(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id                      INT AUTO_INCREMENT PRIMARY KEY,
            company_name            VARCHAR(255) NOT NULL UNIQUE,
            annual_report_parsed    TINYINT DEFAULT 0,
            audit_report_parsed     TINYINT DEFAULT 0,
            screener_parsed         TINYINT DEFAULT 0,
            exchange_filings_parsed TINYINT DEFAULT 0,
            bank_statements_parsed  TINYINT DEFAULT 0,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialised — all tables ready.")


if __name__ == "__main__":
    # Run directly to initialise the database:
    # python db.py
    init_db()
