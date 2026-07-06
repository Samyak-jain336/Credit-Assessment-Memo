"""
reset.py — Utility script to completely wipe all pipeline data.

Drops and recreates all MySQL tables, and deletes all ChromaDB documents.
Used when switching to a new company or resetting a failed test run.

No if __name__ == "__main__" block — will be called by an orchestrator later.
"""

import os
from db import get_connection, init_db
from vector_store import init_vector_store


# ---------------------------------------------------------------------------
# FUNCTION 1 — Reset MySQL Database
# ---------------------------------------------------------------------------

def reset_mysql() -> None:
    """Drop and recreate all MySQL tables for a clean slate.

    Disables foreign key checks during dropping so tables can be removed safely
    without dependency constraint errors, then calls init_db() to rebuild the schema.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Disable foreign key checks so tables can be dropped without constraint errors
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

    # Drop all tables in clean order using IF EXISTS to avoid errors if they don't exist yet.
    # Note: ingestion_runs must be reset! Otherwise, when switching to a new company or
    # re-running a test, the parser agent will see the old flags as DONE (1) and skip
    # all ingestion steps.
    tables_to_drop = [
        "ingestion_runs",
        "bank_statements",
        "market_data",
        "financials",
        "companies",
    ]

    for table in tables_to_drop:
        print(f"  Dropping MySQL table: {table}...")
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # Re-enable foreign key checks after dropping is complete
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    conn.commit()
    cursor.close()
    conn.close()

    # Recreate all tables fresh with default schema
    print("  Recreating MySQL tables via init_db()...")
    init_db()

    print("MySQL reset complete.\n")


# ---------------------------------------------------------------------------
# FUNCTION 2 — Reset ChromaDB Collection
# ---------------------------------------------------------------------------

def reset_chromadb(gemini_api_key: str) -> None:
    """Delete all document chunks from the ChromaDB collection.

    We delete by IDs rather than using a 'where' filter because the filter syntax
    varies across different ChromaDB versions, whereas fetching all IDs and passing
    them to delete(ids=...) is universally supported and reliable.
    """
    collection, _ = init_vector_store(api_key=gemini_api_key)

    before_count = collection.count()
    print(f"  ChromaDB documents before reset: {before_count}")

    # Fetch all existing document IDs
    all_ids = collection.get()["ids"]
    if all_ids:
        print(f"  Deleting {len(all_ids)} document chunks from ChromaDB...")
        collection.delete(ids=all_ids)
    else:
        print("  No documents found in ChromaDB to delete.")

    after_count = collection.count()
    print(f"  ChromaDB documents after reset: {after_count}")
    print("ChromaDB reset complete.\n")


# ---------------------------------------------------------------------------
# FUNCTION 3 — Reset All Pipeline Data
# ---------------------------------------------------------------------------

def reset_all(gemini_api_key: str) -> None:
    """Wipe both MySQL and ChromaDB data completely."""
    print("WARNING: This will permanently delete ALL pipeline data.")
    print(" MySQL: all companies, financials, bank statements, ingestion state.")
    print(" ChromaDB: all document chunks from all companies.")
    print(" This cannot be undone.\n")

    print("--- Resetting MySQL ---")
    reset_mysql()

    print("--- Resetting ChromaDB ---")
    reset_chromadb(gemini_api_key)

    print("ALL DATA RESET COMPLETE — ready for new company ingestion.")
