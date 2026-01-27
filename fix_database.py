#!/usr/bin/env python3
"""
Database verification and fix script for the evaluation system.
Run this after pulling updates to ensure your database schema is current.

Usage:
    python fix_database.py          # Check and fix schema
    python fix_database.py --reset  # Delete and recreate database
"""

import sqlite3
import sys
from pathlib import Path

# Database path (same as used by the app)
DB_PATH = Path("data/metrics.db")

# All required columns with their types and defaults
# This covers the full schema from database.py
COLUMN_DEFAULTS = {
    'total_tokens': ('INTEGER', '0'),
    'total_cost_usd': ('REAL', '0.0'),
    'run_id': ('TEXT', "''"),
    'run_date': ('TEXT', "''"),
    'timestamp': ('TEXT', "''"),
    'scenarios_run': ('INTEGER', '0'),
    'scenarios_passed': ('INTEGER', '0'),
    'scenarios_failed': ('INTEGER', '0'),
    'overall_status': ('TEXT', "'unknown'"),
    'alerts': ('TEXT', "'[]'"),
    'hillclimb_suggestions': ('TEXT', "'[]'"),
}

REQUIRED_COLUMNS = list(COLUMN_DEFAULTS.keys())


def check_schema():
    """Check if database has all required columns."""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return False, []
    
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    # Check if daily_runs table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_runs'")
    if not c.fetchone():
        print("❌ Table 'daily_runs' does not exist")
        conn.close()
        return False, ["table_missing"]
    
    # Get existing columns
    c.execute("PRAGMA table_info(daily_runs)")
    existing_columns = {row[1] for row in c.fetchall()}
    
    # Check for missing columns
    missing = [col for col in REQUIRED_COLUMNS if col not in existing_columns]
    
    if missing:
        print(f"❌ Missing columns: {', '.join(missing)}")
    else:
        print("✓ All required columns present")
    
    conn.close()
    return len(missing) == 0, missing


def add_missing_columns(missing_columns):
    """Add missing columns to the daily_runs table."""
    if not DB_PATH.exists():
        print("Cannot fix - database doesn't exist")
        return
    
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    for col in missing_columns:
        if col == "table_missing":
            print("⚠️  Table 'daily_runs' is missing. Run an evaluation first to create it,")
            print("   or use --reset to start fresh.")
            continue
            
        if col in COLUMN_DEFAULTS:
            col_type, default_val = COLUMN_DEFAULTS[col]
            try:
                c.execute(f"ALTER TABLE daily_runs ADD COLUMN {col} {col_type} DEFAULT {default_val}")
                print(f"  ✓ Added column: {col} ({col_type})")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    print(f"  ⓘ Column {col} already exists")
                else:
                    print(f"  ❌ Error adding {col}: {e}")
        else:
            # This shouldn't happen if COLUMN_DEFAULTS is complete, but warn if it does
            print(f"  ⚠️  Unknown column '{col}' - cannot add automatically.")
            print(f"      Manual intervention may be needed, or use --reset to start fresh.")
    
    conn.commit()
    conn.close()


def show_data_summary():
    """Show summary of data in the database."""
    if not DB_PATH.exists():
        print("No database to summarize")
        return
    
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    # First check what tables exist
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    
    if not tables:
        print("\n📊 Database exists but has no tables yet.")
        print("   Run an evaluation first to initialize the schema.")
        conn.close()
        return
    
    print(f"\n📊 Database Summary:")
    print(f"   Tables found: {', '.join(tables)}")
    
    # Daily runs
    if 'daily_runs' in tables:
        try:
            c.execute("SELECT COUNT(*) FROM daily_runs")
            count = c.fetchone()[0]
            print(f"   daily_runs: {count} records")
            
            if count > 0:
                c.execute("SELECT run_id, total_tokens, total_cost_usd FROM daily_runs ORDER BY timestamp DESC LIMIT 3")
                recent = c.fetchall()
                print("   Recent runs:")
                for run_id, tokens, cost in recent:
                    print(f"     - {run_id}: {tokens or 0} tokens, ${cost or 0:.4f}")
        except Exception as e:
            print(f"   daily_runs: Error reading - {e}")
    
    # Metrics
    if 'metrics' in tables:
        try:
            c.execute("SELECT COUNT(*) FROM metrics")
            count = c.fetchone()[0]
            print(f"   metrics: {count} records")
        except Exception as e:
            print(f"   metrics: Error reading - {e}")
    
    # Test results
    if 'test_results' in tables:
        try:
            c.execute("SELECT COUNT(*) FROM test_results")
            count = c.fetchone()[0]
            print(f"   test_results: {count} records")
        except Exception as e:
            print(f"   test_results: Error reading - {e}")
    
    conn.close()


def reset_database():
    """Delete and let the app recreate the database."""
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"✓ Deleted {DB_PATH}")
    else:
        print(f"No database to delete at {DB_PATH}")
    
    print("The database will be recreated on next app run with correct schema.")


def main():
    print("=" * 50)
    print("Database Verification and Fix Script")
    print("=" * 50)
    
    # Check for reset flag
    if "--reset" in sys.argv:
        print("\n⚠️  RESET MODE - This will delete all data!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            reset_database()
        else:
            print("Reset cancelled.")
        return
    
    # Check schema
    print("\n1. Checking database schema...")
    ok, missing = check_schema()
    
    # Fix if needed
    if not ok and missing:
        print("\n2. Fixing missing columns...")
        add_missing_columns(missing)
        
        # Verify fix
        print("\n3. Verifying fix...")
        check_schema()
    
    # Show summary
    show_data_summary()
    
    print("\n" + "=" * 50)
    print("Done! If issues persist, try: python fix_database.py --reset")
    print("=" * 50)


if __name__ == "__main__":
    main()
