#!/usr/bin/env python3
"""
Database Verification and Fix Script

Run this to:
1. Check if the database has the correct schema (including cost columns)
2. Fix any missing columns
3. Optionally reset the database to start fresh

Usage:
    python fix_database.py          # Check and fix
    python fix_database.py --reset  # Delete and recreate database
"""

import sqlite3
import os
import sys
from pathlib import Path

DB_PATH = Path("data/metrics.db")


def check_schema():
    """Check if database has all required columns."""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return False, []
    
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    # Get column info for daily_runs table
    c.execute("PRAGMA table_info(daily_runs)")
    columns = {row[1] for row in c.fetchall()}
    
    required_columns = {
        'run_id', 'run_date', 'timestamp', 'scenarios_run',
        'scenarios_passed', 'scenarios_failed', 'overall_status',
        'alerts', 'hillclimb_suggestions', 'total_tokens', 'total_cost_usd'
    }
    
    missing = required_columns - columns
    
    conn.close()
    
    if missing:
        print(f"⚠️  Missing columns in daily_runs: {missing}")
        return False, list(missing)
    else:
        print("✓ Schema looks correct - all columns present")
        return True, []


def add_missing_columns(missing_columns):
    """Add missing columns to the database."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    column_defaults = {
        'total_tokens': 'INTEGER DEFAULT 0',
        'total_cost_usd': 'REAL DEFAULT 0.0'
    }
    
    for col in missing_columns:
        if col in column_defaults:
            try:
                sql = f"ALTER TABLE daily_runs ADD COLUMN {col} {column_defaults[col]}"
                c.execute(sql)
                print(f"  ✓ Added column: {col}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    print(f"  - Column {col} already exists")
                else:
                    print(f"  ❌ Error adding {col}: {e}")
    
    conn.commit()
    conn.close()


def show_data_summary():
    """Show summary of data in the database."""
    if not DB_PATH.exists():
        print("No database to summarize")
        return
    
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    
    print("\n📊 Database Summary:")
    
    # Daily runs
    try:
        c.execute("SELECT COUNT(*) FROM daily_runs")
        count = c.fetchone()[0]
        print(f"  daily_runs: {count} records")
        
        if count > 0:
            c.execute("SELECT run_id, total_tokens, total_cost_usd FROM daily_runs ORDER BY timestamp DESC LIMIT 3")
            recent = c.fetchall()
            print("  Recent runs:")
            for run_id, tokens, cost in recent:
                print(f"    - {run_id}: {tokens or 0} tokens, ${cost or 0:.4f}")
    except Exception as e:
        print(f"  daily_runs: Error - {e}")
    
    # Metrics
    try:
        c.execute("SELECT COUNT(*) FROM metrics")
        count = c.fetchone()[0]
        print(f"  metrics: {count} records")
    except Exception as e:
        print(f"  metrics: Error - {e}")
    
    # Test results
    try:
        c.execute("SELECT COUNT(*) FROM test_results")
        count = c.fetchone()[0]
        print(f"  test_results: {count} records")
    except Exception as e:
        print(f"  test_results: Error - {e}")
    
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
