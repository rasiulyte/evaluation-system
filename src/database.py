"""
Database abstraction layer supporting SQLite (local) and PostgreSQL (Supabase/cloud).
"""
import os
import json
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

# Check if we should use PostgreSQL (Supabase) or SQLite
DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = DATABASE_URL is not None

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
else:
    import sqlite3


class Database:
    """Unified database interface for SQLite and PostgreSQL."""
    
    def __init__(self):
        self.use_postgres = USE_POSTGRES
        self._init_db()
    
    def _get_connection(self):
        if self.use_postgres:
            return psycopg2.connect(DATABASE_URL)
        else:
            os.makedirs("data", exist_ok=True)
            return sqlite3.connect("data/metrics.db")
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        c = conn.cursor()
        
        if self.use_postgres:
            c.execute("""
                CREATE TABLE IF NOT EXISTS daily_runs (
                    run_id TEXT PRIMARY KEY,
                    run_date TEXT,
                    timestamp TEXT,
                    scenarios_run INTEGER,
                    scenarios_passed INTEGER,
                    scenarios_failed INTEGER,
                    overall_status TEXT,
                    alerts TEXT,
                    hillclimb_suggestions TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT,
                    scenario TEXT,
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    threshold_min REAL,
                    threshold_max REAL,
                    unit TEXT,
                    status TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT,
                    scenario TEXT,
                    test_case_id TEXT,
                    prompt_id TEXT,
                    ground_truth TEXT,
                    prediction TEXT,
                    confidence REAL,
                    correct BOOLEAN,
                    llm_output TEXT,
                    timestamp TEXT
                )
            """)
            try:
                c.execute("CREATE INDEX idx_metrics_scenario ON metrics(scenario)")
            except:
                pass
            try:
                c.execute("CREATE INDEX idx_metrics_timestamp ON metrics(timestamp)")
            except:
                pass
        else:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS daily_runs (
                    run_id TEXT PRIMARY KEY,
                    run_date TEXT,
                    timestamp TEXT,
                    scenarios_run INTEGER,
                    scenarios_passed INTEGER,
                    scenarios_failed INTEGER,
                    overall_status TEXT,
                    alerts TEXT,
                    hillclimb_suggestions TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    scenario TEXT,
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    threshold_min REAL,
                    threshold_max REAL,
                    unit TEXT,
                    status TEXT
                );
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    scenario TEXT,
                    test_case_id TEXT,
                    prompt_id TEXT,
                    ground_truth TEXT,
                    prediction TEXT,
                    confidence REAL,
                    correct BOOLEAN,
                    llm_output TEXT,
                    timestamp TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_scenario ON metrics(scenario);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_test_results_run ON test_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_test_results_scenario ON test_results(scenario);
            """)
        
        conn.commit()
        conn.close()
    
    def save_daily_run(self, run_id, run_date, timestamp, scenarios_run, scenarios_passed, 
                       scenarios_failed, overall_status, alerts, hillclimb_suggestions):
        """Save a daily run summary."""
        conn = self._get_connection()
        c = conn.cursor()
        
        if self.use_postgres:
            c.execute("""
                INSERT INTO daily_runs (run_id, run_date, timestamp, scenarios_run, 
                    scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    scenarios_run = EXCLUDED.scenarios_run,
                    scenarios_passed = EXCLUDED.scenarios_passed,
                    scenarios_failed = EXCLUDED.scenarios_failed,
                    overall_status = EXCLUDED.overall_status,
                    alerts = EXCLUDED.alerts,
                    hillclimb_suggestions = EXCLUDED.hillclimb_suggestions
            """, (run_id, run_date, timestamp, scenarios_run, scenarios_passed,
                  scenarios_failed, overall_status, json.dumps(alerts), json.dumps(hillclimb_suggestions)))
        else:
            c.execute("""
                INSERT OR REPLACE INTO daily_runs (run_id, run_date, timestamp, scenarios_run,
                    scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, run_date, timestamp, scenarios_run, scenarios_passed,
                  scenarios_failed, overall_status, json.dumps(alerts), json.dumps(hillclimb_suggestions)))
        
        conn.commit()
        conn.close()
    
    def save_metric(self, run_id, scenario, timestamp, metric_name, metric_value, 
                    threshold_min, threshold_max, unit, status):
        """Save a single metric."""
        conn = self._get_connection()
        c = conn.cursor()
        
        if self.use_postgres:
            c.execute("""
                INSERT INTO metrics (run_id, scenario, timestamp, metric_name, metric_value,
                    threshold_min, threshold_max, unit, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, scenario, timestamp, metric_name, metric_value,
                  threshold_min, threshold_max, unit, status))
        else:
            c.execute("""
                INSERT INTO metrics (run_id, scenario, timestamp, metric_name, metric_value,
                    threshold_min, threshold_max, unit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, scenario, timestamp, metric_name, metric_value,
                  threshold_min, threshold_max, unit, status))
        
        conn.commit()
        conn.close()
    
    def get_all_metrics(self):
        """Get all metrics as list of dicts."""
        conn = self._get_connection()
        
        if self.use_postgres:
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute("SELECT * FROM metrics ORDER BY timestamp")
            results = [dict(row) for row in c.fetchall()]
        else:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM metrics ORDER BY timestamp")
            results = [dict(row) for row in c.fetchall()]
        
        conn.close()
        return results
    
    def get_run_ids(self):
        """Get all unique run IDs."""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT DISTINCT run_id FROM metrics ORDER BY run_id DESC")
        results = [row[0] for row in c.fetchall()]
        conn.close()
        return results

    def save_test_result(self, run_id, scenario, test_case_id, prompt_id, ground_truth,
                         prediction, confidence, correct, llm_output, timestamp):
        """Save a single test result."""
        conn = self._get_connection()
        c = conn.cursor()

        if self.use_postgres:
            c.execute("""
                INSERT INTO test_results (run_id, scenario, test_case_id, prompt_id, ground_truth,
                    prediction, confidence, correct, llm_output, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, scenario, test_case_id, prompt_id, ground_truth,
                  prediction, confidence, correct, llm_output, timestamp))
        else:
            c.execute("""
                INSERT INTO test_results (run_id, scenario, test_case_id, prompt_id, ground_truth,
                    prediction, confidence, correct, llm_output, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, scenario, test_case_id, prompt_id, ground_truth,
                  prediction, confidence, correct, llm_output, timestamp))

        conn.commit()
        conn.close()

    def get_test_results(self, run_id=None, scenario=None):
        """Get test results, optionally filtered by run_id and/or scenario."""
        conn = self._get_connection()

        query = "SELECT * FROM test_results WHERE 1=1"
        params = []

        if run_id:
            query += " AND run_id = %s" if self.use_postgres else " AND run_id = ?"
            params.append(run_id)
        if scenario:
            query += " AND scenario = %s" if self.use_postgres else " AND scenario = ?"
            params.append(scenario)

        query += " ORDER BY timestamp"

        if self.use_postgres:
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute(query, params)
            results = [dict(row) for row in c.fetchall()]
        else:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(query, params)
            results = [dict(row) for row in c.fetchall()]

        conn.close()
        return results


# Global database instance
db = Database()
