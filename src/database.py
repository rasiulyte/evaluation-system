"""
Database abstraction layer supporting SQLite (local) and PostgreSQL (Supabase/cloud).
"""
import os
import json
import sqlite3
import re
from typing import Dict, Optional, List
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote

load_dotenv()

# Check if we should use PostgreSQL (Supabase) or SQLite
DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = DATABASE_URL is not None
POSTGRES_ERROR = None
PG_PARAMS = None

if USE_POSTGRES:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Parse DATABASE_URL into components
        parsed = urlparse(DATABASE_URL)
        PG_PARAMS = {
            'host': parsed.hostname,
            'port': parsed.port or 5432,
            'database': parsed.path.lstrip('/') or 'postgres',
            'user': unquote(parsed.username) if parsed.username else 'postgres',
            'password': unquote(parsed.password) if parsed.password else '',
            'sslmode': 'require',
            'connect_timeout': 10
        }
    except ImportError:
        USE_POSTGRES = False
        POSTGRES_ERROR = "psycopg2 not installed"


class Database:
    """Unified database interface for SQLite and PostgreSQL."""

    def __init__(self):
        self.use_postgres = USE_POSTGRES
        self.connection_error = None
        self.pg_params = PG_PARAMS

        # Test PostgreSQL connection, fall back to SQLite if it fails
        if self.use_postgres:
            try:
                conn = psycopg2.connect(**PG_PARAMS)
                conn.close()
            except Exception as e:
                self.connection_error = str(e)
                print(f"PostgreSQL connection failed: {e}")
                print(f"Connection params: host={PG_PARAMS.get('host')}, port={PG_PARAMS.get('port')}, user={PG_PARAMS.get('user')}")
                print("Falling back to SQLite...")
                self.use_postgres = False

        self._init_db()

    def _get_connection(self):
        if self.use_postgres:
            return psycopg2.connect(**PG_PARAMS)
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
                    hillclimb_suggestions TEXT,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_usd REAL DEFAULT 0.0
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
                    hillclimb_suggestions TEXT,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_usd REAL DEFAULT 0.0
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
                       scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                       total_tokens=0, total_cost_usd=0.0):
        """Save a daily run summary."""
        conn = self._get_connection()
        c = conn.cursor()

        if self.use_postgres:
            c.execute("""
                INSERT INTO daily_runs (run_id, run_date, timestamp, scenarios_run,
                    scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                    total_tokens, total_cost_usd)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    scenarios_run = EXCLUDED.scenarios_run,
                    scenarios_passed = EXCLUDED.scenarios_passed,
                    scenarios_failed = EXCLUDED.scenarios_failed,
                    overall_status = EXCLUDED.overall_status,
                    alerts = EXCLUDED.alerts,
                    hillclimb_suggestions = EXCLUDED.hillclimb_suggestions,
                    total_tokens = EXCLUDED.total_tokens,
                    total_cost_usd = EXCLUDED.total_cost_usd
            """, (run_id, run_date, timestamp, scenarios_run, scenarios_passed,
                  scenarios_failed, overall_status, json.dumps(alerts), json.dumps(hillclimb_suggestions),
                  total_tokens, total_cost_usd))
        else:
            c.execute("""
                INSERT OR REPLACE INTO daily_runs (run_id, run_date, timestamp, scenarios_run,
                    scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                    total_tokens, total_cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, run_date, timestamp, scenarios_run, scenarios_passed,
                  scenarios_failed, overall_status, json.dumps(alerts), json.dumps(hillclimb_suggestions),
                  total_tokens, total_cost_usd))

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

    def _ensure_test_results_table(self, conn, c):
        """Ensure test_results table exists."""
        if self.use_postgres:
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
            conn.commit()
        else:
            c.execute("""
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
                )
            """)
            conn.commit()

    def save_test_result(self, run_id, scenario, test_case_id, prompt_id, ground_truth,
                         prediction, confidence, correct, llm_output, timestamp):
        """Save a single test result."""
        conn = self._get_connection()
        c = conn.cursor()

        # Ensure table exists (in case it wasn't created during init)
        try:
            self._ensure_test_results_table(conn, c)
        except Exception:
            pass  # Table might already exist

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

    def get_test_results_count(self):
        """Get count of test results for debugging."""
        conn = self._get_connection()
        c = conn.cursor()
        try:
            c.execute("SELECT COUNT(*) FROM test_results")
            count = c.fetchone()[0]
        except Exception as e:
            count = f"Error: {e}"
        conn.close()
        return count

    def debug_info(self):
        """Get debug information about database state."""
        conn = self._get_connection()
        c = conn.cursor()
        info = {"backend": "PostgreSQL" if self.use_postgres else "SQLite"}
        if self.pg_params:
            info["pg_host"] = self.pg_params.get('host', 'N/A')
            info["pg_user"] = self.pg_params.get('user', 'N/A')
            info["pg_port"] = self.pg_params.get('port', 'N/A')
        if self.connection_error:
            info["pg_error"] = self.connection_error
        try:
            c.execute("SELECT COUNT(*) FROM test_results")
            info["test_results_count"] = c.fetchone()[0]
        except Exception as e:
            info["test_results_error"] = str(e)
        try:
            c.execute("SELECT COUNT(*) FROM metrics")
            info["metrics_count"] = c.fetchone()[0]
        except Exception as e:
            info["metrics_error"] = str(e)
        try:
            c.execute("SELECT COUNT(*) FROM daily_runs")
            info["daily_runs_count"] = c.fetchone()[0]
        except Exception as e:
            info["daily_runs_error"] = str(e)
        conn.close()
        return info


# Global database instance
db = Database()
