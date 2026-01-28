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

PG_MODULE = None  # Will be 'psycopg2' or 'pg8000'

if USE_POSTGRES:
    # Try psycopg2 first (Streamlit Cloud), fall back to pg8000 (local/pure Python)
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        PG_MODULE = 'psycopg2'
    except ImportError:
        try:
            import pg8000
            PG_MODULE = 'pg8000'
        except ImportError:
            USE_POSTGRES = False
            POSTGRES_ERROR = "Neither psycopg2 nor pg8000 installed"

    if USE_POSTGRES:
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


class Database:
    """Unified database interface for SQLite and PostgreSQL."""

    def __init__(self):
        self.use_postgres = USE_POSTGRES
        self.connection_error = None
        self.pg_params = PG_PARAMS

        self.pg_module = PG_MODULE

        # Test PostgreSQL connection, fall back to SQLite if it fails
        if self.use_postgres:
            try:
                conn = self._pg_connect()
                conn.close()
            except Exception as e:
                self.connection_error = str(e)
                print(f"PostgreSQL connection failed: {e}")
                print(f"Connection params: host={PG_PARAMS.get('host')}, port={PG_PARAMS.get('port')}, user={PG_PARAMS.get('user')}")
                print("Falling back to SQLite...")
                self.use_postgres = False

        self._init_db()

    def _pg_connect(self):
        """Connect to PostgreSQL using whichever driver is available."""
        if self.pg_module == 'psycopg2':
            return psycopg2.connect(**PG_PARAMS)
        else:
            # pg8000
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            return pg8000.connect(
                host=PG_PARAMS['host'],
                port=int(PG_PARAMS['port']),
                database=PG_PARAMS['database'],
                user=PG_PARAMS['user'],
                password=PG_PARAMS['password'],
                ssl_context=ssl_context,
                timeout=10
            )

    def _get_connection(self):
        if self.use_postgres:
            return self._pg_connect()
        else:
            os.makedirs("data", exist_ok=True)
            return sqlite3.connect("data/metrics.db")

    def _dict_cursor(self, conn):
        """Get a cursor that returns dicts."""
        if self.use_postgres and self.pg_module == 'psycopg2':
            return conn.cursor(cursor_factory=RealDictCursor)
        elif self.use_postgres and self.pg_module == 'pg8000':
            # pg8000 doesn't have dict cursors - we handle this in _fetchall_dicts
            return conn.cursor()
        else:
            conn.row_factory = sqlite3.Row
            return conn.cursor()

    def _fetchall_dicts(self, cursor):
        """Fetch all rows as list of dicts (handles pg8000 which lacks dict cursor)."""
        rows = cursor.fetchall()
        if self.use_postgres and self.pg_module == 'pg8000':
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        else:
            return [dict(row) for row in rows]

    def _fetchone_dict(self, cursor):
        """Fetch one row as dict."""
        row = cursor.fetchone()
        if row is None:
            return None
        if self.use_postgres and self.pg_module == 'pg8000':
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        else:
            return dict(row)
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        c = conn.cursor()
        
        if self.use_postgres:
            # Create tables
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
                    total_cost_usd REAL DEFAULT 0.0,
                    prompt_version TEXT DEFAULT 'v6_calibrated_confidence'
                )
            """)
            c.execute("ALTER TABLE daily_runs ADD COLUMN IF NOT EXISTS prompt_version TEXT DEFAULT 'v6_calibrated_confidence'")
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
            c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_scenario ON metrics(scenario)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.commit()
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
                    total_cost_usd REAL DEFAULT 0.0,
                    prompt_version TEXT DEFAULT 'v6_calibrated_confidence'
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

        # Migrate existing databases to add new columns
        self._migrate_db()

    def _migrate_db(self):
        """Add new columns to existing databases."""
        if not self.use_postgres:
            # SQLite only - PostgreSQL handled via IF NOT EXISTS in CREATE TABLE
            conn = self._get_connection()
            c = conn.cursor()
            for col, typ, default in [
                ("total_tokens", "INTEGER", "0"),
                ("total_cost_usd", "REAL", "0.0"),
                ("prompt_version", "TEXT", "'v6_calibrated_confidence'"),
            ]:
                try:
                    c.execute(f"ALTER TABLE daily_runs ADD COLUMN {col} {typ} DEFAULT {default}")
                except Exception:
                    pass
            conn.commit()
            conn.close()

    def save_daily_run(self, run_id, run_date, timestamp, scenarios_run, scenarios_passed,
                       scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                       total_tokens=0, total_cost_usd=0.0, prompt_version='v6_calibrated_confidence'):
        """Save a daily run summary."""
        conn = self._get_connection()
        c = conn.cursor()

        if self.use_postgres:
            c.execute("""
                INSERT INTO daily_runs (run_id, run_date, timestamp, scenarios_run,
                    scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                    total_tokens, total_cost_usd, prompt_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    scenarios_run = EXCLUDED.scenarios_run,
                    scenarios_passed = EXCLUDED.scenarios_passed,
                    scenarios_failed = EXCLUDED.scenarios_failed,
                    overall_status = EXCLUDED.overall_status,
                    alerts = EXCLUDED.alerts,
                    hillclimb_suggestions = EXCLUDED.hillclimb_suggestions,
                    total_tokens = EXCLUDED.total_tokens,
                    total_cost_usd = EXCLUDED.total_cost_usd,
                    prompt_version = EXCLUDED.prompt_version
            """, (run_id, run_date, timestamp, scenarios_run, scenarios_passed,
                  scenarios_failed, overall_status, json.dumps(alerts), json.dumps(hillclimb_suggestions),
                  total_tokens, total_cost_usd, prompt_version))
        else:
            c.execute("""
                INSERT OR REPLACE INTO daily_runs (run_id, run_date, timestamp, scenarios_run,
                    scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                    total_tokens, total_cost_usd, prompt_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, run_date, timestamp, scenarios_run, scenarios_passed,
                  scenarios_failed, overall_status, json.dumps(alerts), json.dumps(hillclimb_suggestions),
                  total_tokens, total_cost_usd, prompt_version))

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
            c = self._dict_cursor(conn)
            c.execute("SELECT * FROM metrics ORDER BY timestamp")
            results = self._fetchall_dicts(c)
        else:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM metrics ORDER BY timestamp")
            results = self._fetchall_dicts(c)
        
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
            c = self._dict_cursor(conn)
            c.execute(query, params)
            results = self._fetchall_dicts(c)
        else:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(query, params)
            results = self._fetchall_dicts(c)

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

    def get_daily_runs(self):
        """Get all daily run summaries."""
        conn = self._get_connection()

        if self.use_postgres:
            c = self._dict_cursor(conn)
            c.execute("SELECT * FROM daily_runs ORDER BY timestamp DESC")
            results = self._fetchall_dicts(c)
        else:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM daily_runs ORDER BY timestamp DESC")
            results = self._fetchall_dicts(c)

        conn.close()
        return results

    def get_daily_run(self, run_id):
        """Get a single daily run by ID."""
        conn = self._get_connection()

        if self.use_postgres:
            c = self._dict_cursor(conn)
            c.execute("SELECT * FROM daily_runs WHERE run_id = %s", (run_id,))
        else:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM daily_runs WHERE run_id = ?", (run_id,))

        row = c.fetchone()
        result = None
        if row:
            if self.use_postgres and self.pg_module == 'pg8000':
                columns = [desc[0] for desc in c.description]
                result = dict(zip(columns, row))
            else:
                result = dict(row)

        conn.close()
        return result

    def save_complete_run(self, run_data, metrics_list, test_results_list):
        """Save an entire run atomically using a single connection.

        Args:
            run_data: dict with daily_run fields
            metrics_list: list of dicts with metric fields
            test_results_list: list of dicts with test result fields
        """
        conn = self._get_connection()
        c = conn.cursor()

        try:
            # 1. Save all test results
            for tr in test_results_list:
                if self.use_postgres:
                    c.execute("""
                        INSERT INTO test_results (run_id, scenario, test_case_id, prompt_id, ground_truth,
                            prediction, confidence, correct, llm_output, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (tr['run_id'], tr['scenario'], tr['test_case_id'], tr['prompt_id'],
                          tr['ground_truth'], tr['prediction'], tr['confidence'], tr['correct'],
                          tr['llm_output'], tr['timestamp']))
                else:
                    c.execute("""
                        INSERT INTO test_results (run_id, scenario, test_case_id, prompt_id, ground_truth,
                            prediction, confidence, correct, llm_output, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (tr['run_id'], tr['scenario'], tr['test_case_id'], tr['prompt_id'],
                          tr['ground_truth'], tr['prediction'], tr['confidence'], tr['correct'],
                          tr['llm_output'], tr['timestamp']))

            # 2. Save all metrics
            for m in metrics_list:
                if self.use_postgres:
                    c.execute("""
                        INSERT INTO metrics (run_id, scenario, timestamp, metric_name, metric_value,
                            threshold_min, threshold_max, unit, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (m['run_id'], m['scenario'], m['timestamp'], m['metric_name'],
                          m['metric_value'], m['threshold_min'], m['threshold_max'],
                          m['unit'], m['status']))
                else:
                    c.execute("""
                        INSERT INTO metrics (run_id, scenario, timestamp, metric_name, metric_value,
                            threshold_min, threshold_max, unit, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (m['run_id'], m['scenario'], m['timestamp'], m['metric_name'],
                          m['metric_value'], m['threshold_min'], m['threshold_max'],
                          m['unit'], m['status']))

            # 3. Save daily run summary
            prompt_ver = run_data.get('prompt_version', 'v6_calibrated_confidence')
            if self.use_postgres:
                c.execute("""
                    INSERT INTO daily_runs (run_id, run_date, timestamp, scenarios_run,
                        scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                        total_tokens, total_cost_usd, prompt_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        scenarios_run = EXCLUDED.scenarios_run,
                        scenarios_passed = EXCLUDED.scenarios_passed,
                        scenarios_failed = EXCLUDED.scenarios_failed,
                        overall_status = EXCLUDED.overall_status,
                        alerts = EXCLUDED.alerts,
                        hillclimb_suggestions = EXCLUDED.hillclimb_suggestions,
                        total_tokens = EXCLUDED.total_tokens,
                        total_cost_usd = EXCLUDED.total_cost_usd,
                        prompt_version = EXCLUDED.prompt_version
                """, (run_data['run_id'], run_data['run_date'], run_data['timestamp'],
                      run_data['scenarios_run'], run_data['scenarios_passed'],
                      run_data['scenarios_failed'], run_data['overall_status'],
                      json.dumps(run_data['alerts']), json.dumps(run_data['hillclimb_suggestions']),
                      run_data['total_tokens'], run_data['total_cost_usd'], prompt_ver))
            else:
                c.execute("""
                    INSERT OR REPLACE INTO daily_runs (run_id, run_date, timestamp, scenarios_run,
                        scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions,
                        total_tokens, total_cost_usd, prompt_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (run_data['run_id'], run_data['run_date'], run_data['timestamp'],
                      run_data['scenarios_run'], run_data['scenarios_passed'],
                      run_data['scenarios_failed'], run_data['overall_status'],
                      json.dumps(run_data['alerts']), json.dumps(run_data['hillclimb_suggestions']),
                      run_data['total_tokens'], run_data['total_cost_usd'], prompt_ver))

            # Commit everything at once
            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

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
