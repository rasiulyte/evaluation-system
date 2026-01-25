"""
Daily Evaluation Orchestrator
Runs all evaluation scenarios, tracks metrics, and triggers alerts.

Usage:
    python -m src.orchestrator              # Run all enabled scenarios
    python -m src.orchestrator --dry-run    # Preview without running
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import yaml
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import Evaluator
from metrics import MetricsCalculator
from database import db

class EvalScenario(Enum):
    HALLUCINATION_DETECTION = "hallucination_detection"
    FACTUAL_ACCURACY = "factual_accuracy"
    REASONING_QUALITY = "reasoning_quality"
    INSTRUCTION_FOLLOWING = "instruction_following"
    SAFETY_COMPLIANCE = "safety_compliance"
    CONSISTENCY = "consistency"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class MetricResult:
    name: str
    value: float
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    unit: str = ""
    
    @property
    def status(self) -> HealthStatus:
        if self.threshold_min is not None and self.value < self.threshold_min:
            return HealthStatus.CRITICAL
        if self.threshold_max is not None and self.value > self.threshold_max:
            return HealthStatus.CRITICAL
        # Warning if close to threshold (within 5%)
        if self.threshold_min is not None and self.value < self.threshold_min + 0.05:
            return HealthStatus.WARNING
        if self.threshold_max is not None and self.value > self.threshold_max - 0.05:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

@dataclass
class ScenarioResult:
    scenario: str
    run_id: str
    timestamp: str
    prompt_version: str
    model: str
    metrics: Dict[str, MetricResult]
    sample_size: int
    duration_seconds: float
    status: HealthStatus

@dataclass
class DailyRunSummary:
    run_id: str
    run_date: str
    timestamp: str
    scenarios_run: int
    scenarios_passed: int
    scenarios_failed: int
    overall_status: HealthStatus
    alerts: List[str] = field(default_factory=list)
    hillclimb_suggestions: List[str] = field(default_factory=list)

class MetricsDatabase:
    """SQLite storage for metrics history"""
    def __init__(self, db_path: str = "data/metrics.db"):
        self.db_path = db_path
        self._init_db()
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
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
        CREATE TABLE IF NOT EXISTS scenario_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            scenario TEXT,
            timestamp TEXT,
            prompt_version TEXT,
            model TEXT,
            sample_size INTEGER,
            duration_seconds REAL,
            status TEXT,
            metadata TEXT,
            FOREIGN KEY (run_id) REFERENCES daily_runs(run_id)
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
            status TEXT,
            FOREIGN KEY (run_id) REFERENCES daily_runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            scenario TEXT,
            baseline_run_id TEXT,
            experiment_run_id TEXT,
            hypothesis TEXT,
            changes TEXT,
            status TEXT,
            conclusion TEXT,
            created_at TEXT,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_scenario ON metrics(scenario);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_scenario_results_run ON scenario_results(run_id);
        """)
        conn.commit()
        conn.close()
    def save_daily_run(self, summary: DailyRunSummary):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO daily_runs (run_id, run_date, timestamp, scenarios_run, scenarios_passed, scenarios_failed, overall_status, alerts, hillclimb_suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.run_id, summary.run_date, summary.timestamp, summary.scenarios_run, summary.scenarios_passed, summary.scenarios_failed, summary.overall_status.value,
            json.dumps(summary.alerts), json.dumps(summary.hillclimb_suggestions)
        ))
        conn.commit()
        conn.close()

    def save_metrics(self, run_id: str, scenario: str, timestamp: str, metrics: Dict[str, 'MetricResult']):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for metric in metrics.values():
            c.execute('''
                INSERT INTO metrics (run_id, scenario, timestamp, metric_name, metric_value, threshold_min, threshold_max, unit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                scenario,
                timestamp,
                metric.name,
                metric.value,
                metric.threshold_min,
                metric.threshold_max,
                metric.unit,
                metric.status.value
            ))
        conn.commit()
        conn.close()

class EvaluationOrchestrator:
    """Main orchestrator for daily evaluation runs"""
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    def run_daily_evaluation(self, scenarios: List[str] = None, model: str = None, sample_size: int = None, dry_run: bool = False):
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_date = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().isoformat()
        enabled_scenarios = scenarios or [k for k, v in self.config['scenarios'].items() if v.get('enabled', False)]
        scenario_results = []
        scenarios_passed = 0
        scenarios_failed = 0
        alerts = []
        hillclimb_suggestions = []
        all_test_case_results = []

        # Initialize evaluator (will use API key from .env)
        use_model = model or self.config.get('model') or self.config['global']['default_model']

        if dry_run:
            print(f"[DRY RUN] Would run {len(enabled_scenarios)} scenarios with model {use_model}")
            for scenario in enabled_scenarios:
                print(f"  - {scenario}: prompt={self.config['scenarios'][scenario].get('prompt_version')}")
            return None

        try:
            evaluator = Evaluator(model=use_model)
        except Exception as e:
            print(f"Error initializing evaluator: {e}")
            print("Make sure OPENAI_API_KEY is set in .env file")
            return None

        for scenario in enabled_scenarios:
            print(f"\n{'='*50}")
            print(f"Running scenario: {scenario}")
            print(f"{'='*50}")

            start_time = datetime.now()
            scenario_config = self.config['scenarios'][scenario]
            prompt_version = scenario_config.get('prompt_version', 'v1_zero_shot')
            scenario_sample_size = sample_size or scenario_config.get('sample_size', 20)

            # Get test cases (exclude regression cases)
            all_case_ids = [c for c in evaluator.list_test_cases() if not c.startswith("REG_")]

            # Sample test cases randomly
            import random
            import time
            random.seed(time.time())  # Ensure different random selection each run

            if len(all_case_ids) > scenario_sample_size:
                test_case_ids = random.sample(all_case_ids, scenario_sample_size)
            else:
                test_case_ids = all_case_ids.copy()
                random.shuffle(test_case_ids)  # Shuffle even if not sampling

            print(f"  Prompt: {prompt_version}")
            print(f"  Total available cases: {len(all_case_ids)}")
            print(f"  Sample size: {scenario_sample_size}")
            print(f"  Selected {len(test_case_ids)} cases: {test_case_ids[:5]}{'...' if len(test_case_ids) > 5 else ''}")

            # Run evaluation
            try:
                results = evaluator.evaluate_batch(
                    prompt_id=prompt_version,
                    test_case_ids=test_case_ids
                )

                # Add scenario info to results and save to database
                for r in results:
                    r['scenario'] = scenario
                    # Only save results that have actual predictions (skip errors)
                    if 'prediction' in r and r.get('prediction'):
                        db.save_test_result(
                            run_id=run_id,
                            scenario=scenario,
                            test_case_id=r.get('test_case_id', ''),
                            prompt_id=r.get('prompt_id', ''),
                            ground_truth=r.get('ground_truth', ''),
                            prediction=r.get('prediction', ''),
                            confidence=r.get('confidence', 0.0),
                            correct=r.get('correct', False),
                            llm_output=r.get('llm_output', ''),
                            timestamp=r.get('timestamp', timestamp)
                        )
                    elif 'error' in r:
                        print(f"  Error for {r.get('test_case_id')}: {r.get('error')}")
                all_test_case_results.extend(results)

                # Calculate metrics
                y_true = [r.get("ground_truth") for r in results if "ground_truth" in r and "prediction" in r]
                y_pred = [r.get("prediction") for r in results if "ground_truth" in r and "prediction" in r]

                if y_true and y_pred:
                    calc = MetricsCalculator(y_true, y_pred)
                    calc_metrics = calc.all_metrics()

                    # Convert to our MetricResult format
                    metrics = {
                        'f1': MetricResult('f1', calc_metrics['f1'].value, threshold_min=0.75),
                        'tnr': MetricResult('tnr', calc_metrics['tnr'].value, threshold_min=0.65),
                        'bias': MetricResult('bias', abs(calc_metrics['bias'].value), threshold_max=0.15),
                        'accuracy': MetricResult('accuracy', calc_metrics['accuracy'].value, threshold_min=0.70),
                        'precision': MetricResult('precision', calc_metrics['precision'].value, threshold_min=0.70),
                        'recall': MetricResult('recall', calc_metrics['recall'].value, threshold_min=0.70),
                    }

                    print(f"  Results: F1={metrics['f1'].value:.3f}, TNR={metrics['tnr'].value:.3f}, Bias={metrics['bias'].value:.3f}")
                else:
                    print(f"  Warning: No valid predictions for {scenario}")
                    metrics = {
                        'f1': MetricResult('f1', 0.0, threshold_min=0.75),
                        'tnr': MetricResult('tnr', 0.0, threshold_min=0.65),
                        'bias': MetricResult('bias', 0.0, threshold_max=0.15),
                    }

            except Exception as e:
                print(f"  Error running evaluation: {e}")
                metrics = {
                    'f1': MetricResult('f1', 0.0, threshold_min=0.75),
                    'tnr': MetricResult('tnr', 0.0, threshold_min=0.65),
                    'bias': MetricResult('bias', 0.0, threshold_max=0.15),
                }

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Determine status
            status = HealthStatus.HEALTHY if all(m.status == HealthStatus.HEALTHY for m in metrics.values()) else HealthStatus.WARNING
            if any(m.status == HealthStatus.CRITICAL for m in metrics.values()):
                status = HealthStatus.CRITICAL

            if status != HealthStatus.HEALTHY:
                failed_metrics = [m.name for m in metrics.values() if m.status != HealthStatus.HEALTHY]
                alerts.append(f"{scenario}: {status.value.upper()} - {', '.join(failed_metrics)} below threshold")

            if status == HealthStatus.CRITICAL:
                hillclimb_suggestions.append(f"Improve {scenario}: focus on {', '.join(failed_metrics)}")

            # Save metrics to DB
            for metric in metrics.values():
                db.save_metric(
                    run_id=run_id,
                    scenario=scenario,
                    timestamp=timestamp,
                    metric_name=metric.name,
                    metric_value=metric.value,
                    threshold_min=metric.threshold_min,
                    threshold_max=metric.threshold_max,
                    unit=metric.unit,
                    status=metric.status.value
                )

            scenario_results.append(ScenarioResult(
                scenario=scenario,
                run_id=run_id,
                timestamp=timestamp,
                prompt_version=prompt_version,
                model=use_model,
                metrics=metrics,
                sample_size=len(test_case_ids),
                duration_seconds=duration,
                status=status
            ))

            if status == HealthStatus.HEALTHY:
                scenarios_passed += 1
            else:
                scenarios_failed += 1

        # Overall status
        overall_status = HealthStatus.HEALTHY if scenarios_failed == 0 else (
            HealthStatus.CRITICAL if any(r.status == HealthStatus.CRITICAL for r in scenario_results) else HealthStatus.WARNING
        )

        summary = DailyRunSummary(
            run_id=run_id,
            run_date=run_date,
            timestamp=timestamp,
            scenarios_run=len(enabled_scenarios),
            scenarios_passed=scenarios_passed,
            scenarios_failed=scenarios_failed,
            overall_status=overall_status,
            alerts=alerts,
            hillclimb_suggestions=hillclimb_suggestions
        )

        db.save_daily_run(
            run_id=summary.run_id,
            run_date=summary.run_date,
            timestamp=summary.timestamp,
            scenarios_run=summary.scenarios_run,
            scenarios_passed=summary.scenarios_passed,
            scenarios_failed=summary.scenarios_failed,
            overall_status=summary.overall_status.value,
            alerts=summary.alerts,
            hillclimb_suggestions=summary.hillclimb_suggestions
        )

        # Export to JSON
        out_dir = Path("data/daily_runs")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        with open(out_dir / f"{run_id}_summary.json", "w") as f:
            json.dump(summary.__dict__, f, indent=2, default=str)

        # Save detailed test case results
        with open(out_dir / f"{run_id}_results.json", "w") as f:
            json.dump(all_test_case_results, f, indent=2, default=str)

        print(f"\n{'='*50}")
        print(f"DAILY RUN COMPLETE")
        print(f"{'='*50}")
        print(f"Run ID: {run_id}")
        print(f"Status: {overall_status.value.upper()}")
        print(f"Scenarios: {scenarios_passed}/{len(enabled_scenarios)} passed")
        if alerts:
            print(f"Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"  - {alert}")

        return summary
    def _generate_alerts(self, results: List[ScenarioResult]) -> List[str]:
        return [f"{r.scenario}: {r.status.value.upper()}" for r in results if r.status != HealthStatus.HEALTHY]
    def _generate_hillclimb_suggestions(self, results: List[ScenarioResult]) -> List[str]:

        return [f"Hillclimb {r.scenario}: focus on failed metrics" for r in results if r.status == HealthStatus.CRITICAL]

# Main block for direct execution
if __name__ == "__main__":
    orchestrator = EvaluationOrchestrator()
    summary = orchestrator.run_daily_evaluation()
    print(f"Run complete. Status: {summary.overall_status.value}. Run ID: {summary.run_id}")
