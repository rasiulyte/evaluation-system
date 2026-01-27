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
import logging

# Set up file logging for debugging
log_file = Path(__file__).parent.parent / "logs" / "orchestrator.log"
log_file.parent.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(log_file),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    total_tokens: int = 0
    total_cost_usd: float = 0.0

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
    def __init__(self, config_path: str = None):
        # Resolve config path relative to project root
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "settings.yaml"
        self.config = self._load_config(config_path)
    def _load_config(self, config_path):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    def run_daily_evaluation(self, scenarios: List[str] = None, model: str = None, sample_size: int = None, dry_run: bool = False):
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"=== Starting evaluation run: {run_id} ===")
        run_date = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().isoformat()
        enabled_scenarios = scenarios or [k for k, v in self.config['scenarios'].items() if v.get('enabled', False)]
        scenario_results = []
        scenarios_passed = 0
        scenarios_failed = 0
        alerts = []
        hillclimb_suggestions = []
        all_test_case_results = []
        total_tokens = 0
        total_cost_usd = 0.0

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
                saved_count = 0
                skipped_count = 0
                error_count = 0
                for r in results:
                    r['scenario'] = scenario
                    pred = r.get('prediction')
                    # Save all results that have a prediction key (including 'unknown')
                    if 'prediction' in r and pred not in [None, '']:
                        try:
                            db.save_test_result(
                                run_id=run_id,
                                scenario=scenario,
                                test_case_id=r.get('test_case_id', ''),
                                prompt_id=r.get('prompt_id', ''),
                                ground_truth=r.get('ground_truth', ''),
                                prediction=pred,
                                confidence=float(r.get('confidence', 0.0)),
                                correct=bool(r.get('correct', False)),
                                llm_output=r.get('llm_output', ''),
                                timestamp=r.get('timestamp', timestamp)
                            )
                            saved_count += 1
                        except Exception as save_err:
                            error_count += 1
                            print(f"  DB Error for {r.get('test_case_id')}: {save_err}")
                    elif 'error' in r:
                        error_count += 1
                        print(f"  API Error for {r.get('test_case_id')}: {r.get('error')}")
                    else:
                        skipped_count += 1
                        print(f"  Skipped {r.get('test_case_id')}: prediction='{pred}'")
                print(f"  Results: {saved_count} saved, {skipped_count} skipped, {error_count} errors")

                # Accumulate token usage and cost
                for r in results:
                    total_tokens += r.get('total_tokens', 0)
                    total_cost_usd += r.get('cost_usd', 0.0)

                all_test_case_results.extend(results)

                # Calculate metrics - filter to only valid predictions
                valid_labels = {"hallucination", "grounded"}
                valid_results = [
                    r for r in results
                    if "ground_truth" in r
                    and "prediction" in r
                    and r.get("prediction") in valid_labels
                    and r.get("ground_truth") in valid_labels
                ]

                y_true = [r.get("ground_truth") for r in valid_results]
                y_pred = [r.get("prediction") for r in valid_results]

                # Log filtering info
                total_results = len([r for r in results if "ground_truth" in r and "prediction" in r])
                filtered_count = total_results - len(valid_results)
                if filtered_count > 0:
                    print(f"  Filtered {filtered_count} invalid predictions (unknown/empty)")

                # Extract confidence scores if available
                # Note: v6 prompt outputs "confidence in being grounded" but MAE/RMSE
                # expects "probability of hallucination". We need to transform the scores.
                # For v6 prompts: prob_hallucination = 1 - confidence_in_grounded
                y_conf = []
                for r in valid_results:
                    conf = r.get("confidence")
                    if conf is not None:
                        try:
                            conf_val = float(conf)
                            # Transform confidence based on prompt version
                            # v5 and v6 output confidence in "grounded" so we need to invert
                            # for MAE/RMSE which expect probability of hallucination
                            if prompt_version.startswith("v5_") or prompt_version.startswith("v6_"):
                                conf_val = 1.0 - conf_val
                            y_conf.append(conf_val)
                        except (ValueError, TypeError):
                            y_conf.append(None)
                    else:
                        y_conf.append(None)

                # Only use confidence if all values are present and have variance
                if y_conf and all(c is not None for c in y_conf):
                    # Check if confidence values have variance (not all the same)
                    unique_conf = set(y_conf)
                    if len(unique_conf) > 1:
                        y_conf_valid = y_conf
                    else:
                        print(f"  Warning: All confidence scores are identical ({y_conf[0]}), correlation metrics will be skipped")
                        y_conf_valid = None
                else:
                    y_conf_valid = None

                if y_true and y_pred:
                    calc = MetricsCalculator(y_true, y_pred, y_conf=y_conf_valid)
                    calc_metrics = calc.all_metrics()

                    # Convert to our MetricResult format - include ALL available metrics
                    metrics = {
                        # Classification metrics
                        'f1': MetricResult('f1', calc_metrics['f1'].value, threshold_min=0.75),
                        'tnr': MetricResult('tnr', calc_metrics['tnr'].value, threshold_min=0.65),
                        'accuracy': MetricResult('accuracy', calc_metrics['accuracy'].value, threshold_min=0.70),
                        'precision': MetricResult('precision', calc_metrics['precision'].value, threshold_min=0.70),
                        'recall': MetricResult('recall', calc_metrics['recall'].value, threshold_min=0.70),
                        # Agreement metrics
                        'cohens_kappa': MetricResult('cohens_kappa', calc_metrics['cohens_kappa'].value, threshold_min=0.60),
                        # Error metrics
                        'bias': MetricResult('bias', abs(calc_metrics['bias'].value), threshold_max=0.15),
                    }

                    # Add correlation metrics if confidence scores were available
                    if 'spearman' in calc_metrics:
                        metrics['spearman'] = MetricResult('spearman', calc_metrics['spearman'].value, threshold_min=0.60)
                    if 'pearson' in calc_metrics:
                        metrics['pearson'] = MetricResult('pearson', calc_metrics['pearson'].value, threshold_min=0.60)
                    if 'kendalls_tau' in calc_metrics:
                        metrics['kendalls_tau'] = MetricResult('kendalls_tau', calc_metrics['kendalls_tau'].value, threshold_min=0.50)
                    if 'mae' in calc_metrics:
                        metrics['mae'] = MetricResult('mae', calc_metrics['mae'].value, threshold_max=0.20)
                    if 'rmse' in calc_metrics:
                        metrics['rmse'] = MetricResult('rmse', calc_metrics['rmse'].value, threshold_max=0.25)

                    print(f"  Results: F1={metrics['f1'].value:.3f}, TNR={metrics['tnr'].value:.3f}, Kappa={metrics['cohens_kappa'].value:.3f}")
                    if y_conf_valid:
                        print(f"  Correlation metrics available (confidence scores found)")
                else:
                    print(f"  Warning: No valid predictions for {scenario}")
                    metrics = {
                        'f1': MetricResult('f1', 0.0, threshold_min=0.75),
                        'tnr': MetricResult('tnr', 0.0, threshold_min=0.65),
                        'accuracy': MetricResult('accuracy', 0.0, threshold_min=0.70),
                        'precision': MetricResult('precision', 0.0, threshold_min=0.70),
                        'recall': MetricResult('recall', 0.0, threshold_min=0.70),
                        'cohens_kappa': MetricResult('cohens_kappa', 0.0, threshold_min=0.60),
                        'bias': MetricResult('bias', 0.0, threshold_max=0.15),
                    }

            except Exception as e:
                import traceback
                error_msg = f"ERROR in {scenario}: {type(e).__name__}: {e}"
                print(f"  {error_msg}")
                traceback.print_exc()
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                metrics = {
                    'f1': MetricResult('f1', 0.0, threshold_min=0.75),
                    'tnr': MetricResult('tnr', 0.0, threshold_min=0.65),
                    'accuracy': MetricResult('accuracy', 0.0, threshold_min=0.70),
                    'precision': MetricResult('precision', 0.0, threshold_min=0.70),
                    'recall': MetricResult('recall', 0.0, threshold_min=0.70),
                    'cohens_kappa': MetricResult('cohens_kappa', 0.0, threshold_min=0.60),
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

            # Save metrics to DB (convert numpy types to Python floats)
            for metric in metrics.values():
                db.save_metric(
                    run_id=run_id,
                    scenario=scenario,
                    timestamp=timestamp,
                    metric_name=metric.name,
                    metric_value=float(metric.value),
                    threshold_min=float(metric.threshold_min) if metric.threshold_min is not None else None,
                    threshold_max=float(metric.threshold_max) if metric.threshold_max is not None else None,
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
            hillclimb_suggestions=hillclimb_suggestions,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd
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
            hillclimb_suggestions=summary.hillclimb_suggestions,
            total_tokens=summary.total_tokens,
            total_cost_usd=summary.total_cost_usd
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
        print(f"Tokens: {total_tokens:,} | Cost: ${total_cost_usd:.4f}")
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
