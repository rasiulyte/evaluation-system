# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM hallucination detection evaluation system. Measures how well prompts can classify AI-generated text as "hallucination" or "grounded" against labeled test cases. Includes a learning-focused dashboard for exploring LLM-as-Judge evaluation techniques.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/test_metrics.py -v

# Run dashboard (primary)
python -m streamlit run src/dashboard_v2.py

# Run daily evaluation orchestrator
python src/orchestrator.py

# Verify setup
python verify_setup.py
```

## Architecture

**Data Flow**: Test cases -> Evaluator -> LLM API -> Response parser -> MetricsCalculator -> Database -> Dashboard

```
Test Cases (ground truth)
    |
Prompt Registry (v1-v6)
    |
Evaluator (calls LLM with prompt)
    |
Results (predictions + metadata)
    |
Database (SQLite/PostgreSQL)
    |
+---+---+---+---+
|   |   |   |   |
Metrics  A/B Testing  Regression  Monitoring
    |
Dashboard (visualization + alerts)
```

**Core Components**:
- `src/evaluator.py`: Loads test cases and prompts, calls OpenAI API, parses responses, saves timestamped results
- `src/metrics.py`: Computes 12+ metrics (F1, TNR, bias, precision, recall, Spearman, Pearson, etc.) with `MetricResult` dataclass
- `src/database.py`: Database abstraction layer supporting SQLite (local) and PostgreSQL (production)
- `src/orchestrator.py`: Daily evaluation runner executing multiple scenarios with configurable sample sizes
- `src/ab_testing.py`: Compares prompt performance with statistical significance testing
- `src/monitoring.py`: Drift detection comparing current results to baseline (alerts on >10% change)
- `src/regression.py`: Final validation on held-out set with production thresholds
- `src/dashboard_v2.py`: Primary Streamlit dashboard with metrics visualization, slice analysis, and learning guides

**Data Structure**:
- `data/test_cases/ground_truth.json`: 80 training cases (IDs: `FM{1-7}_{3-digit}`)
- `data/test_cases/regression.json`: 20 held-out cases (IDs: `REG_{3-digit}`) - never use for development
- `prompts/v{N}_{desc}.txt`: Prompt templates (v1=zero-shot through v6=calibrated confidence)
- `data/results/`: Evaluation outputs with timestamps
- `data/daily_runs/`: Orchestrator-generated daily evaluation results
- `data/metrics.db`: SQLite database for metrics history

## Key Patterns

**Response Parsing**: `evaluator._parse_response()` handles two modes:
- v5/v6 prompts: Parse JSON with `classification` and `confidence` fields
- v1-v4: Keyword matching for "hallucination"/"grounded"

**Production Thresholds**: F1 >= 0.75, TNR >= 0.65, |bias| <= 0.15

**Test Case Labels**: Binary classification - "hallucination" or "grounded"

**Slice Analysis**: Metrics can be broken down by failure mode, difficulty, and label to identify weaknesses

## Configuration

`config/settings.yaml` controls:
- Model selection (`model: gpt-4o-mini`)
- Prompt version (`prompt_version: v6_calibrated_confidence`)
- API settings (key via `${OPENAI_API_KEY}` or `.env` file)
- Evaluation scenarios (hallucination_detection, factual_accuracy, etc.)
- Regression thresholds, drift threshold (10%), A/B testing alpha (0.05)

## Prompt Versions (v1-v6)

| Version | Strategy | Best For |
|---------|----------|----------|
| v1 | Zero-shot | Baseline measurement |
| v2 | Few-shot | Quick iteration |
| v3 | Chain-of-thought | Complex reasoning |
| v4 | Rubric-based | Domain-specific criteria |
| v5 | Structured output | Production JSON parsing |
| v6 | Calibrated confidence | Best correlation metrics |

## Failure Modes (FM1-7)

Test cases are categorized by failure mode type:
- FM1: Factual Addition (true facts not in context)
- FM2: Fabrication (outright false claims)
- FM3: Subtle Distortion (small changes to facts)
- FM4: Valid Inference (logical inferences)
- FM5: Verbatim Grounded (direct quotes)
- FM6: Fluent Hallucination (well-written but wrong)
- FM7: Partial Grounding (mixed grounded/hallucinated)

## Dashboard Pages

- **Getting Started**: Evaluation workflow and quick start guide
- **Failure Modes**: Detailed explanations of FM1-FM7
- **Prompt Lab**: Compare and test different prompt strategies
- **Understanding Metrics**: Comprehensive metrics guide with mental models
- **Metrics Overview**: Current performance metrics
- **Slice Analysis**: Performance breakdown by failure mode, difficulty, label
- **Trends**: Historical metrics over time
- **Compare Runs**: Side-by-side run comparison
- **Run History**: Detailed view of past evaluations
- **Test Cases**: Browse ground truth test cases
- **Run Evaluation**: Execute new evaluation runs
