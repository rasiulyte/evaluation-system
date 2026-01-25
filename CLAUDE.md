# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM hallucination detection evaluation system. Measures how well prompts can classify AI-generated text as "hallucination" or "grounded" against labeled test cases.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/test_metrics.py -v

# Run dashboard
python -m streamlit run src/dashboard.py

# Verify setup
python verify_setup.py
```

## Architecture

**Data Flow**: Test cases → Evaluator → LLM API → Response parser → MetricsCalculator → Results JSON

**Core Components**:
- `src/evaluator.py`: Loads test cases and prompts, calls OpenAI API, parses responses, saves timestamped results
- `src/metrics.py`: Computes 12+ metrics (F1, TNR, bias, precision, recall, etc.) with `MetricResult` dataclass containing value, intent, interpretation, and target
- `src/ab_testing.py`: Compares prompt performance with statistical significance testing
- `src/monitoring.py`: Drift detection comparing current results to baseline (alerts on >10% change)
- `src/regression.py`: Final validation on held-out set with production thresholds

**Data Structure**:
- `data/test_cases/ground_truth.json`: 80 training cases (IDs: `FM{1-7}_{3-digit}`)
- `data/test_cases/regression.json`: 20 held-out cases (IDs: `REG_{3-digit}`) - never use for development
- `prompts/v{N}_{desc}.txt`: Prompt templates (v1=zero-shot through v5=structured output)
- `data/results/results_{timestamp}_{run_name}.json`: Evaluation outputs

## Key Patterns

**Response Parsing**: `evaluator._parse_response()` handles two modes:
- v5 prompts: Parse JSON with `classification` and `confidence` fields
- All others: Keyword matching for "hallucination"/"grounded"

**Production Thresholds**: F1 ≥ 0.75, TNR ≥ 0.65, |bias| ≤ 0.15

**Test Case Labels**: Binary classification - "hallucination" or "grounded"

## Configuration

`config/settings.yaml` controls:
- Model selection (`model: gpt-4`)
- API settings (key via `${OPENAI_API_KEY}` or `.env` file)
- Regression thresholds, drift threshold (10%), A/B testing alpha (0.05)

## Failure Modes (FM1-7)

Test cases are categorized by failure mode type:
- FM1: Factual Addition (true facts not in context)
- FM2: Fabrication (outright false claims)
- FM3: Subtle Distortion (small changes to facts)
- FM4: Valid Inference (logical inferences)
- FM5: Verbatim Grounded (direct quotes)
- FM6: Fluent Hallucination (well-written but wrong)
- FM7: Partial Grounding (mixed grounded/hallucinated)
