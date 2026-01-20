# AI Coding Agent Instructions for evaluation-system

## Project Overview

This is a **production-ready LLM evaluation system** for detecting hallucinations in AI-generated content. It's a portfolio project demonstrating expertise in AI evaluation methodology, prompt engineering, statistical testing, and monitoring systems.

**Core Purpose**: Measure and improve hallucination detection accuracy through systematic testing, A/B comparison, and production monitoring.

---

## Architecture & Component Structure

### Core Layers

1. **Data Layer** (`data/test_cases/`):
   - `ground_truth.json`: 80 labeled test cases across 7 failure modes (training)
   - `regression.json`: 20 held-out cases (validation; never used during development)
   - Each case: id, context, response, label, hallucination_span, reasoning, difficulty

2. **Prompt Layer** (`prompts/`):
   - `v1_zero_shot.txt` → `v5_structured_output.txt`: Progressive complexity
   - `prompt_registry.json`: Version tracking, A/B test metadata, production pointer
   - Adding prompts: Create `.txt` file + register in `prompt_registry.json`

3. **Metrics Layer** (`src/metrics.py`):
   - `MetricsCalculator`: All 12+ metrics (F1, TNR, bias, correlation, consistency)
   - `MetricResult` dataclass: name, value, intent, interpretation, target, meets_target
   - Each metric has documented intent ("WHY measure this?") and interpretation guide
   - **Pattern**: All metrics include target threshold and human-readable interpretation

4. **Evaluation Layer** (`src/evaluator.py`):
   - Loads test cases + prompts, calls OpenAI-compatible API, parses responses
   - Returns timestamped results with predictions and confidences
   - Supports multiple runs for consistency measurement
   - **Pattern**: Results always saved with timestamps for tracking

5. **Analysis Layer**:
   - `ab_testing.py`: Compare prompts, statistical significance (McNemar's test)
   - `monitoring.py`: Drift detection, baseline comparison, alert generation
   - `regression.py`: Held-out test validation with conservative thresholds
   - `dashboard.py`: Streamlit visualization of all metrics/results

### Data Flow

```
Test Cases (ground_truth.json)
    ↓
Prompt Registry (v1→v5 + metadata)
    ↓
Evaluator (batch evaluation with streaming)
    ↓
Results (timestamped predictions + confidences)
    ↓
Metrics Calculator (F1, TNR, bias, etc.)
    ↓
A/B Testing (compare prompts)
    ↓
Regression Testing (held-out validation)
    ↓
Monitoring (drift detection)
    ↓
Dashboard (visualization + alerts)
```

---

## Key Development Workflows

### 1. Running an Evaluation

```python
from src.evaluator import Evaluator
from src.metrics import MetricsCalculator

evaluator = Evaluator(model="gpt-4")  # Uses OPENAI_API_KEY env var

# Run on training set (exclude regression cases)
results = evaluator.evaluate_batch(
    prompt_id="v1_zero_shot",
    test_case_ids=[c for c in evaluator.list_test_cases() if not c.startswith("REG_")]
)

# Calculate metrics
calc = MetricsCalculator([r["ground_truth"] for r in results], 
                         [r["prediction"] for r in results])
metrics = calc.all_metrics()

# Save for tracking
evaluator.save_results(results, run_name="v1_baseline")
```

### 2. Adding Test Cases

1. Create case object: `{"id": "FM1_016", "failure_mode": "factual_addition", "context": "...", "response": "...", "label": "grounded", "hallucination_span": null, "reasoning": "...", "difficulty": "hard", ...}`
2. Add to `ground_truth.json` OR `regression.json` (never both; regression is held-out)
3. Ensure ID follows `FM{1-7}_{3-digit}` pattern for training or `REG_{3-digit}` for regression
4. Test with: `evaluator.get_test_case("FM1_016")`

### 3. Adding Prompts

1. Create `prompts/v6_your_strategy.txt` with template using `{context}` and `{response}`
2. Update `prompt_registry.json`: add entry to `prompts[]` array with metadata
3. Run evaluation: `evaluator.evaluate_batch("v6_your_strategy")`
4. A/B test against current production: `ABTester().run_test(results_v5, results_v6, ...)`

### 4. A/B Testing & Production Promotion

```python
from src.ab_testing import ABTester

tester = ABTester()
test_result = tester.run_test(results_v3, results_v4, "v3_cot", "v4_rubric")

if test_result["winner"] == "b":
    tester.set_production_prompt("v4_rubric")  # Promote winner
    tester.update_registry_with_results(test_result)
```

### 5. Regression Testing (Final Validation)

```python
from src.regression import RegressionTester

# Evaluate production prompt on held-out set
regression_results = evaluator.evaluate_batch("v3_chain_of_thought", 
    test_case_ids=[c for c in evaluator.list_test_cases() if c.startswith("REG_")])

tester = RegressionTester()
passed, details = tester.run_regression_test(regression_results)
print(tester.format_report(passed, details))
# Fails if: F1 < 0.75 OR TNR < 0.65 OR |bias| > 0.15
```

### 6. Monitoring Production

```python
from src.monitoring import DriftMonitor

monitor = DriftMonitor(baseline_results_path="data/results/results_baseline.json", 
                       drift_threshold=0.10)
drift = monitor.compare_to_baseline(new_results)
alerts = monitor.get_alerts(drift)  # Human-readable
if monitor.should_rollback(drift):
    print("ALERT: Roll back to previous version")
```

### 7. Dashboard

```bash
# Terminal
streamlit run src/dashboard.py
# Opens http://localhost:8501
```

---

## Project Conventions & Patterns

### Test Case Labeling

- **Conservative labeling**: Only label "hallucination" when confident
- **Hallucination span**: Always specify exact text that's hallucinated (for debugging)
- **Reasoning**: Clear chain of logic for why case is labeled this way
- **Difficulty**: easy/medium/hard based on classification obviousness

### Metrics Interpretation

Every metric has structured interpretation:
```python
result = calc.f1()
print(f"{result.name}: {result.value:.3f}")
print(f"Interpretation: {result.interpretation}")
print(f"Target: {result.target}")
print(f"Meets target: {result.meets_target}")
```

### Production Thresholds (Primary Metrics)

- **F1 ≥ 0.75**: Balanced correctness (50/50 precision/recall importance)
- **TNR ≥ 0.65**: Don't over-flag grounded claims (specificity matters more than recall)
- **|Bias| ≤ 0.15**: Balanced FP/FN (no systematic patterns)

### Failure Modes (7 Categories)

When creating test cases, target specific failure modes:

| Mode | Definition | Risk | Strategy |
|------|-----------|------|----------|
| FM1 | Factual addition (facts not in context) | Medium | Use general knowledge + context facts |
| FM2 | Fabrication (outright false) | Critical | Wrong dates, false attributions |
| FM3 | Subtle distortion (number changes) | High | Inversion, percentage changes |
| FM4 | Valid inference (should NOT flag) | False positive | Syllogisms, logical chains |
| FM5 | Verbatim grounded (easy negative) | Sanity check | Direct quotes, paraphrases |
| FM6 | Fluent hallucination (well-written false) | Deceptive | Confident-sounding false claims |
| FM7 | Partial grounding (mixed) | Mixed signal | Some true + some false |

### Error Handling

- **Results with errors**: Captured in result dict with `"error"` field (don't crash)
- **API retries**: Not implemented; wrap evaluator calls in retry logic if needed
- **JSON parsing**: v5_structured_output gracefully falls back to keyword matching if JSON invalid

### Naming Conventions

- **Test cases**: `FM{1-7}_{3-digit}` or `REG_{3-digit}`
- **Prompts**: `v{N}_{description}.txt` (v1_zero_shot, v2_few_shot, etc.)
- **Results files**: `results_{YYYYMMDD_HHMMSS}_{run_name}.json`
- **Metrics**: snake_case (f1, tnr, cohens_kappa, etc.)

---

## Critical Design Principles

1. **Never use regression cases during development**: Held-out set validates final model only
2. **All results timestamped**: Enables tracking, comparison, drift detection
3. **Metrics document intent**: Not just "what" but "why" measure each metric
4. **Primary metrics for go/no-go**: F1, TNR, Bias decide production readiness
5. **Failure modes guide test design**: Systematic coverage of hallucination types
6. **A/B test winners selected by score**: Weighted comparison of F1/TNR/Bias
7. **Conservative regression thresholds**: Ensure production-grade quality
8. **Drift alerts on metric changes**: Detect 10%+ shifts from baseline

---

## Dependencies & Integration

### External APIs
- **OpenAI-compatible API**: Configured via OPENAI_API_KEY env var
- **Model selection**: Set in config or constructor (default: gpt-4)
- **Alternative endpoints**: Pass `base_url` to Evaluator for local models

### Key Packages
- `numpy`, `scipy`: Statistical calculations
- `scikit-learn`: Metrics (accuracy, precision, recall, F1, kappa)
- `pandas`: Data manipulation
- `plotly`: Dashboard charts
- `streamlit`: Web dashboard
- `openai`: LLM API calls
- `pyyaml`: Configuration

---

## Getting Started for AI Agents

1. **Understand the goal**: Detect hallucinations via LLM evaluation
2. **Review structure**: 100 test cases → 5 prompts → metrics → A/B test → regression → monitoring
3. **Look at docs**: DESIGN.md (methodology), METRICS.md (interpretation), FAILURE_MODES.md (cases)
4. **Start simple**: Run v1_zero_shot baseline on training set, calculate metrics
5. **Iterate methodically**: A/B test new prompts, validate on regression set, monitor drift

---

**Last Updated**: January 20, 2026  
**Status**: Production-ready (all systems implemented)
