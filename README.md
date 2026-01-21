# Hallucination Detection Evaluation System

A production-ready simplified LLM evaluation system for detecting hallucinations in AI-generated content. This project demonstrates  evaluation methodology, prompt engineering, and AI system monitoring.

## Project Goals

- **Measure hallucination detection performance** across multiple prompting strategies
- **Identify and test failure modes** systematically
- **Compare prompt effectiveness** through A/B testing
- **Monitor production metrics** with drift detection
- **Ensure model reliability** through regression testing on held-out data

## Key Features

✅ **100 hand-labeled test cases** across 7 failure modes  
✅ **5 prompting strategies** (v1: zero-shot to v5: structured output)  
✅ **12+ evaluation metrics** (accuracy, F1, TNR, bias, consistency, correlation)  
✅ **A/B testing framework** with statistical significance  
✅ **Drift monitoring** for production safety  
✅ **Regression testing** on held-out set (never used in development)  
✅ **Streamlit dashboard** for visualization  
✅ **Comprehensive documentation** of design and methodology  

## Project Structure

```
evaluation-system/
├── docs/
│   ├── DESIGN.md                 # Architecture & methodology
│   ├── METRICS.md                # All metrics with interpretation
│   ├── FAILURE_MODES.md          # 7 failure modes explained
│   └── PROMPTING_STRATEGIES.md   # 5 strategies compared
├── data/
│   ├── test_cases/
│   │   ├── ground_truth.json     # 80 training cases
│   │   └── regression.json       # 20 held-out cases
│   └── results/                  # Timestamped evaluation runs
├── prompts/
│   ├── v1_zero_shot.txt          # Baseline
│   ├── v2_few_shot.txt           # With examples
│   ├── v3_chain_of_thought.txt   # Step-by-step reasoning
│   ├── v4_rubric_based.txt       # Explicit criteria
│   ├── v5_structured_output.txt  # JSON format
│   └── prompt_registry.json      # Version tracking
├── src/
│   ├── metrics.py                # All metric calculations
│   ├── evaluator.py              # LLM evaluation engine
│   ├── ab_testing.py             # A/B test framework
│   ├── monitoring.py             # Drift detection
│   ├── regression.py             # Regression testing
│   └── dashboard.py              # Streamlit visualization
├── tests/
│   └── test_metrics.py           # Unit tests
├── config/
│   └── settings.yaml             # Configuration
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key (Secure Method)

**Option A: Using `.env` file (Recommended - Most Secure)**

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your actual API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. The code will automatically load from `.env` - **no manual setup needed**
4. **IMPORTANT**: Never commit `.env` to git (it's in `.gitignore`)

**Option B: Using Environment Variable (Less Secure)**

```bash
# PowerShell
$env:OPENAI_API_KEY = "sk-your-api-key"

# Bash/Linux
export OPENAI_API_KEY="sk-your-api-key"
```

### 3. Run Evaluation

```python
from src.evaluator import Evaluator
from src.metrics import MetricsCalculator

# Initialize evaluator (automatically loads API key from .env)
evaluator = Evaluator(model="gpt-4")

# Run on training set with v1 prompt
results = evaluator.evaluate_batch(
    prompt_id="v1_zero_shot",
    test_case_ids=[case for case in evaluator.list_test_cases() if not case.startswith("REG_")]
)

# Calculate metrics
y_true = [r["ground_truth"] for r in results]
y_pred = [r["prediction"] for r in results]

calc = MetricsCalculator(y_true, y_pred)
metrics = calc.all_metrics()

for name, metric in metrics.items():
    print(f"{name}: {metric.value:.3f} - {metric.interpretation}")

# Check production readiness
is_ready, details = calc.production_ready()
print(f"Production ready: {is_ready}")

# Save results
evaluator.save_results(results, run_name="v1_baseline")
```

### 4. View Dashboard

```bash
# Using Python module (recommended)
python -m streamlit run src/dashboard.py

# Or with full Python path
C:\Users\rasar\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run src/dashboard.py
```

Navigate to `http://localhost:8501`

### Optional: Set Up PowerShell Aliases (Recommended)

To make commands shorter and easier to remember:

```powershell
# Run this in PowerShell (in the project directory):
. .\profile.ps1

# Now you can use:
python -m streamlit run src/dashboard.py  # Shorter: just "python"
Start-Dashboard                            # Dashboard shortcut
Run-Tests                                  # Test shortcut
Evaluate-Baseline                          # Evaluation helper
```

## Evaluation Workflow

### Phase 1: Prompt Comparison (Training Set)

1. **Baseline** (v1): Establish zero-shot performance
2. **Few-shot** (v2): Add examples if baseline < 0.70
3. **Chain-of-thought** (v3): Add reasoning if few-shot insufficient
4. **Rubric-based** (v4): Explicit criteria
5. **Structured Output** (v5): JSON parsing

### Phase 2: A/B Testing

```python
from src.ab_testing import ABTester

tester = ABTester()

# Run both prompts on training set
results_v2 = evaluator.evaluate_batch("v2_few_shot", ...)
results_v3 = evaluator.evaluate_batch("v3_chain_of_thought", ...)

# A/B test
test_result = tester.run_test(results_v2, results_v3, "v2_few_shot", "v3_chain_of_thought")
print(f"Winner: {test_result['winner']}")
print(f"F1 improvement: {test_result['summary']['f1_improvement']:.1%}")

# Set winner as production
tester.set_production_prompt("v3_chain_of_thought")
tester.update_registry_with_results(test_result)
```

### Phase 3: Regression Testing (Held-out Set)

```python
from src.regression import RegressionTester

tester = RegressionTester()

# Evaluate on regression set
regression_results = evaluator.evaluate_batch(
    prompt_id="v3_chain_of_thought",  # Production prompt
    test_case_ids=[c for c in evaluator.list_test_cases() if c.startswith("REG_")]
)

# Check thresholds
passed, details = tester.run_regression_test(regression_results)

if passed:
    print("✓ Ready for production!")
else:
    print(tester.format_report(passed, details))
```

### Phase 4: Monitoring & Drift Detection

```python
from src.monitoring import DriftMonitor

monitor = DriftMonitor(baseline_results_path="data/results/results_baseline.json")

# Check current run for drift
drift_analysis = monitor.compare_to_baseline(current_results)

print(f"Health: {drift_analysis['overall_health']}")

for alert in monitor.get_alerts(drift_analysis):
    print(alert)

if monitor.should_rollback(drift_analysis):
    print("⚠️ Recommend rollback!")
```

## Core Metrics

### Primary (Production Threshold)
- **F1**: ≥ 0.75 (balanced correctness)
- **TNR**: ≥ 0.65 (specificity; avoid over-flagging)
- **Bias**: |bias| ≤ 0.15 (balanced FP/FN)

### Secondary (Diagnostics)
- **Precision**: When we flag hallucinations, are we right?
- **Recall**: Do we catch all hallucinations?
- **Accuracy**: Overall correctness
- **Cohen's Kappa**: Consistency across runs

### Tertiary (Optional)
- **Spearman/Pearson**: Correlation (if scores provided)
- **MAE/RMSE**: Calibration quality
- **Variance**: Consistency across runs
- **Self-consistency**: Per-case consistency

See [docs/METRICS.md](docs/METRICS.md) for complete interpretation guide.

## Test Cases: 7 Failure Modes

| Mode | Definition | Cases | Difficulty |
|------|-----------|-------|-----------|
| FM1: Factual Addition | True facts not in context | 15 | Easy-Hard |
| FM2: Fabrication | Outright false claims | 15 | Easy-Medium |
| FM3: Subtle Distortion | Small changes to facts | 15 | Medium-Hard |
| FM4: Valid Inference | Logical inferences | 15 | Easy-Hard |
| FM5: Verbatim Grounded | Direct quotes | 15 | Easy-Hard |
| FM6: Fluent Hallucination | Well-written but wrong | 15 | Hard |
| FM7: Partial Grounding | Mixed grounded/hallucinated | 10 | Hard |

See [docs/FAILURE_MODES.md](docs/FAILURE_MODES.md) for detailed analysis.

## Prompting Strategies

| Strategy | Token Cost | Expected F1 | Best For |
|----------|-----------|------------|----------|
| V1: Zero-shot | 1x | 0.55 | Baseline |
| V2: Few-shot | 1.3x | 0.70 | Quick iteration |
| V3: Chain-of-thought | 1.8x | 0.75 | Complex reasoning |
| V4: Rubric-based | 1.6x | 0.75 | Domain-specific |
| V5: Structured | 2.0x | 0.80 | Production systems |

See [docs/PROMPTING_STRATEGIES.md](docs/PROMPTING_STRATEGIES.md) for pros/cons.

## Production Checklist

- [ ] F1 ≥ 0.75 on training set
- [ ] TNR ≥ 0.65 (don't over-flag)
- [ ] |Bias| ≤ 0.15 (balanced)
- [ ] Regression test passed on held-out set
- [ ] A/B test winner selected
- [ ] Baseline metrics recorded
- [ ] Drift thresholds configured
- [ ] Dashboard monitoring active
- [ ] Error handling implemented
- [ ] Logging/alerts configured

## Documentation

- [DESIGN.md](docs/DESIGN.md) - Complete evaluation methodology
- [METRICS.md](docs/METRICS.md) - Every metric explained
- [FAILURE_MODES.md](docs/FAILURE_MODES.md) - Test case catalog
- [PROMPTING_STRATEGIES.md](docs/PROMPTING_STRATEGIES.md) - Strategy comparison

## Running Tests

```bash
python -m pytest tests/test_metrics.py -v
```

## API Reference

### Evaluator

```python
evaluator = Evaluator(model="gpt-4", api_key="...")
results = evaluator.evaluate_batch("v1_zero_shot")
evaluator.save_results(results, run_name="baseline")
```

### Metrics Calculator

```python
calc = MetricsCalculator(y_true, y_pred)
f1 = calc.f1()           # Individual metric
metrics = calc.all_metrics()  # All metrics
is_ready, details = calc.production_ready()  # Production check
```

### A/B Testing

```python
tester = ABTester()
test_result = tester.run_test(results_a, results_b, "v2", "v3")
tester.set_production_prompt("v3")
```

### Monitoring

```python
monitor = DriftMonitor(baseline_results_path="...")
drift = monitor.compare_to_baseline(current_results)
alerts = monitor.get_alerts(drift)
should_rollback = monitor.should_rollback(drift)
```
### Dashboard
https://evaluation-system.streamlit.app/

### Regression Testing

```python
tester = RegressionTester()
passed, details = tester.run_regression_test(regression_results)
report = tester.format_report(passed, details)
```

## Contributing

When adding new features:
1. Update test cases if adding failure modes
2. Document new metrics in METRICS.md
3. Add unit tests in tests/
4. Update this README

## License

Portfolio project - Open for review and discussion

## Questions?

See the documentation files for detailed explanations. Each module includes comprehensive docstrings.

---

**Last Updated**: January 2026  
**Status**: Production-ready for demonstration purposes
