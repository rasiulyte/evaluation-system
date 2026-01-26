
# AI Coding Agent Instructions for `evaluation-system`

## Project Purpose
This system evaluates LLM hallucination detection using labeled test cases, prompt variants, and robust metrics. It is designed for systematic A/B testing, regression validation, and production monitoring of hallucination detection strategies. The project also serves as a learning sandbox for understanding LLM-as-Judge evaluation techniques.

## Architecture Overview
- **Data Layer**: `data/test_cases/ground_truth.json` (train), `regression.json` (held-out). Each case: id, context, response, label, hallucination_span, reasoning, difficulty.
- **Prompt Layer**: `prompts/` holds prompt templates (`v1_zero_shot.txt` ... `v6_calibrated_confidence.txt`) and `prompt_registry.json` for versioning and A/B test metadata.
- **Evaluation Layer**: `src/evaluator.py` loads test cases/prompts, calls OpenAI-compatible API, saves timestamped results.
- **Metrics Layer**: `src/metrics.py` computes 12+ metrics (F1, TNR, bias, Spearman, Pearson, etc.) with intent and interpretation for each.
- **Database Layer**: `src/database.py` provides SQLite/PostgreSQL abstraction for metrics history.
- **Orchestration Layer**: `src/orchestrator.py` runs daily evaluations across multiple scenarios.
- **Analysis Layer**: `src/ab_testing.py`, `monitoring.py`, `regression.py` for A/B tests, drift, and regression.
- **Dashboard Layer**: `src/dashboard_v2.py` (primary) for visualization, slice analysis, and learning guides.

## Key Workflows
- **Run Evaluation**: Use `Evaluator` to run a prompt on test cases, then `MetricsCalculator` for metrics. Save results with timestamps.
- **Run Dashboard**: `python -m streamlit run src/dashboard_v2.py`
- **Add Test Cases**: Add to `ground_truth.json` (FM1-7) or `regression.json` (REG). Never use regression cases for dev.
- **Add Prompts**: Create new `.txt` in `prompts/`, register in `prompt_registry.json`, evaluate, then A/B test.
- **A/B Testing**: Use `ABTester` to compare prompt results. Promote winners by updating production pointer in registry.
- **Regression Testing**: Use `RegressionTester` on held-out set. Fails if F1 < 0.75, TNR < 0.65, or |bias| > 0.15.
- **Monitoring**: Use `DriftMonitor` to compare new results to baseline and detect drift >10%.
- **Daily Runs**: Use `orchestrator.py` to run automated daily evaluations across scenarios.

## Project Conventions
- **Test Case IDs**: `FM{1-7}_{3-digit}` (train), `REG_{3-digit}` (regression)
- **Prompt Files**: `v{N}_{desc}.txt` (e.g., `v6_calibrated_confidence.txt`)
- **Results**: `results_{YYYYMMDD_HHMMSS}_{run_name}.json`
- **Daily Runs**: Stored in `data/daily_runs/`
- **Metrics**: snake_case (e.g., `f1`, `tnr`, `cohens_kappa`, `spearman`)
- **Labeling**: Only label "hallucination" when confident; always specify hallucination_span and reasoning.
- **Error Handling**: Results with errors have an `error` field; v5/v6 prompts fall back to keyword matching if JSON invalid.

## Prompt Versions
| Version | Strategy | Best For |
|---------|----------|----------|
| v1 | Zero-shot | Baseline measurement |
| v2 | Few-shot | Quick iteration |
| v3 | Chain-of-thought | Complex reasoning |
| v4 | Rubric-based | Domain-specific criteria |
| v5 | Structured output | Production JSON parsing |
| v6 | Calibrated confidence | Best correlation metrics (recommended) |

## Integration & Dependencies
- **OpenAI API**: Set `OPENAI_API_KEY` in env. Model defaults to gpt-4o-mini; override in config or constructor.
- **Key Packages**: `numpy`, `scipy`, `scikit-learn`, `pandas`, `plotly`, `streamlit`, `openai`, `pyyaml`.

## Design Principles
- Never use regression cases for dev; only for final validation.
- All results are timestamped for tracking/drift.
- Metrics document both intent and interpretation.
- F1, TNR, Bias are primary go/no-go metrics.
- Failure modes (FM1-7) guide test design; see `FAILURE_MODES.md`.
- A/B test winners by score (F1/TNR/Bias weighted).
- Regression thresholds are conservative for production.
- Drift alerts on >10% metric change from baseline.
- Slice analysis reveals hidden weaknesses in aggregate metrics.

## References
- See `docs/DESIGN.md`, `docs/METRICS.md`, `docs/FAILURE_MODES.md`, `docs/PROMPTING_STRATEGIES.md`, `docs/SLICE_ANALYSIS.md` for methodology.
- Key code: `src/evaluator.py`, `src/metrics.py`, `src/database.py`, `src/orchestrator.py`, `src/dashboard_v2.py`
- Legacy files (deprecated): `src/dashboard.py`, `src/eval_ui.py`

---
**Last Updated**: January 26, 2026
