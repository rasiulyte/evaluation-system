# Hallucination Detection Evaluation System - Design Document

## Problem Definition

Large Language Models (LLMs) generate text that can be fluent and convincing while containing factual inaccuracies ("hallucinations"). The challenge is distinguishing between:
- **Grounded responses**: Claims supported by context or general knowledge
- **Hallucinations**: False or unsupported claims presented as facts

This evaluation system measures an LLM's ability to generate only grounded claims within a given context.

## Quality Dimensions

1. **Correctness**: Does the LLM accurately classify grounded vs. hallucinated claims?
2. **Reliability**: Is performance consistent across runs and variations?
3. **Robustness**: Does it handle edge cases and adversarial inputs?
4. **Fairness**: Are false negatives (missed hallucinations) and false positives (incorrect flags) balanced?

## Failure Modes

The system explicitly tests seven failure modes:

| Mode | Definition | Risk | Count |
|------|-----------|------|-------|
| FM1: Factual Addition | LLM adds true facts not in context | Hallucination | 15 |
| FM2: Fabrication | LLM creates false claims | Critical hallucination | 15 |
| FM3: Subtle Distortion | LLM modifies facts slightly | Deceptive hallucination | 15 |
| FM4: Valid Inference | LLM makes logical inferences | False positive risk | 15 |
| FM5: Verbatim Grounded | LLM copies context directly | Baseline negative | 15 |
| FM6: Fluent Hallucination | Well-written but false claim | Confidence trap | 15 |
| FM7: Partial Grounding | Some claims grounded, some not | Mixed signal | 10 |

**Total: 100 cases** (80 training + 20 held-out regression)

## Ground Truth Strategy

Each case is hand-labeled with:
- **Label**: "hallucination" or "grounded"
- **Hallucination span**: Exact text that is hallucinated (if applicable)
- **Reasoning**: Why this label, with explicit reasoning chain
- **Difficulty**: easy/medium/hard (based on how obvious the classification is)

Labels are conservative: only claim "hallucination" when confident, erring on side of "grounded" for ambiguous cases.

## Evaluation Protocol

### Single Run
1. Load test case
2. Present context + response to LLM via prompt
3. Parse LLM output (binary classification or confidence score)
4. Compare to ground truth label
5. Record prediction + metadata

### Multiple Runs
Repeat single run 3+ times to measure consistency (variance, Cohen's kappa across runs)

### Test Sets
- **Training**: 80 cases used to evaluate and iterate on prompts
- **Regression**: 20 held-out cases used only for final validation (never during development)

## Success Criteria

### Minimum Requirements
- **F1 Score**: ≥ 0.75 (balanced precision/recall)
- **TNR (Specificity)**: ≥ 0.65 (catch at least 65% of real hallucinations)
- **Bias**: |precision - recall| ≤ 0.15 (no extreme FP/FN imbalance)

### Consistency
- Cohen's Kappa ≥ 0.70 across three runs (substantial agreement)
- Variance in F1 across runs ≤ 0.05

### Production Readiness
- Prompt selection via A/B testing on training set
- Regression tests pass on held-out set
- Monitoring/drift detection in place
- Documented failure modes and known limitations

## Evaluation Metrics

See [METRICS.md](METRICS.md) for comprehensive metric definitions and interpretations.

## Prompting Strategies

Six progressively sophisticated approaches:

1. **Zero-shot (V1)**: Minimal context, baseline performance
2. **Few-shot (V2)**: 3-5 labeled examples including edge cases
3. **Chain-of-Thought (V3)**: Step-by-step reasoning required
4. **Rubric-based (V4)**: Explicit evaluation criteria provided
5. **Structured Output (V5)**: JSON format for easier parsing
6. **Calibrated Confidence (V6)**: Explicit confidence calibration for better correlation metrics

See [PROMPTING_STRATEGIES.md](PROMPTING_STRATEGIES.md) for details.

## Architecture

```
Test Cases (ground truth)
    ↓
Prompt Registry (v1-v6)
    ↓
Orchestrator (daily evaluation runner)
    ↓
Evaluator (calls LLM with prompt on each test case)
    ↓
Results (predictions + metadata)
    ↓
Database (SQLite/PostgreSQL)
    ↓
+---+---+---+---+
|   |   |   |   |
Metrics  A/B Testing  Regression  Monitoring
    ↓
Dashboard (visualization + slice analysis + alerts)
```

## Known Limitations

- LLM hallucination evaluation is inherently subjective; ground truth labels reflect one perspective
- System measures only binary classification (grounded/hallucinated), not severity gradients
- Evaluation is context-dependent; prompts must be adapted for different domains
- Model outputs may vary significantly based on temperature, sampling method, and version
