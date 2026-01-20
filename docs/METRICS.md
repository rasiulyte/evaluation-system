# Metrics Reference Guide

## Overview

Each metric has:
- **Formula**: Mathematical definition
- **Intent**: Why we measure it
- **Interpretation**: What different values mean
- **Target**: Ideal range or value
- **When to Use**: What scenario calls for this metric

---

## Agreement Metrics

### Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Intent**: Overall correctness across all cases
- **Interpretation**:
  - 0.90+ : Excellent classification
  - 0.75-0.90 : Good, production-ready
  - 0.60-0.75 : Acceptable but needs improvement
  - <0.60 : Poor, needs redesign
- **Target**: ≥ 0.75
- **When to Use**: Baseline health check; quick comparison between prompts

### Cohen's Kappa
- **Formula**: (p_o - p_e) / (1 - p_e), where p_o = observed agreement, p_e = expected by chance
- **Intent**: Agreement between runs, correcting for chance
- **Interpretation**:
  - 0.81+ : Almost perfect consistency
  - 0.61-0.80 : Substantial consistency
  - 0.41-0.60 : Moderate consistency
  - 0.21-0.40 : Fair consistency
  - <0.20 : Poor consistency
- **Target**: ≥ 0.70 across 3+ runs
- **When to Use**: Measure reproducibility; is this prompt stable?

---

## Classification Metrics

### Precision
- **Formula**: TP / (TP + FP)
- **Intent**: When we predict "hallucination", how often are we correct?
- **Interpretation**:
  - 0.90+ : Very few false hallucination alarms
  - 0.75-0.90 : Acceptable false alarm rate
  - <0.75 : Users will distrust hallucination warnings
- **Target**: ≥ 0.75
- **When to Use**: Optimize if false positives (incorrectly flagging grounded claims) are costly

### Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Intent**: Of actual hallucinations, how many do we catch?
- **Interpretation**:
  - 0.90+ : Catch almost all hallucinations
  - 0.75-0.90 : Acceptable miss rate
  - <0.75 : Too many undetected hallucinations
- **Target**: ≥ 0.75
- **When to Use**: Optimize if false negatives (missing hallucinations) are dangerous

### F1 Score
- **Formula**: 2 * (Precision × Recall) / (Precision + Recall)
- **Intent**: Harmonic mean of precision and recall; balanced correctness
- **Interpretation**:
  - 0.85+ : Excellent balance
  - 0.75-0.85 : Good, production-ready
  - <0.75 : Needs improvement
- **Target**: ≥ 0.75 (minimum threshold)
- **When to Use**: Primary metric; use when both FP and FN costs are similar

### TNR (Specificity)
- **Formula**: TN / (TN + FP)
- **Intent**: Of grounded claims, how many do we correctly accept?
- **Interpretation**:
  - 0.90+ : Rarely incorrectly flag grounded claims
  - 0.75-0.90 : Acceptable false alarm rate
  - 0.65-0.75 : Marginal; some false alarms
  - <0.65 : Too aggressive; blocks good content
- **Target**: ≥ 0.65 (minimum threshold)
- **When to Use**: Ensure system doesn't over-flag grounded content as hallucinations

---

## Correlation Metrics

### Spearman Correlation
- **Formula**: Ranked correlation between two orderings
- **Intent**: Measure monotonic relationship when model outputs confidence scores
- **Interpretation**:
  - 0.85+ : Strong monotonic relationship
  - 0.60-0.85 : Moderate relationship
  - <0.60 : Weak relationship
- **Target**: ≥ 0.70
- **When to Use**: When LLM outputs confidence scores; check if confidence correlates with correctness

### Pearson Correlation
- **Formula**: Covariance(X, Y) / (σ_X × σ_Y)
- **Intent**: Linear relationship between variables
- **Interpretation**: Same as Spearman; Pearson assumes linearity
- **Target**: ≥ 0.70
- **When to Use**: When assuming linear relationships; less robust than Spearman for LLM outputs

---

## Error Metrics

### Mean Absolute Error (MAE)
- **Formula**: Σ|predicted_score - true_score| / n
- **Intent**: Average magnitude of prediction error on confidence scores
- **Interpretation**:
  - <0.10 : Excellent confidence calibration
  - 0.10-0.20 : Acceptable calibration
  - >0.20 : Model is poorly calibrated
- **Target**: <0.15
- **When to Use**: When model outputs continuous scores; measure calibration quality

### Root Mean Squared Error (RMSE)
- **Formula**: √(Σ(predicted - true)² / n)
- **Intent**: Penalizes large errors more heavily than MAE
- **Interpretation**: Same scale as MAE but sensitive to outliers
- **Target**: <0.20
- **When to Use**: When large errors are particularly problematic

### Bias
- **Formula**: (FP - FN) / (TP + TN + FP + FN) or (Precision - Recall) / 2
- **Intent**: Systematic over/under-prediction
- **Interpretation**:
  - >0.15 : Biased toward over-predicting hallucinations (too many false positives)
  - -0.15 to 0.15 : Balanced
  - <-0.15 : Biased toward under-predicting hallucinations (too many false negatives)
- **Target**: |bias| ≤ 0.15
- **When to Use**: Detect systematic patterns in errors; ensure balanced behavior

---

## Consistency Metrics

### Variance Across Runs
- **Formula**: σ² of metric values across 3+ runs
- **Intent**: How much does performance fluctuate run-to-run?
- **Interpretation**:
  - <0.01 : Highly stable
  - 0.01-0.05 : Good stability
  - >0.05 : Poor stability; don't trust individual runs
- **Target**: <0.05
- **When to Use**: Measure system reproducibility; flag prompts with high variance

### Self-Consistency Rate
- **Formula**: Instances where model makes same decision on 3+ runs / total instances
- **Intent**: Per-case consistency; does model always make same call?
- **Interpretation**:
  - 0.95+ : Almost always consistent
  - 0.80-0.95 : Acceptable; some borderline cases
  - <0.80 : Poor; high per-case variance
- **Target**: ≥ 0.90
- **When to Use**: Identify ambiguous test cases; flag cases where model is uncertain

---

## Per-Failure-Mode Breakdown

For each failure mode (FM1-FM7), calculate:
- **Accuracy**: Can we distinguish this mode?
- **F1**: Balanced correctness on this mode
- **Recall**: Do we catch hallucinations in this mode?

**Interpretation**: If accuracy on FM2 (fabrication) is 0.95 but on FM3 (subtle distortion) is 0.60, the model struggles with nuanced hallucinations.

---

## Metrics Hierarchy

1. **Primary**: F1, TNR, Bias (production-readiness thresholds)
2. **Secondary**: Precision, Recall, Accuracy (understand the gap)
3. **Tertiary**: Kappa, Variance, Self-Consistency (production confidence)
4. **Correlation**: Spearman, Pearson (when scores provided)
5. **Per-Mode Breakdown**: Understand strength/weakness by failure mode

Use primary metrics to decide "go/no-go"; use others for diagnostics.
