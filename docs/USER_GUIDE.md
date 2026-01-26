# LLM Hallucination Evaluation System - User Guide

## What This Tool Does

This evaluation dashboard helps you measure how well AI language models (like GPT-4) detect **hallucinations** - when an AI generates information that sounds plausible but is factually incorrect or not supported by the given context.

This guide also serves as a **practical learning resource** for understanding evaluation metrics. Each metric is explained with real-world intuition, when to use it, and how to interpret the results.

---

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Understanding Metrics - The Complete Guide](#understanding-metrics---the-complete-guide)
   - [Agreement Metrics](#agreement-metrics)
   - [Classification Metrics](#classification-metrics)
   - [Correlation Metrics](#correlation-metrics)
   - [Error Metrics](#error-metrics)
   - [Consistency Metrics](#consistency-metrics)
3. [Metrics Decision Flowchart](#metrics-decision-flowchart)
4. [Test Cases and Failure Modes](#test-cases-and-failure-modes)
5. [Practical Recommendations](#practical-recommendations)
6. [Troubleshooting](#troubleshooting)
7. [Glossary](#glossary)

---

## Quick Start Guide

### Step 1: Access the Dashboard

1. Open the dashboard URL in your browser
2. You'll see a sidebar with navigation options

### Step 2: Run Your First Evaluation

1. Click **🚀 Run Evaluation** in the sidebar
2. Select which scenarios to test (start with just one)
3. Choose a sample size (start with 10-20 for quick tests)
4. Enter the password and click **Start Evaluation**
5. Wait for results (typically 30-60 seconds for 20 samples)

### Step 3: Review Results

1. Click **📈 Current Metrics** to see performance scores
2. Click **📅 Daily Runs** to see historical trends
3. Click **🔄 Compare Runs** to compare different evaluations

---

## Understanding Metrics - The Complete Guide

This section explains every metric in the system with practical intuition for AI evaluation.

### The Confusion Matrix - Foundation of All Metrics

Before diving into metrics, understand the confusion matrix:

```
                    PREDICTED
                 Hallucination | Grounded
              ┌────────────────┬──────────────┐
ACTUAL  Hall. │      TP        │      FN      │
              │ (True Positive)│(False Negative)│
              ├────────────────┼──────────────┤
       Ground │      FP        │      TN      │
              │(False Positive)│(True Negative)│
              └────────────────┴──────────────┘
```

- **TP (True Positive)**: Correctly identified hallucination
- **TN (True Negative)**: Correctly identified grounded content
- **FP (False Positive)**: Wrongly flagged grounded content as hallucination
- **FN (False Negative)**: Missed an actual hallucination

---

### Agreement Metrics

These metrics measure overall agreement between predictions and ground truth.

#### Accuracy

**Formula**: `(TP + TN) / (TP + TN + FP + FN)`

**What it measures**: The percentage of all predictions that were correct.

**Intuition**: If you made 100 predictions and 85 were correct, accuracy = 0.85

**When to use**:
- Quick health check
- When classes are balanced (similar number of hallucinations and grounded)

**When NOT to use**:
- Imbalanced datasets (e.g., 95% grounded, 5% hallucination)
- When false positives and false negatives have different costs

**Target**: ≥ 0.75

| Score | Interpretation |
|-------|----------------|
| 0.90+ | Excellent |
| 0.75-0.90 | Good, production-ready |
| 0.60-0.75 | Needs improvement |
| < 0.60 | Poor, needs redesign |

---

#### Cohen's Kappa (κ)

**Formula**: `κ = (observed agreement - chance agreement) / (1 - chance agreement)`

**What it measures**: Agreement between predictions and ground truth, corrected for chance.

**Intuition**: Imagine two people randomly guessing "hallucination" or "grounded". They'd agree sometimes just by chance. Kappa measures how much better your model is than random guessing.

**Real-world analogy**: Two doctors diagnosing patients. If they agree 80% of the time, but would agree 50% just by chance, Kappa tells you how much of that agreement is "real" skill.

**When to use**:
- Comparing model consistency across multiple runs
- When you want to account for class imbalance
- Measuring inter-rater reliability

**Target**: ≥ 0.70

| Score | Interpretation |
|-------|----------------|
| 0.81+ | Almost perfect agreement |
| 0.61-0.80 | Substantial agreement |
| 0.41-0.60 | Moderate agreement |
| 0.21-0.40 | Fair agreement |
| < 0.20 | Poor agreement (near chance) |

**Example**:
- Kappa = 0.0: Model is no better than random guessing
- Kappa = 0.70: Model shows substantial skill beyond chance
- Kappa = 1.0: Perfect agreement

---

### Classification Metrics

These metrics focus on the quality of positive (hallucination) and negative (grounded) predictions.

#### Precision

**Formula**: `TP / (TP + FP)`

**What it measures**: When the model says "hallucination", how often is it correct?

**Intuition**: Think of precision as "trust in alarms". If a fire alarm goes off 100 times and only 60 are real fires, precision = 0.60. People stop trusting the alarm.

**Real-world analogy**: A spam filter with low precision marks too many legitimate emails as spam. Users get frustrated and disable it.

**When to prioritize precision**:
- When false positives are costly (blocking good content)
- When users will lose trust if warnings are often wrong
- Content moderation systems

**Target**: ≥ 0.75

| Score | Interpretation |
|-------|----------------|
| 0.90+ | Very few false alarms |
| 0.75-0.90 | Acceptable false alarm rate |
| < 0.75 | Users will distrust warnings |

---

#### Recall (Sensitivity)

**Formula**: `TP / (TP + FN)`

**What it measures**: Of all actual hallucinations, how many did the model catch?

**Intuition**: Think of recall as "catch rate". If there are 100 hallucinations in your data and the model finds 80, recall = 0.80.

**Real-world analogy**: A medical test for cancer. Low recall means many cancers go undetected - dangerous!

**When to prioritize recall**:
- When missing positives is dangerous (medical, safety)
- When hallucinations could cause real harm
- Legal or financial applications

**Target**: ≥ 0.75

| Score | Interpretation |
|-------|----------------|
| 0.90+ | Catches almost all hallucinations |
| 0.75-0.90 | Acceptable miss rate |
| < 0.75 | Too many undetected hallucinations |

---

#### F1 Score (Primary Metric)

**Formula**: `2 × (Precision × Recall) / (Precision + Recall)`

**What it measures**: Harmonic mean of precision and recall - a balanced measure.

**Intuition**: F1 punishes extreme imbalances. You can't game it by only optimizing one metric.

**Why harmonic mean?**:
- Arithmetic mean of 0.90 precision and 0.10 recall = 0.50 (seems okay)
- Harmonic mean (F1) = 0.18 (reveals the problem!)

**When to use**:
- Primary metric for most evaluations
- When both false positives and false negatives matter
- When you need a single number to compare models

**Target**: ≥ 0.75

| Score | Interpretation |
|-------|----------------|
| 0.85+ | Excellent balance |
| 0.75-0.85 | Good, production-ready |
| < 0.75 | Needs improvement |

---

#### True Negative Rate (TNR / Specificity)

**Formula**: `TN / (TN + FP)`

**What it measures**: Of all grounded content, how much did we correctly accept?

**Intuition**: TNR is the "false alarm avoidance" rate. High TNR means the model rarely cries wolf.

**Real-world analogy**: An antivirus that flags every file as malware has 0% TNR - it's useless even if it catches all viruses.

**When to prioritize TNR**:
- When over-flagging disrupts workflows
- When grounded content must flow freely
- Customer-facing applications where false alarms annoy users

**Target**: ≥ 0.65

| Score | Interpretation |
|-------|----------------|
| 0.90+ | Rarely blocks good content |
| 0.75-0.90 | Acceptable false alarm rate |
| 0.65-0.75 | Marginal, some user friction |
| < 0.65 | Too aggressive, blocks too much |

---

### Correlation Metrics

These metrics measure relationships between model confidence and actual correctness. **Only calculated when confidence scores are provided.**

#### Spearman Correlation (ρ)

**Formula**: Correlation of ranks (not raw values)

**What it measures**: Monotonic relationship - when confidence goes up, does correctness tend to go up?

**Intuition**: Spearman doesn't care about exact values, only order. If the model ranks its most confident predictions as most likely correct, Spearman will be high.

**When to use**:
- When you have confidence scores
- When the relationship might not be linear
- More robust to outliers than Pearson

**Target**: ≥ 0.70

| Score | Interpretation |
|-------|----------------|
| 0.70+ | Strong monotonic relationship |
| 0.50-0.70 | Moderate relationship |
| < 0.50 | Weak relationship - confidence is unreliable |

---

#### Pearson Correlation (r)

**Formula**: Covariance / (std_x × std_y)

**What it measures**: Linear relationship between confidence and correctness.

**Intuition**: Pearson measures if the relationship is a straight line. If confidence of 0.6 means 60% correct and 0.9 means 90% correct, that's a perfect linear relationship.

**When to use**:
- When you expect a linear relationship
- When data is normally distributed
- For well-calibrated models

**Difference from Spearman**:
- Pearson: "Is the relationship a straight line?"
- Spearman: "When one goes up, does the other go up?"

**Target**: ≥ 0.70

---

#### Kendall's Tau (τ)

**Formula**: `(concordant pairs - discordant pairs) / total pairs`

**What it measures**: Ordinal association based on concordant vs discordant pairs.

**Intuition**: Compare every pair of predictions. If higher confidence usually means higher correctness (concordant), tau is positive. If they're often reversed (discordant), tau is negative.

**When to use**:
- Small sample sizes (more robust than Spearman)
- When there are many ties in the data
- When you want interpretable results (directly based on pair comparisons)

**Why Kendall's Tau vs Spearman?**:
- Tau is more robust with small samples
- Tau handles ties better
- Tau is directly interpretable (proportion of concordant pairs)
- Tau values are typically lower than Spearman for the same data

**Target**: ≥ 0.60 (lower than Spearman because tau values are inherently smaller)

| Score | Interpretation |
|-------|----------------|
| 0.70+ | Strong ordinal association |
| 0.50-0.70 | Moderate association |
| 0.30-0.50 | Weak association |
| < 0.30 | Very weak or no association |

**Example**:
```
Predictions ranked by confidence: A > B > C > D
Actual correctness ranked:        A > B > D > C

Pairs: (A,B)✓ (A,C)✓ (A,D)✓ (B,C)✓ (B,D)✓ (C,D)✗
Concordant: 5, Discordant: 1
Tau = (5-1)/6 = 0.67
```

---

### Error Metrics

These metrics measure the magnitude and direction of errors.

#### Bias

**Formula**: `(FP - FN) / Total`

**What it measures**: Systematic over-prediction or under-prediction.

**Intuition**:
- Positive bias: Model says "hallucination" too often (trigger-happy)
- Negative bias: Model says "grounded" too often (too lenient)
- Zero bias: Balanced errors

**Real-world analogy**: A scale that always reads 2 pounds heavy has positive bias. Even if readings vary, they're systematically too high.

**When to use**:
- Detecting systematic patterns in errors
- Ensuring model doesn't favor one class
- Identifying calibration issues

**Target**: |bias| ≤ 0.15

| Score | Interpretation |
|-------|----------------|
| > 0.15 | Biased toward hallucination (too many FP) |
| -0.15 to 0.15 | Balanced |
| < -0.15 | Biased toward grounded (too many FN) |

---

#### Mean Absolute Error (MAE)

**Formula**: `average(|confidence - actual|)`

**What it measures**: Average magnitude of confidence calibration error.

**Intuition**: If model says 80% confident but is only correct 60% of the time, that's a 0.20 error. MAE averages these across all predictions.

**When to use**:
- When you have confidence scores
- Measuring calibration quality
- When all errors are equally important

**Target**: < 0.15

| Score | Interpretation |
|-------|----------------|
| < 0.10 | Excellent calibration |
| 0.10-0.20 | Acceptable calibration |
| > 0.20 | Poorly calibrated |

---

#### Root Mean Squared Error (RMSE)

**Formula**: `sqrt(average((confidence - actual)²))`

**What it measures**: Like MAE but penalizes large errors more.

**Intuition**: Squaring magnifies big errors. A few wildly wrong confidence scores will spike RMSE even if most are okay.

**When to use**:
- When large errors are especially problematic
- When you want to penalize outliers
- Comparing models sensitive to worst-case performance

**Target**: < 0.20

---

### Consistency Metrics

These metrics measure reproducibility across multiple runs.

#### Variance Across Runs

**Formula**: Statistical variance of a metric across 3+ runs

**What it measures**: How much does performance fluctuate?

**Intuition**: Run the same evaluation 5 times. If F1 varies from 0.70 to 0.90, you can't trust any single run.

**When to use**:
- Before trusting single-run results
- Measuring system stability
- Identifying unreliable evaluation setups

**Target**: < 0.05

| Score | Interpretation |
|-------|----------------|
| < 0.01 | Highly stable |
| 0.01-0.05 | Good stability |
| > 0.05 | Poor stability - don't trust single runs |

---

#### Self-Consistency Rate

**Formula**: `cases with same prediction across all runs / total cases`

**What it measures**: Per-case consistency across runs.

**Intuition**: For each test case, does the model always make the same decision? Low self-consistency reveals ambiguous cases.

**When to use**:
- Identifying borderline/ambiguous test cases
- Measuring decision stability
- Finding cases that need clearer ground truth

**Target**: ≥ 0.90

---

## Metrics Decision Flowchart

Use this flowchart to choose which metrics to prioritize:

```
START: What's your primary concern?
│
├─→ "Missing hallucinations is dangerous"
│   └─→ Prioritize RECALL
│       Secondary: F1, Bias (check for FN bias)
│
├─→ "False alarms annoy users"
│   └─→ Prioritize PRECISION and TNR
│       Secondary: F1, Bias (check for FP bias)
│
├─→ "Need balanced performance"
│   └─→ Prioritize F1
│       Secondary: Precision, Recall, Bias
│
├─→ "Comparing multiple models"
│   └─→ Use F1 for ranking
│       Use Cohen's Kappa for consistency
│       Check Bias to ensure fairness
│
├─→ "Model outputs confidence scores"
│   └─→ Add correlation metrics:
│       - Spearman (general)
│       - Kendall's Tau (small samples/ties)
│       - Pearson (if expecting linear relationship)
│       Add calibration metrics: MAE, RMSE
│
└─→ "Need reproducible results"
    └─→ Run 3+ evaluations
        Check Variance and Self-Consistency
        Use Cohen's Kappa across runs
```

---

## Metrics at a Glance - Quick Reference

| Metric | Formula | Target | Higher is Better? |
|--------|---------|--------|-------------------|
| Accuracy | (TP+TN)/Total | ≥ 0.75 | Yes |
| Cohen's Kappa | Agreement corrected for chance | ≥ 0.70 | Yes |
| Precision | TP/(TP+FP) | ≥ 0.75 | Yes |
| Recall | TP/(TP+FN) | ≥ 0.75 | Yes |
| F1 Score | Harmonic mean of P & R | ≥ 0.75 | Yes |
| TNR | TN/(TN+FP) | ≥ 0.65 | Yes |
| Spearman | Rank correlation | ≥ 0.70 | Yes |
| Pearson | Linear correlation | ≥ 0.70 | Yes |
| Kendall's Tau | Concordant pairs | ≥ 0.60 | Yes |
| Bias | (FP-FN)/Total | \|x\| ≤ 0.15 | Closer to 0 |
| MAE | Avg absolute error | < 0.15 | No (lower better) |
| RMSE | Root mean squared error | < 0.20 | No (lower better) |
| Variance | Fluctuation across runs | < 0.05 | No (lower better) |

---

## Test Cases and Failure Modes

The system tests against 7 types of hallucination failure modes:

| Code | Failure Mode | Description | Example |
|------|--------------|-------------|---------|
| FM1 | Fabricated Facts | Inventing statistics or facts | "Studies show 73% of..." (no such study) |
| FM2 | Entity Confusion | Mixing up names, dates, places | Attributing a quote to wrong person |
| FM3 | Temporal Errors | Wrong dates or time sequences | "Founded in 1985" when it was 1995 |
| FM4 | Causal Hallucination | Inventing cause-effect relationships | "This caused that" without evidence |
| FM5 | Overconfidence | Stating uncertainty as certainty | "Definitely" when context says "possibly" |
| FM6 | Context Ignorance | Contradicting provided context | Saying opposite of what context states |
| FM7 | False Attribution | Misattributing sources or quotes | "According to X..." when X never said it |

---

## Practical Recommendations

### Getting Started

1. **Start small**: Run 1 scenario with 20 samples first
2. **Establish baseline**: Run 3-5 evaluations to understand normal variance
3. **Document everything**: Note which prompt version and model you used

### Recommended Evaluation Schedule

| Frequency | What to Run | Purpose |
|-----------|-------------|---------|
| Daily | 1 scenario, 20 samples | Quick sanity check |
| Weekly | All scenarios, 50 samples | Full performance assessment |
| After changes | All scenarios, 100 samples | Validate changes |
| Monthly | All scenarios, 3 runs each | Consistency measurement |

### Interpreting Dashboard Results

**GREEN (Healthy)**: All metrics meet thresholds. Safe for production.

**YELLOW (Warning)**: Some metrics degrading. Investigate which failure modes are causing issues.

**RED (Critical)**: Do not deploy. Analyze failed test cases to understand root cause.

### Improving Performance

| If this metric is low... | Try this... |
|--------------------------|-------------|
| F1 | Check which failure modes have lowest accuracy; add examples |
| Precision | Model is trigger-happy; add grounded examples to prompt |
| Recall | Model is too lenient; add more hallucination examples |
| TNR | Model flags too much; make prompt more conservative |
| Bias (positive) | Too many false positives; balance training examples |
| Bias (negative) | Too many false negatives; emphasize catching hallucinations |
| Kendall's Tau | Confidence is unreliable; consider not using confidence scores |

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "No metrics found in database" | Run an evaluation first |
| All API calls failing | Check OPENAI_API_KEY is valid |
| Metrics are all zeros | API errors - check logs |
| Test Results count is 0 | Run new evaluation (old runs didn't save details) |
| High variance across runs | Increase sample size; use more consistent prompts |

---

## Glossary

| Term | Definition |
|------|------------|
| **Hallucination** | AI-generated content that is factually incorrect or unsupported |
| **Grounded** | AI response that is supported by the provided context |
| **Concordant pair** | Two samples where higher confidence = higher correctness |
| **Discordant pair** | Two samples where higher confidence = lower correctness |
| **Calibration** | How well confidence scores match actual accuracy |
| **Precision** | Accuracy of positive predictions |
| **Recall** | Coverage of actual positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **TNR** | True Negative Rate - correctly identifying valid content |
| **Bias** | Systematic tendency toward one prediction class |
| **Kappa** | Agreement metric corrected for chance |
| **Tau** | Kendall's rank correlation coefficient |

---

## Additional Resources

- **In Dashboard**: Click 🔧 Debug Info to see database status
- **Logs**: On Streamlit Cloud, click "Manage app" for detailed logs
- **Metrics Code**: See `src/metrics.py` for implementation details

---

*Last updated: January 2026*
