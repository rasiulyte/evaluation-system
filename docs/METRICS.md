# Metrics Reference Guide

A beginner-friendly guide to understanding AI evaluation metrics. Each metric includes interpretation scales and mental models to help you think about what the numbers mean.

---

## Quick Reference: What Do These Numbers Mean?

| Score Range | General Interpretation |
|-------------|----------------------|
| 0.90 - 1.00 | Excellent - Production ready |
| 0.75 - 0.89 | Good - Acceptable for most uses |
| 0.60 - 0.74 | Fair - Needs improvement |
| 0.40 - 0.59 | Poor - Significant issues |
| 0.00 - 0.39 | Failing - Major problems |

---

## Classification Metrics

These metrics tell you how well the system classifies responses as "hallucination" or "grounded."

### F1 Score (Primary Metric)

**Mental Model**: Think of F1 as a "balanced grade" for your detector. It's like a student who needs to be good at BOTH finding problems AND not crying wolf.

**What it measures**: The balance between catching hallucinations and not falsely flagging good content.

**Interpretation Scale**:
| F1 Score | Interpretation | What to Do |
|----------|---------------|------------|
| 0.95 - 1.00 | Exceptional | Ready for high-stakes production |
| 0.85 - 0.94 | Excellent | Production ready |
| 0.75 - 0.84 | Good | Acceptable, minor tuning helpful |
| 0.65 - 0.74 | Fair | Needs improvement before production |
| 0.50 - 0.64 | Poor | Significant prompt/model changes needed |
| < 0.50 | Failing | Fundamental redesign required |

**Target**: >= 0.75

**Real-world analogy**: Like a smoke detector - F1 measures how well it catches real fires (recall) while not going off when you're just cooking (precision).

---

### Precision

**Mental Model**: "When I say there's a hallucination, how often am I right?"

**What it measures**: Of all the times the system flags something as a hallucination, what percentage are actually hallucinations?

**Interpretation Scale**:
| Precision | Interpretation | User Impact |
|-----------|---------------|-------------|
| 0.95 - 1.00 | Exceptional | Users fully trust warnings |
| 0.90 - 0.94 | Excellent | Rare false alarms |
| 0.80 - 0.89 | Good | Occasional false alarms, acceptable |
| 0.70 - 0.79 | Fair | Noticeable false alarms |
| 0.60 - 0.69 | Poor | Users start ignoring warnings |
| < 0.60 | Failing | Warnings are meaningless |

**Target**: >= 0.75

**Real-world analogy**: Like a doctor's diagnosis accuracy - high precision means when they say "you have X," they're almost always right. Low precision means lots of unnecessary worry.

---

### Recall (Sensitivity)

**Mental Model**: "Of all the actual hallucinations out there, how many do I catch?"

**What it measures**: Of all the actual hallucinations, what percentage does the system successfully detect?

**Interpretation Scale**:
| Recall | Interpretation | Risk Level |
|--------|---------------|------------|
| 0.95 - 1.00 | Exceptional | Almost no hallucinations slip through |
| 0.90 - 0.94 | Excellent | Very few missed |
| 0.80 - 0.89 | Good | Some slip through, usually acceptable |
| 0.70 - 0.79 | Fair | Notable blind spots |
| 0.60 - 0.69 | Poor | Many hallucinations missed |
| < 0.60 | Failing | More hallucinations missed than caught |

**Target**: >= 0.75

**Real-world analogy**: Like airport security - high recall means they catch almost all prohibited items. Low recall means dangerous items get through regularly.

---

### TNR (True Negative Rate / Specificity)

**Mental Model**: "When content is actually good, do I correctly leave it alone?"

**What it measures**: Of all the grounded (good) content, what percentage does the system correctly accept?

**Interpretation Scale**:
| TNR | Interpretation | User Experience |
|-----|---------------|-----------------|
| 0.95 - 1.00 | Exceptional | Good content flows freely |
| 0.85 - 0.94 | Excellent | Rare blocking of good content |
| 0.75 - 0.84 | Good | Occasional false blocks |
| 0.65 - 0.74 | Fair | Users notice false blocks |
| 0.55 - 0.64 | Poor | Frustrating user experience |
| < 0.55 | Failing | System blocks more good than bad |

**Target**: >= 0.65

**Real-world analogy**: Like email spam filters - high TNR means important emails rarely go to spam. Low TNR means you're constantly checking your spam folder for real emails.

---

### Accuracy

**Mental Model**: "Overall, what percentage of my decisions are correct?"

**What it measures**: Total correct predictions (both hallucinations caught AND grounded content accepted) divided by all predictions.

**Interpretation Scale**:
| Accuracy | Interpretation | Reliability |
|----------|---------------|-------------|
| 0.95 - 1.00 | Exceptional | Highly reliable |
| 0.85 - 0.94 | Excellent | Very reliable |
| 0.75 - 0.84 | Good | Generally reliable |
| 0.65 - 0.74 | Fair | Sometimes unreliable |
| 0.50 - 0.64 | Poor | Often wrong |
| < 0.50 | Failing | Worse than random guessing |

**Target**: >= 0.75

**Real-world analogy**: Like a weather forecast - 90% accuracy means 9 out of 10 forecasts are correct. 50% accuracy is just flipping a coin.

**Caution**: Accuracy can be misleading with imbalanced data. If 90% of content is grounded, a system that always says "grounded" gets 90% accuracy but catches zero hallucinations!

---

## Agreement Metrics

These metrics measure consistency and reliability.

### Cohen's Kappa

**Mental Model**: "How much better than random chance am I performing?"

**What it measures**: Agreement between predictions and ground truth, adjusted for chance. Kappa = 0 means you're no better than guessing; Kappa = 1 means perfect agreement.

**Interpretation Scale** (Landis & Koch, 1977):
| Kappa | Interpretation | Meaning |
|-------|---------------|---------|
| 0.81 - 1.00 | Almost Perfect | Extremely reliable |
| 0.61 - 0.80 | Substantial | Good reliability |
| 0.41 - 0.60 | Moderate | Fair reliability |
| 0.21 - 0.40 | Fair | Limited reliability |
| 0.00 - 0.20 | Slight | Poor reliability |
| < 0.00 | Poor | Worse than chance |

**Target**: >= 0.60

**Real-world analogy**: Like two doctors independently diagnosing the same patients - Kappa measures how much they agree beyond what you'd expect from luck.

---

## Correlation Metrics

These metrics measure how well confidence scores relate to actual correctness.

### Spearman Correlation

**Mental Model**: "When I'm confident, am I actually more likely to be right?"

**What it measures**: Whether the model's confidence scores rank cases in the same order as their actual correctness. High Spearman means confident predictions tend to be correct.

**Interpretation Scale**:
| Spearman | Interpretation | Confidence Quality |
|----------|---------------|-------------------|
| 0.80 - 1.00 | Strong | Confidence is very trustworthy |
| 0.60 - 0.79 | Moderate | Confidence is useful but imperfect |
| 0.40 - 0.59 | Weak | Confidence only somewhat helpful |
| 0.20 - 0.39 | Very Weak | Confidence is barely meaningful |
| 0.00 - 0.19 | None | Confidence is random noise |
| < 0.00 | Negative | Confidence is backwards! |

**Target**: >= 0.60

**Real-world analogy**: Like a student's self-assessment - high correlation means when they say "I'm sure I got that right," they usually did. Low correlation means their confidence doesn't match reality.

---

### Pearson Correlation

**Mental Model**: "Is there a straight-line relationship between confidence and correctness?"

**What it measures**: Linear relationship between confidence scores and actual outcomes. Assumes the relationship is a straight line.

**Interpretation Scale**: Same as Spearman

**Target**: >= 0.60

**When to use**: Pearson assumes linear relationships. Spearman is usually better for LLM confidence scores.

---

### Kendall's Tau

**Mental Model**: "Do my confidence rankings match the correctness rankings?"

**What it measures**: How often pairs of predictions are ranked in the same order by confidence and by correctness. More robust to outliers than Spearman.

**Interpretation Scale**:
| Tau | Interpretation | Reliability |
|-----|---------------|-------------|
| 0.70 - 1.00 | Strong | Excellent ranking alignment |
| 0.50 - 0.69 | Moderate | Good ranking alignment |
| 0.30 - 0.49 | Weak | Some ranking alignment |
| 0.10 - 0.29 | Very Weak | Minimal alignment |
| < 0.10 | None | Rankings are unrelated |

**Target**: >= 0.50

**Real-world analogy**: Like ranking horses before a race - high Tau means your predictions of which horses will place where actually match the results.

---

## Error Metrics

These metrics measure how far off predictions are from reality.

### Bias

**Mental Model**: "Does the system lean toward over-flagging or under-flagging?"

**What it measures**: Systematic tendency to predict more or fewer hallucinations than actually exist.

**Interpretation Scale**:
| Bias | Interpretation | System Behavior |
|------|---------------|-----------------|
| > +0.20 | Strong Positive | Way too aggressive - flags too much |
| +0.10 to +0.20 | Moderate Positive | Somewhat aggressive |
| -0.10 to +0.10 | Balanced | Fair and unbiased |
| -0.20 to -0.10 | Moderate Negative | Somewhat lenient |
| < -0.20 | Strong Negative | Way too lenient - misses too much |

**Target**: |bias| <= 0.15

**Real-world analogy**: Like a scale that's not zeroed properly - positive bias is like a scale that always reads 2 lbs heavy, negative bias reads 2 lbs light.

---

### MAE (Mean Absolute Error)

**Mental Model**: "On average, how far off are my confidence scores?"

**What it measures**: Average distance between predicted confidence and actual correctness (0 or 1).

**Interpretation Scale**:
| MAE | Interpretation | Calibration Quality |
|-----|---------------|---------------------|
| 0.00 - 0.10 | Excellent | Near-perfect calibration |
| 0.10 - 0.20 | Good | Well calibrated |
| 0.20 - 0.30 | Fair | Acceptable calibration |
| 0.30 - 0.40 | Poor | Needs calibration work |
| > 0.40 | Failing | Severely miscalibrated |

**Target**: < 0.20

**Real-world analogy**: Like a GPS accuracy - MAE of 0.1 is like GPS being off by 10% of the distance. MAE of 0.4 means you're often in the wrong neighborhood.

---

### RMSE (Root Mean Squared Error)

**Mental Model**: "How bad are my worst mistakes?"

**What it measures**: Like MAE, but penalizes large errors more heavily. A few big mistakes hurt RMSE more than many small ones.

**Interpretation Scale**:
| RMSE | Interpretation | Error Distribution |
|------|---------------|-------------------|
| 0.00 - 0.15 | Excellent | Small, consistent errors |
| 0.15 - 0.25 | Good | Reasonable error distribution |
| 0.25 - 0.35 | Fair | Some large errors present |
| 0.35 - 0.50 | Poor | Frequent large errors |
| > 0.50 | Failing | Major errors common |

**Target**: < 0.25

**Real-world analogy**: Like judging an archer - MAE counts total distance from bullseye, but RMSE heavily penalizes the shots that completely miss the target.

---

## Consistency Metrics

### Variance Across Runs

**Mental Model**: "Do I get similar results each time I run this?"

**What it measures**: How much metric values fluctuate when you run the same evaluation multiple times.

**Interpretation Scale**:
| Variance | Interpretation | Stability |
|----------|---------------|-----------|
| 0.00 - 0.01 | Excellent | Highly stable results |
| 0.01 - 0.03 | Good | Stable enough to trust |
| 0.03 - 0.05 | Fair | Some run-to-run variation |
| 0.05 - 0.10 | Poor | Results vary significantly |
| > 0.10 | Failing | Results are unpredictable |

**Target**: < 0.05

**Real-world analogy**: Like measuring your blood pressure multiple times - low variance means consistent readings, high variance means something's unstable.

---

### Self-Consistency Rate

**Mental Model**: "For the same input, do I give the same answer?"

**What it measures**: Percentage of cases where the model makes the same prediction across multiple runs.

**Interpretation Scale**:
| Rate | Interpretation | Reliability |
|------|---------------|-------------|
| 0.95 - 1.00 | Excellent | Almost always consistent |
| 0.90 - 0.94 | Good | Highly consistent |
| 0.80 - 0.89 | Fair | Mostly consistent |
| 0.70 - 0.79 | Poor | Noticeable inconsistency |
| < 0.70 | Failing | Answers seem random |

**Target**: >= 0.90

**Real-world analogy**: Like asking someone the same question three times - high self-consistency means they give the same answer each time.

---

## Metrics Hierarchy: What to Look At First

### 1. Primary Metrics (Production Readiness)
- **F1 Score** - Overall balanced performance
- **TNR** - Are we blocking good content?
- **Bias** - Is the system fair?

### 2. Secondary Metrics (Diagnosis)
- **Precision** - False alarm rate
- **Recall** - Miss rate
- **Accuracy** - Overall correctness

### 3. Tertiary Metrics (Confidence)
- **Cohen's Kappa** - Better than chance?
- **Variance** - Is it stable?

### 4. Calibration Metrics (When using confidence scores)
- **Spearman** - Are confidence rankings meaningful?
- **MAE/RMSE** - Are confidence values accurate?

---

## Quick Decision Guide

**Is my system ready for production?**
1. F1 >= 0.75? If no, stop here - needs more work
2. TNR >= 0.65? If no, you're blocking too much good content
3. |Bias| <= 0.15? If no, system is unfair in one direction

**Should I trust the confidence scores?**
1. Spearman >= 0.60? If yes, confidence rankings are meaningful
2. MAE < 0.20? If yes, confidence values are calibrated

**Is my system stable?**
1. Variance < 0.05? If yes, results are reproducible
2. Self-consistency >= 0.90? If yes, same input gives same output
