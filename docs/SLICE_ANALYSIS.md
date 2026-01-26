# Slice Analysis Guide

## What is Slice Analysis?

**Slice Analysis** breaks down your overall metrics into subsets (slices) of your data to reveal hidden performance patterns. While aggregate metrics like F1 = 0.80 look good, slice analysis might reveal:

- FM3 (Subtle Distortion) has F1 = 0.45 - a major weakness
- Hard cases have 30% lower accuracy than easy cases
- The model is biased toward predicting "grounded"

## Why Slice Analysis Matters

Aggregate metrics can hide critical problems:

| Scenario | Aggregate F1 | Hidden Problem |
|----------|--------------|----------------|
| 0.82 overall | FM6 (Fluent Hallucination) F1 = 0.40 | Model fooled by well-written false content |
| 0.78 overall | Hard cases accuracy = 0.50 | Model only works on easy cases |
| 0.80 overall | Hallucination recall = 0.60 | Missing 40% of actual hallucinations |

## Slice Dimensions

### 1. By Failure Mode (FM1-FM7)

Break down performance by the type of hallucination:

| Failure Mode | What It Tests | Expected Difficulty |
|--------------|---------------|---------------------|
| FM1: Factual Addition | True facts not in context | Medium |
| FM2: Fabrication | Outright false claims | Easy |
| FM3: Subtle Distortion | Small changes to facts | Hard |
| FM4: Valid Inference | Logical inferences | Medium |
| FM5: Verbatim Grounded | Direct quotes | Easy |
| FM6: Fluent Hallucination | Well-written but wrong | Hard |
| FM7: Partial Grounding | Mixed grounded/hallucinated | Hard |

**What to look for:**
- FM2 and FM5 should have high accuracy (>90%) - these are baseline cases
- FM3, FM6, FM7 are expected to be harder - but F1 < 0.60 indicates a problem
- Large variance between failure modes suggests the prompt needs refinement

### 2. By Difficulty Level

Test cases are labeled easy, medium, or hard based on classification challenge:

| Difficulty | Expected Accuracy | If Lower Than Expected |
|------------|-------------------|------------------------|
| Easy | >85% | Fundamental prompt issue |
| Medium | 70-85% | Prompt may need examples |
| Hard | 55-75% | Expected - focus on other areas first |

**What to look for:**
- Easy cases with low accuracy indicate a broken prompt
- Large drop-off from easy to medium suggests lack of examples
- Hard cases below 50% may indicate the task is too difficult for the model

### 3. By Ground Truth Label

Compare performance on "grounded" vs "hallucination" cases:

| Metric | What It Reveals |
|--------|-----------------|
| Grounded accuracy | How well we identify safe content (TNR) |
| Hallucination accuracy | How well we catch problems (Recall) |
| Difference | Bias toward one class |

**What to look for:**
- Large difference (>15%) indicates class bias
- Low hallucination accuracy = dangerous (missing real problems)
- Low grounded accuracy = annoying (too many false alarms)

## How to Use Slice Analysis

### Step 1: Check Baseline Cases First

Start with FM2 (Fabrication) and FM5 (Verbatim Grounded):
- These should be near-perfect (>90% accuracy)
- If not, your prompt has fundamental issues

### Step 2: Identify Weakest Slices

Sort failure modes by F1 score:
- Focus on the bottom 2-3 performers
- These are your improvement opportunities

### Step 3: Diagnose the Problem

For each weak slice, examine:
1. **Example cases**: Read actual predictions and ground truth
2. **Error patterns**: Are errors consistent (same mistake) or random?
3. **Confidence**: Does the model know when it's wrong?

### Step 4: Targeted Improvements

Based on diagnosis:

| Problem | Solution |
|---------|----------|
| Model doesn't understand the category | Add examples of this failure mode to prompt |
| Model is overconfident on wrong answers | Add chain-of-thought reasoning |
| Model misses subtle details | Use rubric-based prompt with explicit criteria |
| Random errors | May need more test cases or different model |

## Slice Analysis in the Dashboard

The dashboard's **Slice Analysis** page provides:

1. **Performance by Failure Mode**
   - F1, Precision, Recall for each FM1-FM7
   - Color-coded cards (green = good, yellow = warning, red = poor)

2. **Performance by Difficulty**
   - Accuracy breakdown for easy/medium/hard cases
   - Visual comparison chart

3. **Performance by Label**
   - Grounded vs Hallucination accuracy
   - Bias indicator

4. **Key Insights**
   - Auto-generated observations about strongest/weakest areas
   - Actionable recommendations

## Example Analysis

**Scenario**: Overall F1 = 0.78, but slice analysis reveals:

```
Failure Mode Analysis:
- FM5 (Verbatim): F1 = 0.95  [Baseline OK]
- FM2 (Fabrication): F1 = 0.92  [Baseline OK]
- FM4 (Inference): F1 = 0.85  [Good]
- FM1 (Factual Addition): F1 = 0.75  [Acceptable]
- FM7 (Partial): F1 = 0.62  [Needs work]
- FM3 (Subtle Distortion): F1 = 0.55  [Problem!]
- FM6 (Fluent Hallucination): F1 = 0.48  [Critical!]

Difficulty Analysis:
- Easy: 88% accuracy
- Medium: 72% accuracy
- Hard: 58% accuracy

Label Analysis:
- Grounded: 85% accuracy
- Hallucination: 71% accuracy
- Bias: Toward grounded (14% difference)
```

**Interpretation**:
1. Baselines (FM2, FM5) are healthy - prompt fundamentals are OK
2. FM6 and FM3 are critical weaknesses - model struggles with subtle/fluent hallucinations
3. Hard cases show expected drop-off
4. Model has slight bias toward "grounded" - may need more hallucination examples

**Action Plan**:
1. Add FM6 examples to prompt (fluent but false content)
2. Add explicit criteria for detecting subtle distortions (FM3)
3. Consider chain-of-thought prompt to catch nuanced errors
4. Balance training examples (more hallucination cases)

## Best Practices

1. **Always check slices before celebrating aggregate metrics**
   - High overall F1 can hide critical weaknesses

2. **Prioritize baseline failure modes**
   - FM2 and FM5 should work first

3. **Focus on highest-impact slices**
   - Improve the weakest 2-3 slices before optimizing strong ones

4. **Track slice metrics over time**
   - Use trends to ensure improvements don't regress other slices

5. **Use slice analysis for prompt iteration**
   - Add examples targeting specific failure modes
   - Verify improvement on that slice without hurting others
