# Failure Modes Catalog

## Overview

Each failure mode represents a distinct way hallucinations can occur or be misclassified. By explicitly testing each, we understand where our evaluation system is strong or weak.

---

## FM1: Factual Addition

**Definition**: LLM states true facts that exist in general knowledge but are not present in the provided context.

**Why It Matters**: 
- Tests whether system conflates "grounded in general knowledge" with "grounded in provided context"
- Real systems must be context-aware; facts not in context may be outdated or domain-specific

**Example**:
```
Context: "The Eiffel Tower is in Paris."
Response: "The Eiffel Tower is in Paris and was completed in 1889."
Label: grounded (fact is true, but 1889 is not in context)
```

**Classification Challenge**: Distinguishing between:
- Legitimate inference from context (acceptable)
- Pulling from external knowledge (may be hallucination depending on use case)

**Test Strategy**:
- Include 5 cases where additional fact is well-known (Paris, 1889)
- Include 5 cases where additional fact is obscure (architect's middle name)
- Include 5 cases where additional fact is outdated (population statistics)

**Target Metric**: High recall needed; false negatives here are acceptable

---

## FM2: Fabrication

**Definition**: LLM creates completely false claims with no basis in context or knowledge.

**Why It Matters**:
- Most egregious hallucination type; critical to catch
- Well-known LLM failure mode
- High-confidence false statements

**Example**:
```
Context: "Python is a programming language."
Response: "Python was invented in 1923 by Thomas Watson."
Label: hallucination (fabricated; Python is from 1989, Watson was not inventor)
```

**Classification Challenge**: 
- May be internally consistent and plausible-sounding
- LLM confidence can be high despite falsity

**Test Strategy**:
- Include 5 obvious fabrications (wrong dates, famous wrong claims)
- Include 5 subtle fabrications (plausible-sounding false attributions)
- Include 5 partially true + false combinations

**Target Metric**: Must have recall ≥ 0.90; this is non-negotiable

---

## FM3: Subtle Distortion

**Definition**: LLM modifies facts from context slightly—changing numbers, reversing claims, or altering nuance.

**Why It Matters**:
- Most deceptive hallucination type; easy to miss on casual reading
- Often passes semantic similarity checks
- Adversarial in nature

**Example**:
```
Context: "The study found that 20% of users preferred X over Y."
Response: "The study found that 80% of users preferred X over Y."
Label: hallucination (inversion of percentage)
```

**Classification Challenge**:
- Requires careful comparison to context
- May require mathematical or logical reasoning
- Hard for both humans and LLMs

**Test Strategy**:
- 5 cases with altered numbers (±10-30%)
- 5 cases with reversed/negated claims
- 5 cases with substituted but plausible alternatives

**Target Metric**: Difficult; target F1 ≥ 0.70 acceptable

---

## FM4: Valid Inference

**Definition**: LLM makes logical conclusions from context that are reasonable and correct.

**Why It Matters**:
- False positive risk; we must NOT flag valid inferences as hallucinations
- Tests system's ability to recognize reasoning, not just citation

**Example**:
```
Context: "All mammals breathe air. Dogs are mammals."
Response: "Dogs breathe air."
Label: grounded (valid logical inference)
```

**Classification Challenge**:
- Requires understanding logical validity
- May require multi-step reasoning
- Chain-of-thought prompts should help here

**Test Strategy**:
- 5 cases with simple syllogisms
- 5 cases with multi-step inference
- 5 cases with valid conditional reasoning

**Target Metric**: High precision needed; false positives here damage system credibility

---

## FM5: Verbatim Grounded

**Definition**: LLM copies or closely paraphrases text directly from context.

**Why It Matters**:
- Baseline negative case; should be trivial to classify
- Tests system's ability to recognize explicit grounding
- Sanity check

**Example**:
```
Context: "The Python programming language was created by Guido van Rossum."
Response: "Python was created by Guido van Rossum."
Label: grounded (direct quote, verbatim grounding)
```

**Classification Challenge**: None; this should be near 100% accuracy

**Test Strategy**:
- 5 direct quotes
- 5 close paraphrases (5-10% word changes)
- 5 structural remixes (same info, different order)

**Target Metric**: Expect accuracy ≥ 0.95; investigate if lower

---

## FM6: Fluent Hallucination

**Definition**: Well-written, confident-sounding claims that are factually false.

**Why It Matters**:
- Distinguishes between content quality and factuality
- Tests whether fluency tricks the evaluator
- Common in high-temperature generation

**Example**:
```
Context: "Renewable energy sources are important."
Response: "Solar panels are the oldest renewable energy technology, dating back to the 1920s and predating hydroelectric power by 50 years."
Label: hallucination (false; hydroelectric is older)
```

**Classification Challenge**:
- High surface-level plausibility
- Requires factual knowledge to contradict

**Test Strategy**:
- 5 cases with confident false attribution
- 5 cases with false but plausible-sounding statistics
- 5 cases with eloquent but baseless claims

**Target Metric**: Difficult; target F1 ≥ 0.70

---

## FM7: Partial Grounding

**Definition**: Response contains mix of grounded and hallucinated claims.

**Why It Matters**:
- Real-world scenario; fully grounded or fully hallucinated is cleaner
- Tests system's ability to identify mixed signals
- Requires fine-grained analysis

**Example**:
```
Context: "Machine learning requires labeled data. Neural networks are a type of ML model."
Response: "Machine learning requires labeled data. Neural networks were invented in 1974 by Geoffrey Hinton, who also founded Google DeepMind in 2006."
Label: hallucination (partial; first claim grounded, last two are false/distorted)
```

**Classification Challenge**:
- How to label mixed cases? (we treat as hallucination if any claim is hallucinated)
- Requires context awareness to distinguish parts

**Test Strategy**:
- 5 cases: 1 false claim in 3-sentence response
- 3 cases: 2 false claims in 5-sentence response
- 2 cases: False claims at start (recency bias test)

**Target Metric**: Moderate; target F1 ≥ 0.65

---

## Test Strategy Summary

| Mode | Cases | Difficulty | Key Challenge | Metric Target |
|------|-------|-----------|---|---|
| FM1 | 15 | Medium | Context-aware inference | Recall ≥ 0.85 |
| FM2 | 15 | Easy | Confidence detection | Recall ≥ 0.90 |
| FM3 | 15 | Hard | Subtle distortion recognition | F1 ≥ 0.70 |
| FM4 | 15 | Medium | Valid logic recognition | Precision ≥ 0.85 |
| FM5 | 15 | Easy | Sanity check | Accuracy ≥ 0.95 |
| FM6 | 15 | Hard | Fluence vs factuality | F1 ≥ 0.70 |
| FM7 | 10 | Hard | Partial analysis | F1 ≥ 0.65 |

---

## Usage

When implementing test cases:
1. Assign each to specific failure mode
2. Set difficulty based on challenge classification above
3. Include in appropriate .json file (ground_truth.json or regression.json)
4. Use in per-failure-mode breakdown analysis
5. Use for prompt diagnosis ("Why does prompt X fail on FM3?")
