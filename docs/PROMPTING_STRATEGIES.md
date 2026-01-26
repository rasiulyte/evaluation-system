# Prompting Strategies for Hallucination Detection

## Overview

Five progressively sophisticated prompting techniques, each testing a different hypothesis about what improves hallucination detection.

---

## V1: Zero-Shot Baseline

**Description**: Minimal instructions; LLM decides how to approach the task.

**Hypothesis**: Establish baseline performance without guidance.

**When to Use**: 
- Initial evaluation
- Comparison baseline for other approaches
- Quick iteration cycles

**Pros**:
- Simplest to implement
- Fewest tokens (cost-effective)
- Reveals LLM's intrinsic ability

**Cons**:
- May produce unstructured responses
- High variance in output format
- Low upper bound on performance

**Example Prompt**:
```
Determine whether the following response is grounded in the provided context.

Context: {context}

Response: {response}

Is this response grounded or hallucinated? Answer with "grounded" or "hallucinated".
```

**Expected Performance**:
- Baseline; typically F1 = 0.50-0.65
- High variance across runs

---

## V2: Few-Shot Learning

**Description**: Provide 3-5 labeled examples including boundary cases.

**Hypothesis**: Examples teach the model task structure and ambiguity boundaries.

**When to Use**:
- When baseline performance is below 0.70
- To improve consistency on edge cases
- When task is non-obvious from instructions alone

**Pros**:
- Significantly improves performance
- Illustrates boundary cases
- Format consistency from examples

**Cons**:
- More tokens (cost increase ~30%)
- Sensitive to example selection
- May overfit to example style

**Example Prompt Structure**:
```
You are an AI trained to detect hallucinations in LLM responses.

Example 1:
Context: "Python is a programming language created in 1989."
Response: "Python is a programming language."
Classification: grounded
Reasoning: The response is directly supported by the context.

Example 2:
Context: "The Earth orbits the Sun."
Response: "The Earth orbits the Sun and is 150 million km away."
Classification: grounded
Reasoning: Distance is general knowledge, not contradicted by context.

Example 3:
Context: "Paris is a city in France."
Response: "Paris is a city in England."
Classification: hallucinated
Reasoning: Direct contradiction of context.

Example 4:
Context: "The study included 100 participants."
Response: "The study included 50 participants."
Classification: hallucinated
Reasoning: Subtle distortion; number contradicts context.

Example 5:
Context: "All mammals breathe air. Dogs are mammals."
Response: "Dogs breathe air."
Classification: grounded
Reasoning: Valid logical inference from context.

Now classify this:
Context: {context}
Response: {response}
Classification: [grounded/hallucinated]
Reasoning: [Your reasoning]
```

**Expected Performance**:
- Improvement; typically F1 = 0.65-0.75
- More consistent output format
- Better handling of edge cases

---

## V3: Chain-of-Thought Reasoning

**Description**: Require step-by-step reasoning before classification.

**Hypothesis**: Explicit reasoning steps improve rigor and catch hidden assumptions.

**When to Use**:
- To improve recall on subtle distortions (FM3)
- When few-shot isn't sufficient
- For complex reasoning-dependent cases

**Pros**:
- Improves accuracy on complex cases
- Provides interpretability
- Better performance on FM3, FM6

**Cons**:
- Significantly more tokens (cost increase ~80%)
- Slower inference
- May produce verbose outputs

**Example Prompt Structure**:
```
Determine if the response is grounded in the context through careful analysis.

Context: {context}

Response: {response}

Think through this step by step:
1. What are the main claims in the response?
2. Which claims are explicitly in the context?
3. Which claims require inference or external knowledge?
4. Are there any contradictions between response and context?
5. Are any numbers, dates, or specific facts altered?
6. Could any claim be a valid inference from context?

After analysis:
Classification: [grounded/hallucinated]
Reasoning: [Your reasoning from steps above]
```

**Expected Performance**:
- Significant improvement; typically F1 = 0.70-0.80
- Better performance on FM3, FM4 (nuanced cases)
- Slower but more thoughtful responses

---

## V4: Rubric-Based Evaluation

**Description**: Provide explicit evaluation criteria and scoring guidelines.

**Hypothesis**: Clear rubric reduces ambiguity and aligns LLM with human judges.

**When to Use**:
- When specific failure modes need emphasis
- To improve consistency on FM6 (fluent hallucinations)
- When domain-specific criteria matter

**Pros**:
- Explicit criteria align LLM with evaluation goals
- Good for domain-specific adjustments
- Transparent evaluation rationale

**Cons**:
- Requires careful rubric design
- More tokens
- Rubric biases can propagate

**Example Prompt Structure**:
```
Use the following rubric to evaluate hallucination:

RUBRIC:
1. Direct Grounding (0-25 pts): Is claim explicitly stated in context?
   - 25: Verbatim or clear paraphrase
   - 15: Slightly reformulated but same meaning
   - 5: Related but modified meaning
   - 0: Not in context

2. Factual Correctness (0-25 pts): Is claim true based on context + common knowledge?
   - 25: Definitely true
   - 15: Likely true, no contradictions
   - 5: Ambiguous or needs verification
   - 0: False or contradicted by context

3. Inference Validity (0-25 pts): If not directly grounded, is it valid inference?
   - 25: Clear logical consequence
   - 15: Reasonable inference
   - 5: Possible but stretch
   - 0: No valid inference

4. Numerical Accuracy (0-25 pts): Are all numbers, dates, percentages exact?
   - 25: All exact or no numbers
   - 10: One minor variation
   - 0: Any significant variation

SCORING:
- 75-100: GROUNDED
- 50-74: AMBIGUOUS (treat as hallucinated)
- 0-49: HALLUCINATED

Context: {context}
Response: {response}

Score each dimension and provide total score and classification.
```

**Expected Performance**:
- Typically F1 = 0.72-0.78
- Excellent on FM2 (fabrication), good on FM6
- Very explicit reasoning

---

## V5: Structured Output Format

**Description**: Require machine-readable JSON output with detailed analysis.

**Hypothesis**: Structured output enables easier parsing, higher consistency, and confidence scoring.

**When to Use**:
- For production pipelines requiring JSON parsing
- When confidence scores are valuable
- When span-level analysis is needed

**Pros**:
- Deterministic parsing (no string matching)
- Confidence scores for threshold tuning
- Span identification helps debugging
- Best for production systems

**Cons**:
- More tokens
- Requires JSON parsing error handling
- Model must support structured output reliably

**Example Prompt Structure**:
```
Analyze the response for hallucinations and return JSON:

Context: {context}

Response: {response}

Return valid JSON with this structure:
{
  "classification": "grounded" | "hallucinated",
  "confidence": 0.0-1.0,
  "hallucinated_spans": ["span1", "span2"],
  "grounded_claims": ["claim1", "claim2"],
  "reasoning": "Brief explanation",
  "chain_of_thought": ["step1", "step2", "step3"]
}

Respond with ONLY valid JSON, no other text.
```

**Expected Performance**:
- Typically F1 = 0.75-0.82 (best performance)
- Confidence scores useful for threshold optimization
- Excellent for production use
- Most tokens (cost ~200% of zero-shot)

---

## V6: Calibrated Confidence

**Description**: Structured JSON output with explicit confidence calibration guidelines.

**Hypothesis**: Explicit confidence-to-probability mapping improves correlation metrics (Spearman, Pearson, Kendall's Tau) by teaching the model what confidence values mean.

**When to Use**:
- When correlation metrics (Spearman) are critical
- When confidence scores need to be well-calibrated
- For production systems requiring reliable uncertainty quantification

**Pros**:
- Dramatically improves correlation metrics (Spearman improved from ~0.26 to ~0.84 in testing)
- Better calibrated confidence scores
- Simple, concise prompt
- JSON output for easy parsing

**Cons**:
- Requires model to understand calibration guidelines
- May not improve classification metrics (F1, accuracy)
- Less explainability than rubric-based

**Example Prompt Structure**:
```
You are an AI trained to detect hallucinations in LLM responses.

Context: {context}

Response to analyze: {response}

Analyze whether this response is grounded in the provided context or contains hallucinations.

IMPORTANT - Confidence Calibration Guidelines:
- confidence 0.95-1.0: Response directly quotes or paraphrases the context with no additions
- confidence 0.80-0.94: Response is clearly supported by context with minor inference
- confidence 0.60-0.79: Response requires moderate inference from context
- confidence 0.40-0.59: Uncertain - evidence is ambiguous
- confidence 0.20-0.39: Response likely adds information not in context
- confidence 0.00-0.19: Response clearly contradicts or fabricates beyond context

Respond with ONLY valid JSON:
{"classification": "grounded", "confidence": 0.85, "reasoning": "Brief explanation"}
```

**Expected Performance**:
- F1: Similar to V5 (~0.75-0.80)
- Spearman correlation: 0.80+ (significant improvement)
- Best for calibration-critical applications

---

## Comparative Summary

| Dimension | V1 | V2 | V3 | V4 | V5 | V6 |
|-----------|----|----|----|----|-----|-----|
| Expected F1 | 0.55 | 0.70 | 0.75 | 0.75 | 0.80 | 0.78 |
| Token Cost | 1x | 1.3x | 1.8x | 1.6x | 2.0x | 1.5x |
| Format Consistency | Low | Medium | High | High | Very High | Very High |
| Explainability | Low | Medium | High | Very High | High | Medium |
| Correlation (Spearman) | N/A | N/A | N/A | N/A | ~0.50 | ~0.84 |
| Speed | Fastest | Fast | Medium | Medium | Slow | Fast |
| Production Ready | No | Partial | Yes | Yes | Yes | Yes |
| Best For | Baseline | Quick eval | Complex cases | Domain-specific | Production | Calibration |

---

## Recommendation Flow

1. **Start with V1** (establish baseline, understand problem)
2. **If F1 < 0.65**: Move to V2 (add examples)
3. **If F1 < 0.75**: Move to V3 (add reasoning) or V4 (add rubric)
4. **If still < 0.75**: Revisit test cases (may be flawed)
5. **For Production**: Use V5 or V6 with confidence threshold tuning on regression set
6. **If correlation metrics matter**: Use V6 (calibrated confidence) for best Spearman/Pearson scores

---

## Example A/B Test

Compare V2 (few-shot) vs V3 (chain-of-thought):
- Run both on 80-case training set
- Calculate F1, TNR, recall for each
- Measure consistency (run 3x, calculate variance)
- Statistical significance test
- Winner goes to regression testing
