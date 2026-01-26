# LLM Hallucination Evaluation System - User Guide

## What This Tool Does

This evaluation dashboard helps you measure how well AI language models (like GPT-4) detect **hallucinations** - when an AI generates information that sounds plausible but is factually incorrect or not supported by the given context.

### Why Evaluate Hallucinations?

AI systems can confidently state false information. Before deploying AI in production, you need to know:
- How often does your AI correctly identify false statements?
- How often does it incorrectly flag true statements as false?
- Is your AI biased toward saying "yes" or "no"?

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

## Understanding the Dashboard Pages

### 🚀 Run Evaluation

This is where you run new tests against the AI system.

**Settings you can configure:**

| Setting | What it means | Recommended value |
|---------|---------------|-------------------|
| **Scenarios** | Different types of tests to run | Start with 1-2, expand later |
| **Model** | Which AI model to test | `gpt-4` for best results |
| **Sample Size** | How many test cases to run | 20 for quick tests, 50+ for accurate results |

**Scenarios explained:**

- **hallucination_detection**: Core test - can the AI identify false statements?
- **factual_accuracy**: Tests if AI can verify facts against given context
- **reasoning_quality**: Tests logical reasoning capabilities
- **instruction_following**: Tests if AI follows specific instructions
- **safety_compliance**: Tests for safe/appropriate responses
- **consistency**: Tests if AI gives consistent answers

### 📈 Current Metrics

Shows the most recent evaluation results with key performance indicators.

**Key Metrics:**

| Metric | What it measures | Good score | Warning | Critical |
|--------|------------------|------------|---------|----------|
| **F1 Score** | Overall accuracy (balance of precision & recall) | > 0.75 | 0.70-0.75 | < 0.70 |
| **TNR (True Negative Rate)** | Correctly identifying valid statements | > 0.65 | 0.60-0.65 | < 0.60 |
| **Bias** | Tendency to favor one answer | < 0.15 | 0.15-0.20 | > 0.20 |
| **Precision** | When AI says "hallucination", how often is it right? | > 0.70 | 0.65-0.70 | < 0.65 |
| **Recall** | Of all hallucinations, how many did AI catch? | > 0.70 | 0.65-0.70 | < 0.65 |

### 📅 Daily Runs

Shows history of all evaluation runs over time. Use this to:
- Track improvement or degradation over time
- Identify when problems started
- Compare performance across different dates

### 🔄 Compare Runs

Side-by-side comparison of two evaluation runs. Use this to:
- Compare different AI models
- Compare different prompt versions
- Measure impact of changes

---

## Understanding the Metrics

### F1 Score (Most Important)

The F1 score is your primary measure of success. It balances:
- **Precision**: Not crying wolf (avoiding false alarms)
- **Recall**: Not missing real problems (catching all hallucinations)

**Interpreting F1 scores:**
- **0.90+**: Excellent - production ready
- **0.80-0.90**: Good - minor improvements needed
- **0.70-0.80**: Fair - significant room for improvement
- **< 0.70**: Poor - needs major work before production

### True Negative Rate (TNR)

TNR measures how well the AI avoids false positives. A low TNR means the AI is too aggressive - it's flagging valid statements as hallucinations.

**Why TNR matters:**
- High false positive rate annoys users
- People stop trusting warnings if they're often wrong
- Business impact: lost productivity from unnecessary reviews

### Bias

Bias measures whether the AI tends to always say "hallucination" or always say "grounded".

- **Bias = 0**: Perfectly balanced predictions
- **Bias > 0**: Tends to predict "hallucination" too often
- **Bias < 0**: Tends to predict "grounded" too often

**Target: Keep bias between -0.15 and +0.15**

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

### Viewing Test Case Details

1. Go to **📈 Current Metrics**
2. Scroll to **🧪 Test Set & Case Results**
3. Select a test case from the dropdown
4. View:
   - The original context given to the AI
   - The response the AI was asked to evaluate
   - The AI's prediction vs. the correct answer
   - The AI's reasoning (if available)

---

## Practical Recommendations

### Getting Started

1. **Start small**: Run 1 scenario with 20 samples first
2. **Establish baseline**: Run 3-5 evaluations to understand normal variance
3. **Document everything**: Note which prompt version and model you used

### Running Regular Evaluations

**Recommended schedule:**
- **Daily**: Quick sanity check (1 scenario, 20 samples)
- **Weekly**: Full evaluation (all scenarios, 50 samples each)
- **After changes**: Always evaluate after changing prompts or models

### Interpreting Results

**When metrics are GREEN (healthy):**
- System is performing within acceptable thresholds
- Continue monitoring for regression

**When metrics are YELLOW (warning):**
- Performance is degrading but not critical
- Investigate which failure modes are causing issues
- Consider prompt improvements

**When metrics are RED (critical):**
- Immediate attention required
- Do not deploy to production
- Analyze failed test cases to understand root cause

### Improving Performance

**If F1 is low:**
1. Check which failure modes have lowest accuracy
2. Add examples of that failure mode to your prompt
3. Try chain-of-thought prompting (v3_chain_of_thought)

**If TNR is low (too many false positives):**
1. AI is too aggressive in flagging hallucinations
2. Add examples of grounded (valid) statements to prompt
3. Adjust prompt to be more conservative

**If Bias is high:**
1. Check class distribution in test cases
2. Add more examples of the underrepresented class
3. Explicitly instruct AI to consider both possibilities

### Comparing Models

To compare two AI models:
1. Run evaluation with Model A
2. Run evaluation with Model B (same scenarios, same sample size)
3. Use **🔄 Compare Runs** to see differences
4. Look for:
   - Overall F1 improvement
   - Specific failure modes where one model excels
   - Cost/speed tradeoffs

### Comparing Prompts

The system includes 5 prompt versions:

| Prompt | Strategy | Best for |
|--------|----------|----------|
| v1_zero_shot | Simple, no examples | Quick baseline |
| v2_few_shot | Includes examples | Better accuracy |
| v3_chain_of_thought | Step-by-step reasoning | Complex cases |
| v4_rubric_based | Scoring rubric | Consistent evaluation |
| v5_structured_output | JSON output | Programmatic use |

**Recommendation:** Start with v2_few_shot, then try v3_chain_of_thought if accuracy is insufficient.

---

## Troubleshooting

### "No metrics found in database"
- Run an evaluation first using 🚀 Run Evaluation

### Evaluation shows all errors
- Check that OPENAI_API_KEY is valid
- Verify API key has sufficient credits
- Check internet connectivity

### Metrics are all zeros
- API calls are failing - check logs
- Verify API key is correctly formatted (no line breaks)

### Test Results count is 0
- Run a new evaluation (old runs didn't save test details)
- Check for errors in the evaluation logs

### Database connection errors
- Verify DATABASE_URL is correct
- Check password doesn't have special characters
- Ensure you're using the pooler connection (port 6543)

---

## Glossary

| Term | Definition |
|------|------------|
| **Hallucination** | AI-generated content that is factually incorrect or unsupported |
| **Grounded** | AI response that is supported by the provided context |
| **Precision** | Of predictions labeled "hallucination", % that were correct |
| **Recall** | Of actual hallucinations, % that were detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **TNR** | True Negative Rate - correctly identifying valid content |
| **Bias** | Systematic tendency toward one prediction class |
| **Prompt** | The instruction template given to the AI |
| **Test Case** | A context + response pair with known correct answer |
| **Scenario** | A category of evaluation tests |

---

## Best Practices Summary

1. **Evaluate regularly** - AI performance can drift over time
2. **Use sufficient sample sizes** - At least 50 samples for reliable metrics
3. **Track trends** - Single evaluations have variance; look at patterns
4. **Test after changes** - Any prompt or model change needs re-evaluation
5. **Investigate failures** - Look at specific failed test cases, not just metrics
6. **Document baselines** - Know what "normal" looks like for your system
7. **Set alerts** - Define thresholds that trigger review

---

## Getting Help

- **Debug Info**: Expand the 🔧 Debug Info section in the sidebar to see database connection status
- **Logs**: On Streamlit Cloud, click "Manage app" to view detailed logs
- **Issues**: Report bugs at the project's GitHub repository

---

*Last updated: January 2026*
