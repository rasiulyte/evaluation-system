#!/usr/bin/env python
"""Summary of project status"""

import json
import os
from pathlib import Path

print("=" * 70)
print("EVALUATION SYSTEM - PROJECT STATUS SUMMARY")
print("=" * 70)

# Check test data
print("\n📊 TEST DATA:")
with open("data/test_cases/ground_truth.json") as f:
    training = json.load(f)
with open("data/test_cases/regression.json") as f:
    regression = json.load(f)

print(f"  • Training cases: {len(training)}")
print(f"  • Regression (held-out): {len(regression)}")
print(f"  • Total: {len(training) + len(regression)}")

# Check results
results_dir = Path("data/results")
result_files = list(results_dir.glob("*.json"))
print(f"\n✅ RESULTS FILES ({len(result_files)}):")
for f in sorted(result_files)[-3:]:
    size = os.path.getsize(f) / 1024
    print(f"  • {f.name} ({size:.1f} KB)")

if result_files:
    latest = sorted(result_files)[-1]
    with open(latest) as f:
        results = json.load(f)
    print(f"\n📈 LATEST RESULTS ({len(results)} cases):")
    correct = sum(1 for r in results if r.get('prediction') == r.get('ground_truth'))
    print(f"  • Accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")

# Check prompts
print(f"\n📝 PROMPTS AVAILABLE:")
with open("prompts/prompt_registry.json") as f:
    registry = json.load(f)
for p in registry["prompts"]:
    status = "📍 PRODUCTION" if p["id"] == registry["production_prompt"] else ""
    print(f"  • {p['id']}: {p['description'][:40]}... {status}")

print("\n" + "=" * 70)
print("🚀 NEXT STEPS:")
print("=" * 70)
print("""
1. VIEW DASHBOARD:
   http://localhost:8501 (currently running!)

2. EXPAND BASELINE (20 cases):
   python -c "
from src.evaluator import Evaluator
from src.metrics import MetricsCalculator

evaluator = Evaluator()
cases = [c for c in evaluator.list_test_cases() if not c.startswith('REG_')]
results = evaluator.evaluate_batch('v1_zero_shot', test_case_ids=cases[:20])

y_true = [r['ground_truth'] for r in results]
y_pred = [r['prediction'] for r in results]
calc = MetricsCalculator(y_true, y_pred)

print('\\nBASELINE (20 cases):')
for name, m in calc.primary_metrics().items():
    print(f'  {name}: {m.value:.3f}')

evaluator.save_results(results, run_name='v1_baseline_20')
"

3. A/B TEST PROMPTS (v1 vs v3):
   python -c "
from src.evaluator import Evaluator
from src.ab_testing import ABTester

evaluator = Evaluator()
cases = [c for c in evaluator.list_test_cases() if not c.startswith('REG_')][:15]

print('Comparing prompts...')
r1 = evaluator.evaluate_batch('v1_zero_shot', test_case_ids=cases)
r3 = evaluator.evaluate_batch('v3_chain_of_thought', test_case_ids=cases)

tester = ABTester()
result = tester.run_test(r1, r3, 'v1_zero_shot', 'v3_chain_of_thought')
print(f'Winner: {result[\"winner_prompt\"]}')
tester.set_production_prompt(result['winner_prompt'])
"

4. REGRESSION TEST (held-out set):
   python -c "
from src.evaluator import Evaluator
from src.regression import RegressionTester

evaluator = Evaluator()
reg_cases = [c for c in evaluator.list_test_cases() if c.startswith('REG_')]
results = evaluator.evaluate_batch('v1_zero_shot', test_case_ids=reg_cases)

tester = RegressionTester()
passed, details = tester.run_regression_test(results)
print(tester.format_report(passed, details))
"
""")
