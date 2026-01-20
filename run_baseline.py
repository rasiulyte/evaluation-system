#!/usr/bin/env python
"""Quick baseline evaluation script"""

from src.evaluator import Evaluator
from src.metrics import MetricsCalculator

print('=' * 70)
print('HALLUCINATION DETECTION - BASELINE EVALUATION')
print('=' * 70)

evaluator = Evaluator(model='gpt-4')
all_cases = evaluator.list_test_cases()
training_cases = [c for c in all_cases if not c.startswith('REG_')]

print(f'\nRunning on {len(training_cases)} training cases')
print('Evaluating first 5 cases...\n')

results = evaluator.evaluate_batch('v1_zero_shot', test_case_ids=training_cases[:5])

print(f'Got {len(results)} results\n')

# Filter out errors
valid_results = [r for r in results if 'error' not in r]
error_results = [r for r in results if 'error' in r]

if error_results:
    print(f'ERRORS ({len(error_results)}):')
    for r in error_results:
        print(f'  - {r.get("test_case_id", "unknown")}: {r["error"]}')
    print()

print(f'RESULTS ({len(valid_results)}):')
for i, r in enumerate(valid_results, 1):
    match = r['prediction'] == r['ground_truth']
    status = 'PASS' if match else 'FAIL'
    print(f'  {i}. [{status}] {r["test_case_id"]}: pred={r["prediction"]}, truth={r["ground_truth"]}')

if valid_results:
    y_true = [r['ground_truth'] for r in valid_results]
    y_pred = [r['prediction'] for r in valid_results]

    calc = MetricsCalculator(y_true, y_pred)
    metrics = calc.primary_metrics()

    print('\nMETRICS:')
    for name, m in metrics.items():
        print(f'  {name}: {m.value:.3f}')

    evaluator.save_results(results, run_name='v1_baseline_test')
    print('\nSaved to data/results/')
else:
    print('No valid results to process')
