#!/usr/bin/env python
"""Test API connectivity only"""

from src.evaluator import Evaluator

print("Testing API connection...")

try:
    evaluator = Evaluator(model='gpt-4')
    print("✓ Evaluator initialized")
    
    # Try one simple call
    print("\nTesting single evaluation...")
    cases = evaluator.list_test_cases()
    test_case_id = [c for c in cases if not c.startswith('REG_')][0]
    
    print(f"Evaluating: {test_case_id}")
    result = evaluator.evaluate_single(test_case_id, 'v1_zero_shot')
    
    if 'error' in result:
        print(f"✗ ERROR: {result['error']}")
    else:
        print(f"✓ Success!")
        print(f"  Prediction: {result.get('prediction', 'N/A')}")
        print(f"  Ground truth: {result.get('ground_truth', 'N/A')}")
        print(f"  Correct: {result.get('prediction') == result.get('ground_truth')}")
        
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}")
    print(f"  {e}")
