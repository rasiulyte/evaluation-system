#!/usr/bin/env python3
"""
Quick Start Guide: Run this script to verify the evaluation system is set up correctly.

This script demonstrates:
1. Loading test cases
2. Running an evaluation with a mock response
3. Calculating metrics
4. Checking production readiness
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import Evaluator
from metrics import MetricsCalculator


def main():
    print("="*70)
    print("HALLUCINATION DETECTION EVALUATION SYSTEM - QUICK START")
    print("="*70)
    
    # 1. Initialize evaluator
    print("\n[1/4] Initializing evaluator...")
    try:
        evaluator = Evaluator()
        print(f"✓ Evaluator initialized")
        print(f"  - Test cases loaded: {len(evaluator.list_test_cases())}")
        print(f"  - Prompts available: {', '.join(evaluator.list_prompts())}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 2. Display sample test case
    print("\n[2/4] Sample test case...")
    try:
        test_case = evaluator.get_test_case("FM1_001")
        print(f"✓ Test case FM1_001 (Factual Addition - Easy)")
        print(f"  Context: {test_case['context'][:60]}...")
        print(f"  Response: {test_case['response'][:60]}...")
        print(f"  Label: {test_case['label']}")
        print(f"  Difficulty: {test_case['difficulty']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 3. Create sample results for demonstration
    print("\n[3/4] Calculating metrics with sample data...")
    try:
        # Simulate 10 evaluations
        y_true = [
            "hallucination", "grounded", "hallucination", "grounded", 
            "hallucination", "grounded", "hallucination", "grounded",
            "hallucination", "grounded"
        ]
        y_pred = [
            "hallucination", "grounded", "hallucination", "grounded",
            "grounded", "grounded", "hallucination", "grounded",
            "hallucination", "grounded"
        ]
        
        calc = MetricsCalculator(y_true, y_pred)
        metrics = calc.primary_metrics()
        
        print("✓ Primary metrics calculated:")
        for name, result in metrics.items():
            status = "✓" if result.meets_target else "✗"
            print(f"  {status} {result.name.upper()}: {result.value:.3f}")
            print(f"     Target: {result.target}")
            print(f"     Interpretation: {result.interpretation}")
        
        # Check production readiness
        is_ready, details = calc.production_ready()
        print(f"\n  Production Ready: {'✓ YES' if is_ready else '✗ NO'}")
        print(f"  Thresholds: F1={details['f1_pass']}, TNR={details['tnr_pass']}, Bias={details['bias_pass']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 4. Display test case breakdown
    print("\n[4/4] Test set composition...")
    try:
        failure_modes = {}
        for case in evaluator.test_cases.values():
            mode = case.get("failure_mode", "unknown")
            if mode not in failure_modes:
                failure_modes[mode] = 0
            failure_modes[mode] += 1
        
        print("✓ Test cases by failure mode:")
        for mode, count in sorted(failure_modes.items()):
            print(f"  - {mode}: {count} cases")
        
        # Check split
        training_count = sum(1 for c in evaluator.test_cases.values() if not c["id"].startswith("REG_"))
        regression_count = sum(1 for c in evaluator.test_cases.values() if c["id"].startswith("REG_"))
        print(f"\n✓ Data split:")
        print(f"  - Training (ground_truth.json): {training_count} cases")
        print(f"  - Held-out (regression.json): {regression_count} cases")
        print(f"  - Total: {len(evaluator.test_cases)} cases")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "="*70)
    print("✓ SYSTEM READY")
    print("="*70)
    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run evaluation: python -c \"from src.evaluator import Evaluator; ...\"")
    print("3. View dashboard: streamlit run src/dashboard.py")
    print("\nFor more information, see:")
    print("- README.md: Project overview and quick start")
    print("- docs/DESIGN.md: Methodology and architecture")
    print("- docs/METRICS.md: All metrics explained")
    print("- .github/copilot-instructions.md: AI agent guide")


if __name__ == "__main__":
    main()
