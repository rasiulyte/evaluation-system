#!/usr/bin/env python3
"""
Quick Start Guide: Verify the evaluation system structure without needing API key.

This script demonstrates:
1. Loading test cases
2. Examining prompts
3. Calculating metrics on sample data
4. Exploring the system architecture
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from metrics import MetricsCalculator


def main():
    print("="*70)
    print("HALLUCINATION DETECTION EVALUATION SYSTEM - QUICK START")
    print("="*70)
    
    # 1. Load and verify test cases
    print("\n[1/5] Loading test cases...")
    try:
        with open("data/test_cases/ground_truth.json") as f:
            training_cases = json.load(f)
        
        with open("data/test_cases/regression.json") as f:
            regression_cases = json.load(f)
        
        print(f"✓ Test cases loaded")
        print(f"  - Training cases: {len(training_cases)}")
        print(f"  - Regression cases: {len(regression_cases)}")
        print(f"  - Total: {len(training_cases) + len(regression_cases)}")
        
    except Exception as e:
        print(f"✗ Error loading test cases: {e}")
        return
    
    # 2. Verify test case composition
    print("\n[2/5] Test case composition...")
    try:
        failure_modes = {}
        for case in training_cases + regression_cases:
            mode = case.get("failure_mode", "unknown")
            if mode not in failure_modes:
                failure_modes[mode] = 0
            failure_modes[mode] += 1
        
        print("✓ Cases by failure mode:")
        for mode, count in sorted(failure_modes.items()):
            print(f"  - {mode}: {count} cases")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 3. Load and verify prompts
    print("\n[3/5] Loading prompts...")
    try:
        with open("prompts/prompt_registry.json") as f:
            registry = json.load(f)
        
        prompts = registry.get("prompts", [])
        print(f"✓ Prompts available: {len(prompts)}")
        for prompt in prompts:
            status = "✓" if prompt["status"] == "active" else "○"
            print(f"  {status} {prompt['id']}: {prompt['description']}")
        
        print(f"\n  Production prompt: {registry.get('current_production', 'not set')}")
        
    except Exception as e:
        print(f"✗ Error loading prompts: {e}")
        return
    
    # 4. Sample test cases and metrics
    print("\n[4/5] Sample data and metrics...")
    try:
        print("✓ Sample test cases:")
        
        # Show one from each failure mode
        shown_modes = set()
        for case in training_cases:
            mode = case["failure_mode"]
            if mode not in shown_modes:
                print(f"\n  FM: {mode} (ID: {case['id']})")
                print(f"    Context: {case['context'][:50]}...")
                print(f"    Response: {case['response'][:50]}...")
                print(f"    Label: {case['label']} | Difficulty: {case['difficulty']}")
                shown_modes.add(mode)
                if len(shown_modes) >= 3:
                    print(f"    ... and {len(failure_modes) - 3} more failure modes")
                    break
        
        # Calculate metrics on simulated results
        print(f"\n✓ Metrics calculation (simulated results):")
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
        primary = calc.primary_metrics()
        
        print("  Primary metrics:")
        for name, result in primary.items():
            status = "✓" if result.meets_target else "✗"
            print(f"    {status} {result.name.upper()}: {result.value:.3f} (target: {result.target})")
        
        is_ready, details = calc.production_ready()
        print(f"\n  Production Ready: {'✓ YES' if is_ready else '✗ NO'}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # 5. System status
    print("\n[5/5] System status...")
    try:
        files_to_check = [
            ("docs/DESIGN.md", "Design Document"),
            ("docs/METRICS.md", "Metrics Guide"),
            ("docs/FAILURE_MODES.md", "Failure Modes"),
            ("docs/PROMPTING_STRATEGIES.md", "Prompting Strategies"),
            ("src/metrics.py", "Metrics Module"),
            ("src/evaluator.py", "Evaluator Module"),
            ("src/ab_testing.py", "A/B Testing Module"),
            ("src/monitoring.py", "Monitoring Module"),
            ("src/regression.py", "Regression Module"),
            ("src/dashboard.py", "Dashboard"),
            ("tests/test_metrics.py", "Test Suite"),
            ("README.md", "README"),
            (".github/copilot-instructions.md", "AI Agent Instructions"),
        ]
        
        print("✓ Project files:")
        all_exist = True
        for filepath, description in files_to_check:
            exists = Path(filepath).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {description}")
            if not exists:
                all_exist = False
        
        if all_exist:
            print("\n✓ ALL FILES PRESENT")
        else:
            print("\n⚠ Some files missing")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "="*70)
    print("✓ SYSTEM READY FOR DEVELOPMENT")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the documentation:")
    print("   - docs/DESIGN.md: System architecture")
    print("   - docs/METRICS.md: Understanding each metric")
    print("   - docs/FAILURE_MODES.md: Test case categories")
    print("   - docs/PROMPTING_STRATEGIES.md: Prompt comparison")
    print("\n2. Set up your environment:")
    print("   pip install -r requirements.txt")
    print("   export OPENAI_API_KEY='your-key'")
    print("\n3. Run an evaluation:")
    print("   python -c \"\"\"")
    print("   from src.evaluator import Evaluator")
    print("   from src.metrics import MetricsCalculator")
    print("   # See README.md for complete example")
    print("   \"\"\"")
    print("\n4. View the dashboard:")
    print("   streamlit run src/dashboard.py")
    print("\n5. Read the AI Agent guide:")
    print("   .github/copilot-instructions.md")


if __name__ == "__main__":
    main()
