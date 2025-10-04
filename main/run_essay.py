#!/usr/bin/env python3
"""Run complete essay dataset experiments with optimal learning rates.

This script provides a user-friendly interface to run essay dataset experiments
using the optimal learning rates found by the learning rate finder on the 7:2:1 data split.

It directly imports and runs the test_essay module for better performance and error handling.
"""

import os
import sys
from datetime import datetime

# Ensure project root on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_experiments():
    """Run essay dataset experiments with optimal learning rates."""
    
    print("=" * 80)
    print("ESSAY DATASET EXPERIMENTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data split: 7:2:1 (train:dev:test)")
    print(f"Models: BERT, RoBERTa, GPT2, LSTM")
    print(f"Pipelines: PLMs (Standard), CBE-PLMs (Joint)")
    print("=" * 80)
    
    try:
        print("\n🚀 Starting essay experiments...")
        
        # Import and run the test_essay module directly
        from tests.test_essay import main as test_essay_main
        test_essay_main()
        
        print("✅ Experiments completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return True

def main():
    """Main entry point."""
    success = run_experiments()
    
    if success:
        print("\n🎉 All experiments completed successfully!")
        print("📁 Results saved to: tests/test_results/result_essay.csv")
        print("📊 Analysis summary: tests/test_results/lr_analysis_summary.md")
    else:
        print("\n💥 Experiments failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
