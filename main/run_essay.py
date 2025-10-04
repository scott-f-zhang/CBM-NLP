#!/usr/bin/env python3
"""Run complete essay dataset experiments with optimal learning rates.

This script provides a user-friendly interface to run essay dataset experiments
using the optimal learning rates found by the learning rate finder on the 7:2:1 data split.

It directly imports and runs the test_essay module for better performance and error handling.

Usage:
    python run_essay.py                    # Default: no early stopping
    python run_essay.py --early-stopping   # Enable early stopping
    python run_essay.py --no-early-stopping # Explicitly disable early stopping
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure project root on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_experiments(use_early_stopping=False):
    """Run essay dataset experiments with optimal learning rates.
    
    Args:
        use_early_stopping (bool): Whether to use early stopping (default: False)
    """
    
    print("=" * 80)
    print("ESSAY DATASET EXPERIMENTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data split: 7:2:1 (train:dev:test)")
    print(f"Models: BERT, RoBERTa, GPT2, LSTM")
    print(f"Pipelines: PLMs (Standard), CBE-PLMs (Joint)")
    print(f"Early stopping: {'Enabled' if use_early_stopping else 'Disabled (Default)'}")
    print("=" * 80)
    
    try:
        print("\n🚀 Starting essay experiments...")
        
        if use_early_stopping:
            # Import and run the test_essay module (with early stopping)
            from tests.test_essay import main as test_essay_main
            test_essay_main()
        else:
            # Import and run the test_essay_no_early_stopping module
            from tests.test_essay_no_early_stopping import main as test_essay_main
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
    parser = argparse.ArgumentParser(
        description="Run essay dataset experiments with configurable early stopping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_essay.py                    # Default: no early stopping (recommended)
  python run_essay.py --early-stopping   # Enable early stopping
  python run_essay.py --no-early-stopping # Explicitly disable early stopping
        """
    )
    
    # Early stopping options (mutually exclusive)
    early_stopping_group = parser.add_mutually_exclusive_group()
    early_stopping_group.add_argument(
        '--early-stopping', 
        action='store_true',
        help='Enable early stopping (patience=5 epochs)'
    )
    early_stopping_group.add_argument(
        '--no-early-stopping', 
        action='store_true',
        help='Explicitly disable early stopping (default behavior)'
    )
    
    args = parser.parse_args()
    
    # Determine early stopping setting
    use_early_stopping = args.early_stopping
    
    success = run_experiments(use_early_stopping=use_early_stopping)
    
    if success:
        print("\n🎉 All experiments completed successfully!")
        if use_early_stopping:
            print("📁 Results saved to: tests/test_results/result_essay_early_stopping_dataset_optimal.csv")
        else:
            print("📁 Results saved to: tests/test_results/result_essay_no_early_stopping_dataset_optimal.csv")
        print("📊 Analysis summary: tests/test_results/lr_analysis_summary.md")
        print("📊 Early stopping analysis: tests/test_results/early_stopping_analysis.md")
    else:
        print("\n💥 Experiments failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
