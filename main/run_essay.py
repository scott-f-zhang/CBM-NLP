#!/usr/bin/env python3
"""Run complete essay dataset experiments with optimal learning rates.

This script runs the full essay dataset experiments using the optimal learning rates
found by the learning rate finder on the 7:2:1 data split.
"""

import os
import sys
import subprocess
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
    
    # Change to project directory
    os.chdir(ROOT_DIR)
    
    # Run the essay experiments
    try:
        print("\n🚀 Starting essay experiments...")
        result = subprocess.run([
            sys.executable, "tests/test_essay.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Experiments completed successfully!")
            print("\n📊 Results:")
            print(result.stdout)
        else:
            print("❌ Experiments failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Experiments timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
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
