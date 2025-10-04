#!/usr/bin/env python3
"""Run essay experiments with both learning rate types for comparison.

This script runs experiments with both dataset-optimal and universal learning rates
to compare their performance.
"""

import os
import sys
import subprocess
from datetime import datetime

# Ensure project root on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_experiment_with_lr_type(lr_type: str):
    """Run experiment with specified learning rate type."""
    
    print("=" * 80)
    print(f"RUNNING ESSAY EXPERIMENTS WITH {lr_type.upper()} LEARNING RATES")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Learning rate type: {lr_type}")
    print("=" * 80)
    
    # Change to project directory
    os.chdir(ROOT_DIR)
    
    try:
        # Import and modify the LR_TYPE in test_essay
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_essay", "main/tests/test_essay.py")
        test_essay = importlib.util.module_from_spec(spec)
        
        # Modify the LR_TYPE before loading
        with open("main/tests/test_essay.py", "r") as f:
            content = f.read()
        
        # Replace the LR_TYPE setting
        modified_content = content.replace(
            f'LR_TYPE = "dataset_optimal"',
            f'LR_TYPE = "{lr_type}"'
        )
        
        # Write temporary file
        temp_file = f"main/tests/test_essay_{lr_type}.py"
        with open(temp_file, "w") as f:
            f.write(modified_content)
        
        # Run the modified script
        result = subprocess.run([
            sys.executable, temp_file
        ], capture_output=True, text=True, timeout=3600)
        
        # Clean up temporary file
        os.remove(temp_file)
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            print("\n📊 Results:")
            print(result.stdout)
            return True
        else:
            print("❌ Experiment failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Experiment timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return False
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return True

def compare_results():
    """Compare results from both learning rate types."""
    
    print("\n" + "=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)
    
    try:
        import pandas as pd
        
        # Load results
        dataset_optimal_file = "main/tests/test_results/result_essay_dataset_optimal.csv"
        universal_file = "main/tests/test_results/result_essay_universal.csv"
        
        if os.path.exists(dataset_optimal_file) and os.path.exists(universal_file):
            df_optimal = pd.read_csv(dataset_optimal_file)
            df_universal = pd.read_csv(universal_file)
            
            print("📊 Performance Comparison:")
            print("\nDataset-Optimal Learning Rates:")
            print(df_optimal[['model', 'function', 'score_avg']].to_string(index=False))
            
            print("\nUniversal Learning Rates:")
            print(df_universal[['model', 'function', 'score_avg']].to_string(index=False))
            
        else:
            print("⚠️  Results files not found. Run experiments first.")
            
    except Exception as e:
        print(f"❌ Error comparing results: {e}")

def main():
    """Main function to run both experiments and compare results."""
    
    print("🚀 Starting Essay Dataset Learning Rate Comparison")
    print("This will run experiments with both learning rate types.")
    
    # Run with dataset-optimal learning rates
    print("\n" + "="*60)
    print("PHASE 1: Dataset-Optimal Learning Rates")
    print("="*60)
    success1 = run_experiment_with_lr_type("dataset_optimal")
    
    # Run with universal learning rates
    print("\n" + "="*60)
    print("PHASE 2: Universal Learning Rates")
    print("="*60)
    success2 = run_experiment_with_lr_type("universal")
    
    # Compare results
    if success1 and success2:
        compare_results()
        print("\n🎉 Comparison completed successfully!")
        print("📁 Results saved to:")
        print("  - main/tests/test_results/result_essay_dataset_optimal.csv")
        print("  - main/tests/test_results/result_essay_universal.csv")
    else:
        print("\n💥 Some experiments failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
