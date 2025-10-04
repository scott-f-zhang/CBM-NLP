#!/usr/bin/env python3
"""Learning rate finder for essay dataset across different models.

This script performs quick learning rate tests (1-3 epochs) to find optimal
learning rates for different model backbones on the essay dataset.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Ensure project root on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import (
    get_cbm_standard,
    get_cbm_joint,
)
from main.config.defaults import RunConfig


def get_learning_rate_candidates(model_name: str) -> List[float]:
    """Get learning rate candidates for a specific model."""
    base_lrs = {
        'bert-base-uncased': [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
        'roberta-base': [1e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5],
        'gpt2': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        'lstm': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    }
    return base_lrs.get(model_name, [1e-6, 1e-5, 2e-5, 5e-5])


def quick_lr_test(model_name: str, pipeline: str, variant: str = 'manual', 
                  num_epochs: int = 3) -> Dict[float, Dict]:
    """
    Quick learning rate test for a specific model and pipeline.
    
    Args:
        model_name: Model backbone name
        pipeline: 'standard' or 'joint'
        variant: Dataset variant ('manual' or 'generated')
        num_epochs: Number of epochs for quick test (default: 3)
    
    Returns:
        Dict mapping learning rate to results
    """
    print(f"\n=== Testing {pipeline.upper()} with {model_name} ===")
    
    lr_candidates = get_learning_rate_candidates(model_name)
    results = {}
    
    # Base configuration
    base_config = {
        'model_name': model_name,
        'dataset': 'essay',
        'variant': variant,
        'num_epochs': num_epochs,
        'max_len': 512,
        'batch_size': 8,
    }
    
    for lr in lr_candidates:
        print(f"Testing LR: {lr:.0e}")
        
        try:
            if pipeline == 'standard':
                result = get_cbm_standard(optimizer_lr=lr, **base_config)
                # Standard pipeline returns list of (acc, f1) tuples
                if isinstance(result, list) and len(result) > 0:
                    task_acc, task_f1 = result[0]
                    concept_acc, concept_f1 = 0.0, 0.0
                else:
                    task_acc, task_f1 = 0.0, 0.0
                    concept_acc, concept_f1 = 0.0, 0.0
            else:  # joint
                result = get_cbm_joint(optimizer_lr=lr, **base_config)
                # Joint pipeline returns dict with 'task' and 'concept' keys
                if isinstance(result, dict):
                    task_scores = result.get('task', [])
                    concept_scores = result.get('concept', [])
                    
                    if task_scores and len(task_scores) > 0:
                        task_acc, task_f1 = task_scores[0]
                    else:
                        task_acc, task_f1 = 0.0, 0.0
                        
                    if concept_scores and len(concept_scores) > 0:
                        concept_acc, concept_f1 = concept_scores[0]
                    else:
                        concept_acc, concept_f1 = 0.0, 0.0
                else:
                    task_acc, task_f1 = 0.0, 0.0
                    concept_acc, concept_f1 = 0.0, 0.0
            
            results[lr] = {
                'task_acc': task_acc,
                'task_f1': task_f1,
                'concept_acc': concept_acc,
                'concept_f1': concept_f1,
                'combined_score': task_acc + concept_acc,  # Simple combination
            }
            
            print(f"  Task: {task_acc:.4f}/{task_f1:.4f}, "
                  f"Concept: {concept_acc:.4f}/{concept_f1:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[lr] = {
                'task_acc': 0.0,
                'task_f1': 0.0,
                'concept_acc': 0.0,
                'concept_f1': 0.0,
                'combined_score': 0.0,
            }
    
    return results


def find_best_lr(results: Dict[float, Dict], metric: str = 'combined_score') -> Tuple[float, Dict]:
    """Find the best learning rate based on specified metric."""
    if not results:
        return None, {}
    
    best_lr = max(results.keys(), key=lambda lr: results[lr][metric])
    best_result = results[best_lr]
    
    return best_lr, best_result


def run_lr_finder():
    """Run learning rate finder for all models and pipelines."""
    
    # Models to test
    models = ['bert-base-uncased', 'roberta-base', 'gpt2', 'lstm']
    
    # Pipelines to test
    pipelines = ['standard', 'joint']
    
    # Results storage
    all_results = {}
    
    print("=" * 60)
    print("LEARNING RATE FINDER FOR ESSAY DATASET")
    print("=" * 60)
    
    for model in models:
        all_results[model] = {}
        
        for pipeline in pipelines:
            print(f"\n{'='*20} {model.upper()} - {pipeline.upper()} {'='*20}")
            
            # Run quick test
            results = quick_lr_test(model, pipeline, variant='manual', num_epochs=3)
            
            # Find best learning rate
            best_lr, best_result = find_best_lr(results, 'combined_score')
            
            all_results[model][pipeline] = {
                'all_results': results,
                'best_lr': best_lr,
                'best_result': best_result,
            }
            
            if best_lr:
                print(f"\nBest LR for {model} ({pipeline}): {best_lr:.0e}")
                print(f"Best scores: Task={best_result['task_acc']:.4f}, "
                      f"Concept={best_result['concept_acc']:.4f}")
            else:
                print(f"\nNo valid results for {model} ({pipeline})")
    
    return all_results


def save_results(results: Dict, output_file: str = "lr_finder_results.csv"):
    """Save learning rate finder results to CSV."""
    
    rows = []
    
    for model, model_results in results.items():
        for pipeline, pipeline_results in model_results.items():
            best_lr = pipeline_results.get('best_lr')
            best_result = pipeline_results.get('best_result', {})
            
            if best_lr:
                rows.append({
                    'model': model,
                    'pipeline': pipeline,
                    'best_lr': best_lr,
                    'task_acc': best_result.get('task_acc', 0.0),
                    'task_f1': best_result.get('task_f1', 0.0),
                    'concept_acc': best_result.get('concept_acc', 0.0),
                    'concept_f1': best_result.get('concept_f1', 0.0),
                    'combined_score': best_result.get('combined_score', 0.0),
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


def print_summary(results: Dict):
    """Print a summary of the best learning rates."""
    
    print("\n" + "=" * 60)
    print("LEARNING RATE SUMMARY")
    print("=" * 60)
    
    for model, model_results in results.items():
        print(f"\n{model.upper()}:")
        
        for pipeline, pipeline_results in model_results.items():
            best_lr = pipeline_results.get('best_lr')
            best_result = pipeline_results.get('best_result', {})
            
            if best_lr:
                print(f"  {pipeline.upper()}: LR={best_lr:.0e}, "
                      f"Task={best_result.get('task_acc', 0.0):.4f}, "
                      f"Concept={best_result.get('concept_acc', 0.0):.4f}")
            else:
                print(f"  {pipeline.upper()}: No valid results")


def main():
    """Main function to run learning rate finder."""
    
    # Run learning rate finder
    results = run_lr_finder()
    
    # Save results to test_results directory
    import os
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main", "tests")
    test_results_dir = os.path.join(tests_dir, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)
    output_file = os.path.join(test_results_dir, "lr_finder_results.csv")
    
    df = save_results(results, output_file)
    
    # Print summary
    print_summary(results)
    
    # Print detailed results table
    print("\n" + "=" * 60)
    print("DETAILED RESULTS TABLE")
    print("=" * 60)
    print(df.to_string(index=False, float_format='%.4f'))
    
    return results, df


if __name__ == "__main__":
    results, df = main()
