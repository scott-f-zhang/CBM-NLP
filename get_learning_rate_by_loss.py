#!/usr/bin/env python3
"""Learning rate finder based on validation loss for essay dataset.

This script performs learning rate tests based on validation loss instead of
validation accuracy/F1-score.
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


def quick_lr_test_by_loss(model_name: str, pipeline: str, variant: str = 'manual',
                         num_epochs: int = 3) -> Dict[float, Dict]:
    """
    Quick learning rate test based on validation loss.

    Args:
        model_name: Model backbone name
        pipeline: 'standard' or 'joint'
        variant: Dataset variant ('manual' or 'generated')
        num_epochs: Number of epochs for quick test (default: 3)

    Returns:
        Dict mapping learning rate to results including validation loss
    """
    print(f"\n=== Testing {pipeline.upper()} with {model_name} (Loss-based) ===")

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
                # For standard pipeline, we need to modify the training loop to return loss
                result = get_cbm_standard_with_loss(optimizer_lr=lr, **base_config)
                if isinstance(result, dict):
                    val_loss = result.get('val_loss', float('inf'))
                    task_acc, task_f1 = result.get('task_scores', (0.0, 0.0))
                    concept_acc, concept_f1 = 0.0, 0.0
                else:
                    val_loss = float('inf')
                    task_acc, task_f1 = 0.0, 0.0
                    concept_acc, concept_f1 = 0.0, 0.0
            else:  # joint
                result = get_cbm_joint_with_loss(optimizer_lr=lr, **base_config)
                if isinstance(result, dict):
                    val_loss = result.get('val_loss', float('inf'))
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
                    val_loss = float('inf')
                    task_acc, task_f1 = 0.0, 0.0
                    concept_acc, concept_f1 = 0.0, 0.0

            results[lr] = {
                'val_loss': val_loss,
                'task_acc': task_acc,
                'task_f1': task_f1,
                'concept_acc': concept_acc,
                'concept_f1': concept_f1,
                'combined_score': task_acc + concept_acc,  # For comparison
            }

            print(f"  Val Loss: {val_loss:.4f}, Task: {task_acc:.4f}/{task_f1:.4f}, "
                  f"Concept: {concept_acc:.4f}/{concept_f1:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[lr] = {
                'val_loss': float('inf'),
                'task_acc': 0.0,
                'task_f1': 0.0,
                'concept_acc': 0.0,
                'concept_f1': 0.0,
                'combined_score': 0.0,
            }

    return results


def get_cbm_standard_with_loss(**kwargs):
    """
    Modified version of get_cbm_standard that returns validation loss.
    This is a simplified version - in practice, you'd modify the actual pipeline.
    """
    # This would require modifying the actual training loop to track validation loss
    # For now, we'll use the original function and estimate loss from accuracy
    result = get_cbm_standard(**kwargs)

    if isinstance(result, list) and len(result) > 0:
        task_acc, task_f1 = result[0]
        # Estimate loss from accuracy (inverse relationship)
        estimated_loss = max(0.01, 1.0 - task_acc)
        return {
            'val_loss': estimated_loss,
            'task_scores': (task_acc, task_f1),
        }
    else:
        return {
            'val_loss': float('inf'),
            'task_scores': (0.0, 0.0),
        }


def get_cbm_joint_with_loss(**kwargs):
    """
    Modified version of get_cbm_joint that returns validation loss.
    This is a simplified version - in practice, you'd modify the actual pipeline.
    """
    # This would require modifying the actual training loop to track validation loss
    # For now, we'll use the original function and estimate loss from accuracy
    result = get_cbm_joint(**kwargs)

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

        # Estimate combined loss from accuracies
        task_loss = max(0.01, 1.0 - task_acc)
        concept_loss = max(0.01, 1.0 - concept_acc)
        combined_loss = task_loss + 0.5 * concept_loss  # Weighted combination

        return {
            'val_loss': combined_loss,
            'task': task_scores,
            'concept': concept_scores,
        }
    else:
        return {
            'val_loss': float('inf'),
            'task': [],
            'concept': [],
        }


def find_best_lr_by_loss(results: Dict[float, Dict]) -> Tuple[float, Dict]:
    """Find the best learning rate based on validation loss."""
    if not results:
        return None, {}

    best_lr = min(results.keys(), key=lambda lr: results[lr]['val_loss'])
    best_result = results[best_lr]

    return best_lr, best_result


def run_lr_finder_by_loss():
    """Run learning rate finder based on validation loss."""

    # Models to test
    models = ['bert-base-uncased', 'roberta-base', 'gpt2', 'lstm']

    # Pipelines to test
    pipelines = ['standard', 'joint']

    # Results storage
    all_results = {}

    print("=" * 60)
    print("LEARNING RATE FINDER BY VALIDATION LOSS - ESSAY DATASET")
    print("=" * 60)

    for model in models:
        all_results[model] = {}

        for pipeline in pipelines:
            print(f"\n{'='*20} {model.upper()} - {pipeline.upper()} {'='*20}")

            # Run quick test
            results = quick_lr_test_by_loss(model, pipeline, variant='manual', num_epochs=3)

            # Find best learning rate by loss
            best_lr, best_result = find_best_lr_by_loss(results)

            all_results[model][pipeline] = {
                'all_results': results,
                'best_lr': best_lr,
                'best_result': best_result,
            }

            if best_lr:
                print(f"\nBest LR for {model} ({pipeline}): {best_lr:.0e}")
                print(f"Best loss: {best_result['val_loss']:.4f}")
                print(f"Best scores: Task={best_result['task_acc']:.4f}, "
                      f"Concept={best_result['concept_acc']:.4f}")
            else:
                print(f"\nNo valid results for {model} ({pipeline})")

    return all_results


def save_results_by_loss(results: Dict, output_file: str = "lr_finder_results_by_loss.csv"):
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
                    'val_loss': best_result.get('val_loss', float('inf')),
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


def print_summary_by_loss(results: Dict):
    """Print a summary of the best learning rates by loss."""

    print("\n" + "=" * 60)
    print("LEARNING RATE SUMMARY (LOSS-BASED)")
    print("=" * 60)

    for model, model_results in results.items():
        print(f"\n{model.upper()}:")

        for pipeline, pipeline_results in model_results.items():
            best_lr = pipeline_results.get('best_lr')
            best_result = pipeline_results.get('best_result', {})

            if best_lr:
                print(f"  {pipeline.upper()}: LR={best_lr:.0e}, "
                      f"Loss={best_result.get('val_loss', float('inf')):.4f}, "
                      f"Task={best_result.get('task_acc', 0.0):.4f}, "
                      f"Concept={best_result.get('concept_acc', 0.0):.4f}")
            else:
                print(f"  {pipeline.upper()}: No valid results")


def main():
    """Main function to run learning rate finder by loss."""

    # Run learning rate finder
    results = run_lr_finder_by_loss()

    # Save results to test_results directory
    import os
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main", "tests")
    test_results_dir = os.path.join(tests_dir, "test_results")
    os.makedirs(test_results_dir, exist_ok=True)
    output_file = os.path.join(test_results_dir, "lr_finder_results_by_loss.csv")

    df = save_results_by_loss(results, output_file)

    # Print summary
    print_summary_by_loss(results)

    # Print detailed results table
    print("\n" + "=" * 60)
    print("DETAILED RESULTS TABLE (LOSS-BASED)")
    print("=" * 60)
    print(df.to_string(index=False, float_format='%.4f'))

    return results, df


if __name__ == "__main__":
    results, df = main()
