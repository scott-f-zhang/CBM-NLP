#!/usr/bin/env python3
"""Test Early Stopping functionality on essay dataset.

This script tests the Early Stopping implementation with a small number of epochs
to verify that it works correctly.
"""
import os
import sys

# Ensure project root on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import (
    get_cbm_standard,
    get_cbm_joint,
)


def test_early_stopping():
    """Test Early Stopping with essay dataset."""
    
    print("=" * 60)
    print("TESTING EARLY STOPPING ON ESSAY DATASET")
    print("=" * 60)
    
    # Test PLMs (Standard) with Early Stopping
    print("\n1. Testing PLMs (Standard) with Early Stopping...")
    print("-" * 50)
    
    result_plm = get_cbm_standard(
        model_name='bert-base-uncased',
        dataset='essay',
        variant='manual',
        num_epochs=20,  # Set high to test early stopping
        max_len=512,
        batch_size=8,
        optimizer_lr=2e-5,
    )
    
    print(f"\nPLMs Result: {result_plm}")
    
    # Test CBE-PLMs (Joint) with Early Stopping
    print("\n2. Testing CBE-PLMs (Joint) with Early Stopping...")
    print("-" * 50)
    
    result_cbm = get_cbm_joint(
        model_name='bert-base-uncased',
        dataset='essay',
        variant='manual',
        num_epochs=20,  # Set high to test early stopping
        max_len=512,
        batch_size=8,
        optimizer_lr=2e-5,
    )
    
    print(f"\nCBE-PLMs Result: {result_cbm}")
    
    print("\n" + "=" * 60)
    print("EARLY STOPPING TEST COMPLETED")
    print("=" * 60)
    print("Check the output above to verify:")
    print("1. Training stops early when validation accuracy doesn't improve")
    print("2. 'Early stopping at epoch X' message appears")
    print("3. Best model is still saved and used for testing")


if __name__ == "__main__":
    test_early_stopping()
