#!/usr/bin/env python3
"""
Simple test script to verify model loading works.
"""

import os
import sys
import torch
from model_loader import load_model_and_tokenizer

# Add main module to path for loading pickled models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_lstm():
    print("Testing LSTM model loading...")
    
    # Load tokenizer and model
    model, tokenizer, hidden_size = load_model_and_tokenizer('lstm')
    print(f"✓ Model loaded successfully, hidden_size: {hidden_size}")
    
    # Test model loading
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "saved_models", "original", "lstm", "lstm_model_standard.pth")
    head_path = os.path.join(project_root, "saved_models", "original", "lstm", "lstm_classifier_standard.pth")
    
    print(f"Model path: {model_path}")
    print(f"Head path: {head_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Head exists: {os.path.exists(head_path)}")
    
    if os.path.exists(model_path) and os.path.exists(head_path):
        device = torch.device("cpu")
        trained_model = torch.load(model_path, weights_only=False, map_location=device)
        trained_head = torch.load(head_path, weights_only=False, map_location=device)
        print("✓ Trained models loaded successfully")
        
        # Test tokenization
        text = "This restaurant has amazing food and great service!"
        inputs = tokenizer(text, padding='max_length', truncation=True, 
                          max_length=128, return_tensors='pt')
        print(f"✓ Tokenization successful, input shape: {inputs['input_ids'].shape}")
        
        return True
    else:
        print("✗ Model files not found")
        return False

if __name__ == "__main__":
    test_lstm()
