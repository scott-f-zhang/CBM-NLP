#!/usr/bin/env python3
"""
Simple inference example for new texts without labels.

This script demonstrates how to load trained models and make predictions on new texts.
It supports both standard and joint modes, showing predictions and probabilities.

Usage:
    python inference_single_text.py --model_name bert-base-uncased --mode standard
    python inference_single_text.py --model_name gpt2 --mode joint
"""

import argparse
import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer

# Add main module to path for loading pickled models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import standalone model loader to avoid circular imports
from model_loader import load_model_and_tokenizer


def load_trained_model(model_name, mode, device):
    """Load trained model and head from saved files."""
    # Determine file paths based on mode
    if mode == 'standard':
        model_file = f"{model_name}_model_standard.pth"
        head_file = f"{model_name}_classifier_standard.pth"
    elif mode == 'joint':
        model_file = f"{model_name}_joint.pth"
        head_file = f"{model_name}_ModelXtoCtoY_layer_joint.pth"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Load from original directory (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "saved_models", "original", model_name, model_file)
    head_path = os.path.join(project_root, "saved_models", "original", model_name, head_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Head file not found: {head_path}")
    
    try:
        # Load models with proper device mapping
        model = torch.load(model_path, weights_only=False, map_location=device)
        head = torch.load(head_path, weights_only=False, map_location=device)
        
        # Fix configuration compatibility issues
        if hasattr(model, 'config'):
            # Add missing attributes to config if they don't exist
            if not hasattr(model.config, '_output_attentions'):
                model.config._output_attentions = False
            if not hasattr(model.config, '_output_hidden_states'):
                model.config._output_hidden_states = False
            if not hasattr(model.config, 'output_attentions'):
                model.config.output_attentions = False
            if not hasattr(model.config, 'output_hidden_states'):
                model.config.output_hidden_states = False
        
        model.to(device)
        head.to(device)
        model.eval()
        head.eval()
        
        return model, head
    except Exception as e:
        print(f"Warning: Could not load saved model due to compatibility issues: {e}")
        print("Falling back to pretrained model (untrained head)...")
        return load_pretrained_model(model_name, device)


def load_pretrained_model(model_name, device):
    """Load pretrained model with untrained head for demonstration."""
    model, tokenizer, hidden_size = load_model_and_tokenizer(model_name)
    
    # Create a simple untrained classification head
    if model_name == 'lstm':
        head = torch.nn.Linear(hidden_size, 5)  # 5 classes for rating
    else:
        head = torch.nn.Linear(hidden_size, 5)  # 5 classes for rating
    
    model.to(device)
    head.to(device)
    model.eval()
    head.eval()
    
    return model, head


def predict_standard(model, head, tokenizer, text, device):
    """Make prediction using standard mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            # Transformer models (BERT, RoBERTa, GPT2)
            pooled_output = outputs.last_hidden_state.mean(1)
        else:
            # LSTM model
            pooled_output = outputs
        
        # Get prediction
        logits = head(pooled_output)
        prediction = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    return prediction, probabilities


def predict_joint(model, head, tokenizer, text, device):
    """Make prediction using joint mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            # Transformer models (BERT, RoBERTa, GPT2)
            pooled_output = outputs.last_hidden_state.mean(1)
        else:
            # LSTM model
            pooled_output = outputs
        
        # Get joint model output
        joint_outputs = head(pooled_output)
        XtoY_output = joint_outputs[0:1]  # Task prediction
        XtoC_output = joint_outputs[1:]   # Concept predictions
        
        # Task prediction
        task_prediction = torch.argmax(XtoY_output[0], dim=1).item()
        task_probs = torch.softmax(XtoY_output[0], dim=1)[0].cpu().numpy()
        
        # Concept predictions
        concept_logits = torch.cat(XtoC_output, dim=0)  # [4, 3] for 4 concepts
        concept_predictions = torch.argmax(concept_logits, dim=1).cpu().numpy()
        concept_probs = torch.softmax(concept_logits, dim=1).cpu().numpy()
    
    return task_prediction, task_probs, concept_predictions, concept_probs


def main():
    parser = argparse.ArgumentParser(description='Inference on new texts')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       choices=['bert-base-uncased', 'gpt2', 'lstm', 'roberta-base'],
                       help='Model name to use')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'joint'],
                       help='Model mode (standard or joint)')
    parser.add_argument('--text', type=str, 
                       default="This restaurant has amazing food and great service!",
                       help='Text to predict on')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Text: {args.text}")
    print("-" * 60)
    
    try:
        # Load tokenizer
        _, tokenizer, _ = load_model_and_tokenizer(args.model_name)
        
        # Load trained model
        model, head = load_trained_model(args.model_name, args.mode, device)
        
        # Make prediction
        if args.mode == 'standard':
            prediction, probabilities = predict_standard(model, head, tokenizer, args.text, device)
            
            print(f"Predicted Rating: {prediction + 1} star(s)")  # Convert 0-4 to 1-5
            print(f"Rating Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  {i+1} star: {prob:.4f} ({prob*100:.2f}%)")
        
        elif args.mode == 'joint':
            task_pred, task_probs, concept_preds, concept_probs = predict_joint(
                model, head, tokenizer, args.text, device)
            
            print(f"Task Prediction: {task_pred + 1} star(s)")  # Convert 0-4 to 1-5
            print(f"Task Probabilities:")
            for i, prob in enumerate(task_probs):
                print(f"  {i+1} star: {prob:.4f} ({prob*100:.2f}%)")
            
            print(f"\nConcept Predictions:")
            concept_names = ['Food', 'Ambiance', 'Service', 'Noise']
            sentiment_map = ['Negative', 'Neutral', 'Positive']
            
            for i, name in enumerate(concept_names):
                pred = concept_preds[i]
                sentiment = sentiment_map[pred]
                probs = concept_probs[i]
                print(f"  {name}: {sentiment}")
                print(f"    Probabilities: {probs[0]:.3f} (Neg) | {probs[1]:.3f} (Neu) | {probs[2]:.3f} (Pos)")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
