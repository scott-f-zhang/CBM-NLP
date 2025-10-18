#!/usr/bin/env python3
"""
Complete inference example with metrics calculation.

This script demonstrates how to load trained models, make predictions on texts with true labels,
and calculate accuracy and F1 scores. It supports batch processing and CSV file input.

Usage:
    python inference_with_metrics.py --model_name bert-base-uncased --mode standard
    python inference_with_metrics.py --model_name gpt2 --mode joint --csv_file test_data.csv
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add main module to path for loading pickled models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import standalone model loader to avoid circular imports
from model_loader import load_model_and_tokenizer


class TextDataset(Dataset):
    """Dataset for text classification with labels."""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


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


def predict_batch_standard(model, head, dataloader, device):
    """Make predictions using standard mode on a batch."""
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
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
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def predict_batch_joint(model, head, dataloader, device):
    """Make predictions using joint mode on a batch."""
    all_task_predictions = []
    all_labels = []
    all_task_probabilities = []
    all_concept_predictions = []
    all_concept_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
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
            task_predictions = torch.argmax(XtoY_output[0], dim=1)
            task_probs = torch.softmax(XtoY_output[0], dim=1)
            
            # Concept predictions
            concept_logits = torch.cat(XtoC_output, dim=0)  # [4, batch_size, 3]
            concept_predictions = torch.argmax(concept_logits, dim=2).transpose(0, 1)  # [batch_size, 4]
            concept_probs = torch.softmax(concept_logits, dim=2).transpose(0, 1)  # [batch_size, 4, 3]
            
            all_task_predictions.extend(task_predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_task_probabilities.extend(task_probs.cpu().numpy())
            all_concept_predictions.extend(concept_predictions.cpu().numpy())
            all_concept_probabilities.extend(concept_probs.cpu().numpy())
    
    return (np.array(all_task_predictions), np.array(all_labels), 
            np.array(all_task_probabilities), np.array(all_concept_predictions),
            np.array(all_concept_probabilities))


def create_sample_data():
    """Create sample data for demonstration."""
    sample_data = [
        {"text": "This restaurant has amazing food and great service!", "label": 4},
        {"text": "Food was terrible and service was awful.", "label": 0},
        {"text": "Average experience, nothing special.", "label": 2},
        {"text": "Great food but bad service.", "label": 2},
        {"text": "Perfect in every way! Highly recommended.", "label": 4},
        {"text": "Overpriced and disappointing.", "label": 1},
        {"text": "Good atmosphere but food was mediocre.", "label": 2},
        {"text": "Excellent quality and friendly staff.", "label": 4},
    ]
    return pd.DataFrame(sample_data)


def main():
    parser = argparse.ArgumentParser(description='Inference with metrics calculation')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       choices=['bert-base-uncased', 'gpt2', 'lstm', 'roberta-base'],
                       help='Model name to use')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'joint'],
                       help='Model mode (standard or joint)')
    parser.add_argument('--csv_file', type=str, default=None,
                       help='CSV file with text and label columns (uses sample data if not provided)')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of text column in CSV')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of label column in CSV')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--show_details', action='store_true',
                       help='Show detailed per-sample results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print("-" * 60)
    
    try:
        # Load data
        if args.csv_file and os.path.exists(args.csv_file):
            print(f"Loading data from: {args.csv_file}")
            df = pd.read_csv(args.csv_file)
            texts = df[args.text_column].tolist()
            labels = df[args.label_column].tolist()
        else:
            print("Using sample data (use --csv_file to load your own data)")
            df = create_sample_data()
            texts = df['text'].tolist()
            labels = df['label'].tolist()
        
        print(f"Loaded {len(texts)} samples")
        
        # Load tokenizer and model
        _, tokenizer, _ = load_model_and_tokenizer(args.model_name)
        model, head = load_trained_model(args.model_name, args.mode, device)
        
        # Create dataset and dataloader
        dataset = TextDataset(texts, labels, tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        # Make predictions
        if args.mode == 'standard':
            predictions, true_labels, probabilities = predict_batch_standard(
                model, head, dataloader, device)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            macro_f1 = f1_score(true_labels, predictions, average='macro')
            weighted_f1 = f1_score(true_labels, predictions, average='weighted')
            
            print(f"\nResults:")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}")
            
            if args.show_details:
                print(f"\nDetailed Results:")
                for i, (text, true_label, pred, probs) in enumerate(zip(texts, true_labels, predictions, probabilities)):
                    correct = "✓" if pred == true_label else "✗"
                    print(f"{i+1:2d}. {text[:50]}...")
                    print(f"    True: {true_label+1} star, Predicted: {pred+1} star {correct}")
                    print(f"    Probabilities: {probs}")
                    print()
        
        elif args.mode == 'joint':
            (task_predictions, true_labels, task_probabilities, 
             concept_predictions, concept_probabilities) = predict_batch_joint(
                model, head, dataloader, device)
            
            # Calculate task metrics
            task_accuracy = accuracy_score(true_labels, task_predictions)
            task_macro_f1 = f1_score(true_labels, task_predictions, average='macro')
            task_weighted_f1 = f1_score(true_labels, task_predictions, average='weighted')
            
            print(f"\nTask Results:")
            print(f"Accuracy: {task_accuracy:.4f} ({task_accuracy*100:.2f}%)")
            print(f"Macro F1: {task_macro_f1:.4f}")
            print(f"Weighted F1: {task_weighted_f1:.4f}")
            
            if args.show_details:
                print(f"\nDetailed Results:")
                concept_names = ['Food', 'Ambiance', 'Service', 'Noise']
                sentiment_map = ['Negative', 'Neutral', 'Positive']
                
                for i, (text, true_label, task_pred, task_probs, concept_preds, concept_probs) in enumerate(
                    zip(texts, true_labels, task_predictions, task_probabilities, 
                        concept_predictions, concept_probabilities)):
                    
                    correct = "✓" if task_pred == true_label else "✗"
                    print(f"{i+1:2d}. {text[:50]}...")
                    print(f"    True: {true_label+1} star, Predicted: {task_pred+1} star {correct}")
                    print(f"    Task Probabilities: {task_probs}")
                    print(f"    Concept Predictions:")
                    
                    for j, name in enumerate(concept_names):
                        pred = concept_preds[j]
                        sentiment = sentiment_map[pred]
                        probs = concept_probs[j]
                        print(f"      {name}: {sentiment} ({probs[0]:.2f}|{probs[1]:.2f}|{probs[2]:.2f})")
                    print()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
