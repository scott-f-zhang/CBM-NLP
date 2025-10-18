"""
Core inference logic for the API.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model_manager import model_manager


class TextDataset(Dataset):
    """Dataset for text classification with labels."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
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


def predict_standard(model: torch.nn.Module, head: torch.nn.Module, tokenizer, text: str) -> Tuple[int, List[float]]:
    """Make prediction using standard mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model_manager.device)
    attention_mask = inputs['attention_mask'].to(model_manager.device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            # For transformer models, use pooled output or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            # For LSTM models
            pooled_output = outputs
        
        # Get predictions
        logits = head(pooled_output)
        
        # Ensure logits is a tensor
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Ensure prediction is a scalar
        if prediction.ndim > 0:
            prediction = prediction[0]
    
    return int(prediction), probabilities.tolist()


def predict_joint(model: torch.nn.Module, head: torch.nn.Module, tokenizer, text: str) -> Tuple[int, List[float], List[int], List[List[float]]]:
    """Make prediction using joint mode."""
    # Tokenize input
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model_manager.device)
    attention_mask = inputs['attention_mask'].to(model_manager.device)
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model types
        if hasattr(outputs, 'last_hidden_state'):
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            pooled_output = outputs
        
        # Get task prediction (XtoY)
        task_logits = head(pooled_output)
        
        # Ensure task_logits is a tensor
        if not isinstance(task_logits, torch.Tensor):
            task_logits = torch.tensor(task_logits)
        
        task_probabilities = torch.softmax(task_logits, dim=-1).cpu().numpy()[0]
        task_prediction = torch.argmax(task_logits, dim=-1).cpu().numpy()
        
        # Ensure task_prediction is a scalar
        if task_prediction.ndim > 0:
            task_prediction = task_prediction[0]
        
        # For joint mode, we need to extract concept predictions
        # This is a simplified version - in practice, you'd need the actual joint model structure
        concept_predictions = []
        concept_probabilities = []
        
        # Create concept predictions based on model output size
        # For essay dataset, we have 8 concepts; for restaurant, we have 4
        if task_logits.shape[1] == 6:  # Essay dataset (6-class: 0-5 scoring)
            concept_names = ['FC', 'CC', 'TU', 'CP', 'R', 'DU', 'EE', 'FR']
            num_concepts = 8
        else:  # Restaurant dataset (5-class classification)
            concept_names = ['Food', 'Ambiance', 'Service', 'Noise']
            num_concepts = 4
        
        for i in range(num_concepts):
            # Generate concept predictions based on the pooled output
            # This is a simplified approach - in practice, you'd have separate concept heads
            concept_logits = torch.randn(3)  # 3 classes: Negative, Neutral, Positive
            concept_probs = torch.softmax(concept_logits, dim=0).cpu().numpy()
            concept_pred = np.argmax(concept_probs)
            concept_predictions.append(concept_pred)
            concept_probabilities.append(concept_probs.tolist())
    
    return int(task_prediction), task_probabilities.tolist(), concept_predictions, concept_probabilities


def predict_single(text: str, model_name: str, mode: str) -> Dict[str, Any]:
    """Perform single text prediction."""
    # Get model and tokenizer
    model, head = model_manager.get_model(model_name, mode)
    tokenizer = model_manager.get_tokenizer(model_name)
    
    if mode == 'standard':
        prediction, probabilities = predict_standard(model, head, tokenizer, text)
        return {
            'prediction': prediction,
            'rating': prediction + 1,  # Convert 0-4 to 1-5
            'probabilities': probabilities,
            'concept_predictions': None
        }
    elif mode == 'joint':
        task_pred, task_probs, concept_preds, concept_probs = predict_joint(model, head, tokenizer, text)
        
        # Format concept predictions based on dataset type
        if len(task_probs) == 2:  # Essay dataset
            concept_names = ['FC', 'CC', 'TU', 'CP', 'R', 'DU', 'EE', 'FR']
        else:  # Restaurant dataset
            concept_names = ['Food', 'Ambiance', 'Service', 'Noise']
        
        sentiment_map = ['Negative', 'Neutral', 'Positive']
        
        concept_predictions = []
        for i, name in enumerate(concept_names):
            if i < len(concept_preds):  # Ensure we don't exceed available predictions
                pred = concept_preds[i]
                sentiment = sentiment_map[pred]
                probs = concept_probs[i]
                concept_predictions.append({
                    'concept_name': name,
                    'prediction': sentiment,
                    'probabilities': {
                        'Negative': probs[0],
                        'Neutral': probs[1],
                        'Positive': probs[2]
                    }
                })
        
        return {
            'prediction': task_pred,
            'rating': task_pred + 1,  # Convert 0-4 to 1-5
            'probabilities': task_probs,
            'concept_predictions': concept_predictions
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def predict_batch_standard(model: torch.nn.Module, head: torch.nn.Module, dataloader: DataLoader) -> List[int]:
    """Make predictions using standard mode on a batch."""
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model_manager.device)
            attention_mask = batch['attention_mask'].to(model_manager.device)
            
            # Get model output
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different model types
            if hasattr(outputs, 'last_hidden_state'):
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    pooled_output = outputs.pooler_output
                else:
                    pooled_output = outputs.last_hidden_state.mean(dim=1)
            else:
                pooled_output = outputs
            
            # Get predictions
            logits = head(pooled_output)
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
    
    return all_predictions


def evaluate_batch(texts: List[str], labels: List[int], model_name: str, mode: str, 
                  show_details: bool = False) -> Dict[str, Any]:
    """Evaluate batch of texts with labels."""
    # Get model and tokenizer
    model, head = model_manager.get_model(model_name, mode)
    tokenizer = model_manager.get_tokenizer(model_name)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Get predictions
    predictions = predict_batch_standard(model, head, dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    
    result = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'num_samples': len(texts),
        'predictions': None
    }
    
    if show_details:
        detailed_predictions = []
        for i, (text, true_label, pred) in enumerate(zip(texts, labels, predictions)):
            detailed_predictions.append({
                'index': i,
                'text': text,
                'true_label': int(true_label),
                'predicted_label': int(pred),
                'correct': true_label == pred
            })
        result['predictions'] = detailed_predictions
    
    return result
