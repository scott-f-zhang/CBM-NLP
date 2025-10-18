"""
Standalone model loader to avoid circular imports.
This is a copy of the load_model_and_tokenizer function from cbm.models.loaders
"""

import os
import torch
from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
from typing import Tuple, Optional

try:
    from gensim.models import FastText
except Exception:
    FastText = None  # Optional dependency


class BiLSTMWithDotAttention(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, embeddings_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if embeddings_weight is not None:
            self.embedding.weight = torch.nn.Parameter(embeddings_weight)
            self.embedding.weight.requires_grad = False
        else:
            # If no pretrained embeddings available, allow the embedding to learn
            self.embedding.weight.requires_grad = True
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        # Match run_cebab: project pooled bi-LSTM (2*hidden_dim) to hidden_dim
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )

    def forward(self, input_ids, attention_mask):
        output, _ = self.lstm(self.embedding(input_ids))
        weights = torch.softmax(torch.bmm(output, output.transpose(1, 2)), dim=2)
        attention = torch.bmm(weights, output)
        # Return a hidden_dim-sized representation like original run_cebab
        logits = self.classifier(attention.mean(1))
        return logits


def load_model_and_tokenizer(model_name: str, fasttext_path: Optional[str] = None) -> Tuple[object, object, Optional[int]]:
    """Load model and tokenizer for inference."""
    if model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name)
        hidden_size = model.config.hidden_size
        return model, tokenizer, hidden_size
    if model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        hidden_size = model.config.hidden_size
        return model, tokenizer, hidden_size
    if model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained(model_name)
        hidden_size = model.config.hidden_size
        return model, tokenizer, hidden_size
    if model_name == 'lstm':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # For LSTM, we don't load FastText embeddings in inference mode
        model = BiLSTMWithDotAttention(len(tokenizer.vocab), 300, 128, None)
        # Expose hidden size as 128 to match the projected representation
        hidden_size = model.hidden_dim
        return model, tokenizer, hidden_size
    raise ValueError(f"Unsupported model_name: {model_name}")
