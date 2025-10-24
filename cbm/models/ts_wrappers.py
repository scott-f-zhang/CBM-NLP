import torch
import torch.nn as nn


class StandardTSWrapper(nn.Module):
    """
    TorchScript-friendly wrapper that encapsulates encoder + head for standard pipeline.
    - For HF encoders: expects outputs with last_hidden_state; uses mean pooling.
    - For LSTM encoder in this repo: encoder forward returns a fixed-size representation.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module, is_lstm: bool):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.is_lstm = is_lstm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.is_lstm:
            pooled = outputs  # LSTM wrapper returns a tensor representation directly
        else:
            # HuggingFace models return an object with last_hidden_state tensor
            pooled = outputs.last_hidden_state.mean(1)
        logits = self.head(pooled)
        return logits


class JointTSWrapper(nn.Module):
    """
    TorchScript-friendly wrapper for joint pipeline.
    Returns a tuple: (task_logits, concept_logits_concat)
    where concept logits are concatenated across attributes to a 2D tensor.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module, is_lstm: bool):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.is_lstm = is_lstm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.is_lstm:
            pooled = outputs
        else:
            pooled = outputs.last_hidden_state.mean(1)
        out = self.head(pooled)
        # out[0:1] -> X->Y logits list (first element); out[1:] -> concept logits list
        task_logits = out[0]
        concept_list = out[1:]
        concept_logits = torch.cat(concept_list, dim=0)
        return task_logits, concept_logits


