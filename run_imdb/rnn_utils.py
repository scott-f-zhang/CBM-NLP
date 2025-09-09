import torch
import torch.nn.functional as F


class BiLSTMWithDotAttention(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=None, pretrained_embeddings=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = torch.nn.Parameter(torch.tensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = False
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_ids, attention_mask):
        input_lengths = attention_mask.sum(dim=1)
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        weights = F.softmax(torch.bmm(output, output.transpose(1, 2)), dim=2)
        attention = torch.bmm(weights, output)
        return attention


