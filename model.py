import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        scores = self.attn(encoder_outputs)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * encoder_outputs, dim=1)
        return context, weights

class ProbLSTMAttention(nn.Module):
    def __init__(self, n_features, hidden_dim, output_len, n_quantiles=3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_features * output_len * n_quantiles)
        self.n_features = n_features
        self.output_len = output_len
        self.n_quantiles = n_quantiles

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)
        out = out.view(-1, self.output_len, self.n_features, self.n_quantiles)
        return out, attn_weights
