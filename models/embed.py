import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataset.vocab import Vocabulary


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LabelEmbedding(nn.Module):
    def __init__(self, d_model):
        super(LabelEmbedding, self).__init__()
        self.vocab_size = Vocabulary().vocab_size
        self.label_embedding = nn.Embedding(self.vocab_size, d_model)

    def forward(self, x):
        x = self.label_embedding(x[:, :, 1].long())
        return x


class TokenEmbedding(nn.Module):    # for CarCode
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(7, d_model)

    def forward(self, x):
        x = self.token_embedding(x[:, :, 0].long())
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, d_model):
        super(ValueEmbedding, self).__init__()
        self.value_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        x = self.value_conv(x[:, :, -1].unsqueeze(-1).permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.label_embedding = LabelEmbedding(d_model)
        self.value_embedding = ValueEmbedding(d_model)
        self.token_embedding = TokenEmbedding(d_model)
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x_tmp = self.value_embedding(x) \
                + self.label_embedding(x) \
                + self.token_embedding(x) \
                + self.position_embedding(x)

        return self.dropout(x_tmp)
