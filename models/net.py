from config import WINDOW_SIZE
import torch
import torch.nn as nn
import math

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seperable = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.seperable(x)
        return x

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seperable = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.seperable(x)
        return x    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class TrasnformerEncoder(nn.Module) :
    def __init__(self, d_model, nhead, num_layers, dropout) :
        super(TrasnformerEncoder, self).__init__()
        self.pe = PositionalEncoding(d_model, max_len = 100)
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                   nhead = nhead,
                                                   dropout = dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x) :
        x = x.permute(0, 2, 1)
        x = self.pe(x)
        out = self.transformer_encoder(x)
        out_pooled = out.mean(dim=1)
        return out_pooled