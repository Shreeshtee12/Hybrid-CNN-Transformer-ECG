"""
models/torch_models/cnn_transformer.py

This is the REAL hybrid architecture used to produce the thesis's reported
results (97% accuracy, matching the Methodology chapter's description:
four residual CNN blocks [64,128,256,512], six Transformer encoder layers,
8-head attention, sinusoidal positional encoding).

Ported directly from notebooks/archive/individual_notebooks/CNN_Transformer_Hybrid.ipynb
(the OptimizedHybrid class), NOT from an earlier, smaller, unrelated model that
previously lived in this file under the same name.

IMPORTANT: this model was trained/evaluated at 100Hz (1000 samples per 10s
record, using filename_lr), not the 500Hz (5000-sample, filename_hr) signals
used elsewhere in src/. Pass --seq-len 1000 and --sr 100 when training this
model so the data pipeline matches what this architecture expects.
"""

import torch
import torch.nn as nn
import numpy as np


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(kernel_size=2)

        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            ) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return self.pool(out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe       = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(*([self.norm1(x)] * 3))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class OptimizedHybrid(nn.Module):
    """
    Hybrid CNN-Transformer for Multi-Label ECG Classification.
    Input:  (B, 12, seq_len)
    Output: (B, num_classes) — raw logits
    """
    def __init__(self, input_channels=12, seq_len=1000, num_classes=8,
                 cnn_channels=[64, 128, 256, 512],
                 transformer_layers=6, num_heads=8, dropout=0.2):
        super().__init__()

        # CNN Frontend -- four residual blocks, each halves the sequence length
        self.cnn1 = ResidualCNNBlock(input_channels,  cnn_channels[0])
        self.cnn2 = ResidualCNNBlock(cnn_channels[0], cnn_channels[1])
        self.cnn3 = ResidualCNNBlock(cnn_channels[1], cnn_channels[2])
        self.cnn4 = ResidualCNNBlock(cnn_channels[2], cnn_channels[3])
        # e.g. 1000 -> 500 -> 250 -> 125 -> 62 time steps at seq_len=1000

        embed_dim = cnn_channels[3]
        # NOTE: dynamically computed from seq_len (four halvings), rather than
        # the notebook's hardcoded max_len=62, so this generalizes correctly
        # if seq_len is ever something other than 1000.
        pooled_len = seq_len
        for _ in range(4):
            pooled_len = pooled_len // 2
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=pooled_len)

        # Transformer Backend
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(transformer_layers)
        ])

        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Expects (B, C, T). If fed (B, T, C), permute before calling.
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = x.transpose(1, 2)              # (B, T', embed_dim)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)                  # Global avg pool
        return self.head(x)


def build_model(input_shape, num_classes, cnn_channels=[64, 128, 256, 512],
                 transformer_layers=6, num_heads=8, dropout=0.2, **kwargs):
    """
    Constructs the real Hybrid CNN-Transformer (matches thesis Methodology chapter).
    Args:
        input_shape (tuple): (channels, sequence_length)
        num_classes (int): number of output classes
    """
    channels, seq_len = input_shape
    return OptimizedHybrid(
        input_channels=channels,
        seq_len=seq_len,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        transformer_layers=transformer_layers,
        num_heads=num_heads,
        dropout=dropout,
    )