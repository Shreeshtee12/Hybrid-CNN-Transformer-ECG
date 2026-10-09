# models/torch_models/cnn1d.py
#
# Plain 1D CNN baseline: stacked Conv-BN-ReLU-MaxPool blocks, global average
# pooling, linear head. Deliberately NO residual connections, NO recurrence and
# NO attention, so it isolates what those components add (ResNet1D / xResNet1D
# add skips; xLSTM adds a BiLSTM; the hybrid adds a Transformer).

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class CNN1d(nn.Module):
    def __init__(self, in_channels=12, num_classes=8, widths=(64, 128, 256, 256, 512), dropout=0.3):
        super().__init__()
        blocks, prev = [], in_channels
        for w in widths:
            blocks.append(ConvBlock(prev, w))
            prev = w
        self.blocks = nn.Sequential(*blocks)      # Grad-CAM hooks blocks[-1]
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x):                          # x: (B, C, T) -> logits (B, num_classes)
        x = self.blocks(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(self.drop(x))


def build_model(input_shape, num_classes, **kwargs):
    """input_shape = (channels, sequence_length)"""
    return CNN1d(in_channels=input_shape[0], num_classes=num_classes, **kwargs)