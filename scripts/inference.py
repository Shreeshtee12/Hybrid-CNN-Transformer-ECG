
import argparse
import torch
import torch.nn as nn
import numpy as np
import wfdb

class ECG_Transformer(nn.Module):
    def __init__(self, seq_len=5000, num_features=12, d_model=32, nhead=2, num_layers=2, num_classes=10):
        super(ECG_Transformer, self).__init__()
        self.input_linear = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# Argument parser for ECG file path
parser = argparse.ArgumentParser(description="Run ECG inference")
parser.add_argument('--record', type=str, help="Path to the WFDB record (without extension)", required=True)
args = parser.parse_args()

# Load model
checkpoint = torch.load('ecg_transformer.pt', map_location='cpu')
model = ECG_Transformer(num_classes=len(checkpoint['classes']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load ECG signal
signal, _ = wfdb.rdsamp(args.record)
signal = torch.tensor(signal[:5000], dtype=torch.float32).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(signal)
    probs = torch.sigmoid(output).squeeze().tolist()

# Display results
for label, prob in zip(checkpoint['classes'], probs):
    if prob > 0.5:
        print(f"{label}: {prob:.2f}")
