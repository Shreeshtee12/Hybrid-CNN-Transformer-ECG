import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import wfdb
import ast
import wandb

from models.model_selector import get_model

# Args & Config

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='transformer', help='Model name from model_selector.py')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--signal_path', type=str, default='data/raw/ptbxl/records500')
parser.add_argument('--sr', type=int, default=500)
args = parser.parse_args()

wandb.init(project="ptbxl-ecg-models", config=vars(args))


# Dataset

class PTBXL_Dataset(Dataset):
    def __init__(self, records, labels, signal_path, sr=500):
        self.records = records
        self.labels = labels
        self.signal_path = signal_path
        self.sr = sr

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = self.labels[idx]
        signal, _ = wfdb.rdsamp(os.path.join(self.signal_path, rec))
        signal = signal[:5000] if self.sr == 500 else signal[:1000]
        signal = torch.tensor(signal, dtype=torch.float32)  # (T, C)
        signal = signal.transpose(0, 1)  # (C, T)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label


# Load Labels & Records

df = pd.read_csv('data/raw/ptbxl/ptbxl_database.csv')
scp_statements = pd.read_csv('data/raw/ptbxl/scp_statements.csv')
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
df['diagnostic_superclass'] = df['scp_codes'].apply(lambda x: list(x.keys()))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['diagnostic_superclass'])
records = df['filename_hr'].str.replace('.hea', '', regex=False).values
train_rec, test_rec, y_train, y_test = train_test_split(records, y, test_size=0.2, random_state=42)

train_ds = PTBXL_Dataset(train_rec, y_train, args.signal_path)
test_ds = PTBXL_Dataset(test_rec, y_test, args.signal_path)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size)


# Model, Loss, Optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = (12, 5000)  # (channels, sequence_length)
num_classes = len(mlb.classes_)
model = get_model(args.model, input_shape=input_shape, num_classes=num_classes).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
wandb.watch(model)


# Training Loop

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        signals, labels = signals.to(device), labels.to(device)

        if args.model == 'transformer':
            signals = signals.permute(0, 2, 1)  # Transformer expects (B, T, C)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"train_loss": loss.item()})

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")


# Evaluation

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for signals, labels in test_loader:
        signals, labels = signals.to(device), labels.to(device)

        if args.model == 'transformer':
            signals = signals.permute(0, 2, 1)

        outputs = model(signals)
        y_true.append(labels.cpu().numpy())
        y_pred.append(torch.sigmoid(outputs).cpu().numpy())

macro_auc = roc_auc_score(np.vstack(y_true), np.vstack(y_pred), average='macro')
print("Classification Report:")
print(classification_report(np.vstack(y_true) > 0.5, np.vstack(y_pred) > 0.5, target_names=mlb.classes_))
wandb.log({"macro_auc": macro_auc})


# Save Model

model_path = f"{args.model}.pt"
torch.save({'model_state_dict': model.state_dict(), 'classes': mlb.classes_}, model_path)
wandb.save(model_path)

