import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(14)  # Channels should match input shape
        self.conv1 = nn.Conv1d(14, 128, kernel_size=10, stride=1, padding="same")
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Switch to (batch, channels, sequence)
        x = self.batch_norm1(x)
        x = F.relu(self.conv1(x))
        x = self.batch_norm2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Reorder for LSTM (batch, sequence, input_size)
        x, _ = self.lstm(x)
        x = self.batch_norm3(x[:, -1, :])  # Use the last LSTM output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)
