"""
1D-CNN Model for Plant Health (with Timestamp-enabled features)
"""

import torch
import torch.nn as nn


class PlantHealthCNN(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.35)
        self.relu = nn.ReLU()

        self.flatten_size = 64 * 5  # After pooling

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x, plant_ids=None):
        x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
