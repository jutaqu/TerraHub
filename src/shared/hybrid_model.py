"""
Shared Model Architecture
"""

import torch
import torch.nn as nn


class PlantEmbedding(nn.Module):
    def __init__(self, num_plants: int, embedding_dim: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(num_plants, embedding_dim)

    def forward(self, plant_ids):
        return self.embedding(plant_ids)


class PlantHealthSimple(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_plants: int,
        embedding_dim: int = 12,
        num_classes: int = 3,
    ):
        super().__init__()
        self.plant_embedding = PlantEmbedding(num_plants, embedding_dim)

        self.fc1 = nn.Linear(input_size + embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, x, plant_ids):
        # x shape: (batch, seq_len, features)
        x = x.mean(dim=1)  # Average over sequence
        emb = self.plant_embedding(plant_ids)
        x = torch.cat([x, emb], dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
