"""
Hybrid Models for TerraHub Plant Health Classification
====================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlantEmbedding(nn.Module):
    """Embedding layer for plant species."""

    def __init__(self, num_plants: int, embedding_dim: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(num_plants, embedding_dim)

    def forward(self, plant_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(plant_ids)


class PlantHealthSimple(nn.Module):
    """Simpler Feed-Forward Model - Recommended for small datasets."""

    def __init__(
        self,
        input_size: int = 20,
        num_plants: int = 8,
        embedding_dim: int = 12,
        num_classes: int = 3,
    ):
        super().__init__()

        self.plant_embedding = PlantEmbedding(num_plants, embedding_dim)

        self.fc1 = nn.Linear(input_size + embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor, plant_ids: torch.Tensor) -> torch.Tensor:
        # Average across time dimension
        x = x.mean(dim=1)

        # Add plant embedding
        emb = self.plant_embedding(plant_ids)
        x = torch.cat([x, emb], dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# Quick test
if __name__ == "__main__":
    model = PlantHealthSimple(input_size=20, num_plants=8, num_classes=3)
    print("PlantHealthSimple loaded successfully!")
    print(model)
