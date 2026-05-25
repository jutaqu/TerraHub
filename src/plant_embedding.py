# models/plant_embedding.py
import torch
import torch.nn as nn


class PlantEmbedding(nn.Module):
    """
    Modular embedding layer for plant types.
    Designed for easy expansion when adding new plants.
    """

    def __init__(self, num_plants: int, embedding_dim: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_plants, embedding_dim=embedding_dim
        )

    def forward(self, plant_ids: torch.Tensor):
        return self.embedding(plant_ids)
