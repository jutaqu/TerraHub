# config.py
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Embedding
    num_plants: int = 8  # Start with 8, easy to increase
    embedding_dim: int = 24  # Scalable dimension

    # LSTM
    sequence_length: int = 30  # Timesteps per input
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    dropout: float = 0.3

    # Output
    num_health_classes: int = 3  # Healthy, Moderate, High Stress

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
