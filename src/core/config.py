"""
Central Configuration for Simplified TerraHub Project
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    sequence_length: int = 20
    batch_size: int = 32
    num_epochs: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    embedding_dim: int = 12
    patience: int = 12
    random_seed: int = 42

    # Focal species setup
    focal_plants: List[str] = None  # e.g. ["Monstera Deliciosa"]
    transfer_plants: List[str] = None  # Plants to transfer to

    def print_config(self):
        print("\n=== Training Configuration ===")
        print(f"Sequence Length : {self.sequence_length}")
        print(f"Batch Size      : {self.batch_size}")
        print(f"Epochs          : {self.num_epochs}")
        print(f"Learning Rate   : {self.learning_rate}")
        print(f"Focal Plants    : {self.focal_plants}")
        print(f"Transfer Plants : {self.transfer_plants}")
        print("=============================\n")
