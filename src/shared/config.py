"""
Central Configuration with Debug
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    sequence_length: int = 20
    batch_size: int = 32
    num_epochs: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    embedding_dim: int = 12
    patience: int = 8
    random_seed: int = 42

    plant_filter: Optional[str] = None  # None = all plants (general mode)
    use_synthetic: bool = False
    model_save_name: str = "best_model.pth"

    def print_config(self):
        print("\n=== Training Configuration ===")
        print(f"Sequence Length : {self.sequence_length}")
        print(f"Batch Size      : {self.batch_size}")
        print(f"Epochs          : {self.num_epochs}")
        print(f"Learning Rate   : {self.learning_rate}")
        print(
            f"Plant Filter    : {self.plant_filter if self.plant_filter else 'ALL PLANTS (General)'}"
        )
        print(f"Model Save Name : {self.model_save_name}")
        print("=============================\n")
