"""
PlantHealthDataset - Clean Sensor-Only Mode
"""

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class PlantHealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int = 20):
        self.sequence_length = sequence_length

        # Core sensor features only
        self.feature_cols = [
            "soil_moisture",
            "soil_temp",
            "air_temp",
            "humidity",
            "light_lux",
            "soil_ph",
            "ec",
            "vpd",
        ]

        # Keep only features that actually exist
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]

        print(f"Using features: {self.feature_cols}")

        self.scaler = MinMaxScaler()
        self.features = self.scaler.fit_transform(df[self.feature_cols])

        # Labels
        label_map = {"Healthy": 0, "Moderate Stress": 1, "High Stress": 2}
        self.labels = df["health_status"].map(label_map).values

        # Create sequences
        self.sequences = []
        self.seq_labels = []

        for i in range(len(self.features) - sequence_length + 1):
            self.sequences.append(self.features[i : i + sequence_length])
            self.seq_labels.append(self.labels[i + sequence_length - 1])

        print(f"Created {len(self.sequences)} sequences of length {sequence_length}\n")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.seq_labels[idx], dtype=torch.long)

        # Return dummy plant_id (ignored in sensor-only mode)
        plant_id = torch.tensor(0, dtype=torch.long)
        return x, plant_id, y
