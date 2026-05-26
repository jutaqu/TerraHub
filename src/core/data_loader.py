"""
Data Loader
"""

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset


class PlantHealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int = 20):
        self.sequence_length = sequence_length

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
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]

        print(f"Using features: {self.feature_cols}")

        self.scaler = MinMaxScaler()
        self.plant_encoder = LabelEncoder()

        self.features = self.scaler.fit_transform(df[self.feature_cols])
        self.plant_ids = self.plant_encoder.fit_transform(df["plant_name"])

        label_map = {"Healthy": 0, "Moderate Stress": 1, "High Stress": 2}
        self.labels = df["health_status"].map(label_map).values

        self.sequences = []
        self.seq_plant_ids = []
        self.seq_labels = []

        for i in range(len(self.features) - sequence_length + 1):
            self.sequences.append(self.features[i : i + sequence_length])
            self.seq_plant_ids.append(self.plant_ids[i])
            self.seq_labels.append(self.labels[i + sequence_length - 1])

        print(f"Created {len(self.sequences)} sequences of length {sequence_length}\n")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        plant_id = torch.tensor(self.seq_plant_ids[idx], dtype=torch.long)
        y = torch.tensor(self.seq_labels[idx], dtype=torch.long)
        return x, plant_id, y
