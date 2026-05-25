"""
Data Loader for TerraHub Plant Health Model
===========================================
Uses simplified core sensor features.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class PlantHealthDataset(Dataset):
    """Dataset using core TerraHub sensor features."""

    def __init__(self, df: pd.DataFrame, sequence_length: int = 20):
        self.seq_len = sequence_length

        # Core features only
        self.feature_cols = [
            "soil_moisture",
            "air_temp",
            "soil_temp",
            "humidity",
            "light_lux",
            "soil_ph",
            "ec",
            "vpd",
            "soil_moisture_deviation",
            "light_lux_deviation",
        ]

        print(f"Using {len(self.feature_cols)} core features")

        df = df.copy()

        # Encode plant names
        self.plant_encoder = LabelEncoder()
        df["plant_id_encoded"] = self.plant_encoder.fit_transform(df["plant_name"])

        # Scale features
        self.scaler = MinMaxScaler()
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])

        self.X = df[self.feature_cols].values
        self.plant_ids = df["plant_id_encoded"].values
        self.y = (
            df["health_status"]
            .map({"Healthy": 0, "Moderate Stress": 1, "High Stress": 2})
            .values
        )

        # Create sequences
        self.sequences = []
        self.seq_plant_ids = []
        self.seq_labels = []

        for plant_id in np.unique(self.plant_ids):
            mask = self.plant_ids == plant_id
            plant_data = self.X[mask]
            plant_labels = self.y[mask]

            for i in range(len(plant_data) - self.seq_len):
                self.sequences.append(plant_data[i : i + self.seq_len])
                self.seq_plant_ids.append(plant_id)
                self.seq_labels.append(plant_labels[i + self.seq_len])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.seq_plant_ids = np.array(self.seq_plant_ids, dtype=np.int64)
        self.seq_labels = np.array(self.seq_labels, dtype=np.int64)

        print(f"Created {len(self)} sequences of length {self.seq_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.seq_plant_ids[idx]),
            torch.tensor(self.seq_labels[idx]),
        )


# Quick test
if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from feature_engineering import engineer_features
    from utils import load_plant_care_profiles

    df = preprocess_data()
    profiles = load_plant_care_profiles()
    df = engineer_features(df, profiles)

    dataset = PlantHealthDataset(df, sequence_length=20)
    print("DataLoader ready!")
