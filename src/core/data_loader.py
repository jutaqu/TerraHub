import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PlantHealthDataset(Dataset):
    def __init__(self, df, feature_cols=None, sequence_length=20):
        self.sequence_length = sequence_length

        if feature_cols is None:
            exclude = ["health_status", "plant_name", "Timestamp", "Plant_ID"]
            self.feature_cols = [
                col
                for col in df.columns
                if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
            ]
        else:
            self.feature_cols = [col for col in feature_cols if col in df.columns]

        print(f"Using features: {self.feature_cols}")

        self.X = df[self.feature_cols].values.astype(np.float32)
        self.y = pd.Categorical(df["health_status"]).codes

        # Create sequences
        self.sequences = []
        self.labels = []

        for i in range(len(self.X) - sequence_length + 1):
            seq = self.X[i : i + sequence_length]
            label = self.y[i + sequence_length - 1]
            self.sequences.append(seq)
            self.labels.append(label)

        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            0,  # dummy plant_id (no embedding)
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
