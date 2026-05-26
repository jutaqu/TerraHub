import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FocalLoss(nn.Module):
    """Stronger Focal Loss - Heavy penalty for missing Healthy"""

    def __init__(self, alpha=[8.0, 1.0, 1.0], gamma=4.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class PlantHealthCNN(nn.Module):
    def __init__(self, input_size=7, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_focal_species():
    print("\n=== Training 1D-CNN on Monstera Deliciosa ===")

    from core.data_loader import PlantHealthDataset
    from core.data_preprocessing import preprocess_data
    from core.feature_engineering import engineer_features

    df = preprocess_data(target_plant="Monstera Deliciosa")
    df = engineer_features(df)

    # Chronological split
    split_idx = int(0.80 * len(df))
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_holdout = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(
        f"✓ Chronological split → Train: {len(df_train)} rows | Holdout (unseen): {len(df_holdout)} rows"
    )

    # Stronger oversampling for Healthy
    healthy_mask = df_train["health_status"] == "Healthy"
    if healthy_mask.sum() > 0:
        df_healthy = df_train[healthy_mask].sample(n=600, replace=True, random_state=42)
        df_train = pd.concat([df_train, df_healthy], ignore_index=True)
        print(
            f"✓ Oversampled Healthy class in training set → {df_train['health_status'].value_counts().get('Healthy', 0)} samples"
        )

    feature_cols = [
        "soil_moisture",
        "soil_temp",
        "air_temp",
        "humidity",
        "light_lux",
        "soil_ph",
        "ec",
    ]

    train_dataset = PlantHealthDataset(
        df_train, feature_cols=feature_cols, sequence_length=20
    )
    val_dataset = PlantHealthDataset(
        df_holdout, feature_cols=feature_cols, sequence_length=20
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PlantHealthCNN(input_size=len(feature_cols)).to(device)
    criterion = FocalLoss(alpha=[8.0, 1.0, 1.0], gamma=4.0)  # Stronger focus
    optimizer = optim.AdamW(model.parameters(), lr=0.00025, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=25, factor=0.5
    )

    best_acc = 0.0
    patience = 70
    counter = 0

    for epoch in range(180):
        model.train()
        for x, _, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, _, y in val_loader:
                x = x.to(device)
                outputs = model(x)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        scheduler.step(val_acc)

        print(
            f"  Epoch {epoch + 1:3d} → Val Acc: {val_acc:.2f}% | Predictions: {Counter(all_preds)}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_focal_monstera_cnn.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("  → Early stopping triggered.")
                break

    print(f"\n✅ Training completed! Best Val Acc on Unseen Holdout: {best_acc:.2f}%")
    print("✅ Final 1D-CNN model saved.")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_focal_species()
