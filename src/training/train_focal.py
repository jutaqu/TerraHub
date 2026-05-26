"""
Focal Species Training - Deep Training on 1-2 Main Plants
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from core.config import TrainingConfig
from core.data_loader import PlantHealthDataset
from core.data_preprocessing import preprocess_data
from core.feature_engineering import engineer_features
from core.model import PlantHealthModel
from core.utils import ensure_dir, load_plant_care_profiles, set_seed


def train_focal_species():
    ensure_dir("models")
    ensure_dir("runs")

    config = TrainingConfig(
        focal_plants=["Monstera Deliciosa"],  # Change this to your focal plant(s)
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=1e-5,
        embedding_dim=12,
        patience=15,
    )

    set_seed(config.random_seed)
    config.print_config()

    df = preprocess_data(target_plant=config.focal_plants[0])
    profiles = load_plant_care_profiles()
    df = engineer_features(df, profiles)

    print(f"\n=== Training on Focal Plant: {config.focal_plants[0]} ===\n")

    train_df, val_df = train_test_split(
        df,
        test_size=0.25,
        random_state=config.random_seed,
        stratify=df["health_status"],
    )

    train_dataset = PlantHealthDataset(train_df, sequence_length=config.sequence_length)
    val_dataset = PlantHealthDataset(val_df, sequence_length=config.sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_counts = Counter(train_df["health_status"])
    weights = [
        1.0 / class_counts.get(c, 1)
        for c in ["Healthy", "Moderate Stress", "High Stress"]
    ]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = PlantHealthModel(
        input_size=len(train_dataset.feature_cols),
        num_plants=len(profiles),
        embedding_dim=config.embedding_dim,
        num_classes=3,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val_acc = 0.0
    print("Starting focal species training...\n")

    for epoch in range(config.num_epochs):
        model.train()
        correct = 0
        total = 0
        for x, plant_ids, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x, plant_ids)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100.0 * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, plant_ids, y in val_loader:
                outputs = model(x, plant_ids)
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch {epoch + 1:2d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                f"models/best_focal_{config.focal_plants[0].replace(' ', '_')}.pth",
            )

    print(f"\n✅ Focal training completed! Best Val Acc: {best_val_acc:.2f}%")
    torch.save(model.state_dict(), "models/final_focal_model.pth")


if __name__ == "__main__":
    train_focal_species()
