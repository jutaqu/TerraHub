"""
Training Script - Real Data Only + Per-Plant Evaluation
=======================================================
"""

import os
from collections import Counter
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import PlantHealthDataset
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from hybrid_model import PlantHealthSimple
from utils import load_plant_care_profiles, set_seed


def train_real_only():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = preprocess_data()
    profiles = load_plant_care_profiles()
    df = engineer_features(df, profiles)

    print(f"\n=== Training on REAL DATA ONLY - Total samples: {len(df)} ===\n")

    train_df, val_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["health_status"]
    )

    train_dataset = PlantHealthDataset(train_df, sequence_length=20)
    val_dataset = PlantHealthDataset(val_df, sequence_length=20)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class_counts = Counter(train_df["health_status"])
    weights = [
        1 / class_counts.get(c, 1)
        for c in ["Healthy", "Moderate Stress", "High Stress"]
    ]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = PlantHealthSimple(
        input_size=len(train_dataset.feature_cols),
        num_plants=len(profiles),
        embedding_dim=12,
        num_classes=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    writer = SummaryWriter(
        f"runs/real_only_perplant_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )

    best_val_acc = 0.0
    print("Starting training...\n")

    for epoch in range(80):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, plant_ids, y in train_loader:
            x, plant_ids, y = x.to(device), plant_ids.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x, plant_ids)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100.0 * correct / total

        # Validation with per-plant breakdown
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        plant_correct = Counter()
        plant_total = Counter()

        with torch.no_grad():
            for x, plant_ids, y in val_loader:
                x, plant_ids, y = x.to(device), plant_ids.to(device), y.to(device)
                outputs = model(x, plant_ids)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

                # Per-plant accuracy
                for p, true, pred in zip(
                    plant_ids.cpu().numpy(), y.cpu().numpy(), predicted.cpu().numpy()
                ):
                    plant_name = train_dataset.plant_encoder.inverse_transform([p])[0]
                    plant_total[plant_name] += 1
                    if true == pred:
                        plant_correct[plant_name] += 1

        val_acc = 100.0 * val_correct / val_total
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1:2d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        # Print per-plant accuracy
        print("  Per-Plant Validation Accuracy:")
        for plant in sorted(plant_total.keys()):
            acc = 100 * plant_correct[plant] / plant_total[plant]
            print(
                f"    {plant:25}: {acc:5.1f}% ({plant_correct[plant]}/{plant_total[plant]})"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_real_only_perplant.pth")

    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%")
    torch.save(model.state_dict(), "models/plant_health_model_real_only_final.pth")
    writer.close()


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_real_only()
