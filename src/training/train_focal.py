"""
Focal 1D-CNN Training - Fixed Evaluation
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from core.config import TrainingConfig
from core.data_loader import PlantHealthDataset
from core.data_preprocessing import preprocess_data
from core.feature_engineering import engineer_features
from core.model import PlantHealthCNN
from core.utils import ensure_dir, set_seed


def evaluate_model(model, loader, description=""):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, _, y in loader:
            outputs = model(x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_acc = (all_preds == all_labels).mean() * 100

    print(
        f"\n  {description} → Val Acc: {val_acc:.2f}% | Predictions: {Counter(all_preds)}"
    )

    # Safe classification report
    from sklearn.metrics import classification_report

    unique_labels = np.unique(all_labels)
    target_names = ["Healthy", "Moderate Stress", "High Stress"]

    print(
        classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=[target_names[i] for i in unique_labels],
            zero_division=0,
        )
    )
    return val_acc


def train_focal_species():
    ensure_dir("models")
    ensure_dir("runs")

    config = TrainingConfig(
        focal_plants=["Monstera Deliciosa"],
        num_epochs=180,
        learning_rate=0.0003,
        weight_decay=8e-4,
        patience=40,
    )

    set_seed(config.random_seed)
    config.print_config()

    df = preprocess_data(target_plant=config.focal_plants[0])
    df = engineer_features(df)

    if "plant_name" in df.columns or "Plant_ID" in df.columns:
        df = df.drop(
            columns=[col for col in ["plant_name", "Plant_ID"] if col in df.columns]
        )

    print(f"=== Training 1D-CNN on {config.focal_plants[0]} ===\n")

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

    model = PlantHealthCNN(input_size=len(train_dataset.feature_cols), num_classes=3)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([2.0, 1.0, 5.0], dtype=torch.float32)
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.4, patience=12
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    print("Starting 1D-CNN training...\n")

    for epoch in range(config.num_epochs):
        model.train()
        correct = total = 0
        for x, _, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100.0 * correct / total
        val_acc = evaluate_model(model, val_loader, f"Epoch {epoch + 1:3d}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_focal_monstera_cnn.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"  → Early stopping triggered.")
                break

    print(f"\n✅ Training completed! Best Val Acc: {best_val_acc:.2f}%")
    torch.save(model.state_dict(), "models/final_focal_cnn.pth")
    print("✅ Final 1D-CNN model saved.")


if __name__ == "__main__":
    train_focal_species()
