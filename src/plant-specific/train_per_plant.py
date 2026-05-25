"""
Plant-Specific Training Pipeline with Early Stopping
====================================================
Trains one specialized model per plant with early stopping.
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

from shared.config import TrainingConfig
from shared.data_loader import PlantHealthDataset
from shared.data_preprocessing import preprocess_data
from shared.feature_engineering import engineer_features
from shared.hybrid_model import PlantHealthSimple
from shared.utils import ensure_dir, load_plant_care_profiles, set_seed


def train_for_single_plant(target_plant: str, patience: int = 12):
    print(f"\n{'=' * 75}")
    print(f"TRAINING SPECIALIZED MODEL FOR: {target_plant}")
    print(f"{'=' * 75}\n")

    set_seed(42)

    # Load and prepare plant-specific data
    df_base = preprocess_data(target_plant=None)
    profiles = load_plant_care_profiles()

    df_plant = df_base.copy()
    df_plant["plant_name"] = target_plant
    df_plant = preprocess_data(
        target_plant=target_plant
    )  # Re-label for this plant only
    df_plant = engineer_features(df_plant, profiles)

    train_df, val_df = train_test_split(
        df_plant, test_size=0.25, random_state=42, stratify=df_plant["health_status"]
    )

    train_dataset = PlantHealthDataset(train_df, sequence_length=20)
    val_dataset = PlantHealthDataset(val_df, sequence_length=20)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PlantHealthSimple(
        input_size=len(train_dataset.feature_cols),
        num_plants=len(profiles),
        embedding_dim=12,
        num_classes=3,
    )

    class_counts = Counter(train_df["health_status"])
    weights = [
        1.0 / class_counts.get(c, 1)
        for c in ["Healthy", "Moderate Stress", "High Stress"]
    ]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = f"models/best_{target_plant.replace(' ', '_')}.pth"

    print(
        f"Starting training for {target_plant} (Early Stopping patience = {patience})\n"
    )

    for epoch in range(100):  # Max epochs
        # Train
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

        # Validate
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

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"  → New best model saved for {target_plant} ({val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  → Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"✅ Finished {target_plant}. Best Val Acc: {best_val_acc:.2f}%\n")
    return best_val_acc


def main():
    ensure_dir("models")
    profiles = load_plant_care_profiles()

    results = {}
    for plant in sorted(profiles.keys()):
        acc = train_for_single_plant(plant, patience=12)
        results[plant] = acc

    print("\n" + "=" * 70)
    print("FINAL PER-PLANT TRAINING RESULTS")
    print("=" * 70)
    for plant, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{plant:28}: {acc:.2f}%")

    print("\n✅ Per-plant training with early stopping completed!")


if __name__ == "__main__":
    main()
