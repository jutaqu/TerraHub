import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from core.model import PlantHealthCNN

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=[8.0, 1.0, 1.0], gamma=4.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(DEVICE)
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_focal_species():
    print("\n=== Training 1D-CNN on Monstera Deliciosa ===")
    from core.data_loader import PlantHealthDataset
    from core.data_preprocessing import preprocess_data
    from core.feature_engineering import engineer_features

    df = preprocess_data(target_plant="Monstera Deliciosa")

    # Chronological split
    split_idx = int(0.80 * len(df))
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_holdout = df.iloc[split_idx:].copy().reset_index(drop=True)

    df_train = engineer_features(df_train)
    df_holdout = engineer_features(df_holdout)

    print(
        f"✓ Chronological split → Train: {len(df_train)} rows | Holdout: {len(df_holdout)} rows"
    )

    # Oversampling Healthy class
    healthy_mask = df_train["health_status"] == "Healthy"
    if healthy_mask.sum() > 0:
        df_healthy = df_train[healthy_mask].sample(n=600, replace=True, random_state=42)
        df_train = pd.concat([df_train, df_healthy], ignore_index=True)
        print(
            f"✓ Oversampled Healthy → {df_train['health_status'].value_counts().get('Healthy', 0)} samples"
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

    # ====================== MODEL SETUP ======================
    model = PlantHealthCNN(input_size=len(feature_cols)).to(DEVICE)

    criterion = FocalLoss(alpha=[8.0, 1.0, 1.0], gamma=4.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.00025, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=25, factor=0.5
    )

    # ====================== METRIC TRACKING ======================
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0
    patience = 20
    counter = 0

    print("\nStarting Training...\n")

    for epoch in range(100):
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if len(batch) == 2:
                seq, labels = batch
            else:
                seq, _, labels = batch
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            preds = outputs.argmax(1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        epoch_val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    seq, labels = batch
                else:
                    seq, _, labels = batch
                seq, labels = seq.to(DEVICE), labels.to(DEVICE)

                outputs = model(seq)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d} → Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_focal_monstera_cnn.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("\n→ Early stopping triggered.")
                break

    # ====================== FINAL EVALUATION & SAVE ======================
    print("\n" + "=" * 70)
    print(f"Best Validation Accuracy : {best_acc:.2f}%\n")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["Healthy", "Moderate Stress", "High Stress"],
            zero_division=0,
        )
    )

    # Visualize results
    os.makedirs("reports", exist_ok=True)
    results = {
        "y_true": [int(x) for x in all_labels],
        "y_pred": [int(x) for x in all_preds],
        "feature_cols": feature_cols,
        "best_acc": float(best_acc),
        "class_names": ["Healthy", "Moderate Stress", "High Stress"],
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(all_labels)),
        "n_features": int(len(feature_cols)),
        "X_test": df_holdout[feature_cols].values.tolist(),
        "y_prob": [[float(p) for p in prob] for prob in all_probs],
        "history": {
            "train_loss": [float(x) for x in train_losses],
            "val_loss": [float(x) for x in val_losses],
            "train_acc": [float(x) for x in train_accs],
            "val_acc": [float(x) for x in val_accs],
        },
    }

    with open("reports/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✓ Rich training results saved to reports/training_results.json")
    print(f"✅ Training completed! Best Validation Accuracy: {best_acc:.2f}%")
    print("✅ Best model saved to models/best_focal_monstera_cnn.pth")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_focal_species()
