import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from core.data_loader import PlantHealthDataset
from core.data_preprocessing import preprocess_data
from core.feature_engineering import engineer_features
from src.training.train_focal import PlantHealthCNN  # Import the model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def test_best_model():
    print("\n=== Testing Best Model on Unseen Holdout Data ===\n")

    # 1. Load and preprocess data
    df = preprocess_data(target_plant="Monstera Deliciosa")
    df = engineer_features(df)

    # 2. Create holdout set (last 20% of data as unseen)
    holdout_size = int(0.20 * len(df))
    df_holdout = df.iloc[-holdout_size:].copy().reset_index(drop=True)
    print(f"✓ Using last {len(df_holdout)} samples as unseen holdout data")

    feature_cols = [
        "soil_moisture",
        "soil_temp",
        "air_temp",
        "humidity",
        "light_lux",
        "soil_ph",
        "ec",
    ]

    dataset = PlantHealthDataset(
        df_holdout, feature_cols=feature_cols, sequence_length=20
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. Load best model
    model = PlantHealthCNN(input_size=len(feature_cols), num_classes=3).to(device)
    model.load_state_dict(
        torch.load("models/best_focal_monstera_cnn.pth", map_location=device)
    )
    model.eval()
    print("✓ Loaded best saved model")

    # 4. Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    # 5. Final Results
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🎯 Final Test Accuracy on Unseen Data: {accuracy:.2f}%\n")

    print("Prediction Distribution:")
    print(Counter(all_preds))

    print("\n=== Detailed Classification Report ===")
    target_names = ["Healthy", "Moderate Stress", "High Stress"]
    print(
        classification_report(
            all_labels, all_preds, target_names=target_names, digits=3
        )
    )

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Optional: Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix - Best Model on Unseen Data")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix_holdout.png")
    print("\n✅ Confusion matrix saved as 'models/confusion_matrix_holdout.png'")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    test_best_model()
