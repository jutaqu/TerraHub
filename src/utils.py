"""
Utility Functions for TerraHub Plant Health Model (CSCI500)
==========================================================
Contains helper functions for data handling, metrics, and visualization.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def load_plant_care_profiles(json_path: str = "plant_care_profiles.json") -> Dict:
    """Load plant optimal conditions from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def save_model(model: torch.nn.Module, path: str = "models/saved/best_model.pth"):
    """Save PyTorch model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✓ Model saved to {path}")


def load_model(model: torch.nn.Module, path: str = "models/saved/best_model.pth"):
    """Load PyTorch model."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"✓ Model loaded from {path}")
    return model


def get_device() -> torch.device:
    """Return best available device."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    return device


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None
):
    """Compute and return classification metrics."""
    if class_names is None:
        class_names = ["Healthy", "Moderate Stress", "High Stress"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm}


def print_feature_summary(df: pd.DataFrame):
    """Print summary of engineered features."""
    deviation_cols = [col for col in df.columns if "deviation" in col]
    print(f"\n=== Feature Summary ===")
    print(f"Total samples     : {len(df)}")
    print(f"Total features    : {len(df.columns)}")
    print(f"Deviation features: {len(deviation_cols)}")
    print(f"Health distribution:\n{df['health_status'].value_counts()}")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")
