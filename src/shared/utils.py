"""
Shared Utilities - Comprehensive
"""

import json
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def load_plant_care_profiles() -> Dict:
    """Load plant care profiles from JSON."""
    try:
        path = "plant_care_profiles.json"
        with open(path, "r") as f:
            profiles = json.load(f)
        print(f"✓ Loaded {len(profiles)} plant profiles from JSON")
        return profiles
    except FileNotFoundError:
        print("❌ plant_care_profiles.json not found!")
        return {}
    except Exception as e:
        print(f"❌ Error loading profiles: {e}")
        return {}


def print_feature_summary(df: pd.DataFrame):
    """Print dataset summary."""
    print(f"\n=== Feature Summary ===")
    print(f"Total samples     : {len(df)}")
    print(f"Total features    : {len(df.columns)}")

    dev_cols = [col for col in df.columns if "deviation" in col.lower()]
    if dev_cols:
        print(f"Deviation features: {len(dev_cols)}")

    print(f"Health distribution:\n{df['health_status'].value_counts()}")
    print("✅ Feature summary completed!\n")


def ensure_dir(directory: str):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Ensured directory exists: {directory}")


def get_device():
    """Return best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device
