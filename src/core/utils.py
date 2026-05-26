"""
Core Utilities
"""

import json
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def load_plant_care_profiles() -> Dict:
    try:
        with open("plant_care_profiles.json", "r") as f:
            profiles = json.load(f)
        print(f"✓ Loaded {len(profiles)} plant profiles from JSON")
        return profiles
    except Exception as e:
        print(f"❌ Error loading profiles: {e}")
        return {}


def print_feature_summary(df: pd.DataFrame, title="Feature Summary"):
    print(f"\n=== {title} ===")
    print(f"Total samples     : {len(df)}")
    print(f"Total features    : {len(df.columns)}")
    print(f"Health distribution:\n{df['health_status'].value_counts()}")
    print("✅ Summary completed!\n")


def ensure_dir(directory: str):
    os.makedirs(directory, exist_ok=True)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device
