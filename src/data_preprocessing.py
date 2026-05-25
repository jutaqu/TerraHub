"""
Data Preprocessing Module - Probabilistic Labeling
==================================================
"""

import json
from typing import Dict

import numpy as np
import pandas as pd

from utils import print_feature_summary, set_seed


def load_plant_care_profiles() -> Dict:
    """Load plant care profiles from JSON."""
    try:
        with open("plant_care_profiles.json", "r") as f:
            profiles = json.load(f)
        print(f"✓ Loaded {len(profiles)} plant profiles from JSON")
        return profiles
    except Exception as e:
        print(f"❌ Error loading profiles: {e}")
        return {}


def derive_health_labels(df: pd.DataFrame, profiles: Dict) -> pd.DataFrame:
    """Probabilistic labeling - adds realistic noise to force genuine learning."""
    print("Deriving health status labels using probabilistic rule-based logic...")
    df = df.copy()

    for plant_name, profile in profiles.items():
        mask = df["plant_name"] == plant_name
        if not mask.any():
            continue

        subset = df[mask].copy()

        # Safe defaults
        moisture = subset.get("soil_moisture", pd.Series([25.0] * len(subset)))
        light = subset.get("light_lux", pd.Series([600.0] * len(subset)))
        vpd = subset.get("vpd", pd.Series([1.2] * len(subset)))

        moisture_dev = abs(
            moisture
            - (
                profile.get("soil_moisture", {"min": 20, "max": 40})["min"]
                + profile.get("soil_moisture", {"min": 20, "max": 40})["max"]
            )
            / 2
        ) / (
            profile.get("soil_moisture", {"min": 20, "max": 40})["max"]
            - profile.get("soil_moisture", {"min": 20, "max": 40})["min"]
            + 1e-8
        )

        light_dev = abs(
            light
            - (
                profile.get("light_lux", {"min": 400, "max": 800})["min"]
                + profile.get("light_lux", {"min": 400, "max": 800})["max"]
            )
            / 2
        ) / (
            profile.get("light_lux", {"min": 400, "max": 800})["max"]
            - profile.get("light_lux", {"min": 400, "max": 800})["min"]
            + 1e-8
        )

        vpd_dev = abs(vpd - 1.2)

        stress_score = 0.45 * moisture_dev + 0.30 * light_dev + 0.25 * vpd_dev

        # Probabilistic assignment with noise
        labels = []
        for score in stress_score:
            rand = np.random.random()
            if score < 0.7:
                label = "Healthy" if rand < 0.88 else "Moderate Stress"
            elif score < 1.5:
                label = (
                    "Moderate Stress"
                    if rand < 0.70
                    else ("Healthy" if rand < 0.92 else "High Stress")
                )
            else:
                label = "High Stress" if rand < 0.82 else "Moderate Stress"
            labels.append(label)

        df.loc[mask, "health_status"] = labels

    print(
        f"✓ Probabilistic labels applied. Distribution:\n{df['health_status'].value_counts()}"
    )
    return df


def preprocess_data() -> pd.DataFrame:
    print("=== Starting Data Preprocessing ===\n")
    set_seed(42)

    df = pd.read_csv("data/raw/plant_health_data.csv")
    print(f"✓ Loaded {len(df)} sensor readings.")

    profiles = load_plant_care_profiles()

    plant_list = list(profiles.keys())
    if plant_list:
        df["plant_name"] = [plant_list[i % len(plant_list)] for i in range(len(df))]
        print(f"✓ Assigned {len(plant_list)} plant types.")

    df = derive_health_labels(df, profiles)

    print_feature_summary(df)
    print("✅ Preprocessing completed!")
    return df
