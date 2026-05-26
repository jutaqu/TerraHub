"""
Data Preprocessing - Focal & Transfer Ready
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .utils import load_plant_care_profiles, print_feature_summary, set_seed


def derive_health_labels(
    df: pd.DataFrame, profiles: Dict, target_plant: Optional[str] = None
) -> pd.DataFrame:
    print("Deriving health status labels using probabilistic logic...")
    df = df.copy()

    for plant_name, profile in profiles.items():
        if target_plant is not None and plant_name != target_plant:
            continue

        mask = (
            df["plant_name"] == plant_name
            if target_plant is None
            else pd.Series(True, index=df.index)
        )
        if not mask.any():
            continue

        subset = df[mask].copy()

        moisture = subset.get("soil_moisture", pd.Series([25.0] * len(subset)))
        light = subset.get("light_lux", pd.Series([600.0] * len(subset)))
        vpd = subset.get("vpd", pd.Series([1.2] * len(subset)))

        moisture_opt = profile.get("soil_moisture", {"min": 20, "max": 40})
        light_opt = profile.get("light_lux", {"min": 400, "max": 800})

        moisture_dev = abs(
            moisture - (moisture_opt["min"] + moisture_opt["max"]) / 2
        ) / (moisture_opt["max"] - moisture_opt["min"] + 1e-8)

        light_dev = abs(light - (light_opt["min"] + light_opt["max"]) / 2) / (
            light_opt["max"] - light_opt["min"] + 1e-8
        )

        vpd_dev = abs(vpd - 1.2)

        stress_score = 0.45 * moisture_dev + 0.30 * light_dev + 0.25 * vpd_dev

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

    print(f"✓ Labels applied. Distribution:\n{df['health_status'].value_counts()}")
    return df


def preprocess_data(target_plant: Optional[str] = None) -> pd.DataFrame:
    print("=== Starting Data Preprocessing ===\n")
    set_seed(42)

    df = pd.read_csv("data/raw/plant_health_data.csv")
    print(f"✓ Loaded {len(df)} sensor readings.")

    profiles = load_plant_care_profiles()

    plant_list = list(profiles.keys())
    if plant_list:
        if target_plant:
            df["plant_name"] = target_plant
            print(f"✓ Forced all data to focal plant: {target_plant}")
        else:
            df["plant_name"] = [plant_list[i % len(plant_list)] for i in range(len(df))]
            print(f"✓ Assigned {len(plant_list)} plant types (General mode)")

    df = derive_health_labels(df, profiles, target_plant)
    print_feature_summary(df)
    print("✅ Preprocessing completed!\n")
    return df
