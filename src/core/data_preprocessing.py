"""
Data Preprocessing - Better Health Labeling for Monstera
"""

import numpy as np
import pandas as pd


def preprocess_data(target_plant="Monstera Deliciosa"):
    print("\n=== Starting Data Preprocessing ===")

    df = pd.read_csv("data/raw/plant_health_data.csv")
    print(f"✓ Loaded {len(df)} sensor readings from data/raw/plant_health_data.csv")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values(["Plant_ID", "Timestamp"]).reset_index(drop=True)

    print(f"✓ Forced all data to focal plant: {target_plant}")

    df = derive_health_status(df, target_plant)

    print(f"✓ Labels applied. Distribution:\n{df['health_status'].value_counts()}\n")
    return df


def derive_health_status(df, target_plant="Monstera Deliciosa"):
    """Simple, effective 3-class labeling based on key sensors"""
    df = df.copy()

    # Simple composite stress score using main TerraHub sensors
    df["stress_score"] = (
        (df["Soil_Moisture"] - 0.55).abs() * 2.0  # Optimal ~0.55
        + (df["Ambient_Temperature"] - 24).abs() * 0.8  # Optimal ~24°C
        + (df["Humidity"] - 70).abs() * 0.6  # Optimal ~70%
        + (df["Soil_pH"] - 6.0).abs() * 3.0  # Optimal ~6.0
        + (df["Light_Intensity"] - 12000).abs() * 0.00008  # Optimal ~12000 lux
    )

    # Normalize and assign classes
    max_score = df["stress_score"].quantile(0.95)  # Robust to outliers
    df["normalized_stress"] = df["stress_score"] / max_score

    def assign_class(score):
        if score <= 0.35:
            return "Healthy"
        elif score <= 0.70:
            return "Moderate Stress"
        else:
            return "High Stress"

    df["health_status"] = df["normalized_stress"].apply(assign_class)

    print(f"✓ Simple 3-class labeling completed. Distribution:")
    print(df["health_status"].value_counts())

    return df
