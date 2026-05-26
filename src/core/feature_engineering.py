"""
Feature Engineering - Core Sensors + Valuable Temporal Features
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame):
    df = df.copy()

    print("\n=== Starting Core Feature Engineering ===")

    # Standardize column names
    rename_map = {
        "Soil_Moisture": "soil_moisture",
        "Ambient_Temperature": "air_temp",
        "Soil_Temperature": "soil_temp",
        "Humidity": "humidity",
        "Light_Intensity": "light_lux",
        "Soil_pH": "soil_ph",
        "Electrochemical_Signal": "ec",
    }

    for old, new in rename_map.items():
        if old in df.columns:
            df[new] = df[old]

    core_sensors = [
        "soil_moisture",
        "air_temp",
        "soil_temp",
        "humidity",
        "light_lux",
        "soil_ph",
        "ec",
    ]
    core_sensors = [col for col in core_sensors if col in df.columns]

    print(f"Core Sensors: {core_sensors}")

    # === Valuable Temporal Features ===
    for col in core_sensors:
        # Rolling mean (trend)
        df[f"{col}_roll_mean_5"] = df[col].rolling(window=5, min_periods=1).mean()

        # Rolling volatility
        df[f"{col}_roll_std_5"] = (
            df[col].rolling(window=5, min_periods=1).std().fillna(0)
        )

        # Rate of change (most important for plant stress)
        df[f"{col}_delta"] = df[col].diff().fillna(0)

    # High-value interactions
    if "humidity" in df.columns and "air_temp" in df.columns:
        df["humidity_temp_interaction"] = df["humidity"] * df["air_temp"]

    if "soil_moisture" in df.columns and "soil_temp" in df.columns:
        df["moisture_temp_ratio"] = df["soil_moisture"] / (df["soil_temp"] + 1)

    # Final column selection
    keep_cols = (
        core_sensors
        + [f"{col}_roll_mean_5" for col in core_sensors]
        + [f"{col}_delta" for col in core_sensors]
        + ["humidity_temp_interaction", "moisture_temp_ratio", "health_status"]
    )

    df = df[keep_cols]

    print(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")
    print("✅ Feature engineering completed!\n")

    return df
