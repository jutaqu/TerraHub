"""
Shared Feature Engineering - Core sensors only
"""

from typing import Dict

import numpy as np
import pandas as pd

from utils import print_feature_summary


def calculate_vpd(temperature: float, humidity: float) -> float:
    svp = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
    return svp * (1 - humidity / 100)


def engineer_features(df: pd.DataFrame, profiles: Dict) -> pd.DataFrame:
    print("\n=== Starting Core Feature Engineering ===\n")

    df = df.copy()

    # Robust column mapping
    col_map = {
        "Soil_Moisture": "soil_moisture",
        "Ambient_Temperature": "air_temp",
        "Soil_Temperature": "soil_temp",
        "Humidity": "humidity",
        "Light_Intensity": "light_lux",
        "Soil_pH": "soil_ph",
        "Electrochemical_Signal": "ec",
    }
    df = df.rename(columns=col_map)

    core_cols = [
        "soil_moisture",
        "soil_temp",
        "air_temp",
        "humidity",
        "light_lux",
        "soil_ph",
        "ec",
    ]

    # Always try to calculate VPD
    if "air_temp" in df.columns and "humidity" in df.columns:
        df["vpd"] = df.apply(
            lambda row: calculate_vpd(row["air_temp"], row["humidity"]), axis=1
        )
        print("✓ Added VPD feature")
    else:
        print("⚠️ Warning: Could not calculate VPD (missing air_temp or humidity)")
        df["vpd"] = 1.2  # safe default

    # Final column selection
    available_core = [col for col in core_cols if col in df.columns]
    keep_cols = available_core + ["vpd", "plant_name", "health_status"]
    keep_cols = [col for col in keep_cols if col in df.columns]

    df = df[keep_cols].copy()

    print(f"Final columns kept: {keep_cols}")

    print_feature_summary(df)
    print("✅ Core feature engineering completed!\n")
    return df
