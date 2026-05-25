"""
Feature Engineering - Fixed for Actual Column Names
"""

from typing import Dict

import numpy as np
import pandas as pd

from utils import print_feature_summary


def calculate_vpd(air_temp_c: float, relative_humidity: float) -> float:
    if pd.isna(air_temp_c) or pd.isna(relative_humidity):
        return np.nan
    svp = 0.6108 * np.exp((17.27 * air_temp_c) / (air_temp_c + 237.3))
    avp = svp * (relative_humidity / 100.0)
    return round(svp - avp, 4)


def engineer_features(df: pd.DataFrame, profiles: Dict) -> pd.DataFrame:
    print("\n=== Starting Core Feature Engineering ===\n")
    print(f"Input columns: {list(df.columns)}")

    # Map actual column names from your CSV
    col_map = {
        "Soil_Moisture": "soil_moisture",
        "Soil_Temperature": "soil_temp",
        "Ambient_Temperature": "air_temp",
        "Humidity": "humidity",
        "Light_Intensity": "light_lux",
        "Soil_pH": "soil_ph",
        "Electrochemical_Signal": "ec",
    }

    # Rename columns to standard lowercase
    df = df.rename(columns=col_map)

    # Core features
    core_cols = [
        "soil_moisture",
        "soil_temp",
        "air_temp",
        "humidity",
        "light_lux",
        "soil_ph",
        "ec",
    ]

    available_core = [col for col in core_cols if col in df.columns]
    df = df[available_core + ["plant_name", "health_status"]].copy()

    print(f"Kept core columns: {available_core}")

    # Add VPD
    if "air_temp" in df.columns and "humidity" in df.columns:
        df["vpd"] = df.apply(
            lambda row: calculate_vpd(row["air_temp"], row["humidity"]), axis=1
        )
        print("✓ Added VPD feature")

    # Add key deviation features
    for plant_name, profile in profiles.items():
        mask = df["plant_name"] == plant_name
        if not mask.any():
            continue

        for sensor_key, col_name in [
            ("soil_moisture", "soil_moisture"),
            ("light_lux", "light_lux"),
        ]:
            if sensor_key in profile and col_name in df.columns:
                opt = profile[sensor_key]
                midpoint = (opt["min"] + opt["max"]) / 2
                range_width = opt["max"] - opt["min"] + 1e-8
                df.loc[mask, f"{col_name}_deviation"] = (
                    df.loc[mask, col_name] - midpoint
                ) / range_width

    print_feature_summary(df)
    print("\n✅ Core feature engineering completed!")
    return df
