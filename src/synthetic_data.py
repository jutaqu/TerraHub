"""
Synthetic Data Generation - Improved Large Scale
===============================================
Better control on soil_moisture bias while generating thousands of samples.
"""

from typing import Dict

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils import load_plant_care_profiles, set_seed


def generate_synthetic_data(
    df: pd.DataFrame, profiles: Dict = None, n_synthetic_per_plant: int = 800
) -> pd.DataFrame:
    if profiles is None:
        profiles = load_plant_care_profiles()

    set_seed(42)
    print(f"\n=== Generating large synthetic dataset ===\n")

    synthetic_list = []

    for plant_name in profiles.keys():
        subset = df[df["plant_name"] == plant_name].copy()
        if len(subset) < 10:
            continue

        print(f"Processing {plant_name} ({len(subset)} real samples)...")

        # Strict numeric selection
        exclude_cols = [
            "plant_id",
            "timestamp",
            "health_status",
            "plant_category",
            "Plant_Health_Status",
        ]
        numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        X = subset[feature_cols].copy()
        y = subset["health_status"].copy()

        # Force at least 2 classes for SMOTE
        if y.nunique() < 2:
            print(
                f"  → Only 1 class for {plant_name}, skipping SMOTE and duplicating with noise"
            )
            synth_plant = subset.copy()
            synth_plant = synth_plant.sample(
                n=n_synthetic_per_plant, replace=True, random_state=42
            )
            synthetic_list.append(synth_plant)
            continue

        le = LabelEncoder()
        X["plant_name_encoded"] = le.fit_transform(subset["plant_name"].astype(str))

        scaler = MinMaxScaler()
        X[feature_cols] = scaler.fit_transform(X[feature_cols])

        smote = SMOTENC(
            categorical_features=[X.columns.get_loc("plant_name_encoded")],
            random_state=42,
            k_neighbors=min(5, len(X) - 1),
        )

        X_res, y_res = smote.fit_resample(X, y)

        synth_plant = pd.DataFrame(X_res, columns=X.columns)
        synth_plant["health_status"] = y_res
        synth_plant["plant_name"] = le.inverse_transform(
            synth_plant["plant_name_encoded"].astype(int)
        )
        synth_plant = synth_plant.drop(columns=["plant_name_encoded"])
        synth_plant[feature_cols] = scaler.inverse_transform(synth_plant[feature_cols])

        synthetic_list.append(synth_plant)

    final_synthetic = pd.concat(synthetic_list, ignore_index=True)

    print(f"\n✅ Generated {len(final_synthetic)} synthetic samples")
    print(f"Label distribution:\n{final_synthetic['health_status'].value_counts()}")

    print("Applying domain-constrained noise...")
    final_synthetic = apply_domain_noise(final_synthetic, profiles, base_noise=0.015)

    return final_synthetic


def apply_domain_noise(
    df: pd.DataFrame, profiles: Dict, base_noise: float = 0.012
) -> pd.DataFrame:
    """Very conservative pure noise approach - no mean correction."""
    df = df.copy()

    # Extremely conservative noise factors
    noise_factors = {
        "soil_moisture": base_noise * 0.15,  # Very low to minimize upward drift
        "soil_temp": base_noise * 0.32,
        "air_temp": base_noise * 0.35,
        "humidity": base_noise * 0.42,
        "soil_ph": base_noise * 0.08,
        "ec": base_noise * 0.35,
    }

    for plant_name, profile in profiles.items():
        mask = df["plant_name"] == plant_name
        if not mask.any():
            continue

        for sensor, col in [
            ("soil_moisture", "soil_moisture"),
            ("soil_temp", "soil_temp"),
            ("air_temp", "air_temp"),
            ("humidity", "humidity"),
            ("soil_ph", "soil_ph"),
            ("ec", "ec"),
        ]:
            if col in df.columns and sensor in profile:
                opt = profile[sensor]
                current = df.loc[mask, col]

                noise_factor = noise_factors.get(sensor, base_noise)
                noise = np.random.normal(
                    0, noise_factor * (opt["max"] - opt["min"]), len(current)
                )
                new_values = current + noise

                # Very tight clamping to stay close to real values
                lower = opt["min"] * 0.93
                upper = opt["max"] * 1.07

                df.loc[mask, col] = np.clip(new_values, lower, upper)

    return df


# Standalone test
if __name__ == "__main__":
    from data_preprocessing import preprocess_data

    real_df = preprocess_data()
    profiles = load_plant_care_profiles()

    synthetic_df = generate_synthetic_data(real_df, profiles, n_synthetic=600)
    print("\nSynthetic data generation completed.")
