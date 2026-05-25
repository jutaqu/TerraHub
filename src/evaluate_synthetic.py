"""
Synthetic Data Evaluation Module for TerraHub Plant Health Model
================================================================
Evaluates real and synthetic data with full feature engineering
and statistical comparison.
"""

import pandas as pd

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from synthetic_data import generate_synthetic_data
from utils import load_plant_care_profiles, print_feature_summary


def compare_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """Compare key feature distributions between real and synthetic data."""
    print("\n=== Statistical Distribution Comparison (Real vs Synthetic) ===\n")

    key_features = [
        "soil_moisture",
        "soil_temp",
        "air_temp",
        "humidity",
        "light_lux",
        "soil_ph",
        "ec",
        "vpd",
        "total_deviation_score",
    ]

    results = []
    for col in key_features:
        if col in real_df.columns and col in synthetic_df.columns:
            real = real_df[col].dropna()
            synth = synthetic_df[col].dropna()

            results.append(
                {
                    "feature": col,
                    "real_mean": round(real.mean(), 3),
                    "synth_mean": round(synth.mean(), 3),
                    "real_std": round(real.std(), 3),
                    "synth_std": round(synth.std(), 3),
                    "real_min": round(real.min(), 3),
                    "synth_min": round(synth.min(), 3),
                    "real_max": round(real.max(), 3),
                    "synth_max": round(synth.max(), 3),
                }
            )

    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))
    return comparison_df


def evaluate_synthetic_data():
    """Run full evaluation pipeline."""
    print("=== Synthetic Data Quality Evaluation ===\n")

    # 1. Preprocess real data
    real_df = preprocess_data()
    profiles = load_plant_care_profiles()

    # 2. Feature engineering on real data
    print("\nApplying feature engineering to REAL data...")
    real_df = engineer_features(real_df, profiles)

    # 3. Generate synthetic data (per-species)
    synthetic_df = generate_synthetic_data(real_df, profiles, n_synthetic_per_plant=300)

    # 4. Feature engineering on synthetic data
    print("\nApplying feature engineering to SYNTHETIC data...")
    synthetic_df = engineer_features(synthetic_df, profiles)

    # 5. Statistical comparison
    compare_distributions(real_df, synthetic_df)

    # 6. Final summary
    print("\n=== Final Synthetic Dataset Summary ===")
    print_feature_summary(synthetic_df)

    print("\n✅ Synthetic data evaluation completed successfully!")


if __name__ == "__main__":
    evaluate_synthetic_data()
