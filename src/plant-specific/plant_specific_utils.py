"""
Plant-Specific Utilities
"""

import pandas as pd

from shared.utils import print_feature_summary


def create_plant_specific_dataset(df: pd.DataFrame, target_plant: str, profiles: dict):
    """Force entire dataset to one plant and re-label using only that plant's profile."""
    print(f"\n=== Creating Plant-Specific Dataset for: {target_plant} ===")

    df = df.copy()

    # Force all rows to this plant
    df["plant_name"] = target_plant

    print(f"→ Forced all {len(df)} rows to plant: {target_plant}")

    # Re-label using only this plant's profile
    if target_plant in profiles:
        profile = profiles[target_plant]
        print(f"→ Using {target_plant}'s optimal ranges for labeling.")
    else:
        print(f"⚠️ {target_plant} not found in profiles.")
        profile = None

    # We will let data_preprocessing handle the actual labeling with target_plant flag
    print(f"Dataset prepared for {target_plant} — {len(df)} samples")
    return df
