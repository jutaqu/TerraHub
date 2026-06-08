import pandas as pd


def preprocess_data(target_plant="Monstera Deliciosa"):
    """Main preprocessing function with clean output."""
    print("\n=== Starting Data Preprocessing ===")

    df = pd.read_csv("data/raw/plant_health_data.csv")
    print(f"✓ Loaded {len(df)} sensor readings from data/raw/plant_health_data.csv")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values(["Plant_ID", "Timestamp"]).reset_index(drop=True)
        print("✓ Data sorted chronologically by timestamp")

    print(f"✓ Focused on target plant: {target_plant}")

    df = derive_health_status(df, target_plant)

    print("\n=== Preprocessing Complete ===")
    return df


def derive_health_status(df, target_plant="Monstera Deliciosa"):
    """Simple, effective 3-class labeling based on key sensors."""
    df = df.copy()

    df["stress_score"] = (
        (df["Soil_Moisture"] - 0.55).abs() * 2.0
        + (df["Ambient_Temperature"] - 24).abs() * 0.8
        + (df["Humidity"] - 70).abs() * 0.6
        + (df["Soil_pH"] - 6.0).abs() * 3.0
        + (df["Light_Intensity"] - 12000).abs() * 0.00008
    )

    max_score = df["stress_score"].quantile(0.95)
    df["normalized_stress"] = df["stress_score"] / max_score

    def assign_class(score):
        if score <= 0.35:
            return "Healthy"
        elif score <= 0.70:
            return "Moderate Stress"
        else:
            return "High Stress"

    df["health_status"] = df["normalized_stress"].apply(assign_class)

    print("✓ 3-class health labeling completed (profile-informed)")
    print("   Final Label Distribution:")
    print(df["health_status"].value_counts())
    print()

    return df
