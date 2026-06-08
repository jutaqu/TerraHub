import pandas as pd


def engineer_features(df: pd.DataFrame):
    df = df.copy()

    print("\n=== Starting Core Feature Engineering ===")

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

    for col in core_sensors:
        df[f"{col}_roll_mean_5"] = df[col].rolling(window=5, min_periods=1).mean()
        df[f"{col}_delta"] = df[col].diff().fillna(0)

    if "humidity" in df.columns and "air_temp" in df.columns:
        df["humidity_temp_interaction"] = df["humidity"] * df["air_temp"]

    if "soil_moisture" in df.columns and "soil_temp" in df.columns:
        df["moisture_temp_ratio"] = df["soil_moisture"] / (df["soil_temp"] + 1)

    print(f"Final columns ({len(df.columns)}): {df.columns.tolist()}")
    print("✅ Feature engineering completed!\n")

    return df
