import os
from pathlib import Path

# Create data directory
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Plant-Health-Data dataset...")

try:
    import kaggle

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(
        dataset="ziya07/plant-health-data", path=data_dir, unzip=True
    )
    print("✅ Dataset downloaded successfully!")
    print(f"Files saved to: {data_dir.resolve()}")

except Exception as e:
    print(f"❌ Kaggle API error: {e}")
    print("\nAlternative: Please download manually from:")
    print("https://www.kaggle.com/datasets/ziya07/plant-health-data")

# List downloaded files
print("\nDownloaded files:")
for file in data_dir.glob("*.csv"):
    print(f"   - {file.name}")
