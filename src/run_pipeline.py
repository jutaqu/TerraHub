import os
import subprocess
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))


def run_script(script_path: str, description: str):
    print(f"\n{'=' * 80}")
    print(f"🚀 Running: {description}")
    print(f"{'=' * 80}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=Path(__file__).parent.parent,
            check=True,
            text=True,
            capture_output=False,
        )
        print(f"✅ {description} completed successfully.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        return False


def main():
    print("🌱 Starting Plant Health Monitoring Complete Pipeline\n")

    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    # Step 1: Download / Prepare Dataset
    download_script = src_dir / "utils" / "download_dataset.py"
    if download_script.exists():
        success = run_script(str(download_script), "Dataset Download & Preparation")
        if not success:
            print("⚠️  Pipeline stopped due to dataset download failure.")
            sys.exit(1)
    else:
        print("⚠️  download_dataset.py not found. Skipping...")

    # Step 2: Train the Model
    train_script = src_dir / "training" / "train_focal.py"
    success = run_script(str(train_script), "Model Training (1D-CNN with Focal Loss)")
    if not success:
        print("❌ Training failed. Stopping pipeline.")
        sys.exit(1)

    # Step 3: Generate Visualizations
    viz_script = src_dir / "utils" / "visualizations.py"
    success = run_script(str(viz_script), "Visualization Generation & Report Saving")
    if not success:
        print("⚠️  Visualizations failed, but training completed.")

    print("\n" + "=" * 80)
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("📁 Check the following folders:")
    print("   - models/          → Best model checkpoint")
    print("   - reports/         → Training results + JSON")
    print("   - (visualizations) → All generated plots")
    print("=" * 80)


if __name__ == "__main__":
    main()
