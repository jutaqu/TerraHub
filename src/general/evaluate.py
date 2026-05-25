"""
General Pipeline - Evaluation
"""

import torch

from shared.data_loader import PlantHealthDataset
from shared.data_preprocessing import preprocess_data
from shared.evaluator import evaluate_per_plant
from shared.feature_engineering import engineer_features
from shared.hybrid_model import PlantHealthSimple
from shared.utils import get_device, load_plant_care_profiles


def evaluate_general_model():
    device = get_device()

    # Load best model
    profiles = load_plant_care_profiles()
    df = preprocess_data()
    df = engineer_features(df, profiles)

    # Use last 25% as holdout for final evaluation
    _, val_df = torch.utils.data.random_split(
        range(len(df)), [int(0.75 * len(df)), len(df) - int(0.75 * len(df))]
    )
    val_df = df.iloc[val_df.indices]

    dataset = PlantHealthDataset(val_df, sequence_length=20)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = PlantHealthSimple(
        input_size=len(dataset.feature_cols),
        num_plants=len(profiles),
        embedding_dim=12,
        num_classes=3,
    )

    model.load_state_dict(
        torch.load("models/best_general_model.pth", map_location=device)
    )
    model.eval()

    print("\n=== Final Evaluation on General Model (Holdout Set) ===\n")
    evaluate_per_plant(model, loader, dataset, device)

    print("\n✅ General model evaluation completed.")


if __name__ == "__main__":
    evaluate_general_model()
