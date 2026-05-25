"""
Shared Evaluation Utilities
"""

from collections import Counter

import torch
from sklearn.metrics import classification_report


@torch.no_grad()
def evaluate_per_plant(model, val_loader, dataset, device):
    """Evaluate and return per-plant accuracy."""
    model.eval()
    plant_correct = Counter()
    plant_total = Counter()

    all_preds = []
    all_labels = []

    for x, plant_ids, y in val_loader:
        x, plant_ids, y = x.to(device), plant_ids.to(device), y.to(device)
        outputs = model(x, plant_ids)
        _, predicted = outputs.max(1)

        for p, true, pred in zip(
            plant_ids.cpu().numpy(), y.cpu().numpy(), predicted.cpu().numpy()
        ):
            plant_name = dataset.plant_encoder.inverse_transform([p])[0]
            plant_total[plant_name] += 1
            if true == pred:
                plant_correct[plant_name] += 1

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    print("\n=== Per-Plant Validation Accuracy ===")
    for plant in sorted(plant_total.keys()):
        acc = 100 * plant_correct[plant] / plant_total[plant]
        print(
            f"    {plant:25}: {acc:5.1f}%  ({plant_correct[plant]}/{plant_total[plant]})"
        )

    return all_preds, all_labels
