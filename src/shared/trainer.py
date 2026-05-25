"""
Shared Training Logic with Enhanced Debugging (Option B)
"""

from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import ensure_dir, get_device, set_seed


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, config, experiment_name="shared"
    ):
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = None

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=config.patience
        )

        self.writer = SummaryWriter(
            f"runs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        self.best_val_acc = 0.0

        print(
            f"Trainer initialized — Total parameters: {sum(p.numel() for p in model.parameters())}"
        )

    def set_class_weights(self, class_weights: torch.Tensor):
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        print(f"✅ Class weights applied: {class_weights.cpu().numpy()}")

    def train_epoch(self):
        self.model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for x, plant_ids, y in self.train_loader:
            x, plant_ids, y = (
                x.to(self.device),
                plant_ids.to(self.device),
                y.to(self.device),
            )

            self.optimizer.zero_grad()
            outputs = self.model(x, plant_ids)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        acc = 100.0 * correct / total
        print(
            f"  [Train] Loss: {running_loss / len(self.train_loader):.4f} | Acc: {acc:.2f}%"
        )
        return acc

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]

        with torch.no_grad():
            for x, plant_ids, y in self.val_loader:
                x, plant_ids, y = (
                    x.to(self.device),
                    plant_ids.to(self.device),
                    y.to(self.device),
                )
                outputs = self.model(x, plant_ids)
                _, predicted = outputs.max(1)

                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                # Per-class statistics
                for label, pred in zip(y.cpu().numpy(), predicted.cpu().numpy()):
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        acc = 100.0 * correct / total

        print(f"  [Val]   Acc: {acc:.2f}%")
        print(
            f"  Per-class support → Healthy: {class_total[0]} | Moderate: {class_total[1]} | High: {class_total[2]}"
        )
        print(
            f"  Per-class accuracy → Healthy: {100 * class_correct[0] / class_total[0]:.1f}% | "
            f"Moderate: {100 * class_correct[1] / class_total[1]:.1f}% | High: {100 * class_correct[2] / class_total[2]:.1f}%"
        )

        return acc

    def train(self, num_epochs: int = None):
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        self.config.print_config()

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1:2d}/{num_epochs} ---")
            train_acc = self.train_epoch()
            val_acc = self.validate()
            self.scheduler.step(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), f"models/{self.config.model_save_name}"
                )
                print(f"  ★ New best model saved! ({val_acc:.2f}%)")

        print(
            f"\n✅ Training completed! Best Validation Accuracy: {self.best_val_acc:.2f}%"
        )
        return self.best_val_acc
