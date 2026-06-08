import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_training_results():
    path = "reports/training_results.json"
    if not os.path.exists(path):
        path = "../reports/training_results.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: reports/training_results.json")

    with open(path, "r") as f:
        data = json.load(f)
    print(f"✓ Loaded training results from {path}")
    return data


def plot_class_distribution(y_true, save_path="reports/class_distribution.png"):
    plt.figure(figsize=(10, 6))
    counts = pd.Series(y_true).value_counts().sort_index()
    class_names = ["Healthy", "Moderate Stress", "High Stress"]

    ax = sns.barplot(
        x=[class_names[i] for i in counts.index],
        y=counts.values,
        palette="viridis",
        hue=None,
        legend=False,
    )

    plt.title("Class Distribution - Monstera Deliciosa", fontsize=16, fontweight="bold")
    plt.xlabel("Health Status")
    plt.ylabel("Number of Samples")

    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Class distribution plot saved → {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path="reports/confusion_matrix.png"):
    class_names = ["Healthy", "Moderate Stress", "High Stress"]
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized %)", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Confusion matrix saved → {save_path}")
    plt.close()


def plot_training_history(history, save_path="reports/training_history.png"):
    if not history or history.get("train_loss") is None:
        print("⚠ No training history available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2, color="red")
    axes[0].plot(
        history["val_loss"], label="Validation Loss", linewidth=2, color="gold"
    )
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2, color="red")
    axes[1].plot(
        history["val_acc"], label="Validation Accuracy", linewidth=2, color="gold"
    )
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Training history plot saved → {save_path}")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, save_path="reports/per_class_metrics.png"):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Healthy", "Moderate Stress", "High Stress"],
        output_dict=True,
        zero_division=0,
    )

    metrics = ["precision", "recall", "f1-score"]
    class_names = ["Healthy", "Moderate Stress", "High Stress"]
    data = []

    for cls in class_names:
        for metric in metrics:
            data.append(
                {
                    "Class": cls,
                    "Metric": metric.capitalize(),
                    "Score": report[cls][metric],
                }
            )

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="Class", y="Score", hue="Metric", palette="viridis")
    plt.title("Per-Class Performance Metrics", fontsize=16, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Per-class metrics plot saved → {save_path}")
    plt.close()


def plot_prediction_confidence(y_prob, save_path="reports/prediction_confidence.png"):
    if not y_prob:
        print("⚠ No prediction probabilities available.")
        return

    y_prob = np.array(y_prob)
    confidence = np.max(y_prob, axis=1)

    plt.figure(figsize=(10, 6))
    sns.histplot(confidence, bins=30, kde=True, color="skyblue")
    plt.title(
        "Model Prediction Confidence Distribution", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.axvline(x=0.8, color="red", linestyle="--", label="High Confidence Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Prediction confidence plot saved → {save_path}")
    plt.close()


def plot_sensor_distributions(
    X_test, y_true, feature_cols, save_path="reports/sensor_distributions.png"
):
    """Fixed: Handle length mismatch between X_test and y_true"""
    min_len = min(len(X_test), len(y_true))
    X_trim = X_test[:min_len]
    y_trim = y_true[:min_len]

    df_plot = pd.DataFrame(X_trim, columns=feature_cols)
    class_names = ["Healthy", "Moderate Stress", "High Stress"]
    df_plot["Health_Status"] = [class_names[label] for label in y_trim]

    n_features = len(feature_cols)
    rows = (n_features + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes]

    for i, col in enumerate(feature_cols):
        if i < len(axes):
            sns.boxplot(
                data=df_plot, x="Health_Status", y=col, ax=axes[i], palette="viridis"
            )
            axes[i].set_title(f"{col.replace('_', ' ').title()} by Health Status")
            axes[i].tick_params(axis="x", rotation=15)

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Sensor distributions (boxplots) saved → {save_path}")
    plt.close()


def plot_feature_correlation(
    X_test, feature_cols, save_path="reports/feature_correlation.png"
):
    """Real correlation heatmap"""
    df = pd.DataFrame(X_test, columns=feature_cols)
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap (Test Set)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Real feature correlation plot saved → {save_path}")
    plt.close()


def generate_full_report():
    data = load_training_results()

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    best_acc = data.get("best_acc", 0.0)
    history = data.get("history", {})
    y_prob = data.get("y_prob", None)
    X_test = data.get("X_test", None)
    feature_cols = data["feature_cols"]

    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT VISUALIZATIONS")
    print("=" * 80)

    plot_class_distribution(y_true)
    plot_confusion_matrix(y_true, y_pred)
    plot_training_history(history)
    plot_per_class_metrics(y_true, y_pred)
    plot_prediction_confidence(y_prob)

    if X_test:
        plot_sensor_distributions(X_test, y_true, feature_cols)
        plot_feature_correlation(X_test, feature_cols)

    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Best Validation Accuracy : {best_acc:.2f}%\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Healthy", "Moderate Stress", "High Stress"],
            zero_division=0,
        )
    )

    print(f"\n✅ All visualizations saved to the 'reports/' folder!")
    print("=" * 80)


if __name__ == "__main__":
    generate_full_report()
