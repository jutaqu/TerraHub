import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Optional: UMAP
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print(
        "⚠️  umap-learn not installed. UMAP plot will be skipped. (pip install umap-learn)"
    )

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


class PlantHealthVisualizer:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.colors = {
            "Healthy": "#2ecc71",
            "Moderate Stress": "#f1c40f",
            "High Stress": "#e74c3c",
        }
        self.class_order = ["Healthy", "Moderate Stress", "High Stress"]

    def save_plot(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {path}")
        plt.close()

    def plot_class_distribution(self, df):
        plt.figure(figsize=(8, 5))
        counts = df["health_status"].value_counts()
        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            palette=self.colors,
            legend=False,
        )
        plt.title("Class Distribution (Severe Imbalance)", fontsize=14)
        plt.ylabel("Number of Samples")
        for i, v in enumerate(counts.values):
            plt.text(i, v + 8, str(v), ha="center", fontsize=11)
        plt.tight_layout()
        self.save_plot("class_distribution.png")

    def plot_sensor_distributions(self, df):
        sensors = [
            "soil_moisture",
            "air_temp",
            "humidity",
            "light_lux",
            "soil_ph",
            "ec",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        for i, sensor in enumerate(sensors):
            if sensor in df.columns:
                sns.boxplot(
                    data=df,
                    x="health_status",
                    y=sensor,
                    hue="health_status",
                    ax=axes[i],
                    palette=self.colors,
                    legend=False,
                )
                axes[i].set_title(
                    f"{sensor.replace('_', ' ').title()} by Health Status"
                )
                axes[i].tick_params(axis="x", rotation=15)
        plt.tight_layout()
        self.save_plot("sensor_boxplots.png")

    def plot_correlation_heatmap(self, df):
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        plt.figure(figsize=(12, 9))
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        self.save_plot("correlation_heatmap.png")

    def plot_tsne(self, df, feature_cols):
        X = df[feature_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)

        plt.figure(figsize=(11, 8))
        for label in self.class_order:
            if label in df["health_status"].values:
                mask = df["health_status"] == label
                plt.scatter(
                    X_tsne[mask, 0],
                    X_tsne[mask, 1],
                    label=label,
                    alpha=0.75,
                    s=45,
                    edgecolors="black",
                    linewidth=0.4,
                )
        plt.title("t-SNE Projection of Sensor Features", fontsize=14)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        self.save_plot("tsne_projection.png")

    def plot_umap(self, df, feature_cols):
        if not UMAP_AVAILABLE:
            print("⚠️  Skipping UMAP plot (umap-learn not installed)")
            return
        X = df[feature_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=20, min_dist=0.1
        )
        X_umap = reducer.fit_transform(X_scaled)

        plt.figure(figsize=(11, 8))
        for label in self.class_order:
            if label in df["health_status"].values:
                mask = df["health_status"] == label
                plt.scatter(
                    X_umap[mask, 0], X_umap[mask, 1], label=label, alpha=0.8, s=40
                )
        plt.title("UMAP Projection of Sensor Features", fontsize=14)
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        self.save_plot("umap_projection.png")

    def plot_pca_3d(self, df, feature_cols):
        X = df[feature_cols].values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        for label in self.class_order:
            if label in df["health_status"].values:
                mask = df["health_status"] == label
                ax.scatter(
                    X_pca[mask, 0],
                    X_pca[mask, 1],
                    X_pca[mask, 2],
                    label=label,
                    alpha=0.7,
                    s=40,
                )
        ax.set_title("3D PCA Projection of Sensor Features")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.tight_layout()
        self.save_plot("pca_3d_projection.png")

    def plot_radar_chart(self, df, feature_cols):
        avg = df.groupby("health_status")[feature_cols].mean()
        scaler = StandardScaler()
        avg_scaled = pd.DataFrame(
            scaler.fit_transform(avg), index=avg.index, columns=feature_cols
        )

        categories = feature_cols
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(11, 8), subplot_kw=dict(polar=True))
        for label in avg_scaled.index:
            values = avg_scaled.loc[label].values.tolist()
            values += values[:1]
            ax.plot(
                angles, values, linewidth=2.5, label=label, color=self.colors[label]
            )
            ax.fill(angles, values, alpha=0.25, color=self.colors[label])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([col.replace("_", " ").title() for col in categories])
        ax.set_title(
            "Average Sensor Profile per Health Status (Radar Chart)",
            fontsize=14,
            pad=30,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.0))
        plt.tight_layout()
        self.save_plot("radar_profiles.png")

    def plot_parallel_coordinates(self, df, feature_cols, sample_size=400):
        sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
        scaler = StandardScaler()
        normalized = scaler.fit_transform(sample[feature_cols])
        norm_df = pd.DataFrame(normalized, columns=feature_cols)
        norm_df["health_status"] = sample["health_status"].values

        plt.figure(figsize=(14, 7))
        pd.plotting.parallel_coordinates(
            norm_df,
            "health_status",
            color=[
                self.colors.get(c, "#95a5a6") for c in norm_df["health_status"].unique()
            ],
            alpha=0.55,
        )
        plt.title("Parallel Coordinates: Sensor Profiles by Health Status")
        plt.ylabel("Normalized Value")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        self.save_plot("parallel_coordinates.png")

    def generate_all(self, df, feature_cols=None):
        if feature_cols is None:
            feature_cols = [
                "soil_moisture",
                "air_temp",
                "soil_temp",
                "humidity",
                "light_lux",
                "soil_ph",
                "ec",
            ]

        print("🚀 Generating all visualizations for report...\n")

        self.plot_class_distribution(df)
        self.plot_sensor_distributions(df)
        self.plot_correlation_heatmap(df)
        self.plot_tsne(df, feature_cols)
        if UMAP_AVAILABLE:
            self.plot_umap(df, feature_cols)
        self.plot_pca_3d(df, feature_cols)
        self.plot_radar_chart(df, feature_cols)
        self.plot_parallel_coordinates(df, feature_cols)

        print(f"\n✅ All plots successfully generated in '{self.output_dir}/' folder!")


# =============== RUN ALL PLOTS ===============
if __name__ == "__main__":
    from core.data_preprocessing import preprocess_data
    from core.feature_engineering import engineer_features

    df = preprocess_data()
    df = engineer_features(df)

    viz = PlantHealthVisualizer()
    viz.generate_all(df)
