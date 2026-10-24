"""Implement PCA with a covariance matrix and eigenvalue decomposition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "pca"


def get_data_path():
    candidates = [
        ROOT / "datasets" / "whole sale customer" / "Wholesale customers data.csv",
        ROOT / "wholesale-customers" / "Wholesale customers data.csv",
        ROOT.parent / "datasets" / "whole sale customer" / "Wholesale customers data.csv",
        ROOT.parent / "wholesale-customers" / "Wholesale customers data.csv",
        Path("datasets/whole sale customer/Wholesale customers data.csv"),
        Path("wholesale-customers/Wholesale customers data.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return ROOT / "wholesale-customers" / "Wholesale customers data.csv"


DATA_PATH = get_data_path()
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
CHANNELS = {1: "Horeca", 2: "Retail"}


def load_data():
    return pd.read_csv(DATA_PATH)


def spending_matrix(data):
    """Annual spending is heavily right-skewed, so use log(1 + x) before standardizing."""
    return np.log1p(data[FEATURES].to_numpy(dtype=float))


class PCAFromScratch:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        standardized = (X - self.mean_) / self.scale_
        covariance = np.cov(standardized, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[order]
        self.eigenvectors_ = eigenvectors[:, order]
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X, n_components):
        standardized = (X - self.mean_) / self.scale_
        return standardized @ self.eigenvectors_[:, :n_components]


def compare_with_sklearn(pca, X):
    """Eigenvalues should match exactly; eigenvectors may differ only in sign."""
    reference = PCA().fit((X - pca.mean_) / pca.scale_)
    eigenvalue_gap = np.abs(reference.explained_variance_ - pca.eigenvalues_).max()
    eigenvector_gap = np.abs(np.abs(reference.components_.T) - np.abs(pca.eigenvectors_)).max()
    return eigenvalue_gap, eigenvector_gap


def plot_explained_variance(summary, path):
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(summary["component"], summary["explained_variance_ratio"], alpha=0.6, label="Individual")
    axis.plot(summary["component"], summary["cumulative_explained_variance"], marker="o", color="black", label="Cumulative")
    axis.axhline(0.90, color="gray", linestyle="--", label="90%")
    for component, ratio in zip(summary["component"], summary["cumulative_explained_variance"]):
        axis.annotate(f"{ratio:.1%}", (component, ratio), textcoords="offset points", xytext=(0, 8), ha="center")
    axis.set(
        title="Explained variance by principal component",
        xlabel="Principal component",
        ylabel="Explained variance ratio",
        xticks=summary["component"],
        ylim=(0, 1.1),
    )
    axis.legend(loc="center right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_first_two_components(reduced, channels, pca, path):
    """PC1 vs PC2 coloured by sales channel, with feature loadings drawn as arrows."""
    figure, axis = plt.subplots(figsize=(8, 6))
    for code, label in CHANNELS.items():
        mask = channels == code
        axis.scatter(reduced[mask, 0], reduced[mask, 1], alpha=0.6, edgecolors="none", label=label)
    arrow_scale = np.abs(reduced[:, :2]).max() * 0.8
    loadings = pca.eigenvectors_[:, :2]
    # Alternate label distance by angle so labels of near-parallel arrows do not overlap.
    angle_rank = np.argsort(np.argsort(np.arctan2(loadings[:, 1], loadings[:, 0])))
    for feature, (x, y), rank in zip(FEATURES, loadings, angle_rank):
        axis.arrow(0, 0, x * arrow_scale, y * arrow_scale, color="black", width=0.01, head_width=0.12, alpha=0.8)
        label_distance = arrow_scale * (1.1 if rank % 2 == 0 else 1.3)
        axis.text(x * label_distance, y * label_distance, feature, ha="center", va="center", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
    ratios = pca.explained_variance_ratio_
    axis.set(
        title="Wholesale customers on the first two principal components",
        xlabel=f"PC1 ({ratios[0]:.1%} of variance)",
        ylabel=f"PC2 ({ratios[1]:.1%} of variance)",
    )
    axis.legend(title="Channel")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_first_three_components(reduced, channels, pca, path):
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(projection="3d")
    for code, label in CHANNELS.items():
        mask = channels == code
        axis.scatter(reduced[mask, 0], reduced[mask, 1], reduced[mask, 2], alpha=0.6, label=label)
    ratios = pca.explained_variance_ratio_
    axis.set(
        title="Wholesale customers on the first three principal components",
        xlabel=f"PC1 ({ratios[0]:.1%})",
        ylabel=f"PC2 ({ratios[1]:.1%})",
        zlabel=f"PC3 ({ratios[2]:.1%})",
    )
    axis.legend(title="Channel")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    X = spending_matrix(data)
    pca = PCAFromScratch().fit(X)
    component_names = [f"PC{index}" for index in range(1, len(pca.eigenvalues_) + 1)]

    summary = pd.DataFrame({
        "component": np.arange(1, len(pca.eigenvalues_) + 1),
        "eigenvalue": pca.eigenvalues_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": pca.cumulative_explained_variance_,
    })
    summary.to_csv(OUTPUT_DIR / "eigenvalues_and_explained_variance.csv", index=False)
    pd.DataFrame(pca.eigenvectors_, index=FEATURES, columns=component_names).to_csv(OUTPUT_DIR / "eigenvectors.csv")

    reduced_2 = pca.transform(X, 2)
    reduced_3 = pca.transform(X, 3)
    pd.DataFrame(reduced_2, columns=component_names[:2]).to_csv(OUTPUT_DIR / "data_2_components.csv", index=False)
    pd.DataFrame(reduced_3, columns=component_names[:3]).to_csv(OUTPUT_DIR / "data_3_components.csv", index=False)

    channels = data["Channel"].to_numpy()
    plot_explained_variance(summary, OUTPUT_DIR / "cumulative_explained_variance.png")
    plot_first_two_components(reduced_2, channels, pca, OUTPUT_DIR / "first_two_components.png")
    plot_first_three_components(reduced_3, channels, pca, OUTPUT_DIR / "first_three_components.png")

    eigenvalue_gap, eigenvector_gap = compare_with_sklearn(pca, X)
    print(summary.to_string(index=False))
    print(f"\nVariance kept: 2 PCs = {pca.cumulative_explained_variance_[1]:.1%}, 3 PCs = {pca.cumulative_explained_variance_[2]:.1%}")
    print(f"Max difference vs sklearn PCA: eigenvalues {eigenvalue_gap:.2e}, |eigenvectors| {eigenvector_gap:.2e}")


if __name__ == "__main__":
    main()
