"""Implement PCA with a covariance matrix and eigenvalue decomposition."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "wholesale-customers" / "Wholesale customers data.csv"
OUTPUT_DIR = ROOT / "outputs" / "pca"
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES].to_numpy(dtype=float)
    pca = PCAFromScratch().fit(X)

    summary = pd.DataFrame({
        "component": np.arange(1, len(pca.eigenvalues_) + 1),
        "eigenvalue": pca.eigenvalues_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": pca.cumulative_explained_variance_,
    })
    summary.to_csv(OUTPUT_DIR / "eigenvalues_and_explained_variance.csv", index=False)
    pd.DataFrame(pca.eigenvectors_, index=FEATURES).to_csv(OUTPUT_DIR / "eigenvectors.csv")
    pd.DataFrame(pca.transform(X, 2), columns=["PC1", "PC2"]).to_csv(OUTPUT_DIR / "data_2_components.csv", index=False)
    pd.DataFrame(pca.transform(X, 3), columns=["PC1", "PC2", "PC3"]).to_csv(OUTPUT_DIR / "data_3_components.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(summary["component"], summary["cumulative_explained_variance"], marker="o")
    axis.axhline(0.90, color="gray", linestyle="--", label="90%")
    axis.set(title="Cumulative explained variance", xlabel="Number of components", ylabel="Cumulative ratio", ylim=(0, 1.05))
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "cumulative_explained_variance.png", dpi=160)
    plt.close(figure)

    reduced = pca.transform(X, 2)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(reduced[:, 0], reduced[:, 1], alpha=0.65, edgecolors="none")
    axis.set(title="Wholesale customers projected onto first two PCs", xlabel="PC1", ylabel="PC2")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "first_two_components.png", dpi=160)
    plt.close(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
