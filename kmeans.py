"""Run K-Means from scratch on wholesale customer spending data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pca import CHANNELS, FEATURES, PCAFromScratch, load_data, spending_matrix


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "kmeans"
K_VALUES = range(2, 7)


class KMeansScratch:
    def __init__(self, n_clusters, max_iter=300, tolerance=1e-4, n_init=10, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_init = n_init
        self.random_state = random_state

    def _run_once(self, X, rng):
        centroids = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
        for _ in range(self.max_iter):
            distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            updated = centroids.copy()
            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                else:
                    updated[cluster] = X[rng.integers(len(X))]
            shift = np.sqrt(((updated - centroids) ** 2).sum(axis=1)).max()
            centroids = updated
            if shift <= self.tolerance:
                break
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        wcss = float(distances[np.arange(len(X)), labels].sum())
        return centroids, labels, wcss

    def fit(self, X):
        master_rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            seed = int(master_rng.integers(0, 2**32 - 1))
            candidate = self._run_once(X, np.random.default_rng(seed))
            if best is None or candidate[2] < best[2]:
                best = candidate
        self.centroids_, self.labels_, self.wcss_ = best
        return self


def standardize(values):
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1
    return (values - mean) / scale


def find_elbow(k_values, wcss):
    """Elbow = the K whose point lies farthest from the line joining the first and last points."""
    k = np.asarray(k_values, dtype=float)
    w = np.asarray(wcss, dtype=float)
    k_norm = (k - k.min()) / (k.max() - k.min())
    w_norm = (w - w.min()) / (w.max() - w.min())
    distances = np.abs(k_norm + w_norm - 1) / np.sqrt(2)
    return int(k[distances.argmax()])


def plot_elbow(results, elbow_k, path):
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(results["k"], results["wcss"], marker="o")
    for k, wcss in zip(results["k"], results["wcss"]):
        axis.annotate(f"{wcss:.0f}", (k, wcss), textcoords="offset points", xytext=(8, 6))
    elbow_wcss = results.loc[results["k"] == elbow_k, "wcss"].iloc[0]
    axis.scatter([elbow_k], [elbow_wcss], s=250, facecolors="none", edgecolors="red", linewidths=2, label=f"Elbow (K={elbow_k})")
    axis.set(
        title="K-Means elbow curve",
        xlabel="Number of clusters (K)",
        ylabel="Within-Cluster Sum of Squares (WCSS)",
        xticks=list(results["k"]),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_clusters_on_pca(X_log, X_standardized, model, path):
    pca = PCAFromScratch().fit(X_log)
    reduced = pca.transform(X_log, 2)
    # PCA standardizes with the same mean/std, so centroids project directly onto the eigenvectors.
    centroids = model.centroids_ @ pca.eigenvectors_[:, :2]
    figure, axis = plt.subplots(figsize=(8, 6))
    for cluster in range(model.n_clusters):
        mask = model.labels_ == cluster
        axis.scatter(reduced[mask, 0], reduced[mask, 1], alpha=0.6, edgecolors="none", label=f"Cluster {cluster} (n={mask.sum()})")
    axis.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, color="black", label="Centroids")
    ratios = pca.explained_variance_ratio_
    axis.set(
        title=f"K-Means clusters (K={model.n_clusters}) on the first two principal components",
        xlabel=f"PC1 ({ratios[0]:.1%} of variance)",
        ylabel=f"PC2 ({ratios[1]:.1%} of variance)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    X_log = spending_matrix(data)
    X = standardize(X_log)

    rows = []
    fitted_models = {}
    for k in K_VALUES:
        model = KMeansScratch(n_clusters=k).fit(X)
        fitted_models[k] = model
        reference = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({
            "k": k,
            "wcss": model.wcss_,
            "sklearn_wcss": reference.inertia_,
            "silhouette": silhouette_score(X, model.labels_),
            "cluster_sizes": " / ".join(str(size) for size in np.bincount(model.labels_)),
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "wcss_results.csv", index=False)

    elbow_k = find_elbow(results["k"], results["wcss"])
    plot_elbow(results, elbow_k, OUTPUT_DIR / "elbow_curve.png")

    chosen = fitted_models[elbow_k]
    labelled = data.assign(cluster=chosen.labels_)
    labelled[["Channel", "Region", "cluster"]].to_csv(OUTPUT_DIR / "cluster_labels.csv", index_label="customer")
    profile = labelled.groupby("cluster")[FEATURES].mean().round(0)
    profile.insert(0, "customers", labelled["cluster"].value_counts().sort_index())
    for code, name in CHANNELS.items():
        profile[f"{name.lower()}_share"] = labelled.groupby("cluster")["Channel"].apply(lambda channel: (channel == code).mean()).round(3)
    profile.to_csv(OUTPUT_DIR / "cluster_profiles.csv")
    plot_clusters_on_pca(X_log, X, chosen, OUTPUT_DIR / "clusters_on_pca.png")

    print(results.to_string(index=False))
    print(f"\nElbow (farthest point from the K=2..6 chord): K={elbow_k}")
    print(f"\nMean annual spending per cluster (K={elbow_k}):")
    print(profile.to_string())


if __name__ == "__main__":
    main()
