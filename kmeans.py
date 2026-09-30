"""Run K-Means from scratch on wholesale customer spending data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "kmeans"


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
        self.centroids, self.labels_, self.wcss_ = best
        return self


def standardize(values):
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1
    return (values - mean) / scale


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    X = standardize(data[FEATURES].to_numpy(dtype=float))
    rows = []
    fitted_models = {}
    for k in range(2, 7):
        model = KMeansScratch(n_clusters=k)
        model.fit(X)
        fitted_models[k] = model
        rows.append({"k": k, "wcss": model.wcss_})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "wcss_results.csv", index=False)
    best_k = int(results.loc[results["wcss"].diff().abs().idxmax(), "k"])
    fitted_models[best_k].labels_.tofile(OUTPUT_DIR / "cluster_labels.csv", sep=",", format="%d")

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(results["k"], results["wcss"], marker="o")
    axis.set(title="K-Means elbow curve", xlabel="Number of clusters (K)", ylabel="Within-Cluster Sum of Squares")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "elbow_curve.png", dpi=160)
    plt.close(figure)
    print(results.to_string(index=False))
    print(f"The script records the WCSS for K=2..6. Select the elbow by visual inspection of elbow_curve.png.")


if __name__ == "__main__":
    main()
