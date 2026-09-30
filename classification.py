"""Compare KNN, a decision tree, and an RBF SVM on the WDBC dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "classification"


def get_data_path():
    candidates = [
        ROOT / "datasets" / "cancer dataset" / "wdbc.data",
        ROOT / "breast-cancer" / "wdbc.data",
        ROOT.parent / "datasets" / "cancer dataset" / "wdbc.data",
        ROOT.parent / "breast-cancer" / "wdbc.data",
        Path("datasets/cancer dataset/wdbc.data"),
        Path("breast-cancer/wdbc.data"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return ROOT / "breast-cancer" / "wdbc.data"


DATA_PATH = get_data_path()


def load_data():
    names = ["id", "diagnosis"]
    for statistic in ("mean", "se", "worst"):
        names.extend(f"{statistic}_{feature}" for feature in (
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry",
            "fractal_dimension",
        ))
    data = pd.read_csv(DATA_PATH, header=None, names=names)
    features = data.drop(columns=["id", "diagnosis"])
    target = data["diagnosis"].map({"B": 0, "M": 1})
    return features, target


def plot_confusion_matrix(matrix, title, path):
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    searches = {
        "knn": GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 9, 11, 15, 21]},
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        ),
        "decision_tree": GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [2, 3, 4, 5, 8, None], "min_samples_leaf": [1, 2, 5]},
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        ),
        "rbf_svm": GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=42))]),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
            },
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        ),
    }

    metrics = []
    roc_data = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        if hasattr(search, "predict_proba"):
            try:
                scores = search.predict_proba(X_test)[:, 1]
            except Exception:
                scores = search.decision_function(X_test)
        else:
            scores = search.decision_function(X_test)
        matrix = confusion_matrix(y_test, predictions)
        metrics.append({
            "model": name,
            "best_parameters": str(search.best_params_),
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, scores),
        })
        plot_confusion_matrix(matrix, f"{name} confusion matrix", OUTPUT_DIR / f"{name}_confusion_matrix.png")
        roc_data[name] = roc_curve(y_test, scores)

    results = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    results.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, (false_positive_rate, true_positive_rate, _) in roc_data.items():
        auc = results.loc[results["model"] == name, "roc_auc"].iloc[0]
        axis.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], "k--", label="Chance")
    axis.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160)
    plt.close(figure)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
