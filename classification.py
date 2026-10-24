"""Compare KNN, a decision tree, an RBF SVM, and a 1D CNN on the WDBC dataset."""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import nn


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "classification"
RANDOM_STATE = 42
STATISTICS = ("mean", "se", "worst")
MEASUREMENTS = (
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension",
)
DISPLAY_NAMES = {
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "rbf_svm": "RBF SVM",
    "cnn": "1D CNN",
}


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
    for statistic in STATISTICS:
        names.extend(f"{statistic}_{measurement}" for measurement in MEASUREMENTS)
    data = pd.read_csv(DATA_PATH, header=None, names=names)
    features = data.drop(columns=["id", "diagnosis"])
    target = data["diagnosis"].map({"B": 0, "M": 1})
    return features, target


def build_searches(cv):
    return {
        "knn": GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 9, 11, 15, 21]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
        "decision_tree": GridSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [2, 3, 4, 5, 8, None], "min_samples_leaf": [1, 2, 5]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
        "rbf_svm": GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf"))]),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
    }


def plot_complexity_curves(X_train, y_train, cv):
    """Train vs cross-validation accuracy as each model's complexity parameter changes."""
    curves = [
        (
            "knn",
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
            "model__n_neighbors",
            np.arange(1, 32, 2),
            "K neighbours (small K = more complex)",
            False,
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            "max_depth",
            np.arange(1, 16),
            "Maximum depth (deeper = more complex)",
            False,
        ),
        (
            "rbf_svm",
            Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", C=10))]),
            "model__gamma",
            np.logspace(-4, 1, 11),
            "gamma, C=10 (large gamma = more complex)",
            True,
        ),
    ]
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (name, estimator, parameter, values, label, log_scale) in zip(axes, curves):
        train_scores, cv_scores = validation_curve(
            estimator, X_train, y_train,
            param_name=parameter, param_range=values,
            cv=cv, scoring="accuracy", n_jobs=-1,
        )
        train_mean, cv_mean = train_scores.mean(axis=1), cv_scores.mean(axis=1)
        cv_std = cv_scores.std(axis=1)
        axis.plot(values, train_mean, marker="o", label="Train accuracy")
        axis.plot(values, cv_mean, marker="o", label="CV accuracy")
        axis.fill_between(values, cv_mean - cv_std, cv_mean + cv_std, alpha=0.2)
        if log_scale:
            axis.set_xscale("log")
        axis.set(title=DISPLAY_NAMES[name], xlabel=label, ylabel="Accuracy")
        axis.legend()
        rows.extend(
            {"model": name, "parameter": parameter, "value": value,
             "train_accuracy": train, "cv_accuracy": validation}
            for value, train, validation in zip(values, train_mean, cv_mean)
        )
    figure.suptitle("Model complexity vs overfitting")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "complexity_curves.png", dpi=160)
    plt.close(figure)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "complexity_curves.csv", index=False)


class WDBCConvNet(nn.Module):
    """1D CNN over the ten nucleus measurements, with mean/se/worst as input channels."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(len(STATISTICS), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * len(MEASUREMENTS), 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def train_cnn(X_train, y_train, X_test, epochs=300, patience=25, batch_size=32):
    """Train with early stopping on a 15% validation split taken from the training set."""
    torch.manual_seed(RANDOM_STATE)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )
    scaler = StandardScaler().fit(X_fit)

    def to_tensor(values):
        scaled = scaler.transform(values).reshape(-1, len(STATISTICS), len(MEASUREMENTS))
        return torch.tensor(scaled, dtype=torch.float32)

    X_fit_t, X_val_t = to_tensor(X_fit), to_tensor(X_val)
    y_fit_t = torch.tensor(y_fit.to_numpy(), dtype=torch.float32)
    y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    model = WDBCConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    history = []
    best_loss, best_state, best_epoch, stale_epochs = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(X_fit_t), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss = loss_function(model(X_fit_t[batch]), y_fit_t[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_function(model(X_fit_t), y_fit_t).item()
            val_loss = loss_function(model(X_val_t), y_val_t).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, stale_epochs = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(to_tensor(X_train))).numpy()
        test_scores = torch.sigmoid(model(to_tensor(X_test))).numpy()
    return train_scores, test_scores, pd.DataFrame(history), best_epoch


def plot_cnn_history(history, best_epoch, path):
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(history["epoch"], history["train_loss"], label="Train loss")
    axis.plot(history["epoch"], history["val_loss"], label="Validation loss")
    axis.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    axis.set(title="1D CNN training history", xlabel="Epoch", ylabel="Binary cross-entropy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


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


def evaluate(name, y_train, train_predictions, y_test, test_predictions, test_scores, extra):
    plot_confusion_matrix(
        confusion_matrix(y_test, test_predictions),
        f"{DISPLAY_NAMES[name]} confusion matrix",
        OUTPUT_DIR / f"{name}_confusion_matrix.png",
    )
    return {
        "model": name,
        **extra,
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions),
        "recall": recall_score(y_test, test_predictions),
        "f1": f1_score(y_test, test_predictions),
        "roc_auc": roc_auc_score(y_test, test_scores),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics = []
    roc_data = {}
    for name, search in build_searches(cv).items():
        search.fit(X_train, y_train)
        scores = search.predict_proba(X_test)[:, 1] if hasattr(search, "predict_proba") else search.decision_function(X_test)
        metrics.append(evaluate(
            name, y_train, search.predict(X_train), y_test, search.predict(X_test), scores,
            {"best_parameters": str(search.best_params_), "cv_roc_auc": search.best_score_},
        ))
        roc_data[name] = roc_curve(y_test, scores)

    train_scores, test_scores, history, best_epoch = train_cnn(X_train, y_train, X_test)
    history.to_csv(OUTPUT_DIR / "cnn_training_history.csv", index=False)
    plot_cnn_history(history, best_epoch, OUTPUT_DIR / "cnn_training_history.png")
    metrics.append(evaluate(
        "cnn", y_train, (train_scores >= 0.5).astype(int), y_test, (test_scores >= 0.5).astype(int), test_scores,
        {"best_parameters": f"early stopping at epoch {best_epoch}", "cv_roc_auc": np.nan},
    ))
    roc_data["cnn"] = roc_curve(y_test, test_scores)

    plot_complexity_curves(X_train, y_train, cv)

    results = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    results.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, (false_positive_rate, true_positive_rate, _) in roc_data.items():
        auc = results.loc[results["model"] == name, "roc_auc"].iloc[0]
        axis.plot(false_positive_rate, true_positive_rate, label=f"{DISPLAY_NAMES[name]} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], "k--", label="Chance")
    axis.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160)
    plt.close(figure)
    print(results.to_string(index=False))
    best = results.iloc[0]
    print(f"\nBest classifier by ROC-AUC: {DISPLAY_NAMES[best['model']]} (AUC={best['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
