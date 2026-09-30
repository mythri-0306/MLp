"""Compare gradient-descent, polynomial, Ridge, and Lasso regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "regression"


def get_data_path():
    candidates = [
        ROOT / "datasets" / "real estate" / "Real estate valuation data set.xlsx",
        ROOT / "real-estate" / "Real estate valuation data set.xlsx",
        ROOT.parent / "datasets" / "real estate" / "Real estate valuation data set.xlsx",
        ROOT.parent / "real-estate" / "Real estate valuation data set.xlsx",
        Path("datasets/real estate/Real estate valuation data set.xlsx"),
        Path("real-estate/Real estate valuation data set.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return ROOT / "real-estate" / "Real estate valuation data set.xlsx"


DATA_PATH = get_data_path()


class GradientDescentLinearRegression(RegressorMixin, BaseEstimator):
    def __init__(self, learning_rate=0.03, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_with_intercept = np.column_stack([np.ones(len(X_arr)), X_arr])
        self.weights_ = np.zeros(X_with_intercept.shape[1])
        self.loss_history_ = []
        for _ in range(self.epochs):
            errors = X_with_intercept @ self.weights_ - y_arr
            gradient = (2 / len(X_arr)) * (X_with_intercept.T @ errors)
            self.weights_ -= self.learning_rate * gradient
            self.loss_history_.append(float(np.mean(errors ** 2)))
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X_arr)), X_arr]) @ self.weights_


def load_data():
    data = pd.read_excel(DATA_PATH)
    X = data.drop(columns=["No", "Y house price of unit area"])
    y = data["Y house price of unit area"]
    return X, y


def score_model(name, model, X_train, X_test, y_train, y_test, rows):
    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    rows.append({
        "model": name,
        "train_mse": mean_squared_error(y_train, train_prediction),
        "test_mse": mean_squared_error(y_test, test_prediction),
        "test_rmse": mean_squared_error(y_test, test_prediction) ** 0.5,
        "test_r2": r2_score(y_test, test_prediction),
    })
    return model, train_prediction, test_prediction


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    rows = []

    gd_model = Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientDescentLinearRegression()),
    ])
    gd_model, _, _ = score_model("linear_gradient_descent", gd_model, X_train, X_test, y_train, y_test, rows)

    degrees = [1, 2, 3, 5, 8, 12]
    degree_rows = []
    for degree in degrees:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ])
        model.fit(X_train, y_train)
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
        degree_rows.append({
            "degree": degree,
            "train_mse": mean_squared_error(y_train, train_prediction),
            "test_mse": mean_squared_error(y_test, test_prediction),
            "test_rmse": mean_squared_error(y_test, test_prediction) ** 0.5,
            "test_r2": r2_score(y_test, test_prediction),
        })
    degree_results = pd.DataFrame(degree_rows)
    degree_results.to_csv(OUTPUT_DIR / "polynomial_degree_comparison.csv", index=False)
    best_degree = int(degree_results.loc[degree_results["test_mse"].idxmin(), "degree"])
    best_polynomial = Pipeline([
        ("scale", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=best_degree, include_bias=False)),
        ("model", LinearRegression()),
    ])
    score_model(f"polynomial_degree_{best_degree}", best_polynomial, X_train, X_test, y_train, y_test, rows)

    for name, estimator in (("ridge", Ridge()), ("lasso", Lasso(max_iter=100000))):
        search = GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", estimator)]),
            {"model__alpha": np.logspace(-4, 4, 17)},
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        fitted, _, _ = score_model(name, search, X_train, X_test, y_train, y_test, rows)
        rows[-1]["best_parameters"] = str(fitted.best_params_)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(degree_results["degree"], degree_results["train_mse"], marker="o", label="Train MSE")
    axis.plot(degree_results["degree"], degree_results["test_mse"], marker="o", label="Test MSE")
    axis.set(title="Polynomial regression: underfitting and overfitting", xlabel="Polynomial degree", ylabel="MSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "polynomial_fit_complexity.png", dpi=160)
    plt.close(figure)

    gd_inner = gd_model.named_steps["model"]
    figure, axis = plt.subplots(figsize=(7, 5))
    loss_data = getattr(gd_inner, "loss_history_", getattr(gd_inner, "loss_history", []))
    axis.plot(loss_data)
    axis.set(title="Gradient-descent training loss", xlabel="Epoch", ylabel="MSE")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "gradient_descent_loss.png", dpi=160)
    plt.close(figure)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
