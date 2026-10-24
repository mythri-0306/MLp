"""Compare gradient-descent, polynomial, Ridge, and Lasso regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "regression"
RANDOM_STATE = 42
TARGET = "Y house price of unit area"
DEMO_FEATURE = "X3 distance to the nearest MRT station"
DEGREES = range(1, 7)
REGULARIZED_DEGREE = 3
ALPHA_GRIDS = {
    "ridge": np.logspace(-3, 4, 15),
    "lasso": np.logspace(-2, 1, 13),
}


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
    X = data.drop(columns=["No", TARGET])
    y = data[TARGET]
    return X, y


def polynomial_pipeline(degree, estimator):
    return Pipeline([
        ("scale", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", estimator),
    ])


def regression_scores(y_train, train_prediction, y_test, test_prediction):
    train_mse = mean_squared_error(y_train, train_prediction)
    test_mse = mean_squared_error(y_test, test_prediction)
    return {
        "train_mse": train_mse,
        "train_rmse": train_mse ** 0.5,
        "train_r2": r2_score(y_train, train_prediction),
        "test_mse": test_mse,
        "test_rmse": test_mse ** 0.5,
        "test_r2": r2_score(y_test, test_prediction),
    }


def score_model(name, model, X_train, X_test, y_train, y_test, rows, best_parameters=""):
    model.fit(X_train, y_train)
    rows.append({
        "model": name,
        **regression_scores(y_train, model.predict(X_train), y_test, model.predict(X_test)),
        "best_parameters": best_parameters,
    })
    return model


def compare_polynomial_degrees(X_train, X_test, y_train, y_test, cv):
    """Train, 5-fold CV, and test MSE for each degree; CV (not test) MSE picks the degree."""
    rows = []
    for degree in DEGREES:
        model = polynomial_pipeline(degree, LinearRegression())
        cv_mse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        model.fit(X_train, y_train)
        rows.append({
            "degree": degree,
            "n_features": model.named_steps["polynomial"].n_output_features_,
            "cv_mse": cv_mse.mean(),
            **regression_scores(y_train, model.predict(X_train), y_test, model.predict(X_test)),
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "polynomial_degree_comparison.csv", index=False)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(results["degree"], results["train_mse"], marker="o", label="Train MSE")
    axis.plot(results["degree"], results["cv_mse"], marker="o", label="5-fold CV MSE")
    axis.plot(results["degree"], results["test_mse"], marker="o", label="Test MSE")
    axis.set_yscale("log")
    axis.set(
        title="Polynomial regression: underfitting and overfitting",
        xlabel="Polynomial degree (all six features)",
        ylabel="MSE (log scale)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "polynomial_fit_complexity.png", dpi=160)
    plt.close(figure)
    return int(results.loc[results["cv_mse"].idxmin(), "degree"]), results


def plot_single_feature_fits(X_train, X_test, y_train, y_test):
    """Price vs MRT distance with an underfitting, a good, and an overfitting degree."""
    cases = [(1, "Underfitting"), (3, "Good fit"), (10, "Overfitting")]
    x_train, x_test = X_train[[DEMO_FEATURE]], X_test[[DEMO_FEATURE]]
    grid = pd.DataFrame({DEMO_FEATURE: np.linspace(x_train.min().iloc[0], x_train.max().iloc[0], 500)})
    figure, axes = plt.subplots(1, len(cases), figsize=(15, 4.5), sharey=True)
    for axis, (degree, label) in zip(axes, cases):
        model = polynomial_pipeline(degree, LinearRegression()).fit(x_train, y_train)
        train_mse = mean_squared_error(y_train, model.predict(x_train))
        test_mse = mean_squared_error(y_test, model.predict(x_test))
        axis.scatter(x_train, y_train, s=12, alpha=0.5, label="Train")
        axis.scatter(x_test, y_test, s=12, alpha=0.7, label="Test")
        axis.plot(grid, model.predict(grid), color="black", linewidth=2, label=f"Degree {degree}")
        axis.set(
            title=f"{label}: degree {degree}\ntrain MSE={train_mse:.1f}, test MSE={test_mse:.1f}",
            xlabel="Distance to nearest MRT station (m)",
            ylim=(0, y_train.max() * 1.1),
        )
        axis.legend()
    axes[0].set_ylabel("House price per unit area")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "single_feature_fits.png", dpi=160)
    plt.close(figure)


def plot_regularization_paths(searches):
    figure, axes = plt.subplots(1, len(searches), figsize=(12, 4.5))
    for axis, (name, search) in zip(axes, searches.items()):
        alphas = search.cv_results_["param_model__alpha"].data.astype(float)
        cv_mse = -search.cv_results_["mean_test_score"]
        best_alpha = search.best_params_["model__alpha"]
        axis.plot(alphas, cv_mse, marker="o")
        axis.axvline(best_alpha, color="gray", linestyle="--", label=f"Best alpha = {best_alpha:.3g}")
        axis.set_xscale("log")
        axis.set(
            title=f"{name.capitalize()} (degree-{REGULARIZED_DEGREE} features)",
            xlabel="alpha (regularization strength)",
            ylabel="5-fold CV MSE",
        )
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "regularization_cv_curves.png", dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    gd_model = Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientDescentLinearRegression()),
    ])
    score_model("linear_gradient_descent", gd_model, X_train, X_test, y_train, y_test, rows)
    closed_form = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(X_train, y_train)
    gd_weights = gd_model.named_steps["model"].weights_
    exact_weights = np.concatenate([[closed_form.named_steps["model"].intercept_], closed_form.named_steps["model"].coef_])
    weight_gap = np.abs(gd_weights - exact_weights).max()

    best_degree, _ = compare_polynomial_degrees(X_train, X_test, y_train, y_test, cv)
    score_model(
        f"polynomial_degree_{best_degree}", polynomial_pipeline(best_degree, LinearRegression()),
        X_train, X_test, y_train, y_test, rows, "degree chosen by 5-fold CV",
    )
    score_model(
        f"polynomial_degree_{REGULARIZED_DEGREE}_unregularized",
        polynomial_pipeline(REGULARIZED_DEGREE, LinearRegression()),
        X_train, X_test, y_train, y_test, rows, "baseline for Ridge/Lasso",
    )

    searches = {}
    for name, estimator in (("ridge", Ridge()), ("lasso", Lasso(max_iter=100000))):
        search = GridSearchCV(
            polynomial_pipeline(REGULARIZED_DEGREE, estimator),
            {"model__alpha": ALPHA_GRIDS[name]},
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        score_model(f"{name}_degree_{REGULARIZED_DEGREE}", search, X_train, X_test, y_train, y_test, rows)
        coefficients = search.best_estimator_.named_steps["model"].coef_
        rows[-1]["best_parameters"] = (
            f"alpha={search.best_params_['model__alpha']:.4g}, "
            f"non-zero coefficients={np.count_nonzero(coefficients)}/{len(coefficients)}"
        )
        searches[name] = search

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    plot_regularization_paths(searches)
    plot_single_feature_fits(X_train, X_test, y_train, y_test)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(gd_model.named_steps["model"].loss_history_)
    axis.set_yscale("log")
    axis.set(title="Gradient-descent training loss", xlabel="Epoch", ylabel="MSE (log scale)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "gradient_descent_loss.png", dpi=160)
    plt.close(figure)

    print(results.to_string(index=False))
    print(f"\nMax |gradient descent - closed-form| weight difference: {weight_gap:.2e}")
    print(f"Polynomial degree selected by cross-validation: {best_degree}")


if __name__ == "__main__":
    main()
