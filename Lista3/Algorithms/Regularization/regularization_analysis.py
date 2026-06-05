import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from pandas import DataFrame

from Lista3.Algorithms.Regularization.regularized_gd import fit_regularized_gd_correct
from Lista3.Metrics.metrics import mse, mae, predict_linear

logger = logging.getLogger(__name__)

def _ensure_feature_names(n_features, feature_names=None):
    if feature_names is None:
        return [f"x{i+1}" for i in range(n_features)]
    return list(feature_names)


def fit_penalty_path(X, y, alphas, penalty, learning_rate=0.01, epochs=2000):
    alphas = list(alphas)
    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty must be 'l1' or 'l2'")

    n_features = X.shape[1]
    weights = np.zeros((len(alphas), n_features + 1))

    for i, alpha in enumerate(alphas):
        alpha_epochs = epochs
        if alpha > 0.1:
            alpha_epochs = int(epochs * 1.5)

        w, history = fit_regularized_gd_correct(
            X,
            y,
            learning_rate=learning_rate,
            alpha=alpha,
            epochs=alpha_epochs,
            penalty=penalty,
            include_reg_in_loss=False,
        )

        weights[i] = w

        if i % max(1, len(alphas) // 4) == 0:
            final_loss = history[-1] if history else np.nan
            logger.debug(f"  alpha={alpha:.6f}: final_mse={final_loss:.6f}, "
                  f"mean_|w|={np.mean(np.abs(w[1:])):.6f}, "
                  f"zero_count={np.sum(np.abs(w[1:]) <= 1e-4)}")

    return np.array(alphas, dtype=float), weights


def build_stats_table(alphas, weights, penalty, zero_tol=1e-6):
    coef = weights[:, 1:]
    abs_mean = np.mean(np.abs(coef), axis=1)
    zero_count = np.sum(np.abs(coef) <= zero_tol, axis=1)

    return pd.DataFrame({
        "alpha": alphas,
        "penalty": penalty,
        "mean_abs_weight": abs_mean,
        "zero_count": zero_count,
    })


def plot_weight_paths(alphas, weights, feature_names=None, max_features=10, include_bias=False):
    coef = weights[:, 1:] if not include_bias else weights
    n_features = coef.shape[1]
    names = _ensure_feature_names(n_features, feature_names)

    # Pick features with largest max |w| across the path for a clearer plot
    max_abs = np.max(np.abs(coef), axis=0)
    top_idx = np.argsort(max_abs)[:max_features]

    plt.figure(figsize=(10, 6))
    for idx in top_idx:
        plt.plot(alphas, coef[:, idx], label=names[idx])

    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("weight")
    plt.title("Weight paths")
    plt.legend(loc="best", fontsize=8)

    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Próg (y=0)')
    plt.tight_layout()


def compare_lasso_ridge(X, y, alphas, learning_rate=0.01, epochs=2000, zero_tol=1e-6, feature_names=None):
    l1_alphas, l1_weights = fit_penalty_path(
        X, y, alphas, penalty="l1", learning_rate=learning_rate, epochs=epochs
    )
    l2_alphas, l2_weights = fit_penalty_path(
        X, y, alphas, penalty="l2", learning_rate=learning_rate, epochs=epochs
    )

    lasso_stats = build_stats_table(l1_alphas, l1_weights, penalty="l1", zero_tol=zero_tol)
    ridge_stats = build_stats_table(l2_alphas, l2_weights, penalty="l2", zero_tol=zero_tol)

    # results = {
    #     "lasso": {"alphas": l1_alphas, "weights": l1_weights, "stats": lasso_stats},
    #     "ridge": {"alphas": l2_alphas, "weights": l2_weights, "stats": ridge_stats},
    # }

    stats_compare = lasso_stats[["alpha", "zero_count", "mean_abs_weight"]].rename(
        columns={
            "zero_count": "lasso_zero_count",
            "mean_abs_weight": "lasso_mean_abs_weight",
        }
    )
    stats_compare["ridge_zero_count"] = ridge_stats["zero_count"].values
    stats_compare["ridge_mean_abs_weight"] = ridge_stats["mean_abs_weight"].values


    # Convenience plots for notebooks
    plot_weight_paths(l1_alphas, l1_weights, feature_names=feature_names, max_features=6)
    plot_weight_paths(l2_alphas, l2_weights, feature_names=feature_names, max_features=6)

    # return results
    return stats_compare


def regularized_regression_results(X_train_s, X_test_s, y_train, y_test, alphas, learning_rate=0.001, epochs=2000):
    rows = []
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    for penalty in ["l1", "l2"]:
        for alpha in alphas:
            weights, _ = fit_regularized_gd_correct(
                X_train_s,
                y_train_arr,
                learning_rate=learning_rate,
                alpha=float(alpha),
                epochs=epochs,
                penalty=penalty,
            )
            train_pred = predict_linear(X_train_s, weights)
            test_pred = predict_linear(X_test_s, weights)
            rows.append({
                "penalty": penalty,
                "alpha": float(alpha),
                "train_mse": mse(y_train_arr, train_pred),
                "test_mse": mse(y_test_arr, test_pred),
                "train_mae": mae(y_train_arr, train_pred),
                "test_mae": mae(y_test_arr, test_pred),
                "zero_count": int(np.sum(np.abs(weights[1:]) <= 1e-3)),
                "mean_abs_weight": float(np.mean(np.abs(weights[1:]))),
            })
    return pd.DataFrame(rows)


def plot_comparison_results(regression_results: DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for penalty, label in [("l1", "Lasso"), ("l2", "Ridge")]:
        subset = regression_results[regression_results["penalty"] == penalty].sort_values("alpha")
        axes[0].plot(subset["alpha"], subset["train_mse"], marker="o", label=f"{label} train")
        axes[0].plot(subset["alpha"], subset["test_mse"], marker="o", linestyle="--", label=f"{label} test")
        axes[1].plot(subset["alpha"], subset["zero_count"], marker="o", label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Train/Test MSE vs alpha")
    axes[0].legend(loc="best")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("number of ~zero weights")
    axes[1].set_title("Lasso zeroes weights, Ridge shrinks them")
    axes[1].legend(loc="best")
    plt.tight_layout()