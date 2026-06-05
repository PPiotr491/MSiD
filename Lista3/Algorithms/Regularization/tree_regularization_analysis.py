import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


TREE_METRICS = {
    "accuracy": accuracy_score,
    "precision_weighted": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "recall_weighted": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
    "f1_weighted": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
}


def fit_tree_classifier(X_train, y_train, **tree_kwargs):
    model = DecisionTreeClassifier(**tree_kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_tree_classifier(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision_weighted": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_precision_weighted": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_recall_weighted": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_recall_weighted": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_f1_weighted": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_f1_weighted": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "train_pred": y_train_pred,
        "test_pred": y_test_pred,
        "depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
    }


def _stringify_param(value):
    if value is None:
        return "None"
    return str(value)


def sweep_tree_parameter(
    X_train,
    y_train,
    X_test,
    y_test,
    param_name,
    param_values,
    base_params=None,
):
    base_params = dict(base_params or {})
    rows = []

    for value in param_values:
        params = dict(base_params)
        params[param_name] = value
        model = fit_tree_classifier(X_train, y_train, **params)
        metrics = evaluate_tree_classifier(model, X_train, y_train, X_test, y_test)
        rows.append(
            {
                "param_name": param_name,
                "param_value": value,
                "param_label": _stringify_param(value),
                **{k: v for k, v in metrics.items() if not k.endswith("_pred")},
            }
        )

    return pd.DataFrame(rows)


def find_best_row(results_df, metric="test_f1_weighted"):
    if results_df.empty:
        raise ValueError("results_df must not be empty")
    return results_df.sort_values(metric, ascending=False).iloc[0]


def plot_train_test_curves(results_df, metric="f1_weighted", param_label="parameter"):
    if results_df.empty:
        raise ValueError("results_df must not be empty")

    x = np.arange(len(results_df))
    labels = results_df["param_label"].tolist()

    plt.figure(figsize=(10, 5))
    plt.plot(x, results_df[f"train_{metric}"].values, marker="o", label=f"train_{metric}")
    plt.plot(x, results_df[f"test_{metric}"].values, marker="o", label=f"test_{metric}")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel(param_label)
    plt.ylabel(metric)
    plt.title(f"Tree regularization: {metric} vs {param_label}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()


def summarize_tree_sweep(results_df, metric="test_f1_weighted"):
    best_row = find_best_row(results_df, metric=metric)
    summary = results_df[["param_label", "train_accuracy", "test_accuracy", "train_f1_weighted", "test_f1_weighted", "depth", "n_leaves"]].copy()
    summary = summary.rename(columns={"param_label": "setting"})
    return summary, best_row

