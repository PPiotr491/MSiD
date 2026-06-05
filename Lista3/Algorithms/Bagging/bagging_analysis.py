import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compare_bagging_variants(X_train, y_train, X_test, y_test, bagging_class,
                               n_estimators_list=None, max_depths=None):
    if n_estimators_list is None:
        n_estimators_list = [1, 5, 10, 20, 50]

    if max_depths is None:
        max_depths = [None]

    results = []

    for max_depth in max_depths:
        for n_est in n_estimators_list:
            model = bagging_class(
                n_estimators=n_est,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Predykcje
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metryki
            results.append({
                'n_estimators': n_est,
                'max_depth': max_depth,
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'test_precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'test_recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'test_f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
            })

    return pd.DataFrame(results)


def compare_single_vs_bagging(X_train, y_train, X_test, y_test, single_model,
                               bagging_model):
    single_train_pred = single_model.predict(X_train)
    single_test_pred = single_model.predict(X_test)

    bagging_train_pred = bagging_model.predict(X_train)
    bagging_test_pred = bagging_model.predict(X_test)

    comparison = {
        'Model': ['Single Tree', 'Bagging'],
        'Train Accuracy': [
            accuracy_score(y_train, single_train_pred),
            accuracy_score(y_train, bagging_train_pred)
        ],
        'Test Accuracy': [
            accuracy_score(y_test, single_test_pred),
            accuracy_score(y_test, bagging_test_pred)
        ],
        'Train F1': [
            f1_score(y_train, single_train_pred, average='weighted', zero_division=0),
            f1_score(y_train, bagging_train_pred, average='weighted', zero_division=0)
        ],
        'Test F1': [
            f1_score(y_test, single_test_pred, average='weighted', zero_division=0),
            f1_score(y_test, bagging_test_pred, average='weighted', zero_division=0)
        ],
        'Overfitting (Acc diff)': [
            accuracy_score(y_train, single_train_pred) - accuracy_score(y_test, single_test_pred),
            accuracy_score(y_train, bagging_train_pred) - accuracy_score(y_test, bagging_test_pred)
        ]
    }

    return pd.DataFrame(comparison)


def plot_n_estimators_impact(results_df, metric='test_accuracy', figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    # Jeśli mamy różne max_depths, wykreśl osobne linie
    max_depths = results_df['max_depth'].unique()

    for max_depth in sorted(max_depths, key=lambda x: (x is None, x)):
        subset = results_df[results_df['max_depth'] == max_depth]
        label = f"max_depth={max_depth}" if max_depth is not None else "max_depth=None (unlimited)"
        ax.plot(subset['n_estimators'], subset[metric], marker='o', label=label, linewidth=2)

    ax.set_xlabel('Number of Estimators', fontsize=11)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f'Impact of Ensemble Size on {metric.replace("_", " ").title()}', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_train_test_comparison(results_df, metric='accuracy', figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    train_col = f'train_{metric}'
    test_col = f'test_{metric}'

    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width/2, results_df[train_col], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df[test_col], width, label='Test', alpha=0.8)

    ax.set_xlabel('Model Configuration', fontsize=11)
    ax.set_ylabel(metric.title(), fontsize=11)
    ax.set_title(f'Train vs Test {metric.title()} - Bagging Variants', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n_est}" for n_est in results_df['n_estimators']])
    ax.legend(loc='best')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


def plot_comparison_bars(comparison_df, figsize=(10, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Dokładność
    metrics_acc = ['Train Accuracy', 'Test Accuracy']
    x = np.arange(len(metrics_acc))
    width = 0.35

    for i, model in enumerate(comparison_df['Model'].values):
        values = [comparison_df.loc[comparison_df['Model'] == model, m].values[0] for m in metrics_acc]
        axes[0].bar(i, values[0], width=0.4, alpha=0.8, label=f'{model} (Train)' if i == 0 else f'{model} (Train)')
        axes[0].bar(i + width, values[1], width=0.4, alpha=0.8, label=f'{model} (Test)' if i == 0 else f'{model} (Test)')

    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Accuracy Comparison: Single Tree vs Bagging', fontsize=12)
    n_models = len(comparison_df['Model'].values)
    center_positions = np.arange(n_models) + width / 2  # Środek między grupami barów
    axes[0].set_xticks(center_positions)
    axes[0].set_xticklabels(comparison_df['Model'].values)
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Overfitting
    x = np.arange(len(comparison_df))
    axes[1].bar(x, comparison_df['Overfitting (gap)'], alpha=0.8, color=['steelblue', 'coral'])
    axes[1].set_ylabel('Train - Test Accuracy (Overfitting)', fontsize=11)
    axes[1].set_title('Overfitting Gap', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_df['Model'].values)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    return fig


def plot_feature_importances(single_model, bagging_model, feature_names, figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    single_imp = single_model.feature_importances_
    if hasattr(bagging_model, 'get_feature_importances'):
        bagging_imp = bagging_model.get_feature_importances()
    else:
        bagging_imp = np.array([tree.feature_importances_ for tree in bagging_model.estimators_]).mean(axis=0)

    # Sort by single tree importances
    sorted_idx = np.argsort(single_imp)[::-1][:10]  # Top 10

    x = np.arange(len(sorted_idx))
    width = 0.35

    axes[0].barh(x - width/2, single_imp[sorted_idx], width, label='Single Tree', alpha=0.8)
    axes[0].barh(x + width/2, bagging_imp[sorted_idx], width, label='Bagging', alpha=0.8)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
    axes[0].set_xlabel('Feature Importance', fontsize=11)
    axes[0].set_title('Top 10 Feature Importances', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, axis='x', alpha=0.3)

    # Różnica w ważności
    diff = bagging_imp - single_imp
    sorted_diff_idx = np.argsort(np.abs(diff))[::-1][:10]

    x = np.arange(len(sorted_diff_idx))
    colors = ['coral' if d > 0 else 'steelblue' for d in diff[sorted_diff_idx]]
    axes[1].barh(x, diff[sorted_diff_idx], color=colors, alpha=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([feature_names[i] for i in sorted_diff_idx], fontsize=9)
    axes[1].set_xlabel('Importance Difference (Bagging - Single)', fontsize=11)
    axes[1].set_title('Top 10 Importance Differences', fontsize=12)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    return fig

