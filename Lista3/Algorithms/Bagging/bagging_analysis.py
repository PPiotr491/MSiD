import numpy as np
import matplotlib.pyplot as plt


def plot_comparison_bars(comparison_df, figsize=(10, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

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
    center_positions = np.arange(n_models) + width / 2
    axes[0].set_xticks(center_positions)
    axes[0].set_xticklabels(comparison_df['Model'].values)
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, axis='y', alpha=0.3)

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

