import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_and_plot_moe_vs_global(global_model, moe_model, X_train, y_train, X_test, y_test):
    y_train_pred_global = global_model.predict(X_train)
    y_test_pred_global = global_model.predict(X_test)

    y_train_pred_moe = moe_model.predict(X_train)
    y_test_pred_moe = moe_model.predict(X_test)

    mse_train_gl = mean_squared_error(y_train, y_train_pred_global)
    mse_test_gl = mean_squared_error(y_test, y_test_pred_global)
    r2_test_gl = r2_score(y_test, y_test_pred_global)

    mse_train_moe = mean_squared_error(y_train, y_train_pred_moe)
    mse_test_moe = mean_squared_error(y_test, y_test_pred_moe)
    r2_test_moe = r2_score(y_test, y_test_pred_moe)

    print(f"{'Architektura':<25} | {'Train MSE':<10} | {'Test MSE':<10} | {'Test R2':<10}")
    print("-" * 75)
    print(f"{'Globalny Ekspert':<25} | {mse_train_gl:<10.4f} | {mse_test_gl:<10.4f} | {r2_test_gl:<10.4f}")
    print(f"{'Mixture of Experts':<25} | {mse_train_moe:<10.4f} | {mse_test_moe:<10.4f} | {r2_test_moe:<10.4f}")

    labels = ['Globalny Ekspert', 'Mixture of Experts']
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    rects1_mse = axes[0].bar(x - width / 2, [mse_train_gl, mse_train_moe], width, label='Train MSE', color='steelblue',
                             alpha=0.85)
    rects2_mse = axes[0].bar(x + width / 2, [mse_test_gl, mse_test_moe], width, label='Test MSE', color='coral',
                             alpha=0.85)
    axes[0].set_title('Porównanie błędów MSE', fontsize=13, pad=10)
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    rects_r2 = axes[1].bar(labels, [r2_test_gl, r2_test_moe], width=0.4, color='mediumseagreen', alpha=0.85)
    axes[1].set_title('Współczynnik determinacji $R^2$ na zbiorze testowym', fontsize=13,
                      pad=10)
    axes[1].set_ylabel('$R^2$ Score')
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    axes[1].set_ylim([0, max(r2_test_gl, r2_test_moe) * 1.2])

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            offset = 3 if height >= 0 else -12
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=9, weight='bold')

    autolabel(rects1_mse, axes[0])
    autolabel(rects2_mse, axes[0])
    autolabel(rects_r2, axes[1])

    plt.tight_layout()
    plt.show()