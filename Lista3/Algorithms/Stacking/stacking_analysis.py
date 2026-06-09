import numpy as np
import matplotlib.pyplot as plt
from Lista3.Metrics.metrics import mse, mae



def evaluate_and_plot_models(models_dict, X_train, y_train, X_test, y_test):
    names = []
    train_mses, test_mses = [], []
    train_maes, test_maes = [], []

    print(f"{'Nazwa Modelu':<40} | {'Train MSE':<10} | {'Test MSE':<10} | {'Train MAE':<10} | {'Test MAE':<10}")
    print("-" * 90)

    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        mse_train = mse(y_train, train_preds)
        mse_test = mse(y_test, test_preds)
        mae_train = mae(y_train, train_preds)
        mae_test = mae(y_test, test_preds)

        print(f"{name:<40} | {mse_train:<10.4f} | {mse_test:<10.4f} | {mae_train:<10.4f} | {mae_test:<10.4f}")

        names.append(name)
        train_mses.append(mse_train)
        test_mses.append(mse_test)
        train_maes.append(mae_train)
        test_maes.append(mae_test)

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    rects1_mse = axes[0].bar(x - width / 2, train_mses, width, label='Train MSE', color='steelblue', alpha=0.85)
    rects2_mse = axes[0].bar(x + width / 2, test_mses, width, label='Test MSE', color='coral', alpha=0.85)
    axes[0].set_title('Porównanie modeli: Błąd MSE', fontsize=14)
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    rects1_mae = axes[1].bar(x - width / 2, train_maes, width, label='Train MAE', color='steelblue', alpha=0.85)
    rects2_mae = axes[1].bar(x + width / 2, test_maes, width, label='Test MAE', color='coral', alpha=0.85)
    axes[1].set_title('Porównanie modeli: Błąd MAE', fontsize=14)
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            offset = 3 if height >= 0 else -12
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    autolabel(rects1_mse, axes[0])
    autolabel(rects2_mse, axes[0])
    autolabel(rects1_mae, axes[1])
    autolabel(rects2_mae, axes[1])

    plt.tight_layout()
    plt.show()