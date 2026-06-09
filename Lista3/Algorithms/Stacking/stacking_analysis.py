import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def plot_train_test_metrics(y_train, y_train_pred, y_test, y_test_pred, model_name="Stacking Model"):
    """
    Rysuje wykres słupkowy porównujący Accuracy i F1 Score na zbiorze treningowym i testowym.
    """
    # Obliczanie metryk
    metrics = {
        'Accuracy': (
            accuracy_score(y_train, y_train_pred),
            accuracy_score(y_test, y_test_pred)
        ),
        'F1 Weighted': (
            f1_score(y_train, y_train_pred, average='weighted'),
            f1_score(y_test, y_test_pred, average='weighted')
        )
    }

    labels = list(metrics.keys())
    train_scores = [metrics[label][0] for label in labels]
    test_scores = [metrics[label][1] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, train_scores, width, label='Train', color='steelblue', alpha=0.8)
    rects2 = ax.bar(x + width / 2, test_scores, width, label='Test', color='coral', alpha=0.8)

    ax.set_ylabel('Wartość metryki')
    ax.set_title(f'Wyniki modelu: {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.0, 1.1])
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    # Dodanie etykiet tekstowych na słupkach
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 punkty przesunięcia w pionie
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


def plot_train_test_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, classes=None):
    """
    Rysuje macierze pomyłek dla zbioru treningowego i testowego obok siebie.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Macierz pomyłek - Zbiór Treningowy
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes, cbar=False)
    axes[0].set_title('Confusion Matrix - Zbiór Treningowy')
    axes[0].set_ylabel('Prawdziwa etykieta')
    axes[0].set_xlabel('Przewidziana etykieta')

    # Macierz pomyłek - Zbiór Testowy
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=classes, yticklabels=classes, cbar=False)
    axes[1].set_title('Confusion Matrix - Zbiór Testowy')
    axes[1].set_ylabel('Prawdziwa etykieta')
    axes[1].set_xlabel('Przewidziana etykieta')

    plt.tight_layout()
    plt.show()