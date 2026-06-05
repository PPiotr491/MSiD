import logging
import numpy as np

from Lista3.Metrics.metrics import mse

logger = logging.getLogger(__name__)

def fit_regularized_gd(X, y, learning_rate=0.01, alpha = 1.0, epochs=2000, lasso=True):
    n_samples, n_features = X.shape

    # Inicjalizuj wagi (bias + coefficients)
    w = np.zeros(n_features + 1)

    # Dodaj bias do X
    X_with_bias = np.column_stack([np.ones(n_samples), X])

    history_loss = []

    for epoch in range(epochs):
        y_pred = X_with_bias @ w

        error = y_pred - y

        gradient = (2.0 / n_samples) * (X_with_bias.T @ error)

        reg = np.zeros_like(w)
        if lasso:
            reg[1:] = np.sign(w[1:])
            # gradient += alpha * np.sign(w)
        else:
            # gradient += alpha * 2 * w
            reg[1:] = 2 * w[1:]
        gradient += alpha * reg

        # Aktualizacja wag
        w = w - learning_rate * gradient

        # Śledź MSE
        loss = mse(y, y_pred)
        history_loss.append(loss)

        if (epoch + 1) % 100 == 0:
            logger.debug(f"Epoch {epoch + 1}: MSE = {loss:.6f}")

    return w, history_loss


def fit_regularized_gd_correct(X, y, learning_rate=0.01, alpha=1.0, epochs=2000, penalty="l2", include_reg_in_loss=False, w_init=None):
    n_samples, n_features = X.shape

    w = np.zeros(n_features + 1)

    # Dodaj bias do X
    X_with_bias = np.column_stack([np.ones(n_samples), X])

    history_loss = []
    use_reg = penalty in {"l1", "l2"} and alpha != 0.0

    for epoch in range(epochs):
        # Predykcja
        y_pred = X_with_bias @ w

        # Błąd
        error = y_pred - y

        # Gradient MSE
        gradient = (2.0 / n_samples) * (X_with_bias.T @ error)

        # Regularizacja bez biasu (w[0])
        if use_reg:
            reg = np.zeros_like(w)
            if penalty == "l1":
                reg[1:] = np.sign(w[1:])
            else:
                reg[1:] = 2 * w[1:]
            gradient += alpha * reg

        # Aktualizacja wag
        w = w - learning_rate * gradient

        # Śledź MSE lub pełną funkcję celu
        loss = mse(y, y_pred)
        if include_reg_in_loss and use_reg:
            if penalty == "l1":
                loss += alpha * np.sum(np.abs(w[1:]))
            else:
                loss += alpha * np.sum(w[1:] ** 2)
        history_loss.append(loss)

        if (epoch + 1) % 100 == 0:
            logger.debug(f"Epoch {epoch + 1}: MSE = {loss:.6f}")

    return w, history_loss
