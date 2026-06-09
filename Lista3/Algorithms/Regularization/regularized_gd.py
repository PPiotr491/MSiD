import logging
import numpy as np
from sklearn.base import BaseEstimator

from Lista3.Metrics.metrics import mse

logger = logging.getLogger(__name__)

class RegularizedGD(BaseEstimator):
    def __init__(self, learning_rate=0.005, alpha=0.1, epochs=3000, penalty="l2", is_classifier=False, threshold=0.5):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.penalty = penalty
        self.is_classifier = is_classifier
        self.threshold = threshold
        self.w = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        w = np.zeros(n_features + 1)

        # Dodaj bias do X
        X_with_bias = np.column_stack([np.ones(n_samples), X])

        use_reg = self.penalty in {"l1", "l2"} and self.alpha != 0.0

        for epoch in range(self.epochs):
            # Predykcja
            y_pred = X_with_bias @ w

            # Błąd
            error = y_pred - y

            # Gradient MSE
            gradient = (2.0 / n_samples) * (X_with_bias.T @ error)

            # Regularizacja bez biasu (w[0])
            if use_reg:
                reg = np.zeros_like(w)
                if self.penalty == "l1":
                    reg[1:] = np.sign(w[1:])
                else:
                    reg[1:] = 2 * w[1:]
                gradient += self.alpha * reg

            # Aktualizacja wag
            w = w - self.learning_rate * gradient

            # Śledź MSE lub pełną funkcję celu
            loss = mse(y, y_pred)
            if use_reg:
                if self.penalty == "l1":
                    loss += self.alpha * np.sum(np.abs(w[1:]))
                else:
                    loss += self.alpha * np.sum(w[1:] ** 2)

            if (epoch + 1) % 100 == 0:
                logger.debug(f"Epoch {epoch + 1}: MSE = {loss:.6f}")

        self.w = w
        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        n_samples = X_arr.shape[0]
        X_with_bias = np.column_stack([np.ones(n_samples), X_arr])

        predictions = X_with_bias @ self.w

        if self.is_classifier:
            return (predictions >= self.threshold).astype(int)
        return predictions
