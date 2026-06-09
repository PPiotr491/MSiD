import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


class SimpleGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.base_pred_ = None
        self.trees_ = []

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        # Krok 1: Inicjalizacja bazowa (przewidywanie średniej wartości z całego zbioru Y)
        self.base_pred_ = np.mean(y_arr)

        # Inicjalizujemy wektor obecnych predykcji wartością średniej
        current_predictions = np.full(shape=y_arr.shape, fill_value=self.base_pred_)

        self.trees_ = []

        # Krok 2: Sekwencyjne budowanie M płytkich drzew (n_estimators)
        for m in range(self.n_estimators):
            # Krok A: Obliczanie wartości resztkowych (residuals)
            residuals = y_arr - current_predictions

            # Krok B: Tworzenie nowego, płytkiego drzewa (max_depth=3)
            # Używamy zmiennego random_state dla każdego drzewa, aby zachować powtarzalność
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + m if self.random_state is not None else None
            )

            # Trenujemy drzewo na oryginalnych cechach X, ale TARGETEM są obliczone reszty!
            tree.fit(X_arr, residuals)

            # Krok C: Aktualizacja głównego modelu o predykcje nowego drzewa z learning_rate
            current_predictions += self.learning_rate * tree.predict(X_arr)

            # Zapisujemy wytrenowane drzewo w komitecie
            self.trees_.append(tree)

        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)

        # Rozpoczynamy predykcję od zapamiętanej wartości średniej
        predictions = np.full(shape=(X_arr.shape[0],), fill_value=self.base_pred_)

        # Dodajemy osłabione przez learning_rate wkłady od każdego drzewa z komitetu
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X_arr)

        return predictions