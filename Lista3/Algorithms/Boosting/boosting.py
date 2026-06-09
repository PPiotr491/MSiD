import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor


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

        self.base_pred_ = np.mean(y_arr)

        current_predictions = np.full(shape=y_arr.shape, fill_value=self.base_pred_)

        self.trees_ = []

        for m in range(self.n_estimators):
            residuals = y_arr - current_predictions

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + m if self.random_state is not None else None
            )

            tree.fit(X_arr, residuals)

            current_predictions += self.learning_rate * tree.predict(X_arr)

            self.trees_.append(tree)

        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)

        predictions = np.full(shape=(X_arr.shape[0],), fill_value=self.base_pred_)

        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X_arr)

        return predictions