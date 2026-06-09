import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge


class MoERegressor(BaseEstimator, RegressorMixin):
    def __init__(self, k_clusters=3, random_state=42):
        self.k_clusters = k_clusters
        self.random_state = random_state

        self.kmeans = KMeans(n_clusters=self.k_clusters, random_state=self.random_state)

        self.router = RandomForestClassifier(n_estimators=50, random_state=self.random_state)

        self.experts = [Ridge(alpha=1.0) for _ in range(self.k_clusters)]

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        cluster_labels = self.kmeans.fit_predict(X_arr)

        self.router.fit(X_arr, cluster_labels)

        for cluster_id in range(self.k_clusters):
            mask = (cluster_labels == cluster_id)
            X_subset = X_arr[mask]
            y_subset = y_arr[mask]

            if len(X_subset) > 0:
                self.experts[cluster_id].fit(X_subset, y_subset)
            else:
                self.experts[cluster_id].fit(X_arr, y_arr)

        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        predictions = np.zeros(X_arr.shape[0])

        predicted_clusters = self.router.predict(X_arr)

        for cluster_id in range(self.k_clusters):
            mask = (predicted_clusters == cluster_id)

            if np.any(mask):
                X_routed = X_arr[mask]
                predictions[mask] = self.experts[cluster_id].predict(X_routed)

        return predictions