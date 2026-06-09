import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge


class MoERegressor(BaseEstimator, RegressorMixin):
    def __init__(self, k_clusters=3, random_state=42):
        self.k_clusters = k_clusters
        self.random_state = random_state

        # 1. Narzędzie do odkrycia "naturalnych grup" w danych
        self.kmeans = KMeans(n_clusters=self.k_clusters, random_state=self.random_state)

        # 2. Sieć-Bramka (Router), która uczy się na pamięć przyporządkowania do klastrów
        self.router = RandomForestClassifier(n_estimators=50, random_state=self.random_state)

        # 3. Zespół Ekspertów (użyjemy modelu Ridge z regularyzacją L2, żeby byli stabilni)
        # self.experts = [Ridge(alpha=1.0) for _ in range(self.k_clusters)]
        self.experts = [RandomForestRegressor(n_estimators=50, random_state=self.random_state) for _ in range(self.k_clusters)]

    def fit(self, X, y):
        # Konwersja do numpy (dla bezpieczeństwa indeksowania)
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        # ETAP A: Podział treningowego świata na K części (Klastrowanie)
        # KMeans przypisze każdemu wierszowi etykietę klastra (0, 1 lub 2)
        cluster_labels = self.kmeans.fit_predict(X_arr)

        # ETAP B: Trening Bramki (Routera)
        # Bramka uczy się, jak rozpoznać klaster na podstawie cech X
        self.router.fit(X_arr, cluster_labels)

        # ETAP C: Trening Ekspertów (Wąska specjalizacja)
        for cluster_id in range(self.k_clusters):
            # Wybieramy tylko te wiersze, które należą do danego klastra
            mask = (cluster_labels == cluster_id)
            X_subset = X_arr[mask]
            y_subset = y_arr[mask]

            # Jeśli w klastrze są jakieś dane, trenujemy przypisanego do niego eksperta
            if len(X_subset) > 0:
                self.experts[cluster_id].fit(X_subset, y_subset)
            else:
                # Ochrona na wypadek "pustego" klastra – ekspert uczy się na wszystkim
                self.experts[cluster_id].fit(X_arr, y_arr)

        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        predictions = np.zeros(X_arr.shape[0])

        # ETAP D: Wnioskowanie (Testowanie)
        # 1. Pytamy Bramkę: "Do którego eksperta wysłać każdą z próbek testowych?"
        predicted_clusters = self.router.predict(X_arr)

        # 2. Delegujemy pracę: każdy ekspert obsługuje tylko swoje wiersze
        for cluster_id in range(self.k_clusters):
            mask = (predicted_clusters == cluster_id)

            # Jeśli Bramka wysłała do tego eksperta jakiekolwiek próbki, niech je rozwiąże
            if np.any(mask):
                X_routed = X_arr[mask]
                predictions[mask] = self.experts[cluster_id].predict(X_routed)

        return predictions