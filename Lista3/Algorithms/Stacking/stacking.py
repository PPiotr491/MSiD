import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict

class StackingModel(BaseEstimator):

    def __init__(self, base_models, meta_model, cv=5, method='predict'):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.method = method

    def fit(self, X, y):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)

        # Tablica przechowująca nowe cechy dla meta-modelu (predykcje modeli Level-0)
        meta_features = np.zeros((n_samples, len(self.base_models)))

        for i, model in enumerate(self.base_models):
            meta_features[:, i] = cross_val_predict(
                clone(model), X, y, cv=self.cv, method=self.method
            )
            # 2. Po wygenerowaniu rzetelnych cech, ostatecznie trenujemy bazowy model na wszystkich danych
            model.fit(X, y)

        # 3. Uczymy "szefa" (Level-1), komu z Level-0 warto ufać przy błędnych przewidywaniach
        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        meta_features = np.zeros((n_samples, len(self.base_models)))

        # 1. Pobranie predykcji od wytrenowanych modeli bazowych na nowych danych
        for i, model in enumerate(self.base_models):
            if self.method == 'predict_proba':
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)

        # 2. Ostateczna decyzja na podstawie modelu na Poziomie 1
        return self.meta_model.predict(meta_features)