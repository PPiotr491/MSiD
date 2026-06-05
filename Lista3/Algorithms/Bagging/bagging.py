import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=None, criterion='gini',
                 bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.estimators_ = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                y_sample = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
                # y_sample = y[indices] if isinstance(y, np.ndarray) else y.iloc[indices]
            else:
                X_sample, y_sample = X, y

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state if self.random_state is None else self.random_state + i
            )
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        if not self.estimators_:
            raise ValueError("Model musi być wytrenowany przed predykcją!")

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_estimators), dtype=int)

        # Zbierz predykcje ze wszystkich drzew
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)

        # Hard voting: wyznacz mode (najczęstszą klasę)
        y_pred = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Режим (najczęstsza klasa)
            y_pred[i] = np.bincount(predictions[i]).argmax()

        return y_pred

    def predict_proba(self, X):
        if not self.estimators_:
            raise ValueError("Model musi być wytrenowany przed predykcją!")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))

        # Zbierz prawdopodobieństwa ze wszystkich drzew
        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)
            probas += tree_proba

        # Uśrednij (soft voting)
        probas /= self.n_estimators

        return probas

    def get_feature_importances(self):
        if not self.estimators_:
            raise ValueError("Model musi być wytrenowany!")

        importances = np.zeros(self.estimators_[0].n_features_in_)
        for tree in self.estimators_:
            importances += tree.feature_importances_

        importances /= self.n_estimators
        return importances

