import numpy as np

def predict_linear(X, w):
    n_samples = X.shape[0]
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    return X_with_bias @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))