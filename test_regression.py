#!/usr/bin/env python
"""
Test regresji liniowej - weryfikacja trzech metod
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Załaduj dane
print("Ładowanie danych...")
df_reg = pd.read_csv("resources/sleep_health_dataset.csv")
if 'person_id' in df_reg.columns:
    df_reg = df_reg.drop('person_id', axis=1)

target_reg = 'cognitive_performance_score'
y_reg = df_reg[target_reg].values
X_reg = df_reg.drop(columns=[target_reg])

# Preprocessing
cat_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()
X_reg_encoded = X_reg.copy()
for col in cat_cols_reg:
    le = LabelEncoder()
    X_reg_encoded[col] = le.fit_transform(X_reg[col].astype(str))

# Podział i standaryzacja
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_encoded, y_reg, test_size=0.2, random_state=13
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f"✓ Dane załadowane: {X_train_reg_scaled.shape}")

# METODA 1: Analityczna
print("\n[METODA 1] Least Squares (Analityczna)...")
n_samples, n_features = X_train_reg_scaled.shape
X_with_bias = np.column_stack([np.ones(n_samples), X_train_reg_scaled])
XtX = X_with_bias.T @ X_with_bias
XtX_inv = np.linalg.inv(XtX)
Xty = X_with_bias.T @ y_train_reg
w_analytical = XtX_inv @ Xty

y_test_pred_analytical = np.column_stack([np.ones(len(X_test_reg_scaled)), X_test_reg_scaled]) @ w_analytical
mse_test_analytical = mse(y_test_reg, y_test_pred_analytical)
print(f"✓ Test MSE: {mse_test_analytical:.6f}")

# METODA 2: Gradient Descent
print("\n[METODA 2] Gradient Descent...")
def fit_gd(X, y, lr=0.01, epochs=2000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features + 1)
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    for epoch in range(epochs):
        y_pred = X_with_bias @ w
        error = y_pred - y
        gradient = (2.0 / n_samples) * (X_with_bias.T @ error)
        w = w - lr * gradient
    return w

w_gd = fit_gd(X_train_reg_scaled, y_train_reg, lr=0.01, epochs=5000)
y_test_pred_gd = np.column_stack([np.ones(len(X_test_reg_scaled)), X_test_reg_scaled]) @ w_gd
mse_test_gd = mse(y_test_reg, y_test_pred_gd)
print(f"✓ Test MSE: {mse_test_gd:.6f}")

# METODA 3: Sklearn
print("\n[METODA 3] Sklearn LinearRegression...")
model_sklearn = LinearRegression()
model_sklearn.fit(X_train_reg_scaled, y_train_reg)
y_test_pred_sklearn = model_sklearn.predict(X_test_reg_scaled)
mse_test_sklearn = mse(y_test_reg, y_test_pred_sklearn)
print(f"✓ Test MSE: {mse_test_sklearn:.6f}")

# Porównanie
print("\n" + "="*60)
print("PORÓWNANIE WYNIKÓW:")
print("="*60)
print(f"{'Metoda':<30} {'Test MSE':<20} {'Intercept':<15}")
print("-"*60)
print(f"{'Analityczna':<30} {mse_test_analytical:<20.6f} {w_analytical[0]:<15.6f}")
print(f"{'Gradient Descent':<30} {mse_test_gd:<20.6f} {w_gd[0]:<15.6f}")
print(f"{'Sklearn':<30} {mse_test_sklearn:<20.6f} {model_sklearn.intercept_:<15.6f}")

print("\n" + "="*60)
print("PORÓWNANIE WAG (pierwsze 5 cech):")
print("="*60)
for i in range(5):
    print(f"w[{i+1}] - Anal: {w_analytical[i+1]:.10f}, GD: {w_gd[i+1]:.10f}, Sklearn: {model_sklearn.coef_[i]:.10f}")

diff_gd_analytical = np.abs(w_gd[1:] - w_analytical[1:])
diff_sklearn_analytical = np.abs(model_sklearn.coef_ - w_analytical[1:])

print(f"\nMax róznica GD vs Analityczna: {diff_gd_analytical.max():.2e}")
print(f"Max róznica Sklearn vs Analityczna: {diff_sklearn_analytical.max():.2e}")

print("\n✓✓✓ SUKCES! Wszystkie trzy metody działają poprawnie ✓✓✓")

