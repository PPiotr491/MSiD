#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decision Tree Classification Model - Sleep Health Dataset
Stwórz model decyzyjny wykorzystujący DecisionTreeClassifier ze scikit-learn
Model wytrenowany na zbiorze treningowym z preprocessingiem i metrykami.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

plt.style.use("seaborn-v0_8-darkgrid")
np.random.seed(42)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ============================================================================
# WCZYTANIE I EKSPLORACJA DANYCH
# ============================================================================
print("=" * 70)
print("WCZYTANIE I EKSPLORACJA DANYCH")
print("=" * 70)

file_path = "../resources/sleep_health_dataset.csv"
df = pd.read_csv(file_path)

print(f"\nRozmiar całego zbioru danych: {df.shape[0]} wierszy x {df.shape[1]} kolumn")
print(f"\nKolumny: {df.columns.tolist()}")
print(f"\nTypy danych:")
print(df.dtypes)
print(f"\nBrakujące wartości:")
print(df.isnull().sum())
print(f"\nRozkład zmiennej celu 'felt_rested':")
print(df['felt_rested'].value_counts())

# ============================================================================
# PREPROCESSING DANYCH
# ============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING DANYCH")
print("=" * 70)

# Usunięcie kolumny person_id jeśli istnieje
if 'person_id' in df.columns:
    df = df.drop('person_id', axis=1)

# Identyfikacja zmiennej celu i cech
target_col = 'felt_rested'
y = df[target_col]
X = df.drop(columns=[target_col])

print(f"\nZmienne niezależne (cechy): {X.columns.tolist()}")
print(f"\nZmienna zależna (target): {target_col}")
print(f"Klasy: {sorted(y.unique())}")
print(f"\nRozkład klas:")
print(y.value_counts())
print(f"\nProcent klas:")
print(y.value_counts(normalize=True) * 100)

# Preprocessing: obsługa zmiennych kategorycznych
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nZmienne kategoryczne: {categorical_cols}")
print(f"Zmienne numeryczne: {numerical_cols}")

# Kodowanie zmiennych kategorycznych
label_encoders = {}
X_processed = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"\nKodowanie {col}:")
    for i, class_name in enumerate(le.classes_):
        print(f"  {class_name} -> {i}")

# Kodowanie zmiennej celu jeśli jest kategoryczna
if y.dtype == 'object':
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"\nKodowanie zmiennej celu {target_col}:")
    for i, class_name in enumerate(le_target.classes_):
        print(f"  {class_name} -> {i}")
else:
    y_encoded = y
    le_target = None

# ============================================================================
# PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
# ============================================================================
print("\n" + "=" * 70)
print("PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nRozmiar zbioru treningowego: {X_train.shape[0]} wierszy x {X_train.shape[1]} kolumn")
print(f"Rozmiar zbioru testowego: {X_test.shape[0]} wierszy x {X_test.shape[1]} kolumn")
print(f"\nRozkład klas w zbiorze treningowym:")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nRozkład klas w zbiorze testowym:")
print(pd.Series(y_test).value_counts().sort_index())

# ============================================================================
# TRENOWANIE MODELU DECISION TREE CLASSIFIER
# ============================================================================
print("\n" + "=" * 70)
print("TRENOWANIE MODELU DECISION TREE CLASSIFIER")
print("=" * 70)

dt_classifier = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42,
    min_samples_split=10
)
dt_classifier.fit(X_train, y_train)

print("\nModel Decision Tree Classifier wytrenowany pomyslnie!")
print(f"Glebokosc drzewa: {dt_classifier.get_depth()}")
print(f"Liczba lisci: {dt_classifier.get_n_leaves()}")

# Feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nWaznosc cech (Feature Importances):")
print(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['feature'], feature_importances['importance'], color='steelblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances - Decision Tree Classifier')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()
print("\nWykres Feature Importances zapisany do pliku: feature_importances.png")

# ============================================================================
# PREDYKCJE I OCENA MODELU
# ============================================================================
print("\n" + "=" * 70)
print("PREDYKCJE I OCENA MODELU")
print("=" * 70)

# Predykcje na zbiorze treningowym i testowym
y_train_pred = dt_classifier.predict(X_train)
y_test_pred = dt_classifier.predict(X_test)

# ============================================================================
# METRYKI KLASYFIKACJI NA ZBIORZE TESTOWYM
# ============================================================================
print("\n" + "=" * 70)
print("METRYKI KLASYFIKACJI NA ZBIORZE TESTOWYM")
print("=" * 70)

# Obliczenie metryk
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"\nAccuracy (dokladnosc):  {accuracy:.4f}")
print(f"Precision (precyzja):   {precision:.4f}")
print(f"Recall (czulosc):       {recall:.4f}")
print(f"F1 Score:               {f1:.4f}")

# Szczegółowy raport klasyfikacji
print("\n" + "-" * 70)
print("RAPORT KLASYFIKACJI NA ZBIORZE TESTOWYM:")
print("-" * 70)
if le_target is not None:
    target_names = le_target.classes_
else:
    target_names = [str(i) for i in sorted(np.unique(y_test))]

print(classification_report(y_test, y_test_pred, target_names=target_names))

# ============================================================================
# METRYKI DLA KAŻDEJ KLASY (ważne dla niezbalansowanych danych)
# ============================================================================
print("\n" + "-" * 70)
print("METRYKI DLA KAŻDEJ KLASY (wazne dla niezbalansowanych danych):")
print("-" * 70)

precision_per_class = precision_score(y_test, y_test_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_test, y_test_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_test, y_test_pred, average=None, zero_division=0)

for i, class_name in enumerate(target_names):
    print(f"\nKlasa: {class_name}")
    print(f"  Precision: {precision_per_class[i]:.4f}")
    print(f"  Recall:    {recall_per_class[i]:.4f}")
    print(f"  F1 Score:  {f1_per_class[i]:.4f}")

# ============================================================================
# PORÓWNANIE: ZBIÓR TRENINGOWY vs TESTOWY
# ============================================================================
print("\n" + "=" * 70)
print("POROWNIANIE: ZBIOR TRENINGOWY vs TESTOWY")
print("=" * 70)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n{'Metryka':<20} {'Train':<15} {'Test':<15} {'Roznica':<15}")
print("-" * 60)
print(f"{'Accuracy':<20} {train_accuracy:<15.4f} {test_accuracy:<15.4f} {train_accuracy - test_accuracy:<15.4f}")
print(f"{'F1 Score':<20} {train_f1:<15.4f} {test_f1:<15.4f} {train_f1 - test_f1:<15.4f}")

# ============================================================================
# MACIERZ POMYŁEK (CONFUSION MATRIX)
# ============================================================================
print("\n" + "=" * 70)
print("MACIERZ POMYLEK (CONFUSION MATRIX)")
print("=" * 70)

cm = confusion_matrix(y_test, y_test_pred)
print("\nMacierz pomylek:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax, cmap='Blues')
ax.set_title('Macierz Pomylek - Zbior Testowy')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Macierz pomylek zapisana do pliku: confusion_matrix.png")

# ============================================================================
# WIZUALIZACJA DRZEWA DECYZYJNEGO
# ============================================================================
print("\n" + "=" * 70)
print("WIZUALIZACJA DRZEWA DECYZYJNEGO")
print("=" * 70)

plt.figure(figsize=(25, 15))
plot_tree(dt_classifier,
          feature_names=X_train.columns.tolist(),
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Struktura Decision Tree Classifier', fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print("Struktura drzewa zapisana do pliku: decision_tree_structure.png")

# ============================================================================
# EKSPORT REGUŁ TEKSTOWYCH
# ============================================================================
print("\n" + "=" * 70)
print("REGULY TEKSTOWE DRZEWA DECYZYJNEGO:")
print("=" * 70)

tree_rules = export_text(dt_classifier, feature_names=X_train.columns.tolist())
print("\n" + tree_rules)

# Zapisz reguły do pliku
with open('tree_rules.txt', 'w', encoding='utf-8') as f:
    f.write("REGULY TEKSTOWE DRZEWA DECYZYJNEGO:\n")
    f.write("=" * 70 + "\n\n")
    f.write(tree_rules)
print("\nReguły zapisane do pliku: tree_rules.txt")

# ============================================================================
# PODSUMOWANIE MODELU
# ============================================================================
print("\n" + "=" * 70)
print("PODSUMOWANIE MODELU")
print("=" * 70)

print(f"\nModel: Decision Tree Classifier")
print(f"Liczba cech: {X_train.shape[1]}")
print(f"Liczba probek treningowych: {X_train.shape[0]}")
print(f"Liczba probek testowych: {X_test.shape[0]}")
print(f"Liczba klas: {len(target_names)}")
print(f"Glebokosc drzewa: {dt_classifier.get_depth()}")
print(f"Liczba lisci: {dt_classifier.get_n_leaves()}")

print("\nWyniki na zbiorze testowym:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

print("\n" + "=" * 70)
print("ZAKOŃCZENIE ANALIZY")
print("=" * 70)

