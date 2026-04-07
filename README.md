# Analiza i przewidywanie zdrowia oraz jakości snu

## Opis Projektu
Projekt stanowi wstęp do uczenia maszynowego i analizy danych. 
Jego głównym celem jest badanie fizjologicznych oraz behawioralnych czynników 
wpływających na zdrowy sen, a także zbudowanie podstawowych systemów decyzyjnych 
przewidujących kluczowe wskaźniki zdrowotne pacjentów.

## Zbiór Danych
Dane wykorzystane w projekcie znajdują się w pliku `sleep_health_dataset.csv`. 
Zbiór zawiera m.in. informacje o wieku, jakości i długości snu, poziomie stresu, 
liczbie przebudzeń, a także o wydajności poznawczej. 

Główne zmienne docelowe analizowane w projekcie to:
* **Klasyfikacja:** `felt_rested`
* **Regresja:** `cognitive_performance_score`

## Główne Etapy Projektu
1. **Eksploracyjna Analiza Danych:** Analiza rozkładów zmiennych ciągłych oraz kategorialnych. Sprawdzenie balansu klas.
2. **Przygotowanie Danych:** Odpowiedni podział zbioru na dane treningowe i testowe, aby uniknąć wycieku danych.
3. **Model Klasyfikacyjny:** Zaprojektowanie prostego i rozbudowanego systemu decyzjynjego przewidującego poczucie wypoczęcia na podstawie wybranych cech.
4. **Model Regresyjny:** Stworzenie systemu przypisującego pacjentów do grup i zwracania średniej wartości sprawności poznawczej obliczonej na zbiorze treningowym.

## Technologie
* **Język:** Python 3.13
* **Biblioteki:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## Jak uruchomić projekt?
1. Sklonuj repozytorium używając komendy:
    ```bash 
   git clone https://github.com/PPiotr491/MSiD.git.
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt