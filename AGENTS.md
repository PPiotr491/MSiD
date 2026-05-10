# Copilot Instructions

## Summary
- Small, single-repo Python 3.13 project for sleep-health analysis and prediction.
- Main targets: classification `felt_rested` and regression `cognitive_performance_score`.
- Core data file is `resources/sleep_health_dataset.csv`.

## High-level repo facts
- Size: small, single-project repo (no packages, no CI workflows).
- Languages: Python; notebooks are used for exploratory work.
- Frameworks/libs: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, Jupyter stack.
- OS assumptions: some scripts use Windows-style paths.

## Build / run / test / lint (validated steps)
**Runtime verified (2026-05-08)**
- `python --version` -> 3.13.9
- `python -m pip --version` -> pip 26.1
- `Lista2\venv\Scripts\python.exe --version` -> 3.13.9

**Bootstrap (dependencies)**
- Local dependencies are already installed in `Lista2/venv` (per user). Prefer running scripts with the venv Python.
- Command: `python -m pip install -r requirements.txt`
- Status: not executed in this session (tool call was skipped).

**Run / test (scripts are the validation)**
- `Lista2\venv\Scripts\python.exe test_regression.py`
  - Status: SUCCESS.
  - Note: emits a `Pandas4Warning` about `select_dtypes(include=['object'])` behavior changes.
- `Lista2\venv\Scripts\python.exe Lista2\decision_tree_model.py`
  - Status: SUCCESS when run from `Lista2/`.
  - Produces PNGs and `tree_rules.txt` in the current working directory.
- `Lista2\venv\Scripts\python.exe Lista2\overfitting_underfitting_analysis.py`
  - Status: not run (helper script for notebook insertion).

**Lint**
- No lint configuration or scripts found.

**CI / checks**
- No `.github/workflows` or other CI configuration found.

## Project layout and architecture
**Root files**
- `README.md`
- `requirements.txt`
- `test_regression.py`

**Top-level directories**
- `resources/` (dataset)
- `Lista1/` (exploratory / notebook-oriented work)
- `Lista2/` (supervised modeling scripts and notebooks)
- `.git/`, `.idea/`

**Key scripts and notebooks**
- `test_regression.py`: linear regression comparison on `cognitive_performance_score`.
- `Lista2/decision_tree_model.py`: DecisionTreeClassifier on `felt_rested`, saves artifacts in CWD.
- `Lista2/overfitting_underfitting_analysis.py`: helper code intended to be copied into `Lista2/sleep_health_prediciton_models.ipynb`.
- `Lista2/sleep_health_prediciton_models.ipynb`: główny notebook projektu — zawiera pełny przepływ analizy i modeli (klasyfikacja i regresja), w tym sekcje korzystające z kodu z `Lista2/overfitting_underfitting_analysis.py`.
- `Lista2/wprowadzenie_do_listy_2.ipynb`: wprowadzenie do tematów rozwijanych dalej w projekcie; traktuj jako kontekst merytoryczny dla `Lista2/sleep_health_prediciton_models.ipynb`.
- `Lista1/sleep_health_analysis.ipynb`, `Lista2/sleep_health_prediciton_models.ipynb`, `Lista2/wprowadzenie_do_listy_2.ipynb`.

**Data paths (important)**
- Root scripts read `resources/sleep_health_dataset.csv`.
- `Lista2` scripts read `../resources/sleep_health_dataset.csv` (relative to `Lista2/`).

**Generated artifacts**
- `Lista2/decision_tree_model.py` writes:
  - `feature_importances.png`
  - `confusion_matrix.png`
  - `decision_tree_structure.png`
  - `tree_rules.txt`

**Non-obvious dependency note**
- `Lista1/main.py` imports `kagglehub` and hardcodes `C:/Users/Super/...` paths; `kagglehub` is not in `requirements.txt`.

**Local virtual environment**
- A checked-in venv exists at `Lista2/venv/`; ignore it for edits and searches.

## README snapshot (for quick context)
- Describes an ML intro project focused on sleep health.
- Targets: `felt_rested` (classification) and `cognitive_performance_score` (regression).
- Declares Python 3.13 and core libs; suggests `pip install -r requirements.txt`.

## Search guidance
- Trust this file first; only search the repo if information here is missing or contradicted by files you open.
