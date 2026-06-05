# AGENTS.md

## Big picture
- Repo is a set of lab-style ML exercises on sleep-health data; each `Lista*/` folder contains a `task_description/` PDF and a final Jupyter notebook with solutions.
- Shared dataset lives in `resources/sleep_health_dataset.csv`; most code reads it relative to the notebook/script location, usually as `../resources/sleep_health_dataset.csv`.
- `Lista3/` turns repeated notebook logic into reusable modules for preprocessing, logging, metrics, and custom regularized gradient descent.

## Workflow for new tasks
- Before adding code, check whether the needed helper already exists in a notebook or module; the repo intentionally reuses functions like `load_dataset()`, `df_label_encoding()`, `predict_linear()`, and the custom metric helpers.
- When notebook files are large, inspect them in parts and search for existing `def`/`class` blocks before inventing new utilities.
- Final deliverable for each list is the notebook, not a standalone package or CLI.

## Task map by list
- `Lista1/sleep_health_analysis.ipynb`: exploratory analysis of continuous and categorical variables, association measures (`cramers_v`, correlation ratio), simple rule-based classifier for `felt_rested`, more complex decision rules, and a regression baseline for `cognitive_performance_score`.
- `Lista2/wprowadzenie_do_listy_2.ipynb`: intro to pipeline vs algorithm, decision tree as `if/else`, classification metrics, linear regression, bias–variance discussion, and a bonus Naive Bayes section.
- `Lista2/sleep_health_prediciton_models.ipynb`: end-to-end `DecisionTreeClassifier` for `felt_rested`, entropy/information gain analysis, threshold inspection, and regression comparison of analytical solution, gradient descent, and `sklearn`.
- `Lista3/sleep_healt_advanced_models.ipynb`: reusable preprocessing, label encoding, linear regression helper, L1/L2 regularization analysis with custom GD (warm_start for smooth paths), decision tree regularization (min_samples_split, min_samples_leaf, max_features), and **Bagging (Bootstrap Aggregating)** ensemble methods with custom implementation vs sklearn RandomForest comparison.

## Notebook method inventory
- `Lista1/sleep_health_analysis.ipynb`: `cramers_v` (categorical-categorical association), `correlation_ratio` (categorical-numerical association), `encode_data` (label-encodes dataframe columns), `calculateThreshold` (finds split threshold), `predict` (simple classifier), `predict_complex` (multi-rule classifier), `predict_regression` (group-based regression baseline).
- `Lista2/wprowadzenie_do_listy_2.ipynb`: `mse` / `mae` (basic regression metrics), `gini` (impurity), `entropy_quick` (entropy shortcut), `predict_line` (y = ax + b), `fit_poly_and_predict` (polynomial fit + prediction).
- `Lista2/sleep_health_prediciton_models.ipynb`: `mse` / `mae`, `entropy` (class entropy), `information_gain` (split quality), `get_first_threshold_for_feature` (first tree threshold for a feature), `analyze_global_split` (evaluate a global split), `fit_linear_regression_analytical`, `predict_linear`, `fit_linear_regression_gd`.
- `Lista3/sleep_healt_advanced_models.ipynb`: 
  - **Regularization section**: `fit_penalty_path()` (Lasso/Ridge paths with warm_start), `build_stats_table()` (compile alpha vs zero_count/mean_weight), `plot_weight_paths()`, `plot_zero_counts()`.
  - **Tree regularization section**: `fit_tree_classifier()`, `evaluate_tree_classifier()`, `sweep_tree_parameter()`, `plot_train_test_curves()`, `summarize_tree_sweep()`.
  - **Bagging section**: `BaggingClassifier` (custom), `compare_single_vs_bagging()`, `plot_comparison_bars()`, `plot_feature_importances()`. Also compares sklearn `RandomForestClassifier` and custom `BaggingClassifier`.

## Key components and data flow
- `Lista2/decision_tree_model.py` is a full linear pipeline: load CSV -> drop `person_id` -> label-encode categoricals -> train/test split -> train `DecisionTreeClassifier` -> save plots and rules (`feature_importances.png`, `confusion_matrix.png`, `decision_tree_structure.png`, `tree_rules.txt`).
- `Lista3/data_preprocessing.py` centralizes dataset loading and encoding; prefer `load_dataset()`, `extract_target()`, `df_label_encoding()` instead of duplicating preprocessing.
- `Lista3/Algorithms/regularized_gd.py` and `Lista3/Algorithms/regularization_analysis.py` implement custom L1/L2 GD and plotting helpers with warm_start support; they expect `numpy` arrays, not pandas frames.
- `Lista3/Algorithms/tree_regularization_analysis.py` analyzes tree structure regularization (max_depth, min_samples_split, min_samples_leaf, max_features) with sweep and evaluation utilities.
- `Lista3/Algorithms/bagging.py` implements `BaggingClassifier` from scratch: bootstrap sampling, training N independent weak learners (decision trees), hard/soft voting for predictions, and feature importance aggregation. Key methods: `fit()` (bootstrap + train N trees), `predict()` (hard voting), `predict_proba()` (soft voting), `get_feature_importances()` (average across trees).
- `Lista3/Algorithms/bagging_analysis.py` provides comparison and visualization utilities: `compare_bagging_variants()` (sweep n_estimators), `compare_single_vs_bagging()` (baseline vs ensemble), `plot_*()` functions for train/test curves, feature importances, and overfitting gaps.

## Project-specific conventions
- Categorical encoding uses `LabelEncoder` per column; see `Lista2/decision_tree_model.py` and `Lista3/data_preprocessing.py` for the exact pattern.
- Plot files are written to the current working directory; do not assume an `outputs/` folder exists.
- `Lista3/Logging/setup_logging.py` provides stdout logging, while modules use `logging.getLogger(__name__)` for debug output.
- `Lista3/Algorithms/regularized_gd.py` keeps the bias term unregularized (`w[0]`), which matters when extending the gradient-descent helpers.
- **Bagging convention**: `BaggingClassifier.fit()` uses `bootstrap=True` by default (sampling with replacement). Each tree i gets `random_state=base_seed + i` for reproducibility. `predict()` uses **hard voting** (mode), `predict_proba()` uses **soft voting** (average probabilities). Feature importances are averaged across all estimators.

## Developer workflows and integrations
- Dependencies are pinned in `requirements.txt`; README documents `pip install -r requirements.txt` for setup.
- `Lista1/main.py` shows external downloading with `kagglehub`, but the rest of the repo expects a local CSV in `resources/`.
- Metrics helpers live in `Lista3/Metrics/metrics.py` and are reused by `Lista3/Algorithms/regularized_gd.py`.
- **Bagging tests**: `test_bagging.py` verifies bootstrap sampling, voting mechanisms, and accuracy improvements. Simple smoke test — does not create notebooks or extra artifacts. Use `BaggingClassifier` just like sklearn's API: `model.fit(X_train, y_train)`, `model.predict(X_test)`.
- **Integration**: Bagging is demonstrated in the notebook alongside regularization and tree parameter tuning, all on the same target (`felt_rested` classification). Comparison includes single tree, custom Bagging, and sklearn RandomForest to illustrate variance reduction.

