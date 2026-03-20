# ML for Bot Detection

Binary classification project comparing **Logistic Regression** and **XGBoost** for detecting social media bots. Built as an assignment for the Introduction to Machine Learning course at Tilburg University.

## What it does

Classifies social media accounts as bots or humans using 59 profile features (mix of numerical and categorical). The dataset (`bots_vs_users.csv`) contains 5,874 accounts — perfectly balanced at 50/50 bots vs. users.

## Dataset

`bots_vs_users.csv` — 5,874 accounts × 59 features + 1 binary target column.

Feature groups:
- **Numerical** (14): `avg_views`, `avg_likes`, `posting_frequency_days`, `subscribers_count`, etc.
- **Categorical binary** (41): `has_photo`, `has_website`, `is_verified`, etc.
- **Categorical multi-class** (4): `city`, `gender`, `marital_status`, `occupation_type`

**Missing data is extensive**: 4,483 accounts have no numerical data at all; many categorical columns have 2,500–5,000 "Unknown" entries.

## Approach

### Preprocessing (`DataCleaner` class)

Applied in this order, fit only on the training set to prevent leakage:

1. **Minimal profile flag** — binary feature marking accounts with no numerical data at all
2. **Subscribers count** — cleaned from messy categorical strings to a proper numeric column
3. **Yeo-Johnson scaling** — removes skew and standardizes numerical features
4. **Median imputation** — fills remaining NaNs with training-set medians
5. **Categorical encoding** — binary columns label-encoded; multi-class columns one-hot encoded with rare categories (< 30 members) grouped into an "other" bin (reduced city from 362 → 6 categories)

### Models

| Model | Tuning | Best params |
|-------|--------|-------------|
| Dummy (baseline) | — | stratified random |
| K-Nearest Neighbors | Grid search (k=1..20) | k=6 |
| Logistic Regression | Optuna TPE (10 runs × 30 trials) | C≈2.5 |
| XGBoost | Optuna TPE (100 trials) | n_estimators=281, lr≈0.20, max_depth=3 |
| Neural Network | Fixed architecture | 64→32 dense with dropout |

All hyperparameters optimized via stratified k-fold cross-validation on the training set using F1 score.

### Results

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| Logistic Regression | 97.28% | ~0.974 | — |
| XGBoost | 97.70% | ~0.976 | — |

Top predictive features (both models): `city`, `has_status`, `has_website`, `has_photo`, `subscribers_count`.

SHAP analysis confirms city (Kostomuksha strongly predicts bot) and profile completeness signals are the most influential features.

## Tech Stack

- **Python** with Jupyter notebooks
- **pandas / numpy** — data manipulation
- **scikit-learn** — preprocessing, KNN, Logistic Regression, cross-validation
- **XGBoost** — gradient boosting
- **TensorFlow/Keras** — neural network baseline
- **Optuna** — Bayesian hyperparameter optimization
- **SHAP** — model interpretability
- **matplotlib / seaborn** — visualization

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost tensorflow optuna shap matplotlib seaborn
jupyter notebook code.ipynb
```

`code.ipynb` is the full notebook (EDA → preprocessing → model training → evaluation → SHAP).
`code_simple.ipynb` is a cleaned-up version without the hyperparameter search exploration.

Both notebooks expect `bots_vs_users.csv` in the same directory.

## Files

| File | Description |
|------|-------------|
| `code.ipynb` | Main notebook — full pipeline with exploration |
| `code_simple.ipynb` | Streamlined notebook |
| `bots_vs_users.csv` | Dataset |
| `Report.pdf` | Written report |
| `Figures/` | Saved plots (ROC, PR curves, SHAP plots) |
