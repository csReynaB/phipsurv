# Survival XGBoost–Cox Pipeline

This repository contains a modular and reproducible pipeline for **survival analysis using gradient-boosted trees (XGBoost) with a Cox proportional hazards objective**.  
It is designed for **biomedical cohorts**, supports **nested cross-validation**, and integrates **model interpretation (SHAP)** and **survival-specific evaluation metrics**.

The scripts have been tested for **HPC/SLURM environments**, and can also be run locally.

---

## 🚀 Features

- **XGBoost Cox survival models**
- **Nested cross-validation** (outer / inner folds)
- **Bayesian hyperparameter optimization** (scikit-optimize)
- **Time-dependent AUC and concordance index**
- **Kaplan–Meier and log-rank testing**
- **SHAP-based feature interpretation**
- HPC-ready **SLURM array execution**

---

## 📁 Repository structure

```text
survival_project/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── run_survival.sh
│
├── src/
│   └── survival/
│       ├── cli/
│       │   └── main_survival_trainTest.py
│       ├── io/
│       │   └── dataHandler.py
│       ├── ml/
│       │   └── ML_survival_helpers.py
│       ├── plots/
│       │   ├── metricsPlots_survival_helpers.py
│       │   └── plots_helpers.py
│       └── utils/
│           └── peptides_filter.py
│
├── configs/
│   ├── survival.yaml
│
├── data/
│
│
├── notebooks/
│   └── BC-Engl_survivalAnalysis.ipynb
│
``` 

---

## 🧠 Method overview

The pipeline implements a **Cox proportional hazards model via XGBoost**, allowing non-linear effects and interactions while preserving survival-time censoring.

Key steps:
1. Data loading and preprocessing
2. Feature filtering (prevalence thresholds, optional covariates)
3. Nested cross-validation
4. Bayesian hyperparameter tuning
5. Model fitting and evaluation
6. Survival-specific metrics and plots
7. SHAP-based feature interpretation

---

## ⚙️ Requirements

The pipeline is designed to run in a **conda environment**.

Core dependencies:
- `numpy`, `pandas`, `scipy`
- `scikit-learn`
- `xgboost`
- `scikit-survival`
- `lifelines`
- `shap`
- `matplotlib`, `seaborn`
- `joblib`, `pyyaml`, `tqdm`

Formatting / linting (optional):
- `black`, `isort`, `ruff`

Main dependencies are documented in `pyproject.toml`.  
Nothing is installed automatically.

---

## 🧪 Environment setup (example)

```bash
conda create -n survival_xgb python=3.10
conda activate survival_xgb
conda install -c conda-forge \
  numpy pandas scipy joblib tqdm pyyaml \
  scikit-learn xgboost scikit-optimize \
  scikit-survival lifelines shap \
  matplotlib seaborn
conda install -c conda-forge black isort ruff 

# or like
conda env create -f ML_env.yml --prefix /path/to/envs/survival_xgb

## Running array in SLURM

sbatch --array=1-100 \
  scripts/run_survival.sh \
  seeds.txt \ # one random number per line
  configs/survival.yaml \
  results/ \
  project_results

## Internally running
python -m survival.cli.main_survival_trainTest --help
