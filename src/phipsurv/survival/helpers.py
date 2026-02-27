# ======================
# Standard library
# ======================
import logging
from typing import Any, Dict, Optional, Tuple

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv

# ======================
# Global configuration
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_config(transform_output="pandas")


def convert_to_survival_format(y):
    """
    Convert survival data from signed format to scikit-survival format.

    Parameters:
    - y: 1D array where negative values indicate censored observations,
         positive values indicate events.

    Returns:
    - Structured survival array compatible with scikit-survival functions.
    """
    events = np.where(y < 0, 0, 1)  # 0 for censored, 1 for deceased
    times = np.abs(y)  # Absolute values for survival times
    return Surv.from_arrays(event=events, time=times)


#############################
#         Classes           #
#############################


class StratifiedKFoldSurv:
    def __init__(self, n_splits=5, shuffle=True, random_state=420):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y, groups=None):
        # Define labels based on the sign of y (negative for censored, positive for deceased)

        if isinstance(y, np.ndarray):  # y is a numpy array from Surv.from_arrays
            labels = y[
                "event"
            ]  # Event status is the first column (event = 1, censored = 0)
        else:
            labels = np.where(y < 0, 0, 1)  # 0 for censored, 1 for deceased

        # Stratify based on these labels
        return self.skf.split(X, labels)

    def get_n_splits(self, X=None, y=None, groups=None):
        # We don't need to use the `groups` argument here, just return `n_splits`
        return self.n_splits


class CoxnetWrapper(CoxnetSurvivalAnalysis):
    """
    A wrapper for CoxnetSurvivalAnalysis that automatically converts raw survival data
    (with negative values indicating censoring) into a structured array and selects
    coefficients corresponding to the final (smallest) alpha.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Coxnet model to X and y. If y is not already structured, it is converted.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            1D array of survival times; negative values indicate censoring.

        Returns
        -------
        self : CoxnetWrapper
            The fitted model.
        """
        # Convert y to structured format if needed
        if not (hasattr(y, "dtype") and y.dtype.names is not None):
            y = convert_to_survival_format(y)
        # Call the parent class's fit method
        super().fit(X, y)
        # If self.coef_ has multiple columns (one per alpha), choose the coefficients for the last alpha
        # if self.coef_.ndim > 1:
        # logger.info(f"Coefficient shape before selection: {self.coef_.shape}")
        self.coef_ = self.coef_[:, -1]

        return self


#############################
#         Functions         #
#############################


def filter_validation_data(y_train, y_valid, scores=None):
    """
    Filter validation data to only include samples within the training time range.

    Parameters:
    - y_train: Training survival times (negative for censored, positive for events)
    - y_valid: Validation survival times (negative for censored, positive for events)
    - scores: Optional prediction scores to filter along with y_valid

    Returns:
    - Filtered y_valid, and optionally filtered scores
    """
    valid_indices = y_valid.abs() <= y_train.abs().max()
    y_valid_filtered = y_valid[valid_indices]

    if scores is not None:
        scores_filtered = scores[valid_indices]
        return y_valid_filtered, scores_filtered

    return y_valid_filtered


def c_index_scorer(y_true, y_pred):
    """
    Custom scorer to calculate Concordance Index (C-index) for single-column `y_true`.
    Negative values in `y_true` indicate censored data; positive values indicate events.

    Parameters:
    - y_true: 1D array of survival times (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).

    Returns:
    - C-index: Concordance Index for the predictions.
    """
    # Convert survival data to structured format
    y_surv = convert_to_survival_format(y_true)

    # Calculate and return the C-index
    return concordance_index_censored(y_surv["event"], y_surv["time"], y_pred)[0]


def c_index_scorer_ipcw(
    y_train: pd.Series, y_val: pd.Series, y_pred: np.ndarray, tau: int = None
) -> float:
    """
    Custom scorer to calculate Concordance Index (C-index).

    Parameters:
    - y_train: 1D array of survival times from training set (negative for censored, positive for events).
    - y_val: 1D array of survival times from validation set (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).

    Returns:
    - C-index: Concordance Index for the predictions.
    """

    # Convert survival data to structured format
    y_val_filtered, y_pred_filtered = filter_validation_data(y_train, y_val, y_pred)

    # Filter validation data to only include times within training range
    y_train_surv = convert_to_survival_format(y_train)
    y_val_surv_filtered = convert_to_survival_format(y_val_filtered)

    if tau is None:
        c_index = concordance_index_ipcw(
            y_train_surv, y_val_surv_filtered, y_pred_filtered
        )[0]
    else:
        c_index = concordance_index_ipcw(
            y_train_surv, y_val_surv_filtered, y_pred_filtered, tau=tau
        )[0]

    # Calculate and return the C-index
    return c_index


def search_best_survival_model(
    estimator: Any,
    param_grid: Dict,
    X_train,
    y_train,
    method: str = "bayesian",  # "random", "grid", or "bayesian"
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 420,
    n_jobs: int = -1,
    **kwargs,
) -> Any:
    """
    Tune hyperparameters for a model using one of three methods:
      - RandomizedSearchCV ("random")
      - GridSearchCV ("grid")
      - BayesSearchCV ("bayesian", if scikit-optimize is installed)

    Parameters
    ----------
    estimator : Any
        A scikit-learn style survival estimator.
    param_grid : Dict
        - For 'grid', a dict of parameter lists, e.g. {'param': [1, 2, 3]}.
        - For 'random', a dict of parameter distributions or lists.
        - For 'bayesian', a dict of parameter search spaces (from skopt.space).
    X_train : array-like or DataFrame
        Training feature data.
    y_train : array-like or structured array
        Training survival target (time + event).
    method : str, default="random"
        Which search method to use: "random", "grid", or "bayesian".
    n_splits : int, default=5
        Number of folds for StratifiedKFoldSurv cross-validation.
    n_iter : int, default=30
        - For 'random', number of draws from param distributions.
        - For 'bayesian', number of parameter settings to sample.
        - Ignored for 'grid'.
    random_state : int, default=420
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.
    **kwargs :
        Additional keyword arguments passed to the underlying search class.

    Returns
    -------
    best_estimator_ : Any
        The best-fitted estimator from the search.
    """
    # Create a custom scorer based on the c-index
    custom_scorer = make_scorer(c_index_scorer, greater_is_better=True)
    cv = StratifiedKFoldSurv(n_splits=n_splits, random_state=random_state)

    method = method.lower()
    if method == "bayesian":
        # BayesSearchCV from scikit-optimize
        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_grid,
            n_iter=n_iter,
            scoring=custom_scorer,
            cv=cv,
            refit=True,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif method == "random":
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=custom_scorer,
            cv=cv,
            refit=True,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
    elif method == "grid":
        # GridSearchCV
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=custom_scorer,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            **kwargs,
        )
    else:
        raise ValueError("method must be 'bayesian', 'random' or 'grid'.")

    # Run the search
    search.fit(X_train, y_train)

    # Return the best model
    return search.best_estimator_


def search_best_model(
    estimator,
    param_grid,
    X_train,
    y_train,
    n_splits=5,
    n_iter=30,
    random_state=420,
    n_jobs=1,
):
    # Create a custom scorer using C-index
    custom_scorer = make_scorer(c_index_scorer, greater_is_better=True)
    cv_inner = StratifiedKFoldSurv(n_splits=n_splits, random_state=random_state)
    search = BayesSearchCV(
        estimator,
        search_spaces=param_grid,
        cv=cv_inner,
        n_iter=n_iter,
        scoring=custom_scorer,
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    search.fit(X_train, y_train)

    return search.best_estimator_


##########################################################
###Compute scores or coefficients for feature selection###
##########################################################


def univariate_cox_score_single(j: int, X: np.ndarray, y_surv) -> float:
    """
    Computes the concordance score for a single feature (column j) using a univariate Cox model.

    Parameters
    ----------
    j : int
        Index of the feature to evaluate.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_surv : structured array
        Survival data as a structured array (e.g., from Surv.from_arrays).

    Returns
    -------
    float
        The score (e.g., concordance index) for feature j. Returns 0.0 if an error occurs.
    """
    model = CoxPHSurvivalAnalysis()
    try:
        # Extract the j-th feature as a 2D array
        Xj = X[:, j : j + 1]
        model.fit(Xj, y_surv)
        score = model.score(Xj, y_surv)
        return score
    except Exception:
        # logger.warning(f"Feature index {j} failed with error: {e}")
        return 0.0


def univariate_cox_score(X: np.ndarray, y: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """
    Computes univariate Cox scores for each feature in X in parallel.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        1D array of survival times; negative values indicate censored observations.
    n_jobs : int, optional
        Number of parallel jobs to run (default is -1 to use all available cores).

    Returns
    -------
    np.ndarray
        Array of scores for each feature (shape: (n_features,)).
    """
    # Convert y into a structured survival array: event==1 indicates event occurred,
    # and negative y-values indicate censored observations.
    y_surv = convert_to_survival_format(y)

    n_features = X.shape[1]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(univariate_cox_score_single)(j, X, y_surv) for j in range(n_features)
    )
    return np.array(scores)


# ##############################
#      Build Pipeline          #
# ##############################


def make_pipeline(peptide_cols, demog_cols, estimator, random_state):
    """
    Build a pipeline which:
      - on peptide_cols: does variance threshold + SelectFromModel(coxnet survival)
      - on demog_cols : just passes them through untouched
      - then fits whatever `estimator` you give it
    """
    transformers = []
    if peptide_cols:
        transformers.append(
            (
                "peptides",
                Pipeline(
                    [
                        ("variance_removal", VarianceThreshold(threshold=0.0)),
                        (
                            "feature_selection",
                            SelectFromModel(
                                CoxnetWrapper(
                                    l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100
                                ),
                                threshold=1e-5,
                            ),
                        ),
                    ]
                ),
                peptide_cols,
            )
        )
    if demog_cols:
        transformers.append(("demographics", "passthrough", demog_cols))

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",  # drop anything not in peptide_cols or demog_cols
        verbose_feature_names_out=False,  # <— disable automatic "peptides__…" prefixes
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    return pipe


def build_pipeline(X_train, random_state=420, all_demog=None):
    """
    Create a pipeline using demographic and peptide columns for XGBoost or RandomForest.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data with features (including 'Age' and 'Sex').
    random_state : int
        Seed for reproducibility.
    all_demog : list of extra features

    Returns
    -------
    pipeline : sklearn.Pipeline
        Preprocessing + model pipeline.
    """
    if all_demog is None:
        all_demog = {"Sex", "Age"}
    # Column split
    peptide_cols = [c for c in X_train.columns if c not in all_demog]
    demog_cols = [c for c in X_train.columns if c in all_demog]

    # Define estimator
    estimator = xgb.XGBRegressor(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    # Build pipeline
    return make_pipeline(
        peptide_cols=peptide_cols,
        demog_cols=demog_cols,
        estimator=estimator,
        random_state=random_state,
    )


#####################
#     Fit model     #
#####################


def _build_and_fit_pipeline(
    pipeline, X_train, y_train, param_grid, n_splits, n_iter, random_state, n_jobs
):
    """
    Helper function to build pipeline and perform hyperparameter tuning.

    Parameters
    ----------
    pipeline : Pipeline or None
        Existing pipeline or None to build default
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training targets
    param_grid : dict or None
        Hyperparameter grid for tuning
    n_splits : int
        Number of CV splits for tuning
    n_iter : int
        Number of iterations for Bayesian optimization
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    best_estimator : Pipeline
        Fitted pipeline
    """
    if pipeline is None:
        pipeline = build_pipeline(X_train, random_state=random_state)

    # Perform hyperparameter tuning if param_grid is provided.
    if param_grid is not None:
        valid_params = set(pipeline.get_params().keys())
        # keep only those entries whose key is in valid_params
        param_grid = {k: v for k, v in param_grid.items() if k in valid_params}
        # Use your search_best_model function or similar with BayesSearchCV
        best_estimator = search_best_survival_model(
            pipeline,
            param_grid,
            X_train,
            y_train,
            method="bayesian",
            n_splits=n_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        best_estimator = pipeline
        best_estimator.fit(X_train, y_train)

    return best_estimator


def calculate_cumulative_dynamic_auc(y_train, y_valid, scores, time_points):
    y_valid, scores = filter_validation_data(y_train, y_valid, scores)

    # Convert survival data to structured format
    y_train_surv = convert_to_survival_format(y_train)
    y_valid_surv = convert_to_survival_format(y_valid)
    # Get the observed time range from the validation fold
    min_time_point = y_valid.abs().min()
    max_time_point = y_valid.abs().max() - 0.001
    # Filter desired time points to only those within the observed range
    time_points_highlight = time_points[
        (time_points >= min_time_point) & (time_points <= max_time_point)
    ]

    # Compute time-dependent AUC for this fold:
    auc_values, mean_auc_values = cumulative_dynamic_auc(
        y_train_surv,
        y_valid_surv,
        scores,
        time_points_highlight,  # generate or use time points appropriate for this fold
    )

    return pd.Series(auc_values, index=time_points_highlight), mean_auc_values


######################
#   Nested models    #
######################


def nested_cv_single(
    train_idx,
    valid_idx,
    X_train,
    y_time_train,
    pipeline=None,
    param_grid=None,
    n_splits=5,
    n_iter=30,
    random_state=420,
    n_jobs=-1,
):

    set_config(transform_output="pandas")

    X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_fold, y_valid_fold = (
        y_time_train.iloc[train_idx],
        y_time_train.iloc[valid_idx],
    )

    best_estimator = _build_and_fit_pipeline(
        pipeline,
        X_train_fold,
        y_train_fold,
        param_grid,
        n_splits,
        n_iter,
        random_state,
        n_jobs,
    )

    # Predict risk scores on the validation fold
    risk_scores_fold = best_estimator.predict(X_valid_fold)

    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_valid_fold = best_estimator[:-1].transform(X_valid_fold)
            logger.info(f"shape valid fold:{X_valid_fold.shape}")
        except Exception as e:
            logger.error(f"Error transforming validation data in fold: {e}")
            return None

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator[-1])
    shap_values_fold = explainer.shap_values(X_valid_fold)
    shap_values_fold_df = pd.DataFrame(
        shap_values_fold, index=X_valid_fold.index, columns=X_valid_fold.columns
    )

    # Compute performance score (e.g., c-index)
    # fold_cindex = c_index_scorer(y_valid_fold, risk_scores_fold)
    cindex_fold = c_index_scorer_ipcw(y_train_fold, y_valid_fold, risk_scores_fold)

    return valid_idx, risk_scores_fold, shap_values_fold_df, best_estimator, cindex_fold


def nested_cv(
    X_train: pd.DataFrame,
    y_time_train: pd.Series,
    pipeline: Optional[Pipeline] = None,
    param_grid: Optional[Dict] = None,
    n_splits: int = 10,
    n_splits_inner: int = 5,
    n_iter: int = 30,
    max_time_point=None,
    random_state: int = 420,
    n_jobs: int = 1,
    n_jobs_inner: int = -1,
):
    """
    Perform nested cross-validation to tune hyperparameters and feature selection,
    and aggregate SHAP values and risk scores for each outer fold.

    For each outer fold:
      - Split data into training and validation sets.
      - Run hyperparameter tuning (BayesSearchCV) on the inner folds (if param_grid is provided)
        to select the best estimator.
      - Use the best estimator to predict risk scores on the validation set.
      - Compute SHAP values on the validation set. Note: because the pipeline's
        feature selection step may select a different subset of features per fold,
        the returned SHAP values DataFrame may have different columns.
      - Store the risk scores and SHAP values (with sample indices).

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_time_train : pd.Series
        Survival times for training samples.
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    n_splits : int, default 10
        Number of outer CV splits.
    n_splits_inner : int, default 5
        Number of inner CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    max_time_point : int, default None
        Max time point for time-dependent AUC estimation
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs to run in the outer CV (default is all cores -1).
    n_jobs_inner : int, default 1
        Number of parallel jobs to run in the inner CV (default is 1).
    Returns
    -------
    model_list : List[Pipeline]
        List of best estimators (one per fold).
    shap_values : pd.DataFrame
        DataFrame concatenating SHAP values from all folds, indexed by sample.
    risk_scores : pd.Series
        Series of risk scores from all folds, indexed by sample.
    validation_indices : List[np.ndarray]
        validation indices for each outer fold.
    time_dependent_auc : pd.DataFrame
        DataFrame with time-dependent AUC for each fold.
    c_index : List[float]
        List of performance c-index scores for each outer fold.
    mean_auc : List[float]
        List of performance mean auc scores for each outer fold.
    """
    # outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFoldSurv(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    # Run the outer folds in parallel.
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(nested_cv_single)(
            train_idx,
            valid_idx,
            X_train,
            y_time_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=n_splits_inner,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs_inner,
        )
        for train_idx, valid_idx in outer_cv.split(X_train, y_time_train)
    )
    # Filter out any folds that returned None (e.g., due to transformation errors).
    fold_results = [result for result in fold_results if result is not None]

    # Initialize master containers.
    model_list = []
    validation_indices = []
    c_index = []

    risk_scores = pd.Series(index=X_train.index, name="Risk score")

    shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)

    max_time_point = (
        y_time_train.abs().max() if max_time_point is None else max_time_point
    )
    time_points = np.arange(1, max_time_point, step=1)
    time_dependent_auc = pd.DataFrame(index=np.arange(n_splits), columns=time_points)
    mean_auc = []

    # Aggregate results from each fold.
    i = 0
    for fold_result in fold_results:
        valid_idx, fold_risk, fold_shap_df, model, fold_cindex = fold_result

        model_list.append(model)
        validation_indices.append(valid_idx)
        c_index.append(fold_cindex)

        # Assign risk scores for the validation fold.
        risk_scores.iloc[valid_idx] = fold_risk

        # Update master SHAP values. Since folds are disjoint, direct assignment works.
        shap_values.loc[fold_shap_df.index, fold_shap_df.columns] = fold_shap_df
        del fold_shap_df

        # Build y_valid (same order as fold_risk) and y_train (complement)
        y_valid_fold = y_time_train.iloc[valid_idx]
        y_train_fold = y_time_train.drop(y_valid_fold.index)

        auc_values_fold, mean_auc_fold = calculate_cumulative_dynamic_auc(
            y_train_fold,  # y_time_train.loc[y_time_train.index.difference(y_valid_fold.index)],
            y_valid_fold,
            fold_risk,
            time_points,
        )

        time_dependent_auc.loc[i, auc_values_fold.index] = auc_values_fold.values
        mean_auc.append(mean_auc_fold)
        i = i + 1

    logger.info(
        f"Mean Concordance Index (C-index) across folds: {np.mean(c_index):.4f}"
    )
    logger.info(f"Mean Time-Dependent AUC across folds: {np.mean(mean_auc):.4f}")

    return (
        model_list,
        shap_values,
        risk_scores,
        validation_indices,
        time_dependent_auc,
        c_index,
        mean_auc,
    )


def nested_cv_allfolds(
    X_train: pd.DataFrame,
    y_time_train: pd.Series,
    y_event_train: pd.Series,
    pipeline: Optional[Pipeline] = None,
    param_grid: Optional[Dict] = None,
    n_splits: int = 10,
    n_splits_inner: int = 5,
    n_iter: int = 30,
    max_time_point=None,
    random_state: int = 420,
    n_jobs: int = -1,
):
    """
    Perform nested cross-validation to tune hyperparameters and feature selection,
    and aggregate SHAP values and risk scores for each outer fold.

    For each outer fold:
      - Split data into training and validation sets.
      - Run hyperparameter tuning (BayesSearchCV) on the inner folds (if param_grid is provided)
        to select the best estimator.
      - Use the best estimator to predict risk scores on the validation set.
      - Compute SHAP values on the validation set. Note: because the pipeline's
        feature selection step may select a different subset of features per fold,
        the returned SHAP values DataFrame may have different columns.
      - Compute time-dependent AUC scores for the validation set.
      - Store the risk scores and SHAP values (with sample indices).

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_time_train : pd.Series
        Survival times for training samples.
    y_event_train : pd.Series
        Event status (1 if event occurred, 0 if censored) for training samples.
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    n_splits : int, default 10
        Number of outer CV splits.
    n_splits_inner : int, default 5
        Number of inner CV splits
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    max_time_point : int, default None
        Max time point for time-dependent AUC estimation
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs: int, default 1
        Number of jobs in parallel in the inner CV

    Returns
    -------
    model_list : List[Pipeline]
        List of best estimators (one per fold).
    shap_values : pd.DataFrame
        DataFrame concatenating SHAP values from all folds, indexed by sample.
    risk_scores : pd.Series
        Series of risk scores from all folds, indexed by sample.
    time_dependent_auc : pd.DataFrame
        DataFrame with time-dependent AUC for each fold.
    time_dependent_auc_mean : pd.Series
        Series with AUC mean
    c_index : List[float]
        List of performance scores (e.g., c-index) for each outer fold.
    validation_indices : List[np.ndarray]
        validation indices for each outer fold.
    """
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    c_index = []
    model_list = []
    validation_indices = []
    risk_scores = pd.Series(index=X_train.index, name="Risk score")
    shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)
    max_time_point = (
        y_time_train.abs().max() if max_time_point is None else max_time_point
    )
    time_points = np.arange(1, max_time_point, step=1)
    time_dependent_auc = pd.DataFrame(index=np.arange(n_splits), columns=time_points)
    mean_auc = pd.Series(index=np.arange(n_splits))
    i = 0
    for train_idx, valid_idx in outer_cv.split(X_train, y_event_train):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = (
            y_time_train.iloc[train_idx],
            y_time_train.iloc[valid_idx],
        )

        if pipeline is None:
            params = {
                "objective": "survival:cox",
                "eval_metric": "cox-nloglik",
                "tree_method": "hist",
                "random_state": random_state,
                "n_jobs": -1,
            }
            # Build the pipeline for this fold.
            pipeline = Pipeline(
                [
                    ("variance_removal", VarianceThreshold(threshold=0.0)),
                    (
                        "feature_selection",
                        SelectFromModel(
                            CoxnetWrapper(
                                l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100
                            ),
                            threshold=1e-5,
                        ),
                    ),
                    ("regressor", xgb.XGBRegressor(**params)),
                ]
            )

            # Perform hyperparameter tuning if param_grid is provided.
        if param_grid is not None:
            # Use your search_best_model function or similar with BayesSearchCV
            best_estimator = search_best_survival_model(
                pipeline,
                param_grid,
                X_train_fold,
                y_train_fold,
                method="bayesian",
                n_splits=n_splits,
                n_iter=n_iter,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        else:
            best_estimator = pipeline
            best_estimator.fit(X_train_fold, y_train_fold)

        model_list.append(best_estimator)

        # Predict risk scores on the validation fold
        risk_scores_fold = best_estimator.predict(X_valid_fold)
        risk_scores.iloc[valid_idx] = risk_scores_fold

        # Transform validation data for SHAP computation
        if len(best_estimator) > 1:
            try:
                X_valid_fold = best_estimator[:-1].transform(X_valid_fold)
            except Exception as e:
                logger.error(f"Error transforming validation data in fold: {e}")
                continue

        # Compute SHAP values using the regressor (last step)
        explainer = shap.TreeExplainer(best_estimator["regressor"])
        shap_values_fold = explainer.shap_values(X_valid_fold)
        shap_values_fold_df = pd.DataFrame(
            shap_values_fold, index=X_valid_fold.index, columns=X_valid_fold.columns
        )
        shap_values.loc[shap_values_fold_df.index, shap_values_fold_df.columns] = (
            shap_values_fold_df
        )

        fold_cindex = c_index_scorer_ipcw(y_train_fold, y_valid_fold, risk_scores_fold)
        c_index.append(fold_cindex)

        auc_values_fold, mean_auc_values_fold = calculate_cumulative_dynamic_auc(
            y_train_fold, y_valid_fold, risk_scores_fold, time_points
        )
        time_dependent_auc.loc[i, auc_values_fold.index] = auc_values_fold.values
        mean_auc.loc[i] = mean_auc_values_fold

        validation_indices.append(
            valid_idx
        )  # Track SampleNames corresponding to the validation set
        i = i + 1
    logger.info(
        f"Mean Concordance Index (C-index) across validation folds: {np.mean(c_index):.4f}"
    )
    logger.info(
        f"Mean Time-Dependent AUC across folds: {time_dependent_auc.mean(skipna=True).mean():.4f}"
    )

    return (
        model_list,
        shap_values,
        risk_scores,
        validation_indices,
        time_dependent_auc,
        c_index,
        mean_auc,
    )


################################
#  External Validation model   #
################################

def align_external_to_train(X_train, X_ext, fill_value=0, min_overlap=0.7):
    train_cols = list(X_train.columns)
    overlap = len(set(train_cols) & set(X_ext.columns)) / len(train_cols)
    if overlap < min_overlap:
        raise ValueError(f"External set overlaps only {overlap:.1%} of training features.")
    X_ext_aligned = X_ext.reindex(columns=train_cols, fill_value=fill_value)
    return X_ext_aligned

def train_and_validate_model(
    X_train: pd.DataFrame,
    y_time_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_time_test: Optional[pd.Series] = None,
    pipeline: Optional[Pipeline] = None,
    param_grid: Optional[Dict] = None,
    best_estimator: Optional[Pipeline] = None,  # need to be pipeline
    n_splits: int = 10,
    n_iter: int = 30,
    max_time_point=None,
    random_state: int = 420,
    n_jobs: int = -1,
    get_only_model: bool = False,
) -> (
    None | Pipeline | Tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series, float, float]
):
    """
    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_time_train : pd.Series
        Survival times for training samples (negative:censored, positive:event).
    X_test : pd.DataFrame, optional
        Feature matrix for testing.
    y_time_test : pd.Series, optional
        Survival times for testing samples (negative:censored, positive:event).
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    best_estimator : Pipeline, optional
        Best estimator to predict on test data
    n_splits : int, default 10
        Number of outer CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    max_time_point : int, default None
        Max time point for time-dependent AUC estimation
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs to run in the outer CV (default is all cores -1).
    get_only_model : bool, default False
        return only the fitted model
    Returns
    -------
    best_estimator : Pipeline
        best estimator
    shap_values : pd.DataFrame
        DataFrame SHAP values, indexed by sample.
    risk_scores : pd.Series
        Series of risk scores, indexed by sample.
    time_dependent_auc : pd.Series
        DataFrame with time-dependent AUC, indexed by timepoint.
    c_index : float
        Performance score (e.g., c-index).
    mean_auc_values : float
        Time-dependent AUC mean based on given timepoints
    """

    # if the user handed us an already‐trained model, use that
    if best_estimator is None:
        best_estimator = _build_and_fit_pipeline(
            pipeline,
            X_train,
            y_time_train,
            param_grid,
            n_splits,
            n_iter,
            random_state,
            n_jobs,
        )

        if get_only_model:
            return best_estimator

    X_test = align_external_to_train(X_train, X_test, fill_value=0, min_overlap=0.7)
    shap_values_df = pd.DataFrame(0.0, index=X_test.index, columns=X_test.columns)

    # Predict risk scores on the validation fold
    risk_scores = best_estimator.predict(X_test)

    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_test = best_estimator[:-1].transform(X_test)
            logger.info(f"shape test data:{X_test.shape}")
        except Exception as e:
            logger.error(f"Error transforming Testing data: {e}")
            return None

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator[-1])
    shap_values = explainer.shap_values(X_test)
    shap_values = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns)
    shap_values_df.loc[shap_values.index, shap_values.columns] = shap_values

    # Compute performance score (e.g., c-index)
    # c_index = c_index_scorer(y_test, risk_scores)
    c_index = c_index_scorer_ipcw(y_time_train, y_time_test, risk_scores)

    max_time_point = (
        y_time_train.abs().max() if max_time_point is None else max_time_point
    )
    time_points = np.arange(1, max_time_point, step=1)
    time_dependent_auc, mean_auc_values = calculate_cumulative_dynamic_auc(
        y_time_train, y_time_test, risk_scores, time_points
    )
    risk_scores = pd.Series(risk_scores, index=X_test.index, name="Risk score")

    logger.info(f"Mean Concordance Index (C-index) in testing set: {c_index:.4f}")
    logger.info(f"Mean Time-Dependent AUC in testing set: {mean_auc_values:.4f}")

    return (
        best_estimator,
        shap_values_df,
        risk_scores,
        time_dependent_auc,
        mean_auc_values,
        c_index,
    )
