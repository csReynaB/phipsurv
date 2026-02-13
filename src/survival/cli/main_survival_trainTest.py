# ======================
# Standard library
# ======================
import argparse
import json
import logging
import time

# ======================
# Third-party libraries
# ======================
import joblib
from sklearn import set_config

# ======================
# Local / project imports
# ======================
from survival.io.dataHandler import (
    Config,
    FeatureManager,
    MetadataHandler,
    OligosHandler,
)
from survival.ml.ML_survival_helpers import (
    build_pipeline,
    nested_cv,
    train_and_validate_model,
)

# ======================
# Global configuration
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_config(transform_output="pandas")


def process_survival_data(feature_manager_obj):
    """
    Process features and target data for survival analysis.

    Parameters
    ----------
    feature_manager_obj : FeatureManager
        Feature manager instance to get data from

    Returns
    -------
    X : pd.DataFrame
        Feature matrix without OS months column
    y_time : pd.Series
        Time data transformed for survival analysis (positive for events, negative for censored)
    y_event : pd.Series
        Event indicator
    """
    X, y_event = feature_manager_obj.get_features_target()
    y_time = X["OS months"]
    X.drop(columns=["OS months"], inplace=True)
    y_time = y_time.where(y_event == 1, -y_time)  # np.where(y_event, y_time, -y_time)
    logger.info(f"shape data {X.shape}")
    return X, y_time, y_event


def setup_feature_manager(config, filters_metadata, args):
    """
    Helper function to set up feature manager with common configuration.
    """
    config.filters_metadata = filters_metadata
    metadata_handler = MetadataHandler(config)
    oligos_handler = OligosHandler(config)
    feature_manager = FeatureManager(
        config,
        metadata_handler,
        oligos_handler,
        subgroup=args.subgroup,
        with_oligos=args.with_oligos,
        with_additional_features=args.with_additional_features,
        prevalence_threshold_min=args.prevalence_threshold_min,
        prevalence_threshold_max=args.prevalence_threshold_max,
    )
    return feature_manager


def str2bool(x):
    xl = x.lower()
    if xl in ("yes", "true", "t", "y", "1"):
        return True
    if xl in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {x!r}")


if __name__ == "__main__":
    # Parse the command-line argument for the random seed
    parser = argparse.ArgumentParser(
        description="Run nested CV and validation with custom random seed and metadata filters for survival models."
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        nargs="?",
        default=420,
        help="Random seed (default: 420)",
    )
    parser.add_argument(
        "--config",
        "-cf",
        type=str,
        default="config.yaml",
        help="Full path of the config file to use (default: config.yaml in script directory)",
    )
    parser.add_argument(
        "--run_nested_cv",
        "-ncv",
        type=str2bool,
        default=True,
        help="Run nested cv for training set (default: True)",
    )
    parser.add_argument(
        "--use_pretrained",
        "-upr",
        type=str2bool,
        default=False,
        help="Load pre trained model from joblib file (default: False)",
    )
    parser.add_argument(
        "--only_train_model",
        "-otm",
        type=str2bool,
        default=False,
        help="Whether to only train model or return predictions as well(True|False).",
    )

    parser.add_argument(
        "--subgroup",
        "-sub",
        type=str,
        default="all",
        help="What subgroup of peptides to include in the analysis. Default = all",
    )
    parser.add_argument(
        "--with_oligos",
        "-wo",
        type=str2bool,
        default=True,
        help="Include or not peptides in the analysis. Default = True",
    )
    parser.add_argument(
        "--with_additional_features",
        "-wa",
        type=str2bool,
        default=False,
        help="Include or not additional features in the analysis (e.g. Sex and Age). Default = False",
    )
    parser.add_argument(
        "--prevalence_threshold_min",
        "-min",
        type=float,
        default=2.0,
        help="Minimum prevalence threshold filter for training set. Default = 2.0",
    )
    parser.add_argument(
        "--prevalence_threshold_max",
        "-max",
        type=float,
        default=98.0,
        help="Maximum prevalence threshold filter for training set. Default = 98.0",
    )

    parser.add_argument(
        "--outer_cv_split",
        "-ocv",
        type=int,
        default=5,
        help="Number of k folds for outer cross-validation. Default = 5",
    )
    parser.add_argument(
        "--inner_cv_split",
        "-icv",
        type=int,
        default=5,
        help="Number of k folds for inner cross-validation. Default = 5",
    )

    parser.add_argument(
        "--max_timepoint",
        "-maxT",
        type=int,
        default=25,
        help="Max time point for survival analysis AUC (default: 25)",
    )

    parser.add_argument(
        "--train",
        "-t",
        type=json.loads,
        default={},
        help=(
            "JSON dict of metadata for train, e.g. "
            '\'{"group_test":"Controls","other_key":"value"}\'. '
            "Default = {}"
        ),
    )
    # Instead of two separate filter_val flags, we do:
    #   --validate '{"treatment":"ICI"}' Cirrhosis-ICI-H
    #   --validate '{"treatment":"TKI"}' Cirrhosis-ICI-TKI

    parser.add_argument(
        "-v",
        "--validate",
        nargs=2,  # two arguments per occurrence
        action="append",
        default=[],
        metavar=("FILTER_JSON", "OUT_BASENAME"),
        help='One validation set: JSON filter and output‐base, e.g. \'{"treatment":"ICI"} Cirrhosis-ICI-H\'.',
    )

    parser.add_argument(
        "--input_dir",
        "-id",
        type=str,
        default=".",
        help="Base name for directory to input joblib files (default: .)",
    )
    parser.add_argument(
        "--out_dir",
        "-d",
        type=str,
        default=".",
        help="Base name for directory to save files (default: .)",
    )
    parser.add_argument(
        "--input_name",
        "-i",
        type=str,
        default="input_name",
        help="Base name for pre_trained model files for test predictions (default: input_name)",
    )
    parser.add_argument(
        "--out_name",
        "-o",
        type=str,
        default="out_name",
        help="Base name for nested‐CV and train_test split predictions (default: out_name)",
    )

    args = parser.parse_args()

    random_seed = args.seed
    outer_cv_split = args.outer_cv_split
    inner_cv_split = args.inner_cv_split

    val_specs = [(json.loads(filt), outname) for filt, outname in (args.validate or [])]

    config_file = args.config
    config = Config(config_file)
    config.get_bayesian_param_grid_from_dict_items(
        "xgboost"
    )  # format bayesian param grid from config file

    # Check if training set should be split
    start_time = time.time()
    feature_manager = setup_feature_manager(config, args.train, args)
    X_train, y_time_train, y_event_train = process_survival_data(feature_manager)

    pipeline = build_pipeline(X_train, random_state=random_seed)
    if args.run_nested_cv:
        start_time = time.time()
        (
            model_list,
            train_shap_values,
            risk_scores_train,
            validation_indices,
            time_dependent_auc_train,
            c_index_train,
            mean_auc_train,
        ) = nested_cv(
            X_train,
            y_time_train,
            pipeline=pipeline,
            param_grid=config.param_grid,
            n_splits=outer_cv_split,
            n_splits_inner=inner_cv_split,
            n_iter=100,
            max_time_point=args.max_timepoint,
            random_state=random_seed,
            n_jobs=1,
            n_jobs_inner=-1,
        )

        # Save the results as a dictionary
        results = {
            "model_list": model_list,
            "train_shap_values": train_shap_values,
            "risk_scores_train": risk_scores_train,
            "validation_indices_train": validation_indices,
            "time_dependent_auc_train": time_dependent_auc_train,
            "c_index_train": c_index_train,
            "mean_auc_train": mean_auc_train,
        }
        joblib.dump(
            results,
            f"{args.out_dir}/nested_XGB-survivalCox_{args.out_name}_{random_seed}.joblib",
        )

        end_time = time.time()
        logger.info(
            f"nested cv runtime for {args.out_name}: {end_time - start_time:.2f} seconds"
        )

    if val_specs:
        start_time = time.time()
        if args.use_pretrained:
            input_file = f"{args.input_dir}/validation_XGB-survivalCox_{args.input_name}_{random_seed}.joblib"
            best_estimator = joblib.load(input_file)[
                "best_estimator"
            ]  # ensure there is best_estimator
            end_time = time.time()
            logger.info(
                f"load best model with {input_file} runtime: {end_time - start_time:.2f} seconds"
            )
        else:
            best_estimator = train_and_validate_model(
                X_train,
                y_time_train,
                X_test=None,
                y_time_test=None,  # we won’t score yet
                pipeline=pipeline,
                param_grid=config.param_grid,
                n_splits=outer_cv_split,
                n_iter=100,
                max_time_point=25,
                random_state=random_seed,
                n_jobs=-1,
                get_only_model=True,  # <— return only the fitted estimator
            )
            end_time = time.time()
            logger.info(
                f"train best model with {args.out_name} runtime: {end_time - start_time:.2f} seconds"
            )

        if args.only_train_model:
            results = {"best_estimator": best_estimator}
            joblib.dump(
                results,
                f"{args.out_dir}/training_XGB-survivalCox_{args.out_name}_{random_seed}.joblib",
            )
        else:
            # run validation if some sets were given
            for filter_val, out_val in val_specs:
                start_time = time.time()

                config.filters_metadata = filter_val
                feature_manager.prevalence_threshold_min = 0.0
                feature_manager.prevalence_threshold_max = 100.0
                X_test, y_time_test, y_event_test = process_survival_data(
                    feature_manager
                )

                (
                    best_estimator,
                    test_shap_values,
                    risk_scores_test,
                    time_dependent_auc_test,
                    time_dependent_auc_mean_test,
                    c_index_test,
                ) = train_and_validate_model(
                    X_train,
                    y_time_train,
                    X_test,
                    y_time_test,
                    param_grid=config.param_grid,
                    best_estimator=best_estimator,
                    n_splits=outer_cv_split,
                    n_iter=100,
                    max_time_point=args.max_timepoint,
                    random_state=random_seed,
                    n_jobs=-1,
                    get_only_model=False,
                )

                # Save the results as a dictionary
                results = {
                    "best_estimator": best_estimator,
                    "test_shap_values": test_shap_values,
                    "risk_scores_test": risk_scores_test,
                    "time_dependent_auc_test": time_dependent_auc_test,
                    "time_dependent_auc_mean_test": time_dependent_auc_mean_test,
                    "c_index_test": c_index_test,
                }

                joblib.dump(
                    results,
                    f"{args.out_dir}/validation_XGB-survivalCox_{out_val}_{random_seed}.joblib",
                )

                end_time = time.time()
                logger.info(
                    f"validation for {out_val} runtime: {end_time - start_time:.2f} seconds"
                )
