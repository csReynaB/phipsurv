# ======================
# Local / project imports
# ======================
from phipsurv.io.data_handler import FeatureManager, MetadataHandler, OligosHandler


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
