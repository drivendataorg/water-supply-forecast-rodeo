import numpy as np
import lightgbm as lgb
from src.config import cfg
from src.utils import seed_everything
from loguru import logger


def get_features(df, remove_features):
    df = df.copy()
    remove_features_ = [col for col in remove_features if col in list(df)]
    all_features = [col for col in list(df) if col not in (remove_features_)]

    return all_features


def as_categorical(df, cat_features):
    df = df.copy()
    df[cat_features] = df[cat_features].astype("category")

    return df


def train_lgb(
    df,
    quantiles=[50, 10, 90],
    params=None,
    cfg=cfg,
    is_full_train=False,
    is_print=True,
    use_log=False,
    verbose=200,
    model_output=None,
    # eval_output=None,
    pred_output=None,
    pred_iters=None,
):
    """
    Args:
        model_output: path and filename of the model output,
            e.g. f"{output_dir}/models/{EXP_NAME}_y{year}"
        eval_output: path and filename of train/valid iteration score output,
            e.g. f"{output_dir}/evals/{EXP_NAME}_y{year}"
        pred_output: path and filename of valid/test prediction output for specified iterations
            e.g. f"{output_dir}/preds/{EXP_NAME}_y{year}"
            Function will return latest value from pred_epochs list
        pred_iters: list of specified iterations
    """
    df = df.copy()
    params = params.copy()
    df = as_categorical(df, cfg["cat_features"])
    all_features = get_features(df, cfg["remove_features"])

    if use_log:
        df[cfg["target"]] = np.log1p(df[cfg["target"]])

    if is_print:
        logger.info(all_features)

    train_mask = df["cat"] == "train"
    train_data = lgb.Dataset(
        df[train_mask][all_features], label=df[train_mask][cfg["target"]]
    )
    logger.info(
        "Train data frame size: ({}, {})".format(
            len(train_mask[train_mask]), len(all_features)
        )
    )

    if not is_full_train:
        valid_mask = df["cat"] == "val"
        valid_data = lgb.Dataset(
            df[valid_mask][all_features], label=df[valid_mask][cfg["target"]]
        )
        logger.info("Validation year {}".format(df[valid_mask]["year"].unique()))
        logger.info(
            "Validation data frame size: ({}, {})".format(
                len(valid_mask[valid_mask]), len(all_features)
            )
        )

    logger.info("Start training")
    seed_everything(cfg["seed"])

    estimators = []
    df_pred = df[~train_mask][
        ["site_id"] + cfg["remove_features"] + ["month", "day"]
    ].copy()
    for quantile in quantiles:
        params.update({"alpha": quantile / 100})
        estimator = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data]
            if (not is_full_train)
            else [train_data],
            verbose_eval=verbose,
        )
        pred_iters = pred_iters or [estimator.current_iteration()]
        logger.info("Prediction at iteration: {}".format(pred_iters))
        for pred_iter in pred_iters:
            df_pred[f"pred_volume_{quantile}"] = estimator.predict(
                df[~train_mask][all_features], num_iteration=pred_iter
            )
            if use_log:
                df_pred[f"pred_volume_{quantile}"] = np.expm1(
                    df_pred[f"pred_volume_{quantile}"]
                )
                df_pred[cfg["target"]] = np.expm1(df_pred[cfg["target"]])
            if pred_output:
                df_pred.to_csv(
                    f"{pred_output}_p{quantile}_e{pred_iter}.csv", index=False
                )
        if model_output:
            estimator.save_model(f"{model_output}_p{quantile}.bin")

    estimators.append(estimator)

    return estimators, df_pred


def predict_lgb(df, quantiles, cfg, model_output):
    df = df.copy()
    df = as_categorical(df, cfg["cat_features"])
    all_features = get_features(df, cfg["remove_features"])

    df_pred = df[["site_id", "year", "month", "day", "volume", "diff"]].copy()

    for quantile in quantiles:
        mdl = lgb.Booster(model_file=f"{model_output}_p{quantile}.bin")
        df_pred[f"pred_volume_{quantile}"] = mdl.predict(df[all_features])

    return df_pred
