import os
import pandas as pd
from src.data.base import *
from src.models.lgb import *


config_base = "afs_base_syc"
config_ngm = "afs_ngm_syc"
dir_main = "main"

exp_dict = {
    config_ngm: [
        "ngm_ua",
        "ngm_pdsi_ua_s51",
        "ngm_pdsi_era5_s51",
        "ngm_pdsi_ua_era5_s51",
    ],
    config_base: [
        "base_swe",
        "swe_ua",
        "pdsi_swe_s51",
        "pdsi_swe_era5",
        "pdsi_swe_era5_s51",
    ],
}
exp_list = exp_dict[config_base] + exp_dict[config_ngm]


def read_features_input(val_year=2023):
    df_features_base = pd.read_feather(
        f"runs/main/afs_base_syc/pdsi_swe_era5_s51/features/features_{val_year}.feather"
    )
    df_features_ngm = pd.read_feather(
        f"runs/main/afs_ngm_syc/ngm_pdsi_ua_era5_s51/features/features_{val_year}.feather"
    )
    df_features = pd.merge(
        df_features_base[df_features_base["year"] == val_year],
        df_features_ngm[df_features_ngm["year"] == val_year],
    )
    df_features = as_categorical(df_features, ["site_id", "rfc"])

    # Used for features input contextual
    df_features_all = pd.merge(
        df_features_base[df_features_base["year"] != val_year].query("year>=1991 & year<=2020"),
        df_features_ngm[df_features_ngm["year"] != val_year].query("year>=1991 & year<=2020"),
    )

    return df_features, df_features_all


def read_features_shap(val_year=2023, dirname="data/explainability"):
    out_file = f"{dirname}/shap_{val_year}.feather"
    if not os.path.isfile(out_file):
        df_features, _ = read_features_input(val_year)
        mdl_list = dict()
        pred_output = list()
        for model_type, exp_list in exp_dict.items():
            for exp_name in exp_list:
                mdl_list[exp_name] = dict()
                for quantile in [10, 50, 90]:
                    if quantile == 50:
                        model_output = f"runs/{dir_main}/{model_type}/{exp_name}/models/{exp_name}_reg_y{val_year}"
                    else:
                        model_output = f"runs/{dir_main}/{model_type}/{exp_name}/models/{exp_name}_y{val_year}"
                    mdl = lgb.Booster(model_file=f"{model_output}_p{quantile}.bin")
                    mdl_list[exp_name][f"q{quantile}"] = mdl
                    _pred_output = pd.DataFrame(
                        mdl.predict(df_features[mdl.feature_name()], pred_contrib=True),
                        columns=mdl.feature_name() + ["base_value"],
                    )
                    _pred_output.columns = ["shap__" + x for x in _pred_output.columns]
                    _pred_output = pd.concat([df_features[["site_id", "year", "md_id"]], _pred_output], axis=1)
                    if quantile == 50:
                        _pred_output["pred"] = mdl.predict(df_features[mdl.feature_name()], num_iteration=1200)
                    else:
                        _pred_output["pred"] = mdl.predict(df_features[mdl.feature_name()], num_iteration=500)
                    _pred_output["exp_name"] = exp_name
                    _pred_output["quantile"] = f"q{quantile}"
                    pred_output.append(_pred_output)
        pred_output = pd.concat(pred_output)
        pred_output = pred_output.reset_index(drop=True)
        pred_output = pred_output.set_index(["exp_name", "quantile", "site_id", "year", "md_id", "pred"])
        pred_output["shap__total"] = pred_output.sum(axis=1)
        pred_output["shap__total_no_base"] = pred_output["shap__total"] - pred_output["shap__base_value"]
        pred_output = pred_output.reset_index()
        os.makedirs(dirname, exist_ok=True)
        pred_output.to_feather(out_file)
    else:
        pred_output = pd.read_feather(out_file)

    return pred_output


def read_contextual_data(val_year=2023, dirname="data/explainability"):
    out_file = f"{dirname}/train_stats_{val_year}.feather"
    if not os.path.isfile(out_file):
        df_train = read_train(is_forecast=True)
        df_train = df_train[df_train["year"] != val_year]
        df_train_stats = pd.concat(
            [
                df_train.query("year>=1981 & year<=2023")
                .sort_values("volume")
                .groupby("site_id")
                .first()
                .rename(columns={"year": "year_min", "volume": "volume_min"}),
                df_train.query("year>=1981 & year<=2023")
                .sort_values("volume")
                .groupby("site_id")
                .last()
                .rename(columns={"year": "year_max", "volume": "volume_max"}),
                df_train.query("year>=1991 & year<=2020")
                .groupby("site_id")
                .agg(volume_median=("volume", "median"), volume_mean=("volume", "mean")),
            ],
            axis=1,
        ).reset_index()
        os.makedirs(dirname, exist_ok=True)
        df_train_stats.to_feather(out_file)
    else:
        df_train_stats = pd.read_feather(out_file)

    return df_train_stats


if __name__ == "__main__":
    pred_output = read_features_shap(val_year=2023)
    df_train_stats = read_contextual_data(val_year=2023)
