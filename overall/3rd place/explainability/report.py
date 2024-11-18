import pandas as pd
import dataframe_image as dfi
import plotly.graph_objs as go
import datetime
from pandas.api.types import CategoricalDtype
from IPython.display import display

from explainability.shap import *
from src.models.postprocess import *

import warnings

warnings.filterwarnings("ignore")

feature_mapping = {
    "base_value": "base_value",
    "month": "issue_date",
    "day": "issue_date",
    "yday": "issue_date",
    "t2m_mean": "t2m_fcst",
    "t2m_p10": "t2m_fcst",
    "t2m_p90": "t2m_fcst",
    "sd_mean": "sd_fcst",
    "sd_p10": "sd_fcst",
    "sd_p90": "sd_fcst",
    "tprate_mean": "prec_fcst",
    "tprate_p10": "prec_fcst",
    "tprate_p90": "prec_fcst",
    "swe_lag1": "gm_swe",
    "swe_lag8": "gm_swe",
    "swe_lag15": "gm_swe",
    "prec_cml_lag1": "gm_prec_cml",
    "prec_cml_lag8": "gm_prec_cml",
    "prec_cml_lag15": "gm_prec_cml",
    "uaswe_huc6_lag1": "ua_swe",
    "uaswe_huc8_lag1": "ua_swe",
    "uaswe_lag1": "ua_swe",
    "uaprec_cml_huc6_lag1": "ua_prec_cml",
    "uaprec_cml_huc8_lag1": "ua_prec_cml",
    "elevation": "static_features",
    "latitude": "static_features",
    "longitude": "static_features",
    "drainage_area": "static_features",
    "rfc": "static_features",
    "n_months": "static_features",
    "season_start_month": "static_features",
    "season_end_month": "static_features",
    "is_use_diff": "supp_features",
    "is_use_diff_extra": "supp_features",
    "lai_lv": "lai",
    "lai_hv": "lai",
    "swvl1": "swvl",
    "swvl2": "swvl",
    "swvl3": "swvl",
    "swvl4": "swvl",
    "pdsi_lag5": "pdsi",
    "snowc": "snowc",
    "asn": "asn",
    "sd": "sd",
    "t2m": "t2m",
}

feature_grouping_mapping = {
    #     "base": ["base_value"],
    "base": ["base_value", "site_id", "issue_date", "static_features", "supp_features"],
    "swe": ["gm_swe", "ua_swe", "sd"],
    "prec_cml": ["gm_prec_cml", "ua_prec_cml"],
    "drought": ["pdsi", "swvl"],
    "others": ["t2m", "snowc", "asn", "lai"],
    "fcst": ["sd_fcst", "prec_fcst", "t2m_fcst"],
}
feature_grouping = {v: k for k, l in feature_grouping_mapping.items() for v in l}

exp_code = {
    "base_swe": "a5",
    "swe_ua": "a4",
    "pdsi_swe_s51": "a3",
    "pdsi_swe_era5": "a2",
    "pdsi_swe_era5_s51": "a1",
    "ngm_ua": "b4",
    "ngm_pdsi_ua_s51": "b3",
    "ngm_pdsi_era5_s51": "b2",
    "ngm_pdsi_ua_era5_s51": "b1",
}

label_mapping = {
    "base": "Base",
    "swe": "Snowpack",
    "gm_swe": "SNOTEL/CDEC",
    "ua_swe": "UA-SWANN",
    "sd": "ERA5-Land",
    "prec_cml": "Precipitation",
    "gm_prec_cml": "SNOTEL/CDEC",
    "ua_prec_cml": "UA-SWANN",
    "drought": "Drought",
    "pdsi": "PDSI",
    "swvl": "Soil water",
    "others": "Others",
    "t2m": "Temperature",
    "snowc": "Snow cover",
    "asn": "Snow albedo",
    "lai": "Leaf area index",
    "fcst": "SEAS51 Forecast",
    "sd_fcst": " Snowpack",
    "prec_fcst": " Precipitation",
    "t2m_fcst": "Temperature",
}

group_cols = ["Base", "Snowpack", "Precipitation", "Drought", "Others", "SEAS51 Forecast"]

weight_map = {1: 0.2, 2: 0.8, 8: 0.8, 9: 0.2}


def transform_features_shap_long(df):
    df = df.copy()
    df = (
        pd.melt(
            df,
            id_vars=["exp_name", "quantile", "site_id", "year", "md_id", "pred"],
            value_vars=[x for x in df.filter(regex="shap__") if "total" not in x],
        )
        .sort_values(["site_id", "exp_name", "quantile", "md_id", "variable"])
        .query("value==value")
    )
    df["variable"] = df["variable"].str.replace("shap__", "")
    df = df.rename(columns={"value": "shap"})
    df = df.assign(
        shap_total=lambda x: x.groupby(["exp_name", "quantile", "site_id", "md_id"])["shap"].transform("sum")
    )
    df = df.rename(columns={"variable": "feature", "value": "shap"}).assign(
        feature_map=lambda x: x["feature"].map(feature_mapping).combine_first(x["feature"]),
        feature_group=lambda x: x["feature_map"].map(feature_grouping),
    )
    df = get_md_id(df)
    return df


def transform_features_shap_agg(df, groupby_cols=["feature_group"]):
    df = df.copy()
    df = (
        df.assign(site_id=lambda x: x["site_id"].astype("str"))
        .groupby(["exp_name", "quantile", "site_id", "year", "md_id"] + groupby_cols, as_index=False)
        .agg(pred=("pred", "mean"), shap_total=("shap_total", "mean"), shap=("shap", "sum"))
        .assign(shap_abs=lambda x: x["shap"].abs())
    )
    df = get_md_id(df)
    return df


def transform_features_shap_pred(df, df_features):
    df = df.copy()
    df = pd.merge(
        df_features[["site_id", "year", "md_id", "volume", "diff", "volume_max", "volume_actual"]],
        df[["exp_name", "quantile", "site_id", "year", "md_id", "pred"]],
    )
    df = add_diff(rescale(df, method="max"))
    df = df.assign(
        idx=lambda x: x.groupby(["exp_name", "site_id", "year", "md_id"])["pred"].rank(method="first", ascending=True),
        quantile_orig=lambda x: x["quantile"],
        quantile=lambda x: x["idx"].map({1: "q10", 2: "q50", 3: "q90"}),
    )
    df = df.assign(
        idx=lambda x: x.groupby(["quantile", "site_id", "year", "md_id"])["pred"].rank(method="first", ascending=True)
    )
    df = (
        pd.concat(
            [
                df.query('quantile=="q50"'),
                df.query('quantile=="q10" & idx in [1,2]'),
                df.query('quantile=="q90" & idx in [8,9]'),
            ]
        )
        .sort_values(["site_id", "year", "md_id", "quantile"])
        .reset_index(drop=True)
    )
    df["weight"] = df["idx"].map(weight_map)
    df = get_md_id(df)
    return df


def get_uncertainty_model_list(df):
    df = df.copy()
    df = pd.merge(
        df.query('quantile=="q10"')
        .sort_values("idx")
        .assign(site_id=lambda x: x["site_id"].astype("str"))
        .groupby(["site_id", "year", "md_id"], as_index=False)
        .agg({"exp_name": ",".join})
        .rename(columns={"exp_name": "exp_name_p10"}),
        df.query('quantile=="q90"')
        .sort_values("idx")
        .assign(site_id=lambda x: x["site_id"].astype("str"))
        .groupby(["site_id", "year", "md_id"], as_index=False)
        .agg({"exp_name": ",".join})
        .rename(columns={"exp_name": "exp_name_p90"}),
    )
    df = get_md_id(df)
    return df


def get_final_prediction(df, df_train_stats):
    df = df.copy()
    df = pd.concat(
        [
            df.assign(site_id=lambda x: x["site_id"].astype("str"), pred=lambda x: x["pred"] * x["weight"])
            .query('quantile=="q10"')
            .groupby(["site_id", "year", "md_id"])
            .agg(volume=("volume", "mean"), diff=("diff", "mean"), pred_volume_10=("pred", "sum")),
            df.assign(site_id=lambda x: x["site_id"].astype("str"))
            .query('quantile=="q50"')
            .groupby(["site_id", "year", "md_id"])
            .agg(pred_volume_50=("pred", "mean")),
            df.assign(site_id=lambda x: x["site_id"].astype("str"), pred=lambda x: x["pred"] * x["weight"])
            .query('quantile=="q90"')
            .groupby(["site_id", "year", "md_id"])
            .agg(pred_volume_90=("pred", "sum")),
        ],
        axis=1,
    ).reset_index()
    df = pd.merge(df, df_train_stats)
    df = get_md_id(df)
    return df


def get_quantile(array, q):
    n = len(array)
    index = (n - 1) * q
    left = int(index)
    if left == index:
        right = left
    else:
        right = left + 1
    fraction = index - left
    val = array[left] + (array[right] - array[left]) * fraction
    return val, index, left, right, fraction


def get_md_id(df, parse_date=True):
    df = df.copy()
    df_base = (
        pd.merge(
            pd.DataFrame({"month": list(range(1, 8))}).assign(a=0),
            pd.DataFrame({"day": [1, 8, 15, 22]}).assign(a=0),
        )
        .reset_index()
        .rename(columns={"index": "md_id"})
    ).drop(columns="a")
    df = pd.merge(df, df_base)
    if parse_date:
        df = df.assign(
            date=lambda x: pd.to_datetime(
                x["year"].astype(str) + "-" + x["month"].astype(str) + "-" + x["day"].astype(str)
            )
        )
    return df


def get_feature_shap_quantile(
    pred_output_long_map, pred_output_transform, site_id="skagit_ross_reservoir", issue_date="2023-07-22", quantile=50
):
    """
    Summarize all models SHAP values of a site for a single issue date and quantile
    """
    _example = pd.merge(pred_output_long_map, pred_output_transform.drop(columns=["pred"])).query(
        f'site_id=="{site_id}" & quantile=="q{quantile}"'
    )
    _example = _example[_example["date"] == issue_date]
    _example["weight"] = np.where(_example["quantile"] == "q50", 1, _example["weight"])
    _example["shap"] = _example["shap"] * _example["weight"]
    _example = (
        _example.assign(exp_name=lambda x: x["exp_name"].map(exp_code))
        .drop(columns=["quantile", "year", "md_id"])
        .pivot(index=["site_id", "feature_group", "feature_map"], columns="exp_name", values="shap")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    _example["feature_map"] = pd.Categorical(_example["feature_map"], categories=feature_grouping.keys(), ordered=True)
    _example = _example.drop(columns=["site_id"])
    _example = _example.sort_values(["feature_map"]).set_index(["feature_group", "feature_map"])

    return _example


def get_feature_contrib(
    pred_output_long_map,
    pred_output_transform,
    site_id="skagit_ross_reservoir",
    issue_date="2023-07-22",
    is_print=True,
    include_base_value=False,
    use_direction=True,
):
    """
    Summarize percentage SHAP contribution of a site for a single date and all quantiles
    """
    example_contrib = []
    example_shap = []
    for quantile in [50, 10, 90]:
        _example = get_feature_shap_quantile(
            pred_output_long_map, pred_output_transform, site_id, issue_date, quantile
        )
        if not include_base_value:
            _example = _example[1:]
        _example_contrib = 100 * np.abs(_example) / (np.abs(_example).sum(axis=0))
        _example_contrib = _example_contrib.fillna(0).mean(axis=1)
        if use_direction:
            _example_direction = _example.sum(axis=1) / np.abs(_example.sum(axis=1))
            _example_contrib = _example_contrib * _example_direction
        _example_contrib = _example_contrib.to_frame(f"q{quantile}")
        example_contrib.append(_example_contrib)
        example_shap.append(_example)
    example_contrib = pd.concat(example_contrib, axis=1)
    example_shap = pd.concat(example_shap, axis=1)
    if is_print:
        print(f"{site_id}, {issue_date}")

    return example_contrib


def get_quantile_models(pred_output_transform, site_id="skagit_ross_reservoir", issue_date="2023-07-22"):
    """
    Print list of models used for 10th and 90th percentile
    """
    q10 = " + ".join(
        pred_output_transform.query('site_id==@site_id & date==@issue_date & quantile=="q10"')
        .assign(exp_name=lambda x: x["weight"].astype(str) + "*" + x["exp_name"])
        .sort_values("idx")
        .exp_name.tolist()
    )
    q90 = " + ".join(
        pred_output_transform.query('site_id==@site_id & date==@issue_date & quantile=="q90"')
        .assign(exp_name=lambda x: x["weight"].astype(str) + "*" + x["exp_name"])
        .sort_values("idx")
        .exp_name.tolist()
    )

    print("P10 model:", q10)
    print("P90 model:", q90)


def visualize_pred(
    df_pred,
    year=2023,
    site_id="skagit_ross_reservoir",
    issue_date="2023-07-22",
    show_list=["median", "mean", "min", "max"],
    save_image=False,
    width=900,
    height=500,
):
    _res = df_pred.query("year==@year and site_id==@site_id")
    df_meta = read_meta()
    df_meta = df_meta.assign(
        season_start=lambda x: pd.to_datetime("1990-" + x["season_start_month"].astype("str") + "-1")
        .dt.month_name()
        .str.slice(stop=3),
        season_end=lambda x: pd.to_datetime("1990-" + x["season_end_month"].astype("str") + "-1")
        .dt.month_name()
        .str.slice(stop=3),
        season_month=lambda x: x["season_start"] + "-" + x["season_end"],
    )
    season_month = df_meta.query("site_id==@site_id").season_month.values[0]
    _res["diff"] = _res["diff"].mask(_res["diff"] == 0)
    _cols = [
        "pred_volume_10",
        "pred_volume_50",
        "pred_volume_90",
        "diff",
    ]
    _res[_cols] = _res[_cols].mask(_res["date"] > issue_date, np.NaN)

    _figs = []
    if "median" in show_list:
        _fig_median = go.Scatter(
            name="Volume (median)",
            x=_res["date"],
            y=_res["volume_median"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.25)", dash="dash"),
            hovertemplate="%{y:.2f}",
        )
        _figs.append(_fig_median)
    if "mean" in show_list:
        _fig_mean = go.Scatter(
            name="Volume (mean)",
            x=_res["date"],
            y=_res["volume_mean"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.25)", dash="dot"),
            hovertemplate="%{y:.2f}",
        )
        _figs.append(_fig_mean)
    if "min" in show_list:
        _fig_min = go.Scatter(
            name=f'Volume (min, {_res["year_min"].min()})',
            x=_res["date"],
            y=_res["volume_min"],
            mode="lines",
            line=dict(color="rgba(226,167,111,0.8)"),
            hovertemplate="%{y:.2f}",
        )
        _figs.append(_fig_min)
    if "max" in show_list:
        _fig_max = go.Scatter(
            name=f'Volume (max, {_res["year_max"].max()})',
            x=_res["date"],
            y=_res["volume_max"],
            mode="lines",
            line=dict(color="rgba(150,0,0,0.7)"),
            hovertemplate="%{y:.2f}",
        )
        _figs.append(_fig_max)
    _figs = _figs + [
        go.Scatter(
            name="Latest known volume",
            x=_res["date"],
            y=_res["diff"],
            mode="lines",
            line=dict(color="rgb(34,139,34)", dash="dot"),
            hovertemplate="%{y:.2f}",
        ),
        go.Scatter(
            name="Quantile 0.1 forecast",
            x=_res["date"],
            y=_res["pred_volume_10"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            showlegend=False,
            hovertemplate="%{y:.2f}",
        ),
        go.Scatter(
            name="80% prediction interval",
            x=_res["date"],
            y=_res["pred_volume_90"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            # fillcolor='#94BBCF',
            fillcolor="rgba(148, 187, 207, 0.4)",
            fill="tonexty",
            hoverinfo="skip",
        ),
        go.Scatter(
            name="Quantile 0.9 forecast",
            x=_res["date"],
            y=_res["pred_volume_90"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            showlegend=False,
            hovertemplate="%{y:.2f}",
        ),
        go.Scatter(
            name="Point forecast",
            x=_res["date"],
            y=_res["pred_volume_50"],
            mode="lines",
            line=dict(color="#384D7E"),
            hovertemplate="%{y:.2f}",
        ),
    ]

    fig = go.Figure(_figs)
    fig.update_layout(
        xaxis_title="Issue Date",
        yaxis_title="Volume (KAF)",
        hovermode="x",
        template="plotly_white",
        width=width,
        height=height,
        title=f"{site_id} - {season_month} {year} forecast",
    )
    if save_image:
        os.makedirs(f"figs/{site_id}", exist_ok=True)
        fig.write_image(f"figs/{site_id}/plot_{site_id}_{issue_date}.png", width=1000, height=500, scale=2)

    return fig


def style_negative(v, props=""):
    return props if v < 0 else None


def rounder(x):
    if isinstance(x, float):
        x = f"{x:,.2f}"
    return x


def to_positive(x):
    if isinstance(x, float):
        if x == 0:
            x = ""
        elif x > 0:
            x = f"{x:,.1f}%"
        elif x < 0:
            x = f"{abs(x):,.1f}%"
    return x


def get_contextual_features_long(df_features, df_features_all):
    """
    Get contextual features using percent of median and percent of mean in long format
    """

    lag_features = {
        k: v
        for k, v in feature_mapping.items()
        if v not in ["base_value", "issue_date", "static_features", "supp_features"]
    }
    lag_features = [x for x in lag_features.keys() if "lag8" not in x and "lag15" not in x]

    df_features_all_long = pd.melt(
        df_features_all.set_index(["site_id", "year", "md_id"])[lag_features].reset_index(),
        id_vars=["site_id", "year", "md_id"],
        value_vars=lag_features,
    )

    smr_feature_all = (
        df_features_all_long.groupby(["site_id", "md_id", "variable"])
        .agg(
            value_min=("value", "min"),
            value_median=("value", "median"),
            value_mean=("value", "mean"),
            value_max=("value", "max"),
        )
        .reset_index()
    )

    df_features_long = pd.merge(
        pd.melt(
            df_features.set_index(["site_id", "year", "md_id"])[lag_features].reset_index(),
            id_vars=["site_id", "year", "md_id"],
            value_vars=lag_features,
        ),
        smr_feature_all,
    )

    df_features_pom = df_features_long.assign(
        pmedian=lambda x: np.where(
            (x["value_median"] > 0) & (x["value_median"] != x["value_mean"]),
            100 * x["value"] / x["value_median"],
            np.NaN,
        ),
        pmean=lambda x: np.where(
            (x["value_median"] > 0) & (x["value_median"] != x["value_mean"]),
            100 * x["value"] / x["value_mean"],
            np.NaN,
        ),
        value_amedian=lambda x: x["value"] - x["value_median"],
        value_amean=lambda x: x["value"] - x["value_mean"],
    )
    df_features_pom = get_md_id(df_features_pom)

    return df_features_pom, df_features_long


def get_contextual_data(df_features_pom, site_id="skagit_ross_reservoir", issue_date="2023-07-22"):
    smr_features_pom = df_features_pom.assign(
        feature_map=lambda x: x["variable"].map(feature_mapping).combine_first(x["variable"]),
        value_shown=lambda x: np.where(
            x["feature_map"].isin(["gm_swe", "gm_prec_cml"]),
            x["pmedian"].combine_first(x["value"]),
            np.where(
                x["feature_map"].isin(["t2m_fcst", "t2m"]),
                x["value_amean"],
                np.where(
                    x["feature_map"].isin(["snowc", "lai", "pdsi"]),
                    x["value"],
                    x["pmean"].combine_first(x["value_amean"]),
                ),
            ),
        ),
        use_pom=lambda x: np.where(
            x["feature_map"].isin(["snowc", "t2m_fcst", "t2m", "lai", "pdsi"]), False, ~x["pmedian"].isna()
        ),
        value_group_median=lambda x: x.groupby(["feature_map", "site_id", "md_id"])["value_shown"].transform("median"),
        value_ratio=lambda x: x["value_shown"] / x["value_group_median"],
        value_shown_final=lambda x: np.where(
            (x["use_pom"]) & (x["value_group_median"] > 100) & (x["value_ratio"] > 2.5), np.NaN, x["value_shown"]
        ),
    ).query("site_id==@site_id & date==@issue_date")

    smr_features_pom = (
        smr_features_pom.groupby(["feature_map", "use_pom"], as_index=False)
        .agg(
            n_values=("value_shown_final", "nunique"),
            value_mean=("value_mean", "mean"),
            value_min=("value_shown_final", "min"),
            value_max=("value_shown_final", "max"),
        )
        .sort_values(["n_values", "use_pom"])
        .groupby("feature_map", as_index=False)
        .tail(1)
        .assign(
            value_min=lambda x: np.where(
                (x["use_pom"]) | (x["feature_map"] == "snowc"),
                np.where(
                    x["value_min"] < 10, x["value_min"].map("{:.1f}%".format), x["value_min"].map("{:.0f}%".format)
                ),
                x["value_min"].map("{:.2f}".format),
            ),
            value_max=lambda x: np.where(
                (x["use_pom"]) | (x["feature_map"] == "snowc"),
                np.where(
                    x["value_max"] < 10, x["value_max"].map("{:.1f}%".format), x["value_max"].map("{:.0f}%".format)
                ),
                x["value_max"].map("{:.2f}".format),
            ),
            value_mean=lambda x: np.where(
                (x["use_pom"]) | (x["feature_map"] == "snowc"),
                np.where(
                    x["value_mean"] < 10, x["value_mean"].map("{:.1f}%".format), x["value_mean"].map("{:.0f}%".format)
                ),
                x["value_mean"].map("{:.2f}".format),
            ),
            val=lambda x: np.where(
                x["n_values"] == 1,
                x["value_min"],
                np.where(
                    x["feature_map"] == "t2m_fcst",
                    x["value_min"] + "; " + x["value_max"],
                    x["value_min"] + "-" + x["value_max"],
                ),
            ),
        )
        .assign(val=lambda x: np.where(x["feature_map"] == "snowc", x["val"] + " (" + x["value_mean"] + ")", x["val"]))
        .set_index("feature_map")[["val"]]
    )

    return smr_features_pom


def get_feature_explanation(
    pred_output_long_map,
    pred_output_transform,
    df_features_pom,
    site_id="skagit_ross_reservoir",
    issue_date="2023-07-22",
    is_print=True,
    include_base_value=False,
    use_direction=True,
):
    _res = get_feature_contrib(
        pred_output_long_map, pred_output_transform, site_id, issue_date, is_print, include_base_value, use_direction
    )
    _res_ctx = get_contextual_data(df_features_pom, site_id, issue_date)
    _res = pd.merge(_res.reset_index(), _res_ctx.reset_index(), how="left").set_index(["feature_group", "feature_map"])
    _res = _res[["val", "q50", "q10", "q90"]]
    _res_total = (
        np.abs(_res.drop(columns=["val"]))
        .groupby("feature_group", sort=False)
        .sum()
        .reset_index()
        .assign(feature_map="total")
    )
    _res_total = _res_total.mask(_res_total == 0, np.NaN)
    _cat = CategoricalDtype(categories=feature_grouping_mapping.keys(), ordered=True)
    _cat_all = CategoricalDtype(categories=["total"] + _res.reset_index().feature_map.tolist(), ordered=True)
    _res_final = pd.concat([_res.reset_index(), _res_total])
    _res_final["feature_group"] = _res_final["feature_group"].astype(_cat)
    _res_final["feature_map"] = _res_final["feature_map"].astype(_cat_all)
    _res_total["feature_group"] = _res_total["feature_group"].astype(_cat)
    _res_total = _res_total.sort_values("feature_group").reset_index(drop=True)

    return _res_final, _res_total


def get_forecast(df_pred, site_id="skagit_ross_reservoir", issue_date="2023-07-22", print_gt=True):
    _df_pred = df_pred.query("site_id==@site_id & date==@issue_date")
    if print_gt:
        print(f"Actual volume: {_df_pred.volume.values[0]:.2f}")
    _res = (
        pd.concat(
            [
                _df_pred.filter(like="pred").assign(idx="forecast"),
                (_df_pred.filter(like="pred") / _df_pred.volume_mean.values[0]).assign(idx="percent_mean"),
                (_df_pred.filter(like="pred") / _df_pred.volume_median.values[0]).assign(idx="percent_median"),
            ]
        )
        .set_index("idx")
        .T.style.format(
            {
                "forecast": "{:,.2f}".format,
                "percent_mean": "{:,.0%}".format,
                "percent_median": "{:,.0%}".format,
            }
        )
    )
    return _res


def get_feature_global(pred_output_long_map, pred_output_transform, drop_cols=["feature_group"]):
    _example = pd.merge(
        pred_output_long_map, pred_output_transform[["site_id", "date", "exp_name", "quantile", "weight"]]
    )
    _example["weight"] = np.where(_example["quantile"] == "q50", 1, _example["weight"])
    _example["shap"] = _example["shap"] * _example["weight"]
    # _example = _example.query('quantile=="q10"')
    _example = (
        _example.assign(exp_name=lambda x: x["exp_name"].map(exp_code))
        .drop(columns=["year", "md_id"])
        .pivot(
            index=["quantile", "site_id", "date", "feature_group", "feature_map"], columns="exp_name", values="shap"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    _example["feature_map"] = pd.Categorical(_example["feature_map"], categories=feature_grouping.keys(), ordered=True)
    _example = _example.sort_values(["quantile", "site_id", "date", "feature_map"]).set_index(
        ["quantile", "site_id", "date", "feature_group", "feature_map"]
    )
    _example = _example.query('feature_map != "base_value"')

    _example_contrib_all = (
        100
        * np.abs(_example)
        / np.abs(_example).groupby(["quantile", "site_id", "date"])[list(_example)].transform("sum")
    )
    _example_contrib_all = _example_contrib_all.fillna(0).mean(axis=1)
    _example_contrib_all = _example_contrib_all.to_frame("feature_contrib")
    _example_contrib_all = _example_contrib_all.assign(
        feature_contrib=lambda x: 100
        * x["feature_contrib"]
        / x.groupby(["quantile", "site_id", "date"])["feature_contrib"].transform("sum")
    )
    _example_contrib_global = _example_contrib_all.groupby(["quantile", "date", "feature_group", "feature_map"]).mean()
    _example_contrib_global = _example_contrib_global.assign(
        feature_contrib=lambda x: 100
        * x["feature_contrib"]
        / x.groupby(["quantile", "date"])["feature_contrib"].transform("sum")
    )
    _example_contrib_global = (
        _example_contrib_global.reset_index()
        .pivot(index=["date", "feature_group", "feature_map"], columns="quantile", values="feature_contrib")
        .reset_index()
        .rename_axis(None, axis=1)
        .dropna(subset=["q10", "q50", "q90"], how="any")
    )
    _example_contrib_global = _example_contrib_global.sort_values(["date", "feature_map"])
    _example_contrib_global = _example_contrib_global.set_index(["date", "feature_group", "feature_map"])
    _example_contrib_global = _example_contrib_global[["q50", "q10", "q90"]]
    _example_contrib_global_total = (
        _example_contrib_global.groupby(["date", "feature_group"], sort=False)
        .sum()
        .reset_index()
        .assign(feature_map="total")
    )
    _example_contrib_global_total = _example_contrib_global_total.sort_values(["date", "feature_map"])

    _cat = CategoricalDtype(
        categories=_example_contrib_global_total[["feature_group"]].drop_duplicates().feature_group.tolist(),
        ordered=True,
    )
    _cat_all = CategoricalDtype(
        categories=["total"]
        + _example_contrib_global.reset_index()[["feature_map"]].drop_duplicates().feature_map.tolist(),
        ordered=True,
    )
    _example_contrib_final = pd.concat([_example_contrib_global.reset_index(), _example_contrib_global_total])
    _example_contrib_final["feature_group"] = _example_contrib_final["feature_group"].astype(_cat)
    _example_contrib_final["feature_map"] = _example_contrib_final["feature_map"].astype(_cat_all)

    # Outer step
    _example_contrib_final = _example_contrib_final[
        ~_example_contrib_final["feature_map"].isin(["site_id", "issue_date", "static_features", "supp_features"])
    ]
    _example_contrib_final = _example_contrib_final.sort_values(["feature_group", "feature_map"]).reset_index(
        drop=True
    )
    _example_contrib_final = _example_contrib_final.assign(
        feature_group=lambda x: x["feature_group"].map(label_mapping),
        _temp="Total (absolute)",
        feature_map=lambda x: x["feature_map"].map(label_mapping).combine_first(x["feature_group"]),
    ).drop(columns=["_temp"] + drop_cols)

    return _example_contrib_final


def generate_report(
    pred_output_long_map,
    pred_output_transform,
    df_features_pom,
    df_pred,
    site_id="skagit_ross_reservoir",
    issue_date="2023-07-22",
    save_tbl_img=True,
    save_plot_img=False,
    lag_week=1,
):
    get_quantile_models(pred_output_transform, site_id, issue_date)
    print("")
    display(get_forecast(df_pred, site_id=site_id, issue_date=issue_date))
    _res_final, _res_total = get_feature_explanation(
        pred_output_long_map, pred_output_transform, df_features_pom, site_id, issue_date
    )
    _res_final = _res_final[
        ~_res_final["feature_map"].isin(["site_id", "issue_date", "static_features", "supp_features"])
    ]
    _res_final = _res_final.sort_values(["feature_group", "feature_map"]).reset_index(drop=True)
    _example_contrib_final = get_feature_global(pred_output_long_map, pred_output_transform)
    _rel = np.abs(_res_final[["q50", "q10", "q90"]]) / _example_contrib_final.query("date==@issue_date")[
        ["q50", "q10", "q90"]
    ].reset_index(drop=True)
    _res_final = pd.concat([_res_final, _rel[["q50"]]], axis=1)

    prev_issue_date = (pd.to_datetime(issue_date) - datetime.timedelta(weeks=lag_week)).strftime("%Y-%m-%d")
    _res_final_prev, _res_total_prev = get_feature_explanation(
        pred_output_long_map, pred_output_transform, df_features_pom, site_id, prev_issue_date
    )
    _res_final_prev.columns = [x + "_prev" if "feature" not in x else x for x in _res_final_prev.columns]
    _res_total_prev.columns = [x + "_prev" if "feature" not in x else x for x in _res_total_prev.columns]
    _res_final = pd.merge(_res_final, _res_final_prev)
    _res_total = pd.merge(_res_total, _res_total_prev)

    _res_final = _res_final.assign(
        feature_group=lambda x: x["feature_group"].map(label_mapping),
        _temp="Total (absolute)",
        feature_map=lambda x: x["feature_map"].map(label_mapping).combine_first(x["feature_group"]),
    ).drop(columns=["_temp", "feature_group"])

    summary_cols = [
        (" ", " ", "Feature"),
        (f"Issue Date: {issue_date}", "", "Value"),
        (f"Issue Date: {issue_date}", "% Feature contribution", "Q0.5"),
        (f"Issue Date: {issue_date}", "% Feature contribution", "Q0.1"),
        (f"Issue Date: {issue_date}", "% Feature contribution", "Q0.9"),
        (" ", " ", "Rel"),
    ]

    if lag_week:
        lag_summary_cols = [
            (f"Previous Issue Date: {prev_issue_date}", "", "Value"),
            (f"Previous Issue Date: {prev_issue_date}", "% Feature contribution", "Q0.5"),
            (f"Previous Issue Date: {prev_issue_date}", "% Feature contribution", "Q0.1"),
            (f"Previous Issue Date: {prev_issue_date}", "% Feature contribution", "Q0.9"),
        ]
        summary_cols = summary_cols + lag_summary_cols
    _res_final.columns = pd.MultiIndex.from_tuples(summary_cols)

    print("\n SHAP feature contribution")
    _fc_cols = [col for col in _res_final.columns if "% Feature contribution" in col]
    _val_cols = [col for col in _res_final.columns if "Value" in col or "Rel" in col]
    _tbl = (
        _res_final.style.applymap(style_negative, subset=_fc_cols, props="color:#c79854")
        .format(to_positive, subset=_fc_cols, na_rep="")
        .format(rounder, subset=_val_cols, na_rep="")
        .bar(
            subset=(_res_final[~_res_final[(" ", " ", "Feature")].isin(group_cols)].index, _fc_cols),
            align=0,
            height=50,
            vmin=_res_final.select_dtypes(include="number").min().min(),
            vmax=_res_final.select_dtypes(include="number").max().max(),
            width=50,
            props="width: 50px;",
            color=["#DABB8E", "#1ca3ec"],
        )
        .set_table_styles(
            [
                {"selector": "th.col_heading.level0", "props": [("text-align", "center")]},
                {"selector": "th.col_heading.level1", "props": [("text-align", "center")]},
                {"selector": "th.col_heading.level2", "props": [("text-align", "center")]},
                {"selector": "tr", "props": "line-height: 21px;"},
                {"selector": "td,th", "props": "line-height: inherit; padding-top: 0; padding-bottom: 0;"},
            ]
        )
        .set_properties(
            subset=[(" ", " ", "Feature")],
            **{
                "text-align": "left",
                "font-weight": "bold",
                "border-right": "1px solid #D3D3D3",
            },
        )
        .set_properties(
            subset=[lag_summary_cols[0]],
            **{
                "border-left": "1px solid #D3D3D3",
            },
        )
        .set_properties(
            subset=(_res_final[~_res_final[(" ", " ", "Feature")].isin(group_cols)].index, _res_final.columns[0]),
            **{"text-indent": "1em", "font-weight": "normal"},
        )
        .set_properties(
            subset=(_res_final[_res_final[(" ", " ", "Feature")].isin(group_cols)].index, _res_final.columns),
            **{"font-weight": "bold", "border-top": "1px solid #D3D3D3"},
        )
        .apply_index(
            lambda x: ["border-right: 1px solid #D3D3D3;" if v in [" ", "Feature", "Rel"] else "" for v in x],
            axis="columns",
            level=[0, 1, 2],
        )
        .hide()
    )
    display(_tbl)
    if save_tbl_img:
        os.makedirs(f"figs/{site_id}", exist_ok=True)
        dfi.export(_tbl, f"figs/{site_id}/tbl_{site_id}_{issue_date}.png", table_conversion="selenium", dpi=200)

    print("\nAggregated SHAP feature contribution")
    display(
        _res_total.drop(columns="feature_map")
        .set_index("feature_group")
        .style.format(to_positive, na_rep="")
        .highlight_max(subset=["q50", "q10", "q90"], props="font-weight:bold")
    )
    if save_plot_img:
        display(visualize_pred(df_pred, site_id=site_id, issue_date=issue_date, save_image=True))
    else:
        display(visualize_pred(df_pred, site_id=site_id, issue_date=issue_date))


if __name__ == "__main__":
    df_features, df_features_all = read_features_input(val_year=2023)
    df_train_stats = read_contextual_data(val_year=2023)
    pred_output = read_features_shap(val_year=2023)
    pred_output_long = transform_features_shap_long(pred_output)
    pred_output_long_map = transform_features_shap_agg(pred_output_long, groupby_cols=["feature_group", "feature_map"])
    pred_output_long_agg = transform_features_shap_agg(pred_output_long, groupby_cols=["feature_group"])
    pred_output_transform = transform_features_shap_pred(pred_output, df_features)
    df_model_unc = get_uncertainty_model_list(pred_output_transform)
    df_pred = get_final_prediction(pred_output_transform, df_train_stats)

    # Convert total precipitation rate to be more interpretable
    tprate_cols = list(df_features.filter(like="tprate"))
    df_features[tprate_cols] = df_features[tprate_cols] * 24 * 3600 * 1000
    df_features_all[tprate_cols] = df_features_all[tprate_cols] * 24 * 3600 * 1000

    df_features_pom, df_features_long = get_contextual_features_long(df_features, df_features_all)

    for site_id in ["owyhee_r_bl_owyhee_dam", "pueblo_reservoir_inflow"]:
        for issue_date in ["2023-03-15", "2023-05-15"]:
            generate_report(
                pred_output_long_map,
                pred_output_transform,
                df_features_pom,
                df_pred,
                site_id=site_id,
                issue_date=issue_date,
                save_tbl_img=True,
                save_plot_img=True,
            )
