import numpy as np
import pandas as pd
from loguru import logger


def parse_time_features(df, cols=["date"], level=["year", "month", "day", "yday", "wyear", "wyday"]):
    df = df.copy()
    for col in cols:
        df[col] = pd.to_datetime(df[col])
        if "year" in level:
            df["year"] = df[col].dt.year
        if "month" in level:
            df["month"] = df[col].dt.month
        if "day" in level:
            df["day"] = df[col].dt.day
        if "yday" in level:
            df["yday"] = df[col].dt.dayofyear
        if "wyear" in level:
            df["wyear"] = df[col].dt.year.where(df[col].dt.month < 10, df[col].dt.year + 1)
        if "wyday" in level:
            df["wyday"] = (df[col] - pd.to_datetime((df["wyear"] - 1).astype("str") + "-09-30")).dt.days

    return df


def bearing_degree(lat1, lon1, lat2, lon2, earth_radius=6371):
    # https://bmanikan.medium.com/feature-engineering-all-i-learned-about-geo-spatial-features-649871d16796
    diff_lon = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(diff_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(diff_lon)
    return np.degrees(np.arctan2(y, x))


def haversine(lat1, lon1, lat2, lon2, to_radians=True, to_miles=False, earth_radius=6371):
    # https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    d = earth_radius * 2 * np.arcsin(np.sqrt(a))

    if to_miles:
        d = d / 1.609344
    return d


def combine_sites(site_a, site_b, limit=100, network="snotel"):
    site_a = site_a.copy()
    site_b = site_b.copy()
    site_a["tmp"] = 1
    site_b["tmp"] = 1
    site_comb = pd.merge(site_a, site_b, on="tmp")

    site_comb["dist"] = haversine(
        site_comb["latitude"],
        site_comb["longitude"],
        site_comb[f"{network}_latitude"],
        site_comb[f"{network}_longitude"],
    )
    site_comb = site_comb.sort_values(["site_id", "dist"])
    site_comb["rnk"] = site_comb.groupby(["site_id"]).cumcount() + 1
    site_comb = site_comb[(site_comb["rnk"] <= limit) | (site_comb["dist"] <= 50)]

    site_comb["bearing_degree"] = bearing_degree(
        site_comb["latitude"],
        site_comb["longitude"],
        site_comb[f"{network}_latitude"],
        site_comb[f"{network}_longitude"],
    )

    site_comb = site_comb.drop("tmp", axis=1)
    site_comb = site_comb.reset_index(drop=True)

    return site_comb


def generate_base_train(df, use_new_format=False):
    """
    Expand train data based on forecast date
    """
    df = df.copy()
    df_base = (
        pd.merge(
            pd.DataFrame({"month": list(range(1, 8))}).assign(a=0),
            pd.DataFrame({"day": [1, 8, 15, 22]}).assign(a=0),
        )
        .reset_index()
        .rename(columns={"index": "md_id"})
    )
    df = pd.merge(df.assign(a=0), df_base).drop(columns=["a"])
    if use_new_format:
        df = df.query('(site_id=="detroit_lake_inflow" & md_id>=24)==False')

    return df


def generate_base_year(df_meta, years=[2024]):
    df_meta = df_meta.copy()
    df_base_all = []
    for year in years:
        df_base = df_meta[["site_id"]].assign(year=year)
        df_base_all.append(df_base)
    df_base_all = pd.concat(df_base_all).reset_index(drop=True)

    return df_base_all


def generate_base_forecast(df_meta, years=[2024]):
    """
    Generate base of forecast date given df_meta (sites) and years
    """
    df_meta = df_meta.copy()
    df_base_all = []
    for year in years:
        df_base = (
            pd.merge(
                pd.DataFrame({"month": list(range(1, 8))}).assign(a=0),
                pd.DataFrame({"day": [1, 8, 15, 22]}).assign(a=0),
            )
            .reset_index()
            .rename(columns={"index": "md_id"})
        )
        df_base = pd.merge(df_meta[["site_id"]].assign(a=0, year=year), df_base).drop(columns=["a"])
        df_base_all.append(df_base)
    df_base_all = pd.concat(df_base_all).reset_index(drop=True)

    return df_base_all


def get_lag_features(
    df,
    groupby_cols,
    features,
    start_lag=2,
    lags=1,
    interval=7,
    lag_list=None,
    is_filter_day=True,
):
    # input is daily
    df = df.copy()
    if lag_list:
        lags = lag_list
    else:
        lags = range(start_lag, start_lag + (lags * interval) + 1, interval)

    lag_features = []
    for lag in lags:
        for col in features:
            _lag_col = f"{col}_lag{lag}"
            lag_features.append(_lag_col)
            df[_lag_col] = df.groupby(groupby_cols)[col].shift(lag)
    if is_filter_day:
        df = df[df["day"].isin([1, 8, 15, 22])]
    return df, lag_features


def expand_date(df, date_col="date", groupby_cols=["site_id"], max_mm_dd=None, max_ymd=None, freq=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df_base = df.groupby(groupby_cols, as_index=False).agg(
        start_date=(date_col, "min"),
        end_date=(date_col, "max"),
    )
    if max_mm_dd:
        df_base["end_date"] = df_base["end_date"].dt.strftime(f"%Y-{max_mm_dd}")
    if max_ymd:
        df_base["end_date"] = df_base["end_date"].dt.strftime(f"{max_ymd}")
    if freq:
        df_base["date"] = df_base.apply(
            lambda row: pd.date_range(row["start_date"], row["end_date"], freq=freq), axis=1
        )
    else:
        df_base["date"] = df_base.apply(lambda row: pd.date_range(row["start_date"], row["end_date"]), axis=1)
    df_base = df_base.explode("date").drop(columns=["start_date", "end_date"])
    df = pd.merge(
        df_base.assign(a=0),
        df.assign(a=0),
        how="left",
    ).drop(columns=["a"])

    return df


def generate_synthetic_data(
    df, cols=["volume"], n_synthetic=4, scale_factor=[0.5, 1.5], filter_condition='cat == "train"', seed=0, cols_neg=[]
):
    df = df.copy()
    train_year = (
        df.query(filter_condition)
        .year.value_counts()
        .to_frame("count")
        .reset_index()
        .rename(columns={"index": "year"})
    )
    np.random.seed(seed)
    scale_factor_low = scale_factor[0]
    scale_factor_high = scale_factor[1]
    scale_factor = np.random.uniform(low=scale_factor_low, high=scale_factor_high, size=(len(train_year), n_synthetic))
    scale_factor = pd.DataFrame(scale_factor)
    scale_factor.columns = [f"f{factor}" for factor in range(n_synthetic)]
    train_year = pd.concat([train_year, pd.DataFrame(scale_factor)], axis=1)
    _df_synthetic = []
    cols_pos = [x for x in cols if x not in cols_neg]
    cols_neg = [x for x in cols if x in cols_neg]
    logger.info(f"Synthetic positive features: {cols_pos}")
    logger.info(f"Synthetic negative features: {cols_neg}")
    for _, row in train_year.iterrows():
        selected_year = row["year"]
        # print(selected_year)
        for factor in range(n_synthetic):
            selected_factor = row[f"f{factor}"]
            neg_selected_factor = scale_factor_high + scale_factor_low - selected_factor
            _df = df.query("year == @selected_year").reset_index(drop=True)
            # _df.loc[:, cols] = _df.loc[:, cols] * selected_factor
            # _df.loc[:, cols_pos] = _df.loc[:, cols_pos] * selected_factor
            # _df.loc[:, cols_neg] = _df.loc[:, cols_neg] * neg_selected_factor
            _df[cols_pos] = (
                _df[cols_pos]
                .mask(_df[cols_pos] > 0, _df[cols_pos] * selected_factor)
                .mask(_df[cols_pos] < 0, _df[cols_pos] * neg_selected_factor)
            )
            if len(cols_neg) > 0:
                _df[cols_neg] = (
                    _df[cols_neg]
                    .mask(_df[cols_neg] < 0, _df[cols_neg] * selected_factor)
                    .mask(_df[cols_neg] > 0, _df[cols_neg] * neg_selected_factor)
                )
            _df_synthetic.append(_df)
    _df_synthetic = pd.concat(_df_synthetic)

    return _df_synthetic, train_year


def get_year_cat(df, val_years, test_years):
    """
    Classify year to train / val / test
    """
    val_years = val_years if type(val_years) is list else [val_years]
    test_years = test_years if type(test_years) is list else [test_years]

    df = df.copy()
    df["cat"] = np.where(df["year"].isin(val_years), "val", "train")
    df["cat"] = np.where(df["year"].isin(test_years), "test", df["cat"])

    return df


def generate_target_diff(df_monthly, df_meta):
    """
    Generate "diff" column which is the known previous month cumulative naturalized flow.
    This column will be used to calculate partial gt in month 5-7 in most of the sites
    """
    df_target_diff = (
        pd.merge(
            df_monthly.assign(month_tf=lambda x: x["month"] + 1),
            df_meta[["site_id", "season_start_month", "season_end_month"]],
        )
        .query("season_start_month+1 <= month_tf <= 7")
        .assign(diff=lambda x: x.groupby(["site_id", "year"])["volume"].transform("cumsum"))[
            ["site_id", "year", "month_tf", "diff"]
        ]
    )

    return df_target_diff


def get_month_tf(df, diff_gap=1):
    """
    Intermediary columns to join target diff
    diff_gap is maximum issue day of month which lag t-1 month data cannot be used and switch to lag t-2 month
    For example:
    - diff_gap = 0 means that we can use lag t-1 month data for all issue day of month (1, 8, 15, 22)
    - diff_gap = 1 means that we can use lag t-1 month data after issue day of month > 1 (8, 15, 22)
    - diff_gap = 8 means that we can use lag t-1 month data after issue day of month > 8 (15, 22)
    - diff_gap = 15 means that we can use lag t-1 month data after issue day of month > 15 (22)

    For the training, we assume that lag t-1 month data is available at 1st of the month
    so we use diff_gap=0 on the training config
    """

    df = df.copy()
    df = df.assign(month_tf=lambda x: np.where(x["day"] <= diff_gap, x["month"] - 1, x["month"]))

    return df


def get_volume_target_diff(df, df_diff, diff_gap=1):
    df = df.copy()
    df = pd.merge(
        get_month_tf(df, diff_gap),
        df_diff,
        how="left",
    ).assign(
        diff=lambda x: x["diff"].fillna(0),
        volume_actual=lambda x: x["volume"],
        volume=lambda x: x["volume"] - x["diff"],
        is_use_diff=lambda x: (x["diff"] > 0).astype(int),
    )

    return df


def get_day_diff(df, date_col="date", groupby_cols=["snotel_id", "wyear"]):
    df = df.copy()
    df = df.assign(
        date_lag=lambda x: x.groupby(groupby_cols)[date_col].transform("shift"),
        day_diff=lambda x: (x[date_col] - x["date_lag"]).dt.days,
    )

    return df


def convert_cfs_to_kaf(df, col):
    df = df.copy()
    df[col] = 1.9834591996927 / 1000 * df[col]

    return df


def convert_fahrenheit_to_celcius(df, col):
    df = df.copy()
    df[col] = (df[col] - 32) * 5 / 9

    return df


def forward_fill(df, groupby_cols, limit=None):
    """
    Use latest available value
    """

    df = df.copy()
    df = df.set_index(groupby_cols)
    if limit:
        df = df.groupby(groupby_cols).ffill(limit=limit)
    else:
        df = df.groupby(groupby_cols).ffill()
    df = df.reset_index()

    return df


def fill_missing_value(df, cols, groupby_cols, method="polynomial", limit=5, rounding=None):
    df = df.copy()
    try:
        df[cols] = (
            df.groupby(groupby_cols, as_index=False)[cols]
            .apply(lambda x: x.interpolate(method=method, order=2, limit=limit).clip(lower=0))
            .reset_index(drop=True)
        )

    except:
        try:
            df[cols] = (
                df.groupby(groupby_cols, as_index=False)[cols]
                .apply(lambda x: x.interpolate(method=method, order=1, limit=limit).clip(lower=0))
                .reset_index(drop=True)
            )
            logger.info("Interpolation with linear method instead of polynomial")
        except:
            df[cols] = (
                df.groupby(groupby_cols, as_index=False)[cols]
                .apply(lambda x: x.interpolate(method="linear", limit=limit).clip(lower=0))
                .reset_index(drop=True)
            )
            logger.info("Interpolation with linear method")

    if rounding:
        df[cols] = df[cols].round(rounding)

    return df


def get_meta_features(df, df_meta):
    df = df.copy()
    df = pd.merge(
        df,
        df_meta[
            [
                "site_id",
                "season_start_month",
                "season_end_month",
                "elevation",
                "latitude",
                "longitude",
                "drainage_area",
                "rfc",
            ]
        ].assign(n_months=lambda x: x["season_end_month"] - x["season_start_month"] + 1),
    )

    return df


def filter_season_month(df, df_meta):
    df = df.copy()
    df = pd.merge(df, df_meta[["site_id", "season_start_month", "season_end_month"]])
    df = df.query("season_start_month<=month<=season_end_month")
    df = df.drop(columns=["season_start_month", "season_end_month"])

    return df


def p10(x):
    return np.quantile(x, 0.1)


def p90(x):
    return np.quantile(x, 0.9)


def get_agg_features(df, groupby_cols, features, agg_cols):
    df = df.copy()
    if "p10_p90" in agg_cols:
        agg_cols = [x for x in agg_cols if x != "p10_p90"]
        agg_cols = agg_cols + [p10, p90]
    df_agg = df.groupby(groupby_cols, as_index=False).agg({k: agg_cols for k in features})
    df_agg.columns = ["_".join(col).strip() if "" not in col else col[0] for col in df_agg.columns.values]
    agg_features = [x for x in df_agg.columns if x not in groupby_cols]

    return df_agg, agg_features


def get_month_features(df, groupby_cols, features, days=[1, 8, 15, 22], days_with_prev_month=[1]):
    df = df.copy()
    df_all = []
    for day in days:
        _df = df.copy()
        if day in days_with_prev_month:
            for col in features:
                _df[col] = _df.groupby(groupby_cols)[col].shift(1)
        df_all.append(_df.assign(day=day))
    df_all = pd.concat(df_all)
    df_all = df_all.sort_values(groupby_cols + ["month", "day"]).reset_index(drop=True)

    return df_all
