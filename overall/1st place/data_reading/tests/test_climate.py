import pandas as pd

from wsfr_read.climate.cpc_outlooks import read_cpc_outlooks_precip, read_cpc_outlooks_temp


def test_read_cpc_outlooks():
    for reader_fn in (read_cpc_outlooks_precip, read_cpc_outlooks_temp):
        print(reader_fn.__name__)
        for year in range(2005, 2024, 2):
            issue_date = f"{year}-03-15"
            print(issue_date)
            # Full dataframe
            df = reader_fn(issue_date)
            assert df.shape[0] > 0
            assert (df.dtypes == "float64").all()
            assert df.index.dtypes.iloc[0] == "datetime64[ns]"
            assert (df.index.dtypes.iloc[1:] == "int64").all()
            assert (df.head(1).index.get_level_values("YEAR") == year - 1).all()
            assert (df.head(1).index.get_level_values("MN") == 10).all()
            assert df.index.get_level_values("issue_date").max() < pd.to_datetime(issue_date)
            for ind in df.index.names:
                if ind == "issue_date":
                    assert pd.api.types.is_datetime64_dtype(df.index.get_level_values(ind))
                else:
                    assert pd.api.types.is_integer_dtype(df.index.get_level_values(ind))
            for col in df.columns:
                assert pd.api.types.is_float_dtype(df[col])

            # For one site
            df = reader_fn(issue_date, site_id="hungry_horse_reservoir_inflow")
            assert df.shape[0] > 0
            assert (df.dtypes == "float64").all()
            assert df.index.dtypes.iloc[0] == "datetime64[ns]"
            assert (df.index.dtypes.iloc[1:] == "int64").all()
            assert (df.head(1).index.get_level_values("YEAR") == year - 1).all()
            assert (df.head(1).index.get_level_values("MN") == 10).all()
            assert df.index.get_level_values("issue_date").max() < pd.to_datetime(issue_date)
            assert set(df.index.get_level_values("CD").values) == {20, 21}
            for ind in df.index.names:
                if ind == "issue_date":
                    assert pd.api.types.is_datetime64_dtype(df.index.get_level_values(ind))
                else:
                    assert pd.api.types.is_integer_dtype(df.index.get_level_values(ind))
            for col in df.columns:
                assert pd.api.types.is_float_dtype(df[col])
