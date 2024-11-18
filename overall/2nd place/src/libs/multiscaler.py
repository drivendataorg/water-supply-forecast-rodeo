from typing import Dict, Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
from joblib import load, dump


class MultiScaler:
    def __init__(self):
        self.scalers : Dict[str, StandardScaler] = {}

    def __getitem__(self, site_id: str) -> StandardScaler:
        return self.scalers[site_id]

    def save_to_file(self, filename: str) -> None:
        dump(self.scalers, filename, compress=True)

    @staticmethod
    def from_file(filename: str) -> "MultiScaler":
        res = MultiScaler()
        res.scalers = load(filename)
        return res

    def fit(self, df: pd.DataFrame, site_id_column: str = "site_id", value_column: str = "volume") -> "MultiScaler":
        # self.scalers = {
        #     site_id: StandardScaler(with_mean=False).fit(group[value_column].dropna().values.reshape(-1, 1))
        #     for site_id, group in df.groupby(site_id_column)
        # }
        self.scalers = {
            site_id: MaxAbsScaler().fit([[np.percentile(group[value_column].dropna().values, 99)]])
            for site_id, group in df.groupby(site_id_column)
        }

        return self

    def _transform(
            self,
            site_id: str,
            values: Union[float, List[float], np.ndarray],
            inverse: bool = False
    ) -> Union[float, np.ndarray]:
        single_value = False
        if isinstance(values, np.ndarray):
            pass
        elif isinstance(values, list):
            values = np.array(values)
        else:
            values = np.array(values)
            single_value = True
        values = values.reshape(-1, 1)
        if inverse:
            values = self.scalers[site_id].inverse_transform(values)[:, 0]
        else:
            values = self.scalers[site_id].transform(values)[:, 0]
        if single_value:
            values = values[0]
        return values

    def transform(self, site_id: str, values: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        return self._transform(site_id, values)

    def inverse_transform(self, site_id: str, values: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        return self._transform(site_id, values, inverse=True)

    def _transform_df(
            self,
            df: pd.DataFrame,
            site_id_column: str = "site_id",
            value_column: str = "volume",
            inverse: bool = False
    ) -> pd.DataFrame:
        res_df = []
        for site_id, group in df.groupby(site_id_column):
            group = group.dropna(subset=[value_column]).copy()
            group[value_column] = self._transform(str(site_id), group[value_column].values, inverse=inverse)
            res_df.append(group)
        res_df = pd.concat(res_df)
        return res_df

    def transform_df(self, df: pd.DataFrame, site_id_column: str = "site_id", value_column: str = "volume") -> pd.DataFrame:
        return self._transform_df(df, site_id_column, value_column)

    def inverse_transform_df(self, df: pd.DataFrame, site_id_column: str = "site_id", value_column: str = "volume") -> pd.DataFrame:
        return self._transform_df(df, site_id_column, value_column, inverse=True)
