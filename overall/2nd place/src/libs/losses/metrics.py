import numpy as np
import pandas as pd


def mql(y_true: np.ndarray, y_pred: np.ndarray, quantile: float):
    return 2*(quantile*np.maximum(y_true - y_pred, 0) + (1 - quantile)*np.maximum(y_pred - y_true, 0))


def percentile_by_df(df: pd.DataFrame):
    score50 = mql(df['target'], df['volume_50'], quantile=0.5)
    score10 = mql(df['target'], df['volume_10'], quantile=0.1)
    score90 = mql(df['target'], df['volume_90'], quantile=0.9)
    score = (score50 + score10 + score90)/3
    return np.mean(score)
