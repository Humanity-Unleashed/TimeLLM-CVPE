import numpy as np
import pandas as pd
import os
from typing import List


def top_lagged_cross_correlation(df: pd.DataFrame, target_col: str, max_lag=5, top_k=10):
    scores = {}
    for col in df.columns:
        if col == target_col:
            continue
        max_corr = 0
        for lag in range(max_lag + 1):
            shifted = df[col].shift(lag)
            corr = df[target_col].corr(shifted)
            if pd.notnull(corr):
                max_corr = max(max_corr, abs(corr))
        scores[col] = max_corr

    # Sort by highest correlation and take top_k
    top_vars: List[str] = sorted(scores, key=scores.get, reverse=True)[:top_k-1]
    top_vars.sort(key=lambda x: int(x))
    return top_vars


def create_modified_data(path: str, new_path: str) -> None:
    df = pd.read_csv(path)
    datetime_col = df.columns[0]
    target_col = df.columns[-1]
    df_nums = df.select_dtypes(include=[np.number])

    top_vars = top_lagged_cross_correlation(df_nums, target_col, max_lag=10)
    top_vars = [datetime_col] + top_vars + [target_col]

    df = df[top_vars]
    df.to_csv(new_path, index=False)


if __name__ == '__main__':
    dataset_name = 'electricity'
    path = os.path.join(os.path.dirname(__file__), dataset_name, f'{dataset_name}.csv')
    new_path = os.path.join(os.path.dirname(__file__), dataset_name, f'modified_{dataset_name}.csv')
    create_modified_data(path, new_path)
