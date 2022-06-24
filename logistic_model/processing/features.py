from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def take(x):
            if isinstance(x, str):
                return x[0]
            else:
                return x

        X_loc = X.copy()
        for variable in self.variables:
            X_loc[variable] = X_loc[variable].apply(take)
        return X_loc
