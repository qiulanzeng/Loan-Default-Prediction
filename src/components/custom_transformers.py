from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def bin_term(X):
    X = X.copy()
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=['term'])
    X['term'] = X['term'].map(lambda x: 'long' if x == 360 else 'short')
    return X

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].map(self.freq_maps[col])
        return X_transformed

class LogCapOutlierFlagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, caps: dict):
        self.caps = caps  # {column_name: cap_value}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.caps.keys())

        all_features = []

        for col, cap in self.caps.items():
            capped = np.minimum(X[col], cap)
            log_col = np.log1p(capped)
            flag_col = (X[col] > cap).astype(int)

            log_col.name = f"{col}_log"
            flag_col.name = f"{col}_outlier_flag"

            all_features.append(log_col)
            all_features.append(flag_col)

        return pd.concat(all_features, axis=1)