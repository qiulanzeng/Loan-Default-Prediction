# data_preprocessing.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.components.custom_transformers import bin_term, FrequencyEncoder, LogCapOutlierFlagTransformer


class DataPreprocessing:
    def __init__(self, config=None):
        self.config = config

    def categorize_categorical_features(self, df, dominance_threshold=0.8, high_cardinality_threshold=20):
        one_dominant = []
        multi_frequent = []
        high_cardinality = []

        cat_cols = df.select_dtypes(include='object').columns

        for col in cat_cols:
            n_unique = df[col].nunique(dropna=False)
            counts = df[col].value_counts(normalize=True, dropna=False)
            top_prop = counts.iloc[0]

            if n_unique >= high_cardinality_threshold:
                high_cardinality.append(col)
            elif top_prop >= dominance_threshold:
                one_dominant.append(col)
            else:
                multi_frequent.append(col)

        return {
            'one_dominant': one_dominant,
            'multi_frequent': multi_frequent,
            'high_cardinality': high_cardinality
        }

    def categorize_numerical_features(self, df, skew_moderate=0.5, skew_high=1.0):
        df_numeric = df.select_dtypes(include=np.number).copy()
        feature_categories = {'symmetric': [], 'moderate_skew': [], 'heavy_skew': []}

        for col in df_numeric.columns:
            skew_val = df_numeric[col].skew()
            if abs(skew_val) <= skew_moderate:
                feature_categories['symmetric'].append(col)
            elif abs(skew_val) <= skew_high:
                feature_categories['moderate_skew'].append(col)
            else:
                feature_categories['heavy_skew'].append(col)
        return feature_categories

    def detect_outliers_iqr(self, df, cols):
        outlier_dict = {}
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_dict[col] = {
                'num_outliers': outliers.shape[0],
                'percent_outliers': round(outliers.shape[0] / df.shape[0] * 100, 2),
                'upper_cap': int(upper),
                'lower_cap': int(lower),
                'minimum': int(df[col].min()), 
                'maximum': int(df[col].max())
            }
        return pd.DataFrame(outlier_dict).T.sort_values(by='percent_outliers', ascending=False)

    def drop_invalid_LTV(self, X, y, threshold=250):
        original_count = len(X)
        mask = X['LTV'] > threshold
        filtered_X = X.loc[~mask].copy()
        filtered_y = y.loc[~mask].copy()
        dropped_count = original_count - len(filtered_X)
        print(f"Dropped {dropped_count} rows where LTV > {threshold}")
        return filtered_X, filtered_y

    def calculate_feature_importance(self, X, y):
        X_temp = X.copy()
        features = X_temp.columns

        for col in features:
            if X_temp[col].dtype == 'object':
                X_temp[col] = X_temp[col].fillna(X_temp[col].mode()[0])
                le = LabelEncoder()
                X_temp[col] = le.fit_transform(X_temp[col])
            else:
                X_temp[col] = X_temp[col].fillna(X_temp[col].median())

        X_train, X_val, y_train, y_val = train_test_split(X_temp, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        feature_importances = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values(by='importance', ascending=False)

        # plt.figure(figsize=(12, 8))
        # sns.barplot(data=feature_importances, y='feature', x='importance')
        # plt.title('Random Forest Feature Importances')
        # plt.tight_layout()
        # plt.show()

        return feature_importances

    def build_preprocessor(self, X_train, y_train):
        X_train = X_train.drop(columns=['ID', 'year'])
        X_train = X_train.drop_duplicates()
        
        categorical_features_groups = self.categorize_categorical_features(X_train)
        numerical_features_groups = self.categorize_numerical_features(X_train)
        numerical_features = sum(numerical_features_groups.values(), [])
        categorical_features = sum(categorical_features_groups.values(), [])

        iqr_outliers = self.detect_outliers_iqr(X_train, numerical_features)
        X_train, y_train = self.drop_invalid_LTV(X_train, y_train, threshold=1000)

        log_outlier_cols = ['LTV', 'income', 'property_value', 'Upfront_charges', 'loan_amount']
        caps_dict = {col: iqr_outliers.loc[col, 'upper_cap'] for col in log_outlier_cols}

        feature_importances = self.calculate_feature_importance(X_train, y_train)
        high_importance_features = feature_importances[feature_importances['importance'] >= 0.05]['feature'].tolist()

        exclude_cols = log_outlier_cols + ['term']
        filtered_high_imp_numeric = [f for f in high_importance_features if f in numerical_features and f not in exclude_cols]
        filtered_symmetric = [f for f in numerical_features_groups['symmetric'] if f not in filtered_high_imp_numeric and f not in exclude_cols]
        filtered_moderate_skew = [f for f in numerical_features_groups['moderate_skew'] if f not in filtered_high_imp_numeric and f not in exclude_cols]
        filtered_heavy_skew = [f for f in numerical_features_groups['heavy_skew'] if f not in filtered_high_imp_numeric and f not in exclude_cols]

        filtered_one_dominant = [f for f in categorical_features_groups['one_dominant'] if f not in exclude_cols]
        filtered_multi_frequent = [f for f in categorical_features_groups['multi_frequent'] if f not in exclude_cols]

        high_imp_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        symmetric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        moderate_skew_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        heavy_skew_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        one_dominant_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        freq_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('freq',  OneHotEncoder(handle_unknown='ignore'))#FrequencyEncoder())
        ])

        term_pipeline = Pipeline(steps=[
            ('bin', FunctionTransformer(bin_term, validate=False)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        log_outlier_pipeline = Pipeline(steps=[
            ('log_outlier_flag', LogCapOutlierFlagTransformer(caps=caps_dict)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('term_pipe', term_pipeline, ['term']),
            ('log_flag', log_outlier_pipeline, log_outlier_cols),
            ('high_imp_num', high_imp_transformer, filtered_high_imp_numeric),
            ('symmetric_num', symmetric_transformer, filtered_symmetric),
            ('mod_skew_num', moderate_skew_transformer, filtered_moderate_skew),
            ('heavy_skew_num', heavy_skew_transformer, filtered_heavy_skew),
            ('one_dom_cat', one_dominant_transformer, filtered_one_dominant),
            ('multi_freq_cat', freq_transformer, filtered_multi_frequent)
        ])

        return preprocessor, X_train, y_train
