import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import KFold

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, important_numeric_features, 
                 important_low_card_features, important_high_card_features,
                 cat_option=1, model_type='tree', n_splits=5, skew_threshold=0.5,
                 dominance_threshold=0.8, high_card_threshold=20):
        self.target_col = target_col
        self.important_numeric_features = important_numeric_features
        self.important_low_card_features = important_low_card_features
        self.important_high_card_features = important_high_card_features
        self.cat_option = cat_option
        self.model_type = model_type
        self.n_splits = n_splits
        self.skew_threshold = skew_threshold
        self.dominance_threshold = dominance_threshold
        self.high_card_threshold = high_card_threshold
        
    def fit(self, X, y=None):
        df = X.copy()
        df[self.target_col] = y if y is not None else df[self.target_col]
        self.rules_ = {}
        
        # --- Iterative Imputer for important numeric features ---
        imp_numeric = IterativeImputer(random_state=42)
        imp_numeric.fit(df[self.important_numeric_features])
        self.rules_['iterative_imputer'] = imp_numeric
        
        # --- Important low-cardinality categorical ---
        ordinal_encoders = {}
        onehot_encoders = {}
        categories_mapping = {}
        for col in self.important_low_card_features:
            df[col] = df[col].fillna('UNKNOWN') if self.cat_option in [2,3] else df[col]
            if self.cat_option in [1,2]:
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                enc.fit(df[[col]])
                ordinal_encoders[col] = enc
                categories_mapping[col] = enc.categories_[0]
            elif self.cat_option == 3:
                enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
                enc.fit(df[[col]])
                onehot_encoders[col] = enc
        self.rules_['ordinal_encoders'] = ordinal_encoders
        self.rules_['onehot_encoders'] = onehot_encoders
        self.rules_['categories_mapping'] = categories_mapping
        
        # --- Important high-cardinality categorical: K-fold CV target encoding ---
        cv_target_enc = {}
        for col in self.important_high_card_features:
            encoded, mapping, global_mean = self.cv_target_encoding(df, col, self.target_col)
            df[col+'_target_enc'] = encoded
            cv_target_enc[col] = {'mapping': mapping, 'global_mean': global_mean}
        self.rules_['cv_target_enc'] = cv_target_enc
        
        # --- Other numeric features ---
        num_groups = self.categorize_numeric(df)
        other_numeric = [f for f in num_groups['symmetric']+num_groups['skewed']
                         if f not in self.important_numeric_features]
        self.rules_['other_numeric'] = other_numeric
        self.rules_['other_numeric_groups'] = num_groups
        self.rules_['mean'] = df[num_groups['symmetric']].mean()
        self.rules_['median'] = df[num_groups['skewed']].median()
        
        # --- Other categorical features ---
        cat_groups = self.categorize_categorical(df)
        other_cat_one_dominant = [f for f in cat_groups['one_dominant']
                                  if f not in self.important_low_card_features + self.important_high_card_features]
        other_cat_multi = [f for f in cat_groups['multi_frequent']
                           if f not in self.important_low_card_features + self.important_high_card_features]
        self.rules_['other_cat_one_dominant'] = other_cat_one_dominant
        self.rules_['other_cat_multi'] = other_cat_multi
        
        freq_probs = {}
        for col in other_cat_multi:
            value_counts = df[col].value_counts(normalize=True, dropna=True)
            freq_probs[col] = (value_counts.index.tolist(), value_counts.values)
        self.rules_['freq_probs'] = freq_probs
        
        if other_cat_one_dominant:
            self.rules_['mode'] = df[other_cat_one_dominant].mode().iloc[0]
        else:
            self.rules_['mode'] = {}
        
        self.rules_['drop_high_card'] = [f for f in cat_groups['high_cardinality']
                                         if f not in self.important_high_card_features]
        return self

    def transform(self, X):
        df_proc = X.copy()
        
        # --- Numeric ---
        df_proc[self.important_numeric_features] = self.rules_['iterative_imputer'].transform(
            df_proc[self.important_numeric_features])
        sym_cols = [f for f in self.rules_['other_numeric_groups']['symmetric'] if f in df_proc.columns]
        skew_cols = [f for f in self.rules_['other_numeric_groups']['skewed'] if f in df_proc.columns]
        df_proc[sym_cols] = df_proc[sym_cols].fillna(self.rules_['mean'].to_dict())
        df_proc[skew_cols] = df_proc[skew_cols].fillna(self.rules_['median'].to_dict())
        
        # --- Important low-cardinality categorical ---
        if self.cat_option in [1,2]:
            for col, enc in self.rules_['ordinal_encoders'].items():
                df_proc[[col]] = df_proc[[col]].fillna('UNKNOWN') if self.cat_option==2 else df_proc[[col]]
                df_proc[[col]] = enc.transform(df_proc[[col]])
            
            if self.cat_option == 1:
                for col, categories in self.rules_['categories_mapping'].items():
                    df_proc[col] = df_proc[col].round().astype(int)
                    df_proc[col] = df_proc[col].clip(0, len(categories)-1)
                    df_proc[col] = df_proc[col].map(lambda x: categories[x])
                    
                    if self.model_type != 'tree':
                        df_col_ohe = pd.get_dummies(df_proc[col], prefix=col)
                        df_proc = pd.concat([df_proc.drop(columns=[col]), df_col_ohe], axis=1)
                        
        elif self.cat_option == 3:
            for col, enc in self.rules_['onehot_encoders'].items():
                df_proc[[col]] = df_proc[[col]].fillna('UNKNOWN')
                encoded_cols = enc.transform(df_proc[[col]])
                col_names = [f"{col}_{cat}" for cat in enc.categories_[0]]
                df_encoded = pd.DataFrame(encoded_cols, columns=col_names, index=df_proc.index)
                df_proc = pd.concat([df_proc.drop(columns=[col]), df_encoded], axis=1)
        
        # --- Important high-cardinality categorical ---
        for col, enc_info in self.rules_['cv_target_enc'].items():
            df_proc[col+'_target_enc'] = df_proc[col].map(enc_info['mapping']).fillna(enc_info['global_mean'])
        
        # --- Other categorical ---
        if self.rules_['other_cat_one_dominant']:
            df_proc[self.rules_['other_cat_one_dominant']] = df_proc[
                self.rules_['other_cat_one_dominant']].fillna(self.rules_['mode'])
        for col, (categories, probs) in self.rules_['freq_probs'].items():
            missing_idx = df_proc[df_proc[col].isna()].index
            df_proc.loc[missing_idx, col] = np.random.choice(categories, size=len(missing_idx), p=probs)
        
        # --- Drop unimportant high-cardinality ---
        df_proc.drop(columns=self.rules_['drop_high_card'], inplace=True, errors='ignore')
        
        return df_proc

    # ------------------------
    # Helper functions
    # ------------------------
    def cv_target_encoding(self, df, col, target):
        df_encoded = pd.Series(index=df.index, dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(df):
            means = df.iloc[train_idx].groupby(col)[target].mean()
            df_encoded.iloc[val_idx] = df.iloc[val_idx][col].map(means)
        global_mean = df[target].mean()
        df_encoded = df_encoded.fillna(global_mean)
        full_mapping = df.groupby(col)[target].mean().to_dict()
        return df_encoded, full_mapping, global_mean

    def categorize_numeric(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        symmetric, skewed = [], []
        for col in numeric_cols:
            if abs(df[col].skew()) <= self.skew_threshold:
                symmetric.append(col)
            else:
                skewed.append(col)
        return {'symmetric': symmetric, 'skewed': skewed}

    def categorize_categorical(self, df):
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        one_dominant, multi_frequent, high_card = [], [], []
        for col in cat_cols:
            counts = df[col].value_counts(normalize=True, dropna=True)
            top_prop = counts.iloc[0] if len(counts) > 0 else 0
            n_unique = df[col].nunique(dropna=False)
            if n_unique >= self.high_card_threshold:
                high_card.append(col)
            elif top_prop >= self.dominance_threshold:
                one_dominant.append(col)
            else:
                multi_frequent.append(col)
        return {'one_dominant': one_dominant, 'multi_frequent': multi_frequent, 'high_cardinality': high_card}
    
if __name__=="__main__":
    preprocessor = CustomPreprocessor(
    target_col='Default',
    important_numeric_features=['Interest_rate_spread','Upfront_charges','rate_of_interest','LTV'],
    important_low_card_features=['credit_type'],
    important_high_card_features=['col_high1'],  # replace with actual names
    cat_option=1,
    model_type='linear'  # 'tree' or 'linear'
    )

    # Fit on training data
    preprocessor.fit(df_train, y=df_train['Default'])

    # Transform train/test
    df_train_proc = preprocessor.transform(df_train)
    df_test_proc = preprocessor.transform(df_test)


    import pickle

    # Save the fitted preprocessor
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Load the preprocessor
    with open("preprocessor.pkl", "rb") as f:
        loaded_preprocessor = pickle.load(f)

    # Transform test data with loaded preprocessor
    df_test_proc = loaded_preprocessor.transform(df_test)