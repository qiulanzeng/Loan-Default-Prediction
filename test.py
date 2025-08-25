import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('data/Loan_Default_kaggle.csv')
df = df.drop(columns=['ID', 'year'])
# Basic info
print(df.shape)
print(df.isnull().mean().sort_values(ascending=False))

# --------------------
# MCAR test (approx): Chi-square test between missingness and other features
# --------------------
def test_mcar(df):
    results = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df_temp = df.copy()
            df_temp['missing'] = df[col].isnull().astype(int)
            for other_col in df.columns:
                if other_col != col and df_temp[other_col].notnull().all():
                    try:
                        if df_temp[other_col].dtype == 'object':
                            cont_table = pd.crosstab(df_temp['missing'], df_temp[other_col])
                        else:
                            df_temp['bin'] = pd.qcut(df_temp[other_col], q=4, duplicates='drop')
                            cont_table = pd.crosstab(df_temp['missing'], df_temp['bin'])
                        chi2, p, dof, ex = chi2_contingency(cont_table)
                        results.append((col, other_col, p))
                    except Exception as e:
                        continue
    mcar_df = pd.DataFrame(results, columns=['MissingCol', 'TestedAgainst', 'p_value'])
    return mcar_df[mcar_df['p_value'] < 0.05]  # significant dependencies

# --------------------
# MAR test: Logistic regression of missingness ~ other features
# --------------------
def test_mar(df):
    results = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df_temp = df.copy()
            df_temp = df_temp.dropna(subset=[c for c in df.columns if c != col])
            if df_temp.shape[0] < 100: continue

            df_temp['missing'] = df[col].isnull().astype(int)

            # Simple numeric encoding for categorical
            X = df_temp.drop(columns=[col, 'missing'])
            for c in X.select_dtypes(include='object').columns:
                X[c] = LabelEncoder().fit_transform(X[c].astype(str))

            y = df_temp['missing']
            try:
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                score = model.score(X, y)
                results.append((col, score))
            except:
                continue
    mar_df = pd.DataFrame(results, columns=['Col', 'LogisticAccuracy'])
    return mar_df[mar_df['LogisticAccuracy'] > 0.6]  # high predictive power → MAR

# --------------------
# MNAR test (heuristic): Compare distributions between missing and non-missing
# --------------------
def test_mnar(df):
    results = []
    for col in df.columns:
        if df[col].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[col]):
            missing = df[df[col].isnull()]
            observed = df[df[col].notnull()]
            for other_col in df.select_dtypes(include=np.number).columns:
                if other_col == col: continue
                try:
                    stat, p = ttest_ind(missing[other_col].dropna(), observed[other_col].dropna(), equal_var=False)
                    results.append((col, other_col, p))
                except:
                    continue
    mnar_df = pd.DataFrame(results, columns=['MissingCol', 'TestedAgainst', 'p_value'])
    return mnar_df[mnar_df['p_value'] < 0.05]

# --------------------
# Run tests
# --------------------
print("\n--- MCAR Test ---")
mcar_results = test_mcar(df)
print(mcar_results)

print("\n--- MAR Test ---")
mar_results = test_mar(df)
print(mar_results)

print("\n--- MNAR Test ---")
mnar_results = test_mnar(df)
print(mnar_results)