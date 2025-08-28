import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

df = pd.read_csv("data/Loan_Default_kaggle.csv")
constant_cols = df.columns[df.nunique(dropna=False) == 1]
print("Dropped constant columns:", list(constant_cols))
df = df.drop(columns=constant_cols)
df = df.drop(columns=['ID'])
target = 'Status'
features = df.columns.drop(target)

# Features only
X = df.drop(columns=[target])

# Target only
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def categorize_categorical_features(df, dominance_threshold=0.8, high_cardinality_threshold=20):
    """
    Classify categorical features into:
    - one dominant category
    - multiple frequent categories (low-cardinality)
    - high cardinality
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        dominance_threshold (float): Top category proportion to consider "dominant"
        high_cardinality_threshold (int): Number of unique categories to consider "high cardinality"
    
    Returns:
        dict: {'one_dominant': [...], 'multi_frequent': [...], 'high_cardinality': [...]}
    """
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
categorical_features_groups = categorize_categorical_features(X_train, dominance_threshold=0.8, high_cardinality_threshold=20)
print(categorical_features_groups)


def categorize_numerical_features(df, skew_moderate=0.5, skew_high=1.0):
    """
    Categorize numeric features into symmetric, skewed, heavy-tailed,
    and apply appropriate preprocessing / imputation.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with numeric features
        skew_moderate (float): Threshold for moderate skew
        skew_high (float): Threshold for high skew
    
    Returns:
        dict: Feature categories {'symmetric': [], 'moderate_skew': [], 'heavy_skew': []}
    """
    df_numeric = df.select_dtypes(include=np.number).copy()
    feature_categories = {'symmetric': [], 'moderate_skew': [], 'heavy_skew': []}

    for col in df_numeric.columns:
        skew_val = df_numeric[col].skew()
        
        # Classify feature
        if abs(skew_val) <= skew_moderate:
            category = 'symmetric'
        elif abs(skew_val) <= skew_high:
            category = 'moderate_skew'
        else:
            category = 'heavy_skew'
        
        feature_categories[category].append(col)
    return feature_categories

numerical_features_groups = categorize_numerical_features(X_train)
print(numerical_features_groups)
def detect_outliers_iqr(df, cols):
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
numerical_features = sum(numerical_features_groups.values(), [])
categorical_features = sum(categorical_features_groups.values(), [])
iqr_outliers = detect_outliers_iqr(df, numerical_features)

def bin_term_binary(df):
    df = df.copy() 
    df["term"] = df["term"].map(lambda x: "long" if x == 360 else "short")
    return df

def log_transform_income(df, col, iqr_outliers):
    df = df.copy()  # avoids modifying slices
    upper_cap = iqr_outliers.loc[col, 'upper_cap']
    df[col] = df[col].clip(upper=upper_cap)
    df[col] = np.log1p(df[col]).astype("float64")
    return df

def add_outlier_flag(df, col, iqr_outliers):
    df = df.copy() 
    df[col+'_outlier_flag'] = np.where(
    df[col].isna(),
    0,
    (df[col] > iqr_outliers.loc[col, 'upper_cap']).astype(int))
    return df

rules_lr_nn, rules_tree = {}, {}
rules_lr_nn['preprocess'] ={'term': [bin_term_binary],
                            'LTV': [log_transform_income, add_outlier_flag], 
                            'income': [log_transform_income, add_outlier_flag], 
                            'property_value': [log_transform_income, add_outlier_flag], 
                            'Upfront_charges': [log_transform_income, add_outlier_flag], 
                            'loan_amount': [log_transform_income, add_outlier_flag]}
def apply_preprocess(df, rules, iqr_outliers=None, verbose=True):
    """
    Apply preprocessing functions to a DataFrame according to rules.

    Args:
        df: pandas DataFrame
        rules: dictionary of preprocess rules (like rules_lr_nn['preprocess'])
        iqr_outliers: DataFrame with outlier info from training data
        verbose: whether to print logging info

    Returns:
        df: processed DataFrame
    """
    for col, funcs in rules.items():
        if not isinstance(funcs, list):
            funcs = [funcs]
        
        for func in funcs:
            # Function takes only df
            if func.__code__.co_argcount == 1:
                df = func(df)
            # Function takes df, col, iqr_outliers
            elif func.__code__.co_argcount >= 2:
                if verbose and col in iqr_outliers.index:
                    upper = iqr_outliers.loc[col, 'upper_cap']
                    pct_out = iqr_outliers.loc[col, 'percent_outliers']
                    print(f"[{col}] Applying {func.__name__}: {pct_out}% outliers, upper cap={upper}")
                df = func(df, col, iqr_outliers)
            else:
                raise ValueError(f"Cannot determine arguments for function {func.__name__}")
    if verbose:
        print("Preprocessing completed.\n")
    return df

# Logistic/NN model preprocessing
X_train_lr_nn = apply_preprocess(X_train.copy(), rules_lr_nn['preprocess'], iqr_outliers)
# X_test_lr_nn  = apply_preprocess(X_test.copy(), rules_lr_nn['preprocess'], iqr_outliers)

print(X_train_lr_nn.term)


def calculate_feature_importance(X, y):
    # Copy the original dataframe
    X_temp = X.copy()
    print(X.shape)
    print(y.shape)
    # Encode categorical columns and impute missing values
    features = X_temp.columns
    for col in features:
        if X_temp[col].dtype == 'object' or X_temp[col].dtype.name == 'category':
            X_temp[col] = X_temp[col].fillna(X_temp[col].mode()[0])  # Fill with most frequent
            le = LabelEncoder()
            X_temp[col] = le.fit_transform(X_temp[col])
        else:
            X_temp[col] = X_temp[col].fillna(X_temp[col].median())  # Fill with median

    # Prepare data
 

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y, test_size=0.2, random_state=42)

    # Train RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Extract feature importances
    feature_importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # Print or display
    print(feature_importances)

    # Optional: plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importances, y='feature', x='importance')
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    plt.show()

    return feature_importances


feature_importances = calculate_feature_importance(X_train_lr_nn, y_train)