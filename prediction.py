import joblib
import pandas as pd

class Predictor:
    def __init__(self, model_path='loan_default_model_pipeline.pkl'):
        self.model = joblib.load(model_path)

    def predict_proba(self, df):
        return self.model.predict_proba(df)[:, 1]

    def predict_class(self, df, threshold=0.5):
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)


# Example usage
if __name__ == "__main__":
    X_test = pd.read_csv("data/X_test.csv")
    X_test = X_test.drop(columns=['ID', 'year'])

    print("X_test", X_test.shape)
    predictor = Predictor()
    y_pred  = predictor.predict_class(X_test)
    y_proba = predictor.predict_proba(X_test)
    print("y_pred", y_pred.shape)
    print("y_proba", y_proba.shape)


    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    y_true = pd.read_csv("data/y_test.csv")
    print("y_true", y_true.shape)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_proba))
    print(classification_report(y_true, y_pred))