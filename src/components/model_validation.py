from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class ModelValidator:
    def __init__(self):
        pass

    def evaluate(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        print("✅ Evaluation Metrics:")
        print(confusion_matrix(y_val, y_pred))
        print(classification_report(y_val, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.4f}")
