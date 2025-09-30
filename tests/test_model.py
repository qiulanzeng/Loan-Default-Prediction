import pandas as pd
import numpy as np
from prediction import Predictor
from sklearn.metrics import accuracy_score, roc_auc_score

def test_model_predicts_on_sample():
    # Load sample data
    X = pd.read_csv("tests/sample_data/X_test_sample.csv").drop(columns=["ID", "year"])
    y_true = pd.read_csv("tests/sample_data/y_test_sample.csv")

    # Load model
    predictor = Predictor(model_path="loan_default_model_pipeline.pkl")

    # Run predictions
    y_pred = predictor.predict_class(X)
    y_proba = predictor.predict_proba(X)

    # Assertions
    assert y_pred.shape == y_true.shape, "Mismatch in predicted vs actual labels"
    assert y_proba.shape == y_true.shape, "Mismatch in predicted probabilities"
    assert set(y_pred).issubset({0, 1}), "Predictions are not binary"
    assert ((y_proba >= 0) & (y_proba <= 1)).all(), "Probabilities out of range"

    # Optional performance check
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {auc:.3f}")

    assert acc > 0.6, "Accuracy too low"
    assert auc > 0.7, "AUC too low"
