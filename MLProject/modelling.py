"""
modelling.py (versi MLProject untuk Workflow CI)

Melatih ulang (re-training) model Random Forest menggunakan dataset hasil
preprocessing, dijalankan otomatis oleh MLflow Project melalui GitHub Actions
setiap kali workflow di-trigger (Kriteria 3 - Basic).
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(n_estimators: int, max_depth: int):
    max_depth = None if max_depth is None or max_depth <= 0 else max_depth

    mlflow.set_experiment("breast_cancer_classification_ci")
    mlflow.sklearn.autolog()

    train_df = pd.read_csv("breast_cancer_preprocessing/train.csv")
    test_df = pd.read_csv("breast_cancer_preprocessing/test.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    with mlflow.start_run(run_name="ci_retrain_random_forest"):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall   :", recall_score(y_test, y_pred))
        print("F1-score :", f1_score(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
