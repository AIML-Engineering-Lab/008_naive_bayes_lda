"""
Train pipeline for Naive Bayes & LDA — dual-dataset.
Project 1: Jira Bug Routing  (GaussianNB)
Project 2: Silicon Defect Classification (GaussianNB)
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = {
    "jira": {
        "file": "jira_bug_routing.csv",
        "target": "category",
        "model_name": "gnb_jira.pkl",
    },
    "silicon": {
        "file": "silicon_defect_classification.csv",
        "target": "defect_type",
        "model_name": "gnb_silicon.pkl",
    },
}


def train(key: str):
    cfg = DATASETS[key]
    print(f"\n=== Training {key} ===")
    df = pd.read_csv(DATA_DIR / cfg["file"])
    print(f"  Rows: {len(df)}")

    X = df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB()),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    model_path = MODEL_DIR / cfg["model_name"]
    joblib.dump(pipe, model_path)
    print(f"  Saved → {model_path}")
    return pipe


if __name__ == "__main__":
    for k in DATASETS:
        train(k)
