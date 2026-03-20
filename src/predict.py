"""
Inference for Naive Bayes & LDA — dual-dataset.
Load trained model and run predictions.
"""
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

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


def predict(data: pd.DataFrame, model_path: str = None) -> list:
    if model_path is None:
        model_path = str(MODEL_DIR / "gnb_jira.pkl")
    pipe = joblib.load(model_path)
    return pipe.predict(data).tolist()


if __name__ == "__main__":
    for key, cfg in DATASETS.items():
        df = pd.read_csv(DATA_DIR / cfg["file"])
        features = df.drop(columns=[cfg["target"]]).head(5)
        preds = predict(features, str(MODEL_DIR / cfg["model_name"]))
        print(f"{key}: {preds}")
