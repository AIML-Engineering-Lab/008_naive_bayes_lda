"""Tests for Naive Bayes & LDA — dual-dataset."""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_jira_model_exists():
    assert (ROOT / "models" / "gnb_jira.pkl").exists()


def test_silicon_model_exists():
    assert (ROOT / "models" / "gnb_silicon.pkl").exists()


def test_jira_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "jira_bug_routing.csv")
    features = df.drop(columns=["category"]).head(3)
    preds = predict(features, str(ROOT / "models" / "gnb_jira.pkl"))
    assert len(preds) == 3


def test_silicon_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "silicon_defect_classification.csv")
    features = df.drop(columns=["defect_type"]).head(3)
    preds = predict(features, str(ROOT / "models" / "gnb_silicon.pkl"))
    assert len(preds) == 3


if __name__ == "__main__":
    test_jira_model_exists()
    test_silicon_model_exists()
    test_jira_prediction()
    test_silicon_prediction()
    print("All 4 tests passed.")
