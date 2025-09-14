import os
import argparse
import pandas as pd
import joblib
import yaml
from sklearn.preprocessing import MultiLabelBinarizer
from src.config_loader import load_config
from src.models.abuse_detector import AbuseDetector
from src.models.crisis_detector import CrisisDetector
from src.utils.metrics import multilabel_metrics, binary_metrics

def extract_multilabel(df: pd.DataFrame, label_cols: list[str]) -> list[list[str]]:
    return df.apply(lambda row: [label for label in label_cols if row[label] == 1], axis=1).tolist()

def save_yaml(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/raw/train.csv")
    ap.add_argument("--out_dir", type=str, default="models/")
    ap.add_argument("--report", type=str, default="reports/evaluation/metrics.yaml")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load config and data
    mcfg = load_config(["configs/models.yaml"])
    labels = mcfg["abuse"]["labels"]
    df = pd.read_csv(args.train)

    # Train AbuseDetector
    abuse_labels = extract_multilabel(df, labels)
    abuse_model = AbuseDetector(mcfg["abuse"]).fit(df["comment_text"].tolist(), abuse_labels)
    joblib.dump(abuse_model, os.path.join(args.out_dir, "abuse_detector.joblib"))

    # Evaluate AbuseDetector
    mlb = MultiLabelBinarizer(classes=labels)
    Y_true = mlb.fit_transform(abuse_labels)
    Y_prob = abuse_model.predict_proba(df["comment_text"].tolist())
    abuse_metrics = multilabel_metrics(Y_true, Y_prob, labels=labels)

    # Train CrisisDetector (using 'toxic' as proxy or replace with real crisis column)
    crisis_model = CrisisDetector(mcfg["crisis"]).fit(df["comment_text"].tolist(), df["toxic"].tolist())
    joblib.dump(crisis_model, os.path.join(args.out_dir, "crisis_detector.joblib"))

    # Evaluate CrisisDetector
    crisis_metrics = binary_metrics(df["toxic"].tolist(), crisis_model.predict_proba(df["comment_text"].tolist()))

    # Save metrics
    save_yaml({
        "abuse": abuse_metrics,
        "crisis": crisis_metrics
    }, args.report)

    print("✅ Models trained and saved to:", args.out_dir)
    print("📊 Metrics saved to:", args.report)

if __name__ == "__main__":
    main()
