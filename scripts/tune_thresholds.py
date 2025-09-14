import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.config_loader import load_config
from src.models.abuse_detector import AbuseDetector
from src.utils.metrics import multilabel_metrics
import yaml
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/raw/train.csv")
    ap.add_argument("--out", type=str, default="reports/evaluation/thresholds.yaml")
    args = ap.parse_args()

    mcfg = load_config(["configs/models.yaml"])
    labels = mcfg["abuse"]["labels"]

    df = pd.read_csv(args.train)

    # Extract multi-label targets from binary columns
    df["abuse_labels"] = df.apply(lambda row: [label for label in labels if row[label] == 1], axis=1)

    abuse = AbuseDetector(mcfg["abuse"]).fit(df["comment_text"].tolist(), df["abuse_labels"].tolist())

    mlb = MultiLabelBinarizer(classes=labels)
    Y_true = mlb.fit_transform(df["abuse_labels"].tolist())
    Y_prob = abuse.predict_proba(df["comment_text"].tolist())

    best_thr = {}
    for i, lbl in enumerate(labels):
        candidates = np.linspace(0.2, 0.8, 13)
        best_f1, best_t = -1, 0.5
        for t in candidates:
            f1 = multilabel_metrics(Y_true[:, [i]], Y_prob[:, [i]], threshold=t)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best_thr[lbl] = best_t

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"abuse_thresholds": best_thr}, f)

    print("✅ Tuned thresholds saved to:", args.out)

if __name__ == "__main__":
    main()
