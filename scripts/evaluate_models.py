import argparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from src.config_loader import load_config
from src.models.abuse_detector import AbuseDetector
from src.utils.metrics import multilabel_metrics
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/raw/train.csv")
    ap.add_argument("--test", type=str, default="data/raw/test.csv")
    ap.add_argument("--test_labels", type=str, default="data/raw/test_labels.csv")
    args = ap.parse_args()

    mcfg = load_config(["configs/models.yaml"])
    labels = mcfg["abuse"]["labels"]

    # Load training data
    df = pd.read_csv(args.train)
    df["abuse_labels"] = df[labels].apply(lambda row: [lbl for lbl in labels if row[lbl] == 1], axis=1)

    # Train model
    abuse = AbuseDetector(mcfg["abuse"]).fit(df["comment_text"].tolist(), df["abuse_labels"].tolist())

    # Load test data
    test_df = pd.read_csv(args.test)
    test_labels_df = pd.read_csv(args.test_labels)
    test_df = test_df.merge(test_labels_df, on="id")

    # Filter out rows with missing labels (-1)
    valid_mask = (test_df[labels] != -1).all(axis=1)
    test_df = test_df[valid_mask]

    # Prepare ground truth and predictions
    Y_true = test_df[labels].values
    Y_prob = abuse.predict_proba(test_df["comment_text"].tolist())

    # Evaluate
    metrics = multilabel_metrics(Y_true, Y_prob)
    print("📊 Abuse Model Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
