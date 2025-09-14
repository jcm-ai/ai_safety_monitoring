from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AbuseDetector:
    """
    Multi-label abuse classifier.
    Backend: sklearn (TF-IDF + OneVsRest LogisticRegression).
    Config keys: models.yaml -> abuse
    """
    def __init__(self, config: Dict[str, Any]):
        self.labels: List[str] = config.get("labels", ["toxic", "threat", "insult", "hate", "sexual"])
        self.vectorizer_max_features = config.get("sklearn", {}).get("vectorizer_max_features", 30000)
        self.c = float(config.get("sklearn", {}).get("c", 2.0))
        self.pipeline: Pipeline | None = None
        self.mlb = MultiLabelBinarizer(classes=self.labels)

    def fit(self, texts: List[str], y_labels: List[List[str]]):
        Y = self.mlb.fit_transform(y_labels)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=self.vectorizer_max_features, ngram_range=(1, 2))),
            ("clf", OneVsRestClassifier(LogisticRegression(C=self.c, max_iter=200))),
        ])
        self.pipeline.fit(texts, Y)
        logger.info("AbuseDetector trained", extra={"context": {"labels": self.labels}})
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.pipeline:
            raise RuntimeError("AbuseDetector not fitted")
        return np.array(self.pipeline.predict_proba(texts))

    def predict(self, texts: List[str], thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        probs = self.predict_proba(texts)
        results = []
        for row in probs:
            label_scores = {lbl: float(score) for lbl, score in zip(self.labels, row)}
            label_flags = {lbl: score >= thresholds.get(lbl, 0.5) for lbl, score in label_scores.items()}
            results.append({
                "scores": label_scores,
                "labels": [lbl for lbl, flag in label_flags.items() if flag]
            })
        return results
