from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CrisisDetector:
    """
    Binary crisis classifier.
    Backend: sklearn (TF-IDF + LogisticRegression).
    Config keys: models.yaml -> crisis
    """
    def __init__(self, config: Dict[str, Any]):
        self.vectorizer_max_features = config.get("sklearn", {}).get("vectorizer_max_features", 20000)
        self.c = float(config.get("sklearn", {}).get("c", 1.0))
        self.pipeline: Pipeline | None = None

    def fit(self, texts: List[str], y: List[int]):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=self.vectorizer_max_features, ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=self.c, max_iter=200)),
        ])
        self.pipeline.fit(texts, y)
        logger.info("CrisisDetector trained", extra={"context": {"samples": len(texts)}})
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.pipeline:
            raise RuntimeError("CrisisDetector not fitted")
        return self.pipeline.predict_proba(texts)[:, 1]  # Probability of crisis class

    def predict(self, texts: list[str], threshold: float = 0.5) -> list[dict[str, any]]:
        probs = self.predict_proba(texts)
        results = []
        for i, score in enumerate(probs):
            text = texts[i].lower()
            label_flags = {
                "crisis": score >= threshold,
                "self_harm": any(kw in text for kw in ["hurt myself", "cut myself", "self harm"]),
                "suicide": any(kw in text for kw in ["suicide", "end my life", "kill myself"]),
                "harm": any(kw in text for kw in ["harm", "hurt", "damage", "injure"])
            }
            results.append({
                "score": float(score),
                "label": "crisis" if score >= threshold else "non-crisis",
                "flags": label_flags,
                "labels": [lbl for lbl, flag in label_flags.items() if flag]
            })
        return results
