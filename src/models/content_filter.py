from __future__ import annotations
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ContentFilter:
    """
    Hybrid content filter for age-appropriateness.
    Combines rule-based keyword flags with a classifier.
    Config keys: models.yaml -> content_filter
    """

    def __init__(self, config: Dict[str, Any]):
        self.rules = config.get("rules", {})
        clf_cfg = config.get("classifier", {})
        self.vectorizer_max_features = clf_cfg.get("vectorizer_max_features", 15000)
        self.c = float(clf_cfg.get("c", 1.0))
        self.pipeline: Pipeline | None = None

    def rule_flags(self, text: str) -> Dict[str, bool]:
        """
        Apply rule-based keyword checks.

        Returns:
            Dict of flags: sexual, violence, substances
        """
        def has_any(key: str) -> bool:
            return any(k.lower() in text.lower() for k in self.rules.get(key, []))

        return {
            "sexual": has_any("sexual_keywords"),
            "violence": has_any("violence_keywords"),
            "substances": has_any("substances_keywords"),
        }

    def fit(self, texts: List[str], y_age_class: List[str]):
        """
        Train classifier to predict minimum age class.
        """
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=self.vectorizer_max_features, ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=self.c, max_iter=200, multi_class="auto")),
        ])
        self.pipeline.fit(texts, y_age_class)
        logger.info("ContentFilter trained", extra={"context": {"samples": len(texts)}})
        return self

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict age class and apply rule flags.

        Returns:
            List of dicts with suggested_min_age and rule_flags
        """
        preds = ["13+" for _ in texts]  # default fallback
        if self.pipeline:
            preds = list(self.pipeline.predict(texts))

        results = []
        for text, age in zip(texts, preds):
            flags = self.rule_flags(text)
            results.append({
                "suggested_min_age": age,
                "rule_flags": flags
            })
        return results
