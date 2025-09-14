from __future__ import annotations
import os
import joblib
from typing import Dict, Any
from src.preprocessing.language_detection import detect_language
from src.preprocessing.text_normalization import normalize_text
from src.preprocessing.pii_masking import mask_pii
from src.models.abuse_detector import AbuseDetector
from src.models.crisis_detector import CrisisDetector
from src.models.escalation_tracker import EscalationTracker
from src.models.content_filter import ContentFilter
from src.policy_engine.policy_decision import PolicyEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceOrchestrator:
    """
    Central orchestrator for real-time safety inference.
    Combines preprocessing, model inference, and policy decision.
    """

    def __init__(self, configs: Dict[str, Any]):
        self.pre_cfg = configs.get("preprocessing", {})
        self.models_cfg = configs.get("models", {})
        self.policy_cfg = configs.get("policy", {})
        self.ui_cfg = configs.get("ui", {})

        # Initialize models
        self.abuse = None
        self.crisis = None
        self.escalation = EscalationTracker(**self.models_cfg.get("escalation", {}))
        self.content_filter = ContentFilter(self.models_cfg.get("content_filter", {}))
        self.policy = PolicyEngine(self.policy_cfg)

        self._trained = False

    def load_models_from_disk(self, model_dir: str = "models/"):
        """
        Load trained models from disk.
        """
        try:
            self.abuse = joblib.load(os.path.join(model_dir, "abuse_detector.joblib"))
            self.crisis = joblib.load(os.path.join(model_dir, "crisis_detector.joblib"))
            self._trained = True
            logger.info("✅ Models loaded from disk")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load models from disk: {e}")
            self.load_or_fit_minimal()

    def load_or_fit_minimal(self):
        """
        Fit models with minimal dummy data if not already trained.
        Used as fallback when disk loading fails.
        """
        if self._trained:
            return
        texts = ["hello friend", "i will hurt you", "need help i want to die", "let's watch a movie"]
        abuse_labels = [["toxic"], ["threat"], ["toxic"], []]
        crisis_labels = [0, 0, 1, 0]
        age_labels = ["7+", "16+", "13+", "7+"]
        self.abuse = AbuseDetector(self.models_cfg.get("abuse", {})).fit(texts, abuse_labels)
        self.crisis = CrisisDetector(self.models_cfg.get("crisis", {})).fit(texts, crisis_labels)
        self.content_filter.fit(texts, age_labels)
        self._trained = True
        logger.info("🧪 Orchestrator models fitted with minimal data")

    def preprocess(self, text: str) -> Dict[str, Any]:
        lang_code, lang_conf = ("en", 1.0)
        if self.pre_cfg.get("language_detection", {}).get("enabled", True):
            lang_code, lang_conf = detect_language(text)

        s = normalize_text(
            mask_pii(
                text,
                mask_email=self.pre_cfg.get("pii_masking", {}).get("mask_email", True),
                mask_phone=self.pre_cfg.get("pii_masking", {}).get("mask_phone", True),
                email_token=self.pre_cfg.get("pii_masking", {}).get("email_token", "<EMAIL>"),
                phone_token=self.pre_cfg.get("pii_masking", {}).get("phone_token", "<PHONE>"),
            ),
            lower=self.pre_cfg.get("normalization", {}).get("lower", True),
            strip_urls=self.pre_cfg.get("normalization", {}).get("strip_urls", True),
            strip_punctuation=self.pre_cfg.get("normalization", {}).get("strip_punctuation", True),
            collapse_whitespace=self.pre_cfg.get("normalization", {}).get("collapse_whitespace", True),
            unicode_nfkc=self.pre_cfg.get("normalization", {}).get("unicode_nfkc", True),
        )
        return {"text": s, "lang": lang_code, "lang_conf": lang_conf}

    def infer(self, text: str, age: str) -> Dict[str, Any]:
        if not self._trained:
            self.load_models_from_disk()

        pre = self.preprocess(text)
        texts = [pre["text"]]

        abuse_thr = self.policy_cfg["thresholds"]["abuse"]
        abuse_out = self.abuse.predict(texts, thresholds=abuse_thr)[0]

        crisis_thr = self.policy_cfg["thresholds"]["crisis"]
        crisis_out = self.crisis.predict(texts, threshold=crisis_thr)[0]

        content_out = self.content_filter.predict(texts)[0]

        max_risk = max([crisis_out["score"], *abuse_out["scores"].values()])
        esc = self.escalation.update(max_risk)

        decision = self.policy.decide(
            age=age,
            abuse=abuse_out["scores"],
            crisis=crisis_out["score"],
            escalation={"ewma": esc["ewma"], "slope": esc["slope"]},
            content_flags=content_out["rule_flags"],
            crisis_labels=crisis_out["labels"]
        )

        return {
            "input": {"raw": text, "preprocessed": pre["text"], "lang": pre["lang"]},
            "abuse": abuse_out,
            "crisis": crisis_out,
            "escalation": esc,
            "content": content_out,
            "decision": decision,
        }
