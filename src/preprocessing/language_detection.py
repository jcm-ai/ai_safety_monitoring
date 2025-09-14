from __future__ import annotations
from langdetect import detect, DetectorFactory
from typing import Tuple
from src.utils.logger import get_logger

DetectorFactory.seed = 42  # Ensures consistent results
logger = get_logger(__name__)

SUPPORTED_LANGUAGES = {"en", "hi"}

def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect language of input text.

    Returns:
        (language_code, confidence_score)
        Confidence is approximated since langdetect doesn't expose it directly.
    """
    if not text or len(text.strip()) < 5:
        logger.debug("Text too short for reliable detection", extra={"context": {"text": text}})
        return ("en", 0.0)

    try:
        code = detect(text)
        confidence = 1.0 if code in SUPPORTED_LANGUAGES else 0.5
        return (code if code in SUPPORTED_LANGUAGES else "en", confidence)
    except Exception as e:
        logger.warning("Language detection failed", extra={"context": {"error": str(e), "text": text}})
        return ("en", 0.0)
