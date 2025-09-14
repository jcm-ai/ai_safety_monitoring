import pytest
from src.preprocessing.language_detection import detect_language
from src.preprocessing.text_normalization import normalize_text
from src.preprocessing.pii_masking import mask_pii

def test_language_detection_en():
    lang, conf = detect_language("Hello, how are you?")
    assert lang == "en"
    assert conf >= 0.9

def test_language_detection_hi():
    lang, conf = detect_language("आप कैसे हैं?")
    assert lang == "hi"
    assert conf >= 0.9

def test_normalization_basic():
    raw = "Hello!!! Visit https://example.com NOW."
    clean = normalize_text(raw)
    assert "hello" in clean
    assert "visit" in clean
    assert "example" not in clean  # URL removed
    assert "!" not in clean

def test_pii_masking_email():
    text = "Contact me at jagadish@example.com"
    masked = mask_pii(text)
    assert "<EMAIL>" in masked
    assert "@" not in masked

def test_pii_masking_phone():
    text = "Call me at +91-9876543210"
    masked = mask_pii(text)
    assert "<PHONE>" in masked
    assert "9876" not in masked
