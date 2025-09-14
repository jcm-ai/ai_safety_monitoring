import re
import unicodedata

# Regular expressions for cleaning
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")

def normalize_text(
    text: str,
    *,
    lower: bool = True,
    strip_urls: bool = True,
    strip_punctuation: bool = True,
    collapse_whitespace: bool = True,
    unicode_nfkc: bool = True
) -> str:
    """
    Normalize input text for model ingestion.

    Args:
        text: Raw input string
        lower: Convert to lowercase
        strip_urls: Remove URLs
        strip_punctuation: Remove punctuation
        collapse_whitespace: Replace multiple spaces with one
        unicode_nfkc: Normalize unicode characters (e.g., ligatures, accents)

    Returns:
        Cleaned string
    """
    if not text:
        return ""

    s = text
    if unicode_nfkc:
        s = unicodedata.normalize("NFKC", s)
    if strip_urls:
        s = URL_RE.sub(" ", s)
    if lower:
        s = s.lower()
    if strip_punctuation:
        s = PUNCT_RE.sub(" ", s)
    if collapse_whitespace:
        s = WS_RE.sub(" ", s).strip()

    return s
