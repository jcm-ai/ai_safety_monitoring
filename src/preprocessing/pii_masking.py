import re

# Regex patterns for PII
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}\b")

def mask_pii(
    text: str,
    *,
    mask_email: bool = True,
    mask_phone: bool = True,
    email_token: str = "<EMAIL>",
    phone_token: str = "<PHONE>"
) -> str:
    """
    Mask personally identifiable information (PII) in text.

    Args:
        text: Input string
        mask_email: Whether to mask email addresses
        mask_phone: Whether to mask phone numbers
        email_token: Replacement token for emails
        phone_token: Replacement token for phone numbers

    Returns:
        Text with PII masked
    """
    if not text:
        return ""

    s = text
    if mask_email:
        s = EMAIL_RE.sub(email_token, s)
    if mask_phone:
        s = PHONE_RE.sub(phone_token, s)

    return s
