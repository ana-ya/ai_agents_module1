import re
from src.exceptions import ResearchError

# Константи для валідації
MIN_TOPIC_LENGTH = 3
MAX_TOPIC_LENGTH = 500

# Патерни для перевірки небезпечного контенту
INVALID_CHARS_PATTERN = r'[<>{}]'
SUSPICIOUS_PATTERNS = [
    r'<script[^>]*>',           # Script tags
    r'javascript:',             # JavaScript protocol
    r'on\w+\s*=',              # Event handlers (onclick, onerror, etc.)
    r'(--|;)\s*DROP',          # SQL DROP commands
    r'(--|;)\s*DELETE',        # SQL DELETE commands
    r'(--|;)\s*INSERT',        # SQL INSERT commands
    r'(--|;)\s*UPDATE',        # SQL UPDATE commands
]


def _validate_topic(topic: str) -> str:
    """Validate and sanitize topic"""
    
    # 1. Check if empty
    if not topic or not topic.strip():
        raise ResearchError("Тема не може бути порожньою")
    
    topic = topic.strip()
    
    # 2. Check length
    if len(topic) <  MIN_TOPIC_LENGTH:
        raise ResearchError(f"Тема занадто коротка (мінімум {MIN_TOPIC_LENGTH} символів)")
    elif len(topic) > MAX_TOPIC_LENGTH:
        raise ResearchError(f"Тема занадто довга (максимум {MAX_TOPIC_LENGTH} символів)")
    
    # 3. Check for HTML/code injection attempts
    if re.search(INVALID_CHARS_PATTERN, topic):
        raise ResearchError("Тема містить заборонені символи: < > { }")
    
    # 4. Check for script tags (case-insensitive)
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, topic, re.IGNORECASE):
            raise ResearchError("Тема містить підозрілий контент")
    
    # 5. Remove excessive whitespace
    topic = ' '.join(topic.split())
    
    return topic