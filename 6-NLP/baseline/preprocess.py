"""
KorNLI Preprocessing - 텍스트 전처리
"""

import re
import unicodedata


def preprocess_text(text: str) -> str:
    """기본 텍스트 전처리 (Unicode NFC 정규화 + 공백 정규화)"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
