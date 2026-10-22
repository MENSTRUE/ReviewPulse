import re
import unicodedata


def clean_review(value: object) -> str:
    """Minimal cleaning; preserve wording for fitted TF-IDF pipelines."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text
