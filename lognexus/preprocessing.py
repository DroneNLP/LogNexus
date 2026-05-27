import re


def fix_missing_space_camelcase(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def fix_allcaps_boundary(text: str) -> str:
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)


def fix_colon_semicolon_spacing(text: str) -> str:
    text = re.sub(r"\s+([:;])", r"\1", text)
    return re.sub(r"([:;])(?!\s)", r"\1 ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ensure_ending_punctuation(text: str) -> str:
    text = text.strip()
    if text and not re.search(r"[.!?;:]$", text):
        text += "."
    return text


def preprocess_log(text: str) -> str:
    text = fix_missing_space_camelcase(str(text))
    text = fix_allcaps_boundary(text)
    text = fix_colon_semicolon_spacing(text)
    text = normalize_whitespace(text)
    return ensure_ending_punctuation(text)
