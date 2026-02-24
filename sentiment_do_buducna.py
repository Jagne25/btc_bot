# sentiment.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

_an = SentimentIntensityAnalyzer()

def score_text(text: str):
    if not text or not text.strip():
        return 0.0, "unk", False
    try:
        lang = detect(text)
    except Exception:
        lang = "unk"

    # VADER compound ∈ ⟨-1, +1⟩
    s = _an.polarity_scores(text).get("compound", 0.0)

    # CZ/SK flag (budeme to vedieť filtrovať)
    is_czsk = lang in ("cs", "sk")

    # zjednodušíme kódy
    lang_code = "en" if lang.startswith("en") else ("cs" if lang == "cs" else ("sk" if lang == "sk" else "other"))
    if lang_code in ("cs","sk"): lang_code = "czsk"
    return float(s), lang_code, is_czsk