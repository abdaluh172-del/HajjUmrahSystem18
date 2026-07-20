# -*- coding: utf-8 -*-
"""Professional multilingual sentiment engine (v13).

The old system classified with a small TF-IDF model + keyword lists and made
frequent mistakes, especially outside Arabic/English. This engine replaces it
with a tiered pipeline — every comment (site reviews AND external comments
from YouTube/Reddit/X/any future source) goes through analyze():

TIER 1 — Transformer AI model (strongest, recommended):
    If the environment variable HF_API_TOKEN is set, the text is classified
    by `cardiffnlp/twitter-xlm-roberta-base-sentiment` through the Hugging
    Face Inference API — a real multilingual transformer trained on tweets
    in Arabic, English, Turkish, Urdu, Indonesian, Hindi and many more, so
    it understands each language NATIVELY (no keywords involved). Running
    it via the API keeps the app tiny enough for Render's free plan (a
    local transformer would not fit in 512 MB RAM).

TIER 2 — VADER on the English translation:
    Without a token, the comment is machine-translated to English
    (translation.py) and scored by VADER — a research-grade sentiment
    analyzer that models negation ("not good"), intensifiers ("very bad"),
    contrast ("but"), punctuation/caps emphasis and emojis. Far beyond
    keyword matching, works fully offline on the server.

TIER 3 — Built-in analyzer (always available):
    If VADER isn't installed, an internal analyzer with the same core ideas
    (negation windows, intensifiers, emoji polarity, Arabic + English
    vocabulary) scores the text, blended with the trained ML model's
    probabilities on the English translation.

Whatever tier runs, the result is:
    {"label": positive|negative|neutral, "confidence": float 0..100,
     "scores": {...}, "engine": "transformer"|"vader"|"builtin"}
and errors NEVER propagate: any tier that fails falls through to the next.
"""
import os
import re

import requests

import translation

HF_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TIMEOUT = 20

# VADER is in requirements.txt (installed on Render); the guarded import
# keeps the app working even where it's missing.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None


# ------------------------------------------------------------------ #
# TIER 1 — multilingual transformer via the HF Inference API
# ------------------------------------------------------------------ #
_HF_LABEL = {"positive": "positive", "negative": "negative", "neutral": "neutral",
             "LABEL_2": "positive", "LABEL_0": "negative", "LABEL_1": "neutral"}


def _analyze_transformer(text: str):
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        return None
    try:
        r = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}"},
                          json={"inputs": text[:1500], "options": {"wait_for_model": True}},
                          timeout=HF_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Response shape: [[{"label": "positive", "score": 0.98}, ...]]
        cand = data[0] if data and isinstance(data[0], list) else data
        scores = {}
        for item in cand:
            lbl = _HF_LABEL.get(item.get("label"))
            if lbl:
                scores[lbl] = round(float(item.get("score", 0)) * 100, 1)
        if not scores:
            return None
        label = max(scores, key=scores.get)
        return {"label": label, "confidence": scores[label], "scores": scores,
                "engine": "transformer"}
    except Exception as e:
        print(f"[sentiment] transformer API failed, falling back: {e}")
        return None


# ------------------------------------------------------------------ #
# TIER 2 — VADER on the English translation
# ------------------------------------------------------------------ #
def _analyze_vader(text_en: str):
    if _VADER is None or not text_en:
        return None
    try:
        vs = _VADER.polarity_scores(text_en)
        compound = vs["compound"]  # -1 .. +1
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        confidence = round(min(99.0, 50 + abs(compound) * 50), 1)
        return {"label": label, "confidence": confidence,
                "scores": {"positive": round(vs["pos"] * 100, 1),
                           "negative": round(vs["neg"] * 100, 1),
                           "neutral": round(vs["neu"] * 100, 1)},
                "engine": "vader"}
    except Exception as e:
        print(f"[sentiment] vader failed: {e}")
        return None


# ------------------------------------------------------------------ #
# TIER 3 — built-in analyzer: negation + intensifiers + emojis,
# Arabic and English vocabulary, blended with the trained ML model.
# ------------------------------------------------------------------ #
_POS = {
    # English
    "excellent": 3, "amazing": 3, "wonderful": 3, "awesome": 3, "perfect": 3,
    "great": 2.5, "fantastic": 3, "love": 2.5, "loved": 2.5, "best": 2.5,
    "good": 2, "nice": 2, "helpful": 2, "friendly": 2, "clean": 2, "fast": 1.5,
    "comfortable": 2, "organized": 2, "smooth": 2, "easy": 1.5, "beautiful": 2.5,
    "professional": 2, "thank": 1.5, "thanks": 1.5, "recommend": 2, "enjoyed": 2,
    "spiritual": 1.5, "blessed": 2, "peaceful": 2, "impressive": 2, "satisfied": 2,
    # Arabic
    "ممتاز": 3, "ممتازة": 3, "رائع": 3, "رائعة": 3, "مذهل": 3, "مذهلة": 3,
    "جميل": 2, "جميلة": 2, "جيد": 2, "جيدة": 2, "سريع": 1.5, "سريعة": 1.5,
    "نظيف": 2, "نظيفة": 2, "متعاون": 2, "متعاونين": 2, "مريح": 2, "مريحة": 2,
    "منظم": 2, "منظمة": 2, "تنظيم": 1.5, "محترم": 2, "محترمين": 2,
    "شكرا": 1.5, "شكراً": 1.5, "أشكر": 1.5, "احترافي": 2, "روحانية": 1.5,
    "تسهيل": 1.5, "سلس": 2, "سلسة": 2, "أنصح": 2, "استمتعت": 2, "راضي": 2,
    "مبهر": 3, "أفضل": 2, "حلو": 1.5, "حلوة": 1.5, "يعطيكم": 1, "العافية": 1,
}
_NEG = {
    # English
    "terrible": 3, "horrible": 3, "awful": 3, "worst": 3, "disgusting": 3,
    "bad": 2, "poor": 2, "dirty": 2.5, "rude": 2.5, "slow": 1.5, "late": 1.5,
    "crowded": 2, "crowding": 2, "chaos": 2.5, "chaotic": 2.5, "problem": 1.5,
    "problems": 1.5, "delay": 1.5, "delayed": 1.5, "broken": 2, "unorganized": 2.5,
    "disorganized": 2.5, "disappointed": 2.5, "disappointing": 2.5, "waste": 2,
    "exhausting": 1.5, "complaint": 1.5, "hate": 2.5, "hated": 2.5, "scam": 3,
    "expensive": 1.5, "overpriced": 2, "unhelpful": 2, "difficult": 1.5,
    # Arabic
    "سيء": 2.5, "سيئة": 2.5, "سئ": 2.5, "أسوأ": 3, "فظيع": 3, "فظيعة": 3,
    "ازدحام": 2, "زحمة": 2, "زحام": 2, "بطيء": 1.5, "بطيئة": 1.5,
    "تأخير": 1.5, "تأخر": 1.5, "متسخ": 2.5, "متسخة": 2.5, "وسخ": 2.5,
    "مشكلة": 1.5, "مشاكل": 1.5, "ضعيف": 2, "ضعيفة": 2, "شكوى": 1.5,
    "فوضى": 2.5, "مزعج": 2, "مزعجة": 2, "محبط": 2.5, "خايس": 2.5,
    "غالي": 1.5, "غالية": 1.5, "مقرف": 3, "تعبنا": 1.5, "معاناة": 2,
    "استغلال": 2.5, "احتيال": 3, "نصب": 3, "كارثة": 3, "مأساة": 2.5,
}
_NEGATORS = {"not", "no", "never", "none", "n't", "without", "hardly",
             "لا", "لم", "لن", "ليس", "ليست", "ما", "غير", "بدون", "مو", "مب", "مش"}
_INTENSIFIERS = {"very": 1.5, "so": 1.3, "really": 1.4, "extremely": 1.8, "too": 1.3,
                 "جدا": 1.5, "جداً": 1.5, "للغاية": 1.8, "كثير": 1.3, "مره": 1.4,
                 "مرة": 1.4, "حيل": 1.4, "قوي": 1.3}
_POS_EMOJI = set("😊😀😃😄😁🥰😍🤩👍❤️💚🌟⭐🙏✨😌🕋")
_NEG_EMOJI = set("😠😡🤬😞😔😢😭👎💔😤🤢😖😫")
_TOKEN_RE = re.compile(r"[\w']+|[\U0001F300-\U0001FAFF\u2600-\u27BF]")
# Arabic attached prefixes (و/ف/ب/ال/وال/بال...) hide lexicon words:
# "والتأخير" must still match "تأخير". Strip only when the remainder is a
# known lexicon word, so normal words are never mangled.
_AR_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل")
# These negators attach tightly to the NEXT word only ("غير منظم"), unlike
# English "not" which can sit a couple of words away ("not very good").
_TIGHT_NEGATORS = {"غير", "ليس", "ليست", "مو", "مب", "مش", "بدون"}


def _canon(tok):
    """Canonical lexicon form of a token (handles Arabic attached prefixes)."""
    if tok in _POS or tok in _NEG or tok in _NEGATORS or tok in _INTENSIFIERS:
        return tok
    for p in _AR_PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            stripped = tok[len(p):]
            if stripped in _POS or stripped in _NEG:
                return stripped
    return tok


def _analyze_builtin(text: str, text_en: str, ml_scores: dict):
    """Rule scoring on original + English translation, blended with the ML
    model's probabilities on the English text."""
    score = 0.0
    hits = 0
    for chunk in (text or "", (text_en or "") if text_en != text else ""):
        raw = _TOKEN_RE.findall(chunk.lower())
        tokens = [_canon(tk) for tk in raw]
        used_negators = set()  # each negator flips ONE sentiment word, not all after it
        for i, tok in enumerate(tokens):
            w = _POS.get(tok, 0) - _NEG.get(tok, 0)
            if tok in _POS_EMOJI:
                w += 2
            if tok in _NEG_EMOJI:
                w -= 2
            if w == 0:
                continue
            # nearest unused negator within 3 tokens back (tight ones: 1 token)
            neg_idx = None
            for j in range(i - 1, max(-1, i - 4), -1):
                if j in used_negators or tokens[j] not in _NEGATORS:
                    continue
                if tokens[j] in _TIGHT_NEGATORS and i - j > 1:
                    continue
                neg_idx = j
                break
            if neg_idx is not None:
                w = -w * 0.9
                used_negators.add(neg_idx)
            for prev in tokens[max(0, i - 3):i]:
                if prev in _INTENSIFIERS:
                    w *= _INTENSIFIERS[prev]
            score += w
            hits += 1
    rule_pos = max(0.0, score)
    rule_neg = max(0.0, -score)
    strength = min(1.0, abs(score) / 4.0)
    # Blend: rules dominate when they fired clearly; the ML model fills in
    # when the rules saw nothing it recognizes.
    ml_pos = ml_scores.get("positive", 33.3) / 100
    ml_neg = ml_scores.get("negative", 33.3) / 100
    rule_weight = 0.75 if hits else 0.0
    pos = rule_weight * (strength if score > 0 else 0) + (1 - rule_weight) * ml_pos
    neg = rule_weight * (strength if score < 0 else 0) + (1 - rule_weight) * ml_neg
    if pos - neg > 0.12:
        label = "positive"
    elif neg - pos > 0.12:
        label = "negative"
    else:
        label = "neutral"
    conf = round(min(99.0, 55 + abs(pos - neg) * 45), 1)
    return {"label": label, "confidence": conf,
            "scores": {"positive": round(pos * 100, 1), "negative": round(neg * 100, 1),
                       "neutral": round(max(0.0, 1 - pos - neg) * 100, 1)},
            "engine": "builtin", "_rule_hits": hits}


def analyze(text: str, ml_predict=None) -> dict:
    """Full multilingual analysis of one comment. Never raises.

    ml_predict: optional callable(text) -> {"positive": %, "negative": %, ...}
    from the trained ML model (used by tier 3)."""
    text = (text or "").strip()
    # TIER 1 — multilingual transformer (understands the original language).
    res = _analyze_transformer(text)
    if res:
        return res
    # Tiers 2/3 analyze the ENGLISH translation so any language works —
    # the translation itself keeps the meaning; the original stays untouched.
    text_en = translation.to_english(text)
    res = _analyze_vader(text_en)
    if res:
        return res
    ml_scores = {}
    if ml_predict is not None:
        try:
            ml_scores = ml_predict(text_en) or {}
        except Exception:
            ml_scores = {}
    return _analyze_builtin(text, text_en, ml_scores)


def engine_status() -> dict:
    """For the admin panel: which analysis tier will run."""
    return {
        "transformer_configured": bool(os.environ.get("HF_API_TOKEN")),
        "transformer_model": HF_MODEL,
        "vader_available": _VADER is not None,
        "active_engine": ("transformer" if os.environ.get("HF_API_TOKEN")
                          else "vader" if _VADER is not None else "builtin"),
    }
