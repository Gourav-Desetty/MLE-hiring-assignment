from __future__ import annotations

import re
import unicodedata
from collections import Counter


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[a-zàâçéèêëîïôûùüÿñáéíóúüßäö]+", re.I)

LANG_MARKERS = {
    "fr": {
        "bonjour", "merci", "compte", "facture", "remboursement", "problème",
        "probleme", "annuler", "aide", "avec", "pourquoi", "comment", "je",
        "vous", "mon", "ma", "mes", "le", "la", "les", "des", "une", "être",
    },
    "es": {
        "hola", "gracias", "cuenta", "factura", "reembolso", "problema",
        "cancelar", "ayuda", "con", "porque", "cómo", "como", "quiero",
        "necesito", "mi", "mis", "el", "la", "los", "las", "una", "para",
    },
    "de": {
        "hallo", "danke", "konto", "rechnung", "erstattung", "problem",
        "kündigen", "kundigen", "hilfe", "mit", "warum", "wie", "ich",
        "mein", "meine", "der", "die", "das", "und", "nicht", "bitte",
    },
}

DIACRITIC_HINTS = {
    "fr": "àâçéèêëîïôûùüÿœ",
    "es": "áéíóúñ¿¡",
    "de": "äöüß",
}


def detect_language(text: str) -> str:
    """Return a deterministic ISO 639-1 language code for the ticket text."""
    normalized = unicodedata.normalize("NFKC", text or "")
    if _cjk_ratio(normalized) >= 0.2:
        return "zh"

    lowered = normalized.lower()
    scores = Counter[str]()
    for lang, chars in DIACRITIC_HINTS.items():
        scores[lang] += sum(lowered.count(char) for char in chars) * 2

    words = WORD_RE.findall(lowered)
    for word in words:
        for lang, markers in LANG_MARKERS.items():
            if word in markers:
                scores[lang] += 1

    if not scores:
        return "en"

    lang, score = scores.most_common(1)[0]
    return lang if score >= 2 else "en"


def _cjk_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum(1 for char in visible if CJK_RE.match(char)) / len(visible)
