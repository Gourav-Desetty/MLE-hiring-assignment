from __future__ import annotations

import re

from models import PiiFinding


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,19}(?!\d)")
PHONE_CONTEXT_SKIP_RE = re.compile(r"\b(order|reference|tracking|card|transaction|txn|ticket|case)\s*$", re.I)
ACCOUNT_RE = re.compile(
    r"\b(?:acct|account|user|customer)(?:[_ -]?(?:id|number|no))?\s*[:#-]?\s*"
    r"(?=[A-Za-z0-9_-]*(?:\d|[_-]))[A-Za-z0-9][A-Za-z0-9_-]{5,}\b",
    re.I,
)
ADDRESS_RE = re.compile(r"\b\d{1,6}\s+[A-Za-z0-9 .'-]{2,60}\s+(street|st|road|rd|avenue|ave|lane|ln|drive|dr|blvd|boulevard|apt|suite)\b", re.I)


def detect_and_redact(text: str) -> PiiFinding:
    redacted = text
    kinds: list[str] = []
    safe_contact: str | None = None
    card_tail: str | None = None

    email_match = EMAIL_RE.search(text)
    if email_match:
        kinds.append("email")
        safe_contact = email_match.group(0)
        redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)

    for match in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _is_luhn_card(digits):
            kinds.append("card")
            card_tail = digits[-4:]
            break
    redacted = CARD_RE.sub(_redact_card_match, redacted)

    if SSN_RE.search(redacted):
        kinds.append("ssn")
        redacted = SSN_RE.sub("[REDACTED_SSN]", redacted)
    if any(_is_phone_match(match, redacted) for match in PHONE_RE.finditer(redacted)):
        kinds.append("phone")
        redacted = PHONE_RE.sub(lambda match: _redact_phone_match(match, redacted), redacted)
    if ADDRESS_RE.search(redacted):
        kinds.append("address")
        redacted = ADDRESS_RE.sub("[REDACTED_ADDRESS]", redacted)
    if ACCOUNT_RE.search(redacted):
        kinds.append("account_or_transaction_id")
        redacted = ACCOUNT_RE.sub("[REDACTED_ID]", redacted)

    return PiiFinding(
        detected=bool(kinds),
        redacted_text=redacted,
        kinds=tuple(dict.fromkeys(kinds)),
        safe_contact=safe_contact,
        card_tail=card_tail,
    )


def contains_unredacted_pii(text: str) -> bool:
    if EMAIL_RE.search(text) or SSN_RE.search(text) or ADDRESS_RE.search(text):
        return True
    if any(_is_phone_match(match, text) for match in PHONE_RE.finditer(text)):
        return True
    if ACCOUNT_RE.search(text):
        return True
    for match in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _is_luhn_card(digits):
            return True
    return False


def _redact_card_match(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_CARD]" if _is_luhn_card(digits) else match.group(0)


def _redact_phone_match(match: re.Match[str], text: str) -> str:
    return "[REDACTED_PHONE]" if _is_phone_match(match, text) else match.group(0)


def _is_phone_match(match: re.Match[str], text: str) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    if not 10 <= len(digits) <= 15:
        return False
    prefix = text[max(0, match.start() - 24):match.start()]
    return not PHONE_CONTEXT_SKIP_RE.search(prefix)


def _is_luhn_card(digits: str) -> bool:
    return 13 <= len(digits) <= 19 and _luhn(digits)


def _luhn(digits: str) -> bool:
    total, odd = 0, True
    for d in reversed(digits):
        n = int(d)
        if odd:
            total += n
        else:
            n *= 2
            total += n - 9 if n > 9 else n
        odd = not odd
    return total % 10 == 0
