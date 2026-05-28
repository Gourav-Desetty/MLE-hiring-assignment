from __future__ import annotations

import re

from models import PiiFinding


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,19}(?!\d)")
ACCOUNT_RE = re.compile(r"\b(?:acct|account|user|customer|txn|transaction|ticket|case)[_-]?[A-Za-z0-9-]{4,}\b", re.I)
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
        if 13 <= len(digits) <= 19:
            kinds.append("card")
            card_tail = digits[-4:]
            break
    redacted = CARD_RE.sub("[REDACTED_CARD]", redacted)

    if PHONE_RE.search(redacted):
        kinds.append("phone")
        redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    if SSN_RE.search(redacted):
        kinds.append("ssn")
        redacted = SSN_RE.sub("[REDACTED_SSN]", redacted)
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
    if EMAIL_RE.search(text) or PHONE_RE.search(text) or SSN_RE.search(text) or ADDRESS_RE.search(text):
        return True
    if ACCOUNT_RE.search(text):
        return True
    for match in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19:
            return True
    return False
